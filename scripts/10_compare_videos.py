"""Side-by-side videos of every method on the same held-out layouts, seen from the agentview camera:
open-loop replay of the retargeted demos, the state-based BC policy, the first camera CNN, and the
SmolVLA student after offline distillation and after DAgger. Each tile ends with the outcome of that
rollout; the column headers give the method's success rate on the 100 evaluation layouts.

Usage (GPU for the SmolVLA students; checkpoints under $HC_VLA_ROOT/ckpt):
    python scripts/10_compare_videos.py --pushed-seeds 0,3,9 --nominal-seeds 7,12,15
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hc.datagen import HOLD_S, plan_for  # noqa: E402
from hc.evaluate import AFTER_STEPS, EVAL_SEED_START, HOLD_STEPS, HORIZON, sample_perturbation  # noqa: E402
from hc.policy import PolicyRunner, load_policy  # noqa: E402
from hc.sim import CONTROL_FREQ, displace_cube, make_env, reset  # noqa: E402
from hc.types import load_demos  # noqa: E402
from hc.vision_policy import AgentviewCamera, VisionPolicyRunner, load_vision_policy  # noqa: E402
from hc.vla_env import StudentCameras, student_state  # noqa: E402

ROOT = Path(os.environ.get("HC_VLA_ROOT", "/var/tmp/humanoid_vla"))
OUT = Path("outputs/stack_main/vla")
TEACHER = Path("data/processed/stack_main1600_bc.pt")
DEMOS = Path("data/processed/stack_main_demos.npz")
CNN = Path("outputs/stack_main/vision/vision_bc.pt")
TILE = 256
HEADER = 44


class Replay:
    """Open-loop replay: the demo retargeted once to the initial layout, tracked for its length
    plus HOLD_S, success checked at the end (as in hc.evaluate)."""

    closed_loop = False

    def __init__(self, demos):
        self.demos = demos

    def reset(self, env, obs: dict, seed: int) -> int:
        demo = self.demos[(seed - EVAL_SEED_START) % len(self.demos)]  # same demo as the evaluation
        self.plan, self.ctl = plan_for(obs, demo, 1.0)
        self.k = 0
        return len(self.plan) + round(HOLD_S * CONTROL_FREQ)

    def act(self, obs: dict, images: np.ndarray) -> np.ndarray:
        i = min(self.k, len(self.plan) - 1)
        self.k += 1
        return self.ctl.action(obs, self.plan.pos[i], self.plan.yaw[i], self.plan.grip[i])


class StateBC:
    closed_loop = True

    def __init__(self):
        self.runner = PolicyRunner(load_policy(TEACHER))

    def reset(self, env, obs: dict, seed: int) -> int:
        self.runner.reset()
        return HORIZON

    def act(self, obs: dict, images: np.ndarray) -> np.ndarray:
        return self.runner.act(obs)


class CameraCNN:
    """The first camera policy: renders its own 84x84 agentview, as it was trained."""

    closed_loop = True

    def __init__(self):
        self.runner = VisionPolicyRunner(load_vision_policy(CNN))
        self.cam = AgentviewCamera(self.runner.policy.config.image_size)

    def reset(self, env, obs: dict, seed: int) -> int:
        self.cam.attach(env)
        self.runner.reset()
        return HORIZON

    def act(self, obs: dict, images: np.ndarray) -> np.ndarray:
        return self.runner.act(obs, self.cam.render())


class SmolVLA:
    closed_loop = True

    def __init__(self, ckpt: str, n_action_steps: int = 4):
        import torch

        from hc.vla_student import Student
        self.student = Student.load(ROOT / "ckpt" / ckpt)
        self.n = n_action_steps
        self.generator = torch.Generator().manual_seed(0)

    def reset(self, env, obs: dict, seed: int) -> int:
        self.queue = []
        return HORIZON

    def act(self, obs: dict, images: np.ndarray) -> np.ndarray:
        if not self.queue:
            chunk = self.student.predict(images[None], student_state(obs)[None], self.generator)[0]
            self.queue = list(chunk[:self.n])
        action = self.queue.pop(0).astype(np.float64)
        action[-1] = 1.0 if action[-1] > 0 else -1.0
        return action


def rollout(env, cams: StudentCameras, actor, seed: int, pushed: bool) -> tuple[list[np.ndarray], bool]:
    """One episode on eval layout `seed` with hc.evaluate's push and success rule; returns the
    upright agentview frames."""
    obs = reset(env, seed)
    cams.attach(env)
    steps = actor.reset(env, obs, seed)
    perturb = sample_perturbation(seed, obs) if pushed else None
    frames, held = [], 0
    for t in range(steps):
        if perturb is not None and t == perturb.step:
            displace_cube(env, perturb.cube, perturb.dxy)
            obs = env._get_observations(force_update=True)
        images = cams.render()
        frames.append(np.ascontiguousarray(images[0][:, ::-1]))
        obs, *_ = env.step(actor.act(obs, images))
        held = held + 1 if env._check_success() else 0
        if actor.closed_loop and held >= HOLD_STEPS + AFTER_STEPS:
            break
    success = held >= HOLD_STEPS + AFTER_STEPS if actor.closed_loop else bool(env._check_success())
    return frames, success


def label(img: np.ndarray, text: str, colour: tuple[int, int, int]) -> np.ndarray:
    img = img.copy()
    cv2.rectangle(img, (0, TILE - 30), (TILE, TILE), colour, -1)
    cv2.putText(img, text, (8, TILE - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def header(names: list[str], rates: list[str]) -> np.ndarray:
    bar = np.full((HEADER, TILE * len(names), 3), 255, np.uint8)
    for c, (name, rate) in enumerate(zip(names, rates)):
        cv2.putText(bar, name, (6 + c * TILE, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(bar, rate, (6 + c * TILE, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (90, 90, 90), 1, cv2.LINE_AA)
    return bar


def write_grid(panels: list[list[tuple[list[np.ndarray], bool]]], names: list[str], rates: list[str],
               path: Path, fps: int = 20) -> None:
    """panels[row][col] = (frames, success); finished tiles freeze on their last frame with the outcome."""
    import imageio.v2 as imageio
    n = max(len(f) for row in panels for f, _ in row) + fps
    top = header(names, rates)
    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=7, macro_block_size=4)
    for t in range(n):
        rows = []
        for row in panels:
            tiles = []
            for frames, success in row:
                img = cv2.resize(frames[min(t, len(frames) - 1)], (TILE, TILE))
                if t >= len(frames) - 1:
                    img = label(img, "SUCCESS" if success else "FAILED", (40, 150, 40) if success else (190, 40, 40))
                tiles.append(img)
            rows.append(np.concatenate(tiles, 1))
        writer.append_data(np.concatenate([top] + rows, 0))
    writer.close()


def rate_text(cond: str) -> list[str]:
    """Success on the 100 evaluation layouts (50 for the camera CNN) for each column."""
    ref = json.loads(Path("outputs/stack_main1600/eval/results.json").read_text())
    key = "nominal" if cond == "nominal" else "perturbed"
    pct = lambda xs: f"{100 * np.mean(xs):.0f} %"
    cnn = json.loads((CNN.parent / "results.json").read_text())
    vla = {n: json.loads((OUT / f"eval_{n}.json").read_text())[cond]["rate"] for n in ("r0", "r3_full")}
    cnn_rate = cnn["eval"][cond]["success_rate"]
    return [f"{pct(ref[key]['replay']['success'])} on 100 layouts", f"{pct(ref[key]['bc']['success'])} on 100 layouts",
            f"{100 * cnn_rate:.0f} % on 50 layouts", f"{100 * vla['r0']:.0f} % on 100 layouts",
            f"{100 * vla['r3_full']:.0f} % on 100 layouts"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pushed-seeds", default="0,3,9", help="eval layout indices (offsets from 1e6)")
    parser.add_argument("--nominal-seeds", default="7,12,15")
    args = parser.parse_args()

    env = make_env(horizon=HORIZON)
    cams = StudentCameras()
    actors = [Replay(load_demos(DEMOS)), StateBC(), CameraCNN(), SmolVLA("r0"), SmolVLA("r3")]
    names = ["open-loop replay of my demos", "state BC (sees cube poses)", "camera CNN (first try)",
             "SmolVLA, offline distillation", "SmolVLA, after DAgger"]
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for cond, seeds in (("pushed", args.pushed_seeds), ("nominal", args.nominal_seeds)):
        panels = []
        for s in [EVAL_SEED_START + int(x) for x in seeds.split(",")]:
            row = [rollout(env, cams, actor, s, cond == "pushed") for actor in actors]
            summary[f"{cond}_{s - EVAL_SEED_START}"] = dict(zip(names, [ok for _, ok in row]))
            print(cond, s - EVAL_SEED_START, [ok for _, ok in row], flush=True)
            panels.append(row)
        write_grid(panels, names, rate_text(cond), OUT / f"compare_{cond}.mp4")
    (OUT / "compare_videos.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
