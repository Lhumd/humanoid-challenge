"""Replay each retargeted human demo on the robot, side by side with the phone video, and measure
open-loop replay success on several robot arms (the plan lives in end-effector space, so the same
object-centric plan drives every arm).

python scripts/02_replay_demos.py --demos data/processed/stack_main_demos.npz \
    --video data/raw/stack_main.mp4 --track data/processed/stack_main_track.npz
"""

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hc.datagen import execute_plan, horizon_for, plan_for, replay_success  # noqa: E402
from hc.evaluate import EVAL_SEED_START, eval_seeds  # noqa: E402
from hc.sim import CONTROL_FREQ, make_env, reset  # noqa: E402
from hc.types import INDEX_TIP, THUMB_TIP, Demo, HandTrack, load_demos  # noqa: E402

HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
              (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17),
              (17, 18), (18, 19), (19, 20)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--demos", type=Path, default=ROOT / "data/processed/stack_main_demos.npz")
    p.add_argument("--video", type=Path, default=None, help="phone clip, for the human half")
    p.add_argument("--track", type=Path, default=None, help="hand track, drawn over the video")
    p.add_argument("--robots", nargs="+", default=["Panda", "UR5e", "IIWA", "Kinova3"])
    p.add_argument("--seeds", type=int, default=20, help="eval layouts per demo and robot")
    p.add_argument("--camera", default="agentview")
    p.add_argument("--size", type=int, default=384, help="height of each half of the video")
    p.add_argument("--max-videos", type=int, default=100)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--out", type=Path, default=ROOT / "outputs/stack_main/replay")
    return p.parse_args()


def source_start_frame(demo: Demo) -> int:
    """Demo.source is "<clip>:<start>-<end>", frame indices into the clip."""
    return int(re.fullmatch(r".*:(\d+)-(\d+)", demo.source).group(1))


def read_frames(video: Path, start: int, count: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for _ in range(count):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def draw_hand(frame: np.ndarray, landmarks: np.ndarray, closed: bool) -> None:
    if np.isnan(landmarks).any():
        return
    pts = landmarks.astype(int)
    for a, b in HAND_EDGES:
        cv2.line(frame, tuple(pts[a]), tuple(pts[b]), (255, 255, 255), 2, cv2.LINE_AA)
    pinch = ((landmarks[THUMB_TIP] + landmarks[INDEX_TIP]) / 2).astype(int)
    cv2.circle(frame, tuple(pinch), 9, (0, 0, 255) if closed else (0, 200, 0), -1, cv2.LINE_AA)


def fit_height(frame: np.ndarray, height: int) -> np.ndarray:
    """Resize to `height`, keeping the aspect ratio and an even width (H.264 needs one)."""
    width = 2 * round(frame.shape[1] * height / frame.shape[0] / 2)
    return cv2.resize(frame, (width, height),
                      interpolation=cv2.INTER_AREA)


def label(frame: np.ndarray, text: str) -> np.ndarray:
    cv2.putText(frame, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def render_side_by_side(demos: list[Demo], args: argparse.Namespace) -> None:
    """One MP4 per demo: phone video (left, if given) and the robot replaying the retargeted plan
    on eval layout EVAL_SEED_START + i (right), synchronised on the demo's clock."""
    track = HandTrack.load(args.track) if args.track else None
    env = make_env(horizon=horizon_for(demos, 1.0), camera=args.camera, camera_size=args.size)
    for i, demo in enumerate(demos[:args.max_videos]):
        obs = reset(env, EVAL_SEED_START + i)
        plan, ctl = plan_for(obs, demo, 1.0)
        episode, robot_frames = execute_plan(env, obs, plan, ctl, np.random.default_rng(0),
                                             camera=args.camera)
        human = []
        if args.video:
            start = source_start_frame(demo)
            fps = 1.0 / np.median(np.diff(demo.t))
            clip = read_frames(args.video, start, len(demo.t))
            for k in range(len(robot_frames)):
                idx = min(int(round(plan.human_t[min(k, len(plan) - 1)] * fps)), len(clip) - 1)
                frame = clip[idx].copy()
                if track is not None:
                    draw_hand(frame, track.image_landmarks[start + idx], demo.closed[idx])
                human.append(label(fit_height(frame, args.size)[:, :, ::-1].copy(), "human (phone)"))
        verdict = "stacked" if episode.success else "failed"
        writer = imageio.get_writer(args.out / f"demo{i:02d}_replay.mp4", fps=CONTROL_FREQ,
                                    macro_block_size=1)
        for k, robot in enumerate(robot_frames):
            right = label(robot.copy(), f"robot replay: {verdict}" if k >= len(plan) else "robot replay")
            writer.append_data(np.hstack([human[k], right]) if human else right)
        writer.close()
        print(f"demo {i} ({demo.source}): {len(plan)} plan steps, {verdict}")


def cross_embodiment(demos: list[Demo], args: argparse.Namespace) -> dict:
    seeds = eval_seeds(args.seeds)
    results, skipped = {}, {}
    for robot in args.robots:
        try:
            make_env(robot=robot).close()
        except Exception as err:  # some robosuite robots ship broken assets for this mujoco
            skipped[robot] = str(err).splitlines()[0]
            print(f"{robot}: skipped, failed to load ({skipped[robot]})")
            continue
        per_demo = replay_success(demos, seeds, robot, args.workers)
        results[robot] = {"mean": float(np.mean(per_demo)), "per_demo": per_demo}
        print(f"{robot}: open-loop replay success {100 * np.mean(per_demo):.0f}% "
              f"({len(demos)} demos x {len(seeds)} layouts)")
    (args.out / "cross_embodiment.json").write_text(json.dumps(
        {"seeds": [seeds[0], seeds[-1]], "results": results, "skipped": skipped}, indent=2))
    return results


def plot_cross_embodiment(results: dict, path: Path) -> None:
    robots = list(results)
    fig, ax = plt.subplots(figsize=(1.2 * len(robots) + 1.5, 3))
    bars = ax.bar(robots, [100 * results[r]["mean"] for r in robots], color="#4c72b0", width=0.6)
    ax.bar_label(bars, fmt="%.0f%%", label_type="center", color="white", fontsize=9)
    for x, r in enumerate(robots):
        per_demo = 100 * np.array(results[r]["per_demo"])
        ax.scatter(np.full(len(per_demo), x) + np.linspace(-0.15, 0.15, len(per_demo)), per_demo,
                   s=8, color="k", zorder=3)
    ax.set_ylabel("stack success (%)")
    ax.set_ylim(0, 112)
    ax.set_title("Same retargeted human plans, different arms (dots: individual demos)", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    demos = load_demos(args.demos)
    render_side_by_side(demos, args)
    results = cross_embodiment(demos, args)
    plot_cross_embodiment(results, args.out / "cross_embodiment.png")


if __name__ == "__main__":
    main()
