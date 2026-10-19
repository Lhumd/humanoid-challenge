"""Evaluation on held-out layouts (seeds >= EVAL_SEED_START): the closed-loop BC policy against
open-loop replay of the retargeted human demos, with and without a mid-episode perturbation.

Success criterion for the policy: robosuite's Stack test holds for 3 s in a row (HOLD_STEPS +
AFTER_STEPS) while the policy keeps acting. The replay arm uses hc.datagen.execute_plan, which
tracks the whole plan including the retreat, holds its last target for 1 s and then applies the
same Stack test.
"""

import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np

from hc.datagen import Episode, execute_plan, plan_for
from hc.policy import PolicyRunner, load_policy
from hc.sim import CONTROL_FREQ, displace_cube, make_env, policy_features, reset
from hc.types import load_demos

EVAL_SEED_START = 1_000_000
HORIZON = 400
HOLD_STEPS = CONTROL_FREQ
AFTER_STEPS = 2 * CONTROL_FREQ  # keep acting after the stack holds, as replay does through its retreat
PERTURB_STEP = 15
PERTURB_RANGE = (0.03, 0.05)
MIN_CUBE_GAP = 0.07
MAX_WORKERS = 6


class Perturbation(NamedTuple):
    """Teleport `cube` by `dxy` metres at policy step `step` (same layout as execute_plan's tuple)."""

    step: int
    cube: str
    dxy: np.ndarray


@dataclass
class Task:
    """One evaluation or exploration episode for the worker pool.

    arm: "bc" (policy), "replay" (open-loop retargeted demo) or "explore" (policy with action
    noise, recorded for self-improvement).
    """

    arm: str
    seed: int
    perturbed: bool = False
    camera: str | None = None
    demo_index: int = 0
    action_noise: float = 0.0


def eval_seeds(n: int) -> list[int]:
    return list(range(EVAL_SEED_START, EVAL_SEED_START + n))


def horizon_for_episodes(episodes_path: Path) -> int:
    """Policy rollout length: HORIZON, or 1.25x the longest training episode if that is longer."""
    lengths = np.diff(np.load(episodes_path)["episode_ends"], prepend=0)
    return max(HORIZON, int(1.25 * lengths.max()))


def sample_perturbation(seed: int, obs: dict) -> Perturbation:
    """A 3-5 cm push of cube A in a random direction, drawn from `seed`, rejecting pushes that
    would bring it within MIN_CUBE_GAP of cube B. Both evaluation arms get the same push."""
    rng = np.random.default_rng(seed)
    a, b = obs["cubeA_pos"][:2], obs["cubeB_pos"][:2]
    while True:
        angle, radius = rng.uniform(0, 2 * np.pi), rng.uniform(*PERTURB_RANGE)
        dxy = radius * np.array([np.cos(angle), np.sin(angle)])
        if np.linalg.norm(a + dxy - b) >= MIN_CUBE_GAP:
            return Perturbation(PERTURB_STEP, "A", dxy)


def run_policy(env, runner: PolicyRunner, seed: int, horizon: int = HORIZON,
               perturb: Perturbation | None = None, camera: str | None = None,
               action_noise: float = 0.0) -> tuple[Episode, list[np.ndarray]]:
    """Closed-loop rollout. Records features and the policy's clean actions while executing
    them plus N(0, action_noise) on the arm dimensions. Success means the stack held for
    HOLD_STEPS + AFTER_STEPS consecutive steps while the policy kept acting, so a policy that knocks
    the cube off after letting go fails, as open-loop replay does when its retreat knocks it off."""
    rng = np.random.default_rng(seed)
    obs = reset(env, seed)
    runner.reset()
    features, actions, frames = [], [], []
    held = 0
    for step in range(horizon):
        if perturb is not None and step == perturb.step:
            displace_cube(env, perturb.cube, perturb.dxy)
            obs = env._get_observations(force_update=True)
        action = runner.act(obs)
        features.append(policy_features(obs))
        actions.append(action)
        executed = action.copy()
        executed[:6] = np.clip(action[:6] + rng.normal(0.0, action_noise, 6), -1, 1)
        obs, *_ = env.step(executed)
        if camera is not None:
            frames.append(obs[f"{camera}_image"][::-1])
        held = held + 1 if env._check_success() else 0
        if held >= HOLD_STEPS + AFTER_STEPS:
            break
    episode = Episode(np.array(features, np.float32), np.array(actions, np.float32),
                      held >= HOLD_STEPS + AFTER_STEPS, seed, -1)
    return episode, frames


def rollout_policy(env, runner: PolicyRunner, seed: int, horizon: int = HORIZON,
                   perturb: Perturbation | None = None,
                   camera: str | None = None) -> tuple[bool, list[np.ndarray]]:
    """Noise-free closed-loop rollout -> (success, frames)."""
    episode, frames = run_policy(env, runner, seed, horizon, perturb, camera)
    return episode.success, frames


def replay_demo(env, demo, seed: int, perturb: Perturbation | None = None,
                camera: str | None = None) -> tuple[bool, list[np.ndarray]]:
    """Retarget `demo` onto the initial layout of `seed` and track it open-loop at human speed."""
    obs = reset(env, seed)
    plan, ctl = plan_for(obs, demo, 1.0)
    episode, frames = execute_plan(env, obs, plan, ctl, np.random.default_rng(seed),
                                   perturb=perturb, camera=camera, seed=seed)
    return episode.success, frames


_worker: dict = {}


def _init_worker(policy_path: Path | None, demos_path: Path | None, horizon: int) -> None:
    import torch
    torch.set_num_threads(1)
    _worker["policy"] = load_policy(policy_path) if policy_path else None
    _worker["demos"] = load_demos(demos_path) if demos_path else None
    _worker["horizon"] = horizon
    _worker["envs"] = {}


def _env(camera: str | None):
    if camera not in _worker["envs"]:
        _worker["envs"][camera] = make_env(camera=camera)
    return _worker["envs"][camera]


def _run_task(task: Task) -> dict:
    env = _env(task.camera)
    perturb = sample_perturbation(task.seed, reset(env, task.seed)) if task.perturbed else None
    result = {"task": task}
    if task.arm == "replay":
        result["success"], result["frames"] = replay_demo(
            env, _worker["demos"][task.demo_index], task.seed, perturb, task.camera)
        return result
    episode, frames = run_policy(env, PolicyRunner(_worker["policy"]), task.seed,
                                 _worker["horizon"], perturb, task.camera, task.action_noise)
    result.update(success=episode.success, frames=frames, steps=len(episode.actions))
    if task.arm == "explore":
        result["episode"] = episode
    return result


def run_tasks(tasks: list[Task], policy_path: Path | None = None, demos_path: Path | None = None,
              horizon: int = HORIZON, workers: int = MAX_WORKERS) -> list[dict]:
    """Run tasks in a spawn pool (one env per worker and camera), results in task order."""
    ctx = mp.get_context("spawn")
    with ctx.Pool(min(workers, MAX_WORKERS), _init_worker, (policy_path, demos_path, horizon)) as pool:
        return pool.map(_run_task, tasks, chunksize=1)


def evaluate(seeds: list[int], policy_path: Path | None = None, demos_path: Path | None = None,
             perturbed: bool = False, horizon: int = HORIZON, workers: int = MAX_WORKERS) -> dict:
    """Success of the BC policy (if `policy_path`) and of open-loop replay (if `demos_path`,
    demos assigned round-robin over seeds) on the same layouts."""
    tasks = []
    if policy_path is not None:
        tasks += [Task("bc", s, perturbed) for s in seeds]
    if demos_path is not None:
        n_demos = len(load_demos(demos_path))
        tasks += [Task("replay", s, perturbed, demo_index=i % n_demos) for i, s in enumerate(seeds)]
    results = run_tasks(tasks, policy_path, demos_path, horizon, workers)
    out: dict = {}
    for arm in ("bc", "replay"):
        hits = [bool(r["success"]) for r in results if r["task"].arm == arm]
        if hits:
            out[arm] = {"success_rate": float(np.mean(hits)), "n": len(hits), "success": hits}
    return out
