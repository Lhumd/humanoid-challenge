"""Sim-verified data multiplication: replay retargeted human demos over many random cube layouts
and keep the episodes that stack (MimicGen-style), recording clean tracking actions while
executing noisy ones (DART-style) so the learned policy sees and corrects small deviations.
"""

import json
import multiprocessing as mp
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hc.retarget import GRIP_DWELL_S, MIN_APPROACH_S, SETTLE_S, Plan, choose_grasp_yaw, retarget
from hc.sim import (CONTROL_FREQ, SETTLE_STEPS, TrackingController, displace_cube, make_env,
                    policy_features, reset, scene_from_obs)
from hc.types import Demo

HOLD_S = 1.0  # the plan's last target is held this long before the success check
MAX_POOL_WORKERS = 6


@dataclass
class Episode:
    """One robot rollout: what the policy observed and the clean action it should imitate."""

    features: np.ndarray  # (N, OBS_DIM) float32 (wider on arms whose gripper has more joints)
    actions: np.ndarray  # (N, ACTION_DIM) float32, clean tracking actions
    success: bool
    seed: int
    demo_index: int


def plan_for(obs: dict, demo: Demo, speed: float) -> tuple[Plan, TrackingController]:
    """Retarget `demo` onto the layout read from `obs`; the controller is homed at the current
    end-effector pose."""
    ctl = TrackingController(obs)
    scene = scene_from_obs(obs)
    return retarget(demo, scene, choose_grasp_yaw(ctl, scene), speed), ctl


def execute_plan(env, obs: dict, plan: Plan, ctl: TrackingController, rng: np.random.Generator,
                 action_noise: float = 0.0, perturb: tuple[int, str, np.ndarray] | None = None,
                 camera: str | None = None, seed: int = -1,
                 demo_index: int = -1) -> tuple[Episode, list[np.ndarray]]:
    """Track `plan` open-loop, then hold its last target for HOLD_S seconds.

    Every step records policy_features(obs) and the clean tracking action; the executed action
    adds N(0, action_noise) to the six arm dimensions. `perturb` = (step, "A"|"B", dxy) teleports
    a cube at that step. With `camera`, returns the rendered frames (upright) too.
    """
    steps = len(plan) + round(HOLD_S * CONTROL_FREQ)
    features, actions, frames = [], [], []
    for k in range(steps):
        i = min(k, len(plan) - 1)
        if perturb is not None and k == perturb[0]:
            displace_cube(env, perturb[1], np.asarray(perturb[2]))
        clean = ctl.action(obs, plan.pos[i], plan.yaw[i], plan.grip[i])
        features.append(policy_features(obs))
        actions.append(clean)
        executed = clean.copy()
        executed[:6] = np.clip(clean[:6] + rng.normal(0.0, action_noise, 6), -1, 1)
        obs, *_ = env.step(executed)
        if camera is not None:
            frames.append(obs[f"{camera}_image"][::-1])
    episode = Episode(np.array(features, np.float32), np.array(actions, np.float32),
                      bool(env._check_success()), seed, demo_index)
    return episode, frames


def plan_steps(demo: Demo, speed: float) -> int:
    """Upper bound on the length of a plan retargeted from `demo` at `speed`, plus the hold."""
    dwell_s = 2 * (SETTLE_S + GRIP_DWELL_S)
    human_s = demo.t[-1] - demo.t[0] + MIN_APPROACH_S
    return int(np.ceil((human_s / speed + dwell_s + HOLD_S) * CONTROL_FREQ)) + 3


def horizon_for(demos: list[Demo], min_speed: float) -> int:
    return max(plan_steps(d, min_speed) for d in demos) + SETTLE_STEPS + CONTROL_FREQ


_worker: dict = {}


def _init_worker(demos: list[Demo], robot: str, horizon: int) -> None:
    _worker["env"] = make_env(robot=robot, horizon=horizon)
    _worker["demos"] = demos


def _run_task(task: tuple[int, int, float, float]) -> Episode:
    seed, demo_index, speed, action_noise = task
    env = _worker["env"]
    obs = reset(env, seed)
    plan, ctl = plan_for(obs, _worker["demos"][demo_index], speed)
    episode, _ = execute_plan(env, obs, plan, ctl, np.random.default_rng(seed), action_noise,
                              seed=seed, demo_index=demo_index)
    return episode


def run_tasks(demos: list[Demo], tasks: list[tuple[int, int, float, float]], robot: str,
              workers: int) -> list[Episode]:
    """Run (seed, demo_index, speed, action_noise) tasks in a pool with one env per worker."""
    horizon = horizon_for(demos, min(t[2] for t in tasks))
    ctx = mp.get_context("spawn")
    with ctx.Pool(min(workers, MAX_POOL_WORKERS), _init_worker, (demos, robot, horizon)) as pool:
        return pool.map(_run_task, tasks, chunksize=1)


def generate(demos: list[Demo], n_episodes: int, out_path: Path, workers: int = 6,
             action_noise: float = 0.1, speed_range: tuple[float, float] = (0.8, 1.2),
             seed_offset: int = 0, robot: str = "Panda") -> dict:
    """Layout seed seed_offset + i replays demo i % len(demos) at a random speed with action
    noise; saves the successful episodes to `out_path` and statistics next to it as JSON."""
    assert seed_offset + n_episodes <= 100_000, "datagen seeds must stay below the eval range"
    speeds = np.random.default_rng(seed_offset).uniform(*speed_range, n_episodes)
    tasks = [(seed_offset + i, i % len(demos), float(speeds[i]), action_noise)
             for i in range(n_episodes)]
    start = time.time()
    episodes = run_tasks(demos, tasks, robot, workers)
    wall = time.time() - start
    successes = [e for e in episodes if e.success]
    save_episodes(out_path, successes)
    per_demo = [float(np.mean([e.success for e in episodes if e.demo_index == d]))
                for d in range(len(demos))]
    stats = {"n_attempted": n_episodes, "n_success": len(successes),
             "yield": len(successes) / n_episodes, "per_demo_success": per_demo,
             "wall_time_s": wall, "action_noise": action_noise, "speed_range": list(speed_range),
             "seed_offset": seed_offset, "robot": robot,
             "steps_saved": int(sum(len(e.actions) for e in successes))}
    Path(out_path).with_suffix(".json").write_text(json.dumps(stats, indent=2))
    return stats


def replay_success(demos: list[Demo], seeds: list[int], robot: str = "Panda",
                   workers: int = 6) -> list[float]:
    """Success rate per demo of noise-free, open-loop replay at the human's speed: the
    'retargeted human replay' baseline."""
    tasks = [(seed, d, 1.0, 0.0) for d in range(len(demos)) for seed in seeds]
    episodes = run_tasks(demos, tasks, robot, workers)
    return [float(np.mean([e.success for e in episodes if e.demo_index == d]))
            for d in range(len(demos))]


def save_episodes(path: Path, episodes: list[Episode]) -> None:
    np.savez_compressed(
        path,
        obs=np.concatenate([e.features for e in episodes]).astype(np.float32),
        act=np.concatenate([e.actions for e in episodes]).astype(np.float32),
        episode_ends=np.cumsum([len(e.actions) for e in episodes]),
        success=np.array([e.success for e in episodes]),
        seed=np.array([e.seed for e in episodes]),
        demo_index=np.array([e.demo_index for e in episodes]),
    )


def load_episodes(path: Path) -> list[Episode]:
    d = np.load(path)
    starts = np.r_[0, d["episode_ends"][:-1]]
    return [Episode(d["obs"][a:b], d["act"][a:b], bool(s), int(seed), int(di))
            for a, b, s, seed, di in zip(starts, d["episode_ends"], d["success"], d["seed"],
                                         d["demo_index"])]
