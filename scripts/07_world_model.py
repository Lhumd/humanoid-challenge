"""Learned simulator: fit a feature-space dynamics model on demos plus closed-loop BC rollouts of
four scaling checkpoints, measure open-loop error vs horizon by phase, and score the checkpoints
by rolling them out inside the model from the 100 evaluation layouts.

python scripts/07_world_model.py
"""

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hc.datagen import Episode, load_episodes, save_episodes  # noqa: E402
from hc.evaluate import HORIZON, MAX_WORKERS, eval_seeds, run_policy  # noqa: E402
from hc.policy import PolicyRunner, load_policy  # noqa: E402
from hc.sim import make_env  # noqa: E402
from hc.world_model import (executed_actions, horizon_errors, predicted_success, rollout,  # noqa: E402
                            save_world_model, stacked, train_world_model, transitions)

SCALING = ROOT / "outputs/stack_main/eval/scaling"
CHECKPOINTS = {n: SCALING / f"bc_{n}.pt" for n in (10, 50, 150, 582)}
COLLECT_SEEDS = range(60000, 60300)
HORIZONS = (1, 5, 10, 20, 50)

_worker: dict = {}


def _init_worker() -> None:
    torch.set_num_threads(1)
    _worker["env"] = make_env()
    _worker["policies"] = {n: load_policy(p) for n, p in CHECKPOINTS.items()}


def _collect(task: tuple[int, int]) -> Episode:
    n, seed = task
    episode, _ = run_policy(_worker["env"], PolicyRunner(_worker["policies"][n]), seed, HORIZON)
    return Episode(episode.features, episode.actions, episode.success, seed, n)


def collect(tasks: list[tuple[int, int]], path: Path, workers: int) -> list[Episode]:
    """Closed-loop BC rollouts (checkpoint size stored in demo_index), cached at `path`."""
    if not path.exists():
        with mp.get_context("spawn").Pool(min(workers, MAX_WORKERS), _init_worker) as pool:
            save_episodes(path, pool.map(_collect, tasks, chunksize=1))
    return load_episodes(path)


def split(n: int, frac: float, seed: int) -> np.ndarray:
    """Boolean held-out mask over n episodes."""
    mask = np.zeros(n, bool)
    mask[np.random.default_rng(seed).permutation(n)[:round(frac * n)]] = True
    return mask


def plot_errors(errors: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    colors = {"free": "C0", "holding": "C3", "mixed": "C2"}
    for ax, part in zip(axes, ("eef", "cubeA")):
        for split_name, style in (("held_out", "-"), ("eval_seeds", "--")):
            for phase, color in colors.items():
                hs = [h for h in HORIZONS if errors[split_name][phase][h][part] is not None]
                ax.plot(hs, [1000 * errors[split_name][phase][h][part] for h in hs], style,
                        color=color, marker="o", ms=3, label=f"{phase} ({split_name})")
        hs = [h for h in HORIZONS if errors["held_out"]["free"][h][f"{part}_static"] is not None]
        ax.plot(hs, [1000 * errors["held_out"]["free"][h][f"{part}_static"] for h in hs], ":",
                color="gray", label="no-motion baseline (free, held_out)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(HORIZONS, [str(h) for h in HORIZONS])
        ax.set_xlabel("open-loop horizon (steps, 20 Hz)")
        ax.set_title(f"{part} position error")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("mean error (mm)")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_success(sizes: list[int], true: list[float], pred: list[float], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 3.5))
    x = np.arange(len(sizes))
    ax.bar(x - 0.2, 100 * np.array(true), 0.4, label="true (simulator)")
    ax.bar(x + 0.2, 100 * np.array(pred), 0.4, label="predicted (world model)")
    ax.set_xticks(x, [f"bc_{n}" for n in sizes])
    ax.set_ylabel("success on 100 eval layouts (%)")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--episodes", type=Path, default=ROOT / "data/processed/stack_main_episodes.npz")
    ap.add_argument("--results", type=Path, default=ROOT / "outputs/stack_main/eval/results.json")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/stack_main/world_model")
    ap.add_argument("--datagen-noise", type=float, default=0.1)
    ap.add_argument("--updates", type=int, default=6000)
    ap.add_argument("--rollout-steps", type=int, default=300)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t_start = time.time()
    torch.set_num_threads(min(args.workers, MAX_WORKERS))
    args.out.mkdir(parents=True, exist_ok=True)
    sizes = list(CHECKPOINTS)

    t0 = time.time()
    rollouts = collect([(sizes[i % len(sizes)], s) for i, s in enumerate(COLLECT_SEEDS)],
                       args.out / "rollouts_60000.npz", args.workers)
    eval_eps = collect([(582, s) for s in eval_seeds(100)], args.out / "rollouts_eval_bc582.npz",
                       args.workers)
    collect_s = time.time() - t0

    demos = load_episodes(args.episodes)
    episodes = demos + rollouts
    actions = [executed_actions(e, args.datagen_noise) for e in demos] + [e.actions for e in rollouts]
    held = split(len(episodes), 0.15, args.seed)
    pick = lambda mask: ([e for e, m in zip(episodes, mask) if m], [a for a, m in zip(actions, mask) if m])  # noqa: E731
    train_eps, train_act = pick(~held)
    val_eps, val_act = pick(held)

    t0 = time.time()
    model, train_metrics = train_world_model(transitions(train_eps, train_act),
                                             transitions(val_eps, val_act), updates=args.updates,
                                             seed=args.seed)
    train_s = time.time() - t0
    save_world_model(model, args.out / "world_model.pt", {"held_out_seeds": [e.seed for e in val_eps]})

    val_rollout = [m for e, m in zip(episodes, held) if e.demo_index in CHECKPOINTS]
    errors = {"held_out": horizon_errors(model, val_eps, val_act, HORIZONS),
              "held_out_demos": horizon_errors(model, *pick(held & (np.arange(len(episodes)) < len(demos))), HORIZONS),
              "held_out_rollouts": horizon_errors(model, *pick(held & (np.arange(len(episodes)) >= len(demos))), HORIZONS),
              "eval_seeds": horizon_errors(model, eval_eps, [e.actions for e in eval_eps], HORIZONS)}

    true_results = json.loads(args.results.read_text())
    true_rates = {r["episodes"]: r["success_rate"] for r in true_results["scaling"]}
    true_582 = np.array(true_results["nominal"]["bc"]["success"])
    init = np.stack([e.features[0] for e in eval_eps])
    sim = {}
    for n, path in CHECKPOINTS.items():
        feats, _ = rollout(model, load_policy(path), init, args.rollout_steps)
        ok = predicted_success(feats)
        sim[n] = {"true_rate": true_rates[n], "pred_rate": float(ok.mean()), "pred_success": ok.tolist(),
                  "ever_stacked_20": float(np.array([np.convolve(s, np.ones(20), "valid").max() >= 20
                                                     for s in stacked(feats)]).mean())}
    pred_582 = np.array(sim[582]["pred_success"])
    rho = spearmanr([sim[n]["true_rate"] for n in sizes], [sim[n]["pred_rate"] for n in sizes])
    replayed_582 = np.array([e.success for e in eval_eps])
    criterion_on_real = np.array([stacked(e.features[-20:]).all() for e in rollouts])
    real_success = np.array([e.success for e in rollouts])

    results = {
        "data": {"demo_episodes": len(demos), "rollout_episodes": len(rollouts),
                 "rollout_success_by_checkpoint": {n: float(np.mean([e.success for e in rollouts if e.demo_index == n]))
                                                   for n in sizes},
                 "held_out_episodes": int(held.sum()), "held_out_rollout_episodes": int(sum(val_rollout))},
        "training": train_metrics,
        "open_loop_error_m": errors,
        "learned_simulator": {"rollout_steps": args.rollout_steps, "criterion": "cube A within 1.5 cm (xy) and "
                              "1.5 cm (z) of cubeB + [0,0,0.045], gripper width > 0.06, for the last 20 steps",
                              "per_checkpoint": {n: {k: v for k, v in s.items() if k != "pred_success"}
                                                 for n, s in sim.items()},
                              "spearman_rho": float(rho.statistic), "spearman_p": float(rho.pvalue)},
        "bc_582_per_layout": {"true_succ_pred_succ": int((true_582 & pred_582).sum()),
                              "true_succ_pred_fail": int((true_582 & ~pred_582).sum()),
                              "true_fail_pred_succ": int((~true_582 & pred_582).sum()),
                              "true_fail_pred_fail": int((~true_582 & ~pred_582).sum()),
                              "agreement": float((true_582 == pred_582).mean())},
        "checks": {"eval_rollout_reproduces_results_json": float((replayed_582 == true_582).mean()),
                   "criterion_on_real_rollouts_vs_env_label": float((criterion_on_real == real_success).mean())},
        "wall_time_s": {"collect": collect_s, "train": train_s, "total": time.time() - t_start},
    }
    (args.out / "results.json").write_text(json.dumps(results, indent=2))
    plot_errors(errors, args.out / "error_vs_horizon.png")
    plot_success(sizes, [sim[n]["true_rate"] for n in sizes], [sim[n]["pred_rate"] for n in sizes],
                 args.out / "predicted_vs_true_success.png")
    print(json.dumps({k: results[k] for k in ("training", "learned_simulator", "bc_582_per_layout", "checks",
                                              "wall_time_s")}, indent=2))


if __name__ == "__main__":
    main()
