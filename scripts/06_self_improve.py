"""Success-filtered self-improvement (expert iteration / filtered BC): roll the current policy out
with exploration noise on fresh training layouts, keep the episodes the sim marks as stacked
(recording the policy's clean actions), add them to the dataset and retrain. The only reward is
the sim's success test; no scripted expert is involved. In a fraction of the rollouts cube A is
pushed before the grasp, so the policy's own successful recoveries enter the data.

python scripts/06_self_improve.py --episodes data/processed/stack_main_episodes.npz \
    --policy data/processed/stack_main_bc.pt
python scripts/06_self_improve.py --base-episodes 50   # start from a weaker policy trained here
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hc.datagen import load_episodes, save_episodes  # noqa: E402
from hc.evaluate import (EVAL_SEED_START, Task, eval_seeds, evaluate,  # noqa: E402
                         horizon_for_episodes, run_tasks)
from hc.policy import subset_indices, train_bc  # noqa: E402

EXPLORE_SEED_START = 50_000
TRAIN_SEED_END = 100_000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=Path, default=ROOT / "data/processed/stack_main_episodes.npz")
    p.add_argument("--policy", type=Path, default=ROOT / "data/processed/stack_main_bc.pt",
                   help="round-0 policy, trained on --episodes")
    p.add_argument("--base-episodes", type=int, default=None,
                   help="start instead from a policy trained here on a random subset of this size")
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--rollouts", type=int, default=200, help="exploration episodes per round")
    p.add_argument("--noise", type=float, default=0.1, help="std of the exploration action noise")
    p.add_argument("--perturb-frac", type=float, default=0.5,
                   help="fraction of exploration rollouts in which cube A is pushed before the grasp")
    p.add_argument("--updates", type=int, default=4000, help="training budget per round")
    p.add_argument("--n-eval", type=int, default=100)
    p.add_argument("--work-dir", type=Path, default=ROOT / "data/processed/stack_main_self_improve")
    p.add_argument("--out", type=Path, default=ROOT / "outputs/stack_main/eval/self_improve.json")
    p.add_argument("--workers", type=int, default=6)
    return p.parse_args()


def fresh_seeds(used: set[int], n: int) -> list[int]:
    """The first n training-range seeds from EXPLORE_SEED_START on that no dataset has used."""
    seeds = [s for s in range(EXPLORE_SEED_START, TRAIN_SEED_END) if s not in used][:n]
    assert len(seeds) == n and seeds[-1] < EVAL_SEED_START
    return seeds


def eval_both(policy: Path, seeds: list[int], horizon: int, workers: int) -> dict[str, float]:
    return {cond: evaluate(seeds, policy, None, perturbed, horizon, workers)["bc"]["success_rate"]
            for cond, perturbed in (("nominal", False), ("perturbed", True))}


def plot_rounds(rows: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    rounds = [r["round"] for r in rows]
    for cond, marker in (("nominal", "o-"), ("perturbed", "s--")):
        ax.plot(rounds, [100 * r[cond] for r in rows], marker, label=cond)
    for r in rows:
        ax.annotate(f"{r['dataset_episodes']} ep", (r["round"], 100 * r["nominal"]),
                    textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)
    ax.set_xticks(rounds)
    ax.set_xlabel("self-improvement round")
    ax.set_ylabel("BC success on held-out layouts (%)")
    ax.set_ylim(0, 110)
    ax.margins(x=0.15)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    seeds = eval_seeds(args.n_eval)
    horizon = horizon_for_episodes(args.episodes)
    episodes = load_episodes(args.episodes)
    used = {e.seed for e in episodes}
    policy = args.policy
    if args.base_episodes:
        episodes = [episodes[i] for i in subset_indices(len(episodes), args.base_episodes, 0)]
        save_episodes(args.work_dir / "episodes_round0.npz", episodes)
        policy = args.work_dir / "bc_round0.pt"
        train_bc(args.work_dir / "episodes_round0.npz", policy, updates=args.updates)
    rows = [{"round": 0, "dataset_episodes": len(episodes), **eval_both(policy, seeds, horizon, args.workers)}]
    print(rows[-1], flush=True)
    for rnd in range(1, args.rounds + 1):
        explore = fresh_seeds(used, args.rollouts)
        used |= set(explore)
        pushed = np.random.default_rng(rnd).random(len(explore)) < args.perturb_frac
        tasks = [Task("explore", s, bool(p), action_noise=args.noise) for s, p in zip(explore, pushed)]
        results = run_tasks(tasks, policy, None, horizon, args.workers)
        kept = [r["episode"] for r in results if r["success"]]
        for e in kept:
            e.demo_index = -rnd
        episodes += kept
        data = args.work_dir / f"episodes_round{rnd}.npz"
        save_episodes(data, episodes)
        policy = args.work_dir / f"bc_round{rnd}.pt"
        train_bc(data, policy, updates=args.updates)
        rows.append({"round": rnd, "explore_seeds": [explore[0], explore[-1]],
                     "explore_success": len(kept) / len(explore), "added": len(kept),
                     "added_pushed": int(sum(r["success"] and r["task"].perturbed for r in results)),
                     "dataset_episodes": len(episodes), **eval_both(policy, seeds, horizon, args.workers)})
        print(rows[-1], flush=True)
    args.out.write_text(json.dumps({"n_eval": args.n_eval, "noise": args.noise,
                                    "perturb_frac": args.perturb_frac,
                                    "base_episodes": args.base_episodes, "rounds": rows}, indent=2))
    plot_rounds(rows, args.out.with_suffix(".png"))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
