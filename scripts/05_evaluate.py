"""Evaluate on held-out layouts: open-loop replay of the retargeted human demos vs the BC policy,
both with and without a 3-5 cm push of cube A before the grasp, plus a data-scaling curve and
rollout videos.

python scripts/05_evaluate.py --policy data/processed/stack_main_bc.pt \
    --demos data/processed/stack_main_demos.npz --episodes data/processed/stack_main_episodes.npz \
    --n-eval 100 --scaling
"""

import argparse
import json
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

from hc.evaluate import Task, eval_seeds, evaluate, horizon_for_episodes, run_tasks  # noqa: E402
from hc.policy import train_bc  # noqa: E402
from hc.sim import CONTROL_FREQ  # noqa: E402

GREEN, RED, WHITE = (40, 200, 60), (220, 40, 40), (255, 255, 255)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy", type=Path, default=ROOT / "data/processed/stack_main_bc.pt")
    p.add_argument("--demos", type=Path, default=ROOT / "data/processed/stack_main_demos.npz")
    p.add_argument("--episodes", type=Path, default=ROOT / "data/processed/stack_main_episodes.npz",
                   help="training data for the scaling study")
    p.add_argument("--n-eval", type=int, default=100)
    p.add_argument("--out", type=Path, default=ROOT / "outputs/stack_main/eval/results.json")
    p.add_argument("--scaling", action="store_true", help="retrain on subsets and evaluate each")
    p.add_argument("--sizes", type=int, nargs="+", default=[10, 50, 150, 0],
                   help="0 = all; sizes are capped at the dataset size and deduplicated")
    p.add_argument("--updates", type=int, default=4000, help="training budget per scaling run")
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--camera", default="agentview", help="robosuite camera for the videos")
    p.add_argument("--workers", type=int, default=6)
    return p.parse_args()


def label_frame(frame: np.ndarray, text: str, outcome: bool | None) -> np.ndarray:
    """Caption a frame; once the episode is over, frame it green (success) or red (failure)."""
    img = np.ascontiguousarray(frame)
    cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA)
    if outcome is not None:
        cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), GREEN if outcome else RED, 6)
        cv2.putText(img, "success" if outcome else "fail", (6, img.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, GREEN if outcome else RED, 2, cv2.LINE_AA)
    return img


def write_grid_video(path: Path, results: list[dict], captions: list[str], cols: int = 3,
                     tail_s: float = 1.0) -> None:
    """Tile episodes into a rows x cols grid; finished tiles freeze on their last frame."""
    length = max(len(r["frames"]) for r in results) + int(tail_s * CONTROL_FREQ)
    rows = len(results) // cols
    with imageio.get_writer(path, fps=CONTROL_FREQ, macro_block_size=1) as writer:
        for t in range(length):
            tiles = [label_frame(r["frames"][min(t, len(r["frames"]) - 1)], c,
                                 r["success"] if t >= len(r["frames"]) - 1 else None)
                     for r, c in zip(results, captions)]
            grid = [np.concatenate(tiles[i * cols:(i + 1) * cols], axis=1) for i in range(rows)]
            writer.append_data(np.concatenate(grid, axis=0))


def plot_main(results: dict, path: Path) -> None:
    arms = [("replay", "open-loop replay\nof human demos"), ("bc", "BC policy\n(closed loop)")]
    x = np.arange(len(arms))
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for k, (cond, color) in enumerate([("nominal", "#4C72B0"), ("perturbed", "#DD8452")]):
        rates = [100 * results[cond][arm]["success_rate"] for arm, _ in arms]
        bars = ax.bar(x + (k - 0.5) * 0.38, rates, 0.38, label=cond, color=color)
        ax.bar_label(bars, fmt="%.0f", padding=2, fontsize=8)
    ax.set_xticks(x, [name for _, name in arms])
    ax.set_ylabel("success (%)")
    ax.set_ylim(0, 118)
    ax.set_title(f"Stack, {results['n_eval']} held-out layouts (cube A pushed 3-5 cm when perturbed)",
                 fontsize=8)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_scaling(scaling: list[dict], replay_rate: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    n = [s["episodes"] for s in scaling]
    ax.plot(n, [100 * s["success_rate"] for s in scaling], "o-", label="BC policy")
    ax.axhline(100 * replay_rate, color="gray", ls="--", label="open-loop replay")
    ax.set_xscale("log")
    ax.set_xlabel("training episodes (sim-verified, from the human demos)")
    ax.set_ylabel("success (%)")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def scaling_study(args: argparse.Namespace, seeds: list[int], out_dir: Path) -> list[dict]:
    n_total = len(np.load(args.episodes)["episode_ends"])
    rows = []
    for size in sorted({min(size or n_total, n_total) for size in args.sizes}):
        ckpt = out_dir / f"scaling/bc_{size}.pt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        metrics = train_bc(args.episodes, ckpt, max_episodes=size, updates=args.updates)
        rate = evaluate(seeds, policy_path=ckpt, horizon=args.horizon,
                        workers=args.workers)["bc"]["success_rate"]
        rows.append({"episodes": metrics["episodes"], "success_rate": rate,
                     "final_loss": metrics["final_loss"]})
        print(f"scaling: {metrics['episodes']:4d} episodes -> BC success {100 * rate:.0f}%", flush=True)
    return rows


def render_videos(args: argparse.Namespace, out_dir: Path) -> None:
    seeds = eval_seeds(6)
    tasks = [Task("bc", s, camera=args.camera) for s in seeds]
    tasks += [Task(arm, s, True, args.camera) for arm in ("replay", "bc") for s in seeds[:3]]
    results = run_tasks(tasks, args.policy, args.demos, args.horizon, args.workers)
    write_grid_video(out_dir / "bc_rollouts.mp4", results[:6], [f"BC  seed {s}" for s in seeds])
    write_grid_video(out_dir / "perturbed_replay_vs_bc.mp4", results[6:],
                     [f"replay  seed {s}  push" for s in seeds[:3]]
                     + [f"BC  seed {s}  push" for s in seeds[:3]])


def main() -> None:
    args = parse_args()
    args.horizon = horizon_for_episodes(args.episodes)
    out_dir = args.out.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = eval_seeds(args.n_eval)
    results: dict = {"n_eval": args.n_eval, "eval_seeds": [seeds[0], seeds[-1]],
                     "policy": str(args.policy), "demos": str(args.demos), "horizon": args.horizon}
    for cond, perturbed in (("nominal", False), ("perturbed", True)):
        results[cond] = evaluate(seeds, args.policy, args.demos, perturbed, args.horizon, args.workers)
        print(cond, {arm: f"{100 * r['success_rate']:.0f}%" for arm, r in results[cond].items()},
              flush=True)
    plot_main(results, out_dir / "success.png")
    if args.scaling:
        results["scaling"] = scaling_study(args, seeds, out_dir)
        plot_scaling(results["scaling"], results["nominal"]["replay"]["success_rate"],
                     out_dir / "scaling.png")
    args.out.write_text(json.dumps(results, indent=2))
    if not args.no_video:
        render_videos(args, out_dir)
    print(f"saved {args.out} and figures/videos in {out_dir}")


if __name__ == "__main__":
    main()
