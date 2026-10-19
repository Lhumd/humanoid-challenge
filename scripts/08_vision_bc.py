"""Camera-based BC on the stack task: render datagen episodes from agentview, train a CNN policy
on images + proprioception (no object state), evaluate closed loop nominal and pushed.

Usage:
    python scripts/08_vision_bc.py data  --n-max 700 --budget 270 --size 84 --workers 12
    python scripts/08_vision_bc.py train --updates 100000 --time-budget 300
    python scripts/08_vision_bc.py eval  --n-seeds 50 --workers 14 --eval-budget 480
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hc.evaluate import eval_seeds  # noqa: E402
from hc.types import load_demos  # noqa: E402
from hc.vision_policy import evaluate_vision, generate_camera_data, train_vision_bc  # noqa: E402

OUT = Path("outputs/stack_main/vision")
DATA = OUT / "camera_episodes.npz"
POLICY = OUT / "vision_bc.pt"
DEMOS = Path("data/processed/stack_main_demos.npz")


def _update_results(**fields) -> None:
    path = OUT / "results.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    results.update(fields)
    path.write_text(json.dumps(results, indent=2))


def stage_data(args) -> None:
    stats = generate_camera_data(load_demos(DEMOS), args.n_max, DATA, args.budget, args.workers,
                                 size=args.size, stride=args.stride)
    print(stats, flush=True)
    _update_results(data=stats)


def stage_train(args) -> None:
    metrics = train_vision_bc(DATA, POLICY, args.updates, args.batch_size, args.lr,
                              threads=args.threads, time_budget_s=args.time_budget)
    print({k: v for k, v in metrics.items() if k != "log"}, flush=True)
    _update_results(train=metrics)


def grid_video(frames_list: list[list[np.ndarray]], path: Path, fps: int = 20) -> None:
    import imageio.v2 as imageio
    n = max(len(f) for f in frames_list)
    frames_list = frames_list + [[np.zeros_like(frames_list[0][0])]] * (6 - len(frames_list))
    padded = [f + [f[-1]] * (n - len(f)) for f in frames_list]
    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=6, macro_block_size=4)
    for t in range(n):
        top = np.concatenate([padded[c][t] for c in range(3)], 1)
        bottom = np.concatenate([padded[3 + c][t] for c in range(3)], 1)
        writer.append_data(np.concatenate([top, bottom], 0))
    writer.close()


def stage_eval(args) -> None:
    seeds = eval_seeds(args.n_seeds)
    size = json.loads((OUT / "results.json").read_text())["data"]["image_size"]
    start = time.time()
    results = evaluate_vision(POLICY, seeds, size, set(seeds[:6]), args.workers, args.eval_budget)
    wall = time.time() - start
    out = {}
    for name, pert in (("nominal", False), ("pushed", True)):
        done = [r for r in results if r["perturbed"] == pert]
        hits = [bool(r["success"]) for r in done]
        out[name] = {"success_rate": float(np.mean(hits)), "n": len(hits),
                     "seeds": [r["seed"] for r in done], "success": hits}
        print(f"{name}: {np.mean(hits):.2f} ({sum(hits)}/{len(hits)})", flush=True)
    video = [r["frames"] for r in results if r["frames"]]
    grid_video(video, OUT / "vision_rollouts.mp4")
    _update_results(eval=out, eval_wall_time_s=wall)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["data", "train", "eval"])
    p.add_argument("--n-max", type=int, default=600)
    p.add_argument("--budget", type=float, default=420.0)
    p.add_argument("--size", type=int, default=84)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--updates", type=int, default=3000)
    p.add_argument("--time-budget", type=float, default=None,
                   help="train for as many updates as fit in this many seconds")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--n-seeds", type=int, default=50)
    p.add_argument("--eval-budget", type=float, default=None,
                   help="stop collecting rollouts after this many seconds")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"data": stage_data, "train": stage_train, "eval": stage_eval}[args.stage](args)


if __name__ == "__main__":
    main()
