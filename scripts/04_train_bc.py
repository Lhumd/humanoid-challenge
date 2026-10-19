"""Train the behaviour-cloning policy (MLP + action chunking) on the sim-verified episodes.

python scripts/04_train_bc.py --episodes data/processed/stack_main_episodes.npz \
    --out data/processed/stack_main_bc.pt
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hc.policy import train_bc  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=Path, default=ROOT / "data/processed/stack_main_episodes.npz")
    p.add_argument("--out", type=Path, default=ROOT / "data/processed/stack_main_bc.pt")
    p.add_argument("--max-episodes", type=int, default=None, help="random subset of this size")
    p.add_argument("--updates", type=int, default=4000, help="gradient steps (more overfits)")
    p.add_argument("--obs-noise", type=float, default=0.0, help="input noise, in feature stds")
    p.add_argument("--chunk", type=int, default=8, help="actions predicted per step")
    p.add_argument("--n-obs", type=int, default=1, help="feature frames stacked as input")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train_bc(args.episodes, args.out, chunk=args.chunk, seed=args.seed, n_obs=args.n_obs,
                       max_episodes=args.max_episodes, updates=args.updates,
                       obs_noise=args.obs_noise, log_every=5)
    args.out.with_suffix(".json").write_text(json.dumps(metrics, indent=2))
    print(f"{metrics['episodes']} episodes, {metrics['samples']} samples, {metrics['updates']} "
          f"updates: loss {metrics['first_loss']:.3f} -> {metrics['final_loss']:.3f}")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
