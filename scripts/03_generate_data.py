"""Multiply ~10 human demos into hundreds of sim-verified robot episodes over random cube layouts.

python scripts/03_generate_data.py --demos data/processed/stack_main_demos.npz --n 600
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hc.datagen import generate  # noqa: E402
from hc.types import load_demos  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--demos", type=Path, default=ROOT / "data/processed/stack_main_demos.npz")
    p.add_argument("--n", type=int, default=600, help="layouts to attempt (seeds seed_offset..)")
    p.add_argument("--out", type=Path, default=ROOT / "data/processed/stack_main_episodes.npz")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--noise", type=float, default=0.1, help="std of the executed action noise")
    p.add_argument("--speed", type=float, nargs=2, default=(0.8, 1.2), help="replay speed range")
    p.add_argument("--seed-offset", type=int, default=0)
    p.add_argument("--robot", default="Panda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    demos = load_demos(args.demos)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    stats = generate(demos, args.n, args.out, args.workers, args.noise, tuple(args.speed),
                     args.seed_offset, args.robot)
    per_demo = " ".join(f"{100 * r:.0f}" for r in stats["per_demo_success"])
    print(f"{len(demos)} human demos -> {stats['n_success']}/{stats['n_attempted']} successful "
          f"episodes (yield {100 * stats['yield']:.1f}%) in {stats['wall_time_s']:.0f} s")
    print(f"per-demo success %: {per_demo}")
    print(f"saved {args.out} and {args.out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
