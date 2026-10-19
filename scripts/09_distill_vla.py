"""Distil the privileged state-based teacher into a SmolVLA student that sees only cameras, the
robot's own state and the instruction: offline round 0 on teacher rollouts, then DAgger rounds in
which the student drives and the teacher labels every visited state.

Usage (large files go to $HC_VLA_ROOT, default /var/tmp/humanoid_vla; results to outputs/stack_main/vla):
    python scripts/09_distill_vla.py teacher-data --n 400 --workers 12
    python scripts/09_distill_vla.py train   --round 0 --steps 3000
    python scripts/09_distill_vla.py eval    --ckpt r0 --n 100
    python scripts/09_distill_vla.py collect --round 1 --n 200 --switch-fraction 0.5
    python scripts/09_distill_vla.py train   --round 1 --steps 1500
    python scripts/09_distill_vla.py dagger  --rounds 1,2,3 --switch-fractions 0.5,0.25,0
    python scripts/09_distill_vla.py eval    --teacher --n 100     # same harness, teacher driving
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hc.distill import (STRESSES, DatasetWriter, RolloutPool, dagger_specs, eval_specs,  # noqa: E402
                        round_seeds, teacher_specs)
from hc.evaluate import eval_seeds  # noqa: E402

ROOT = Path(os.environ.get("HC_VLA_ROOT", "/var/tmp/humanoid_vla"))
OUT = Path("outputs/stack_main/vla")
TEACHER = Path("data/processed/stack_main1600_bc.pt")
VAL_EVERY_EPISODE = 20


def data_root(round_index: int) -> Path:
    return ROOT / "data" / f"r{round_index}"


def ckpt_dir(name: str) -> Path:
    return ROOT / "ckpt" / name


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(centre - half), float(centre + half)]


def save_json(name: str, payload: dict) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(json.dumps(payload, indent=1))
    return path


def fresh_root(root: Path) -> Path:
    """A dataset folder left by a run that stopped before writing its summary is incomplete:
    remove it so the stage can be rerun (LeRobotDataset.create refuses an existing folder)."""
    if root.exists():
        shutil.rmtree(root)
    return root


def stage_teacher_data(args) -> None:
    """Teacher rollouts with DART noise, successes only: round 0 (`--name r0`), or extra teacher data
    on other layouts (e.g. `--name t1 --seed-round 5`) for the offline control."""
    start = time.time()
    root = fresh_root(ROOT / "data" / args.name)
    writer = DatasetWriter(root, f"local/{args.name}", keep=lambda r: r.success)
    try:
        with RolloutPool(args.workers, TEACHER) as pool:
            specs = teacher_specs(round_seeds(args.seed_round, args.n), args.noise)
            results = pool.run(specs, on_episode=writer)
    finally:
        writer.finalize()
    summary = {"n": args.n, "noise": args.noise, "seed_round": args.seed_round,
               "success": int(sum(r.success for r in results)), "episodes_kept": writer.episodes,
               "frames": writer.frames, "wall_time_s": time.time() - start}
    print(summary, flush=True)
    save_json(f"data_{args.name}.json", {**summary, "episodes": [r.summary() for r in results]})


def split_round0():
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    from hc.vla_train import open_dataset
    n = LeRobotDatasetMetadata("local/r0", root=data_root(0)).total_episodes
    val = [i for i in range(n) if i % VAL_EVERY_EPISODE == 0]
    train = [i for i in range(n) if i % VAL_EVERY_EPISODE != 0]
    return open_dataset(data_root(0), train), open_dataset(data_root(0), val)


def stage_train(args) -> None:
    from hc.vla_env import CHUNK
    from hc.vla_student import Student, init_checkpoint
    from hc.vla_train import open_dataset, stats_from, train

    r0_train, val = split_round0()
    train_sets = [r0_train] + [open_dataset(data_root(r)) for r in range(1, args.round + 1)]
    extra = args.extra_data.split(",") if args.extra_data else []
    train_sets += [open_dataset(ROOT / "data" / name) for name in extra]
    if args.init:
        student = Student.load(ckpt_dir(args.init), train=True)
    elif args.round == 0:
        student = Student.load(init_checkpoint(), stats=stats_from(r0_train), chunk=CHUNK, train=True)
    else:
        student = Student.load(ckpt_dir(f"r{args.round - 1}"), train=True)
    name = args.name or f"r{args.round}"
    history = train(student, train_sets, val, ckpt_dir(name), args.steps, args.batch_size, args.lr,
                    args.warmup, args.newest_share, args.val_every, workers=args.loader_workers,
                    seed=args.seed)
    save_json(f"train_{name}.json", {"round": args.round, "init": args.init, "extra_data": extra,
                                     "newest_share": args.newest_share, "steps": args.steps,
                                     "batch_size": args.batch_size, "lr": args.lr,
                                     "frames": [len(d) for d in train_sets], **history})


def stage_collect(args) -> None:
    from hc.vla_student import Student

    student = Student.load(ckpt_dir(args.ckpt or f"r{args.round - 1}"))
    start = time.time()
    writer = DatasetWriter(fresh_root(data_root(args.round)), f"local/r{args.round}")
    specs = dagger_specs(round_seeds(args.round, args.n), args.switch_fraction)
    try:
        with RolloutPool(args.workers, TEACHER) as pool:
            results = pool.run(specs, student, n_action_steps=args.n_action_steps, on_episode=writer,
                               noise_seed=args.round)
    finally:
        writer.finalize()
    pure = [r for r in results if r.switch_step is None]
    summary = {"round": args.round, "n": args.n, "switch_fraction": args.switch_fraction,
               "frames": writer.frames, "wall_time_s": time.time() - start,
               "pure_student_episodes": len(pure),
               "student_success": float(np.mean([r.success and r.takeover_step is None for r in pure])),
               "takeovers": {k: sum(r.takeover_reason == k for r in results) for k in ("switch", "stall")},
               "arm_l1": float(np.nanmean([r.arm_l1 for r in results])),
               "grip_disagree": float(np.nanmean([r.grip_disagree for r in results]))}
    print(summary, flush=True)
    save_json(f"data_r{args.round}.json", {**summary, "episodes": [r.summary() for r in results]})


class BlankImages:
    """Student with every camera image replaced by black: shows whether it uses its eyes."""

    def __init__(self, student):
        self.student = student
        self.chunk = student.chunk

    def predict(self, images, states, generator=None):
        return self.student.predict(np.zeros_like(images), states, generator)


def write_video(episodes: list[dict], path: Path, fps: int = 10) -> None:
    import imageio.v2 as imageio
    n = max(len(e) for e in episodes)
    padded = [np.concatenate([e, np.repeat(e[-1:], n - len(e), 0)]) for e in episodes]
    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=6, macro_block_size=4)
    for t in range(n):
        writer.append_data(np.concatenate([p[t] for p in padded], 1))
    writer.close()


def run_eval(student, n: int, workers: int, n_action_steps: int, video_seeds: int,
             teacher: bool, name: str, stress: str | None = None) -> dict:
    """Nominal and pushed rollouts on the held-out layouts, or one stress test (hc.distill.STRESSES)."""
    out = {}
    conditions = [(stress, False)] if stress else [("nominal", False), ("pushed", True)]
    with RolloutPool(workers, TEACHER) as pool:
        for label, perturbed in conditions:
            specs = eval_specs(eval_seeds(n), perturbed, stress)
            for spec in specs:
                spec.driver = "teacher" if teacher else "student"
            for spec in specs[:video_seeds]:
                spec.record = True
            frames: dict[int, np.ndarray] = {}

            def keep_frames(result):
                if result.frames is not None:
                    # undo the LIBERO mirror for viewing; agentview on top of the wrist camera
                    frames[result.seed] = np.concatenate([result.frames["images"][:, 0, :, ::-1],
                                                          result.frames["images"][:, 1, :, ::-1]], 1)

            start = time.time()
            results = pool.run(specs, student, n_action_steps=n_action_steps, on_episode=keep_frames)
            k = int(sum(r.success for r in results))
            out[label] = {"success": k, "n": n, "rate": k / n, "ci95": wilson(k, n),
                          "wall_time_s": time.time() - start,
                          "episodes": [r.summary() for r in results]}
            print(f"{label}: {k}/{n} = {k / n:.2f}  ci95 {np.round(wilson(k, n), 2)}", flush=True)
            if frames:
                out[label]["video"] = str(OUT / f"video_{name}_{label}.mp4")
                OUT.mkdir(parents=True, exist_ok=True)
                write_video([frames[s] for s in sorted(frames)], Path(out[label]["video"]))
    return out


def stage_eval(args) -> None:
    from hc.vla_student import Student

    student = None
    name = "teacher" if args.teacher else args.ckpt
    if not args.teacher:
        student = Student.load(ckpt_dir(args.ckpt))
        if args.blank_images:
            student = BlankImages(student)
            name += "_blank"
    if args.n_action_steps != 4:
        name += f"_k{args.n_action_steps}"
    if args.stress:
        name += f"_{args.stress}"
    out = run_eval(student, args.n, args.workers, args.n_action_steps, args.video_seeds, args.teacher,
                   name, args.stress)
    save_json(f"eval_{name}.json", {"ckpt": name, "n_action_steps": args.n_action_steps,
                                    "stress": args.stress, **out})


def stage_dagger(args) -> None:
    """Collect -> train -> quick eval for each round, unattended. Steps whose summary JSON already
    exists are skipped, so a restarted job resumes at the first unfinished step."""
    rounds = [int(r) for r in args.rounds.split(",")]
    fractions = [float(f) for f in args.switch_fractions.split(",")]
    if len(rounds) != len(fractions):
        raise ValueError(f"{len(rounds)} rounds but {len(fractions)} switch fractions")
    for r, frac in zip(rounds, fractions):
        if not (OUT / f"data_r{r}.json").exists():
            print(f"=== round {r}: collect (switch fraction {frac})", flush=True)
            stage_collect(argparse.Namespace(round=r, n=args.n, switch_fraction=frac, ckpt=None,
                                             workers=args.workers, n_action_steps=args.n_action_steps))
        if not (OUT / f"train_r{r}.json").exists():
            print(f"=== round {r}: train", flush=True)
            stage_train(argparse.Namespace(round=r, init=None, name=None, extra_data="", steps=args.steps,
                                           batch_size=args.batch_size, lr=args.lr, warmup=args.warmup,
                                           newest_share=args.newest_share, val_every=args.val_every,
                                           loader_workers=args.loader_workers, seed=args.seed))
        if not (OUT / f"eval_r{r}.json").exists():
            print(f"=== round {r}: eval", flush=True)
            stage_eval(argparse.Namespace(ckpt=f"r{r}", teacher=False, blank_images=False, stress=None,
                                          n=args.eval_n,
                                          workers=args.workers, n_action_steps=args.n_action_steps,
                                          video_seeds=0))


PLOT_ARMS = [  # (eval json name, label, train json name for the frame count)
    ("teacher", "teacher (sees cube poses)", None),
    ("r0", "student, offline (round 0)", "r0"),
    ("offline_matched", "student, offline, data-matched", "offline_matched"),
    ("r1", "student, DAgger round 1", "r1"),
    ("r2", "student, DAgger round 2", "r2"),
    ("r3_full", "student, DAgger round 3", "r3"),
    ("r3_blank", "round 3, cameras blacked out", None),
]


def stage_plot(args) -> None:
    """Success with Wilson 95 % intervals for every evaluated arm (left) and success against the
    number of labelled frames the student was trained on (right)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = [(name, label, train) for name, label, train in PLOT_ARMS if (OUT / f"eval_{name}.json").exists()]
    evals = {name: json.loads((OUT / f"eval_{name}.json").read_text()) for name, _, _ in arms}
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4.5), gridspec_kw={"width_ratios": [1.6, 1]})
    x = np.arange(len(arms))
    for offset, cond, colour in ((-0.2, "nominal", "#4C72B0"), (0.2, "pushed", "#DD8452")):
        rates = np.array([evals[n][cond]["rate"] for n, _, _ in arms])
        lo = rates - np.array([evals[n][cond]["ci95"][0] for n, _, _ in arms])
        hi = np.array([evals[n][cond]["ci95"][1] for n, _, _ in arms]) - rates
        ax.bar(x + offset, 100 * rates, 0.4, yerr=100 * np.stack([lo, hi]), capsize=3, color=colour,
               label="cube A pushed 3-5 cm" if cond == "pushed" else cond)
    ax.set_xticks(x, [f"{label}\n(n={evals[n]['nominal']['n']})" for n, label, _ in arms], rotation=25,
                  ha="right", fontsize=8)
    ax.set_ylabel("stacking success on held-out layouts (%)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)
    for cond, colour, marker in (("nominal", "#4C72B0", "o"), ("pushed", "#DD8452", "s")):
        for kind, names in (("DAgger", ["r0", "r1", "r2", "r3_full"]), ("offline", ["r0", "offline_matched"])):
            pts = [(sum(json.loads((OUT / f"train_{t}.json").read_text())["frames"]), evals[n][cond]["rate"])
                   for n, _, t in arms if n in names and t and (OUT / f"train_{t}.json").exists()]
            if len(pts) > 1:
                fx, fy = zip(*pts)
                # offline is only measured at its two end points, so its markers stay unconnected
                ax2.plot(np.array(fx) / 1000, 100 * np.array(fy), marker=marker, color=colour,
                         ls="-" if kind == "DAgger" else "none", ms=6 if kind == "DAgger" else 10,
                         mfc=colour if kind == "DAgger" else "none", label=f"{kind}, {cond}")
    ax2.set_xlabel("labelled frames the student was trained on (thousands)")
    ax2.set_ylabel("success (%)")
    ax2.set_ylim(0, 105)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "success.png", dpi=150)
    print(f"saved {OUT / 'success.png'}", flush=True)
    plot_stress(plt)


STRESS_ARMS = [("teacher", "teacher (cube poses)", "#555555"), ("r0", "student, offline round 0", "#C9A227"),
               ("offline_matched", "student, offline data-matched", "#55A868"),
               ("r3", "student, DAgger round 3", "#4C72B0")]
STRESS_LABELS = {"move_b": "cube B moved\nduring carry", "drop": "cube A slips\nafter lift",
                 "noise": "action noise\nσ = 0.15", "wide": "wider layouts\n±14 cm", "dim": "dim lights",
                 "table": "brown table", "camera": "camera moved\n5 cm"}


def plot_stress(plt) -> None:
    """Success with Wilson 95 % intervals per stress test, nominal for reference."""
    tests = [t for t in ("nominal",) + STRESSES
             if any((OUT / f"eval_{a}_{t}.json").exists() for a, _, _ in STRESS_ARMS)]
    if len(tests) < 2:
        return

    def get(arm: str, test: str):
        path = OUT / (f"eval_{arm}.json" if test == "nominal" else f"eval_{arm}_{test}.json")
        if test == "nominal" and arm == "r3":
            path = OUT / "eval_r3_full.json"
        return json.loads(path.read_text())[test] if path.exists() else None

    fig, ax = plt.subplots(figsize=(13, 4.5))
    width = 0.8 / len(STRESS_ARMS)
    x = np.arange(len(tests))
    for i, (arm, label, colour) in enumerate(STRESS_ARMS):
        cells = [get(arm, t) for t in tests]
        xs = [x[j] + (i - 1.5) * width for j, c in enumerate(cells) if c]
        rates = np.array([c["rate"] for c in cells if c])
        err = np.array([[c["rate"] - c["ci95"][0], c["ci95"][1] - c["rate"]] for c in cells if c]).T
        ax.bar(xs, 100 * rates, width, yerr=100 * err, capsize=2, color=colour, label=label)
    ax.set_xticks(x, ["nominal"] + [STRESS_LABELS[t] for t in tests[1:]], fontsize=8)
    ax.set_ylabel("stacking success, 100 held-out layouts (%)")
    ax.set_ylim(0, 105)
    visual = [i for i, t in enumerate(tests) if t in ("dim", "table", "camera")]
    if visual:  # tests right of the line change only the pixels, so the teacher is unaffected
        ax.axvline(visual[0] - 0.5, color="#999999", lw=0.8, ls=":")
    ax.legend(fontsize=8, ncol=4, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "stress.png", dpi=150)
    print(f"saved {OUT / 'stress.png'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)

    def train_args(p):
        p.add_argument("--steps", type=int, default=3000)
        p.add_argument("--batch-size", type=int, default=32)
        p.add_argument("--lr", type=float, default=1e-4)
        p.add_argument("--warmup", type=int, default=300)
        p.add_argument("--newest-share", type=float, default=0.5,
                       help="share of each batch from the last dataset; 0 = uniform over all frames")
        p.add_argument("--val-every", type=int, default=500)
        p.add_argument("--loader-workers", type=int, default=6)
        p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("teacher-data")
    p.add_argument("--name", default="r0")
    p.add_argument("--seed-round", type=int, default=0, help="layouts = round_seeds(seed_round, n)")
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--noise", type=float, default=0.1)
    p.add_argument("--workers", type=int, default=12)

    p = sub.add_parser("train")
    p.add_argument("--round", type=int, default=0)
    p.add_argument("--init", default=None, help="checkpoint name to start from (default: previous round)")
    p.add_argument("--name", default=None)
    p.add_argument("--extra-data", default="", help="comma-separated extra dataset names to train on")
    train_args(p)

    p = sub.add_parser("collect")
    p.add_argument("--round", type=int, required=True)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--switch-fraction", type=float, default=0.5)
    p.add_argument("--ckpt", default=None, help="student driving the rollouts (default: previous round)")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--n-action-steps", type=int, default=4)

    p = sub.add_parser("eval")
    p.add_argument("--ckpt", default="r0")
    p.add_argument("--teacher", action="store_true")
    p.add_argument("--blank-images", action="store_true")
    p.add_argument("--stress", choices=STRESSES, default=None,
                   help="run one stress test instead of the nominal and pushed conditions")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--n-action-steps", type=int, default=4)
    p.add_argument("--video-seeds", type=int, default=0)

    sub.add_parser("plot")

    p = sub.add_parser("dagger")
    p.add_argument("--rounds", default="1,2,3")
    p.add_argument("--switch-fractions", default="0.5,0.25,0")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--eval-n", type=int, default=50)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--n-action-steps", type=int, default=4)
    train_args(p)

    args = parser.parse_args()
    {"teacher-data": stage_teacher_data, "train": stage_train, "collect": stage_collect,
     "eval": stage_eval, "dagger": stage_dagger, "plot": stage_plot}[args.stage](args)


if __name__ == "__main__":
    main()
