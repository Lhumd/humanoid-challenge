"""Phone video -> table-frame pick-and-place demos (data/processed/<clip>_demos.npz).

    python scripts/01_extract_demos.py --video data/raw/stack_main.mp4
    python scripts/01_extract_demos.py --synthetic 10      # same pipeline on synthetic landmarks

Diagnostics go to outputs/perception/: <clip>_timeseries.png, <clip>_reps.png, <clip>_overlay.mp4.
A hand-labelled data/raw/<clip>_events.json ([[grasp_s, release_s], ...]) overrides segmentation.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hc.lifting import lift, table_frame, to_demos, to_table
from hc.segmentation import Rep, load_events, pinch_speed, find_reps, reps_from_events
from hc.synthetic_video import make_synthetic_track
from hc.types import INDEX_TIP, THUMB_TIP, CameraTrajectory, Demo, HandTrack, make_synthetic_demo, save_demos

HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
                    (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17),
                    (17, 18), (18, 19), (19, 20)]
OVERLAY_WIDTH = 960


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", type=Path, default=ROOT / "data/raw/stack_main.mp4")
    p.add_argument("--hfov", type=float, default=66.0, help="horizontal field of view of the phone, degrees")
    p.add_argument("--hand-scale", type=float, default=1.0, help="your hand length / MediaPipe's (~18 cm)")
    p.add_argument("--events", type=Path, default=None, help="default: data/raw/<clip>_events.json if present")
    p.add_argument("--synthetic", type=int, default=0, help="run on N synthetic reps instead of a video")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--retrack", action="store_true", help="ignore the cached hand track")
    p.add_argument("--no-video", action="store_true", help="skip the overlay video")
    p.add_argument("--stabilize", action="store_true",
                   help="experimental camera-motion compensation (drifts on long hand-held clips)")
    return p.parse_args()


def load_track(video: Path, cache: Path, retrack: bool) -> HandTrack:
    if cache.exists() and not retrack:
        return HandTrack.load(cache)
    from hc.hand_tracking import track_video
    track = track_video(video, ROOT / "models/hand_landmarker.task")
    track.save(cache)
    return track


def load_homographies(video: Path, track: HandTrack, cache: Path, retrack: bool) -> np.ndarray:
    """Homographies mapping every frame into frame 0's view."""
    if cache.exists() and not retrack:
        return np.load(cache)["homographies"]
    from hc.stabilize import MIN_INLIERS, estimate_homographies
    homographies, inliers = estimate_homographies(video, track)
    np.savez_compressed(cache, homographies=homographies, inliers=inliers)
    print(f"stabilisation: median {np.median(inliers[1:]):.0f} inliers between consecutive frames, "
          f"{np.mean(inliers[1:] < MIN_INLIERS):.0%} of steps assumed still")
    return homographies


def phase_labels(n: int, reps: list[Rep]) -> list[str]:
    labels = ["idle"] * n
    for i, r in enumerate(reps):
        for f in range(r.start, r.end):
            phase = "reach" if f < r.grasp else "carry" if f < r.release else "return"
            labels[f] = f"rep {i + 1}: {phase}"
    return labels


def plot_timeseries(traj: CameraTrajectory, reps: list[Rep], cam_T_table: np.ndarray, path: Path) -> None:
    height = to_table(traj.pinch, cam_T_table)[:, 2]
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    series = [(100 * height, "pinch height (cm)"), (100 * pinch_speed(traj), "pinch speed (cm/s)"),
              (100 * traj.aperture, "thumb-index aperture (cm)")]
    for ax, (y, label) in zip(axes, series):
        ax.plot(traj.t, y, lw=1, color="k")
        ax.plot(traj.t[~traj.valid], y[~traj.valid], ".", color="0.7", ms=2)
        for r in reps:
            ax.axvline(traj.t[r.grasp], color="tab:green", lw=1)
            ax.axvline(traj.t[r.release], color="tab:red", lw=1)
            ax.axvspan(traj.t[r.start], traj.t[r.end - 1], color="tab:blue", alpha=0.05)
        ax.set_ylabel(label)
    axes[0].set_title("green: grasp, red: release, shaded: repetition, grey dots: interpolated gaps")
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_reps(demos: list[Demo], path: Path) -> None:
    fig, (top, side) = plt.subplots(1, 2, figsize=(12, 5))
    for d in demos:
        p = 100 * d.pinch
        for ax, (i, j) in ((top, (0, 1)), (side, (0, 2))):
            ax.plot(p[:, i], p[:, j], lw=1, alpha=0.7)
            ax.plot(*p[d.grasp_idx, [i, j]], "o", color="tab:green", ms=4)
            ax.plot(*p[d.release_idx, [i, j]], "s", color="tab:red", ms=4)
    top.set(xlabel="x (cm, pick -> place)", ylabel="y (cm)", title="top view, table frame", aspect="equal")
    side.set(xlabel="x (cm)", ylabel="z (cm, above table)", title="side view", aspect="equal")
    for ax in (top, side):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_overlay(video: Path, track: HandTrack, traj: CameraTrajectory, reps: list[Rep], path: Path) -> None:
    """Original frames at OVERLAY_WIDTH with landmarks, pinch point, aperture and phase."""
    scale = OVERLAY_WIDTH / track.width
    size = (OVERLAY_WIDTH, 2 * int(round(track.height * scale / 2)))
    labels = phase_labels(len(traj.t), reps)
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    writer = imageio.get_writer(path, fps=track.fps, codec="libx264", quality=6, macro_block_size=2)
    for i in range(len(track.score)):
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        if track.score[i] > 0:
            pts = (track.image_landmarks[i] * scale).astype(int)
            for a, b in HAND_CONNECTIONS:
                cv2.line(frame, tuple(pts[a]), tuple(pts[b]), (255, 255, 255), 1, cv2.LINE_AA)
            for pt in pts:
                cv2.circle(frame, tuple(pt), 2, (0, 200, 255), -1, cv2.LINE_AA)
            pinch = ((pts[THUMB_TIP] + pts[INDEX_TIP]) / 2).astype(int)
            closed = "carry" in labels[i]
            cv2.circle(frame, tuple(pinch), 6, (0, 0, 255) if closed else (0, 255, 0), 2, cv2.LINE_AA)
        text = f"{labels[i]}   aperture {100 * traj.aperture[i]:.1f} cm   t {traj.t[i]:.2f} s"
        cv2.rectangle(frame, (0, 0), (OVERLAY_WIDTH, 36), (0, 0, 0), -1)
        cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    writer.close()
    cap.release()


def report_ground_truth(demos: list[Demo], truth: list[Demo]) -> None:
    """Synthetic runs only: event timing and trajectory errors against the generating demos."""
    offsets = np.cumsum([0] + [len(d.t) for d in truth])
    fps = 1.0 / np.median(np.diff(truth[0].t))
    for k, (d, gt, off) in enumerate(zip(demos, truth, offsets)):
        start = int(d.source.rsplit(":", 1)[1].split("-")[0])
        frames = np.arange(max(start, off), min(start + len(d.t), off + len(gt.t)))
        err = np.linalg.norm(d.pinch[frames - start] - gt.pinch[frames - off], axis=1)
        print(f"  rep {k + 1}: grasp {(start + d.grasp_idx - off - gt.grasp_idx) / fps:+.2f} s, "
              f"release {(start + d.release_idx - off - gt.release_idx) / fps:+.2f} s, "
              f"pinch error mean {100 * err.mean():.2f} cm max {100 * err.max():.2f} cm")


def main() -> None:
    args = parse_args()
    processed, outputs = ROOT / "data/processed", ROOT / "outputs/perception"
    processed.mkdir(parents=True, exist_ok=True)
    outputs.mkdir(parents=True, exist_ok=True)

    truth: list[Demo] = []
    if args.synthetic:
        clip = f"synthetic{args.synthetic}"
        rng = np.random.default_rng(args.seed)
        truth = [make_synthetic_demo(rng) for _ in range(args.synthetic)]
        track, _ = make_synthetic_track(truth, rng, hfov_deg=args.hfov)
    else:
        clip = args.video.stem
        track = raw_track = load_track(args.video, processed / f"{clip}_track.npz", args.retrack)
        if args.stabilize:
            from hc.stabilize import stabilize_track
            homographies = load_homographies(args.video, track, processed / f"{clip}_homographies.npz", args.retrack)
            track = stabilize_track(track, homographies, args.hfov)
    print(f"{clip}: {len(track.score)} frames at {track.fps:.1f} fps, {track.width}x{track.height}, "
          f"hand found in {100 * np.mean(track.score > 0):.0f}% of frames")

    traj = lift(track, hfov_deg=args.hfov, hand_scale=args.hand_scale)
    events = args.events or ROOT / f"data/raw/{clip}_events.json"
    if events.exists():
        reps = reps_from_events(traj, load_events(events))
        print(f"segmentation: {len(reps)} reps from {events}")
    else:
        reps = find_reps(traj)
        print(f"segmentation: {len(reps)} reps found automatically")
    cam_T_table = table_frame(traj, reps)
    demos = to_demos(traj, reps, cam_T_table, clip)
    save_demos(processed / f"{clip}_demos.npz", demos)

    grasps = np.array([d.pinch[d.grasp_idx] for d in demos])
    releases = np.array([d.pinch[d.release_idx] for d in demos])
    lifts = [d.pinch[d.grasp_idx:d.release_idx, 2].max() - d.pinch[d.grasp_idx, 2] for d in demos]
    print(f"camera height above table {100 * -(cam_T_table[:3, :3].T @ cam_T_table[:3, 3])[2]:.1f} cm; "
          f"grasp height {100 * grasps[:, 2].mean():.1f} cm, release height {100 * releases[:, 2].mean():.1f} cm, "
          f"carry {100 * np.linalg.norm(releases - grasps, axis=1).mean():.1f} cm, lift {100 * np.mean(lifts):.1f} cm")
    print(f"rep durations (s): {np.round([d.t[-1] for d in demos], 1).tolist()}")
    if truth:
        report_ground_truth(demos, truth)

    plot_timeseries(traj, reps, cam_T_table, outputs / f"{clip}_timeseries.png")
    plot_reps(demos, outputs / f"{clip}_reps.png")
    if not args.synthetic and not args.no_video:
        write_overlay(args.video, raw_track, traj, reps, outputs / f"{clip}_overlay.mp4")
    print(f"wrote {processed / f'{clip}_demos.npz'} and diagnostics in {outputs}")


if __name__ == "__main__":
    main()
