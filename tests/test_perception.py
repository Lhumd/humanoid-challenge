"""Perception round trip on synthetic landmark tracks with known ground truth."""

import json

import numpy as np
import pytest

import hc.synthetic_video as synthetic_video
from hc.lifting import focal_length, lift, solve_translation, table_frame, to_demos
from hc.segmentation import find_reps, load_events, reps_from_events
from hc.types import INDEX_TIP, THUMB_TIP, load_demos, make_synthetic_demo, save_demos

FPS = 30.0
N_REPS = 10


def synthetic_session(seed: int, **kwargs):
    rng = np.random.default_rng(seed)
    demos = [make_synthetic_demo(rng) for _ in range(N_REPS)]
    track, cam_T_table = synthetic_video.make_synthetic_track(demos, rng, **kwargs)
    offsets = np.cumsum([0] + [len(d.t) for d in demos])[:-1]
    return demos, offsets, track, cam_T_table


def true_pinch_cam(demos, cam_T_table) -> np.ndarray:
    pinch = np.concatenate([d.pinch for d in demos])
    return pinch @ cam_T_table[:3, :3].T + cam_T_table[:3, 3]


def rotation_error_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(a.T @ b) - 1) / 2, -1, 1))))


def test_pnp_lift_is_exact_without_noise(monkeypatch):
    monkeypatch.setattr(synthetic_video, "WORLD_NOISE", 0.0)
    demos, _, track, cam_T_table = synthetic_session(0, noise_px=0.0)
    seen = track.score > 0
    f = focal_length(track.width, 66.0)
    t = solve_translation(track.image_landmarks[seen], track.world_landmarks[seen], f,
                          track.width / 2, track.height / 2)
    points = track.world_landmarks[seen] + t[:, None]
    pinch = (points[:, THUMB_TIP] + points[:, INDEX_TIP]) / 2
    err = np.linalg.norm(pinch - true_pinch_cam(demos, cam_T_table)[seen], axis=1)
    assert err.max() < 1e-3


def test_lift_with_noise_within_a_centimetre():
    demos, _, track, cam_T_table = synthetic_session(1)
    traj = lift(track)
    err = np.linalg.norm(traj.pinch - true_pinch_cam(demos, cam_T_table), axis=1)
    assert traj.valid.all()
    assert err.mean() < 0.01


@pytest.mark.parametrize("seed", [0, 3])
def test_pipeline_recovers_events_and_table_frame_trajectories(seed):
    demos, offsets, track, cam_T_table = synthetic_session(seed)
    traj = lift(track)
    reps = find_reps(traj)
    assert len(reps) == N_REPS

    # Ground truth puts the release at the end of the place dwell, segmentation at its middle.
    for rep, demo, off in zip(reps, demos, offsets):
        assert abs(rep.grasp - off - demo.grasp_idx) / FPS <= 0.15
        assert abs(rep.release - off - demo.release_idx) / FPS <= 0.2

    estimate = table_frame(traj, reps)
    assert rotation_error_deg(estimate[:3, :3], cam_T_table[:3, :3]) < 5.0

    recovered = to_demos(traj, reps, estimate, "synthetic")
    for rep, demo, gt, off in zip(reps, recovered, demos, offsets):
        frames = np.arange(max(rep.start, off), min(rep.end, off + len(gt.t)))
        err = np.linalg.norm(demo.pinch[frames - rep.start] - gt.pinch[frames - off], axis=1)
        assert err.mean() < 0.02
        yaw_err = np.angle(np.exp(1j * (demo.yaw[frames - rep.start] - gt.yaw[frames - off])))
        assert np.abs(yaw_err).mean() < 0.15


def test_event_file_overrides_segmentation(tmp_path):
    demos, offsets, track, _ = synthetic_session(2)
    events = [[(off + d.grasp_idx) / FPS, (off + d.release_idx) / FPS] for d, off in zip(demos, offsets)]
    path = tmp_path / "clip_events.json"
    path.write_text(json.dumps(events))
    reps = reps_from_events(lift(track), load_events(path))
    assert [r.grasp for r in reps] == [off + d.grasp_idx for d, off in zip(demos, offsets)]
    assert [r.release for r in reps] == [off + d.release_idx for d, off in zip(demos, offsets)]
    assert all(r.start < r.grasp < r.release < r.end for r in reps)
    assert all(a.end <= b.start for a, b in zip(reps[:-1], reps[1:]))


def test_demos_are_saved_with_events(tmp_path):
    demos, _, track, _ = synthetic_session(4)
    traj = lift(track)
    reps = find_reps(traj)
    recovered = to_demos(traj, reps, table_frame(traj, reps), "synthetic")
    save_demos(tmp_path / "demos.npz", recovered)
    loaded = load_demos(tmp_path / "demos.npz")
    assert len(loaded) == N_REPS
    for a, b in zip(recovered, loaded):
        np.testing.assert_allclose(a.pinch, b.pinch)
        assert (a.grasp_idx, a.release_idx, a.source) == (b.grasp_idx, b.release_idx, b.source)
        assert b.closed.sum() == b.release_idx - b.grasp_idx
