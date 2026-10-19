from pathlib import Path

import numpy as np
import pytest

from hc.distill import RECORD_STRIDE, DatasetWriter, EpisodeSpec, RolloutPool, dagger_specs, teacher_specs
from hc.sim import ACTION_DIM, make_env, reset
from hc.vla_env import CHUNK, STATE_DIM, StudentCameras, apply_visual_stress, privileged, student_state
from hc.vla_train import FINAL_LR_RATIO, labels, lr_lambda, open_dataset, sample_weights, stats_from

TEACHER = Path("data/processed/stack_main1600_bc.pt")


def test_student_state_is_libero_layout_without_cube_poses():
    env = make_env()
    obs = reset(env, 70_000)
    state = student_state(obs)
    assert state.shape == (STATE_DIM,) and state.dtype == np.float32
    np.testing.assert_allclose(state[:3], obs["robot0_eef_pos"], atol=1e-6)
    np.testing.assert_allclose(state[6:], obs["robot0_gripper_qpos"], atol=1e-6)
    assert abs(np.linalg.norm(state[3:6]) - np.pi) < 0.3  # gripper points down: ~180 degree rotation
    assert not np.isin(np.round(obs["cubeA_pos"], 5), np.round(state, 5)).any()
    assert privileged(obs).shape == (14,)


def test_cameras_match_robosuite_frames_rotated_180_degrees():
    """LeRobot's LIBERO data are robosuite frames rotated by 180 degrees; StudentCameras must agree."""
    env = make_env(camera="agentview", camera_size=64)
    obs = reset(env, 70_001)
    cams = StudentCameras(64, ("agentview",))
    cams.attach(env)
    ours = cams.render()[0].astype(float)
    libero = obs["agentview_image"][::-1, ::-1].astype(float)
    assert ours.shape == libero.shape == (64, 64, 3)
    assert np.abs(ours - libero).mean() < 1.0
    assert np.abs(ours - obs["agentview_image"][::-1].astype(float)).mean() > 5.0  # not just upright
    cams.close()


def test_reused_renderer_matches_a_fresh_one_after_a_hard_reset():
    env = make_env()
    reset(env, 70_002)
    reused = StudentCameras(64)
    reused.attach(env)
    reset(env, 70_003)
    reused.attach(env)
    fresh = StudentCameras(64)
    fresh.attach(env)
    np.testing.assert_array_equal(reused.render(), fresh.render())
    reused.close()
    fresh.close()


@pytest.fixture(scope="module")
def teacher_episode():
    with RolloutPool(1, TEACHER) as pool:
        (result,) = pool.run(teacher_specs([70_000]), on_episode=lambda r: setattr(r, "kept", r.frames))
    return result


def test_teacher_episode_records_labels_at_every_other_step(teacher_episode):
    frames = teacher_episode.kept
    n = len(frames["state"])
    assert teacher_episode.success and n == (teacher_episode.steps + 1) // RECORD_STRIDE
    assert frames["images"].shape == (n, 2, 256, 256, 3) and frames["images"].dtype == np.uint8
    assert frames["chunk"].shape == (n, CHUNK, ACTION_DIM)
    assert set(np.unique(frames["chunk"][:, :, -1])) <= {-1.0, 1.0}
    assert np.abs(frames["chunk"][:, :, :-1]).max() <= 1.0
    assert (frames["executor"] == 0).all()
    assert (frames["env_step"] == RECORD_STRIDE * np.arange(n)).all()
    assert teacher_episode.frames is None  # dropped after the callback


def test_dataset_roundtrip_and_sampling_weights(teacher_episode, tmp_path):
    writer = DatasetWriter(tmp_path / "r0", "local/r0")
    teacher_episode.frames = teacher_episode.kept
    writer(teacher_episode)
    writer.finalize()
    teacher_episode.frames = None
    dataset = open_dataset(tmp_path / "r0")
    states, chunks = labels(dataset)
    np.testing.assert_allclose(states, teacher_episode.kept["state"], atol=1e-6)
    np.testing.assert_allclose(chunks, teacher_episode.kept["chunk"], atol=1e-6)
    stats = stats_from(dataset)
    assert len(stats.state_mean) == STATE_DIM and len(stats.action_std) == ACTION_DIM
    weights = sample_weights([dataset, dataset], newest_share=0.25)
    n = len(dataset)
    assert np.isclose(weights[n:].sum() / weights.sum(), 0.25)
    grip = np.sign(chunks[:, :, -1])
    switch = (grip != grip[:, :1]).any(1)
    assert switch.any() and weights[:n][switch].min() > weights[:n][~switch].max()
    uniform = sample_weights([dataset, dataset], newest_share=0.0)
    np.testing.assert_array_equal(uniform[:n], uniform[n:])


def test_dagger_specs_mix_switch_once_and_pure_student_episodes():
    specs = dagger_specs(list(range(1000)), switch_fraction=0.3, max_switch=50)
    switched = [s.switch_step for s in specs if s.switch_step is not None]
    assert 0.25 < len(switched) / len(specs) < 0.35
    assert min(switched) >= 0 and max(switched) < 50
    assert all(s.driver == "student" and s.record for s in specs)
    assert [s.switch_step for s in dagger_specs(list(range(1000)), 0.3, 50)] == [s.switch_step for s in specs]


def test_lr_schedule_warms_up_then_decays_to_floor():
    f = lr_lambda(warmup=10, steps=100)
    assert f(0) == pytest.approx(0.1) and f(9) == pytest.approx(1.0)
    assert f(10) == pytest.approx(1.0) and f(100) == pytest.approx(FINAL_LR_RATIO)
    assert all(f(s) >= f(s + 1) for s in range(10, 100))


@pytest.mark.parametrize("stress", ["move_b", "drop"])
def test_physical_stress_fires_once_cube_a_is_carried(stress):
    with RolloutPool(1, TEACHER) as pool:
        (result,) = pool.run([EpisodeSpec(1_000_001, driver="teacher", stress=stress, record=False)],
                             log_every_s=1e9)
    assert result.stress_step is not None and result.stress_step > 15


def test_visual_stresses_change_the_pixels_not_the_state():
    env = make_env()
    obs = reset(env, 1_000_002)
    cams = StudentCameras(64)
    cams.attach(env)
    base = cams.render().astype(float)
    for stress in ("dim", "table", "camera"):
        stressed_obs = reset(env, 1_000_002)
        cams.attach(env)
        apply_visual_stress(env, stress)
        assert np.abs(cams.render().astype(float) - base).mean() > 3.0, stress
        np.testing.assert_allclose(student_state(stressed_obs), student_state(obs))
    cams.close()
