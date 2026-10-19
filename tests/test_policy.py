"""BC policy and evaluation. Fixture data comes from synthetic human demos replayed through the real
retarget + datagen pipeline on two training layouts."""

from pathlib import Path

import numpy as np
import pytest
import torch

from hc.datagen import execute_plan, plan_for, save_episodes
from hc.evaluate import (EVAL_SEED_START, MIN_CUBE_GAP, PERTURB_RANGE, Perturbation, replay_demo,
                         rollout_policy, sample_perturbation)
from hc.policy import (BCPolicy, PolicyConfig, PolicyRunner, TemporalEnsembler, chunk_indices,
                       load_episode_arrays, load_policy, save_policy, train_bc)
from hc.sim import ACTION_DIM, OBS_DIM, make_env, reset
from hc.types import make_synthetic_demo


@pytest.fixture(scope="module")
def env():
    return make_env()


@pytest.fixture(scope="module")
def episodes_path(env, tmp_path_factory) -> Path:
    rng = np.random.default_rng(0)
    episodes = []
    for seed in (0, 1):
        obs = reset(env, seed)
        plan, ctl = plan_for(obs, make_synthetic_demo(rng), 1.0)
        episode, _ = execute_plan(env, obs, plan, ctl, rng, action_noise=0.1, seed=seed, demo_index=0)
        episodes.append(episode)
    path = tmp_path_factory.mktemp("data") / "episodes.npz"
    save_episodes(path, episodes)
    return path


@pytest.fixture(scope="module")
def trained(episodes_path, tmp_path_factory) -> tuple[Path, dict]:
    out = tmp_path_factory.mktemp("ckpt") / "bc.pt"
    return out, train_bc(episodes_path, out, chunk=4, n_obs=2, updates=300,
                         batch_size=64, hidden=64, depth=2)


def test_training_reduces_loss(trained):
    _, metrics = trained
    assert metrics["episodes"] == 2
    assert metrics["final_loss"] < 0.5 * metrics["first_loss"]


def test_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(0)
    policy = BCPolicy(PolicyConfig(n_obs=2, chunk=5, hidden=32, depth=2))
    policy.obs_mean.normal_()
    policy.act_std.uniform_(0.5, 2.0)
    save_policy(policy, tmp_path / "p.pt", {"lr": 1e-3})
    loaded = load_policy(tmp_path / "p.pt")
    x = np.random.default_rng(0).normal(size=(2, OBS_DIM)).astype(np.float32)
    assert loaded.config == policy.config
    np.testing.assert_allclose(loaded.predict_chunk(x), policy.eval().predict_chunk(x), atol=1e-6)


def test_runner_output_shape_and_range(trained):
    path, _ = trained
    runner = PolicyRunner(load_policy(path))
    obs = {"robot0_eef_pos": np.array([0.0, 0.0, 1.0]), "robot0_eef_quat": np.array([1.0, 0, 0, 0]),
           "robot0_gripper_qpos": np.array([0.02, -0.02]), "cubeA_pos": np.array([0.05, 0, 0.83]),
           "cubeA_quat": np.array([0, 0, 0, 1.0]), "cubeB_pos": np.array([-0.05, 0, 0.83])}
    runner.reset()
    for _ in range(6):
        action = runner.act(obs)
        assert action.shape == (ACTION_DIM,)
        assert np.all(np.abs(action[:6]) <= 1.0)
        assert action[6] in (-1.0, 1.0)
    assert len(runner.history) == 2


def test_temporal_ensembling_uniform():
    ens = TemporalEnsembler(chunk=3, k=0.0)
    chunks = [np.array([[1.0], [2.0], [3.0]]), np.array([[10.0], [20.0], [30.0]]),
              np.array([[100.0], [200.0], [300.0]]), np.array([[1000.0], [2000.0], [3000.0]])]
    out = [ens.step(c)[0] for c in chunks]
    assert out == pytest.approx([1.0, (2 + 10) / 2, (3 + 20 + 100) / 3, (30 + 200 + 1000) / 3])


def test_temporal_ensembling_weights_oldest_first():
    ens = TemporalEnsembler(chunk=2, k=1.0)
    ens.step(np.array([[0.0], [1.0]]))
    w = np.exp(-np.arange(2.0))
    assert ens.step(np.array([[5.0], [7.0]]))[0] == pytest.approx((w[0] * 1 + w[1] * 5) / w.sum())
    ens.reset()
    assert ens.step(np.array([[4.0], [4.0]]))[0] == pytest.approx(4.0)


def test_chunk_indices_respect_episode_bounds():
    obs_idx, act_idx, mask = chunk_indices(np.array([3, 5]), n_obs=2, chunk=3)
    np.testing.assert_array_equal(obs_idx[0], [0, 0])
    np.testing.assert_array_equal(obs_idx[3], [3, 3])
    np.testing.assert_array_equal(act_idx[1], [1, 2, 2])
    np.testing.assert_array_equal(mask[1], [True, True, False])
    np.testing.assert_array_equal(act_idx[3], [3, 4, 4])


def test_subsets_are_nested(episodes_path):
    one = load_episode_arrays(episodes_path, max_episodes=1, seed=3)
    both = load_episode_arrays(episodes_path, max_episodes=None, seed=3)
    assert len(one[2]) == 1 and len(both[2]) == 2
    assert any(np.array_equal(one[0], both[0][a:b]) for a, b in [(0, both[2][0]), (both[2][0], both[2][1])])


def test_perturbation_is_a_3_to_5_cm_push_away_from_b(env):
    for seed in range(EVAL_SEED_START, EVAL_SEED_START + 5):
        obs = reset(env, seed)
        p = sample_perturbation(seed, obs)
        assert PERTURB_RANGE[0] <= np.linalg.norm(p.dxy) <= PERTURB_RANGE[1]
        assert np.linalg.norm(obs["cubeA_pos"][:2] + p.dxy - obs["cubeB_pos"][:2]) >= MIN_CUBE_GAP
        assert sample_perturbation(seed, obs).dxy == pytest.approx(p.dxy)


def test_rollout_applies_perturbation(env, trained):
    path, _ = trained
    seed = EVAL_SEED_START
    start = reset(env, seed)["cubeA_pos"].copy()
    push = Perturbation(3, "A", np.array([0.04, 0.0]))
    success, frames = rollout_policy(env, PolicyRunner(load_policy(path)), seed, horizon=5,
                                     perturb=push)
    assert success is False and frames == []
    moved = env.sim.data.body_xpos[env.cubeA_body_id] - start
    assert moved[0] == pytest.approx(0.04, abs=0.01)


def test_replay_of_synthetic_demo_stacks(env):
    demo = make_synthetic_demo(np.random.default_rng(1))
    success, _ = replay_demo(env, demo, EVAL_SEED_START + 1)
    assert success
