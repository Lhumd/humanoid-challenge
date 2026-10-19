"""World model: brief training reduces loss, rollouts have the right shapes, and the batched
policy runner matches PolicyRunner's temporal ensembling."""

import numpy as np
import torch

from hc.datagen import Episode
from hc.policy import BCPolicy, PolicyConfig, TemporalEnsembler
from hc.sim import ACTION_DIM, OBS_DIM
from hc.world_model import (BatchPolicyRunner, WorldModel, WorldModelConfig, executed_actions,
                            horizon_errors, predicted_success, rollout, train_world_model, transitions)

SMALL = WorldModelConfig(hidden=64, depth=2)


def toy_episodes(n: int = 6, length: int = 40) -> list[Episode]:
    """Linear toy dynamics: positions drift with the first three action dims."""
    rng = np.random.default_rng(0)
    episodes = []
    for i in range(n):
        act = rng.uniform(-1, 1, (length, ACTION_DIM)).astype(np.float32)
        feat = np.zeros((length, OBS_DIM), np.float32)
        feat[0] = rng.normal(0, 0.1, OBS_DIM)
        for t in range(1, length):
            feat[t] = feat[t - 1]
            feat[t, :3] += 0.01 * act[t - 1, :3]
            feat[t, 7:10] += 0.005 * act[t - 1, :3]
        episodes.append(Episode(feat, act, False, i, -1))
    return episodes


def test_training_reduces_loss():
    eps = toy_episodes()
    data = transitions(eps, [e.actions for e in eps])
    _, metrics = train_world_model(data, data, SMALL, updates=300, batch_size=128)
    assert metrics["final_train_loss"] < 0.5 * metrics["first_loss"]


def test_rollout_and_error_shapes():
    torch.manual_seed(0)
    model = WorldModel(SMALL).eval()
    policy = BCPolicy(PolicyConfig()).eval()
    feats, acts = rollout(model, policy, np.zeros((3, OBS_DIM), np.float32), steps=12)
    assert feats.shape == (3, 13, OBS_DIM) and acts.shape == (3, 12, ACTION_DIM)
    assert np.allclose(feats[:, :, 15:18], feats[:, :, 7:10] - feats[:, :, 0:3], atol=1e-6)
    assert set(np.unique(acts[..., -1])) <= {-1.0, 1.0}
    assert predicted_success(feats).shape == (3,)
    eps = toy_episodes(2, 30)
    errors = horizon_errors(model, eps, [e.actions for e in eps], (1, 5))
    assert errors["free"][1]["n"] > 0


def test_batch_runner_matches_single_ensembler():
    torch.manual_seed(1)
    policy = BCPolicy(PolicyConfig()).eval()
    feats = np.random.default_rng(0).normal(0, 1, (10, 2, OBS_DIM)).astype(np.float32)
    batch = BatchPolicyRunner(policy)
    singles = [TemporalEnsembler(policy.config.chunk) for _ in range(2)]
    for t in range(10):
        got = batch.act(torch.as_tensor(feats[t])).numpy()
        for b, ens in enumerate(singles):
            want = ens.step(policy.predict_chunk(feats[t, b][None]))
            want[-1] = 1.0 if want[-1] > 0 else -1.0
            assert np.allclose(got[b], want, atol=1e-5)


def test_executed_actions_are_clipped_clean_plus_noise():
    ep = toy_episodes(1, 5)[0]
    executed = executed_actions(ep, 0.1)
    assert np.all(np.abs(executed) <= 1)
    assert np.array_equal(executed[:, 6], ep.actions[:, 6])
    assert not np.allclose(executed[:, :6], ep.actions[:, :6])
