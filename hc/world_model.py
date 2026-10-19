"""Learned dynamics over the 21-D policy features, used as an offline simulator for BC policies.

The model maps (features_t, executed action_t) to the change of the 15 independent features
(eef pos, eef yaw sin/cos, gripper qpos, cube A pos, cube A yaw sin/cos, cube B pos); the two
relative-position blocks are recomputed from the predicted positions so a rollout stays
self-consistent. Inputs and targets are normalised with statistics stored in the checkpoint.
"""

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from hc.datagen import Episode
from hc.policy import ARM_DIMS, BCPolicy, TemporalEnsembler
from hc.sim import ACTION_DIM, OBS_DIM

INDEP_DIMS = 15
EEF, GRIP, CUBE_A, CUBE_B = slice(0, 3), slice(5, 7), slice(7, 10), slice(12, 15)
STACK_OFFSET = np.array([0.0, 0.0, 0.045], np.float32)
OPEN_WIDTH = 0.06
HOLD_WIDTH = (0.03, 0.05)
HOLD_DIST = 0.03
STD_FLOOR = 1e-3


@dataclass
class WorldModelConfig:
    obs_dim: int = OBS_DIM
    act_dim: int = ACTION_DIM
    hidden: int = 512
    depth: int = 3


class WorldModel(nn.Module):
    """MLP (features, action) -> normalised delta of the independent features."""

    def __init__(self, config: WorldModelConfig):
        super().__init__()
        self.config = config
        layers: list[nn.Module] = []
        width = config.obs_dim + config.act_dim
        for _ in range(config.depth):
            layers += [nn.Linear(width, config.hidden), nn.LayerNorm(config.hidden), nn.GELU()]
            width = config.hidden
        layers.append(nn.Linear(width, INDEP_DIMS))
        self.net = nn.Sequential(*layers)
        self.register_buffer("in_mean", torch.zeros(width_in(config)))
        self.register_buffer("in_std", torch.ones(width_in(config)))
        self.register_buffer("delta_mean", torch.zeros(INDEP_DIMS))
        self.register_buffer("delta_std", torch.ones(INDEP_DIMS))

    def forward(self, feat: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """Raw (B, obs_dim) features and (B, act_dim) actions -> (B, 15) normalised deltas."""
        x = (torch.cat([feat, act], -1) - self.in_mean) / self.in_std
        return self.net(x)

    @torch.no_grad()
    def step(self, feat: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """One predicted transition in raw feature units, with derived blocks recomputed."""
        delta = self(feat, act) * self.delta_std + self.delta_mean
        return derive_features(feat[:, :INDEP_DIMS] + delta)


def width_in(config: WorldModelConfig) -> int:
    return config.obs_dim + config.act_dim


def derive_features(indep: torch.Tensor) -> torch.Tensor:
    """(B, 15) independent features -> (B, 21) with cubeA - eef and cubeB - cubeA appended."""
    return torch.cat([indep, indep[:, CUBE_A] - indep[:, EEF], indep[:, CUBE_B] - indep[:, CUBE_A]], -1)


def executed_actions(episode: Episode, action_noise: float) -> np.ndarray:
    """Reconstruct the actions hc.datagen actually executed: clean actions plus the arm noise
    drawn from default_rng(seed), one normal(0, action_noise, 6) draw per step."""
    rng = np.random.default_rng(episode.seed)
    executed = episode.actions.copy()
    for t in range(len(executed)):
        executed[t, :6] = np.clip(executed[t, :6] + rng.normal(0.0, action_noise, 6), -1, 1)
    return executed


def transitions(episodes: list[Episode], actions: list[np.ndarray]) -> tuple[np.ndarray, ...]:
    """Stack (feat_t, act_t, feat_{t+1}) over all consecutive pairs inside each episode."""
    feat = np.concatenate([e.features[:-1] for e in episodes])
    act = np.concatenate([a[:-1] for a in actions])
    nxt = np.concatenate([e.features[1:] for e in episodes])
    return feat, act, nxt


def train_world_model(train: tuple[np.ndarray, ...], val: tuple[np.ndarray, ...],
                      config: WorldModelConfig = WorldModelConfig(), updates: int = 6000,
                      batch_size: int = 1024, lr: float = 1e-3, seed: int = 0) -> tuple[WorldModel, dict]:
    """Fit a WorldModel with MSE on normalised deltas. Returns the model and train/val losses."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    feat, act, nxt = (torch.as_tensor(x) for x in train)
    model = WorldModel(config)
    inputs = torch.cat([feat, act], -1)
    delta = (nxt - feat)[:, :INDEP_DIMS]
    model.in_mean.copy_(inputs.mean(0))
    model.in_std.copy_(inputs.std(0).clamp_min(STD_FLOOR))
    model.delta_mean.copy_(delta.mean(0))
    model.delta_std.copy_(delta.std(0).clamp_min(STD_FLOOR))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=updates, pct_start=0.05)
    losses = []
    for _ in range(updates):
        idx = torch.as_tensor(rng.integers(0, len(feat), batch_size))
        loss = delta_loss(model, feat[idx], act[idx], nxt[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())
    model.eval()
    with torch.no_grad():
        metrics = {"first_loss": float(np.mean(losses[:50])), "final_train_loss": float(np.mean(losses[-200:])),
                   "train_loss": delta_loss(model, feat, act, nxt).item(),
                   "val_loss": delta_loss(model, *(torch.as_tensor(x) for x in val)).item(),
                   "n_train": len(feat), "n_val": len(val[0]), "updates": updates}
    return model, metrics


def delta_loss(model: WorldModel, feat: torch.Tensor, act: torch.Tensor, nxt: torch.Tensor) -> torch.Tensor:
    target = ((nxt - feat)[:, :INDEP_DIMS] - model.delta_mean) / model.delta_std
    return ((model(feat, act) - target) ** 2).mean()


def save_world_model(model: WorldModel, path: Path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(model.config), "meta": meta, "state_dict": model.state_dict()}, path)


def load_world_model(path: Path) -> WorldModel:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = WorldModel(WorldModelConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()


class BatchPolicyRunner:
    """PolicyRunner for B rollouts in lockstep, acting on features instead of obs dicts.

    Reuses TemporalEnsembler on chunks flattened to (chunk, B * ACTION_DIM), so the blending and
    the gripper binarisation are exactly PolicyRunner's. Requires n_obs == 1."""

    def __init__(self, policy: BCPolicy, k: float = 0.01):
        assert policy.config.n_obs == 1
        self.policy = policy
        self.ensembler = TemporalEnsembler(policy.config.chunk, k)

    def reset(self) -> None:
        self.ensembler.reset()

    @torch.no_grad()
    def act(self, feat: torch.Tensor) -> torch.Tensor:
        """(B, obs_dim) features -> (B, ACTION_DIM) actions."""
        out = self.policy(feat[:, None])
        arm = (out[..., :ARM_DIMS] * self.policy.act_std + self.policy.act_mean).clamp(-1, 1)
        grip = 2 * torch.sigmoid(out[..., ARM_DIMS:]) - 1
        chunk = torch.cat([arm, grip], -1).numpy()
        flat = chunk.transpose(1, 0, 2).reshape(chunk.shape[1], -1)
        action = self.ensembler.step(flat).reshape(len(feat), ACTION_DIM)
        action[:, -1] = np.where(action[:, -1] > 0, 1.0, -1.0)
        return torch.as_tensor(action, dtype=torch.float32)


@torch.no_grad()
def rollout(model: WorldModel, policy: BCPolicy, init_feat: np.ndarray,
            steps: int = 300) -> tuple[np.ndarray, np.ndarray]:
    """Closed loop inside the model: the policy acts on predicted features.
    Returns features (B, steps + 1, obs_dim) and actions (B, steps, ACTION_DIM)."""
    runner = BatchPolicyRunner(policy)
    feat = torch.as_tensor(init_feat, dtype=torch.float32)
    feats, acts = [feat], []
    for _ in range(steps):
        act = runner.act(feat)
        feat = model.step(feat, act)
        feats.append(feat)
        acts.append(act)
    return torch.stack(feats, 1).numpy(), torch.stack(acts, 1).numpy()


@torch.no_grad()
def open_loop(model: WorldModel, feat0: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Replay recorded actions (B, H, ACTION_DIM) from start features (B, obs_dim) through the
    model. Returns predicted features (B, H + 1, obs_dim)."""
    feat = torch.as_tensor(feat0, dtype=torch.float32)
    act = torch.as_tensor(actions, dtype=torch.float32)
    feats = [feat]
    for h in range(act.shape[1]):
        feat = model.step(feat, act[:, h])
        feats.append(feat)
    return torch.stack(feats, 1).numpy()


def stacked(feats: np.ndarray, tol: float = 0.015) -> np.ndarray:
    """Per step: cube A within tol horizontally and vertically of its stacked position on
    cube B, with the gripper open. feats (..., obs_dim) -> bool (...)."""
    err = feats[..., CUBE_A] - (feats[..., CUBE_B] + STACK_OFFSET)
    width = feats[..., GRIP][..., 0] - feats[..., GRIP][..., 1]
    return (np.linalg.norm(err[..., :2], axis=-1) < tol) & (np.abs(err[..., 2]) < tol) & (width > OPEN_WIDTH)


def predicted_success(feats: np.ndarray, last: int = 20) -> np.ndarray:
    """(B, T, obs_dim) rollout -> (B,) success: stacked for each of the last `last` steps."""
    return stacked(feats[:, -last:]).all(1)


def holding(feats: np.ndarray) -> np.ndarray:
    """Per step: fingers closed on a 4 cm cube and cube A within HOLD_DIST of the eef."""
    width = feats[..., GRIP][..., 0] - feats[..., GRIP][..., 1]
    near = np.linalg.norm(feats[..., CUBE_A] - feats[..., EEF], axis=-1) < HOLD_DIST
    return (width > HOLD_WIDTH[0]) & (width < HOLD_WIDTH[1]) & near


def horizon_errors(model: WorldModel, episodes: list[Episode], actions: list[np.ndarray],
                   horizons: tuple[int, ...] = (1, 5, 10, 20, 50), stride: int = 2) -> dict:
    """Open-loop position error of the eef and cube A after h model steps, per phase.

    A window [t0, t0 + h] is "free" if cube A is never held in it, "holding" if it is held at
    every step, "mixed" otherwise (grasp or release inside the window). Errors in metres; the
    "static" entries are the error of predicting no motion at all, for scale."""
    h_max = max(horizons)
    rows = {p: {h: {"eef": [], "cubeA": [], "eef_static": [], "cubeA_static": []} for h in horizons}
            for p in ("free", "holding", "mixed")}
    for ep, act in zip(episodes, actions):
        n = len(ep.features)
        starts = np.arange(0, n - 1, stride)
        idx = np.minimum(starts[:, None] + np.arange(h_max + 1)[None], n - 1)
        pred = open_loop(model, ep.features[starts], act[idx[:, :-1]])
        true = ep.features[idx]
        hold = holding(ep.features)[idx]
        for h in horizons:
            valid = starts + h <= n - 1
            window = hold[:, :h + 1]
            phase = np.where(window.all(1), "holding", np.where(window.any(1), "mixed", "free"))
            for part, sl in (("eef", EEF), ("cubeA", CUBE_A)):
                err = np.linalg.norm(pred[:, h, sl] - true[:, h, sl], axis=-1)
                static = np.linalg.norm(true[:, 0, sl] - true[:, h, sl], axis=-1)
                for p in rows:
                    m = valid & (phase == p)
                    rows[p][h][part] += err[m].tolist()
                    rows[p][h][f"{part}_static"] += static[m].tolist()
    return {p: {h: {k: (float(np.mean(v)) if v else None) for k, v in d.items()} | {"n": len(d["eef"])}
                for h, d in by_h.items()} for p, by_h in rows.items()}
