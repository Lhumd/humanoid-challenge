"""Behaviour cloning: an MLP that maps the last few feature frames to a chunk of future actions.

Training targets are the clean tracking actions recorded during data generation. Position and
rotation actions are regressed with L1 in normalised units; the gripper is a binary
open/close decision trained with BCE. At test time `PolicyRunner` queries the network every step
and blends the overlapping chunk predictions with ACT-style temporal ensembling.
"""

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from hc.sim import ACTION_DIM, OBS_DIM, policy_features

ARM_DIMS = ACTION_DIM - 1
STD_FLOOR = 1e-3


@dataclass
class PolicyConfig:
    obs_dim: int = OBS_DIM
    n_obs: int = 1
    chunk: int = 8
    hidden: int = 256
    depth: int = 3


class BCPolicy(nn.Module):
    """MLP from (n_obs * obs_dim) normalised features to (chunk, ACTION_DIM) outputs.

    Output columns 0..5 are normalised arm actions, column 6 is the gripper-close logit.
    Normalisation statistics live in buffers so they travel with the state dict.
    """

    def __init__(self, config: PolicyConfig):
        super().__init__()
        self.config = config
        layers: list[nn.Module] = []
        width = config.n_obs * config.obs_dim
        for _ in range(config.depth):
            layers += [nn.Linear(width, config.hidden), nn.LayerNorm(config.hidden), nn.GELU()]
            width = config.hidden
        layers.append(nn.Linear(width, config.chunk * ACTION_DIM))
        self.net = nn.Sequential(*layers)
        self.register_buffer("obs_mean", torch.zeros(config.obs_dim))
        self.register_buffer("obs_std", torch.ones(config.obs_dim))
        self.register_buffer("act_mean", torch.zeros(ARM_DIMS))
        self.register_buffer("act_std", torch.ones(ARM_DIMS))

    def forward(self, obs_history: torch.Tensor) -> torch.Tensor:
        """obs_history (B, n_obs, obs_dim) raw features -> (B, chunk, ACTION_DIM) raw outputs."""
        x = (obs_history - self.obs_mean) / self.obs_std
        return self.net(x.flatten(1)).view(-1, self.config.chunk, ACTION_DIM)

    @torch.no_grad()
    def predict_chunk(self, obs_history: np.ndarray) -> np.ndarray:
        """(n_obs, obs_dim) raw features -> (chunk, ACTION_DIM) env actions.

        Arm columns are de-normalised and clipped to [-1, 1]; the gripper column is
        2 * p(close) - 1, so its sign is the gripper decision.
        """
        out = self(torch.as_tensor(obs_history, dtype=torch.float32)[None])[0]
        arm = (out[:, :ARM_DIMS] * self.act_std + self.act_mean).clamp(-1, 1)
        grip = 2 * torch.sigmoid(out[:, ARM_DIMS:]) - 1
        return torch.cat([arm, grip], dim=1).numpy()


def subset_indices(n_episodes: int, max_episodes: int | None, seed: int) -> np.ndarray:
    """Sorted indices of a random subset; growing sizes with one seed give nested subsets."""
    return np.sort(np.random.default_rng(seed).permutation(n_episodes)[:max_episodes])


def load_episode_arrays(path: Path, max_episodes: int | None = None,
                        seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read obs, act and episode_ends from an episodes npz, optionally keeping a random subset."""
    d = np.load(path)
    obs, act, ends = d["obs"], d["act"], d["episode_ends"]
    starts = np.concatenate([[0], ends[:-1]])
    keep = subset_indices(len(ends), max_episodes, seed)
    obs = np.concatenate([obs[starts[i]:ends[i]] for i in keep])
    act = np.concatenate([act[starts[i]:ends[i]] for i in keep])
    new_ends = np.cumsum([ends[i] - starts[i] for i in keep])
    return obs, act, new_ends


def chunk_indices(episode_ends: np.ndarray, n_obs: int, chunk: int) -> tuple[np.ndarray, ...]:
    """Index arrays for every training sample.

    Returns obs_idx (S, n_obs) with the history clamped at the episode start, act_idx (S, chunk)
    clamped at the episode end, and a validity mask (S, chunk) that is False on the padding.
    """
    starts = np.concatenate([[0], episode_ends[:-1]])
    obs_idx, act_idx, mask = [], [], []
    for s, e in zip(starts, episode_ends):
        t = np.arange(s, e)
        obs_idx.append(np.maximum(t[:, None] - np.arange(n_obs)[::-1][None], s))
        future = t[:, None] + np.arange(chunk)[None]
        act_idx.append(np.minimum(future, e - 1))
        mask.append(future < e)
    return np.concatenate(obs_idx), np.concatenate(act_idx), np.concatenate(mask)


def bc_loss(policy: BCPolicy, obs_hist: torch.Tensor, act_chunk: torch.Tensor,
            mask: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    """Masked L1 on normalised arm actions plus BCE on the gripper sign."""
    out = policy(obs_hist)
    arm_target = (act_chunk[..., :ARM_DIMS] - policy.act_mean) / policy.act_std
    grip_target = (act_chunk[..., ARM_DIMS] > 0).float()
    w = mask.float()
    arm = ((out[..., :ARM_DIMS] - arm_target).abs().mean(-1) * w).sum() / w.sum()
    grip = (nn.functional.binary_cross_entropy_with_logits(
        out[..., ARM_DIMS], grip_target, reduction="none") * w).sum() / w.sum()
    return arm + grip, {"arm_l1": arm.item(), "grip_bce": grip.item()}


def train_bc(episodes_path: Path, out_path: Path, epochs: int | None = None, chunk: int = 8,
             seed: int = 0, n_obs: int = 1, max_episodes: int | None = None, batch_size: int = 512,
             lr: float = 1e-3, updates: int = 4000, obs_noise: float = 0.0, hidden: int = 256,
             depth: int = 3, log_every: int = 0) -> dict:
    """Train a BCPolicy and save the checkpoint (weights, config, normalisation stats,
    hyper-parameters). Returns training metrics.

    The budget is `updates` gradient steps (rounded up to whole epochs) unless `epochs` is given.
    A fixed update budget keeps datasets of different sizes comparable; in development runs,
    closed-loop success dropped when the policy was optimised several times longer. `obs_noise`
    adds Gaussian noise, in units of each feature's std, to the training inputs.
    """
    torch.set_num_threads(6)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    obs, act, ends = load_episode_arrays(episodes_path, max_episodes, seed)
    obs_idx, act_idx, mask = chunk_indices(ends, n_obs, chunk)

    config = PolicyConfig(obs_dim=obs.shape[1], n_obs=n_obs, chunk=chunk, hidden=hidden, depth=depth)
    policy = BCPolicy(config)
    policy.obs_mean.copy_(torch.as_tensor(obs.mean(0)))
    policy.obs_std.copy_(torch.as_tensor(np.maximum(obs.std(0), STD_FLOOR)))
    policy.act_mean.copy_(torch.as_tensor(act[:, :ARM_DIMS].mean(0)))
    policy.act_std.copy_(torch.as_tensor(np.maximum(act[:, :ARM_DIMS].std(0), STD_FLOOR)))

    obs_t, act_t = torch.as_tensor(obs), torch.as_tensor(act)
    mask_t = torch.as_tensor(mask)
    n_samples = len(obs_idx)
    batches_per_epoch = max(n_samples // batch_size, 1)
    epochs = epochs or int(np.ceil(updates / batches_per_epoch))
    total = epochs * batches_per_epoch
    opt = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=total, pct_start=0.05)

    epoch_losses = []
    for epoch in range(epochs):
        perm = rng.permutation(n_samples)
        running = []
        for b in range(batches_per_epoch):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            obs_hist = obs_t[obs_idx[idx]]
            obs_hist = obs_hist + obs_noise * policy.obs_std * torch.randn_like(obs_hist)
            loss, parts = bc_loss(policy, obs_hist, act_t[act_idx[idx]], mask_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            running.append(loss.item())
        epoch_losses.append(float(np.mean(running)))
        if log_every and (epoch % log_every == 0 or epoch == epochs - 1):
            print(f"epoch {epoch:4d}  loss {epoch_losses[-1]:.4f}  "
                  f"arm {parts['arm_l1']:.4f}  grip {parts['grip_bce']:.4f}", flush=True)

    hparams = {"episodes_path": str(episodes_path), "max_episodes": max_episodes, "epochs": epochs,
               "updates": total, "batch_size": batch_size, "lr": lr, "obs_noise": obs_noise,
               "seed": seed}
    save_policy(policy, out_path, hparams)
    return {"episodes": len(ends), "samples": n_samples, "epochs": epochs, "updates": total,
            "first_loss": epoch_losses[0], "final_loss": epoch_losses[-1],
            "epoch_losses": epoch_losses}


def save_policy(policy: BCPolicy, path: Path, hparams: dict) -> None:
    """Checkpoint = architecture config, training hyper-parameters and the state dict
    (weights and normalisation buffers)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(policy.config), "hparams": hparams,
                "state_dict": policy.state_dict()}, path)


def load_policy(path: Path) -> BCPolicy:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    policy = BCPolicy(PolicyConfig(**ckpt["config"]))
    policy.load_state_dict(ckpt["state_dict"])
    return policy.eval()


class TemporalEnsembler:
    """ACT temporal ensembling: the action for step t averages every stored chunk that covers t,
    weighted exp(-k * j) with j = 0 for the oldest prediction."""

    def __init__(self, chunk: int, k: float = 0.01):
        self.chunk = chunk
        self.k = k
        self.chunks: list[tuple[int, np.ndarray]] = []
        self.t = 0

    def reset(self) -> None:
        self.chunks = []
        self.t = 0

    def step(self, new_chunk: np.ndarray) -> np.ndarray:
        """Add the chunk predicted at the current step and return the blended current action."""
        self.chunks.append((self.t, new_chunk))
        self.chunks = [(t0, c) for t0, c in self.chunks if self.t - t0 < self.chunk]
        preds = np.stack([c[self.t - t0] for t0, c in self.chunks])
        w = np.exp(-self.k * np.arange(len(preds)))
        self.t += 1
        return (w[:, None] * preds).sum(0) / w.sum()


class PolicyRunner:
    """Closed-loop wrapper: keeps the feature history, queries the policy every step and returns
    the temporally ensembled action with the gripper binarised to +1 (close) / -1 (open)."""

    def __init__(self, policy: BCPolicy, k: float = 0.01):
        self.policy = policy
        self.ensembler = TemporalEnsembler(policy.config.chunk, k)
        self.history: list[np.ndarray] = []

    def reset(self) -> None:
        self.ensembler.reset()
        self.history = []

    def act(self, obs: dict) -> np.ndarray:
        feat = policy_features(obs)
        if not self.history:
            self.history = [feat] * self.policy.config.n_obs
        self.history = self.history[1:] + [feat]
        action = self.ensembler.step(self.policy.predict_chunk(np.stack(self.history)))
        action[-1] = 1.0 if action[-1] > 0 else -1.0
        return action
