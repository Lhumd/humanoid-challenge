"""Training the SmolVLA student on the distillation datasets written by hc.distill.

Each sample is one recorded frame: both camera images, the robot state and the teacher's chunk at
that state. DAgger aggregates every round's data; sampling gives the newest round a fixed share of
each batch and up-weights frames whose label chunk opens or closes the gripper, the rare moments
that decide grasp and release timing.
"""

import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

from hc.sim import ACTION_DIM
from hc.vla_env import CHUNK, IMAGE_KEYS
from hc.vla_student import Student, StudentStats

SWITCH_WEIGHT = 3.0
FINAL_LR_RATIO = 0.025  # SmolVLA's cosine schedule ends at 2.5e-6 for a 1e-4 peak


def open_dataset(root: Path, episodes: list[int] | None = None):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    return LeRobotDataset(f"local/{root.name}", root=root, episodes=episodes)


def labels(dataset) -> tuple[np.ndarray, np.ndarray]:
    """(states (N, STATE_DIM), chunks (N, CHUNK, ACTION_DIM)) read from the parquet columns only,
    without decoding video (decoding in the main process breaks DataLoader workers' decoders)."""
    cols = dataset.hf_dataset.with_format("numpy")
    states = np.asarray(cols["observation.state"], np.float32)
    chunks = np.asarray(cols["teacher_chunk"], np.float32).reshape(-1, CHUNK, ACTION_DIM)
    return states, chunks


def stats_from(dataset) -> StudentStats:
    states, chunks = labels(dataset)
    return StudentStats.from_data(states, chunks.reshape(-1, ACTION_DIM))


class Frames(Dataset):
    """LeRobot frames -> (images (2, 3, H, W) in [0, 1], state, teacher chunk (CHUNK, ACTION_DIM))."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, i: int):
        item = self.dataset[i]
        images = torch.stack([item[k] for k in IMAGE_KEYS])
        return images, item["observation.state"], item["teacher_chunk"].view(CHUNK, ACTION_DIM)


def sample_weights(datasets: list, newest_share: float) -> np.ndarray:
    """Per-frame sampling weights over the concatenation of `datasets` (oldest first). The last
    dataset gets `newest_share` of the total weight; 0 leaves frames weighted uniformly."""
    weights = []
    for d in datasets:
        _, chunks = labels(d)
        grip = np.sign(chunks[:, :, -1])
        switch = (grip != grip[:, :1]).any(1)
        weights.append(np.where(switch, SWITCH_WEIGHT, 1.0))
    if len(datasets) > 1 and newest_share > 0:
        old, new = sum(w.sum() for w in weights[:-1]), weights[-1].sum()
        weights[-1] = weights[-1] * newest_share / new * old / (1 - newest_share)
    return np.concatenate(weights)


def lr_lambda(warmup: int, steps: int):
    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = min((step - warmup) / max(steps - warmup, 1), 1.0)
        return FINAL_LR_RATIO + (1 - FINAL_LR_RATIO) * 0.5 * (1 + math.cos(math.pi * progress))
    return f


@torch.no_grad()
def validation_loss(student: Student, loader: DataLoader, seed: int = 0) -> float:
    """Mean loss on the real action dimensions over `loader`, with fixed flow-matching noise and
    times so values are comparable across checkpoints."""
    student.policy.eval()
    dev = student.device
    losses = []
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        for images, states, chunks in loader:
            b = len(states)
            noise = torch.randn(b, CHUNK, student.policy.config.max_action_dim)
            t = torch.distributions.Beta(1.5, 1.0).sample((b,)) * 0.999 + 0.001
            losses.append(student.loss(images.to(dev), states.to(dev), chunks.to(dev),
                                       noise.to(dev), t.to(dev), real_dims_only=True).item())
    student.policy.train()
    return float(np.mean(losses))


def train(student: Student, train_sets: list, val_set, out_dir: Path, steps: int, batch_size: int = 32,
          lr: float = 1e-4, warmup: int = 300, newest_share: float = 0.5, val_every: int = 500,
          log_every: int = 25, save_every: int = 1000, workers: int = 6, seed: int = 0) -> dict:
    """Fine-tune `student` for `steps` updates, saving it to `out_dir` every `save_every` updates
    and at the end. Returns the loss history."""
    torch.manual_seed(seed)
    weights = sample_weights(train_sets, newest_share)
    sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), steps * batch_size,
                                    replacement=True, generator=torch.Generator().manual_seed(seed))
    prefetch = {"prefetch_factor": 4, "persistent_workers": True} if workers > 0 else {}
    loader = DataLoader(ConcatDataset([Frames(d) for d in train_sets]), batch_size, sampler=sampler,
                        num_workers=workers, pin_memory=True, drop_last=True, **prefetch)
    val_loader = DataLoader(Frames(val_set), batch_size, shuffle=False, num_workers=2)
    params = student.trainable_parameters()
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(warmup, steps))
    history = {"step": [], "loss": [], "val_step": [], "val_loss": []}
    history["val_step"].append(0)
    history["val_loss"].append(validation_loss(student, val_loader))
    print(f"step 0  val {history['val_loss'][-1]:.4f}", flush=True)
    student.policy.train()
    dev = student.device
    start, running = time.time(), []
    for step, (images, states, chunks) in enumerate(loader, start=1):
        loss = student.loss(images.to(dev, non_blocking=True), states.to(dev, non_blocking=True),
                            chunks.to(dev, non_blocking=True))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 10.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        sched.step()
        running.append(loss.item())
        if step % log_every == 0:
            history["step"].append(step)
            history["loss"].append(float(np.mean(running)))
            print(f"step {step}  loss {history['loss'][-1]:.4f}  lr {sched.get_last_lr()[0]:.2e}  "
                  f"{(time.time() - start) / step:.2f} s/step", flush=True)
            running = []
        if step % val_every == 0 or step == steps:
            history["val_step"].append(step)
            history["val_loss"].append(validation_loss(student, val_loader))
            print(f"step {step}  val {history['val_loss'][-1]:.4f}", flush=True)
        if step % save_every == 0 and step < steps:
            student.save(out_dir)
    student.save(out_dir)
    history["wall_time_s"] = time.time() - start
    return history
