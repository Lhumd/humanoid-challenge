"""Camera-based behaviour cloning: a small CNN on the agentview image plus proprioception
(end-effector position, sin/cos of 4x its yaw, gripper joint positions) predicts a chunk of
future actions. No object state reaches the policy.

Loss, chunking and temporal ensembling are reused from hc.policy, so the vision policy differs
from the state-based one only in what it observes.
"""

import multiprocessing as mp
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch
from torch import nn

from hc.datagen import HOLD_S, horizon_for, plan_for
from hc.evaluate import AFTER_STEPS, HOLD_STEPS, HORIZON, sample_perturbation
from hc.policy import ARM_DIMS, STD_FLOOR, TemporalEnsembler, bc_loss, chunk_indices
from hc.sim import ACTION_DIM, CONTROL_FREQ, displace_cube, make_env, policy_features, reset
from hc.types import Demo

PROPRIO_DIM = 7
CAMERA = "agentview"


def proprio(obs: dict) -> np.ndarray:
    """First PROPRIO_DIM entries of policy_features: eef_pos, sin/cos(4 yaw), gripper qpos."""
    return policy_features(obs)[:PROPRIO_DIM]


# ---------------------------------------------------------------- data generation

_worker: dict = {}


class AgentviewCamera:
    """Renders `camera` from an env built without robosuite's camera observables.

    robosuite's own offscreen renderer is rebuilt on every hard reset and leaked ~250 MB per
    episode here, and building the env with a camera changes the sampled layouts; a
    mujoco.Renderer on the camera-less env keeps the exact eval layouts. Call `attach` after
    every reset (the hard reset replaces the model). Images are upright (no [::-1] needed)."""

    def __init__(self, size: int, camera: str = CAMERA):
        self.size = size
        self.camera = camera
        self.renderer = None

    def attach(self, env) -> None:
        if self.renderer is not None:
            self.renderer.close()
        self.env = env
        self.renderer = mujoco.Renderer(env.sim.model._model, self.size, self.size)

    def render(self) -> np.ndarray:
        self.renderer.update_scene(self.env.sim.data._data, camera=self.camera)
        return self.renderer.render().copy()


def _init_data_worker(demos: list[Demo], horizon: int, camera: str, size: int) -> None:
    torch.set_num_threads(1)
    _worker["env"] = make_env(horizon=horizon)
    _worker["demos"] = demos
    _worker["cam"] = AgentviewCamera(size, camera)


def _run_data_task(task: tuple[int, int, float, float, int]) -> dict:
    """hc.datagen.execute_plan (same plan, clean/noisy actions and rng draws), additionally
    rendering the observation every `stride` steps. Image j shows the state that the clean
    action at step j * stride was computed from."""
    seed, demo_index, speed, noise, stride = task
    env, cam = _worker["env"], _worker["cam"]
    obs = reset(env, seed)
    cam.attach(env)
    plan, ctl = plan_for(obs, _worker["demos"][demo_index], speed)
    rng = np.random.default_rng(seed)
    props, actions, images = [], [], []
    for k in range(len(plan) + round(HOLD_S * CONTROL_FREQ)):
        i = min(k, len(plan) - 1)
        clean = ctl.action(obs, plan.pos[i], plan.yaw[i], plan.grip[i])
        if k % stride == 0:
            images.append(cam.render())
        props.append(proprio(obs))
        actions.append(clean)
        executed = clean.copy()
        executed[:6] = np.clip(clean[:6] + rng.normal(0.0, noise, 6), -1, 1)
        obs, *_ = env.step(executed)
    return {"seed": seed, "demo_index": demo_index, "success": bool(env._check_success()),
            "proprio": np.array(props, np.float32), "act": np.array(actions, np.float32),
            "images": np.stack(images)}


def generate_camera_data(demos: list[Demo], n_max: int, out_path: Path, time_budget_s: float,
                         workers: int = 16, camera: str = CAMERA, size: int = 84,
                         stride: int = 2, action_noise: float = 0.1) -> dict:
    """Seeds 0..n-1 with the same (demo, speed, noise) tasks as generate(seed_offset=0); stops
    taking results after `time_budget_s` (results arrive in seed order, so the kept seeds are a
    prefix). Saves successful episodes: images every `stride` steps, proprio/actions every step."""
    speeds = np.random.default_rng(0).uniform(0.8, 1.2, n_max)
    tasks = [(i, i % len(demos), float(speeds[i]), action_noise, stride) for i in range(n_max)]
    horizon = horizon_for(demos, float(speeds.min()))
    start = time.time()
    results = []
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(workers, _init_data_worker, (demos, horizon, camera, size))
    for r in pool.imap(_run_data_task, tasks, chunksize=1):
        results.append(r)
        if len(results) % 25 == 0:
            print(f"  {len(results)} episodes, {time.time() - start:.0f}s", flush=True)
        if time.time() - start > time_budget_s:
            break
    pool.terminate()
    pool.join()
    wall = time.time() - start
    ok = [r for r in results if r["success"]]
    np.savez(out_path,
             images=np.concatenate([r["images"] for r in ok]),
             image_ends=np.cumsum([len(r["images"]) for r in ok]),
             proprio=np.concatenate([r["proprio"] for r in ok]).astype(np.float32),
             act=np.concatenate([r["act"] for r in ok]).astype(np.float32),
             episode_ends=np.cumsum([len(r["act"]) for r in ok]),
             seed=np.array([r["seed"] for r in ok]), stride=stride)
    return {"n_attempted": len(results), "n_success": len(ok), "wall_time_s": wall,
            "steps": int(sum(len(r["act"]) for r in ok)),
            "frames": int(sum(len(r["images"]) for r in ok)), "image_size": size,
            "stride": stride, "camera": camera}


# ---------------------------------------------------------------- model


@dataclass
class VisionConfig:
    image_size: int = 84
    proprio_dim: int = PROPRIO_DIM
    channels: int = 32
    chunk: int = 8
    hidden: int = 256
    keypoints: int = 32


class SpatialSoftmax(nn.Module):
    """(B, C, H, W) feature maps -> (B, 2C) expected x/y image coordinates per channel."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        attn = torch.softmax(x.flatten(2), -1).view(b, c, h, w)
        ys = torch.linspace(-1, 1, h, device=x.device)
        xs = torch.linspace(-1, 1, w, device=x.device)
        ex = (attn.sum(2) * xs).sum(-1)
        ey = (attn.sum(3) * ys).sum(-1)
        return torch.cat([ex, ey], 1)


class VisionBCPolicy(nn.Module):
    """CNN (4 conv layers) -> spatial-softmax keypoints, concatenated with normalised proprio,
    -> MLP -> (chunk, ACTION_DIM). Same output convention as hc.policy.BCPolicy so bc_loss
    applies: columns 0..5 normalised arm actions, column 6 the gripper-close logit."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        c = config.channels
        self.encoder = nn.Sequential(
            nn.Conv2d(3, c, 3, stride=2, padding=1), nn.GroupNorm(8, c), nn.GELU(),
            nn.Conv2d(c, c, 3, stride=2, padding=1), nn.GroupNorm(8, c), nn.GELU(),
            nn.Conv2d(c, 2 * c, 3, stride=1, padding=1), nn.GroupNorm(8, 2 * c), nn.GELU(),
            nn.Conv2d(2 * c, config.keypoints, 3, stride=1, padding=1),
            SpatialSoftmax(),
        )
        width = 2 * config.keypoints + config.proprio_dim
        self.head = nn.Sequential(
            nn.Linear(width, config.hidden), nn.LayerNorm(config.hidden), nn.GELU(),
            nn.Linear(config.hidden, config.hidden), nn.LayerNorm(config.hidden), nn.GELU(),
            nn.Linear(config.hidden, config.chunk * ACTION_DIM),
        )
        self.register_buffer("obs_mean", torch.zeros(config.proprio_dim))
        self.register_buffer("obs_std", torch.ones(config.proprio_dim))
        self.register_buffer("act_mean", torch.zeros(ARM_DIMS))
        self.register_buffer("act_std", torch.ones(ARM_DIMS))

    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """inputs = (images (B, H, W, 3) uint8, proprio (B, PROPRIO_DIM) raw) ->
        (B, chunk, ACTION_DIM) raw outputs."""
        images, prop = inputs
        x = images.permute(0, 3, 1, 2).float() / 127.5 - 1.0
        keys = self.encoder(x)
        p = (prop - self.obs_mean) / self.obs_std
        return self.head(torch.cat([keys, p], 1)).view(-1, self.config.chunk, ACTION_DIM)

    @torch.no_grad()
    def predict_chunk(self, image: np.ndarray, prop: np.ndarray) -> np.ndarray:
        out = self((torch.as_tensor(image)[None], torch.as_tensor(prop, dtype=torch.float32)[None]))[0]
        arm = (out[:, :ARM_DIMS] * self.act_std + self.act_mean).clamp(-1, 1)
        grip = 2 * torch.sigmoid(out[:, ARM_DIMS:]) - 1
        return torch.cat([arm, grip], dim=1).numpy()


def random_shift(images: torch.Tensor, pad: int = 4) -> torch.Tensor:
    """DrQ-style augmentation: replicate-pad by `pad` pixels and crop back at a random offset
    per sample. images (B, H, W, 3) uint8 -> same shape."""
    b, h, w, _ = images.shape
    x = images.permute(0, 3, 1, 2).float()
    x = nn.functional.pad(x, (pad, pad, pad, pad), mode="replicate")
    ox = torch.randint(0, 2 * pad + 1, (b,))
    oy = torch.randint(0, 2 * pad + 1, (b,))
    rows = (oy[:, None] + torch.arange(h)[None])[:, None, :, None].expand(b, 3, h, w + 2 * pad)
    x = torch.gather(x, 2, rows)
    cols = (ox[:, None] + torch.arange(w)[None])[:, None, None, :].expand(b, 3, h, w)
    x = torch.gather(x, 3, cols)
    return x.permute(0, 2, 3, 1).round().to(torch.uint8)


def training_samples(image_ends: np.ndarray, episode_ends: np.ndarray, stride: int,
                     chunk: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One sample per stored image: (image index, proprio/step index, act_idx (S, chunk),
    mask (S, chunk)). Image j of an episode shows step j * stride."""
    _, act_idx, mask = chunk_indices(episode_ends, 1, chunk)
    img_starts = np.r_[0, image_ends[:-1]]
    step_starts = np.r_[0, episode_ends[:-1]]
    img_idx, step_idx = [], []
    for i0, i1, s0 in zip(img_starts, image_ends, step_starts):
        j = np.arange(i1 - i0)
        img_idx.append(i0 + j)
        step_idx.append(s0 + j * stride)
    img_idx, step_idx = np.concatenate(img_idx), np.concatenate(step_idx)
    return img_idx, step_idx, act_idx[step_idx], mask[step_idx]


def train_vision_bc(data_path: Path, out_path: Path, updates: int, batch_size: int = 64,
                    lr: float = 1e-3, seed: int = 0, threads: int = 16, config: VisionConfig | None = None,
                    log_every: int = 100, time_budget_s: float | None = None) -> dict:
    """Train with a cosine-decayed lr for `updates` steps, or for as many as fit in
    `time_budget_s` (timed over the first 20 updates). Saves weights, config, normalisation
    buffers and hyper-parameters."""
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    d = np.load(data_path)
    images = torch.as_tensor(d["images"])
    prop, act = d["proprio"], d["act"]
    img_idx, step_idx, act_idx, mask = training_samples(d["image_ends"], d["episode_ends"],
                                                        int(d["stride"]), (config or VisionConfig()).chunk)
    config = config or VisionConfig(image_size=images.shape[1])
    policy = VisionBCPolicy(config)
    policy.obs_mean.copy_(torch.as_tensor(prop.mean(0)))
    policy.obs_std.copy_(torch.as_tensor(np.maximum(prop.std(0), STD_FLOOR)))
    policy.act_mean.copy_(torch.as_tensor(act[:, :ARM_DIMS].mean(0)))
    policy.act_std.copy_(torch.as_tensor(np.maximum(act[:, :ARM_DIMS].std(0), STD_FLOOR)))
    prop_t, act_t, mask_t = torch.as_tensor(prop), torch.as_tensor(act), torch.as_tensor(mask)

    opt = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)
    losses, log = [], []
    start = time.time()
    warmup = 50
    u = 0
    while u < updates:
        if time_budget_s is not None and u == 20:
            updates = max(int(time_budget_s / ((time.time() - start) / 20)), 100)
            print(f"time budget {time_budget_s:.0f}s -> {updates} updates", flush=True)
        for group in opt.param_groups:
            group["lr"] = lr * min(1.0, (u + 1) / warmup) * 0.5 * (1 + np.cos(np.pi * u / updates))
        idx = rng.integers(0, len(img_idx), batch_size)
        batch_images = random_shift(images[img_idx[idx]])
        loss, parts = bc_loss(policy, (batch_images, prop_t[step_idx[idx]]),
                              act_t[act_idx[idx]], mask_t[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if log_every and (u % log_every == 0 or u == updates - 1):
            window = float(np.mean(losses[-log_every:]))
            log.append({"update": u, "loss": window, **parts, "time_s": time.time() - start})
            print(f"update {u:5d}  loss {window:.4f}  arm {parts['arm_l1']:.4f}  "
                  f"grip {parts['grip_bce']:.4f}  {time.time() - start:.0f}s", flush=True)
        u += 1
    hparams = {"data_path": str(data_path), "updates": updates, "batch_size": batch_size, "lr": lr,
               "seed": seed, "augmentation": "random_shift pad 4"}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(config), "hparams": hparams, "state_dict": policy.state_dict()},
               out_path)
    return {"samples": len(img_idx), "updates": updates, "batch_size": batch_size,
            "first_loss": float(np.mean(losses[:log_every or 1])),
            "final_loss": float(np.mean(losses[-(log_every or 1):])), "log": log,
            "train_time_s": time.time() - start}


def load_vision_policy(path: Path) -> VisionBCPolicy:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    policy = VisionBCPolicy(VisionConfig(**ckpt["config"]))
    policy.load_state_dict(ckpt["state_dict"])
    return policy.eval()


class VisionPolicyRunner:
    """Closed-loop wrapper mirroring hc.policy.PolicyRunner: queries the network every step,
    temporally ensembles the chunks and binarises the gripper. `act` takes the env obs (for
    proprio) and the current camera image."""

    def __init__(self, policy: VisionBCPolicy, k: float = 0.01):
        self.policy = policy
        self.ensembler = TemporalEnsembler(policy.config.chunk, k)

    def reset(self) -> None:
        self.ensembler.reset()

    def act(self, obs: dict, image: np.ndarray) -> np.ndarray:
        action = self.ensembler.step(self.policy.predict_chunk(image, proprio(obs)))
        action[-1] = 1.0 if action[-1] > 0 else -1.0
        return action


# ---------------------------------------------------------------- evaluation


def run_vision_policy(env, cam: AgentviewCamera, runner: VisionPolicyRunner, seed: int,
                      horizon: int = HORIZON, perturb=None,
                      keep_frames: bool = False) -> tuple[bool, list[np.ndarray]]:
    """hc.evaluate.run_policy with image observations; identical success rule (stack held
    HOLD_STEPS + AFTER_STEPS consecutive steps while the policy acts) and horizon."""
    obs = reset(env, seed)
    cam.attach(env)
    runner.reset()
    frames = []
    held = 0
    for step in range(horizon):
        if perturb is not None and step == perturb.step:
            displace_cube(env, perturb.cube, perturb.dxy)
            obs = env._get_observations(force_update=True)
        image = cam.render()
        if keep_frames:
            frames.append(image)
        obs, *_ = env.step(runner.act(obs, image))
        held = held + 1 if env._check_success() else 0
        if held >= HOLD_STEPS + AFTER_STEPS:
            break
    return held >= HOLD_STEPS + AFTER_STEPS, frames


def _init_eval_worker(policy_path: Path, size: int) -> None:
    torch.set_num_threads(1)
    _worker["policy"] = load_vision_policy(policy_path)
    _worker["env"] = make_env()
    _worker["cam"] = AgentviewCamera(size)


def _run_eval_task(task: tuple[int, bool, bool]) -> dict:
    seed, perturbed, keep_frames = task
    env = _worker["env"]
    perturb = sample_perturbation(seed, reset(env, seed)) if perturbed else None
    success, frames = run_vision_policy(env, _worker["cam"], VisionPolicyRunner(_worker["policy"]),
                                        seed, HORIZON, perturb, keep_frames)
    return {"seed": seed, "perturbed": perturbed, "success": success, "frames": frames}


def evaluate_vision(policy_path: Path, seeds: list[int], size: int, video_seeds: set[int],
                    workers: int = 16, time_budget_s: float | None = None) -> list[dict]:
    """Nominal and pushed rollouts per seed, interleaved so a time-capped run still covers the
    first seeds in both conditions; results arriving after `time_budget_s` are dropped."""
    tasks = [(s, p, (not p) and s in video_seeds) for s in seeds for p in (False, True)]
    start = time.time()
    results = []
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(workers, _init_eval_worker, (policy_path, size))
    for r in pool.imap_unordered(_run_eval_task, tasks, chunksize=1):
        results.append(r)
        if len(results) % 10 == 0:
            print(f"  {len(results)}/{len(tasks)} rollouts, {time.time() - start:.0f}s", flush=True)
        if time_budget_s is not None and time.time() - start > time_budget_s:
            break
    pool.terminate()
    pool.join()
    return sorted(results, key=lambda r: (r["perturbed"], r["seed"]))
