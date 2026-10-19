"""SmolVLA student: loading, normalisation, batched action-chunk inference and the training loss.

The student is LeRobot's SmolVLA (450M parameters) initialised from `lerobot/smolvla_libero`. It
predicts a chunk of CHUNK actions with flow matching from the two camera images, the 8-D robot
state and the instruction. Targets are the teacher's own predicted chunks at the visited states
(see hc.distill), so training does not rely on LeRobot's action-chunk padding, which 0.4.4 never
masks (it reads `actions_id_pad` instead of `action_is_pad`).

Normalisation statistics are computed once from the round-0 data and kept fixed afterwards, so
the student's output scale does not shift between DAgger rounds.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from safetensors.torch import load_file
from transformers import AutoTokenizer

from hc.sim import ACTION_DIM
from hc.vla_env import CHUNK, IMAGE_KEYS, TASK

INIT_REPO = "lerobot/smolvla_libero"
INIT_REVISION = "a24f1f380461b665158d52acd140152c79e8b36d"
STATS_FILE = "student_stats.json"
STD_FLOOR = 1e-3


def init_checkpoint() -> Path:
    """Local snapshot of the pinned smolvla_libero checkpoint (downloaded on first use, ~0.9 GB)."""
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(INIT_REPO, revision=INIT_REVISION))


@dataclass
class StudentStats:
    state_mean: list[float]
    state_std: list[float]
    action_mean: list[float]
    action_std: list[float]

    @classmethod
    def from_data(cls, states: np.ndarray, actions: np.ndarray) -> "StudentStats":
        """states (N, STATE_DIM), actions (N, ACTION_DIM) -> mean/std with a floor on the std."""
        return cls(states.mean(0).tolist(), np.maximum(states.std(0), STD_FLOOR).tolist(),
                   actions.mean(0).tolist(), np.maximum(actions.std(0), STD_FLOOR).tolist())

    def save(self, directory: Path) -> None:
        (directory / STATS_FILE).write_text(json.dumps(asdict(self), indent=1))

    @classmethod
    def load(cls, directory: Path) -> "StudentStats":
        return cls(**json.loads((directory / STATS_FILE).read_text()))


class Student:
    """SmolVLA plus the fixed normalisation, the tokenised instruction and the device."""

    def __init__(self, policy: SmolVLAPolicy, stats: StudentStats, device: str = "cuda"):
        self.policy = policy
        self.stats = stats
        self.device = device
        self.chunk = policy.config.chunk_size
        to = lambda x: torch.tensor(x, dtype=torch.float32, device=device)
        self.state_mean, self.state_std = to(stats.state_mean), to(stats.state_std)
        self.action_mean, self.action_std = to(stats.action_mean), to(stats.action_std)
        tokenizer = AutoTokenizer.from_pretrained(policy.config.vlm_model_name)
        encoded = tokenizer([TASK + "\n"], padding="max_length", truncation=True,
                            max_length=policy.config.tokenizer_max_length, return_tensors="pt")
        self.tokens = encoded["input_ids"].to(device)
        self.token_mask = encoded["attention_mask"].bool().to(device)

    @classmethod
    def load(cls, path: Path, stats: StudentStats | None = None, chunk: int | None = None,
             train_expert_only: bool = True, train: bool = False, device: str = "cuda") -> "Student":
        """Load a SmolVLA checkpoint directory. `stats` defaults to the checkpoint's saved
        student_stats.json; `chunk` overrides the chunk length (no weight depends on it).

        With `train`, the action expert is kept in fp32 (weights re-read from the file at full
        precision): AdamW on its bf16 weights, as LeRobot trains it, rounds most updates below
        ~1e-4 to zero, which stalls the second half of a cosine schedule."""
        config = PreTrainedConfig.from_pretrained(path)
        config.load_vlm_weights = False
        config.device = device
        if chunk is not None:
            config.chunk_size = chunk
            config.n_action_steps = chunk
        config.train_expert_only = train_expert_only
        config.freeze_vision_encoder = train_expert_only
        policy = SmolVLAPolicy.from_pretrained(path, config=config)
        # load_vlm_weights=False builds the VLM in fp32; the checkpoint was trained with it in bf16
        policy.model.vlm_with_expert.vlm.to(torch.bfloat16)
        if train:
            policy.model.vlm_with_expert.lm_expert.float()
            policy.load_state_dict(load_file(Path(path) / "model.safetensors", device=device))
        return cls(policy, stats or StudentStats.load(Path(path)), device)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.policy.save_pretrained(directory)
        self.stats.save(directory)

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        return [p for p in self.policy.parameters() if p.requires_grad]

    def _batch(self, images: torch.Tensor, states: torch.Tensor) -> dict[str, torch.Tensor]:
        """images (B, 2, 3, H, W) float in [0, 1], states (B, STATE_DIM) raw, both on the device."""
        b = states.shape[0]
        batch = {key: images[:, i] for i, key in enumerate(IMAGE_KEYS)}
        batch["observation.state"] = (states - self.state_mean) / self.state_std
        batch["observation.language.tokens"] = self.tokens.expand(b, -1)
        batch["observation.language.attention_mask"] = self.token_mask.expand(b, -1)
        return batch

    def images_to_device(self, images: np.ndarray) -> torch.Tensor:
        """(B, 2, H, W, 3) uint8 -> (B, 2, 3, H, W) float in [0, 1] on the device."""
        return torch.from_numpy(images).to(self.device).permute(0, 1, 4, 2, 3).float() / 255.0

    @torch.no_grad()
    def predict(self, images: np.ndarray, states: np.ndarray,
                generator: torch.Generator | None = None) -> np.ndarray:
        """images (B, 2, H, W, 3) uint8, states (B, STATE_DIM) -> (B, chunk, ACTION_DIM) env actions
        (arm columns clipped to [-1, 1], gripper column continuous; the caller binarises it)."""
        self.policy.eval()
        batch = self._batch(self.images_to_device(images),
                            torch.from_numpy(states).float().to(self.device))
        noise = None
        if generator is not None:
            shape = (len(states), self.chunk, self.policy.config.max_action_dim)
            noise = torch.randn(shape, generator=generator, device="cpu").to(self.device)
        actions = self.policy.predict_action_chunk(batch, noise=noise)
        actions = actions[..., :ACTION_DIM].float() * self.action_std + self.action_mean
        actions[..., :ACTION_DIM - 1] = actions[..., :ACTION_DIM - 1].clamp(-1, 1)
        return actions.cpu().numpy()

    def loss(self, images: torch.Tensor, states: torch.Tensor, chunks: torch.Tensor,
             noise: torch.Tensor | None = None, time: torch.Tensor | None = None,
             real_dims_only: bool = False) -> torch.Tensor:
        """Flow-matching MSE, as LeRobot's SmolVLA forward computes it.

        images (B, 2, 3, H, W) in [0, 1], states (B, STATE_DIM), chunks (B, chunk, ACTION_DIM) raw
        teacher actions, all on the device. Training keeps the zero-padded action dimensions in the
        loss: inference integrates all of them and feeds them back into the expert, so their
        velocities must stay trained. `real_dims_only` reports the loss on the 7 real ones."""
        batch = self._batch(images, states)
        target = (chunks - self.action_mean) / self.action_std
        model = self.policy.model
        imgs, img_masks = self.policy.prepare_images(batch)
        state = self.policy.prepare_state(batch)
        actions = self.policy.prepare_action({"action": target})
        losses = model.forward(imgs, img_masks, batch["observation.language.tokens"],
                               batch["observation.language.attention_mask"], state, actions,
                               noise, time)
        return losses[..., :ACTION_DIM].mean() if real_dims_only else losses.mean()
