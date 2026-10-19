import numpy as np
import torch

from hc.policy import bc_loss
from hc.sim import ACTION_DIM
from hc.vision_policy import (PROPRIO_DIM, VisionBCPolicy, VisionConfig, random_shift,
                              training_samples)


def _tiny_policy() -> VisionBCPolicy:
    torch.set_num_threads(1)
    return VisionBCPolicy(VisionConfig(image_size=32, channels=8, chunk=4, hidden=32, keypoints=8))


def test_forward_shapes():
    torch.manual_seed(0)
    policy = _tiny_policy()
    images = torch.randint(0, 256, (5, 32, 32, 3), dtype=torch.uint8)
    out = policy((images, torch.randn(5, PROPRIO_DIM)))
    assert out.shape == (5, 4, ACTION_DIM)
    chunk = policy.predict_chunk(images[0].numpy(), np.zeros(PROPRIO_DIM, np.float32))
    assert chunk.shape == (4, ACTION_DIM)
    assert np.all(np.abs(chunk) <= 1.0)


def test_random_shift_keeps_shape_and_dtype():
    images = torch.randint(0, 256, (3, 32, 32, 3), dtype=torch.uint8)
    shifted = random_shift(images)
    assert shifted.shape == images.shape and shifted.dtype == torch.uint8


def test_training_samples_align_images_with_steps():
    img_idx, step_idx, act_idx, mask = training_samples(np.array([3, 5]), np.array([5, 9]), 2, 4)
    assert img_idx.tolist() == [0, 1, 2, 3, 4]
    assert step_idx.tolist() == [0, 2, 4, 5, 7]
    assert act_idx[2].tolist() == [4, 4, 4, 4] and mask[2].tolist() == [True, False, False, False]


def test_loss_decreases_on_tiny_batch():
    torch.manual_seed(0)
    policy = _tiny_policy()
    images = torch.randint(0, 256, (16, 32, 32, 3), dtype=torch.uint8)
    prop = torch.randn(16, PROPRIO_DIM)
    act = torch.rand(16, 4, ACTION_DIM) * 2 - 1
    mask = torch.ones(16, 4, dtype=torch.bool)
    opt = torch.optim.Adam(policy.parameters(), lr=3e-3)
    first = bc_loss(policy, (images, prop), act, mask)[0].item()
    for _ in range(60):
        loss, _ = bc_loss(policy, (images, prop), act, mask)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.7 * first
