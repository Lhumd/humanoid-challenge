import numpy as np
import pytest

from hc.datagen import Episode, execute_plan, load_episodes, plan_for, run_tasks, save_episodes
from hc.retarget import MAX_LINGER_S, MIN_APPROACH_S, Plan, apply_similarity, carry_end, planar_similarity, retarget
from hc.sim import CONTROL_FREQ, OBS_DIM, Scene, make_env, reset, scene_from_obs
from hc.types import Demo, make_synthetic_demo

EVAL_SEEDS = list(range(1_000_000, 1_000_020))


@pytest.fixture(scope="module")
def demos() -> list[Demo]:
    return [make_synthetic_demo(np.random.default_rng(i)) for i in range(10)]


@pytest.fixture(scope="module")
def scene() -> Scene:
    env = make_env()
    return scene_from_obs(reset(env, 1_000_000))


def oblique(demo: Demo) -> Demo:
    """The same demo, but the hand comes down on the box at 45 degrees instead of vertically."""
    pinch = demo.pinch.copy()
    g = demo.grasp_idx
    pinch[:g, 0] -= np.maximum(pinch[:g, 2] - pinch[g, 2], 0)
    return Demo(demo.t, pinch, demo.yaw, demo.grasp_idx, demo.release_idx, demo.source)


def grasp_and_release_steps(plan: Plan) -> tuple[int, int]:
    closing = int(np.argmax(plan.grip > 0))
    opening = closing + int(np.argmax(plan.grip[closing:] < 0))
    return closing, opening


def test_similarity_maps_both_anchors() -> None:
    src_a, src_b = np.array([0.0, 0.0]), np.array([0.25, 0.02])
    dst_a, dst_b = np.array([-0.05, 0.03]), np.array([0.04, -0.06])
    s = planar_similarity(src_a, src_b, dst_a, dst_b)
    out = apply_similarity(np.stack([src_a, src_b]), s, src_a, dst_a)
    np.testing.assert_allclose(out, np.stack([dst_a, dst_b]), atol=1e-12)


def test_plan_anchors(demos: list[Demo], scene: Scene) -> None:
    plan = retarget(demos[0], scene, 0.3)
    closing, opening = grasp_and_release_steps(plan)
    release_target = scene.stack_pos + [0, 0, 0.005]
    np.testing.assert_allclose(plan.pos[0], scene.eef_pos, atol=1e-9)
    assert np.linalg.norm(plan.pos[closing] - scene.cubeA_pos) < 0.005
    assert np.linalg.norm(plan.pos[opening] - release_target) < 0.005
    assert np.all(plan.yaw[closing:opening] == 0.3)
    assert len(plan.pos) == len(plan.yaw) == len(plan.grip) == len(plan.human_t)


def flat_carry(demo: Demo) -> Demo:
    """The same demo, but the hand slides the box up a straight ramp to the release point."""
    pinch = demo.pinch.copy()
    g, r = demo.grasp_idx, demo.release_idx
    pinch[g:r, 2] = np.linspace(pinch[g, 2], pinch[r, 2], r - g)
    return Demo(demo.t, pinch, demo.yaw, g, r, demo.source)


@pytest.mark.parametrize("make", [lambda d: d, flat_carry])
def test_carry_clears_cube_b(demos: list[Demo], scene: Scene, make) -> None:
    plan = retarget(make(demos[0]), scene, 0.0)
    closing, opening = grasp_and_release_steps(plan)
    carry = plan.pos[closing:opening]
    cube_b_top = scene.cubeB_pos[2] + scene.cubeB_half
    near_b = np.linalg.norm(carry[:, :2] - scene.cubeB_pos[None, :2], axis=1) < 0.06
    above_b = carry[near_b]
    stack_xy = np.linalg.norm(above_b[:, :2] - scene.cubeB_pos[None, :2], axis=1) < 0.012
    assert np.all(above_b[~stack_xy, 2] - scene.cubeA_half > cube_b_top + 0.02)
    assert carry[:, 2].max() - scene.cubeA_half > cube_b_top + 0.02
    table_z = scene.cubeA_pos[2] - scene.cubeA_half
    assert plan.pos[closing:opening, 2].max() <= table_z + 0.15 + 1e-9


@pytest.mark.parametrize("make", [lambda d: d, oblique])
def test_final_descent_is_vertical(demos: list[Demo], scene: Scene, make) -> None:
    plan = retarget(make(demos[1]), scene, 0.0)
    closing, _ = grasp_and_release_steps(plan)
    approach = plan.pos[:closing]
    low = approach[:, 2] - scene.cubeA_pos[2] < 0.03
    offsets = np.linalg.norm(approach[low, :2] - scene.cubeA_pos[None, :2], axis=1)
    assert low.sum() > 0 and offsets.max() < 0.006


def test_noise_free_replay_stacks(demos: list[Demo]) -> None:
    tasks = [(seed, seed % len(demos), 1.0, 0.0) for seed in EVAL_SEEDS]
    episodes = run_tasks(demos, tasks, "Panda", workers=6)
    assert np.mean([e.success for e in episodes]) >= 0.8


def test_oblique_demo_replay_stacks(demos: list[Demo]) -> None:
    env = make_env()
    obs = reset(env, 1_000_003)
    plan, ctl = plan_for(obs, oblique(demos[3]), 1.0)
    episode, _ = execute_plan(env, obs, plan, ctl, np.random.default_rng(0))
    assert episode.success


def starts_at_grasp(demo: Demo) -> Demo:
    """The same demo cut so the hand is first seen at the grasp (it entered the view late)."""
    g = demo.grasp_idx
    return Demo(demo.t[g:] - demo.t[g], demo.pinch[g:], demo.yaw[g:], 0, demo.release_idx - g,
                demo.source)


def test_demo_seen_from_the_grasp_replay_stacks(demos: list[Demo]) -> None:
    env = make_env()
    obs = reset(env, 1_000_003)
    plan, ctl = plan_for(obs, starts_at_grasp(demos[3]), 1.0)
    closing, _ = grasp_and_release_steps(plan)
    episode, _ = execute_plan(env, obs, plan, ctl, np.random.default_rng(0))
    assert closing >= MIN_APPROACH_S * CONTROL_FREQ
    assert episode.success


def test_episodes_round_trip(tmp_path) -> None:
    rng = np.random.default_rng(0)
    episodes = [Episode(rng.normal(size=(n, OBS_DIM)).astype(np.float32),
                        rng.uniform(-1, 1, (n, 7)).astype(np.float32), True, seed, seed % 3)
                for n, seed in [(5, 11), (8, 12), (3, 13)]]
    path = tmp_path / "episodes.npz"
    save_episodes(path, episodes)
    loaded = load_episodes(path)
    assert [e.seed for e in loaded] == [11, 12, 13]
    assert [e.demo_index for e in loaded] == [2, 0, 1]
    for a, b in zip(episodes, loaded):
        np.testing.assert_array_equal(a.features, b.features)
        np.testing.assert_array_equal(a.actions, b.actions)
        assert b.success
    raw = np.load(path)
    assert raw["obs"].dtype == np.float32 and list(raw["episode_ends"]) == [5, 13, 16]


def lingering(demo: Demo, seconds: float) -> Demo:
    """The same demo with the hand holding the box over the place spot for `seconds` before
    letting go."""
    r, dt = demo.release_idx, demo.t[1] - demo.t[0]
    n = int(round(seconds / dt))
    pinch = np.concatenate([demo.pinch[:r], np.repeat(demo.pinch[r:r + 1], n, 0), demo.pinch[r:]])
    yaw = np.concatenate([demo.yaw[:r], np.repeat(demo.yaw[r:r + 1], n), demo.yaw[r:]])
    return Demo(np.arange(len(pinch)) * dt, pinch, yaw, demo.grasp_idx, r + n, demo.source)


def test_carry_ends_soon_after_the_hand_arrives(demos):
    demo = lingering(demos[0], 2.0)
    end = carry_end(demo)
    arrive = demo.grasp_idx + int(np.argmax(np.linalg.norm(
        demo.pinch[demo.grasp_idx:demo.release_idx + 1, :2] - demo.pinch[demo.release_idx, :2], axis=1) < 0.02))
    assert demo.t[end] - demo.t[arrive] <= MAX_LINGER_S + 1e-6
    assert demo.t[demo.release_idx] - demo.t[end] > 1.5
    assert carry_end(demos[0]) <= demos[0].release_idx
