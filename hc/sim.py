"""robosuite Stack environment, scene readout, policy features and a Cartesian tracking controller.

Every robot command in this project goes through `TrackingController`: a target end-effector
position, a target yaw about the world z axis and a gripper command become one 7-D OSC_POSE
delta action [dx, dy, dz, drx, dry, drz, grip] in [-1, 1] (grip +1 closes).
"""

import os
from dataclasses import dataclass

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from robosuite.utils import transform_utils as tu

CONTROL_FREQ = 20
POS_STEP_MAX = 0.05  # metres per action unit, robosuite OSC_POSE default output_max
ROT_STEP_MAX = 0.5  # radians per action unit
ACTION_DIM = 7
OBS_DIM = 21
SETTLE_STEPS = 5


def make_env(robot: str = "Panda", horizon: int = 400,
             camera: str | None = None, camera_size: int = 256):
    """Stack task: lift cube A (4 cm, red) and place it on cube B (5 cm, green).

    `camera` enables offscreen rendering from that camera (e.g. "frontview", "agentview");
    without it the env runs at ~100 steps/s on one CPU core.
    """
    cfg = load_composite_controller_config(controller="BASIC", robot=robot)
    render = camera is not None
    return suite.make(
        "Stack", robots=robot, controller_configs=cfg,
        has_renderer=False, has_offscreen_renderer=render, use_camera_obs=render,
        camera_names=[camera] if render else [], camera_heights=camera_size, camera_widths=camera_size,
        control_freq=CONTROL_FREQ, horizon=horizon, ignore_done=True, reward_shaping=False,
    )


def reset(env, seed: int) -> dict:
    """Reset to the layout determined by `seed` (robosuite samples cube poses from np.random),
    then hold still for a few steps so the cubes, spawned 1 cm high, settle on the table."""
    np.random.seed(seed)
    env.reset()
    hold = np.zeros(ACTION_DIM)
    hold[-1] = -1.0
    for _ in range(SETTLE_STEPS):
        obs, *_ = env.step(hold)
    return obs


def quat_yaw(quat_xyzw: np.ndarray) -> float:
    m = tu.quat2mat(quat_xyzw)
    return float(np.arctan2(m[1, 0], m[0, 0]))


def wrap_quarter(angle: float) -> float:
    """Wrap to [-pi/4, pi/4): a cube looks the same every quarter turn."""
    return float((angle + np.pi / 4) % (np.pi / 2) - np.pi / 4)


@dataclass
class Scene:
    """Object-centric description of one episode's initial layout, in world coordinates."""

    eef_pos: np.ndarray
    cubeA_pos: np.ndarray
    cubeA_yaw: float
    cubeB_pos: np.ndarray
    cubeA_half: float = 0.02
    cubeB_half: float = 0.025

    @property
    def stack_pos(self) -> np.ndarray:
        """Where cube A's centre sits once stacked on cube B."""
        return self.cubeB_pos + np.array([0.0, 0.0, self.cubeB_half + self.cubeA_half])


def scene_from_obs(obs: dict) -> Scene:
    return Scene(obs["robot0_eef_pos"].copy(), obs["cubeA_pos"].copy(),
                 quat_yaw(obs["cubeA_quat"]), obs["cubeB_pos"].copy())


def policy_features(obs: dict) -> np.ndarray:
    """Low-dimensional state for the learned policy (OBS_DIM values)."""
    eef, a, b = obs["robot0_eef_pos"], obs["cubeA_pos"], obs["cubeB_pos"]
    eef_yaw = quat_yaw(obs["robot0_eef_quat"])
    a_yaw = quat_yaw(obs["cubeA_quat"])
    return np.concatenate([
        eef, [np.sin(4 * eef_yaw), np.cos(4 * eef_yaw)], obs["robot0_gripper_qpos"],
        a, [np.sin(4 * a_yaw), np.cos(4 * a_yaw)], b, a - eef, b - a,
    ]).astype(np.float32)


class TrackingController:
    """Proportional Cartesian tracking on top of robosuite's OSC_POSE delta controller.

    Yaw targets are relative to the gripper's home orientation, so the same commands work on
    any robosuite arm whose home pose points the gripper down.
    """

    def __init__(self, obs: dict, pos_gain: float = 1.0, rot_gain: float = 1.0):
        self.home_rot = tu.quat2mat(obs["robot0_eef_quat"])
        self.home_yaw = float(np.arctan2(self.home_rot[1, 0], self.home_rot[0, 0]))
        self.pos_gain = pos_gain
        self.rot_gain = rot_gain

    def grasp_yaw(self, cube_yaw: float) -> float:
        """Yaw offset from home that aligns the fingers with the faces of a cube."""
        return wrap_quarter(cube_yaw - self.home_yaw)

    def action(self, obs: dict, target_pos: np.ndarray, yaw_offset: float, grip: float) -> np.ndarray:
        pos_err = target_pos - obs["robot0_eef_pos"]
        target_rot = tu.rotation_matrix(yaw_offset, [0, 0, 1])[:3, :3] @ self.home_rot
        rot_err = tu.quat2axisangle(tu.mat2quat(target_rot @ tu.quat2mat(obs["robot0_eef_quat"]).T))
        act = np.zeros(ACTION_DIM)
        act[:3] = np.clip(self.pos_gain * pos_err / POS_STEP_MAX, -1, 1)
        act[3:6] = np.clip(self.rot_gain * rot_err / ROT_STEP_MAX, -1, 1)
        act[6] = 1.0 if grip > 0 else -1.0
        return act


def displace_cube(env, cube: str, dxy: np.ndarray) -> None:
    """Teleport cube "A" or "B" horizontally by dxy metres (robustness tests)."""
    obj = env.cubeA if cube == "A" else env.cubeB
    joint = obj.joints[0]
    qpos = env.sim.data.get_joint_qpos(joint).copy()
    qpos[:2] += dxy
    env.sim.data.set_joint_qpos(joint, qpos)
    env.sim.data.set_joint_qvel(joint, np.zeros(6))
    env.sim.forward()
