"""What the SmolVLA student observes: two camera images, the robot's own state and an instruction.

The student starts from `lerobot/smolvla_libero`, which was fine-tuned on LIBERO (robosuite, Panda,
20 Hz OSC delta-pose actions), so its observations follow LIBERO's conventions as LeRobot stores them:
- camera1 is the third-person `agentview`, camera2 the wrist camera `robot0_eye_in_hand`, 256x256;
- images are robosuite's frames rotated by 180 degrees, i.e. an upright view mirrored left-right;
- state is [eef position (3), eef axis-angle (3), gripper finger positions (2)].
The action space needs no mapping: our 7-D OSC_POSE delta action (grip +1 closes) is LIBERO's.

Cube positions never reach the student; `privileged` exists only for diagnostics.
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from robosuite.utils import transform_utils as tu

TASK = "put the red cube on the green cube"
IMAGE_SIZE = 256
CAMERAS = ("agentview", "robot0_eye_in_hand")
IMAGE_KEYS = ("observation.images.camera1", "observation.images.camera2")
STATE_DIM = 8
CHUNK = 8  # actions per prediction; the teacher's chunk length, so its chunks are the labels
PRIVILEGED_DIM = 14


def student_state(obs: dict) -> np.ndarray:
    """LIBERO state: eef position, eef axis-angle, gripper finger positions."""
    return np.concatenate([obs["robot0_eef_pos"], tu.quat2axisangle(obs["robot0_eef_quat"]),
                           obs["robot0_gripper_qpos"]]).astype(np.float32)


def privileged(obs: dict) -> np.ndarray:
    """Cube A and cube B poses (position, xyzw quaternion): what the teacher sees and the student
    does not."""
    return np.concatenate([obs["cubeA_pos"], obs["cubeA_quat"],
                           obs["cubeB_pos"], obs["cubeB_quat"]]).astype(np.float32)


class StudentCameras:
    """Renders CAMERAS with mujoco.Renderer on a camera-less env (robosuite's own offscreen renderer
    leaks memory across hard resets) and returns them in LIBERO's convention.

    Collision geoms (group 0) are hidden, which makes the frames identical to robosuite's camera
    observations. Call `attach` after every reset: the hard reset replaces the model with an
    identical one rebuilt from the same XML, so the renderer (GL context, uploaded meshes and
    textures) is kept and only pointed at the new model. Creating a fresh EGL renderer per episode
    made the following hundred env steps about 3x slower."""

    def __init__(self, size: int = IMAGE_SIZE, cameras: tuple[str, ...] = CAMERAS):
        self.size = size
        self.cameras = cameras
        self.renderer = None
        self.option = mujoco.MjvOption()
        self.option.geomgroup[0] = 0

    def attach(self, env) -> None:
        self.env = env
        if self.renderer is None:
            self.renderer = mujoco.Renderer(env.sim.model._model, self.size, self.size)
        else:
            self.renderer._model = env.sim.model._model

    def render(self) -> np.ndarray:
        """(len(cameras), size, size, 3) uint8, upright and mirrored left-right."""
        frames = []
        for camera in self.cameras:
            self.renderer.update_scene(self.env.sim.data._data, camera=camera, scene_option=self.option)
            frames.append(self.renderer.render()[:, ::-1])
        return np.stack(frames)

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
