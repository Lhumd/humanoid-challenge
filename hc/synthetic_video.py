"""Synthetic landmark tracks with ground truth, to test perception before real footage exists.

A rigid-palm model of an adult right hand (wrist to straight middle fingertip ~18 cm) follows the
demos' pinch trajectories and yaw, opens and closes its thumb-index pinch, and is projected into a
static pinhole camera looking down at the table. Only landmarks are rendered, not pixels.
"""

import numpy as np

from hc.types import Demo, HandTrack

APERTURE_OPEN = 0.09
APERTURE_HOLD = 0.06
APERTURE_RAMP_S = 0.2
WORLD_NOISE = 0.002
CAMERA_AIM = np.array([0.10, -0.08, 0.08])


def hand_points(aperture: float) -> np.ndarray:
    """21 MediaPipe landmarks (metres) of a right hand in a top-down grasp, in a hand frame with
    the pinch point at the origin, z up, fingers pointing +y and knuckles INDEX_MCP -> PINKY_MCP
    along +x. Thumb and index tips sit at y = -/+ aperture / 2."""
    half = aperture / 2
    wrist = np.array([0.035, -0.095, 0.085])
    thumb = [np.array([0.015, -0.075, 0.070]), np.array([-0.012, -0.055, 0.055]),
             np.array([-0.008, -half - 0.012, 0.025]), np.array([0.0, -half, 0.0])]
    mcps = [np.array([0.0, 0.0, 0.085]), np.array([0.022, 0.0, 0.088]),
            np.array([0.042, 0.0, 0.085]), np.array([0.060, 0.0, 0.078])]
    tips = [np.array([0.0, half, 0.0]), np.array([0.022, half + 0.004, 0.004]),
            np.array([0.042, half, 0.012]), np.array([0.060, half - 0.008, 0.022])]
    bend = np.array([0.0, 0.025, 0.012])
    points = [wrist, *thumb]
    for mcp, tip in zip(mcps, tips):
        points += [mcp, mcp + 0.45 * (tip - mcp) + bend, mcp + 0.75 * (tip - mcp) + 0.6 * bend, tip]
    return np.array(points)


def aperture_profile(demo: Demo) -> np.ndarray:
    """Thumb-index distance: open before the grasp, closing so it holds the box at grasp_idx,
    opening from release_idx on."""
    fps = 1.0 / np.median(np.diff(demo.t))
    ramp = APERTURE_RAMP_S * fps
    idx = np.arange(len(demo.t))
    closing = np.clip((idx - demo.grasp_idx) / ramp + 1, 0, 1)
    opening = np.clip((demo.release_idx - idx) / ramp + 1, 0, 1)
    closed = np.minimum(closing, opening)
    return APERTURE_OPEN - (APERTURE_OPEN - APERTURE_HOLD) * closed


def rot_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def camera_pose(camera_height: float, pitch_deg: float, yaw_deg: float, roll_deg: float,
                aim: np.ndarray) -> np.ndarray:
    """4x4 cam_T_table for a camera (x right, y down, z forward) `camera_height` above the table
    whose optical axis passes through the table-frame point `aim`."""
    p = np.radians(pitch_deg)
    forward = np.array([0.0, np.cos(p), -np.sin(p)])
    right = np.array([1.0, 0.0, 0.0])
    down = np.cross(forward, right)
    roll = np.radians(roll_deg)
    right, down = np.cos(roll) * right + np.sin(roll) * down, -np.sin(roll) * right + np.cos(roll) * down
    table_R_cam = rot_z(np.radians(yaw_deg)) @ np.stack([right, down, forward], axis=1)
    centre = aim - (camera_height - aim[2]) / np.sin(p) * table_R_cam[:, 2]
    cam_T_table = np.eye(4)
    cam_T_table[:3, :3] = table_R_cam.T
    cam_T_table[:3, 3] = -table_R_cam.T @ centre
    return cam_T_table


def make_synthetic_track(demos: list[Demo], rng: np.random.Generator, hfov_deg: float = 66.0,
                         image_size: tuple[int, int] = (1920, 1080), camera_height: float = 0.45,
                         pitch_deg: float = 55.0, noise_px: float = 1.5) -> tuple[HandTrack, np.ndarray]:
    """HandTrack of the demos played back to back, and the ground-truth cam_T_table.

    The camera is pitched down by `pitch_deg`, aimed at CAMERA_AIM (the middle of the space the
    hand sweeps; the wrist still leaves the frame at the rest spot), with a random
    yaw (+-15 deg) and roll (+-5 deg). Image landmarks get `noise_px` Gaussian pixel noise; world
    landmarks are the camera-frame hand points centred on their mean (MediaPipe convention) with
    WORLD_NOISE metres of noise. About one 2-6 frame dropout occurs every 5 s.
    """
    width, height = image_size
    fps = 1.0 / np.median(np.diff(demos[0].t))
    cam_T_table = camera_pose(camera_height, pitch_deg, rng.uniform(-15, 15), rng.uniform(-5, 5), CAMERA_AIM)
    R, t = cam_T_table[:3, :3], cam_T_table[:3, 3]

    frames = []
    for demo in demos:
        for pinch, yaw, aperture in zip(demo.pinch, demo.yaw, aperture_profile(demo)):
            frames.append(pinch + hand_points(aperture) @ rot_z(yaw).T)
    points_cam = np.array(frames) @ R.T + t

    f = (width / 2) / np.tan(np.radians(hfov_deg) / 2)
    image = f * points_cam[..., :2] / points_cam[..., 2:3] + np.array([width / 2, height / 2])
    image += rng.normal(0, noise_px, image.shape)
    world = points_cam - points_cam.mean(axis=1, keepdims=True)
    world += rng.normal(0, WORLD_NOISE, world.shape)
    score = rng.uniform(0.85, 0.99, len(frames))

    n = len(frames)
    for start in rng.choice(n - 10, size=max(1, int(n / fps / 5)), replace=False):
        stop = start + rng.integers(2, 7)
        image[start:stop] = np.nan
        world[start:stop] = np.nan
        score[start:stop] = 0.0
    return HandTrack(fps, width, height, image, world, score), cam_T_table
