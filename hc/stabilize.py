"""Compensate the motion of a hand-held phone.

The phone moves a little while it films, and that motion would otherwise show up in the hand
trajectory. Each frame is registered to one reference frame by a homography fitted to ORB
features on the static background, with the hand region masked out. Hand-held motion is mostly
rotation, and for a pure rotation the homography is exact at every depth, so the hand's image
landmarks can be moved into the reference view and its 3-D landmarks rotated with
R = K^-1 H K. The lift then sees a camera that never moved.
"""

from pathlib import Path

import cv2
import numpy as np

from hc.lifting import focal_length
from hc.types import HandTrack

WORK_WIDTH = 960
MIN_INLIERS = 40
MAX_STEP_PX = 30.0


def _hand_mask(shape: tuple[int, int], landmarks: np.ndarray, scale: float) -> np.ndarray:
    """255 everywhere except a generous box around the hand and the object it holds."""
    mask = np.full(shape, 255, np.uint8)
    if np.isnan(landmarks).any():
        return mask
    pts = landmarks * scale
    lo, hi = pts.min(0), pts.max(0)
    margin = 0.6 * (hi - lo).max()
    x0, y0 = np.maximum(lo - margin, 0).astype(int)
    x1, y1 = (hi + margin).astype(int)
    mask[y0:y1, x0:x1] = 0
    return mask


def estimate_homographies(video: Path, track: HandTrack) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame homographies (T, 3, 3) mapping full-resolution pixels of frame t into frame 0,
    and the RANSAC inlier count of each frame-to-previous-frame fit.

    Consecutive frames are registered and the steps are chained. Matching every frame to one
    reference fails here because the boxes the hand moves are part of the background. A step
    with too few inliers or an implausible jump is treated as no camera motion."""
    scale = WORK_WIDTH / track.width
    to_work = np.diag([scale, scale, 1.0])
    to_full = np.linalg.inv(to_work)
    size = (WORK_WIDTH, int(round(track.height * scale)))
    orb = cv2.ORB_create(3000)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    centre = np.array([size[0] / 2, size[1] / 2, 1.0])

    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    homographies = np.tile(np.eye(3), (len(track.score), 1, 1))
    inliers = np.zeros(len(track.score), int)
    previous = None
    for i in range(len(track.score)):
        ok, frame = cap.read()
        if not ok:
            homographies[i:] = homographies[i - 1]
            break
        gray = cv2.cvtColor(cv2.resize(frame, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        kp, desc = orb.detectAndCompute(gray, _hand_mask(gray.shape, track.image_landmarks[i], scale))
        step = np.eye(3)
        if previous is not None and desc is not None and previous[1] is not None:
            pairs = matcher.knnMatch(desc, previous[1], k=2)
            good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
            if len(good) >= MIN_INLIERS:
                src = np.float32([kp[m.queryIdx].pt for m in good])
                dst = np.float32([previous[0][m.trainIdx].pt for m in good])
                h, mask = cv2.findHomography(src, dst, cv2.RANSAC, 2.0)
                inliers[i] = 0 if h is None else int(mask.sum())
                if inliers[i] >= MIN_INLIERS:
                    moved = h @ centre
                    if np.linalg.norm(moved[:2] / moved[2] - centre[:2]) < MAX_STEP_PX:
                        step = h
        previous = (kp, desc)
        if i > 0:
            homographies[i] = homographies[i - 1] @ (to_full @ step @ to_work)
    cap.release()
    return homographies, inliers


def stabilize_track(track: HandTrack, homographies: np.ndarray, hfov_deg: float) -> HandTrack:
    """Move every frame's landmarks into the reference camera."""
    f = focal_length(track.width, hfov_deg)
    k = np.array([[f, 0, track.width / 2], [0, f, track.height / 2], [0, 0, 1]])
    image = np.full_like(track.image_landmarks, np.nan)
    world = np.full_like(track.world_landmarks, np.nan)
    for i, h in enumerate(homographies):
        if track.score[i] == 0:
            continue
        pts = np.c_[track.image_landmarks[i], np.ones(len(track.image_landmarks[i]))] @ h.T
        image[i] = pts[:, :2] / pts[:, 2:]
        m = np.linalg.inv(k) @ h @ k
        u, _, vt = np.linalg.svd(m / np.cbrt(np.linalg.det(m)))
        rotation = u @ vt
        world[i] = track.world_landmarks[i] @ rotation.T
    return HandTrack(track.fps, track.width, track.height, image, world, track.score.copy())
