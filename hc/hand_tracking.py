"""MediaPipe hand tracking of one phone video into a HandTrack.

One hand is kept per frame by temporal continuity: the first kept detection is the most
confident hand, later frames keep the hand whose wrist is nearest the previous kept wrist.
Handedness labels are not used because they flip on non-mirrored rear-camera footage.
"""

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from hc.types import NUM_LANDMARKS, WRIST, HandTrack


def _make_landmarker(model_path: Path) -> vision.HandLandmarker:
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO, num_hands=2,
        min_hand_detection_confidence=0.4, min_hand_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )
    return vision.HandLandmarker.create_from_options(options)


def _candidates(result, width: int, height: int) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """(image landmarks in full-resolution pixels, world landmarks, score) for each detected hand."""
    hands = []
    for image_lms, world_lms, handedness in zip(result.hand_landmarks, result.hand_world_landmarks,
                                                result.handedness):
        image = np.array([[lm.x * width, lm.y * height] for lm in image_lms])
        world = np.array([[lm.x, lm.y, lm.z] for lm in world_lms])
        hands.append((image, world, handedness[0].score))
    return hands


def _pick_hand(hands: list, previous_wrist: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, float]:
    if previous_wrist is None:
        return max(hands, key=lambda h: h[2])
    return min(hands, key=lambda h: np.linalg.norm(h[0][WRIST] - previous_wrist))


def track_video(video_path: Path, model_path: Path = Path("models/hand_landmarker.task"),
                max_side: int = 1280) -> HandTrack:
    """Run the hand landmarker on every frame. Inference runs on a copy downscaled so its longer
    side is at most `max_side`; landmarks are returned in pixels of the displayed frame (OpenCV
    applies the phone's rotation metadata). Frames without a hand are NaN with score 0."""
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    fps = cap.get(cv2.CAP_PROP_FPS)
    image_lms, world_lms, scores = [], [], []
    previous_wrist = None
    width = height = 0
    with _make_landmarker(model_path) as landmarker:
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            height, width = frame.shape[:2]
            scale = min(1.0, max_side / max(height, width))
            small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else frame
            rgb = np.ascontiguousarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                                                 int(round(frame_idx * 1000.0 / fps)))
            hands = _candidates(result, width, height)
            if hands:
                image, world, score = _pick_hand(hands, previous_wrist)
                previous_wrist = image[WRIST]
            else:
                image = np.full((NUM_LANDMARKS, 2), np.nan)
                world = np.full((NUM_LANDMARKS, 3), np.nan)
                score = 0.0
            image_lms.append(image)
            world_lms.append(world)
            scores.append(score)
            frame_idx += 1
    cap.release()
    return HandTrack(fps, width, height, np.array(image_lms), np.array(world_lms), np.array(scores))
