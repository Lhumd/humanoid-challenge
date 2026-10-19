"""Data contracts shared by every stage of the pipeline.

video --track--> HandTrack --lift--> CameraTrajectory --segment+table frame--> list[Demo]
list[Demo] --retarget+sim--> episodes.npz --train--> policy
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

NUM_LANDMARKS = 21
WRIST, THUMB_TIP, INDEX_MCP, INDEX_TIP, MIDDLE_MCP, PINKY_MCP = 0, 4, 5, 8, 9, 17


@dataclass
class HandTrack:
    """Raw per-frame MediaPipe output for one video. Missing frames are NaN with score 0."""

    fps: float
    width: int
    height: int
    image_landmarks: np.ndarray  # (T, 21, 2) pixels
    world_landmarks: np.ndarray  # (T, 21, 3) metres, hand-centred, camera-aligned axes
    score: np.ndarray  # (T,) detection confidence

    def save(self, path: Path) -> None:
        np.savez_compressed(path, fps=self.fps, width=self.width, height=self.height,
                            image_landmarks=self.image_landmarks,
                            world_landmarks=self.world_landmarks, score=self.score)

    @classmethod
    def load(cls, path: Path) -> "HandTrack":
        d = np.load(path)
        return cls(float(d["fps"]), int(d["width"]), int(d["height"]),
                   d["image_landmarks"], d["world_landmarks"], d["score"])


@dataclass
class CameraTrajectory:
    """Metric hand trajectory in the camera frame (x right, y down, z forward), after smoothing."""

    t: np.ndarray  # (T,) seconds
    landmarks: np.ndarray  # (T, 21, 3) metres, camera frame
    pinch: np.ndarray  # (T, 3) midpoint of thumb tip and index tip, camera frame
    aperture: np.ndarray  # (T,) thumb-tip to index-tip distance in metres
    valid: np.ndarray  # (T,) bool, False where the hand was never seen nearby in time


@dataclass
class Demo:
    """One pick-and-place repetition expressed in the task-aligned table frame.

    Table frame: z up (table normal), origin at the mean grasp point on the table plane,
    x from mean grasp point toward mean release point. Units: metres and seconds.
    """

    t: np.ndarray  # (T,) seconds from the start of the repetition
    pinch: np.ndarray  # (T, 3) pinch point in the table frame
    yaw: np.ndarray  # (T,) hand yaw about the table normal, radians
    grasp_idx: int  # frame where the hand closes on the object
    release_idx: int  # frame where the hand lets go
    source: str  # clip name and frame range, for traceability

    @property
    def closed(self) -> np.ndarray:
        idx = np.arange(len(self.t))
        return (idx >= self.grasp_idx) & (idx < self.release_idx)


def save_demos(path: Path, demos: list[Demo]) -> None:
    arrays = {}
    for i, d in enumerate(demos):
        arrays[f"{i}_t"] = d.t
        arrays[f"{i}_pinch"] = d.pinch
        arrays[f"{i}_yaw"] = d.yaw
        arrays[f"{i}_events"] = np.array([d.grasp_idx, d.release_idx])
        arrays[f"{i}_source"] = np.array(d.source)
    np.savez_compressed(path, n=len(demos), **arrays)


def load_demos(path: Path) -> list[Demo]:
    d = np.load(path)
    return [Demo(d[f"{i}_t"], d[f"{i}_pinch"], d[f"{i}_yaw"],
                 int(d[f"{i}_events"][0]), int(d[f"{i}_events"][1]), str(d[f"{i}_source"]))
            for i in range(int(d["n"]))]


def make_synthetic_demo(rng: np.random.Generator, fps: float = 30.0) -> Demo:
    """A plausible human pick-and-place in the table frame, for tests and for development before
    real data exists. Reach from a rest spot, grasp at the origin, lift ~10 cm, carry ~25 cm along
    +x, place on a 6 cm box, release, return."""
    rest = np.array([-0.05, -0.20, 0.0]) + rng.normal(0, 0.01, 3) * [1, 1, 0]
    grasp = np.array([0.0, 0.0, 0.03]) + rng.normal(0, 0.005, 3)
    release = np.array([0.25, 0.0, 0.09]) + rng.normal(0, 0.008, 3)
    lift = 0.10 + rng.normal(0, 0.01)

    def seg(a, b, dur, arc=0.0):
        n = max(int(dur * fps), 2)
        s = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n))
        p = a[None] + s[:, None] * (b - a)[None]
        p[:, 2] += arc * np.sin(np.pi * s)
        return p

    parts = [
        seg(rest, rest, 0.3),
        seg(rest, grasp + [0, 0, 0.06], 0.9, arc=0.04),
        seg(grasp + [0, 0, 0.06], grasp, 0.4),
        seg(grasp, grasp, 0.25),
    ]
    grasp_idx = sum(len(p) for p in parts) - 1
    parts += [
        seg(grasp, grasp + [0, 0, lift], 0.5),
        seg(grasp + [0, 0, lift], release + [0, 0, 0.04], 1.0, arc=0.02),
        seg(release + [0, 0, 0.04], release, 0.4),
        seg(release, release, 0.25),
    ]
    release_idx = sum(len(p) for p in parts) - 1
    parts += [seg(release, release + [0, 0, 0.08], 0.4), seg(release + [0, 0, 0.08], rest, 1.0, arc=0.03)]
    pinch = np.concatenate(parts)
    pinch += rng.normal(0, 0.002, pinch.shape)
    t = np.arange(len(pinch)) / fps
    yaw = np.full(len(t), rng.normal(0, 0.1))
    return Demo(t, pinch, yaw, grasp_idx, release_idx, "synthetic")
