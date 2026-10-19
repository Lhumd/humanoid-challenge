"""Lift a 2D hand track to a metric camera-frame trajectory, then express repetitions in the
table frame.

Depth comes from the hand itself: MediaPipe's world landmarks give the hand's metric shape, so
the translation that best reprojects that shape onto the 2D landmarks (a linear PnP with known
rotation) places the hand in the camera frame. No depth sensor or checkerboard is needed; the
cost is that a wrong hand size scales every distance (`hand_scale` corrects it).

Table frame. The phone is static but its pose is unknown, and the video gives no gravity.
- Up axis: mean direction of the first few centimetres of motion away from each grasp and release
  (lift-off, retreat) and toward them, reversed (descents). People lift and lower boxes
  vertically, and the horizontal drift of lifts (toward the place spot) and of placing descents
  (from the pick spot) points in opposite directions, so it largely cancels.
  Rejected: orthogonalising against the carry vector, because the place spot is higher than the
  pick spot (box B), so the carry is not horizontal and that would tilt the frame; a plane fit
  to dwell positions, because rest, pick and place sit at different heights above the table.
- x axis: mean grasp point to mean release point, projected onto the table plane.
- Origin: mean grasp point moved down to the table. Table height is the 5th percentile of pinch
  heights inside repetitions, i.e. the lowest places the hand visits (the rest spot when the hand
  rests on the table, else the grasp). It overestimates the table by the pinch point's height
  above the table at rest (about a fingertip thickness).
"""

import numpy as np
from scipy.signal import butter, filtfilt

from hc.segmentation import Rep
from hc.types import INDEX_MCP, INDEX_TIP, PINKY_MCP, THUMB_TIP, CameraTrajectory, Demo, HandTrack

MOTION_PROBE_DIST = 0.03
TABLE_HEIGHT_PERCENTILE = 5.0
OUTLIER_FACTOR = 4.0
OUTLIER_MIN_PX = 5.0


def focal_length(width: int, hfov_deg: float) -> float:
    return (width / 2) / np.tan(np.radians(hfov_deg) / 2)


def solve_translation(image_lms: np.ndarray, world_lms: np.ndarray, f: float,
                      cx: float, cy: float) -> np.ndarray:
    """Per-frame metric translation t (T, 3) such that world_lms + t projects onto image_lms.

    u_i = f (X_i + tx) / (Z_i + tz) + cx is linear in t: f tx - (u_i - cx) tz = (u_i - cx) Z_i - f X_i,
    and likewise for v and y. Stacking the 21 landmarks gives a 42 x 3 least-squares problem.
    """
    du = image_lms[..., 0] - cx
    dv = image_lms[..., 1] - cy
    X, Y, Z = world_lms[..., 0], world_lms[..., 1], world_lms[..., 2]
    zeros, fs = np.zeros_like(du), np.full_like(du, f)
    rows_u = np.stack([fs, zeros, -du], axis=-1)
    rows_v = np.stack([zeros, fs, -dv], axis=-1)
    A = np.concatenate([rows_u, rows_v], axis=1)
    b = np.concatenate([du * Z - f * X, dv * Z - f * Y], axis=1)
    AtA = np.einsum("tij,tik->tjk", A, A)
    Atb = np.einsum("tij,ti->tj", A, b)
    return np.linalg.solve(AtA, Atb[..., None])[..., 0]


def reprojection_error(points_cam: np.ndarray, image_lms: np.ndarray, f: float,
                       cx: float, cy: float) -> np.ndarray:
    """RMS pixel error (T,) of camera-frame points against the 2D landmarks."""
    uv = f * points_cam[..., :2] / points_cam[..., 2:3] + np.array([cx, cy])
    return np.sqrt(np.mean(np.sum((uv - image_lms) ** 2, axis=-1), axis=-1))


def _fill_gaps(values: np.ndarray, seen: np.ndarray, max_gap: int) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate unseen frames; frames in gaps longer than max_gap are marked invalid."""
    idx = np.arange(len(seen))
    seen_idx = np.flatnonzero(seen)
    flat = values.reshape(len(seen), -1)
    filled = np.stack([np.interp(idx, seen_idx, flat[seen_idx, j]) for j in range(flat.shape[1])], axis=1)
    valid = seen.copy()
    gap_start = None
    for i in range(len(seen) + 1):
        missing = i < len(seen) and not seen[i]
        if missing and gap_start is None:
            gap_start = i
        elif not missing and gap_start is not None:
            if i - gap_start <= max_gap:
                valid[gap_start:i] = True
            gap_start = None
    return filled.reshape(values.shape), valid


def lift(track: HandTrack, hfov_deg: float = 66.0, hand_scale: float = 1.0,
         cutoff_hz: float = 4.0, max_gap_s: float = 0.3) -> CameraTrajectory:
    """Metric camera-frame hand trajectory from a HandTrack (pinhole camera, principal point at
    the image centre). Frames whose reprojection error exceeds OUTLIER_FACTOR times the median
    (and OUTLIER_MIN_PX) count as missed detections; gaps up to max_gap_s are interpolated, then
    everything is low-pass filtered without phase lag."""
    f = focal_length(track.width, hfov_deg)
    cx, cy = track.width / 2, track.height / 2
    seen = (track.score > 0) & np.all(np.isfinite(track.image_landmarks), axis=(1, 2))
    world = hand_scale * track.world_landmarks[seen]
    image = track.image_landmarks[seen]
    translation = solve_translation(image, world, f, cx, cy)
    points = world + translation[:, None, :]
    seen_idx = np.flatnonzero(seen)
    error = reprojection_error(points, image, f, cx, cy)
    seen[seen_idx[error > max(OUTLIER_FACTOR * np.median(error), OUTLIER_MIN_PX)]] = False

    landmarks = np.full(track.world_landmarks.shape, np.nan)
    landmarks[seen_idx] = points
    landmarks, valid = _fill_gaps(landmarks, seen, int(round(max_gap_s * track.fps)))
    b, a = butter(2, cutoff_hz / (track.fps / 2))
    landmarks = filtfilt(b, a, landmarks, axis=0)

    thumb, index = landmarks[:, THUMB_TIP], landmarks[:, INDEX_TIP]
    t = np.arange(len(seen)) / track.fps
    return CameraTrajectory(t, landmarks, (thumb + index) / 2, np.linalg.norm(thumb - index, axis=1), valid)


def _probe_direction(pinch: np.ndarray, anchor: int, frames: np.ndarray) -> np.ndarray | None:
    """Unit vector from pinch[anchor] to the first frame in `frames` at least MOTION_PROBE_DIST away."""
    dist = np.linalg.norm(pinch[frames] - pinch[anchor], axis=1)
    far = np.flatnonzero(dist >= MOTION_PROBE_DIST)
    if len(far) == 0:
        return None
    v = pinch[frames[far[0]]] - pinch[anchor]
    return v / np.linalg.norm(v)


def estimate_up(traj: CameraTrajectory, reps: list[Rep]) -> np.ndarray:
    """Table normal in the camera frame from lift-offs, retreats and reversed descents."""
    directions = []
    for rep in reps:
        for event in (rep.grasp, rep.release):
            after = np.arange(event + 1, rep.end)
            before = np.arange(event - 1, rep.start - 1, -1)
            directions += [_probe_direction(traj.pinch, event, after),
                           _probe_direction(traj.pinch, event, before)]
    up = np.mean([d for d in directions if d is not None], axis=0)
    return up / np.linalg.norm(up)


def table_frame(traj: CameraTrajectory, reps: list[Rep]) -> np.ndarray:
    """4x4 cam_T_table: p_cam = R @ p_table + origin (see the module docstring for the axes)."""
    up = estimate_up(traj, reps)
    grasp_mean = traj.pinch[[r.grasp for r in reps]].mean(axis=0)
    release_mean = traj.pinch[[r.release for r in reps]].mean(axis=0)
    carry = release_mean - grasp_mean
    x = carry - (carry @ up) * up
    x /= np.linalg.norm(x)
    y = np.cross(up, x)

    in_reps = np.concatenate([np.arange(r.start, r.end) for r in reps])
    in_reps = in_reps[traj.valid[in_reps]]
    table_height = np.percentile(traj.pinch[in_reps] @ up, TABLE_HEIGHT_PERCENTILE)
    origin = grasp_mean - (grasp_mean @ up - table_height) * up

    cam_T_table = np.eye(4)
    cam_T_table[:3, :3] = np.stack([x, y, up], axis=1)
    cam_T_table[:3, 3] = origin
    return cam_T_table


def to_table(points_cam: np.ndarray, cam_T_table: np.ndarray) -> np.ndarray:
    """Camera-frame points (..., 3) to the table frame."""
    return (points_cam - cam_T_table[:3, 3]) @ cam_T_table[:3, :3]


def hand_yaw(landmarks_cam: np.ndarray, cam_T_table: np.ndarray) -> np.ndarray:
    """Unwrapped angle of the INDEX_MCP -> PINKY_MCP direction in the table plane, from table x."""
    knuckles = (landmarks_cam[:, PINKY_MCP] - landmarks_cam[:, INDEX_MCP]) @ cam_T_table[:3, :3]
    return np.unwrap(np.arctan2(knuckles[:, 1], knuckles[:, 0]))


def visible_span(valid: np.ndarray, rep: Rep) -> tuple[int, int]:
    """The rep's frame range shrunk to the hand being tracked continuously before the grasp and
    after the release (the hand often enters and leaves the phone's view)."""
    start, end = rep.grasp, rep.release + 1
    while start > rep.start and valid[start - 1]:
        start -= 1
    while end < rep.end and valid[end]:
        end += 1
    return start, end


def to_demos(traj: CameraTrajectory, reps: list[Rep], cam_T_table: np.ndarray, source: str) -> list[Demo]:
    """Cut the trajectory into one Demo per repetition, in the table frame."""
    pinch = to_table(traj.pinch, cam_T_table)
    yaw = hand_yaw(traj.landmarks, cam_T_table)
    demos = []
    for r in reps:
        s, e = visible_span(traj.valid, r)
        demos.append(Demo(traj.t[s:e] - traj.t[s], pinch[s:e], yaw[s:e], r.grasp - s, r.release - s,
                          f"{source}:{s}-{e}"))
    return demos
