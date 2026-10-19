"""Object-centric retargeting: one human pick-and-place Demo -> an end-effector Plan for a robot.

The human motion is anchored on the two contact events. The human pinch point at the grasp maps
to cube A's centre (the robot's eef site sits between its fingertips), the pinch point at the
release maps to the stacked position of cube A on cube B. A planar similarity (rotation about z
plus uniform scale) carries the horizontal motion between the two anchors; heights keep their
metric offsets from the anchors, scaled by one clipped factor so the carried cube clears cube B.

Embodiment-gap constraints applied to the mapped path:
- the robot starts from its own home pose, so the offset between home and the mapped first point
  fades out over the early approach; the approach lasts at least MIN_APPROACH_S, because the
  phone often sees the hand only shortly before the grasp while the gripper starts ~20 cm away;
- human fingers wrap around a box from any angle, a parallel gripper must come down on it from
  straight above: the path may only drop below FUNNEL_HEIGHT over the grasp (and the stack) when it
  is within FUNNEL_RADIUS of it horizontally, which makes the last centimetres of each descent
  near-vertical;
- during the carry the path stays high enough for cube A to clear cube B everywhere except
  close above the pick-up spot and the stack, even when the human slid the box along a low path;
- a parallel gripper's outer finger can land on a nearby cube B, so the grasp uses the pair of
  cube faces whose closing axis is closest to perpendicular to the A->B direction
  (choose_grasp_yaw);
- the gripper needs time to close and open, so the plan holds still at the grasp and at the release;
- the human's wait between setting the box down and opening the hand is cut (carry_end).
"""

from dataclasses import dataclass

import numpy as np

from hc.sim import CONTROL_FREQ, Scene, TrackingController
from hc.types import Demo

RELEASE_CLEARANCE = 0.005  # cube A is let go this far above its stacked position
CARRY_MARGIN = 0.03  # carried cube A's bottom must pass this far above cube B's top
MAX_LIFT = 0.15  # highest end-effector point of the carry above the table
HEIGHT_SCALES = np.linspace(0.3, 3.0, 271)
FUNNEL_HEIGHT = 0.06
FUNNEL_RADIUS = 0.01
LIFT_RADIUS = 0.03  # the carried cube may rise gradually within this distance of where it was
TABLE_CLEARANCE = 0.015
RESIDUAL_FADE = 0.6  # fraction of the approach over which the home-pose offset fades out
MIN_APPROACH_S = 1.5  # shorter human approaches are slowed down to this (at speed 1)
SETTLE_S = 0.2  # hold before the gripper closes or opens, so tracking lag has caught up
GRIP_DWELL_S = 0.4  # hold while the gripper closes or opens
ARRIVAL_RADIUS = 0.02  # horizontal distance at which the hand has arrived over the place spot
MAX_LINGER_S = 0.3  # human time kept between arriving there and letting go


@dataclass
class Plan:
    """End-effector targets at CONTROL_FREQ, in world coordinates."""

    pos: np.ndarray  # (N, 3)
    yaw: np.ndarray  # (N,) yaw offsets from the gripper's home orientation
    grip: np.ndarray  # (N,) +1 closed, -1 open
    human_t: np.ndarray  # (N,) demo time each step reproduces (frozen during dwells)

    def __len__(self) -> int:
        return len(self.pos)


def smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def planar_similarity(src_a: np.ndarray, src_b: np.ndarray,
                      dst_a: np.ndarray, dst_b: np.ndarray) -> complex:
    """Complex factor s (rotation and scale) of the map x -> dst_a + s (x - src_a) that also
    sends src_b to dst_b, all points taken in the xy plane."""
    return complex(*(dst_b[:2] - dst_a[:2])) / complex(*(src_b[:2] - src_a[:2]))


def apply_similarity(xy: np.ndarray, s: complex, src_a: np.ndarray, dst_a: np.ndarray) -> np.ndarray:
    w = (xy[:, 0] + 1j * xy[:, 1] - complex(*src_a[:2])) * s + complex(*dst_a[:2])
    return np.stack([w.real, w.imag], axis=-1)


def resample(demo: Demo, start: int, end: int, speed: float, min_duration: float = 0.0) -> np.ndarray:
    """Demo times from frame `start` to frame `end` (both included) at the robot's control rate,
    with the human's time scaled by 1/speed and stretched to at least min_duration / speed."""
    t0, t1 = demo.t[start], demo.t[end]
    n = max(int(round(max(t1 - t0, min_duration) * CONTROL_FREQ / speed)), 1)
    return np.linspace(t0, t1, n + 1)


def interp_pinch(demo: Demo, tau: np.ndarray) -> np.ndarray:
    return np.stack([np.interp(tau, demo.t, demo.pinch[:, i]) for i in range(3)], axis=-1)


def funnel(pos: np.ndarray, centre: np.ndarray, height: float, radius: float) -> np.ndarray:
    """Lowest allowed z at each point: `height` above `centre`, dropping smoothly to the centre's
    own height within `radius` of it horizontally."""
    dist = np.linalg.norm(pos[:, :2] - centre[None, :2], axis=1)
    return centre[2] + height * smoothstep(dist / radius)


def raise_to(pos: np.ndarray, floor: np.ndarray) -> np.ndarray:
    return np.column_stack([pos[:, :2], np.maximum(pos[:, 2], floor)])


def height_scale(carry_base: np.ndarray, carry_offset: np.ndarray, need: float, cap: float) -> float:
    """The factor k closest to 1 for which the carry heights carry_base + k * carry_offset peak
    between `need` and `cap`; if no k satisfies both, the one with the smallest violation."""
    peaks = (carry_base[None] + HEIGHT_SCALES[:, None] * carry_offset[None]).max(axis=1)
    violation = np.maximum(need - peaks, 0) + np.maximum(peaks - cap, 0)
    return float(HEIGHT_SCALES[np.lexsort((np.abs(HEIGHT_SCALES - 1), violation))[0]])


def carry_end(demo: Demo) -> int:
    """Frame where the carry ends: the human release, or MAX_LINGER_S after the hand first
    arrives over the place spot when it stayed there longer. People set the box down and adjust
    it before letting go; copied into the plan, that wait puts many "keep holding" labels on the
    same state as the few "open" labels, and a policy that sees only the current state never
    learns when to let go."""
    r = demo.release_idx
    xy = np.linalg.norm(demo.pinch[demo.grasp_idx:r + 1, :2] - demo.pinch[r, :2], axis=1)
    arrive = demo.grasp_idx + int(np.argmax(xy < ARRIVAL_RADIUS))
    return min(r, int(np.searchsorted(demo.t, demo.t[arrive] + MAX_LINGER_S)))


def choose_grasp_yaw(ctl: TrackingController, scene: Scene) -> float:
    """Yaw offset for the grasp. A cube can be taken by either pair of opposite faces; pick the
    pair whose finger closing axis is most perpendicular to the direction of cube B, so the outer
    finger cannot land on cube B when the two cubes are close."""
    aligned = ctl.grasp_yaw(scene.cubeA_yaw)
    other = aligned - np.pi / 2 if aligned >= 0 else aligned + np.pi / 2
    home_axis = np.arctan2(ctl.home_rot[1, 1], ctl.home_rot[0, 1])  # fingers close along eef y
    towards_b = np.arctan2(*(scene.cubeB_pos[1::-1] - scene.cubeA_pos[1::-1]))
    return min((aligned, other), key=lambda yaw: abs(np.cos(home_axis + yaw - towards_b)))


def retarget(demo: Demo, scene: Scene, grasp_yaw: float, speed: float = 1.0,
             use_human_yaw: bool = False) -> Plan:
    """Map a table-frame human demo onto the robot scene, keeping the shape of the human motion.

    `grasp_yaw` comes from choose_grasp_yaw (or TrackingController.grasp_yaw). `speed` > 1
    replays the human motion faster. With `use_human_yaw`, the hand's yaw change since the grasp
    is added during the carry (cubes are symmetric, so it is off by default).
    """
    g_h, r_h = demo.pinch[demo.grasp_idx], demo.pinch[demo.release_idx]
    g_s = scene.cubeA_pos.copy()
    r_s = scene.stack_pos + np.array([0.0, 0.0, RELEASE_CLEARANCE])
    table_z = scene.cubeA_pos[2] - scene.cubeA_half
    s = planar_similarity(g_h, r_h, g_s, r_s)

    tau_app = resample(demo, 0, demo.grasp_idx, speed, MIN_APPROACH_S)
    tau_carry = resample(demo, demo.grasp_idx, carry_end(demo), speed)
    tau_ret = resample(demo, demo.release_idx, len(demo.t) - 1, speed)
    h_app, h_carry, h_ret = (interp_pinch(demo, tau) for tau in (tau_app, tau_carry, tau_ret))

    blend = smoothstep((tau_carry - tau_carry[0]) / (tau_carry[-1] - tau_carry[0]))
    carry_base = (1 - blend) * g_s[2] + blend * r_s[2]
    carry_offset = h_carry[:, 2] - ((1 - blend) * g_h[2] + blend * r_h[2])
    need = scene.cubeB_pos[2] + scene.cubeB_half + scene.cubeA_half + CARRY_MARGIN
    k = height_scale(carry_base, carry_offset, need, table_z + MAX_LIFT)

    def mapped(h: np.ndarray, z: np.ndarray) -> np.ndarray:
        return np.column_stack([apply_similarity(h, s, g_h, g_s), z])

    app = mapped(h_app, g_s[2] + k * (h_app[:, 2] - g_h[2]))
    carry = mapped(h_carry, carry_base + k * carry_offset)
    ret = mapped(h_ret, r_s[2] + k * (h_ret[:, 2] - r_h[2]))

    fade = 1 - smoothstep(np.arange(len(app)) / (RESIDUAL_FADE * (len(app) - 1)))
    app += fade[:, None] * (scene.eef_pos - app[0])[None]
    app = raise_to(app, funnel(app, g_s, FUNNEL_HEIGHT, FUNNEL_RADIUS))
    carry = raise_to(carry, np.minimum(funnel(carry, g_s, need - g_s[2], LIFT_RADIUS),
                                       funnel(carry, r_s, need - r_s[2], FUNNEL_RADIUS)))
    ret = raise_to(ret, funnel(ret, r_s, FUNNEL_HEIGHT, FUNNEL_RADIUS))

    yaw_app = grasp_yaw * (1 - fade)
    yaw_carry = np.full(len(carry), grasp_yaw)
    if use_human_yaw:
        human_yaw = np.unwrap(demo.yaw)
        yaw_carry += np.interp(tau_carry, demo.t, human_yaw) - human_yaw[demo.grasp_idx]
    yaw_ret = np.full(len(ret), yaw_carry[-1])

    n_settle, n_dwell = round(SETTLE_S * CONTROL_FREQ), round(GRIP_DWELL_S * CONTROL_FREQ)
    n_hold = n_settle + n_dwell
    hold_grip = np.r_[-np.ones(n_settle), np.ones(n_dwell)]
    pieces = [  # (positions, yaw, grip, human time)
        (app, yaw_app, -np.ones(len(app)), tau_app),
        (np.repeat(g_s[None], n_hold, 0), np.full(n_hold, grasp_yaw), hold_grip,
         np.full(n_hold, tau_app[-1])),
        (carry[1:], yaw_carry[1:], np.ones(len(carry) - 1), tau_carry[1:]),
        (np.repeat(r_s[None], n_hold, 0), np.full(n_hold, yaw_carry[-1]), -hold_grip,
         np.full(n_hold, tau_carry[-1])),
        (ret[1:], yaw_ret[1:], -np.ones(len(ret) - 1), tau_ret[1:]),
    ]
    pos, yaw, grip, human_t = (np.concatenate(part) for part in zip(*pieces))
    pos[:, 2] = np.maximum(pos[:, 2], table_z + TABLE_CLEARANCE)
    return Plan(pos, yaw, grip, human_t)
