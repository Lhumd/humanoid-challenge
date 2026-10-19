"""Split a camera-frame hand trajectory into pick-and-place repetitions.

The hand pauses at three kinds of places: the rest spot, the pick spot and the place spot. Pauses
(dwells) are runs slower than a fraction of the clip's brisk speed, clustered in 3D. The pick
cluster is the one where the hand arrives open and leaves closed on the box; the place cluster is
the reverse. This aperture cue is physical: one must open wider than the box to grasp it and to let
go of it. Each visit to the pick spot pairs with the next visit to the place spot, so the pick
precedes the place in every rep; extra pauses (hovering above a box) stay inside their visit.

Real footage can defeat these heuristics, so `reps_from_events` builds reps from hand-labelled
grasp and release times instead (data/raw/<clip>_events.json, [[grasp_s, release_s], ...]).
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

from hc.types import CameraTrajectory

DWELL_SPEED_FRACTION = 0.12
SPEED_HALF_WINDOW_S = 0.1
DWELL_MIN_S = 0.1
MERGE_GAP_S = 0.3
MERGE_DIST = 0.03
CLUSTER_DIST = 0.07
MAJOR_CLUSTER_FRACTION = 0.4
APERTURE_WINDOW_S = 0.3
REST_KEEP_S = 0.5
EDGE_PAD_S = 1.5


@dataclass
class Rep:
    """Frame indices of one repetition; `end` is exclusive."""

    start: int
    grasp: int
    release: int
    end: int


@dataclass
class Dwell:
    """A pause of the pinch point, frames [start, end)."""

    start: int
    end: int
    pos: np.ndarray

    @property
    def mid(self) -> int:
        return (self.start + self.end) // 2


def pinch_speed(traj: CameraTrajectory) -> np.ndarray:
    """Pinch speed (m/s) from displacement over +-SPEED_HALF_WINDOW_S, which averages out
    landmark jitter that a frame-to-frame difference would amplify."""
    k = max(int(round(SPEED_HALF_WINDOW_S / np.median(np.diff(traj.t)))), 1)
    padded = np.concatenate([np.repeat(traj.pinch[:1], k, 0), traj.pinch, np.repeat(traj.pinch[-1:], k, 0)])
    return np.linalg.norm(padded[2 * k:] - padded[:-2 * k], axis=1) / (traj.t[1] - traj.t[0]) / (2 * k)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def find_dwells(traj: CameraTrajectory) -> list[Dwell]:
    """Runs of at least DWELL_MIN_S slower than DWELL_SPEED_FRACTION of the clip's brisk speed
    (95th percentile), so slow and fast demonstrators are treated alike. Runs separated by a short
    twitch (gap below MERGE_GAP_S, positions within MERGE_DIST) are merged."""
    fps = 1.0 / np.median(np.diff(traj.t))
    speed = pinch_speed(traj)
    slow = (speed < DWELL_SPEED_FRACTION * np.percentile(speed[traj.valid], 95)) & traj.valid
    runs = [(s, e) for s, e in _runs(slow) if e - s >= DWELL_MIN_S * fps]
    merged: list[tuple[int, int]] = []
    for s, e in runs:
        if merged:
            ps, pe = merged[-1]
            close = np.linalg.norm(traj.pinch[ps:pe].mean(0) - traj.pinch[s:e].mean(0)) < MERGE_DIST
            if s - pe <= MERGE_GAP_S * fps and close:
                merged[-1] = (ps, e)
                continue
        merged.append((s, e))
    return [Dwell(int(s), int(e), traj.pinch[s:e].mean(0)) for s, e in merged]


def cluster_dwells(dwells: list[Dwell]) -> np.ndarray:
    """Cluster label per dwell (average linkage, cut at CLUSTER_DIST metres)."""
    positions = np.array([d.pos for d in dwells])
    return fcluster(linkage(positions, method="average"), CLUSTER_DIST, criterion="distance")


def aperture_closing(traj: CameraTrajectory, dwell: Dwell) -> float:
    """Aperture just before arriving minus aperture just after leaving: positive at a grasp,
    negative at a release."""
    w = int(round(APERTURE_WINDOW_S / np.median(np.diff(traj.t))))
    before = traj.aperture[max(dwell.start - w, 0):dwell.start + 1]
    after = traj.aperture[dwell.end - 1:dwell.end + w]
    return float(before.mean() - after.mean())


def pick_place_clusters(traj: CameraTrajectory, dwells: list[Dwell], labels: np.ndarray) -> tuple[int, int, list[int]]:
    """(pick label, place label, major labels). Major clusters hold at least
    MAJOR_CLUSTER_FRACTION as many dwells as the largest one."""
    ids, counts = np.unique(labels, return_counts=True)
    major = [int(i) for i, c in zip(ids, counts) if c >= max(2, MAJOR_CLUSTER_FRACTION * counts.max())]
    if len(major) < 2:
        raise ValueError("fewer than two recurring pause spots; label events in data/raw/<clip>_events.json")
    closing = {c: np.median([aperture_closing(traj, d) for d, l in zip(dwells, labels) if l == c]) for c in major}
    pick = max(major, key=closing.get)
    place = min(major, key=closing.get)
    return pick, place, major


def _rest_keep(traj: CameraTrajectory, rest: Dwell) -> int:
    """Frames of a rest dwell kept on each side of a cut."""
    return min(int(REST_KEEP_S / np.median(np.diff(traj.t))), (rest.end - rest.start) // 2)


def _boundary(traj: CameraTrajectory, left: Dwell, right: Dwell, rests: list[Dwell]) -> tuple[int, int]:
    """(end of the rep ending at `left`, start of the rep starting at `right`)."""
    between = [d for d in rests if d.start >= left.end and d.end <= right.start]
    if not between:
        mid = (left.end + right.start) // 2
        return mid, mid
    rest = max(between, key=lambda d: d.end - d.start)
    keep = _rest_keep(traj, rest)
    return rest.start + keep, rest.end - keep


def _pick_place_pairs(traj: CameraTrajectory, dwells: list[Dwell], labels: np.ndarray,
                      pick: int, place: int, major: list[int]) -> list[tuple[Dwell, Dwell]]:
    """(grasp dwell, release dwell) per rep. Consecutive dwells in one cluster form a visit (e.g. a
    hover above box B, then the placement); the grasp is the pick-visit dwell with the largest
    aperture closing, the release the place-visit dwell with the largest opening. A pick visit
    pairs with the next place visit unless another pick visit comes first."""
    visits: list[tuple[int, list[Dwell]]] = []
    for d, l in zip(dwells, labels):
        if l not in major:
            continue
        if visits and visits[-1][0] == l:
            visits[-1][1].append(d)
        else:
            visits.append((l, [d]))
    closing = lambda d: aperture_closing(traj, d)
    pairs, pending = [], None
    for label, visit in visits:
        if label == pick:
            pending = max(visit, key=closing)
        elif label == place and pending is not None:
            pairs.append((pending, min(visit, key=closing)))
            pending = None
    return pairs


def find_reps(traj: CameraTrajectory) -> list[Rep]:
    """Repetitions from dwell clustering. Grasp = last frame of the pick dwell (just before
    lift-off); release = middle of the place dwell; reps are cut inside rest dwells (keeping up
    to REST_KEEP_S of rest) or halfway between place and the next pick."""
    dwells = find_dwells(traj)
    labels = cluster_dwells(dwells)
    pick, place, major = pick_place_clusters(traj, dwells, labels)
    rests = [d for d, l in zip(dwells, labels) if l in major and l not in (pick, place)]
    pairs = _pick_place_pairs(traj, dwells, labels, pick, place, major)
    if not pairs:
        raise ValueError("no pick visit followed by a place visit; label events in data/raw/<clip>_events.json")
    fps = 1.0 / np.median(np.diff(traj.t))
    pad = int(EDGE_PAD_S * fps)
    first, last = pairs[0][0], pairs[-1][1]
    starts = [max(0, first.start - pad)]
    ends = []
    for (_, place_k), (pick_next, _) in zip(pairs[:-1], pairs[1:]):
        end_k, start_next = _boundary(traj, place_k, pick_next, rests)
        ends.append(end_k)
        starts.append(start_next)
    ends.append(min(len(traj.t), last.end + pad))
    leading = [d for d in rests if d.end <= first.start]
    if leading:
        starts[0] = leading[-1].end - _rest_keep(traj, leading[-1])
    trailing = [d for d in rests if d.start >= last.end]
    if trailing:
        ends[-1] = trailing[0].start + _rest_keep(traj, trailing[0])
    return [Rep(s, p.end - 1, q.start + (q.end - q.start) // 2, e)
            for s, e, (p, q) in zip(starts, ends, pairs)]


def load_events(path: Path) -> list[tuple[float, float]]:
    return [(float(g), float(r)) for g, r in json.loads(Path(path).read_text())]


def reps_from_events(traj: CameraTrajectory, events: list[tuple[float, float]]) -> list[Rep]:
    """Repetitions from labelled (grasp_s, release_s) pairs. A rep keeps at most EDGE_PAD_S before
    its grasp and after its release, and never crosses the midpoint to the neighbouring rep."""
    n = len(traj.t)
    fps = 1.0 / np.median(np.diff(traj.t))
    grasps = [int(round(g * fps)) for g, _ in events]
    releases = [int(round(r * fps)) for _, r in events]
    pad = int(EDGE_PAD_S * fps)
    cuts = [(r + g) // 2 for r, g in zip(releases[:-1], grasps[1:])]
    starts = [max(c, g - pad) for c, g in zip([0] + cuts, grasps)]
    ends = [min(c, r + pad) for c, r in zip(cuts + [n], releases)]
    return [Rep(s, g, r, e) for s, g, r, e in zip(starts, grasps, releases, ends)]
