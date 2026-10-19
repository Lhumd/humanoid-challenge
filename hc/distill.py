"""On-policy distillation (DAgger) of the privileged state-based teacher into the SmolVLA student.

The teacher is the BC policy of hc.policy: it sees cube poses and stacks 94 % of held-out layouts.
The student (hc.vla_student) sees only camera images, the robot's own state and the instruction.

Rollouts run in env worker processes driven in lockstep from the main process, which holds the
student on the GPU and batches its chunk predictions across envs. Whoever drives an episode, every
recorded frame is labelled with the teacher's own predicted chunk at that state (DAgger): the
teacher can be queried at any state because it only needs the simulator's cube poses.

Episode drivers:
- "teacher": the teacher acts (temporal ensembling, as in its evaluation) plus DART noise on the
  arm dimensions; this is round 0, plain offline distillation;
- "student": the student acts; with `switch_step` it hands over to the teacher at that step
  ("switch-once" episodes record the teacher recovering from the student's own states), and a
  student that stalls for STALL_STEPS is taken over as well, so no episode floods the data with
  hundreds of near-identical stuck frames.
"""

import multiprocessing as mp
import os
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from hc.evaluate import AFTER_STEPS, HOLD_STEPS, HORIZON, MIN_CUBE_GAP, PERTURB_RANGE, sample_perturbation
from hc.sim import ACTION_DIM, displace_cube, make_env, policy_features, reset
from hc.vla_env import (CAMERAS, CHUNK, IMAGE_KEYS, IMAGE_SIZE, PRIVILEGED_DIM, STATE_DIM, TASK,
                        StudentCameras, apply_visual_stress, privileged, student_state)

RECORD_STRIDE = 2
N_ACTION_STEPS = 4
STALL_STEPS = 60
STALL_DIST = 0.01
TEACHER_NOISE = 0.1
DATASET_FPS = 20 // RECORD_STRIDE
SINGLE_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")

# Stress tests beyond hc.evaluate's push (evaluation only; never used for training data):
#   move_b  cube B moves 3-5 cm once cube A is carried       drop   cube A slips back onto the table
#   noise   N(0, STRESS_NOISE) on every executed arm action    wide   layouts within +-WIDE_RANGE
#   (STRESS_NOISE 0.15: the teacher stacks 12/12 smoke layouts at 0.15 but 4/12 at 0.2)
#   dim / table / camera   darker lights, a brown table, the agentview camera moved (pixels only)
STRESSES = ("move_b", "drop", "noise", "wide", "dim", "table", "camera")
STRESS_NOISE = 0.15
TRAIN_RANGE = 0.08  # robosuite Stack's default layout range, used for every training episode
WIDE_RANGE = 0.14
LIFT_TRIGGER = 0.03  # cube A counts as carried once 3 cm above where it rested
FRAME_INFO = ("executor", "env_step", "episode_seed", "episode_success")  # executor: 1 = student


@dataclass
class EpisodeSpec:
    seed: int
    driver: str = "student"
    switch_step: int | None = None
    perturbed: bool = False
    stress: str | None = None
    action_noise: float = 0.0
    record: bool = True
    stall_takeover: bool = True


@dataclass
class EpisodeResult:
    seed: int
    driver: str
    perturbed: bool
    success: bool
    steps: int
    switch_step: int | None = None
    takeover_step: int | None = None
    takeover_reason: str | None = None
    student_steps: int = 0
    stress_step: int | None = None
    arm_l1: float = float("nan")
    grip_disagree: float = float("nan")
    frames: dict[str, np.ndarray] | None = field(default=None, repr=False)

    def summary(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "frames"}


# ---------------------------------------------------------------- env worker process


def _pack(env, obs: dict, teacher, ensembler, cams: StudentCameras, held: int, render: bool) -> dict:
    """Everything the main process needs about the current state, including the teacher's chunk
    (gripper binarised) and its ensembled action. The ensembler advances once per state."""
    chunk = teacher.predict_chunk(policy_features(obs)[None])
    action = ensembler.step(chunk.copy())
    chunk[:, -1] = np.where(chunk[:, -1] > 0, 1.0, -1.0)
    action[-1] = 1.0 if action[-1] > 0 else -1.0
    return {"state": student_state(obs), "chunk": chunk.astype(np.float32),
            "teacher_action": action, "privileged": privileged(obs),
            "eef": obs["robot0_eef_pos"].copy(), "stacked": held > 0,
            "done": held >= HOLD_STEPS + AFTER_STEPS,
            "images": cams.render() if render else None}


def _teleport_cube(env, cube: str, xyz: np.ndarray) -> None:
    joint = (env.cubeA if cube == "A" else env.cubeB).joints[0]
    qpos = env.sim.data.get_joint_qpos(joint).copy()
    qpos[:3] = xyz
    env.sim.data.set_joint_qpos(joint, qpos)
    env.sim.data.set_joint_qvel(joint, np.zeros(6))
    env.sim.forward()


def _physical_stress(env, obs: dict, stress: str, seed: int, a_start: np.ndarray) -> None:
    """move_b: cube B moves 3-5 cm; drop: cube A lands back on the table 3-5 cm from where it started.
    Positions within MIN_CUBE_GAP of the other cube are redrawn."""
    rng = np.random.default_rng(seed + 1)
    other = obs["cubeA_pos"][:2] if stress == "move_b" else obs["cubeB_pos"][:2]
    base = obs["cubeB_pos"][:2] if stress == "move_b" else a_start[:2]
    while True:
        angle, radius = rng.uniform(0, 2 * np.pi), rng.uniform(*PERTURB_RANGE)
        xy = base + radius * np.array([np.cos(angle), np.sin(angle)])
        if np.linalg.norm(xy - other) >= MIN_CUBE_GAP:
            break
    if stress == "move_b":
        displace_cube(env, "B", xy - base)
    else:
        _teleport_cube(env, "A", np.array([xy[0], xy[1], a_start[2]]))


def _env_worker(conn, teacher_path: str, image_size: int) -> None:
    import torch

    from hc.policy import TemporalEnsembler, load_policy

    torch.set_num_threads(1)
    env = make_env(horizon=HORIZON)
    cams = StudentCameras(image_size, CAMERAS)
    teacher = load_policy(Path(teacher_path))
    ensembler = TemporalEnsembler(teacher.config.chunk)
    obs, held, t, perturb, stress, stress_step, seed, a_start = None, 0, 0, None, None, None, 0, None
    while True:
        msg = conn.recv()
        if msg[0] == "reset":
            _, seed, perturbed, stress, render = msg
            half = WIDE_RANGE if stress == "wide" else TRAIN_RANGE
            env.placement_initializer.x_range = [-half, half]
            env.placement_initializer.y_range = [-half, half]
            obs = reset(env, seed)
            cams.attach(env)
            if stress in ("dim", "table", "camera"):
                apply_visual_stress(env, stress)
            ensembler.reset()
            held, t, stress_step = 0, 0, None
            a_start = obs["cubeA_pos"].copy()
            perturb = sample_perturbation(seed, obs) if perturbed else None
            conn.send(_pack(env, obs, teacher, ensembler, cams, held, render) | {"stress_step": stress_step})
        elif msg[0] == "step":
            _, action, render = msg
            obs, *_ = env.step(action)
            t += 1
            held = held + 1 if env._check_success() else 0
            if perturb is not None and t == perturb.step:
                displace_cube(env, perturb.cube, perturb.dxy)
                obs = env._get_observations(force_update=True)
            if (stress in ("move_b", "drop") and stress_step is None
                    and obs["cubeA_pos"][2] > a_start[2] + LIFT_TRIGGER):
                _physical_stress(env, obs, stress, seed, a_start)
                obs = env._get_observations(force_update=True)
                stress_step = t
            conn.send(_pack(env, obs, teacher, ensembler, cams, held, render) | {"stress_step": stress_step})
        elif msg[0] == "render":
            conn.send(cams.render())
        elif msg[0] == "close":
            cams.close()
            env.close()
            return


# ---------------------------------------------------------------- lockstep rollouts


class _Slot:
    """Main-process state of one env's current episode."""

    def __init__(self, spec: EpisodeSpec):
        self.spec = spec
        self.driver = spec.driver
        self.t = 0
        self.queue: deque = deque()
        self.rng = np.random.default_rng(spec.seed)
        self.frames: dict[str, list] = {k: [] for k in ("images", "state", "chunk", "executor",
                                                          "privileged", "env_step")}
        self.eef_history: deque = deque(maxlen=STALL_STEPS + 1)
        self.takeover_step: int | None = None
        self.takeover_reason: str | None = None
        self.student_steps = 0
        self.arm_err: list[float] = []
        self.grip_err: list[float] = []

    def needs_images(self, t: int, n_action_steps: int) -> bool:
        record = self.spec.record and t % RECORD_STRIDE == 0
        return record or (self.driver == "student" and t % n_action_steps == 0)

    def hand_over(self, reason: str) -> None:
        self.driver = "teacher"
        self.queue.clear()
        self.takeover_step = self.t
        self.takeover_reason = reason

    def result(self, success: bool) -> EpisodeResult:
        frames = None
        if self.spec.record and self.frames["state"]:
            frames = {k: np.stack(v) for k, v in self.frames.items()}
        return EpisodeResult(
            seed=self.spec.seed, driver=self.spec.driver, perturbed=self.spec.perturbed, success=success,
            steps=self.t, switch_step=self.spec.switch_step, takeover_step=self.takeover_step,
            takeover_reason=self.takeover_reason, student_steps=self.student_steps,
            arm_l1=float(np.mean(self.arm_err)) if self.arm_err else float("nan"),
            grip_disagree=float(np.mean(self.grip_err)) if self.grip_err else float("nan"), frames=frames)


class RolloutPool:
    """`workers` env processes, each with a robosuite Stack env, the two student cameras and a CPU
    copy of the teacher."""

    def __init__(self, workers: int, teacher_path: Path, image_size: int = IMAGE_SIZE):
        ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        saved = {k: os.environ.get(k) for k in SINGLE_THREAD_VARS}
        os.environ.update({k: "1" for k in SINGLE_THREAD_VARS})  # inherited by the spawned workers
        for _ in range(workers):
            parent, child = ctx.Pipe()
            proc = ctx.Process(target=_env_worker, args=(child, str(teacher_path), image_size),
                               daemon=True)
            proc.start()
            self.conns.append(parent)
            self.procs.append(proc)
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k)
            else:
                os.environ[k] = v

    def close(self) -> None:
        for conn in self.conns:
            conn.send(("close",))
        for proc in self.procs:
            proc.join(timeout=30)

    def __enter__(self) -> "RolloutPool":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def run(self, specs: list[EpisodeSpec], student=None, horizon: int = HORIZON,
            n_action_steps: int = N_ACTION_STEPS, on_episode=None, noise_seed: int = 0,
            log_every_s: float = 60.0) -> list[EpisodeResult]:
        """Run every spec to completion and return the results in spec order.

        Episodes start only on steps that are multiples of `n_action_steps`, so every
        student-driven env needs a new chunk on the same steps and one batched forward pass
        serves them all. `on_episode(result)` is called as each episode finishes (e.g. to write it
        to a dataset), after which its frames are dropped from the returned result."""
        import torch

        if student is not None and n_action_steps > student.chunk:
            raise ValueError(f"n_action_steps {n_action_steps} > student chunk {student.chunk}")
        generator = torch.Generator().manual_seed(noise_seed)
        pending = deque(enumerate(specs))
        slots: list[tuple[int, _Slot] | None] = [None] * len(self.conns)
        packs: list[dict | None] = [None] * len(self.conns)
        results: list[EpisodeResult | None] = [None] * len(specs)
        clock, start, last_log = 0, time.time(), time.time()
        while pending or any(s is not None for s in slots):
            if clock % n_action_steps == 0:
                for w in range(len(self.conns)):
                    if slots[w] is None and pending:
                        index, spec = pending.popleft()
                        slot = _Slot(spec)
                        slots[w] = (index, slot)
                        self.conns[w].send(("reset", spec.seed, spec.perturbed, spec.stress,
                                            slot.needs_images(0, n_action_steps)))
                for w in range(len(self.conns)):
                    if slots[w] is not None and slots[w][1].t == 0:
                        packs[w] = self.conns[w].recv()
            active = [w for w in range(len(self.conns)) if slots[w] is not None]
            if not active:
                clock += 1
                continue

            for w in active:
                slot, pack = slots[w][1], packs[w]
                slot.eef_history.append(pack["eef"])
                if slot.driver == "student":
                    if slot.spec.switch_step is not None and slot.t >= slot.spec.switch_step:
                        slot.hand_over("switch")
                    elif (slot.spec.stall_takeover and not pack["stacked"]
                          and len(slot.eef_history) > STALL_STEPS
                          and np.linalg.norm(slot.eef_history[-1] - slot.eef_history[0]) < STALL_DIST):
                        slot.hand_over("stall")
                need = slot.driver == "student" and not slot.queue
                if (need or (slot.spec.record and slot.t % RECORD_STRIDE == 0)) and pack["images"] is None:
                    self.conns[w].send(("render",))
                    pack["images"] = self.conns[w].recv()

            infer = [w for w in active if slots[w][1].driver == "student" and not slots[w][1].queue]
            if infer:
                chunks = student.predict(np.stack([packs[w]["images"] for w in infer]),
                                         np.stack([packs[w]["state"] for w in infer]), generator)
                for w, chunk in zip(infer, chunks):
                    slots[w][1].queue.extend(chunk[:n_action_steps])

            for w in active:
                slot, pack = slots[w][1], packs[w]
                if slot.driver == "student":
                    action = slot.queue.popleft().astype(np.float64)
                    action[-1] = 1.0 if action[-1] > 0 else -1.0
                    slot.student_steps += 1
                    slot.arm_err.append(float(np.abs(action[:-1] - pack["chunk"][0, :-1]).mean()))
                    slot.grip_err.append(float(action[-1] != pack["chunk"][0, -1]))
                else:
                    action = pack["teacher_action"].copy()
                if slot.spec.action_noise > 0:
                    action[:-1] = np.clip(action[:-1] + slot.rng.normal(0.0, slot.spec.action_noise,
                                                                          ACTION_DIM - 1), -1, 1)
                if slot.spec.record and slot.t % RECORD_STRIDE == 0:
                    slot.frames["images"].append(pack["images"])
                    slot.frames["state"].append(pack["state"])
                    slot.frames["chunk"].append(pack["chunk"])
                    slot.frames["executor"].append(float(slot.driver == "student"))
                    slot.frames["privileged"].append(pack["privileged"])
                    slot.frames["env_step"].append(float(slot.t))
                self.conns[w].send(("step", action, slot.needs_images(slot.t + 1, n_action_steps)))

            for w in active:
                index, slot = slots[w]
                packs[w] = self.conns[w].recv()
                slot.t += 1
                if packs[w]["done"] or slot.t >= horizon:
                    result = slot.result(bool(packs[w]["done"]))
                    result.stress_step = packs[w]["stress_step"]
                    if on_episode is not None:
                        on_episode(result)
                    result.frames = None
                    results[index] = result
                    slots[w] = None
            clock += 1
            if time.time() - last_log > log_every_s:
                finished = [r for r in results if r is not None]
                rate = np.mean([r.success for r in finished]) if finished else float("nan")
                print(f"  {len(finished)}/{len(specs)} episodes, success {rate:.2f}, "
                      f"{time.time() - start:.0f}s", flush=True)
                last_log = time.time()
        return results


# ---------------------------------------------------------------- datasets


def dataset_features(image_size: int = IMAGE_SIZE) -> dict:
    image = {"dtype": "video", "shape": (image_size, image_size, 3),
             "names": ["height", "width", "channels"]}
    vector = lambda n: {"dtype": "float32", "shape": (n,), "names": None}
    return {IMAGE_KEYS[0]: image, IMAGE_KEYS[1]: dict(image),
            "observation.state": vector(STATE_DIM), "action": vector(ACTION_DIM),
            "teacher_chunk": vector(CHUNK * ACTION_DIM), "privileged": vector(PRIVILEGED_DIM),
            "frame_info": vector(len(FRAME_INFO))}


class DatasetWriter:
    """Writes recorded episodes to a LeRobot dataset (AV1 video for the cameras) from a background
    thread, so writing does not stall the lockstep rollouts. LeRobot's streaming encoder feeds frames
    straight to per-camera encoder threads; its default path writes PNGs and then spends ~4.5 s per
    episode on image statistics that nothing here reads. An error in the writer thread is re-raised
    in the caller, and `finalize` (call it in a `finally`) leaves a readable dataset of the episodes
    written so far.

    "action" is the teacher's first chunk action; the whole teacher chunk is stored under
    "teacher_chunk", a key LeRobot does not treat as a policy feature. "frame_info" holds the
    FRAME_INFO scalars (shape-(1,) features would go through a deprecated NumPy conversion)."""

    def __init__(self, root: Path, repo_id: str, image_size: int = IMAGE_SIZE, keep=None,
                 max_pending: int = 8):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.dataset = LeRobotDataset.create(repo_id, fps=DATASET_FPS, features=dataset_features(image_size),
                                             root=root, robot_type="panda", use_videos=True,
                                             streaming_encoding=True)
        self.keep = keep
        self.episodes = 0
        self.frames = 0
        self.error: BaseException | None = None
        self.pending: queue.Queue = queue.Queue(max_pending)
        self.thread = threading.Thread(target=self._drain, daemon=True)
        self.thread.start()

    def __call__(self, result: EpisodeResult) -> None:
        if result.frames is None or (self.keep is not None and not self.keep(result)):
            return
        self._put((result.seed, result.success, result.frames))

    def _put(self, item) -> None:
        while self.thread.is_alive():
            try:
                self.pending.put(item, timeout=1.0)
                return
            except queue.Full:
                pass
        self._raise_if_failed()

    def _raise_if_failed(self) -> None:
        if self.error is not None:
            raise RuntimeError("dataset writer thread failed") from self.error

    def _drain(self) -> None:
        try:
            while (item := self.pending.get()) is not None:
                self._write(*item)
        except BaseException as error:
            self.error = error

    def _write(self, seed: int, success: bool, f: dict[str, np.ndarray]) -> None:
        for i in range(len(f["state"])):
            self.dataset.add_frame({
                IMAGE_KEYS[0]: f["images"][i, 0], IMAGE_KEYS[1]: f["images"][i, 1],
                "observation.state": f["state"][i], "action": f["chunk"][i, 0],
                "teacher_chunk": f["chunk"][i].reshape(-1),
                "privileged": f["privileged"][i],
                "frame_info": np.array([f["executor"][i], f["env_step"][i], seed, success], np.float32),
                "task": TASK,
            })
        self.dataset.save_episode()
        self.episodes += 1
        self.frames += len(f["state"])

    def finalize(self) -> None:
        self._put(None)
        self.thread.join()
        self.dataset.finalize()
        self._raise_if_failed()


def round_seeds(round_index: int, n: int) -> list[int]:
    """Training layouts for DAgger round `round_index` (round 0 = teacher data). Disjoint from
    datagen (0-1599), self-improvement (50000+) and evaluation (1e6+)."""
    start = 70_000 + 2_000 * round_index
    return list(range(start, start + n))


def teacher_specs(seeds: list[int], noise: float = TEACHER_NOISE) -> list[EpisodeSpec]:
    return [EpisodeSpec(s, driver="teacher", action_noise=noise) for s in seeds]


def dagger_specs(seeds: list[int], switch_fraction: float, max_switch: int = 120) -> list[EpisodeSpec]:
    """A `switch_fraction` share of switch-once episodes (student for U[0, max_switch) steps, then
    the teacher), the rest driven by the student throughout (with the stall takeover)."""
    rng = np.random.default_rng(seeds[0])
    specs = []
    for s in seeds:
        switch = int(rng.integers(0, max_switch)) if rng.random() < switch_fraction else None
        specs.append(EpisodeSpec(s, driver="student", switch_step=switch))
    return specs


def eval_specs(seeds: list[int], perturbed: bool, stress: str | None = None) -> list[EpisodeSpec]:
    noise = STRESS_NOISE if stress == "noise" else 0.0
    return [EpisodeSpec(s, driver="student", perturbed=perturbed, stress=stress, action_noise=noise,
                        record=False, stall_takeover=False) for s in seeds]
