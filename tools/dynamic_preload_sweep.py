"""Sweep dynamic lateral preload motions for Blue-like in-place load transfer."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np

from sedon_baseline.env import JOINT_NAMES, SedonStandingEnv, load_sedon_config_from_env
from tools.sedon_debug_common import (
    BASE_PROXY_GEOM,
    DEBUG_OUT_DIR,
    DEFAULT_SCENE_PATH,
    FLOOR_GEOM,
    LEFT_FOOT_GEOM,
    RIGHT_FOOT_GEOM,
    require_scene,
)


DEFAULT_OUT_CSV = DEBUG_OUT_DIR / "dynamic_preload_sweep.csv"
HIP_ROLL_INDEX = {"right": JOINT_NAMES.index("R_joint_hip_roll"), "left": JOINT_NAMES.index("L_joint_hip_roll")}
TARGET_RATIO_THRESHOLD = 0.58
UPRIGHT_MIN = 0.98
MAX_PENETRATION_M = 0.0015
TOTAL_FORCE_NORM_MIN = 0.8
TOTAL_FORCE_NORM_MAX = 1.3
BASE_HEIGHT_DROP_MAX = 0.015
CONTACT_FORCE_THRESHOLD_N = 5.0


@dataclass(frozen=True)
class DynamicPreloadCandidate:
    """One dynamic preload motion candidate."""

    side: str
    profile: str
    phase_duration_s: float
    hip_roll_amplitude: float
    lean_amplitude: float
    swing_hip_roll_amplitude: float


@dataclass(frozen=True)
class DynamicPreloadRow:
    """One evaluated dynamic preload result."""

    side: str
    profile: str
    phase_duration_s: float
    hip_roll_amplitude: float
    lean_amplitude: float
    swing_hip_roll_amplitude: float
    peak_target_ratio: float
    min_swing_ratio: float
    duration_above_0_58: float
    both_contact_ratio: float
    contact_none_ratio: float
    max_abs_base_roll: float
    max_abs_base_pitch: float
    base_height_drop: float
    max_penetration: float
    min_upright: float
    min_total_force_normalized: float
    max_total_force_normalized: float
    no_foot_collision: bool
    no_base_proxy_contact: bool
    passed: bool
    score: float


@dataclass(frozen=True)
class FrameMetrics:
    """Per-frame dynamic load-transfer metrics."""

    target_ratio: float
    swing_ratio: float
    total_force_normalized: float
    both_contact: bool
    contact_none: bool
    foot_collision: bool
    base_proxy_contact: bool
    base_height: float
    upright: float
    base_roll: float
    base_pitch: float
    max_penetration: float


def _motion_alpha(profile: str, phase: float) -> float:
    """Return preload interpolation alpha for a normalized phase."""
    phase = float(np.clip(phase, 0.0, 1.0))
    if profile == "smoothstep":
        return phase * phase * (3.0 - 2.0 * phase)
    if profile == "sinusoidal":
        return float(np.sin(0.5 * np.pi * phase))
    raise ValueError(f"Unsupported profile: {profile}")


def _build_target(env: SedonStandingEnv, candidate: DynamicPreloadCandidate, alpha: float) -> np.ndarray:
    """Build an in-place preload target for one motion frame."""
    target = env._nominal_joint_qpos.copy()
    stance = candidate.side
    swing = "right" if stance == "left" else "left"
    if stance == "left":
        target[HIP_ROLL_INDEX[stance]] -= candidate.hip_roll_amplitude * alpha
        target[HIP_ROLL_INDEX[swing]] += (candidate.lean_amplitude + candidate.swing_hip_roll_amplitude) * alpha
    else:
        target[HIP_ROLL_INDEX[stance]] += candidate.hip_roll_amplitude * alpha
        target[HIP_ROLL_INDEX[swing]] -= (candidate.lean_amplitude + candidate.swing_hip_roll_amplitude) * alpha
    return target


def _contact_metrics(env: SedonStandingEnv) -> tuple[float, float, float, bool, bool, bool, bool]:
    """Return left/right vertical load and contact state flags."""
    left_world_z = 0.0
    right_world_z = 0.0
    max_penetration = 0.0
    left_contact = False
    right_contact = False
    foot_collision = False
    base_proxy_contact = False
    wrench = np.zeros(6, dtype=np.float64)
    for contact_index in range(env.data.ncon):
        contact = env.data.contact[contact_index]
        geom1 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1))
        geom2 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2))
        pair = {geom1, geom2}
        if pair == {LEFT_FOOT_GEOM, RIGHT_FOOT_GEOM}:
            foot_collision = True
        if pair == {FLOOR_GEOM, BASE_PROXY_GEOM}:
            base_proxy_contact = True
        if pair == {FLOOR_GEOM, LEFT_FOOT_GEOM}:
            left_contact = True
        elif pair == {FLOOR_GEOM, RIGHT_FOOT_GEOM}:
            right_contact = True
        else:
            continue
        mujoco.mj_contactForce(env.model, env.data, contact_index, wrench)
        contact_frame = np.asarray(contact.frame, dtype=np.float64).reshape(3, 3)
        world_force = contact_frame.T @ wrench[:3]
        world_z = abs(float(world_force[2]))
        max_penetration = max(max_penetration, max(0.0, -float(contact.dist)))
        if pair == {FLOOR_GEOM, LEFT_FOOT_GEOM}:
            left_world_z += world_z
        else:
            right_world_z += world_z
    return (
        left_world_z,
        right_world_z,
        max_penetration,
        foot_collision,
        base_proxy_contact,
        bool(left_contact and right_contact),
        bool((not left_contact) and (not right_contact)),
    )


def _collect_frame_metrics(env: SedonStandingEnv, side: str, initial_base_height: float) -> FrameMetrics:
    """Collect one frame of dynamic preload metrics."""
    left_world_z, right_world_z, max_penetration, foot_collision, base_proxy_contact, both_contact, contact_none = _contact_metrics(env)
    total_world_z = left_world_z + right_world_z
    left_ratio = float(left_world_z / (total_world_z + 1e-6))
    right_ratio = float(right_world_z / (total_world_z + 1e-6))
    target_ratio = left_ratio if side == "left" else right_ratio
    swing_ratio = right_ratio if side == "left" else left_ratio
    total_robot_weight = float(np.sum(env.model.body_mass) * 9.81)
    return FrameMetrics(
        target_ratio=target_ratio,
        swing_ratio=swing_ratio,
        total_force_normalized=float(total_world_z / max(total_robot_weight, 1e-6)),
        both_contact=both_contact,
        contact_none=contact_none,
        foot_collision=foot_collision,
        base_proxy_contact=base_proxy_contact,
        base_height=float(env.data.xpos[env._base_body_id][2]),
        upright=float(env._base_upright()),
        base_roll=float(env._base_roll()),
        base_pitch=float(env._base_pitch()),
        max_penetration=max_penetration,
    )


def _mean_bool(values: list[bool]) -> float:
    """Return fraction of true values."""
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else 0.0


def _evaluate_candidate(env: SedonStandingEnv, candidate: DynamicPreloadCandidate) -> DynamicPreloadRow:
    """Run one dynamic preload candidate and summarize the full motion."""
    env.reset(seed=0)
    initial_base_height = float(env.data.xpos[env._base_body_id][2])
    steps = max(1, int(round(candidate.phase_duration_s / float(env.model.opt.timestep))))
    frames: list[FrameMetrics] = []
    for step_index in range(steps):
        phase = step_index / max(1, steps - 1)
        alpha = _motion_alpha(candidate.profile, phase)
        target = _build_target(env, candidate, alpha)
        env._do_pd_simulation(target)
        frames.append(_collect_frame_metrics(env, candidate.side, initial_base_height))

    peak_target_ratio = max((frame.target_ratio for frame in frames), default=0.0)
    min_swing_ratio = min((frame.swing_ratio for frame in frames), default=0.0)
    duration_above = sum(
        float(env.model.opt.timestep)
        for frame in frames
        if frame.target_ratio >= TARGET_RATIO_THRESHOLD
    )
    both_contact_ratio = _mean_bool([frame.both_contact for frame in frames])
    contact_none_ratio = _mean_bool([frame.contact_none for frame in frames])
    max_abs_base_roll = max((abs(frame.base_roll) for frame in frames), default=0.0)
    max_abs_base_pitch = max((abs(frame.base_pitch) for frame in frames), default=0.0)
    min_base_height = min((frame.base_height for frame in frames), default=initial_base_height)
    base_height_drop = max(0.0, initial_base_height - min_base_height)
    max_penetration = max((frame.max_penetration for frame in frames), default=0.0)
    min_upright = min((frame.upright for frame in frames), default=0.0)
    min_total_force_norm = min((frame.total_force_normalized for frame in frames), default=0.0)
    max_total_force_norm = max((frame.total_force_normalized for frame in frames), default=0.0)
    no_foot_collision = not any(frame.foot_collision for frame in frames)
    no_base_proxy_contact = not any(frame.base_proxy_contact for frame in frames)
    passed = (
        peak_target_ratio >= TARGET_RATIO_THRESHOLD
        and min_swing_ratio < 1.0 - TARGET_RATIO_THRESHOLD
        and both_contact_ratio >= 0.95
        and contact_none_ratio == 0.0
        and min_upright >= UPRIGHT_MIN
        and base_height_drop <= BASE_HEIGHT_DROP_MAX
        and max_penetration <= MAX_PENETRATION_M
        and min_total_force_norm >= TOTAL_FORCE_NORM_MIN
        and max_total_force_norm <= TOTAL_FORCE_NORM_MAX
        and no_foot_collision
    )
    row = DynamicPreloadRow(
        side=candidate.side,
        profile=candidate.profile,
        phase_duration_s=float(candidate.phase_duration_s),
        hip_roll_amplitude=float(candidate.hip_roll_amplitude),
        lean_amplitude=float(candidate.lean_amplitude),
        swing_hip_roll_amplitude=float(candidate.swing_hip_roll_amplitude),
        peak_target_ratio=float(peak_target_ratio),
        min_swing_ratio=float(min_swing_ratio),
        duration_above_0_58=float(duration_above),
        both_contact_ratio=float(both_contact_ratio),
        contact_none_ratio=float(contact_none_ratio),
        max_abs_base_roll=float(max_abs_base_roll),
        max_abs_base_pitch=float(max_abs_base_pitch),
        base_height_drop=float(base_height_drop),
        max_penetration=float(max_penetration),
        min_upright=float(min_upright),
        min_total_force_normalized=float(min_total_force_norm),
        max_total_force_normalized=float(max_total_force_norm),
        no_foot_collision=bool(no_foot_collision),
        no_base_proxy_contact=bool(no_base_proxy_contact),
        passed=bool(passed),
        score=0.0,
    )
    return DynamicPreloadRow(**{**asdict(row), "score": _score(row)})


def _score(row: DynamicPreloadRow) -> float:
    """Return lower-is-better score that prioritizes reaching target load."""
    ratio_reward = max(0.0, TARGET_RATIO_THRESHOLD - row.peak_target_ratio)
    contact_penalty = row.contact_none_ratio + max(0.0, 0.95 - row.both_contact_ratio)
    upright_penalty = max(0.0, UPRIGHT_MIN - row.min_upright) * 5.0
    height_penalty = max(0.0, row.base_height_drop - BASE_HEIGHT_DROP_MAX) * 10.0
    penetration_penalty = max(0.0, row.max_penetration - MAX_PENETRATION_M) * 100.0
    force_penalty = max(0.0, TOTAL_FORCE_NORM_MIN - row.min_total_force_normalized)
    force_penalty += max(0.0, row.max_total_force_normalized - TOTAL_FORCE_NORM_MAX)
    collision_penalty = 1.0 if not row.no_foot_collision else 0.0
    return float(
        ratio_reward
        + contact_penalty
        + upright_penalty
        + height_penalty
        + penetration_penalty
        + force_penalty
        + collision_penalty
        - 0.1 * row.duration_above_0_58
    )


def _candidates() -> list[DynamicPreloadCandidate]:
    """Return the requested dynamic preload grid."""
    durations = [0.2, 0.3, 0.4, 0.5]
    hip_roll_amplitudes = [0.005, 0.01, 0.015, 0.02]
    lean_amplitudes = [0.005, 0.01, 0.015, 0.02]
    swing_hip_roll_amplitudes = [0.005, 0.01, 0.02]
    profiles = ["smoothstep", "sinusoidal"]
    return [
        DynamicPreloadCandidate(side, profile, duration, hip_amp, lean_amp, swing_amp)
        for side in ("left", "right")
        for profile in profiles
        for duration in durations
        for hip_amp in hip_roll_amplitudes
        for lean_amp in lean_amplitudes
        for swing_amp in swing_hip_roll_amplitudes
    ]


def _write_csv(path: Path, rows: list[DynamicPreloadRow]) -> None:
    """Write dynamic preload sweep rows to CSV."""
    fieldnames = list(asdict(rows[0]).keys())
    fieldnames[fieldnames.index("passed")] = "pass"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = asdict(row)
            payload["pass"] = payload.pop("passed")
            writer.writerow(payload)


def _top_rows(rows: list[DynamicPreloadRow], top_k: int) -> list[DynamicPreloadRow]:
    """Return top candidates, preferring passing rows."""
    return sorted(rows, key=lambda row: (not row.passed, row.score))[:top_k]


def run_sweep(scene_path: Path, out_csv: Path, top_k: int) -> list[DynamicPreloadRow]:
    """Run dynamic preload sweep and write CSV."""
    env = SedonStandingEnv(
        scene_path=require_scene(scene_path),
        reset_noise_scale=0.0,
        reward_config=load_sedon_config_from_env(),
    )
    rows = [_evaluate_candidate(env, candidate) for candidate in _candidates()]
    _write_csv(out_csv, rows)
    top_rows = _top_rows(rows, top_k)
    print(f"evaluated {len(rows)} candidates")
    print(f"wrote rows to {out_csv}")
    print(f"top {len(top_rows)} candidates")
    for rank, row in enumerate(top_rows, start=1):
        status = "PASS" if row.passed else "fail"
        print(
            f"{rank:>2}. {status} side={row.side:<5} profile={row.profile:<10} duration={row.phase_duration_s:.1f}s "
            f"hip={row.hip_roll_amplitude:.3f} lean={row.lean_amplitude:.3f} swing={row.swing_hip_roll_amplitude:.3f} "
            f"peak={row.peak_target_ratio:.3f} min_swing={row.min_swing_ratio:.3f} "
            f"above={row.duration_above_0_58:.3f}s both={row.both_contact_ratio:.2f} none={row.contact_none_ratio:.2f} "
            f"roll={row.max_abs_base_roll:.3f} pitch={row.max_abs_base_pitch:.3f} "
            f"drop={row.base_height_drop:.4f} pen={row.max_penetration * 1000.0:.2f}mm "
            f"force=[{row.min_total_force_normalized:.3f},{row.max_total_force_normalized:.3f}] score={row.score:.4f}"
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE_PATH)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--top-k", type=int, default=20)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run dynamic preload sweep CLI."""
    args = build_parser().parse_args(argv)
    run_sweep(scene_path=args.scene, out_csv=args.out_csv, top_k=args.top_k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
