"""Two-stage Seedon preload sweep with stance knee/ankle and swing hip-roll deltas."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np

from seedon_baseline.env import JOINT_NAMES, SeedonStandingEnv, load_seedon_config_from_env
from tools.seedon_debug_common import (
    BASE_PROXY_GEOM,
    DEBUG_OUT_DIR,
    DEFAULT_SCENE_PATH,
    FLOOR_GEOM,
    LEFT_FOOT_GEOM,
    RIGHT_FOOT_GEOM,
    require_scene,
)


DEFAULT_OUT_CSV = DEBUG_OUT_DIR / "seedon_preload_sweep_v2.csv"
HIP_ROLL_INDEX = {"right": JOINT_NAMES.index("R_joint_hip_roll"), "left": JOINT_NAMES.index("L_joint_hip_roll")}
KNEE_PITCH_INDEX = {"right": JOINT_NAMES.index("R_joint_knee_pitch"), "left": JOINT_NAMES.index("L_joint_knee_pitch")}
ANKLE_PITCH_INDEX = {"right": JOINT_NAMES.index("R_joint_ankle_pitch"), "left": JOINT_NAMES.index("L_joint_ankle_pitch")}
TARGET_RATIO_MIN = 0.58
TARGET_RATIO_MAX = 0.65
UPRIGHT_MIN = 0.98
MAX_PENETRATION_M = 0.0015
TOTAL_FORCE_NORM_MIN = 0.8
TOTAL_FORCE_NORM_MAX = 1.2
BASE_HEIGHT_DROP_MAX = 0.015
CONTACT_FORCE_THRESHOLD_N = 5.0
MOVING_AVERAGE_FRAMES = 30


@dataclass(frozen=True)
class PreloadV2Candidate:
    """One preload-v2 target candidate."""

    side: str
    hip_roll: float
    lean_roll: float
    stance_knee_pitch_delta: float
    stance_ankle_pitch_delta: float
    swing_hip_roll_delta: float


@dataclass(frozen=True)
class PreloadV2Row:
    """One evaluated preload-v2 result row."""

    side: str
    hip_roll: float
    lean_roll: float
    stance_knee_pitch_delta: float
    stance_ankle_pitch_delta: float
    swing_hip_roll_delta: float
    force_ratio_left: float
    force_ratio_right: float
    base_height: float
    upright: float
    max_penetration: float
    total_force_normalized: float
    support_side_guess: str
    no_foot_collision: bool
    both_feet_contact: bool
    passed: bool
    base_height_drop: float
    score: float


@dataclass(frozen=True)
class FrameMetrics:
    """Contact and stability metrics for one dynamic frame."""

    left_world_z: float
    right_world_z: float
    max_penetration: float
    base_height: float
    upright: float
    foot_collision: bool
    both_feet_contact: bool


def _parse_float_list(raw_value: str) -> list[float]:
    """Parse a comma-separated list of floats."""
    values = [float(part.strip()) for part in raw_value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one float value.")
    return values


def _support_side_guess(left_force: float, right_force: float) -> str:
    """Return support-side label from left/right vertical force."""
    left_contact = left_force > CONTACT_FORCE_THRESHOLD_N
    right_contact = right_force > CONTACT_FORCE_THRESHOLD_N
    if left_contact and right_contact:
        return "double"
    if left_contact:
        return "left"
    if right_contact:
        return "right"
    return "none"


def _contact_metrics(env: SeedonStandingEnv) -> tuple[float, float, float, bool, bool]:
    """Return foot world-z loads, penetration, foot collision, and both-foot contact."""
    left_world_z = 0.0
    right_world_z = 0.0
    max_penetration = 0.0
    left_contact = False
    right_contact = False
    foot_collision = False
    wrench = np.zeros(6, dtype=np.float64)
    for contact_index in range(env.data.ncon):
        contact = env.data.contact[contact_index]
        geom1 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1))
        geom2 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2))
        pair = {geom1, geom2}
        if pair == {LEFT_FOOT_GEOM, RIGHT_FOOT_GEOM}:
            foot_collision = True
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
    return left_world_z, right_world_z, max_penetration, foot_collision, bool(left_contact and right_contact)


def _build_target(env: SeedonStandingEnv, candidate: PreloadV2Candidate) -> np.ndarray:
    """Build a static in-place preload target without lift or forward motion."""
    target = env._nominal_joint_qpos.copy()
    stance = candidate.side
    swing = "right" if stance == "left" else "left"
    target[HIP_ROLL_INDEX[stance]] += candidate.hip_roll
    if stance == "left":
        target[HIP_ROLL_INDEX[swing]] += candidate.lean_roll + candidate.swing_hip_roll_delta
    else:
        target[HIP_ROLL_INDEX[swing]] -= candidate.lean_roll + candidate.swing_hip_roll_delta
    target[KNEE_PITCH_INDEX[stance]] += candidate.stance_knee_pitch_delta
    target[ANKLE_PITCH_INDEX[stance]] += candidate.stance_ankle_pitch_delta
    return target


def _collect_frame_metrics(env: SeedonStandingEnv) -> FrameMetrics:
    """Collect one frame of preload metrics."""
    left_world_z, right_world_z, max_penetration, foot_collision, both_feet_contact = _contact_metrics(env)
    return FrameMetrics(
        left_world_z=left_world_z,
        right_world_z=right_world_z,
        max_penetration=max_penetration,
        base_height=float(env.data.xpos[env._base_body_id][2]),
        upright=float(env._base_upright()),
        foot_collision=foot_collision,
        both_feet_contact=both_feet_contact,
    )


def _mean(values: list[float]) -> float:
    """Return zero-safe mean."""
    return float(np.mean(values)) if values else 0.0


def _score(row: PreloadV2Row) -> float:
    """Return lower-is-better score, with pass rows naturally ranking first."""
    target_ratio = row.force_ratio_left if row.side == "left" else row.force_ratio_right
    target_mid = 0.5 * (TARGET_RATIO_MIN + TARGET_RATIO_MAX)
    ratio_penalty = abs(target_ratio - target_mid)
    upright_penalty = max(0.0, UPRIGHT_MIN - row.upright) * 5.0
    penetration_penalty = max(0.0, row.max_penetration - MAX_PENETRATION_M) * 100.0
    force_penalty = max(0.0, TOTAL_FORCE_NORM_MIN - row.total_force_normalized)
    force_penalty += max(0.0, row.total_force_normalized - TOTAL_FORCE_NORM_MAX)
    height_penalty = max(0.0, row.base_height_drop - BASE_HEIGHT_DROP_MAX) * 10.0
    collision_penalty = 1.0 if not row.no_foot_collision else 0.0
    both_contact_penalty = 0.05 if not row.both_feet_contact else 0.0
    return float(
        ratio_penalty
        + upright_penalty
        + penetration_penalty
        + force_penalty
        + height_penalty
        + collision_penalty
        + both_contact_penalty
    )


def _evaluate_candidate(
    env: SeedonStandingEnv,
    candidate: PreloadV2Candidate,
    *,
    settle_steps: int,
) -> PreloadV2Row:
    """Evaluate one candidate with dynamic PD settle and no lift/forward command."""
    env.reset(seed=0)
    initial_base_height = float(env.data.xpos[env._base_body_id][2])
    target = _build_target(env, candidate)
    frames: list[FrameMetrics] = []
    for _ in range(settle_steps):
        env._do_pd_simulation(target)
        frames.append(_collect_frame_metrics(env))

    tail = frames[-MOVING_AVERAGE_FRAMES:]
    left_world_z = _mean([frame.left_world_z for frame in tail])
    right_world_z = _mean([frame.right_world_z for frame in tail])
    total_world_z = left_world_z + right_world_z
    force_ratio_left = float(left_world_z / (total_world_z + 1e-6))
    force_ratio_right = float(right_world_z / (total_world_z + 1e-6))
    total_robot_weight = float(np.sum(env.model.body_mass) * 9.81)
    base_height = _mean([frame.base_height for frame in tail])
    upright = _mean([frame.upright for frame in tail])
    max_penetration = max((frame.max_penetration for frame in tail), default=0.0)
    total_force_normalized = float(total_world_z / max(total_robot_weight, 1e-6))
    no_foot_collision = not any(frame.foot_collision for frame in tail)
    both_feet_contact = all(frame.both_feet_contact for frame in tail) if tail else False
    base_height_drop = max(0.0, initial_base_height - base_height)
    target_ratio = force_ratio_left if candidate.side == "left" else force_ratio_right
    passed = (
        TARGET_RATIO_MIN <= target_ratio <= TARGET_RATIO_MAX
        and upright >= UPRIGHT_MIN
        and max_penetration <= MAX_PENETRATION_M
        and TOTAL_FORCE_NORM_MIN <= total_force_normalized <= TOTAL_FORCE_NORM_MAX
        and base_height_drop <= BASE_HEIGHT_DROP_MAX
        and no_foot_collision
    )
    row = PreloadV2Row(
        side=candidate.side,
        hip_roll=float(candidate.hip_roll),
        lean_roll=float(candidate.lean_roll),
        stance_knee_pitch_delta=float(candidate.stance_knee_pitch_delta),
        stance_ankle_pitch_delta=float(candidate.stance_ankle_pitch_delta),
        swing_hip_roll_delta=float(candidate.swing_hip_roll_delta),
        force_ratio_left=force_ratio_left,
        force_ratio_right=force_ratio_right,
        base_height=float(base_height),
        upright=float(upright),
        max_penetration=float(max_penetration),
        total_force_normalized=total_force_normalized,
        support_side_guess=_support_side_guess(left_world_z, right_world_z),
        no_foot_collision=bool(no_foot_collision),
        both_feet_contact=bool(both_feet_contact),
        passed=bool(passed),
        base_height_drop=float(base_height_drop),
        score=0.0,
    )
    return PreloadV2Row(**{**asdict(row), "score": _score(row)})


def _base_candidates() -> list[PreloadV2Candidate]:
    """Return v1 hip-roll + lean-roll base candidates with zero added deltas."""
    left_hip_rolls = [-0.001 * index for index in range(16)]
    right_hip_rolls = [0.001 * index for index in range(16)]
    lean_rolls = [0.005 * index for index in range(5)]
    candidates: list[PreloadV2Candidate] = []
    for hip_roll in left_hip_rolls:
        for lean_roll in lean_rolls:
            candidates.append(PreloadV2Candidate("left", hip_roll, lean_roll, 0.0, 0.0, 0.0))
    for hip_roll in right_hip_rolls:
        for lean_roll in lean_rolls:
            candidates.append(PreloadV2Candidate("right", hip_roll, lean_roll, 0.0, 0.0, 0.0))
    return candidates


def _expand_candidates(
    bases: list[PreloadV2Row],
    *,
    stance_knee_pitch_deltas: list[float],
    stance_ankle_pitch_deltas: list[float],
    swing_hip_roll_deltas: list[float],
) -> list[PreloadV2Candidate]:
    """Expand selected base rows into preload-v2 candidates."""
    candidates: list[PreloadV2Candidate] = []
    seen: set[tuple[object, ...]] = set()
    for base in bases:
        for knee_delta in stance_knee_pitch_deltas:
            for ankle_delta in stance_ankle_pitch_deltas:
                for swing_delta in swing_hip_roll_deltas:
                    key = (
                        base.side,
                        base.hip_roll,
                        base.lean_roll,
                        knee_delta,
                        ankle_delta,
                        swing_delta,
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append(
                        PreloadV2Candidate(
                            side=base.side,
                            hip_roll=base.hip_roll,
                            lean_roll=base.lean_roll,
                            stance_knee_pitch_delta=knee_delta,
                            stance_ankle_pitch_delta=ankle_delta,
                            swing_hip_roll_delta=swing_delta,
                        )
                    )
    return candidates


def _write_csv(path: Path, rows: list[PreloadV2Row]) -> None:
    """Write preload-v2 rows to CSV."""
    fieldnames = [
        "side",
        "hip_roll",
        "lean_roll",
        "stance_knee_pitch_delta",
        "stance_ankle_pitch_delta",
        "swing_hip_roll_delta",
        "force_ratio_left",
        "force_ratio_right",
        "base_height",
        "upright",
        "max_penetration",
        "total_force_normalized",
        "support_side_guess",
        "no_foot_collision",
        "both_feet_contact",
        "pass",
        "base_height_drop",
        "score",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = asdict(row)
            payload["pass"] = payload.pop("passed")
            writer.writerow(payload)


def _top_rows(rows: list[PreloadV2Row], top_k: int) -> list[PreloadV2Row]:
    """Return top rows, preferring pass rows and then score."""
    return sorted(rows, key=lambda row: (not row.passed, row.score))[:top_k]


def run_sweep(
    *,
    scene_path: Path,
    out_csv: Path,
    settle_steps: int,
    top_k: int,
    expand_base_top_k: int,
    stance_knee_pitch_deltas: list[float],
    stance_ankle_pitch_deltas: list[float],
    swing_hip_roll_deltas: list[float],
    exhaustive: bool,
) -> list[PreloadV2Row]:
    """Run preload-v2 sweep and write CSV."""
    if settle_steps < MOVING_AVERAGE_FRAMES:
        raise ValueError(f"settle_steps must be at least {MOVING_AVERAGE_FRAMES}.")
    env = SeedonStandingEnv(
        scene_path=require_scene(scene_path),
        reset_noise_scale=0.0,
        reward_config=load_seedon_config_from_env(),
    )
    base_rows = [
        _evaluate_candidate(env, candidate, settle_steps=settle_steps)
        for candidate in _base_candidates()
    ]
    selected_bases = base_rows if exhaustive else _top_rows(base_rows, expand_base_top_k)
    expanded_candidates = _expand_candidates(
        selected_bases,
        stance_knee_pitch_deltas=stance_knee_pitch_deltas,
        stance_ankle_pitch_deltas=stance_ankle_pitch_deltas,
        swing_hip_roll_deltas=swing_hip_roll_deltas,
    )
    rows = [
        _evaluate_candidate(env, candidate, settle_steps=settle_steps)
        for candidate in expanded_candidates
    ]
    _write_csv(out_csv, rows)
    top_rows = _top_rows(rows, top_k)
    print(f"base candidates evaluated: {len(base_rows)}")
    print(f"v2 candidates evaluated: {len(rows)}")
    print(f"wrote rows to {out_csv}")
    print(f"top {len(top_rows)} candidates")
    for rank, row in enumerate(top_rows, start=1):
        target_ratio = row.force_ratio_left if row.side == "left" else row.force_ratio_right
        status = "PASS" if row.passed else "fail"
        print(
            f"{rank:>2}. {status} side={row.side:<5} hip={row.hip_roll:+.4f} lean={row.lean_roll:+.4f} "
            f"knee={row.stance_knee_pitch_delta:+.4f} ankle={row.stance_ankle_pitch_delta:+.4f} "
            f"swing_roll={row.swing_hip_roll_delta:+.4f} target_ratio={target_ratio:.3f} "
            f"L={row.force_ratio_left:.3f} R={row.force_ratio_right:.3f} upright={row.upright:.3f} "
            f"pen={row.max_penetration * 1000.0:.2f}mm force_norm={row.total_force_normalized:.3f} "
            f"base_drop={row.base_height_drop:.4f} both={row.both_feet_contact} "
            f"foot_collision={not row.no_foot_collision} score={row.score:.4f}"
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE_PATH)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--settle-steps", type=int, choices=(180, 240), default=240)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--expand-base-top-k", type=int, default=40)
    parser.add_argument("--stance-knee-pitch-deltas", type=_parse_float_list, default="0,0.01,0.02")
    parser.add_argument("--stance-ankle-pitch-deltas", type=_parse_float_list, default="0,-0.01,0.01")
    parser.add_argument("--swing-hip-roll-deltas", type=_parse_float_list, default="0,0.01,0.02")
    parser.add_argument("--exhaustive", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run preload-v2 CLI."""
    args = build_parser().parse_args(argv)
    run_sweep(
        scene_path=args.scene,
        out_csv=args.out_csv,
        settle_steps=args.settle_steps,
        top_k=args.top_k,
        expand_base_top_k=args.expand_base_top_k,
        stance_knee_pitch_deltas=args.stance_knee_pitch_deltas,
        stance_ankle_pitch_deltas=args.stance_ankle_pitch_deltas,
        swing_hip_roll_deltas=args.swing_hip_roll_deltas,
        exhaustive=args.exhaustive,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
