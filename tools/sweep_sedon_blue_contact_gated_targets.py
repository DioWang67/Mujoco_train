"""Sweep Blue-like Sedon shift targets and rank candidates for contact-gated support.

This tool does not modify training XML, reward terms, or PPO. It only searches
preview-side deterministic shift/unload poses that might produce a stable
support-side force bias before the swing-lift gate opens.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np

from sedon_baseline.env import SedonStandingEnv, load_sedon_config_from_env
from tools.sedon_debug_common import (
    BASE_PROXY_GEOM,
    DEBUG_OUT_DIR,
    FLOOR_GEOM,
    LEFT_FOOT_GEOM,
    RIGHT_FOOT_GEOM,
    contact_pairs,
)


DEFAULT_OUT_CSV = DEBUG_OUT_DIR / "blue_contact_gated_target_sweep.csv"
HIP_ROLL_INDEX = {"right": 1, "left": 6}
HIP_PITCH_INDEX = {"right": 2, "left": 7}
KNEE_PITCH_INDEX = {"right": 3, "left": 8}
ANKLE_PITCH_INDEX = {"right": 4, "left": 9}
SUPPORT_TO_SWING = {"left": "right", "right": "left"}


@dataclass(frozen=True)
class ShiftCandidate:
    """One shift-target candidate to evaluate."""

    support_side: str
    shift_roll: float
    support_knee_delta: float
    support_ankle_delta: float
    unload_hip_pitch_delta: float
    unload_knee_pitch_delta: float
    unload_ankle_pitch_delta: float


@dataclass(frozen=True)
class ShiftSweepResult:
    """One evaluated sweep result row."""

    support_side: str
    swing_side: str
    shift_roll: float
    support_knee_delta: float
    support_ankle_delta: float
    unload_hip_pitch_delta: float
    unload_knee_pitch_delta: float
    unload_ankle_pitch_delta: float
    steps: int
    terminated: bool
    terminated_step: int
    final_contact_state: str
    both_contact_steps: int
    support_only_steps: int
    none_steps: int
    base_proxy_steps: int
    gate_passed: bool
    gate_first_step: int
    max_support_force_ratio: float
    max_stable_support_force_ratio: float
    final_support_force_ratio: float
    max_support_com_shift: float
    final_support_com_shift: float
    max_abs_base_roll: float
    min_base_z: float
    min_upright: float
    score: float
    preview_flag_fragment: str


def _parse_float_list(raw_value: str) -> list[float]:
    """Parse a comma-separated float list."""
    values = [float(part.strip()) for part in raw_value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one float value.")
    return values


def _overall_com(env: SedonStandingEnv) -> np.ndarray:
    """Return whole-body COM in world coordinates."""
    masses = env.model.body_mass
    total_mass = float(np.sum(masses))
    if total_mass <= 0.0:
        raise ValueError("Model has no positive body mass.")
    return np.sum(env.data.xipos * masses[:, None], axis=0) / total_mass


def _quat_to_roll(quat: np.ndarray) -> float:
    """Return roll angle from a MuJoCo quaternion."""
    w, x, y, z = [float(value) for value in quat]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    return float(math.atan2(sinr_cosp, cosr_cosp))


def _foot_floor_load(env: SedonStandingEnv, side: str) -> tuple[int, float]:
    """Return floor-contact count and normal-force sum for one foot."""
    foot_geom_name = LEFT_FOOT_GEOM if side == "left" else RIGHT_FOOT_GEOM
    contact_count = 0
    normal_force_sum = 0.0
    wrench = np.zeros(6, dtype=np.float64)
    for contact_index in range(env.data.ncon):
        contact = env.data.contact[contact_index]
        name_a = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1))
        name_b = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2))
        if {name_a, name_b} != {FLOOR_GEOM, foot_geom_name}:
            continue
        mujoco.mj_contactForce(env.model, env.data, contact_index, wrench)
        contact_count += 1
        normal_force_sum += abs(float(wrench[0]))
    return contact_count, normal_force_sum


def _contact_flags(env: SedonStandingEnv) -> tuple[bool, bool, bool]:
    """Return left, right, and base-proxy floor-contact flags."""
    left = False
    right = False
    base = False
    for name_a, name_b, _ in contact_pairs(env.model, env.data):
        pair = {name_a, name_b}
        if pair == {FLOOR_GEOM, LEFT_FOOT_GEOM}:
            left = True
        elif pair == {FLOOR_GEOM, RIGHT_FOOT_GEOM}:
            right = True
        elif pair == {FLOOR_GEOM, BASE_PROXY_GEOM}:
            base = True
    return left, right, base


def _contact_state(left_contact: bool, right_contact: bool) -> str:
    """Return a compact floor-contact state label."""
    if left_contact and right_contact:
        return "both"
    if left_contact:
        return "left_only"
    if right_contact:
        return "right_only"
    return "none"


def _smoothstep(value: float) -> float:
    """Return a clipped smooth interpolation."""
    clipped = float(np.clip(value, 0.0, 1.0))
    return clipped * clipped * (3.0 - 2.0 * clipped)


def _apply_support_roll(target: np.ndarray, support_side: str, magnitude: float) -> None:
    """Apply mirrored hip-roll offsets to bias support load."""
    if support_side == "left":
        target[HIP_ROLL_INDEX["right"]] += magnitude
        target[HIP_ROLL_INDEX["left"]] -= magnitude
        return
    if support_side == "right":
        target[HIP_ROLL_INDEX["right"]] -= magnitude
        target[HIP_ROLL_INDEX["left"]] += magnitude
        return
    raise ValueError(f"Unsupported support side: {support_side}")


def _build_shift_target(
    env: SedonStandingEnv,
    candidate: ShiftCandidate,
    *,
    alpha: float,
) -> np.ndarray:
    """Return one both-feet-down shift target candidate."""
    support_side = candidate.support_side
    swing_side = SUPPORT_TO_SWING[support_side]
    target = env._nominal_joint_qpos.copy()
    _apply_support_roll(target, support_side, candidate.shift_roll * alpha)
    target[KNEE_PITCH_INDEX[support_side]] += candidate.support_knee_delta * alpha
    target[ANKLE_PITCH_INDEX[support_side]] += candidate.support_ankle_delta * alpha
    target[HIP_PITCH_INDEX[swing_side]] += candidate.unload_hip_pitch_delta * alpha
    target[KNEE_PITCH_INDEX[swing_side]] += candidate.unload_knee_pitch_delta * alpha
    target[ANKLE_PITCH_INDEX[swing_side]] += candidate.unload_ankle_pitch_delta * alpha
    return env._apply_safe_joint_target_clamps(target)


def _preview_flag_fragment(candidate: ShiftCandidate) -> str:
    """Return preview CLI flags for replaying one candidate."""
    support_side = candidate.support_side
    swing_side = SUPPORT_TO_SWING[support_side]
    parts = [f"--shift-roll {candidate.shift_roll:.4f}"]
    parts.append(f"--shift-support-{support_side}-knee-delta {candidate.support_knee_delta:.4f}")
    parts.append(f"--shift-support-{support_side}-ankle-delta {candidate.support_ankle_delta:.4f}")
    parts.append(
        f"--shift-unload-{swing_side}-hip-pitch-delta {candidate.unload_hip_pitch_delta:.4f}"
    )
    parts.append(
        f"--shift-unload-{swing_side}-knee-pitch-delta {candidate.unload_knee_pitch_delta:.4f}"
    )
    parts.append(
        f"--shift-unload-{swing_side}-ankle-pitch-delta {candidate.unload_ankle_pitch_delta:.4f}"
    )
    return " ".join(parts)


def _candidate_score(
    *,
    gate_passed: bool,
    max_stable_support_force_ratio: float,
    max_support_com_shift: float,
    both_contact_steps: int,
    support_only_steps: int,
    none_steps: int,
    base_proxy_steps: int,
    terminated: bool,
) -> float:
    """Return a simple ranking score for target search."""
    return (
        (1000.0 if gate_passed else 0.0)
        + max_stable_support_force_ratio * 100.0
        + max_support_com_shift * 1000.0
        + both_contact_steps * 0.25
        - support_only_steps * 2.0
        - none_steps * 4.0
        - base_proxy_steps * 6.0
        - (50.0 if terminated else 0.0)
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support-sides", default="left")
    parser.add_argument("--shift-rolls", default="0.10,0.12")
    parser.add_argument("--support-knee-deltas", default="0.0,0.02")
    parser.add_argument("--support-ankle-deltas", default="0.0,-0.01")
    parser.add_argument("--unload-hip-pitch-deltas", default="0.0,-0.01")
    parser.add_argument("--unload-knee-pitch-deltas", default="0.0,-0.03")
    parser.add_argument("--unload-ankle-pitch-deltas", default="0.0,0.01")
    parser.add_argument("--settle-steps", type=int, default=20)
    parser.add_argument("--ramp-steps", type=int, default=40)
    parser.add_argument("--hold-steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gate-force-ratio", type=float, default=0.65)
    parser.add_argument("--gate-com-shift", type=float, default=0.008)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    return parser


def _iter_candidates(args: argparse.Namespace) -> list[ShiftCandidate]:
    """Expand the CLI grids into concrete candidates."""
    support_sides = [part.strip() for part in args.support_sides.split(",") if part.strip()]
    if not support_sides:
        raise ValueError("--support-sides must contain at least one side.")
    for side in support_sides:
        if side not in {"left", "right"}:
            raise ValueError(f"Unsupported support side: {side}")

    shift_rolls = _parse_float_list(args.shift_rolls)
    support_knee_deltas = _parse_float_list(args.support_knee_deltas)
    support_ankle_deltas = _parse_float_list(args.support_ankle_deltas)
    unload_hip_pitch_deltas = _parse_float_list(args.unload_hip_pitch_deltas)
    unload_knee_pitch_deltas = _parse_float_list(args.unload_knee_pitch_deltas)
    unload_ankle_pitch_deltas = _parse_float_list(args.unload_ankle_pitch_deltas)

    return [
        ShiftCandidate(
            support_side=support_side,
            shift_roll=shift_roll,
            support_knee_delta=support_knee_delta,
            support_ankle_delta=support_ankle_delta,
            unload_hip_pitch_delta=unload_hip_pitch_delta,
            unload_knee_pitch_delta=unload_knee_pitch_delta,
            unload_ankle_pitch_delta=unload_ankle_pitch_delta,
        )
        for (
            support_side,
            shift_roll,
            support_knee_delta,
            support_ankle_delta,
            unload_hip_pitch_delta,
            unload_knee_pitch_delta,
            unload_ankle_pitch_delta,
        ) in itertools.product(
            support_sides,
            shift_rolls,
            support_knee_deltas,
            support_ankle_deltas,
            unload_hip_pitch_deltas,
            unload_knee_pitch_deltas,
            unload_ankle_pitch_deltas,
        )
    ]


def _run_candidate(
    candidate: ShiftCandidate,
    *,
    settle_steps: int,
    ramp_steps: int,
    hold_steps: int,
    seed: int,
    gate_force_ratio: float,
    gate_com_shift: float,
) -> ShiftSweepResult:
    """Run one shift candidate and return gate-oriented metrics."""
    reward_config = load_sedon_config_from_env()
    env = SedonStandingEnv(reset_noise_scale=0.0, reward_config=reward_config)
    try:
        env.reset(seed=seed)
        if settle_steps > 0:
            nominal_target = env._apply_safe_joint_target_clamps(env._nominal_joint_qpos.copy())
            for _ in range(settle_steps):
                env._do_pd_simulation(nominal_target)

        support_side = candidate.support_side
        swing_side = SUPPORT_TO_SWING[support_side]
        initial_com_y = float(_overall_com(env)[1])
        left_foot_geom_id = env._geom_id(LEFT_FOOT_GEOM)
        right_foot_geom_id = env._geom_id(RIGHT_FOOT_GEOM)
        terminated = False
        terminated_step = 0
        final_contact_state = "none"
        both_contact_steps = 0
        support_only_steps = 0
        none_steps = 0
        base_proxy_steps = 0
        gate_passed = False
        gate_first_step = 0
        max_support_force_ratio = 0.0
        max_stable_support_force_ratio = 0.0
        final_support_force_ratio = 0.0
        max_support_com_shift = 0.0
        final_support_com_shift = 0.0
        max_abs_base_roll = 0.0
        min_base_z = float("inf")
        min_upright = float("inf")

        total_steps = ramp_steps + hold_steps
        for step in range(1, total_steps + 1):
            alpha = 1.0 if step > ramp_steps else _smoothstep(step / max(ramp_steps, 1))
            target = _build_shift_target(env, candidate, alpha=alpha)
            env._do_pd_simulation(target)

            left_count, left_force = _foot_floor_load(env, "left")
            right_count, right_force = _foot_floor_load(env, "right")
            left_contact, right_contact, base_proxy_contact = _contact_flags(env)
            total_force = left_force + right_force
            if total_force > 1e-9:
                left_force_ratio = left_force / total_force
                right_force_ratio = right_force / total_force
            else:
                left_force_ratio = 0.0
                right_force_ratio = 0.0
            support_force_ratio = left_force_ratio if support_side == "left" else right_force_ratio
            final_support_force_ratio = support_force_ratio
            max_support_force_ratio = max(max_support_force_ratio, support_force_ratio)

            com_y = float(_overall_com(env)[1])
            left_foot_y = float(env.data.geom_xpos[left_foot_geom_id][1])
            right_foot_y = float(env.data.geom_xpos[right_foot_geom_id][1])
            center_y = 0.5 * (left_foot_y + right_foot_y)
            signed_support_com_shift = com_y - center_y if support_side == "left" else center_y - com_y
            max_support_com_shift = max(max_support_com_shift, signed_support_com_shift)
            final_support_com_shift = signed_support_com_shift

            contact_state = _contact_state(left_contact, right_contact)
            final_contact_state = contact_state
            if contact_state == "both":
                both_contact_steps += 1
                max_stable_support_force_ratio = max(
                    max_stable_support_force_ratio,
                    support_force_ratio,
                )
            elif contact_state == f"{support_side}_only":
                support_only_steps += 1
            elif contact_state == "none":
                none_steps += 1
            base_proxy_steps += int(base_proxy_contact)

            if (
                not gate_passed
                and left_contact
                and right_contact
                and support_force_ratio >= gate_force_ratio
                and signed_support_com_shift >= gate_com_shift
            ):
                gate_passed = True
                gate_first_step = step

            base_roll = _quat_to_roll(env.data.xquat[env._base_body_id])
            base_z = env._base_height()
            upright = env._base_upright()
            max_abs_base_roll = max(max_abs_base_roll, abs(base_roll))
            min_base_z = min(min_base_z, float(base_z))
            min_upright = min(min_upright, float(upright))

            observation = env._get_obs()
            terminated = env._is_terminated(base_z, upright, observation)
            if terminated:
                terminated_step = step
                break

        if not terminated:
            terminated_step = total_steps

        score = _candidate_score(
            gate_passed=gate_passed,
            max_stable_support_force_ratio=max_stable_support_force_ratio,
            max_support_com_shift=max_support_com_shift,
            both_contact_steps=both_contact_steps,
            support_only_steps=support_only_steps,
            none_steps=none_steps,
            base_proxy_steps=base_proxy_steps,
            terminated=terminated,
        )

        return ShiftSweepResult(
            support_side=support_side,
            swing_side=swing_side,
            shift_roll=candidate.shift_roll,
            support_knee_delta=candidate.support_knee_delta,
            support_ankle_delta=candidate.support_ankle_delta,
            unload_hip_pitch_delta=candidate.unload_hip_pitch_delta,
            unload_knee_pitch_delta=candidate.unload_knee_pitch_delta,
            unload_ankle_pitch_delta=candidate.unload_ankle_pitch_delta,
            steps=terminated_step,
            terminated=terminated,
            terminated_step=terminated_step,
            final_contact_state=final_contact_state,
            both_contact_steps=both_contact_steps,
            support_only_steps=support_only_steps,
            none_steps=none_steps,
            base_proxy_steps=base_proxy_steps,
            gate_passed=gate_passed,
            gate_first_step=gate_first_step,
            max_support_force_ratio=max_support_force_ratio,
            max_stable_support_force_ratio=max_stable_support_force_ratio,
            final_support_force_ratio=final_support_force_ratio,
            max_support_com_shift=max_support_com_shift,
            final_support_com_shift=final_support_com_shift,
            max_abs_base_roll=max_abs_base_roll,
            min_base_z=min_base_z,
            min_upright=min_upright,
            score=score,
            preview_flag_fragment=_preview_flag_fragment(candidate),
        )
    finally:
        env.close()


def _write_csv(path: Path, rows: list[ShiftSweepResult]) -> None:
    """Write sweep rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows([asdict(row) for row in rows])


def main(argv: list[str] | None = None) -> int:
    """Run the Blue-like shift-target sweep."""
    args = build_parser().parse_args(argv)
    if args.settle_steps < 0:
        raise ValueError("--settle-steps must be non-negative.")
    if args.ramp_steps <= 0:
        raise ValueError("--ramp-steps must be positive.")
    if args.hold_steps <= 0:
        raise ValueError("--hold-steps must be positive.")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive.")

    candidates = _iter_candidates(args)
    results = [
        _run_candidate(
            candidate,
            settle_steps=args.settle_steps,
            ramp_steps=args.ramp_steps,
            hold_steps=args.hold_steps,
            seed=args.seed,
            gate_force_ratio=args.gate_force_ratio,
            gate_com_shift=args.gate_com_shift,
        )
        for candidate in candidates
    ]
    _write_csv(args.out_csv, results)

    ranked = sorted(
        results,
        key=lambda row: (
            row.gate_passed,
            row.score,
            row.max_stable_support_force_ratio,
            row.max_support_com_shift,
        ),
        reverse=True,
    )

    print(f"candidates: {len(results)}")
    print(
        "side gate step stable_ratio max_ratio max_com_shift "
        "both support_only none term score"
    )
    for row in ranked[: args.top_k]:
        print(
            f"{row.support_side:>5} {str(row.gate_passed):>5} {row.gate_first_step:>4} "
            f"{row.max_stable_support_force_ratio:>12.3f} {row.max_support_force_ratio:>9.3f} "
            f"{row.max_support_com_shift:>13.4f} {row.both_contact_steps:>4} "
            f"{row.support_only_steps:>12} {row.none_steps:>4} "
            f"{row.terminated_step:>4} {row.score:>7.2f}"
        )
        print(f"  flags: {row.preview_flag_fragment}")

    print(f"\ncsv: {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
