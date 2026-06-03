"""Run a focused closed-loop Seedon force-unload controller proof.

This is a diagnostic, not a walking policy or PPO reward. It keeps both feet
near the floor and only asks whether joint-space feedback can bias enough
normal force onto one support foot while staying upright.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np

from seedon_baseline.env import SeedonStandingConfig, SeedonStandingEnv
from tools.seedon_debug_common import (
    BASE_PROXY_GEOM,
    DEBUG_OUT_DIR,
    FLOOR_GEOM,
    LEFT_FOOT_GEOM,
    RIGHT_FOOT_GEOM,
    contact_pairs,
)


DEFAULT_OUT_CSV = DEBUG_OUT_DIR / "force_unload_controller.csv"
SUPPORT_TO_SWING = {"left": "right", "right": "left"}
FOOT_GEOM_BY_SIDE = {"left": LEFT_FOOT_GEOM, "right": RIGHT_FOOT_GEOM}
HIP_ROLL_INDEX = {"right": 1, "left": 6}
HIP_PITCH_INDEX = {"right": 2, "left": 7}
KNEE_PITCH_INDEX = {"right": 3, "left": 8}
ANKLE_PITCH_INDEX = {"right": 4, "left": 9}


@dataclass(frozen=True)
class ControllerCase:
    """One force-unload controller candidate."""

    support_side: str
    target_support_fraction: float
    force_kp: float
    force_kd: float
    max_support_roll: float
    max_roll_delta: float
    support_knee_brace: float
    support_ankle_brace: float
    swing_hip_unload: float
    swing_knee_unload: float
    swing_ankle_unload: float

    @property
    def case_name(self) -> str:
        """Return a compact stable case label."""
        return (
            f"{self.support_side}_target_{self.target_support_fraction:g}"
            f"_kp_{self.force_kp:g}_roll_{self.max_support_roll:g}"
        )


@dataclass(frozen=True)
class ControllerSample:
    """One rollout sample from the controller."""

    case_name: str
    step: int
    support_side: str
    swing_side: str
    target_support_fraction: float
    support_fraction: float
    swing_fraction: float
    support_force: float
    swing_force: float
    support_contact_count: int
    swing_contact_count: int
    support_roll_cmd: float
    force_error: float
    com_y: float
    support_foot_y: float
    swing_foot_y: float
    support_margin_y: float
    base_z: float
    upright: float
    contact_state: str
    base_proxy_floor_contact: bool
    terminated: bool


@dataclass(frozen=True)
class ControllerSummary:
    """Aggregate diagnostic result for one controller candidate."""

    case_name: str
    support_side: str
    target_support_fraction: float
    steps: int
    terminated: bool
    terminated_step: int | None
    max_support_fraction: float
    mean_support_fraction_last_50: float
    min_swing_fraction_last_50: float
    stable_gate_steps: int
    longest_stable_gate_streak: int
    max_support_roll_cmd: float
    min_support_margin_y: float
    base_proxy_floor_steps: int
    none_contact_steps: int
    diagnosis: str


def _parse_float_list(raw_value: str) -> list[float]:
    """Parse a comma-separated float list."""
    values = [float(part.strip()) for part in raw_value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one float.")
    return values


def _overall_com(env: SeedonStandingEnv) -> np.ndarray:
    """Return whole-body COM in world coordinates."""
    masses = env.model.body_mass
    total_mass = float(np.sum(masses))
    if total_mass <= 0.0:
        raise ValueError("Seedon model has no positive body mass.")
    return np.sum(env.data.xipos * masses[:, None], axis=0) / total_mass


def _foot_floor_load(env: SeedonStandingEnv, side: str) -> tuple[int, float]:
    """Return floor-contact count and summed normal force for one foot."""
    foot_geom_name = FOOT_GEOM_BY_SIDE[side]
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


def _contact_flags(env: SeedonStandingEnv) -> tuple[bool, bool, bool]:
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
    """Return a compact contact state label."""
    if left_contact and right_contact:
        return "both"
    if left_contact:
        return "left_only"
    if right_contact:
        return "right_only"
    return "none"


def _support_roll_offsets(support_side: str, magnitude: float) -> tuple[float, float]:
    """Return right/left hip-roll offsets for one support side."""
    if support_side == "left":
        return magnitude, -magnitude
    if support_side == "right":
        return -magnitude, magnitude
    raise ValueError(f"Unsupported support_side: {support_side}")


def _build_target(env: SeedonStandingEnv, case: ControllerCase, support_roll_cmd: float) -> np.ndarray:
    """Build the joint target for one closed-loop controller step."""
    target = env._nominal_joint_qpos.copy()
    swing_side = SUPPORT_TO_SWING[case.support_side]
    right_roll, left_roll = _support_roll_offsets(case.support_side, support_roll_cmd)
    target[HIP_ROLL_INDEX["right"]] += right_roll
    target[HIP_ROLL_INDEX["left"]] += left_roll
    target[KNEE_PITCH_INDEX[case.support_side]] += case.support_knee_brace
    target[ANKLE_PITCH_INDEX[case.support_side]] += case.support_ankle_brace
    target[HIP_PITCH_INDEX[swing_side]] += case.swing_hip_unload
    target[KNEE_PITCH_INDEX[swing_side]] += case.swing_knee_unload
    target[ANKLE_PITCH_INDEX[swing_side]] += case.swing_ankle_unload
    return env._apply_safe_joint_target_clamps(target)


def _rate_limit(value: float, previous: float, max_delta: float) -> float:
    """Clamp per-step command changes."""
    delta = float(np.clip(value - previous, -max_delta, max_delta))
    return previous + delta


def _sample(
    env: SeedonStandingEnv,
    *,
    case: ControllerCase,
    step: int,
    support_roll_cmd: float,
    force_error: float,
) -> ControllerSample:
    """Collect one sample after a controller step."""
    swing_side = SUPPORT_TO_SWING[case.support_side]
    support_count, support_force = _foot_floor_load(env, case.support_side)
    swing_count, swing_force = _foot_floor_load(env, swing_side)
    total_force = support_force + swing_force
    support_fraction = support_force / max(total_force, 1e-9)
    swing_fraction = swing_force / max(total_force, 1e-9)
    left_contact, right_contact, base_contact = _contact_flags(env)

    support_geom_id = env._geom_id(FOOT_GEOM_BY_SIDE[case.support_side])
    swing_geom_id = env._geom_id(FOOT_GEOM_BY_SIDE[swing_side])
    com_y = float(_overall_com(env)[1])
    support_foot_y = float(env.data.geom_xpos[support_geom_id][1])
    swing_foot_y = float(env.data.geom_xpos[swing_geom_id][1])
    base_z = env._base_height()
    upright = env._base_upright()
    terminated = env._is_terminated(base_z, upright, env._get_obs())
    return ControllerSample(
        case_name=case.case_name,
        step=step,
        support_side=case.support_side,
        swing_side=swing_side,
        target_support_fraction=case.target_support_fraction,
        support_fraction=support_fraction,
        swing_fraction=swing_fraction,
        support_force=support_force,
        swing_force=swing_force,
        support_contact_count=support_count,
        swing_contact_count=swing_count,
        support_roll_cmd=support_roll_cmd,
        force_error=force_error,
        com_y=com_y,
        support_foot_y=support_foot_y,
        swing_foot_y=swing_foot_y,
        support_margin_y=abs(com_y - support_foot_y),
        base_z=float(base_z),
        upright=float(upright),
        contact_state=_contact_state(left_contact, right_contact),
        base_proxy_floor_contact=base_contact,
        terminated=terminated,
    )


def _longest_true_streak(values: list[bool]) -> int:
    """Return the longest consecutive true streak."""
    longest = 0
    current = 0
    for value in values:
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _diagnosis(
    *,
    terminated: bool,
    longest_stable_gate_streak: int,
    required_stable_steps: int,
    max_support_fraction: float,
    target_support_fraction: float,
) -> str:
    """Return a compact case diagnosis."""
    if longest_stable_gate_streak >= required_stable_steps and not terminated:
        return "scripted_force_unload_success"
    if longest_stable_gate_streak >= required_stable_steps:
        return "force_unload_possible_but_unstable"
    if max_support_fraction >= target_support_fraction:
        return "force_ratio_spikes_but_not_stable"
    return "controller_cannot_unload"


def _run_case(
    case: ControllerCase,
    *,
    steps: int,
    warmup_steps: int,
    stable_fraction_gate: float,
    swing_fraction_gate: float,
    required_stable_steps: int,
    seed: int,
) -> tuple[ControllerSummary, list[ControllerSample]]:
    """Run one closed-loop force-unload candidate."""
    env = SeedonStandingEnv(
        reset_noise_scale=0.0,
        reward_config=SeedonStandingConfig(gait_mode="fsm"),
    )
    samples: list[ControllerSample] = []
    try:
        env.reset(seed=seed)
        support_roll_cmd = 0.0
        previous_error = 0.0
        terminated = False
        terminated_step: int | None = None
        for step in range(1, steps + 1):
            if step <= warmup_steps:
                force_error = 0.0
                desired_roll = 0.0
            else:
                if samples:
                    current_fraction = samples[-1].support_fraction
                else:
                    current_fraction = 0.5
                force_error = case.target_support_fraction - current_fraction
                force_derivative = force_error - previous_error
                desired_roll = support_roll_cmd + case.force_kp * force_error + case.force_kd * force_derivative
                desired_roll = float(np.clip(desired_roll, 0.0, case.max_support_roll))
            support_roll_cmd = _rate_limit(desired_roll, support_roll_cmd, case.max_roll_delta)
            previous_error = force_error
            env._do_pd_simulation(_build_target(env, case, support_roll_cmd))
            sample = _sample(
                env,
                case=case,
                step=step,
                support_roll_cmd=support_roll_cmd,
                force_error=force_error,
            )
            samples.append(sample)
            if sample.terminated:
                terminated = True
                terminated_step = step
                break
    finally:
        env.close()

    if not samples:
        raise RuntimeError(f"No samples collected for case {case.case_name}.")

    tail = samples[-50:]
    stable_flags = [
        sample.support_fraction >= stable_fraction_gate
        and sample.swing_fraction <= swing_fraction_gate
        and sample.contact_state == "both"
        and not sample.base_proxy_floor_contact
        and not sample.terminated
        for sample in samples
    ]
    longest_streak = _longest_true_streak(stable_flags)
    summary = ControllerSummary(
        case_name=case.case_name,
        support_side=case.support_side,
        target_support_fraction=case.target_support_fraction,
        steps=len(samples),
        terminated=terminated,
        terminated_step=terminated_step,
        max_support_fraction=max(sample.support_fraction for sample in samples),
        mean_support_fraction_last_50=float(np.mean([sample.support_fraction for sample in tail])),
        min_swing_fraction_last_50=min(sample.swing_fraction for sample in tail),
        stable_gate_steps=sum(stable_flags),
        longest_stable_gate_streak=longest_streak,
        max_support_roll_cmd=max(abs(sample.support_roll_cmd) for sample in samples),
        min_support_margin_y=min(sample.support_margin_y for sample in samples),
        base_proxy_floor_steps=sum(1 for sample in samples if sample.base_proxy_floor_contact),
        none_contact_steps=sum(1 for sample in samples if sample.contact_state == "none"),
        diagnosis=_diagnosis(
            terminated=terminated,
            longest_stable_gate_streak=longest_streak,
            required_stable_steps=required_stable_steps,
            max_support_fraction=max(sample.support_fraction for sample in samples),
            target_support_fraction=case.target_support_fraction,
        ),
    )
    return summary, samples


def _write_csv(path: Path, rows: list[ControllerSample]) -> None:
    """Write all per-step samples to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows([asdict(row) for row in rows])


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support-sides", default="left,right")
    parser.add_argument("--target-support-fraction", type=float, default=0.65)
    parser.add_argument("--force-kps", type=_parse_float_list, default=[0.04, 0.08, 0.12])
    parser.add_argument("--force-kd", type=float, default=0.0)
    parser.add_argument("--max-support-rolls", type=_parse_float_list, default=[0.04, 0.08, 0.12])
    parser.add_argument("--max-roll-delta", type=float, default=0.004)
    parser.add_argument("--support-knee-brace", type=float, default=0.0)
    parser.add_argument("--support-ankle-brace", type=float, default=0.0)
    parser.add_argument("--swing-hip-unload", type=float, default=0.0)
    parser.add_argument("--swing-knee-unload", type=float, default=0.0)
    parser.add_argument("--swing-ankle-unload", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=220)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--stable-fraction-gate", type=float, default=0.65)
    parser.add_argument("--swing-fraction-gate", type=float, default=0.35)
    parser.add_argument("--required-stable-steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    return parser


def _iter_cases(args: argparse.Namespace) -> list[ControllerCase]:
    """Expand CLI grids into concrete controller cases."""
    sides = [side.strip() for side in args.support_sides.split(",") if side.strip()]
    invalid = sorted(set(sides) - {"left", "right"})
    if invalid:
        raise ValueError(f"Unsupported support side(s): {', '.join(invalid)}")
    return [
        ControllerCase(
            support_side=support_side,
            target_support_fraction=args.target_support_fraction,
            force_kp=force_kp,
            force_kd=args.force_kd,
            max_support_roll=max_support_roll,
            max_roll_delta=args.max_roll_delta,
            support_knee_brace=args.support_knee_brace,
            support_ankle_brace=args.support_ankle_brace,
            swing_hip_unload=args.swing_hip_unload,
            swing_knee_unload=args.swing_knee_unload,
            swing_ankle_unload=args.swing_ankle_unload,
        )
        for support_side in sides
        for force_kp in args.force_kps
        for max_support_roll in args.max_support_rolls
    ]


def main(argv: list[str] | None = None) -> int:
    """Run the closed-loop force-unload proof."""
    args = build_parser().parse_args(argv)
    if args.steps <= 0:
        raise ValueError("--steps must be positive.")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be non-negative.")
    if args.required_stable_steps <= 0:
        raise ValueError("--required-stable-steps must be positive.")
    if not 0.0 < args.target_support_fraction < 1.0:
        raise ValueError("--target-support-fraction must be between 0 and 1.")

    all_samples: list[ControllerSample] = []
    summaries: list[ControllerSummary] = []
    for case in _iter_cases(args):
        summary, samples = _run_case(
            case,
            steps=args.steps,
            warmup_steps=args.warmup_steps,
            stable_fraction_gate=args.stable_fraction_gate,
            swing_fraction_gate=args.swing_fraction_gate,
            required_stable_steps=args.required_stable_steps,
            seed=args.seed,
        )
        summaries.append(summary)
        all_samples.extend(samples)

    _write_csv(args.out_csv, all_samples)
    summaries.sort(
        key=lambda row: (
            row.longest_stable_gate_streak,
            row.mean_support_fraction_last_50,
            -float(row.terminated),
        ),
        reverse=True,
    )

    print(
        "rank side target max_frac mean_last50 min_swing_last50 "
        "stable_steps longest roll max_margin_ok none base terminated diagnosis"
    )
    for rank, row in enumerate(summaries[: args.top_k], start=1):
        print(
            f"{rank:>4} {row.support_side:>5} "
            f"{row.target_support_fraction:>6.2f} "
            f"{row.max_support_fraction:>8.3f} "
            f"{row.mean_support_fraction_last_50:>11.3f} "
            f"{row.min_swing_fraction_last_50:>16.3f} "
            f"{row.stable_gate_steps:>12} "
            f"{row.longest_stable_gate_streak:>7} "
            f"{row.max_support_roll_cmd:>5.3f} "
            f"{row.min_support_margin_y:>13.4f} "
            f"{row.none_contact_steps:>4} "
            f"{row.base_proxy_floor_steps:>4} "
            f"{str(row.terminated):>10} "
            f"{row.diagnosis}"
        )

    print(f"\nCSV: {args.out_csv}")
    if summaries and summaries[0].diagnosis == "scripted_force_unload_success":
        print("interpretation: scripted force unload is controllable; it is reasonable to convert this into curriculum/reward.")
    elif summaries and summaries[0].max_support_fraction >= args.stable_fraction_gate:
        print("interpretation: force ratio can spike, but not stably; tune controller/model before PPO reward.")
    else:
        print("interpretation: this controller cannot unload Seedon; inspect authority/contact/mass before training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
