"""Sweep a simple COM-feedback balance controller for Seedon lateral authority.

This diagnostic does not touch PPO, reward terms, or the committed training
scene. It only replaces the fixed open-loop shift target with a closed-loop
hip-roll controller that tries to move COM laterally while both feet remain on
the floor.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

import mujoco
import numpy as np

from seedon_baseline.env import SeedonStandingEnv, load_seedon_config_from_env
from tools.seedon_debug_common import (
    BASE_PROXY_GEOM,
    DEBUG_OUT_DIR,
    FLOOR_GEOM,
    LEFT_FOOT_GEOM,
    RIGHT_FOOT_GEOM,
    contact_pairs,
)


DEFAULT_OUT_CSV = DEBUG_OUT_DIR / "com_feedback_balance_sweep.csv"
HIP_ROLL_INDEX = {"right": 1, "left": 6}
HIP_PITCH_INDEX = {"right": 2, "left": 7}
KNEE_PITCH_INDEX = {"right": 3, "left": 8}
ANKLE_PITCH_INDEX = {"right": 4, "left": 9}
SUPPORT_TO_SWING = {"left": "right", "right": "left"}


@dataclass(frozen=True)
class FeedbackCandidate:
    """One feedback-controller candidate."""

    support_side: str
    target_support_com_shift: float
    com_kp: float
    com_kd: float
    roll_kp: float
    max_support_roll: float

    @property
    def case_name(self) -> str:
        """Return a stable case label."""
        return (
            f"{self.support_side}"
            f"__target_{self.target_support_com_shift:.3f}"
            f"__kp_{self.com_kp:.2f}"
            f"__kd_{self.com_kd:.2f}"
            f"__roll_{self.roll_kp:.2f}"
            f"__cap_{self.max_support_roll:.3f}"
        )


@dataclass(frozen=True)
class FeedbackSweepResult:
    """One feedback-balance sweep row."""

    case_name: str
    support_side: str
    target_support_com_shift: float
    com_kp: float
    com_kd: float
    roll_kp: float
    max_support_roll: float
    max_abs_com_y_delta: float
    max_support_com_shift: float
    mean_support_com_shift_last_50: float
    max_support_force_ratio: float
    mean_support_force_ratio_last_50: float
    support_only_steps: int
    both_contact_ratio: float
    none_contact_ratio: float
    base_proxy_ratio: float
    terminated: bool
    terminated_step: int
    score: float


def _parse_float_list(raw_value: str) -> list[float]:
    """Parse comma-separated floats."""
    values = [float(part.strip()) for part in raw_value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one float value.")
    return values


def _overall_com(env: SeedonStandingEnv) -> np.ndarray:
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


def _foot_floor_load(env: SeedonStandingEnv, side: str) -> tuple[int, float]:
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


def _contact_state(env: SeedonStandingEnv) -> tuple[str, bool]:
    """Return compact foot-contact state plus base-proxy flag."""
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
    if left and right:
        return "both", base
    if left:
        return "left_only", base
    if right:
        return "right_only", base
    return "none", base


def _apply_support_roll(target: np.ndarray, support_side: str, magnitude: float) -> None:
    """Apply mirrored hip-roll offsets that bias support onto one foot."""
    if support_side == "left":
        target[HIP_ROLL_INDEX["right"]] += magnitude
        target[HIP_ROLL_INDEX["left"]] -= magnitude
        return
    if support_side == "right":
        target[HIP_ROLL_INDEX["right"]] -= magnitude
        target[HIP_ROLL_INDEX["left"]] += magnitude
        return
    raise ValueError(f"Unsupported support side: {support_side}")


def _brace_target(env: SeedonStandingEnv, support_side: str) -> np.ndarray:
    """Return a conservative pre-lift brace pose found from prior target sweeps."""
    swing_side = SUPPORT_TO_SWING[support_side]
    target = env._nominal_joint_qpos.copy()
    target[KNEE_PITCH_INDEX[support_side]] += 0.04
    target[KNEE_PITCH_INDEX[swing_side]] += -0.06
    return target


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support-sides", default="left,right")
    parser.add_argument("--target-support-com-shifts", default="0.005,0.010,0.015")
    parser.add_argument("--com-kp-values", default="4,8,12,16")
    parser.add_argument("--com-kd-values", default="0.0,0.2,0.4")
    parser.add_argument("--roll-kp-values", default="0.0,0.5,1.0")
    parser.add_argument("--max-support-roll-values", default="0.08,0.12,0.16")
    parser.add_argument("--settle-steps", type=int, default=20)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    return parser


def _iter_candidates(args: argparse.Namespace) -> list[FeedbackCandidate]:
    """Expand CLI grids into concrete candidates."""
    support_sides = [part.strip() for part in args.support_sides.split(",") if part.strip()]
    if not support_sides:
        raise ValueError("--support-sides must contain at least one side.")
    for side in support_sides:
        if side not in {"left", "right"}:
            raise ValueError(f"Unsupported support side: {side}")

    return [
        FeedbackCandidate(
            support_side=support_side,
            target_support_com_shift=target_support_com_shift,
            com_kp=com_kp,
            com_kd=com_kd,
            roll_kp=roll_kp,
            max_support_roll=max_support_roll,
        )
        for (
            support_side,
            target_support_com_shift,
            com_kp,
            com_kd,
            roll_kp,
            max_support_roll,
        ) in product(
            support_sides,
            _parse_float_list(args.target_support_com_shifts),
            _parse_float_list(args.com_kp_values),
            _parse_float_list(args.com_kd_values),
            _parse_float_list(args.roll_kp_values),
            _parse_float_list(args.max_support_roll_values),
        )
    ]


def _score_result(
    *,
    max_support_com_shift: float,
    mean_support_com_shift_last_50: float,
    max_support_force_ratio: float,
    both_contact_ratio: float,
    none_contact_ratio: float,
    base_proxy_ratio: float,
    terminated: bool,
) -> float:
    """Return a ranking score for lateral authority under feedback."""
    return (
        max_support_com_shift * 1000.0
        + mean_support_com_shift_last_50 * 1200.0
        + max_support_force_ratio * 40.0
        + both_contact_ratio * 12.0
        - none_contact_ratio * 30.0
        - base_proxy_ratio * 40.0
        - (80.0 if terminated else 0.0)
    )


def _run_candidate(
    candidate: FeedbackCandidate,
    *,
    settle_steps: int,
    steps: int,
    seed: int,
) -> FeedbackSweepResult:
    """Run one closed-loop COM-feedback candidate."""
    reward_config = load_seedon_config_from_env()
    env = SeedonStandingEnv(reset_noise_scale=0.0, reward_config=reward_config)
    try:
        env.reset(seed=seed)
        nominal_target = env._apply_safe_joint_target_clamps(env._nominal_joint_qpos.copy())
        for _ in range(settle_steps):
            env._do_pd_simulation(nominal_target)

        support_side = candidate.support_side
        brace_target = _brace_target(env, support_side)
        left_foot_geom_id = env._geom_id(LEFT_FOOT_GEOM)
        right_foot_geom_id = env._geom_id(RIGHT_FOOT_GEOM)
        initial_com_y = float(_overall_com(env)[1])
        left_foot_y = float(env.data.geom_xpos[left_foot_geom_id][1])
        right_foot_y = float(env.data.geom_xpos[right_foot_geom_id][1])
        center_y = 0.5 * (left_foot_y + right_foot_y)
        initial_support_shift = (
            initial_com_y - center_y if support_side == "left" else center_y - initial_com_y
        )
        previous_support_shift = 0.0
        support_shifts: list[float] = []
        support_force_ratios: list[float] = []
        max_abs_com_y_delta = 0.0
        max_support_com_shift = 0.0
        max_support_force_ratio = 0.0
        support_only_steps = 0
        both_steps = 0
        none_steps = 0
        base_proxy_steps = 0
        terminated = False
        terminated_step = 0

        for step_index in range(1, steps + 1):
            com_y = float(_overall_com(env)[1])
            left_foot_y = float(env.data.geom_xpos[left_foot_geom_id][1])
            right_foot_y = float(env.data.geom_xpos[right_foot_geom_id][1])
            center_y = 0.5 * (left_foot_y + right_foot_y)
            actual_support_shift = com_y - center_y if support_side == "left" else center_y - com_y
            support_shift_delta = actual_support_shift - initial_support_shift
            support_shift_velocity = support_shift_delta - previous_support_shift
            previous_support_shift = support_shift_delta
            desired_support_shift = candidate.target_support_com_shift
            support_shift_error = desired_support_shift - support_shift_delta

            base_roll = _quat_to_roll(env.data.xquat[env._base_body_id])
            control_signal = (
                candidate.com_kp * support_shift_error
                - candidate.com_kd * support_shift_velocity
                - candidate.roll_kp * base_roll
            )
            support_roll = float(np.clip(control_signal, 0.0, candidate.max_support_roll))

            target = brace_target.copy()
            _apply_support_roll(target, support_side, support_roll)
            target = env._apply_safe_joint_target_clamps(target)
            env._do_pd_simulation(target)

            com_y = float(_overall_com(env)[1])
            actual_support_shift = (
                com_y - 0.5 * (float(env.data.geom_xpos[left_foot_geom_id][1]) + float(env.data.geom_xpos[right_foot_geom_id][1]))
                if support_side == "left"
                else 0.5 * (float(env.data.geom_xpos[left_foot_geom_id][1]) + float(env.data.geom_xpos[right_foot_geom_id][1])) - com_y
            )
            support_shift_delta = actual_support_shift - initial_support_shift
            support_shifts.append(support_shift_delta)
            max_support_com_shift = max(max_support_com_shift, support_shift_delta)
            max_abs_com_y_delta = max(max_abs_com_y_delta, abs(com_y - initial_com_y))

            _, left_force = _foot_floor_load(env, "left")
            _, right_force = _foot_floor_load(env, "right")
            total_force = left_force + right_force
            if total_force > 1e-9:
                left_force_ratio = left_force / total_force
                right_force_ratio = right_force / total_force
            else:
                left_force_ratio = 0.0
                right_force_ratio = 0.0
            support_force_ratio = left_force_ratio if support_side == "left" else right_force_ratio
            support_force_ratios.append(support_force_ratio)
            max_support_force_ratio = max(max_support_force_ratio, support_force_ratio)

            contact_state, base_proxy_contact = _contact_state(env)
            if contact_state == "both":
                both_steps += 1
            elif contact_state == f"{support_side}_only":
                support_only_steps += 1
            elif contact_state == "none":
                none_steps += 1
            base_proxy_steps += int(base_proxy_contact)

            base_z = env._base_height()
            upright = env._base_upright()
            observation = env._get_obs()
            terminated = env._is_terminated(base_z, upright, observation)
            if terminated:
                terminated_step = step_index
                break

        if not terminated:
            terminated_step = steps

        executed_steps = max(len(support_shifts), 1)
        shift_tail = support_shifts[-50:] if len(support_shifts) >= 50 else support_shifts
        force_tail = support_force_ratios[-50:] if len(support_force_ratios) >= 50 else support_force_ratios
        mean_support_com_shift_last_50 = float(np.mean(shift_tail)) if shift_tail else 0.0
        mean_support_force_ratio_last_50 = float(np.mean(force_tail)) if force_tail else 0.0
        both_contact_ratio = both_steps / executed_steps
        none_contact_ratio = none_steps / executed_steps
        base_proxy_ratio = base_proxy_steps / executed_steps
        score = _score_result(
            max_support_com_shift=max_support_com_shift,
            mean_support_com_shift_last_50=mean_support_com_shift_last_50,
            max_support_force_ratio=max_support_force_ratio,
            both_contact_ratio=both_contact_ratio,
            none_contact_ratio=none_contact_ratio,
            base_proxy_ratio=base_proxy_ratio,
            terminated=terminated,
        )

        return FeedbackSweepResult(
            case_name=candidate.case_name,
            support_side=support_side,
            target_support_com_shift=candidate.target_support_com_shift,
            com_kp=candidate.com_kp,
            com_kd=candidate.com_kd,
            roll_kp=candidate.roll_kp,
            max_support_roll=candidate.max_support_roll,
            max_abs_com_y_delta=float(max_abs_com_y_delta),
            max_support_com_shift=float(max_support_com_shift),
            mean_support_com_shift_last_50=float(mean_support_com_shift_last_50),
            max_support_force_ratio=float(max_support_force_ratio),
            mean_support_force_ratio_last_50=float(mean_support_force_ratio_last_50),
            support_only_steps=int(support_only_steps),
            both_contact_ratio=float(both_contact_ratio),
            none_contact_ratio=float(none_contact_ratio),
            base_proxy_ratio=float(base_proxy_ratio),
            terminated=bool(terminated),
            terminated_step=int(terminated_step),
            score=float(score),
        )
    finally:
        env.close()


def _write_csv(path: Path, rows: list[FeedbackSweepResult]) -> None:
    """Write sweep rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows([asdict(row) for row in rows])


def main(argv: list[str] | None = None) -> int:
    """Run the COM-feedback balance sweep."""
    args = _build_parser().parse_args(argv)
    if args.settle_steps < 0:
        raise ValueError("--settle-steps must be non-negative.")
    if args.steps <= 0:
        raise ValueError("--steps must be positive.")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive.")

    candidates = _iter_candidates(args)
    results = [
        _run_candidate(
            candidate,
            settle_steps=args.settle_steps,
            steps=args.steps,
            seed=args.seed,
        )
        for candidate in candidates
    ]
    _write_csv(args.out_csv, results)

    ranked = sorted(
        results,
        key=lambda row: (
            row.score,
            not row.terminated,
            row.max_support_com_shift,
            row.max_support_force_ratio,
        ),
        reverse=True,
    )

    print(f"cases: {len(results)}")
    print(
        "rank case max_shift mean_last50 max_force both_ratio "
        "none_ratio term term_step score"
    )
    for index, row in enumerate(ranked[: args.top_k], start=1):
        print(
            f"{index:>4} {row.case_name:>46} {row.max_support_com_shift:>9.4f} "
            f"{row.mean_support_com_shift_last_50:>11.4f} {row.max_support_force_ratio:>9.3f} "
            f"{row.both_contact_ratio:>10.3f} {row.none_contact_ratio:>10.3f} "
            f"{str(row.terminated):>5} {row.terminated_step:>9} {row.score:>8.2f}"
        )

    print(f"\ncsv: {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
