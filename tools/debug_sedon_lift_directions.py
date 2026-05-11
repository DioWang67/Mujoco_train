"""Sweep Sedon swing-leg joint directions and measure actual foot clearance."""

from __future__ import annotations

import argparse
import csv
import itertools
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sedon_baseline.env import JOINT_NAMES, SedonStandingConfig, SedonStandingEnv
from tools.sedon_debug_common import (
    BASE_PROXY_GEOM,
    DEBUG_OUT_DIR,
    FLOOR_GEOM,
    LEFT_FOOT_GEOM,
    RELAXED_FOOT_SIZE,
    RIGHT_FOOT_GEOM,
    apply_foot_size_override,
    contact_pairs,
)


DEFAULT_OUT_CSV = DEBUG_OUT_DIR / "lift_direction_sweep.csv"
FOOT_INDEX_BY_LEG = {"right": 0, "left": 1}
LIFT_JOINT_INDICES = {
    "right": (2, 3, 4),
    "left": (7, 8, 9),
}


@dataclass(frozen=True)
class LiftDirectionResult:
    """Measured result for one swing-leg joint direction candidate."""

    lifted_leg: str
    hip_sign: int
    knee_sign: int
    ankle_sign: int
    max_clearance: float
    max_lifted_foot_bottom_z: float
    final_lifted_foot_bottom_z: float
    min_base_z: float
    min_upright: float
    terminated: bool
    steps: int
    contact_state_counts: dict[str, int]
    base_proxy_floor_steps: int
    saturated_ctrl_steps: int
    final_joint_error_l2: float


def _contact_state(env: SedonStandingEnv) -> tuple[str, bool]:
    """Return foot contact state and base-proxy-floor flag."""
    pairs = [set((name_a, name_b)) for name_a, name_b, _ in contact_pairs(env.model, env.data)]
    right = {FLOOR_GEOM, RIGHT_FOOT_GEOM} in pairs
    left = {FLOOR_GEOM, LEFT_FOOT_GEOM} in pairs
    base = {FLOOR_GEOM, BASE_PROXY_GEOM} in pairs
    if right and left:
        return "both", base
    if right:
        return "right_only", base
    if left:
        return "left_only", base
    return "none", base


def _joint_target(
    env: SedonStandingEnv,
    *,
    lifted_leg: str,
    hip_delta: float,
    knee_delta: float,
    ankle_delta: float,
    support_roll: float,
) -> np.ndarray:
    """Build a static swing-leg target for one candidate direction."""
    target = env._nominal_joint_qpos.copy()
    if lifted_leg == "right":
        target[1] = support_roll
        target[6] = -support_roll
    elif lifted_leg == "left":
        target[1] = -support_roll
        target[6] = support_roll
    else:
        raise ValueError(f"Unsupported lifted leg: {lifted_leg}")

    hip_index, knee_index, ankle_index = LIFT_JOINT_INDICES[lifted_leg]
    target[hip_index] += hip_delta
    target[knee_index] += knee_delta
    target[ankle_index] += ankle_delta
    return target


def _ctrl_saturated(env: SedonStandingEnv, tolerance: float = 1e-6) -> bool:
    """Return whether any actuator command is clipped at its control range."""
    lower = env.model.actuator_ctrlrange[:, 0]
    upper = env.model.actuator_ctrlrange[:, 1]
    return bool(np.any(env.data.ctrl <= lower + tolerance) or np.any(env.data.ctrl >= upper - tolerance))


def _run_candidate(
    *,
    lifted_leg: str,
    hip_sign: int,
    knee_sign: int,
    ankle_sign: int,
    hip_amp: float,
    knee_amp: float,
    ankle_amp: float,
    support_roll: float,
    ramp_steps: int,
    hold_steps: int,
    relaxed_foot: bool,
) -> LiftDirectionResult:
    """Run one direction candidate and return clearance metrics."""
    env = SedonStandingEnv(
        reset_noise_scale=0.0,
        reward_config=SedonStandingConfig(gait_mode="fsm"),
    )
    try:
        env.reset(seed=42)
        if relaxed_foot:
            apply_foot_size_override(env.model, env.data, RELAXED_FOOT_SIZE)
        nominal = env._nominal_joint_qpos.copy()
        target = _joint_target(
            env,
            lifted_leg=lifted_leg,
            hip_delta=hip_sign * hip_amp,
            knee_delta=knee_sign * knee_amp,
            ankle_delta=ankle_sign * ankle_amp,
            support_roll=support_roll,
        )
        lifted_foot_index = FOOT_INDEX_BY_LEG[lifted_leg]
        initial_lifted_bottom_z = float(env._foot_bottom_heights()[lifted_foot_index])
        max_lifted_bottom_z = initial_lifted_bottom_z
        final_lifted_bottom_z = initial_lifted_bottom_z
        min_base_z = float("inf")
        min_upright = float("inf")
        contact_counts: Counter[str] = Counter()
        base_proxy_floor_steps = 0
        saturated_ctrl_steps = 0
        terminated = False
        steps = 0

        for index in range(ramp_steps + hold_steps):
            alpha = min(1.0, (index + 1) / ramp_steps)
            env._do_pd_simulation(nominal + (target - nominal) * alpha)
            if _ctrl_saturated(env):
                saturated_ctrl_steps += 1
            obs = env._get_obs()
            base_z = env._base_height()
            upright = env._base_upright()
            terminated = env._is_terminated(base_z, upright, obs)
            state, base_contact = _contact_state(env)
            contact_counts.update([state])
            if base_contact:
                base_proxy_floor_steps += 1
            lifted_bottom_z = float(env._foot_bottom_heights()[lifted_foot_index])
            max_lifted_bottom_z = max(max_lifted_bottom_z, lifted_bottom_z)
            final_lifted_bottom_z = lifted_bottom_z
            min_base_z = min(min_base_z, base_z)
            min_upright = min(min_upright, upright)
            steps = index + 1
            if terminated:
                break

        joint_error = env._joint_positions() - target
        return LiftDirectionResult(
            lifted_leg=lifted_leg,
            hip_sign=hip_sign,
            knee_sign=knee_sign,
            ankle_sign=ankle_sign,
            max_clearance=max_lifted_bottom_z - initial_lifted_bottom_z,
            max_lifted_foot_bottom_z=max_lifted_bottom_z,
            final_lifted_foot_bottom_z=final_lifted_bottom_z,
            min_base_z=min_base_z,
            min_upright=min_upright,
            terminated=terminated,
            steps=steps,
            contact_state_counts={
                "both": contact_counts["both"],
                "right_only": contact_counts["right_only"],
                "left_only": contact_counts["left_only"],
                "none": contact_counts["none"],
            },
            base_proxy_floor_steps=base_proxy_floor_steps,
            saturated_ctrl_steps=saturated_ctrl_steps,
            final_joint_error_l2=float(np.dot(joint_error, joint_error)),
        )
    finally:
        env.close()


def _result_row(result: LiftDirectionResult) -> dict[str, object]:
    """Return a CSV row for one direction result."""
    counts = result.contact_state_counts
    return {
        "lifted_leg": result.lifted_leg,
        "hip_sign": result.hip_sign,
        "knee_sign": result.knee_sign,
        "ankle_sign": result.ankle_sign,
        "max_clearance": result.max_clearance,
        "max_lifted_foot_bottom_z": result.max_lifted_foot_bottom_z,
        "final_lifted_foot_bottom_z": result.final_lifted_foot_bottom_z,
        "min_base_z": result.min_base_z,
        "min_upright": result.min_upright,
        "terminated": result.terminated,
        "steps": result.steps,
        "both_steps": counts["both"],
        "right_only_steps": counts["right_only"],
        "left_only_steps": counts["left_only"],
        "none_steps": counts["none"],
        "base_proxy_floor_steps": result.base_proxy_floor_steps,
        "saturated_ctrl_steps": result.saturated_ctrl_steps,
        "final_joint_error_l2": result.final_joint_error_l2,
    }


def _write_csv(path: Path, results: list[LiftDirectionResult]) -> None:
    """Write all sweep results to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [_result_row(result) for result in results]
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relaxed-foot", action="store_true")
    parser.add_argument("--ramp-steps", type=int, default=40)
    parser.add_argument("--hold-steps", type=int, default=80)
    parser.add_argument("--support-roll", type=float, default=0.06)
    parser.add_argument("--hip-amp", type=float, default=0.15)
    parser.add_argument("--knee-amp", type=float, default=0.12)
    parser.add_argument("--ankle-amp", type=float, default=0.04)
    parser.add_argument("--top", type=int, default=8)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the swing-leg direction sweep."""
    args = build_parser().parse_args(argv)
    if args.ramp_steps <= 0:
        raise ValueError("--ramp-steps must be positive.")
    if args.hold_steps <= 0:
        raise ValueError("--hold-steps must be positive.")
    if min(args.hip_amp, args.knee_amp, args.ankle_amp) <= 0.0:
        raise ValueError("Joint amplitudes must be positive.")

    results: list[LiftDirectionResult] = []
    for lifted_leg in ("right", "left"):
        for hip_sign, knee_sign, ankle_sign in itertools.product((-1, 1), repeat=3):
            results.append(
                _run_candidate(
                    lifted_leg=lifted_leg,
                    hip_sign=hip_sign,
                    knee_sign=knee_sign,
                    ankle_sign=ankle_sign,
                    hip_amp=args.hip_amp,
                    knee_amp=args.knee_amp,
                    ankle_amp=args.ankle_amp,
                    support_roll=args.support_roll,
                    ramp_steps=args.ramp_steps,
                    hold_steps=args.hold_steps,
                    relaxed_foot=args.relaxed_foot,
                )
            )

    results.sort(
        key=lambda result: (
            result.max_clearance,
            result.max_lifted_foot_bottom_z,
            not result.terminated,
            -result.saturated_ctrl_steps,
        ),
        reverse=True,
    )
    _write_csv(args.out_csv, results)

    print(
        "lifted leg hip knee ankle clearance max_bottom min_z "
        "min_upright term both right_only left_only none sat_ctrl"
    )
    for result in results[: args.top]:
        counts = result.contact_state_counts
        print(
            f"{result.lifted_leg:>6} "
            f"{result.hip_sign:>3} {result.knee_sign:>4} {result.ankle_sign:>5} "
            f"{result.max_clearance:>9.5f} "
            f"{result.max_lifted_foot_bottom_z:>10.5f} "
            f"{result.min_base_z:>6.3f} "
            f"{result.min_upright:>10.3f} "
            f"{str(result.terminated):>5} "
            f"{counts['both']:>4} {counts['right_only']:>10} "
            f"{counts['left_only']:>9} {counts['none']:>4} "
            f"{result.saturated_ctrl_steps:>8}"
        )

    passing = [
        result
        for result in results
        if result.max_lifted_foot_bottom_z >= 0.005
        and result.min_base_z > 0.37
        and result.min_upright > 0.85
        and result.base_proxy_floor_steps == 0
    ]
    print(f"\nCSV: {args.out_csv}")
    print(f"candidates: {len(results)}")
    print(f"clearance_pass_candidates: {len(passing)}")
    if passing:
        best = passing[0]
        print(
            "best_pass: "
            f"leg={best.lifted_leg} hip={best.hip_sign} "
            f"knee={best.knee_sign} ankle={best.ankle_sign} "
            f"max_bottom={best.max_lifted_foot_bottom_z:.5f}"
        )
    else:
        print("best_pass: none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
