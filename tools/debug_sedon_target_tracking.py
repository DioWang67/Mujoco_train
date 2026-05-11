"""Trace Sedon PD target angles, actual qpos, and unload/lift contact behavior."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
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


DEFAULT_OUT_CSV = DEBUG_OUT_DIR / "target_tracking.csv"
FOOT_INDEX_BY_LEG = {"right": 0, "left": 1}
LIFT_JOINT_INDICES = {
    "right": (2, 3, 4),
    "left": (7, 8, 9),
}


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


def _ctrl_saturated(env: SedonStandingEnv, tolerance: float = 1e-6) -> bool:
    """Return whether any actuator command is clipped at its control range."""
    lower = env.model.actuator_ctrlrange[:, 0]
    upper = env.model.actuator_ctrlrange[:, 1]
    return bool(np.any(env.data.ctrl <= lower + tolerance) or np.any(env.data.ctrl >= upper - tolerance))


def _target_components(
    env: SedonStandingEnv,
    *,
    lifted_leg: str,
    support_roll: float,
    hip_delta: float,
    knee_delta: float,
    ankle_delta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return unload-only and full lift targets."""
    unload_target = env._nominal_joint_qpos.copy()
    if lifted_leg == "right":
        unload_target[1] = support_roll
        unload_target[6] = -support_roll
    elif lifted_leg == "left":
        unload_target[1] = -support_roll
        unload_target[6] = support_roll
    else:
        raise ValueError(f"Unsupported lifted leg: {lifted_leg}")

    lift_target = unload_target.copy()
    hip_index, knee_index, ankle_index = LIFT_JOINT_INDICES[lifted_leg]
    lift_target[hip_index] += hip_delta
    lift_target[knee_index] += knee_delta
    lift_target[ankle_index] += ankle_delta
    return unload_target, lift_target


def _step_target(
    *,
    mode: str,
    nominal: np.ndarray,
    unload_target: np.ndarray,
    lift_target: np.ndarray,
    step_index: int,
    unload_steps: int,
    lift_steps: int,
    hold_steps: int,
) -> np.ndarray:
    """Return the current target for an unload/lift scripted sequence."""
    if mode == "unload":
        alpha = min(1.0, (step_index + 1) / unload_steps)
        return nominal + (unload_target - nominal) * alpha
    if mode == "lift":
        alpha = min(1.0, (step_index + 1) / lift_steps)
        return nominal + (lift_target - nominal) * alpha
    if mode != "unload-lift":
        raise ValueError(f"Unsupported mode: {mode}")

    if step_index < unload_steps:
        alpha = min(1.0, (step_index + 1) / unload_steps)
        return nominal + (unload_target - nominal) * alpha
    lift_index = step_index - unload_steps
    if lift_index < lift_steps:
        alpha = min(1.0, (lift_index + 1) / lift_steps)
        return unload_target + (lift_target - unload_target) * alpha
    del hold_steps
    return lift_target


def _fieldnames() -> list[str]:
    """Return CSV fieldnames for target tracking rows."""
    names = [
        "step",
        "mode",
        "lifted_leg",
        "base_z",
        "upright",
        "lifted_foot_bottom_z",
        "support_foot_bottom_z",
        "lifted_foot_clearance",
        "contact_state",
        "base_proxy_floor_contact",
        "ctrl_saturated",
        "joint_error_l2",
        "max_abs_joint_error",
    ]
    for joint_name in JOINT_NAMES:
        names.extend(
            (
                f"{joint_name}_target",
                f"{joint_name}_actual",
                f"{joint_name}_error",
            )
        )
    return names


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    """Write target tracking rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_fieldnames())
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("unload", "lift", "unload-lift"), default="unload-lift")
    parser.add_argument("--lifted-leg", choices=("right", "left"), default="right")
    parser.add_argument("--relaxed-foot", action="store_true")
    parser.add_argument("--support-roll", type=float, default=0.06)
    parser.add_argument("--hip-delta", type=float, default=0.03)
    parser.add_argument("--knee-delta", type=float, default=-0.06)
    parser.add_argument("--ankle-delta", type=float, default=-0.03)
    parser.add_argument("--unload-steps", type=int, default=80)
    parser.add_argument("--lift-steps", type=int, default=80)
    parser.add_argument("--hold-steps", type=int, default=80)
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the target tracking diagnostic."""
    args = build_parser().parse_args(argv)
    if min(args.unload_steps, args.lift_steps, args.hold_steps, args.print_every) <= 0:
        raise ValueError("Step counts and --print-every must be positive.")

    total_steps = {
        "unload": args.unload_steps + args.hold_steps,
        "lift": args.lift_steps + args.hold_steps,
        "unload-lift": args.unload_steps + args.lift_steps + args.hold_steps,
    }[args.mode]

    env = SedonStandingEnv(
        reset_noise_scale=0.0,
        reward_config=SedonStandingConfig(gait_mode="fsm"),
    )
    rows: list[dict[str, object]] = []
    contact_counts: Counter[str] = Counter()
    base_proxy_floor_steps = 0
    saturated_ctrl_steps = 0
    min_base_z = float("inf")
    min_upright = float("inf")
    max_lifted_foot_bottom_z = -float("inf")
    terminated = False
    try:
        env.reset(seed=42)
        if args.relaxed_foot:
            apply_foot_size_override(env.model, env.data, RELAXED_FOOT_SIZE)
        nominal = env._nominal_joint_qpos.copy()
        unload_target, lift_target = _target_components(
            env,
            lifted_leg=args.lifted_leg,
            support_roll=args.support_roll,
            hip_delta=args.hip_delta,
            knee_delta=args.knee_delta,
            ankle_delta=args.ankle_delta,
        )
        lifted_index = FOOT_INDEX_BY_LEG[args.lifted_leg]
        support_index = 1 - lifted_index
        initial_lifted_bottom_z = float(env._foot_bottom_heights()[lifted_index])

        print("step base_z upright lifted_z clearance contact err_l2 sat")
        for step_index in range(total_steps):
            target = _step_target(
                mode=args.mode,
                nominal=nominal,
                unload_target=unload_target,
                lift_target=lift_target,
                step_index=step_index,
                unload_steps=args.unload_steps,
                lift_steps=args.lift_steps,
                hold_steps=args.hold_steps,
            )
            env._do_pd_simulation(target)
            qpos = env._joint_positions()
            error = qpos - target
            saturated = _ctrl_saturated(env)
            if saturated:
                saturated_ctrl_steps += 1
            obs = env._get_obs()
            base_z = env._base_height()
            upright = env._base_upright()
            terminated = env._is_terminated(base_z, upright, obs)
            state, base_contact = _contact_state(env)
            contact_counts.update([state])
            if base_contact:
                base_proxy_floor_steps += 1
            foot_bottoms = env._foot_bottom_heights()
            lifted_bottom_z = float(foot_bottoms[lifted_index])
            support_bottom_z = float(foot_bottoms[support_index])
            clearance = lifted_bottom_z - initial_lifted_bottom_z
            min_base_z = min(min_base_z, base_z)
            min_upright = min(min_upright, upright)
            max_lifted_foot_bottom_z = max(max_lifted_foot_bottom_z, lifted_bottom_z)
            joint_error_l2 = float(np.dot(error, error))
            max_abs_error = float(np.max(np.abs(error)))
            row: dict[str, object] = {
                "step": step_index + 1,
                "mode": args.mode,
                "lifted_leg": args.lifted_leg,
                "base_z": base_z,
                "upright": upright,
                "lifted_foot_bottom_z": lifted_bottom_z,
                "support_foot_bottom_z": support_bottom_z,
                "lifted_foot_clearance": clearance,
                "contact_state": state,
                "base_proxy_floor_contact": base_contact,
                "ctrl_saturated": saturated,
                "joint_error_l2": joint_error_l2,
                "max_abs_joint_error": max_abs_error,
            }
            for index, joint_name in enumerate(JOINT_NAMES):
                row[f"{joint_name}_target"] = float(target[index])
                row[f"{joint_name}_actual"] = float(qpos[index])
                row[f"{joint_name}_error"] = float(error[index])
            rows.append(row)
            if step_index == 0 or (step_index + 1) % args.print_every == 0 or terminated:
                print(
                    f"{step_index + 1:>4} {base_z:>6.3f} {upright:>7.3f} "
                    f"{lifted_bottom_z:>8.4f} {clearance:>9.4f} "
                    f"{state:>10} {joint_error_l2:>7.4f} {str(saturated):>5}"
                )
            if terminated:
                break
    finally:
        env.close()

    _write_rows(args.out_csv, rows)
    print(f"\nCSV: {args.out_csv}")
    print(f"steps: {len(rows)} terminated={terminated}")
    print(f"min_base_z: {min_base_z:.5f}")
    print(f"min_upright: {min_upright:.5f}")
    print(f"max_lifted_foot_bottom_z: {max_lifted_foot_bottom_z:.5f}")
    print(f"base_proxy_floor_steps: {base_proxy_floor_steps}")
    print(f"saturated_ctrl_steps: {saturated_ctrl_steps}")
    print("contact_state_counts:")
    for state in ("both", "right_only", "left_only", "none"):
        print(f"  {state}: {contact_counts[state]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
