"""Sweep Seedon reset base-height offsets and measure contact initialization stability."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import mujoco
import numpy as np

from seedon_baseline.env import SeedonStandingEnv, load_seedon_config_from_env
from tools.seedon_debug_common import DEBUG_OUT_DIR, LEFT_FOOT_GEOM, RIGHT_FOOT_GEOM, contact_pairs


DEFAULT_OUT_CSV = DEBUG_OUT_DIR / "contact_initialization_sweep.csv"
FLOOR_GEOM = "floor"
HIP_ROLL_INDEX = {"right": 1, "left": 6}


@dataclass(frozen=True)
class ContactInitializationResult:
    """One reset-height sweep result row."""

    support_side: str
    base_height_offset: float
    target_base_height: float
    initial_base_z: float
    initial_left_foot_bottom_z: float
    initial_right_foot_bottom_z: float
    initial_left_contact_count: int
    initial_right_contact_count: int
    initial_left_normal_force: float
    initial_right_normal_force: float
    initial_contact_state: str
    settled_base_z: float
    settled_left_foot_bottom_z: float
    settled_right_foot_bottom_z: float
    settled_left_contact_count: int
    settled_right_contact_count: int
    settled_left_normal_force: float
    settled_right_normal_force: float
    settled_contact_state: str
    steps: int
    terminated: bool
    terminated_step: int
    final_contact_state: str
    support_only_steps: int
    both_contact_steps: int
    none_contact_steps: int
    max_abs_com_y_delta: float
    max_abs_base_roll_delta: float
    final_com_y_delta: float
    final_base_roll_delta: float
    min_base_z: float
    min_upright: float


def _parse_float_list(raw_value: str) -> list[float]:
    """Parse comma-separated floats."""
    values = [float(part.strip()) for part in raw_value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one float.")
    return values


def _overall_com(env: SeedonStandingEnv) -> np.ndarray:
    """Return mass-weighted whole-body COM in world coordinates."""
    masses = env.model.body_mass
    total_mass = float(np.sum(masses))
    if total_mass <= 0.0:
        raise ValueError("Model has no positive body mass.")
    return np.sum(env.data.xipos * masses[:, None], axis=0) / total_mass


def _quat_to_roll(quat: np.ndarray) -> float:
    """Return base roll in radians from a MuJoCo quaternion."""
    w, x, y, z = [float(value) for value in quat]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    return float(math.atan2(sinr_cosp, cosr_cosp))


def _contact_state(env: SeedonStandingEnv) -> str:
    """Return compact left/right foot floor-contact state."""
    right = False
    left = False
    for name_a, name_b, _ in contact_pairs(env.model, env.data):
        pair = {name_a, name_b}
        if pair == {FLOOR_GEOM, RIGHT_FOOT_GEOM}:
            right = True
        elif pair == {FLOOR_GEOM, LEFT_FOOT_GEOM}:
            left = True
    if right and left:
        return "both"
    if right:
        return "right_only"
    if left:
        return "left_only"
    return "none"


def _foot_floor_load(env: SeedonStandingEnv, side: str) -> tuple[int, float]:
    """Return floor-contact count and summed normal force for one foot."""
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


def _target_positions(env: SeedonStandingEnv, support_side: str, support_roll: float) -> np.ndarray:
    """Return mirrored hip-roll load-transfer target positions."""
    target = env._nominal_joint_qpos.copy()
    if support_side == "left":
        target[HIP_ROLL_INDEX["right"]] += support_roll
        target[HIP_ROLL_INDEX["left"]] -= support_roll
    elif support_side == "right":
        target[HIP_ROLL_INDEX["right"]] -= support_roll
        target[HIP_ROLL_INDEX["left"]] += support_roll
    else:
        raise ValueError(f"Unsupported support side: {support_side}")
    return env._apply_safe_joint_target_clamps(target)


def _snapshot(env: SeedonStandingEnv) -> dict[str, float | int | str]:
    """Capture foot-bottom/contact snapshot from current state."""
    left_contact_count, left_normal_force = _foot_floor_load(env, "left")
    right_contact_count, right_normal_force = _foot_floor_load(env, "right")
    foot_bottoms = env._foot_bottom_heights()
    return {
        "base_z": float(env._base_height()),
        "left_foot_bottom_z": float(foot_bottoms[1]),
        "right_foot_bottom_z": float(foot_bottoms[0]),
        "left_contact_count": left_contact_count,
        "right_contact_count": right_contact_count,
        "left_normal_force": left_normal_force,
        "right_normal_force": right_normal_force,
        "contact_state": _contact_state(env),
    }


def _write_csv(path: Path, rows: list[ContactInitializationResult]) -> None:
    """Write flat result rows to CSV."""
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
    parser.add_argument("--base-height-offsets", default="-0.005,0.000,0.003,0.005,0.007,0.010,0.015")
    parser.add_argument("--support-sides", default="left,right")
    parser.add_argument("--settle-steps", type=int, default=20)
    parser.add_argument("--support-roll", type=float, default=0.10)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    return parser


def _run_case(
    *,
    base_height_offset: float,
    support_side: str,
    settle_steps: int,
    support_roll: float,
    steps: int,
    seed: int,
) -> ContactInitializationResult:
    """Run one reset-height and support-side case."""
    base_config = load_seedon_config_from_env()
    reward_config = replace(
        base_config,
        target_base_height=base_config.target_base_height + base_height_offset,
    )
    env = SeedonStandingEnv(reset_noise_scale=0.0, reward_config=reward_config)
    try:
        env.reset(seed=seed)
        initial = _snapshot(env)
        initial_com_y = float(_overall_com(env)[1])
        initial_base_roll = _quat_to_roll(env.data.xquat[env._base_body_id])

        settle_target = env._nominal_joint_qpos.copy()
        for _ in range(settle_steps):
            env._do_pd_simulation(settle_target)
        settled = _snapshot(env)

        target_positions = _target_positions(env, support_side, support_roll)
        support_only_label = f"{support_side}_only"
        max_abs_com_y_delta = 0.0
        max_abs_base_roll_delta = 0.0
        min_base_z = float("inf")
        min_upright = float("inf")
        terminated = False
        terminated_step = 0
        final_contact_state = str(settled["contact_state"])
        contact_counts = {"both": 0, "right_only": 0, "left_only": 0, "none": 0}

        for step_index in range(1, steps + 1):
            env._do_pd_simulation(target_positions)
            com_y_delta = float(_overall_com(env)[1] - initial_com_y)
            base_roll_delta = float(_quat_to_roll(env.data.xquat[env._base_body_id]) - initial_base_roll)
            max_abs_com_y_delta = max(max_abs_com_y_delta, abs(com_y_delta))
            max_abs_base_roll_delta = max(max_abs_base_roll_delta, abs(base_roll_delta))
            final_contact_state = _contact_state(env)
            contact_counts[final_contact_state] += 1
            base_z = float(env._base_height())
            upright = float(env._base_upright())
            min_base_z = min(min_base_z, base_z)
            min_upright = min(min_upright, upright)
            obs = env._get_obs()
            terminated = bool(env._is_terminated(base_z, upright, obs))
            if terminated:
                terminated_step = step_index
                break

        if not terminated:
            terminated_step = steps

        return ContactInitializationResult(
            support_side=support_side,
            base_height_offset=base_height_offset,
            target_base_height=float(reward_config.target_base_height),
            initial_base_z=float(initial["base_z"]),
            initial_left_foot_bottom_z=float(initial["left_foot_bottom_z"]),
            initial_right_foot_bottom_z=float(initial["right_foot_bottom_z"]),
            initial_left_contact_count=int(initial["left_contact_count"]),
            initial_right_contact_count=int(initial["right_contact_count"]),
            initial_left_normal_force=float(initial["left_normal_force"]),
            initial_right_normal_force=float(initial["right_normal_force"]),
            initial_contact_state=str(initial["contact_state"]),
            settled_base_z=float(settled["base_z"]),
            settled_left_foot_bottom_z=float(settled["left_foot_bottom_z"]),
            settled_right_foot_bottom_z=float(settled["right_foot_bottom_z"]),
            settled_left_contact_count=int(settled["left_contact_count"]),
            settled_right_contact_count=int(settled["right_contact_count"]),
            settled_left_normal_force=float(settled["left_normal_force"]),
            settled_right_normal_force=float(settled["right_normal_force"]),
            settled_contact_state=str(settled["contact_state"]),
            steps=steps,
            terminated=terminated,
            terminated_step=terminated_step,
            final_contact_state=final_contact_state,
            support_only_steps=contact_counts[support_only_label],
            both_contact_steps=contact_counts["both"],
            none_contact_steps=contact_counts["none"],
            max_abs_com_y_delta=max_abs_com_y_delta,
            max_abs_base_roll_delta=max_abs_base_roll_delta,
            final_com_y_delta=float(_overall_com(env)[1] - initial_com_y),
            final_base_roll_delta=float(
                _quat_to_roll(env.data.xquat[env._base_body_id]) - initial_base_roll
            ),
            min_base_z=min_base_z,
            min_upright=min_upright,
        )
    finally:
        env.close()


def main(argv: list[str] | None = None) -> int:
    """Run the contact initialization sweep and print compact conclusions."""
    args = build_parser().parse_args(argv)
    if args.settle_steps < 0:
        raise ValueError("--settle-steps must be non-negative.")
    if args.steps <= 0:
        raise ValueError("--steps must be positive.")
    base_height_offsets = _parse_float_list(args.base_height_offsets)
    support_sides = [part.strip() for part in args.support_sides.split(",") if part.strip()]
    if not support_sides:
        raise ValueError("--support-sides must contain at least one side.")
    for side in support_sides:
        if side not in {"left", "right"}:
            raise ValueError(f"Unsupported support side: {side}")

    rows: list[ContactInitializationResult] = []
    print(
        "offset side init_zL init_zR initLc initRc settleLc settleRc "
        "max_com_dy support_only both term_step init_state settle_state final_state"
    )
    for base_height_offset in base_height_offsets:
        for support_side in support_sides:
            row = _run_case(
                base_height_offset=base_height_offset,
                support_side=support_side,
                settle_steps=args.settle_steps,
                support_roll=args.support_roll,
                steps=args.steps,
                seed=args.seed,
            )
            rows.append(row)
            print(
                f"{row.base_height_offset:>6.3f} {row.support_side:>5} "
                f"{row.initial_left_foot_bottom_z:>7.4f} {row.initial_right_foot_bottom_z:>7.4f} "
                f"{row.initial_left_contact_count:>6} {row.initial_right_contact_count:>6} "
                f"{row.settled_left_contact_count:>8} {row.settled_right_contact_count:>8} "
                f"{row.max_abs_com_y_delta:>10.4f} {row.support_only_steps:>12} "
                f"{row.both_contact_steps:>4} {row.terminated_step:>9} "
                f"{row.initial_contact_state:>10} {row.settled_contact_state:>12} {row.final_contact_state:>10}"
            )

    _write_csv(args.out_csv, rows)
    print(f"\ncsv: {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
