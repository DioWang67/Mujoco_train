"""Sweep Seedon PD gains under floor contact to isolate load-transfer authority."""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import mujoco
import numpy as np

from seedon_baseline.env import SeedonStandingEnv, load_seedon_config_from_env
from tools.seedon_debug_common import DEBUG_OUT_DIR, contact_pairs


DEFAULT_OUT_CSV = DEBUG_OUT_DIR / "pd_load_transfer_sweep.csv"
RIGHT_FOOT_GEOM = "R_foot_collision"
LEFT_FOOT_GEOM = "L_foot_collision"
FLOOR_GEOM = "floor"
HIP_ROLL_INDEX = {"right": 1, "left": 6}


@dataclass(frozen=True)
class PdLoadTransferResult:
    """One PD-gain sweep result row."""

    support_side: str
    hip_roll_offset: float
    pd_stiffness: float
    pd_damping: float
    steps: int
    terminated: bool
    terminated_step: int
    final_contact_state: str
    support_only_steps: int
    both_contact_steps: int
    none_contact_steps: int
    max_abs_com_y_delta: float
    max_abs_base_y_delta: float
    max_abs_base_roll_delta: float
    final_com_y_delta: float
    final_base_y_delta: float
    final_base_roll_delta: float
    max_abs_support_hip_roll_error: float
    final_support_hip_roll_error: float
    support_hip_roll_target: float
    final_support_hip_roll_qpos: float
    final_left_normal_force: float
    final_right_normal_force: float
    final_left_force_ratio: float
    final_right_force_ratio: float
    max_left_force_ratio: float
    max_right_force_ratio: float
    ctrl_saturation_ratio: float
    max_abs_ctrl_fraction: float
    min_base_z: float
    min_upright: float


def _parse_float_list(raw_value: str) -> list[float]:
    """Parse comma-separated floats."""
    values = [float(part.strip()) for part in raw_value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one float.")
    return values


def _overall_com(env: SeedonStandingEnv) -> np.ndarray:
    """Return the mass-weighted whole-body COM in world coordinates."""
    masses = env.model.body_mass
    total_mass = float(np.sum(masses))
    if total_mass <= 0.0:
        raise ValueError("Model has no positive body mass.")
    return np.sum(env.data.xipos * masses[:, None], axis=0) / total_mass


def _quat_to_roll(quat: np.ndarray) -> float:
    """Return world-frame base roll from a MuJoCo quaternion."""
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


def _target_positions(env: SeedonStandingEnv, support_side: str, hip_roll_offset: float) -> np.ndarray:
    """Return nominal target with mirrored hip-roll offsets."""
    target = env._nominal_joint_qpos.copy()
    if support_side == "left":
        target[HIP_ROLL_INDEX["right"]] += hip_roll_offset
        target[HIP_ROLL_INDEX["left"]] -= hip_roll_offset
    elif support_side == "right":
        target[HIP_ROLL_INDEX["right"]] -= hip_roll_offset
        target[HIP_ROLL_INDEX["left"]] += hip_roll_offset
    else:
        raise ValueError(f"Unsupported support side: {support_side}")
    return env._apply_safe_joint_target_clamps(target)


def _write_csv(path: Path, results: list[PdLoadTransferResult]) -> None:
    """Write sweep results to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not results:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        writer.writerows([asdict(result) for result in results])


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kp-values", default="35,70,140,280")
    parser.add_argument("--kd-values", default="2,4,8,16")
    parser.add_argument("--hip-roll-offsets", default="0.02,0.05,0.08")
    parser.add_argument("--support-sides", default="left,right")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    return parser


def _run_case(
    *,
    support_side: str,
    hip_roll_offset: float,
    pd_stiffness: float,
    pd_damping: float,
    steps: int,
    seed: int,
):
    """Run one with-floor PD case and return metrics."""
    base_config = load_seedon_config_from_env()
    reward_config = replace(
        base_config,
        pd_stiffness=pd_stiffness,
        pd_damping=pd_damping,
    )
    env = SeedonStandingEnv(reset_noise_scale=0.0, reward_config=reward_config)
    try:
        env.reset(seed=seed)
        target_positions = _target_positions(env, support_side, hip_roll_offset)
        initial_base_y = float(env.data.qpos[1])
        initial_com_y = float(_overall_com(env)[1])
        initial_base_roll = _quat_to_roll(env.data.xquat[env._base_body_id])
        support_joint_index = HIP_ROLL_INDEX[support_side]
        support_only_label = f"{support_side}_only"

        max_abs_com_y_delta = 0.0
        max_abs_base_y_delta = 0.0
        max_abs_base_roll_delta = 0.0
        max_abs_support_hip_roll_error = 0.0
        max_left_force_ratio = 0.0
        max_right_force_ratio = 0.0
        max_abs_ctrl_fraction = 0.0
        saturation_hits = 0
        control_updates = 0
        min_base_z = float("inf")
        min_upright = float("inf")
        terminated = False
        terminated_step = 0
        final_contact_state = "none"
        final_left_normal_force = 0.0
        final_right_normal_force = 0.0
        contact_counts: Counter[str] = Counter()

        for step_index in range(1, steps + 1):
            for _ in range(env.frame_skip):
                ctrl = env._pd_control(target_positions)
                env.data.ctrl[:] = ctrl
                ctrl_abs = np.abs(ctrl)
                ctrl_range = np.maximum(np.abs(env._ctrl_range[:, 0]), np.abs(env._ctrl_range[:, 1]))
                ctrl_fraction = ctrl_abs / np.maximum(ctrl_range, 1e-9)
                max_abs_ctrl_fraction = max(max_abs_ctrl_fraction, float(np.max(ctrl_fraction)))
                saturation_hits += int(np.any(ctrl_fraction >= 0.999))
                control_updates += 1
                mujoco.mj_step(env.model, env.data)

            joint_positions = env._joint_positions()
            support_error = float(joint_positions[support_joint_index] - target_positions[support_joint_index])
            base_y_delta = float(env.data.qpos[1] - initial_base_y)
            com_y_delta = float(_overall_com(env)[1] - initial_com_y)
            base_roll_delta = float(_quat_to_roll(env.data.xquat[env._base_body_id]) - initial_base_roll)
            max_abs_support_hip_roll_error = max(max_abs_support_hip_roll_error, abs(support_error))
            max_abs_base_y_delta = max(max_abs_base_y_delta, abs(base_y_delta))
            max_abs_com_y_delta = max(max_abs_com_y_delta, abs(com_y_delta))
            max_abs_base_roll_delta = max(max_abs_base_roll_delta, abs(base_roll_delta))

            _, left_force = _foot_floor_load(env, "left")
            _, right_force = _foot_floor_load(env, "right")
            total_force = max(left_force + right_force, 1e-9)
            left_force_ratio = left_force / total_force
            right_force_ratio = right_force / total_force
            max_left_force_ratio = max(max_left_force_ratio, left_force_ratio)
            max_right_force_ratio = max(max_right_force_ratio, right_force_ratio)
            final_left_normal_force = left_force
            final_right_normal_force = right_force

            final_contact_state = _contact_state(env)
            contact_counts.update([final_contact_state])
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

        final_joint_positions = env._joint_positions()
        final_support_hip_roll_qpos = float(final_joint_positions[support_joint_index])
        final_support_hip_roll_error = float(
            final_support_hip_roll_qpos - target_positions[support_joint_index]
        )
        final_base_y_delta = float(env.data.qpos[1] - initial_base_y)
        final_com_y_delta = float(_overall_com(env)[1] - initial_com_y)
        final_base_roll_delta = float(
            _quat_to_roll(env.data.xquat[env._base_body_id]) - initial_base_roll
        )
        total_force = max(final_left_normal_force + final_right_normal_force, 1e-9)
        final_left_force_ratio = final_left_normal_force / total_force
        final_right_force_ratio = final_right_normal_force / total_force

        return PdLoadTransferResult(
            support_side=support_side,
            hip_roll_offset=hip_roll_offset,
            pd_stiffness=pd_stiffness,
            pd_damping=pd_damping,
            steps=steps,
            terminated=terminated,
            terminated_step=terminated_step,
            final_contact_state=final_contact_state,
            support_only_steps=contact_counts[support_only_label],
            both_contact_steps=contact_counts["both"],
            none_contact_steps=contact_counts["none"],
            max_abs_com_y_delta=max_abs_com_y_delta,
            max_abs_base_y_delta=max_abs_base_y_delta,
            max_abs_base_roll_delta=max_abs_base_roll_delta,
            final_com_y_delta=final_com_y_delta,
            final_base_y_delta=final_base_y_delta,
            final_base_roll_delta=final_base_roll_delta,
            max_abs_support_hip_roll_error=max_abs_support_hip_roll_error,
            final_support_hip_roll_error=final_support_hip_roll_error,
            support_hip_roll_target=float(target_positions[support_joint_index]),
            final_support_hip_roll_qpos=final_support_hip_roll_qpos,
            final_left_normal_force=final_left_normal_force,
            final_right_normal_force=final_right_normal_force,
            final_left_force_ratio=final_left_force_ratio,
            final_right_force_ratio=final_right_force_ratio,
            max_left_force_ratio=max_left_force_ratio,
            max_right_force_ratio=max_right_force_ratio,
            ctrl_saturation_ratio=saturation_hits / max(control_updates, 1),
            max_abs_ctrl_fraction=max_abs_ctrl_fraction,
            min_base_z=min_base_z,
            min_upright=min_upright,
        )
    finally:
        env.close()


def main(argv: list[str] | None = None) -> int:
    """Run the PD gain sweep and print compact conclusions."""
    args = build_parser().parse_args(argv)
    if args.steps <= 0:
        raise ValueError("--steps must be positive.")

    kp_values = _parse_float_list(args.kp_values)
    kd_values = _parse_float_list(args.kd_values)
    hip_roll_offsets = _parse_float_list(args.hip_roll_offsets)
    support_sides = [part.strip() for part in args.support_sides.split(",") if part.strip()]
    if len(kp_values) != len(kd_values):
        raise ValueError("--kp-values and --kd-values must have the same length for paired sweeps.")
    for side in support_sides:
        if side not in {"left", "right"}:
            raise ValueError(f"Unsupported support side: {side}")

    results: list[PdLoadTransferResult] = []
    print(
        "side kp kd offset term term_step max_com_dy max_roll_dy "
        "hip_err forceL forceR support_only both sat_ratio contact"
    )
    for kp_value, kd_value in zip(kp_values, kd_values, strict=True):
        for support_side in support_sides:
            for hip_roll_offset in hip_roll_offsets:
                result = _run_case(
                    support_side=support_side,
                    hip_roll_offset=hip_roll_offset,
                    pd_stiffness=kp_value,
                    pd_damping=kd_value,
                    steps=args.steps,
                    seed=args.seed,
                )
                results.append(result)
                print(
                    f"{result.support_side:>5} {result.pd_stiffness:>4.0f} {result.pd_damping:>3.0f} "
                    f"{result.hip_roll_offset:>6.2f} {str(result.terminated):>5} "
                    f"{result.terminated_step:>9} {result.max_abs_com_y_delta:>10.4f} "
                    f"{result.max_abs_base_roll_delta:>11.4f} {result.max_abs_support_hip_roll_error:>7.4f} "
                    f"{result.max_left_force_ratio:>6.3f} {result.max_right_force_ratio:>6.3f} "
                    f"{result.support_only_steps:>12} {result.both_contact_steps:>4} "
                    f"{result.ctrl_saturation_ratio:>9.3f} {result.final_contact_state:>10}"
                )

    _write_csv(args.out_csv, results)
    print(f"\ncsv: {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
