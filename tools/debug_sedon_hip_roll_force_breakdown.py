"""Break down Sedon hip-roll generalized forces under floor contact.

This diagnostic keeps the normal committed Sedon model and reward config. It
does not edit MJCF, reward, PPO, or the training scene. Instead, it runs an
isolated with-floor hip-roll target and records how actuator, passive, bias,
and constraint forces contribute on the hip-roll DOFs.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
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


DEFAULT_OUT_CSV = DEBUG_OUT_DIR / "hip_roll_force_breakdown.csv"
DEFAULT_CONTACT_ROW_OUT_CSV = DEBUG_OUT_DIR / "hip_roll_contact_row_breakdown.csv"
RIGHT_HIP_ROLL_INDEX = 1
LEFT_HIP_ROLL_INDEX = 6
RIGHT_HIP_ROLL_NAME = "R_joint_hip_roll"
LEFT_HIP_ROLL_NAME = "L_joint_hip_roll"
RIGHT_HIP_ROLL_ACTUATOR = "R_joint_hip_roll_motor"
LEFT_HIP_ROLL_ACTUATOR = "L_joint_hip_roll_motor"


@dataclass(frozen=True)
class DofForceBreakdown:
    """Generalized-force breakdown for one DOF at one step."""

    actuator: float
    constraint_total: float
    passive: float
    bias: float
    joint_limit_constraint: float
    non_limit_constraint: float


@dataclass(frozen=True)
class BreakdownSummary:
    """Compact summary for the force-breakdown rollout."""

    max_abs_right_tracking_error: float
    max_abs_left_tracking_error: float
    peak_error_step: int
    mean_abs_right_actuator_last_50: float
    mean_abs_left_actuator_last_50: float
    mean_abs_right_constraint_last_50: float
    mean_abs_left_constraint_last_50: float
    mean_abs_right_non_limit_constraint_last_50: float
    mean_abs_left_non_limit_constraint_last_50: float
    mean_abs_right_joint_limit_last_50: float
    mean_abs_left_joint_limit_last_50: float
    mean_abs_right_passive_last_50: float
    mean_abs_left_passive_last_50: float
    mean_abs_right_bias_last_50: float
    mean_abs_left_bias_last_50: float
    right_ctrl_saturation_ratio: float
    left_ctrl_saturation_ratio: float
    both_contact_ratio: float
    none_contact_ratio: float
    base_proxy_contact_ratio: float
    terminated_step: int | None
    likely_root_cause: str


def _contact_state(env: SedonStandingEnv) -> tuple[str, bool]:
    """Return foot contact state and base-proxy-floor flag."""
    right = False
    left = False
    base = False
    for name_a, name_b, _ in contact_pairs(env.model, env.data):
        pair = {name_a, name_b}
        if pair == {FLOOR_GEOM, RIGHT_FOOT_GEOM}:
            right = True
        elif pair == {FLOOR_GEOM, LEFT_FOOT_GEOM}:
            left = True
        elif pair == {FLOOR_GEOM, BASE_PROXY_GEOM}:
            base = True
    if right and left:
        return "both", base
    if right:
        return "right_only", base
    if left:
        return "left_only", base
    return "none", base


def _foot_floor_load(env: SedonStandingEnv, side: str) -> tuple[int, float]:
    """Return floor-contact count and summed normal load for one foot."""
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


def _ctrl_saturated(env: SedonStandingEnv, actuator_id: int, tolerance: float = 1e-6) -> bool:
    """Return whether one actuator command is clipped at its control range."""
    lower = float(env.model.actuator_ctrlrange[actuator_id, 0])
    upper = float(env.model.actuator_ctrlrange[actuator_id, 1])
    ctrl_value = float(env.data.ctrl[actuator_id])
    return bool(ctrl_value <= lower + tolerance or ctrl_value >= upper - tolerance)


def _hip_roll_target(env: SedonStandingEnv, support_side: str, hip_roll_offset: float) -> np.ndarray:
    """Return a mirrored hip-roll support target."""
    target = env._nominal_joint_qpos.copy()
    if support_side == "left":
        target[RIGHT_HIP_ROLL_INDEX] += hip_roll_offset
        target[LEFT_HIP_ROLL_INDEX] -= hip_roll_offset
        return target
    if support_side == "right":
        target[RIGHT_HIP_ROLL_INDEX] -= hip_roll_offset
        target[LEFT_HIP_ROLL_INDEX] += hip_roll_offset
        return target
    raise ValueError(f"Unsupported support side: {support_side}")


def _constraint_type_name(efc_type: int) -> str:
    """Return a stable human-readable MuJoCo constraint-type name."""
    mapping = {
        int(mujoco.mjtConstraint.mjCNSTR_EQUALITY): "equality",
        int(mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF): "friction_dof",
        int(mujoco.mjtConstraint.mjCNSTR_FRICTION_TENDON): "friction_tendon",
        int(mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT): "limit_joint",
        int(mujoco.mjtConstraint.mjCNSTR_LIMIT_TENDON): "limit_tendon",
        int(mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS): "contact_frictionless",
        int(mujoco.mjtConstraint.mjCNSTR_CONTACT_PYRAMIDAL): "contact_pyramidal",
        int(mujoco.mjtConstraint.mjCNSTR_CONTACT_ELLIPTIC): "contact_elliptic",
    }
    return mapping.get(int(efc_type), f"unknown_{int(efc_type)}")


def _is_contact_constraint(efc_type: int) -> bool:
    """Return whether a constraint type is one of MuJoCo's contact constraints."""
    return int(efc_type) in {
        int(mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS),
        int(mujoco.mjtConstraint.mjCNSTR_CONTACT_PYRAMIDAL),
        int(mujoco.mjtConstraint.mjCNSTR_CONTACT_ELLIPTIC),
    }


def _constraint_breakdown_by_dof(env: SedonStandingEnv) -> tuple[np.ndarray, np.ndarray]:
    """Return joint-limit and non-limit constraint forces by DOF.

    MuJoCo exposes total constraint generalized force in ``qfrc_constraint``.
    This helper reconstructs per-constraint-row contributions via ``J^T * f``,
    then separates joint-limit rows from all other constraints. In this tool
    the non-limit term is expected to be dominated by floor contact.
    """
    if env.data.nefc <= 0:
        zeros = np.zeros(env.model.nv, dtype=np.float64)
        return zeros, zeros

    efc_force = np.asarray(env.data.efc_force[: env.data.nefc], dtype=np.float64)
    efc_type = np.asarray(env.data.efc_type[: env.data.nefc], dtype=np.int32)
    jacobian = np.asarray(env.data.efc_J, dtype=np.float64).reshape(env.data.nefc, env.model.nv)
    limit_mask = efc_type == int(mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT)
    joint_limit = jacobian[limit_mask].T @ efc_force[limit_mask] if np.any(limit_mask) else np.zeros(env.model.nv)
    non_limit = (
        jacobian[~limit_mask].T @ efc_force[~limit_mask] if np.any(~limit_mask) else np.zeros(env.model.nv)
    )
    return np.asarray(joint_limit, dtype=np.float64), np.asarray(non_limit, dtype=np.float64)


def _foot_contact_side(geom1_name: str, geom2_name: str) -> str:
    """Return left/right side label for one foot-floor contact pair."""
    pair = {geom1_name, geom2_name}
    if pair == {FLOOR_GEOM, LEFT_FOOT_GEOM}:
        return "left"
    if pair == {FLOOR_GEOM, RIGHT_FOOT_GEOM}:
        return "right"
    return "other"


def _contact_row_breakdown(
    env: SedonStandingEnv,
    *,
    step: int,
    support_side: str,
    left_dof_adr: int,
    right_dof_adr: int,
) -> list[dict[str, object]]:
    """Return per-contact-row contributions for foot-floor constraints.

    MuJoCo stores one ``efc_id`` per constraint row. For contact rows that id
    points back to ``data.contact[contact_id]``, so we can attach row-level
    generalized-force contributions to the exact floor-foot contact point.
    """
    if env.data.nefc <= 0 or env.data.ncon <= 0:
        return []

    jacobian = np.asarray(env.data.efc_J, dtype=np.float64).reshape(env.data.nefc, env.model.nv)
    efc_force = np.asarray(env.data.efc_force[: env.data.nefc], dtype=np.float64)
    efc_type = np.asarray(env.data.efc_type[: env.data.nefc], dtype=np.int32)
    efc_id = np.asarray(env.data.efc_id[: env.data.nefc], dtype=np.int32)

    rows: list[dict[str, object]] = []
    for efc_row in range(env.data.nefc):
        row_type = int(efc_type[efc_row])
        if not _is_contact_constraint(row_type):
            continue
        contact_id = int(efc_id[efc_row])
        if contact_id < 0 or contact_id >= env.data.ncon:
            continue

        contact = env.data.contact[contact_id]
        geom1_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1))
        geom2_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2))
        side = _foot_contact_side(geom1_name, geom2_name)
        if side == "other":
            continue

        body1_id = int(env.model.geom_bodyid[int(contact.geom1)])
        body2_id = int(env.model.geom_bodyid[int(contact.geom2)])
        body1_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, body1_id)
        body2_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, body2_id)
        left_jacobian = float(jacobian[efc_row, left_dof_adr])
        right_jacobian = float(jacobian[efc_row, right_dof_adr])
        row_force = float(efc_force[efc_row])
        left_contribution = left_jacobian * row_force
        right_contribution = right_jacobian * row_force
        support_abs_contribution = abs(left_contribution) if support_side == "left" else abs(right_contribution)

        rows.append(
            {
                "step": step,
                "contact_id": contact_id,
                "geom1": geom1_name,
                "geom2": geom2_name,
                "body1": body1_name,
                "body2": body2_name,
                "contact_pos_x": float(contact.pos[0]),
                "contact_pos_y": float(contact.pos[1]),
                "contact_pos_z": float(contact.pos[2]),
                "efc_row": efc_row,
                "efc_type": _constraint_type_name(row_type),
                "efc_force": row_force,
                "jacobian_to_left_hip_roll": left_jacobian,
                "jacobian_to_right_hip_roll": right_jacobian,
                "contribution_to_left_hip_roll": left_contribution,
                "contribution_to_right_hip_roll": right_contribution,
                "is_joint_limit": False,
                "is_contact": True,
                "side": side,
                "_abs_support_contribution": support_abs_contribution,
            }
        )

    rows.sort(key=lambda row: float(row["_abs_support_contribution"]), reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank_abs_contribution"] = rank
        del row["_abs_support_contribution"]
    return rows


def _dof_breakdown(env: SedonStandingEnv, dof_index: int) -> DofForceBreakdown:
    """Return force breakdown for one generalized DOF."""
    joint_limit, non_limit = _constraint_breakdown_by_dof(env)
    return DofForceBreakdown(
        actuator=float(env.data.qfrc_actuator[dof_index]),
        constraint_total=float(env.data.qfrc_constraint[dof_index]),
        passive=float(env.data.qfrc_passive[dof_index]),
        bias=float(env.data.qfrc_bias[dof_index]),
        joint_limit_constraint=float(joint_limit[dof_index]),
        non_limit_constraint=float(non_limit[dof_index]),
    )


def _mean_abs(values: list[float]) -> float:
    """Return mean absolute value with zero fallback."""
    if not values:
        return 0.0
    return float(np.mean(np.abs(np.asarray(values, dtype=np.float64))))


def _likely_root_cause(
    *,
    mean_abs_actuator: float,
    mean_abs_non_limit_constraint: float,
    mean_abs_joint_limit: float,
    ctrl_saturation_ratio: float,
    mean_abs_tracking_error: float,
) -> str:
    """Return a conservative root-cause label."""
    if mean_abs_joint_limit > max(mean_abs_actuator * 0.5, 1.0):
        return "joint_limit_constraint_interference"
    if (
        mean_abs_non_limit_constraint > mean_abs_actuator * 1.2
        and mean_abs_tracking_error > 0.02
    ):
        return "contact_constraint_dominant"
    if ctrl_saturation_ratio > 0.5 and mean_abs_tracking_error > 0.02:
        return "actuator_authority_limited_or_pd_clipped"
    if mean_abs_tracking_error > 0.02:
        return "mixed_tracking_and_contact_suppression"
    return "no_clear_tracking_failure"


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support-side", choices=("left", "right"), default="left")
    parser.add_argument("--hip-roll-offset", type=float, default=0.06)
    parser.add_argument("--settle-steps", type=int, default=20)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--contact-row-out-csv", type=Path, default=DEFAULT_CONTACT_ROW_OUT_CSV)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the with-floor hip-roll force-breakdown diagnostic."""
    args = build_parser().parse_args(argv)
    if args.steps <= 0:
        raise ValueError("--steps must be positive.")
    if args.settle_steps < 0:
        raise ValueError("--settle-steps must be non-negative.")
    if args.print_every <= 0:
        raise ValueError("--print-every must be positive.")

    reward_config = load_sedon_config_from_env()
    env = SedonStandingEnv(reset_noise_scale=0.0, reward_config=reward_config)
    rows: list[dict[str, object]] = []
    contact_row_rows: list[dict[str, object]] = []
    contact_counts: Counter[str] = Counter()
    base_proxy_contact_steps = 0
    right_ctrl_saturated_steps = 0
    left_ctrl_saturated_steps = 0
    max_abs_right_tracking_error = 0.0
    max_abs_left_tracking_error = 0.0
    peak_error_step = 1
    peak_error_value = -1.0
    terminated_step: int | None = None

    try:
        env.reset(seed=args.seed)
        nominal_target = env._apply_safe_joint_target_clamps(env._nominal_joint_qpos.copy())
        for _ in range(args.settle_steps):
            env._do_pd_simulation(nominal_target)

        right_joint_id = env._joint_id(RIGHT_HIP_ROLL_NAME)
        left_joint_id = env._joint_id(LEFT_HIP_ROLL_NAME)
        right_dof_adr = int(env.model.jnt_dofadr[right_joint_id])
        left_dof_adr = int(env.model.jnt_dofadr[left_joint_id])
        right_actuator_id = int(
            mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, RIGHT_HIP_ROLL_ACTUATOR)
        )
        left_actuator_id = int(
            mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, LEFT_HIP_ROLL_ACTUATOR)
        )
        if right_actuator_id < 0 or left_actuator_id < 0:
            raise ValueError("Failed to resolve hip-roll actuators.")

        target = env._apply_safe_joint_target_clamps(
            _hip_roll_target(env, args.support_side, args.hip_roll_offset)
        )

        print(
            "step target_r target_l actual_r actual_l err_r err_l "
            "act_r con_r con_nonlimit_r limit_r bias_r "
            "act_l con_l con_nonlimit_l limit_l bias_l contact"
        )
        for step in range(1, args.steps + 1):
            env._do_pd_simulation(target)
            joint_positions = env._joint_positions()
            right_target = float(target[RIGHT_HIP_ROLL_INDEX])
            left_target = float(target[LEFT_HIP_ROLL_INDEX])
            right_actual = float(joint_positions[RIGHT_HIP_ROLL_INDEX])
            left_actual = float(joint_positions[LEFT_HIP_ROLL_INDEX])
            right_error = right_target - right_actual
            left_error = left_target - left_actual
            right_breakdown = _dof_breakdown(env, right_dof_adr)
            left_breakdown = _dof_breakdown(env, left_dof_adr)
            contact_state, base_proxy_contact = _contact_state(env)
            left_contact_count, left_normal_force = _foot_floor_load(env, "left")
            right_contact_count, right_normal_force = _foot_floor_load(env, "right")
            total_normal_force = left_normal_force + right_normal_force
            if total_normal_force > 1e-9:
                left_force_ratio = left_normal_force / total_normal_force
                right_force_ratio = right_normal_force / total_normal_force
            else:
                left_force_ratio = 0.0
                right_force_ratio = 0.0

            right_ctrl_saturated = _ctrl_saturated(env, right_actuator_id)
            left_ctrl_saturated = _ctrl_saturated(env, left_actuator_id)
            right_ctrl_saturated_steps += int(right_ctrl_saturated)
            left_ctrl_saturated_steps += int(left_ctrl_saturated)
            contact_counts[contact_state] += 1
            base_proxy_contact_steps += int(base_proxy_contact)

            abs_right_error = abs(right_error)
            abs_left_error = abs(left_error)
            max_abs_right_tracking_error = max(max_abs_right_tracking_error, abs_right_error)
            max_abs_left_tracking_error = max(max_abs_left_tracking_error, abs_left_error)
            combined_error = max(abs_right_error, abs_left_error)
            if combined_error > peak_error_value:
                peak_error_value = combined_error
                peak_error_step = step

            row = {
                "step": step,
                "support_side": args.support_side,
                "target_right_hip_roll": right_target,
                "target_left_hip_roll": left_target,
                "actual_right_hip_roll": right_actual,
                "actual_left_hip_roll": left_actual,
                "right_hip_roll_error": right_error,
                "left_hip_roll_error": left_error,
                "right_ctrl": float(env.data.ctrl[right_actuator_id]),
                "left_ctrl": float(env.data.ctrl[left_actuator_id]),
                "right_ctrl_saturated": right_ctrl_saturated,
                "left_ctrl_saturated": left_ctrl_saturated,
                "right_qfrc_actuator": right_breakdown.actuator,
                "right_qfrc_constraint": right_breakdown.constraint_total,
                "right_qfrc_constraint_non_limit": right_breakdown.non_limit_constraint,
                "right_joint_limit_force": right_breakdown.joint_limit_constraint,
                "right_qfrc_passive": right_breakdown.passive,
                "right_qfrc_bias": right_breakdown.bias,
                "left_qfrc_actuator": left_breakdown.actuator,
                "left_qfrc_constraint": left_breakdown.constraint_total,
                "left_qfrc_constraint_non_limit": left_breakdown.non_limit_constraint,
                "left_joint_limit_force": left_breakdown.joint_limit_constraint,
                "left_qfrc_passive": left_breakdown.passive,
                "left_qfrc_bias": left_breakdown.bias,
                "left_contact_count": left_contact_count,
                "right_contact_count": right_contact_count,
                "left_normal_force": left_normal_force,
                "right_normal_force": right_normal_force,
                "left_force_ratio": left_force_ratio,
                "right_force_ratio": right_force_ratio,
                "contact_state": contact_state,
                "base_proxy_contact": base_proxy_contact,
                "base_z": float(env._base_height()),
                "upright": float(env._base_upright()),
                "terminated": bool(env._is_terminated(env._base_height(), env._base_upright(), env._get_obs())),
            }
            rows.append(row)
            contact_row_rows.extend(
                _contact_row_breakdown(
                    env,
                    step=step,
                    support_side=args.support_side,
                    left_dof_adr=left_dof_adr,
                    right_dof_adr=right_dof_adr,
                )
            )

            if step == 1 or step % args.print_every == 0 or row["terminated"]:
                print(
                    f"{step:>4} {right_target:>8.4f} {left_target:>8.4f} "
                    f"{right_actual:>8.4f} {left_actual:>8.4f} "
                    f"{right_error:>7.4f} {left_error:>7.4f} "
                    f"{right_breakdown.actuator:>7.2f} {right_breakdown.constraint_total:>7.2f} "
                    f"{right_breakdown.non_limit_constraint:>13.2f} {right_breakdown.joint_limit_constraint:>7.2f} "
                    f"{right_breakdown.bias:>7.2f} {left_breakdown.actuator:>7.2f} "
                    f"{left_breakdown.constraint_total:>7.2f} {left_breakdown.non_limit_constraint:>13.2f} "
                    f"{left_breakdown.joint_limit_constraint:>7.2f} {left_breakdown.bias:>7.2f} "
                    f"{contact_state:>10}"
                )

            if row["terminated"]:
                terminated_step = step
                break
    finally:
        env.close()

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    args.contact_row_out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.contact_row_out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(contact_row_rows[0].keys()) if contact_row_rows else [],
        )
        if contact_row_rows:
            writer.writeheader()
            writer.writerows(contact_row_rows)

    tail = rows[-50:]
    summary = BreakdownSummary(
        max_abs_right_tracking_error=max_abs_right_tracking_error,
        max_abs_left_tracking_error=max_abs_left_tracking_error,
        peak_error_step=peak_error_step,
        mean_abs_right_actuator_last_50=_mean_abs([float(row["right_qfrc_actuator"]) for row in tail]),
        mean_abs_left_actuator_last_50=_mean_abs([float(row["left_qfrc_actuator"]) for row in tail]),
        mean_abs_right_constraint_last_50=_mean_abs([float(row["right_qfrc_constraint"]) for row in tail]),
        mean_abs_left_constraint_last_50=_mean_abs([float(row["left_qfrc_constraint"]) for row in tail]),
        mean_abs_right_non_limit_constraint_last_50=_mean_abs(
            [float(row["right_qfrc_constraint_non_limit"]) for row in tail]
        ),
        mean_abs_left_non_limit_constraint_last_50=_mean_abs(
            [float(row["left_qfrc_constraint_non_limit"]) for row in tail]
        ),
        mean_abs_right_joint_limit_last_50=_mean_abs([float(row["right_joint_limit_force"]) for row in tail]),
        mean_abs_left_joint_limit_last_50=_mean_abs([float(row["left_joint_limit_force"]) for row in tail]),
        mean_abs_right_passive_last_50=_mean_abs([float(row["right_qfrc_passive"]) for row in tail]),
        mean_abs_left_passive_last_50=_mean_abs([float(row["left_qfrc_passive"]) for row in tail]),
        mean_abs_right_bias_last_50=_mean_abs([float(row["right_qfrc_bias"]) for row in tail]),
        mean_abs_left_bias_last_50=_mean_abs([float(row["left_qfrc_bias"]) for row in tail]),
        right_ctrl_saturation_ratio=right_ctrl_saturated_steps / max(len(rows), 1),
        left_ctrl_saturation_ratio=left_ctrl_saturated_steps / max(len(rows), 1),
        both_contact_ratio=contact_counts["both"] / max(len(rows), 1),
        none_contact_ratio=contact_counts["none"] / max(len(rows), 1),
        base_proxy_contact_ratio=base_proxy_contact_steps / max(len(rows), 1),
        terminated_step=terminated_step,
        likely_root_cause=_likely_root_cause(
            mean_abs_actuator=0.5
            * (
                _mean_abs([float(row["right_qfrc_actuator"]) for row in tail])
                + _mean_abs([float(row["left_qfrc_actuator"]) for row in tail])
            ),
            mean_abs_non_limit_constraint=0.5
            * (
                _mean_abs([float(row["right_qfrc_constraint_non_limit"]) for row in tail])
                + _mean_abs([float(row["left_qfrc_constraint_non_limit"]) for row in tail])
            ),
            mean_abs_joint_limit=0.5
            * (
                _mean_abs([float(row["right_joint_limit_force"]) for row in tail])
                + _mean_abs([float(row["left_joint_limit_force"]) for row in tail])
            ),
            ctrl_saturation_ratio=0.5
            * (
                right_ctrl_saturated_steps / max(len(rows), 1)
                + left_ctrl_saturated_steps / max(len(rows), 1)
            ),
            mean_abs_tracking_error=0.5
            * (
                _mean_abs([float(row["right_hip_roll_error"]) for row in tail])
                + _mean_abs([float(row["left_hip_roll_error"]) for row in tail])
            ),
        ),
    )

    print(f"\ncsv: {args.out_csv}")
    print(f"contact_row_csv: {args.contact_row_out_csv}")
    print(f"steps: {len(rows)}")
    print(f"peak_error_step: {summary.peak_error_step}")
    print(f"max_abs_right_tracking_error: {summary.max_abs_right_tracking_error:.5f}")
    print(f"max_abs_left_tracking_error: {summary.max_abs_left_tracking_error:.5f}")
    print(f"mean_abs_right_actuator_last_50: {summary.mean_abs_right_actuator_last_50:.3f}")
    print(f"mean_abs_left_actuator_last_50: {summary.mean_abs_left_actuator_last_50:.3f}")
    print(f"mean_abs_right_constraint_last_50: {summary.mean_abs_right_constraint_last_50:.3f}")
    print(f"mean_abs_left_constraint_last_50: {summary.mean_abs_left_constraint_last_50:.3f}")
    print(
        "mean_abs_right_non_limit_constraint_last_50: "
        f"{summary.mean_abs_right_non_limit_constraint_last_50:.3f}"
    )
    print(
        "mean_abs_left_non_limit_constraint_last_50: "
        f"{summary.mean_abs_left_non_limit_constraint_last_50:.3f}"
    )
    print(f"mean_abs_right_joint_limit_last_50: {summary.mean_abs_right_joint_limit_last_50:.3f}")
    print(f"mean_abs_left_joint_limit_last_50: {summary.mean_abs_left_joint_limit_last_50:.3f}")
    print(f"mean_abs_right_passive_last_50: {summary.mean_abs_right_passive_last_50:.3f}")
    print(f"mean_abs_left_passive_last_50: {summary.mean_abs_left_passive_last_50:.3f}")
    print(f"mean_abs_right_bias_last_50: {summary.mean_abs_right_bias_last_50:.3f}")
    print(f"mean_abs_left_bias_last_50: {summary.mean_abs_left_bias_last_50:.3f}")
    print(f"right_ctrl_saturation_ratio: {summary.right_ctrl_saturation_ratio:.3f}")
    print(f"left_ctrl_saturation_ratio: {summary.left_ctrl_saturation_ratio:.3f}")
    print(f"both_contact_ratio: {summary.both_contact_ratio:.3f}")
    print(f"none_contact_ratio: {summary.none_contact_ratio:.3f}")
    print(f"base_proxy_contact_ratio: {summary.base_proxy_contact_ratio:.3f}")
    print(f"terminated_step: {summary.terminated_step}")
    print(f"likely_root_cause: {summary.likely_root_cause}")
    if contact_row_rows:
        support_key = (
            "contribution_to_left_hip_roll"
            if args.support_side == "left"
            else "contribution_to_right_hip_roll"
        )
        top_rows = sorted(
            contact_row_rows,
            key=lambda row: abs(float(row[support_key])),
            reverse=True,
        )[:10]
        print("\ntop_10_contact_rows_by_abs_support_side_contribution:")
        for index, row in enumerate(top_rows, start=1):
            print(
                f"{index:>2} step={int(row['step'])} "
                f"contact_id={int(row['contact_id'])} "
                f"efc_row={int(row['efc_row'])} "
                f"side={row['side']} "
                f"contact_y={float(row['contact_pos_y']):+.4f} "
                f"support_contrib={float(row[support_key]):+.4f} "
                f"left_contrib={float(row['contribution_to_left_hip_roll']):+.4f} "
                f"right_contrib={float(row['contribution_to_right_hip_roll']):+.4f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
