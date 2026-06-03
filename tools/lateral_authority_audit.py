"""Audit actuator authority needed to replace Seedon lateral assist."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np

from seedon_baseline.env import JOINT_NAMES, SeedonStandingEnv, load_seedon_config_from_env
from tools.seedon_debug_common import (
    DEBUG_OUT_DIR,
    DEFAULT_SCENE_PATH,
    FLOOR_GEOM,
    LEFT_FOOT_GEOM,
    RIGHT_FOOT_GEOM,
    geom_id,
    require_scene,
)


DEFAULT_OUT_CSV = DEBUG_OUT_DIR / "lateral_authority_audit.csv"
HIP_ROLL_INDEX = {"right": JOINT_NAMES.index("R_joint_hip_roll"), "left": JOINT_NAMES.index("L_joint_hip_roll")}
HIP_YAW_INDEX = {"right": JOINT_NAMES.index("R_joint_hip_yaw"), "left": JOINT_NAMES.index("L_joint_hip_yaw")}
KNEE_PITCH_INDEX = {"right": JOINT_NAMES.index("R_joint_knee_pitch"), "left": JOINT_NAMES.index("L_joint_knee_pitch")}
ANKLE_PITCH_INDEX = {"right": JOINT_NAMES.index("R_joint_ankle_pitch"), "left": JOINT_NAMES.index("L_joint_ankle_pitch")}
HIP_YAW_BODY = {"right": "R_link_hip_yaw", "left": "L_link_hip_yaw"}
BASE_BODY_NAME = "base_link"
MEDIUM_LATERAL_ASSIST_N = 10.0


@dataclass(frozen=True)
class AuthorityCase:
    """One lateral authority replacement case."""

    name: str
    lateral_force_y: float = 0.0
    hip_roll_ctrl_delta: float = 0.0
    hip_yaw_ctrl_delta: float = 0.0
    stance_knee_delta: float = 0.0
    stance_ankle_delta: float = 0.0
    pelvis_lateral_target_delta: float = 0.0


@dataclass(frozen=True)
class AuthorityAuditRow:
    """One lateral authority audit result."""

    case: str
    lateral_assist_force_y: float
    lateral_assist_torque_norm: float
    assist_body: str
    assist_position_x: float
    assist_position_y: float
    assist_position_z: float
    hip_roll_ctrl_delta: float
    hip_yaw_ctrl_delta: float
    stance_knee_delta: float
    stance_ankle_delta: float
    pelvis_lateral_target_delta: float
    peak_support_ratio: float
    min_swing_ratio: float
    total_force_norm: float
    penetration: float
    upright: float
    torque_saturation: float
    max_abs_base_roll: float
    max_abs_base_pitch: float
    max_abs_ctrl: float
    mean_abs_ctrl: float
    diagnosis: str
    score: float


def _smoothstep(phase: float) -> float:
    """Return smoothstep alpha."""
    phase = float(np.clip(phase, 0.0, 1.0))
    return phase * phase * (3.0 - 2.0 * phase)


def _apply_stance_width_scale(env: SeedonStandingEnv, scale: float) -> None:
    """Apply temporary stance-width scale in memory."""
    for side, body_name in HIP_YAW_BODY.items():
        body_id = env._body_id(body_name)
        direction = -1.0 if side == "right" else 1.0
        env.model.body_pos[body_id][1] = direction * abs(float(env.model.body_pos[body_id][1])) * scale


def _build_target(env: SeedonStandingEnv, alpha: float, case: AuthorityCase) -> np.ndarray:
    """Build the audited right-support preload target."""
    target = env._nominal_joint_qpos.copy()
    target[HIP_ROLL_INDEX["right"]] += 0.005 * alpha
    target[HIP_ROLL_INDEX["left"]] -= (0.020 + 0.020) * alpha
    target[KNEE_PITCH_INDEX["right"]] += case.stance_knee_delta * alpha
    target[ANKLE_PITCH_INDEX["right"]] += case.stance_ankle_delta * alpha
    target[HIP_ROLL_INDEX["right"]] += case.pelvis_lateral_target_delta * alpha
    target[HIP_ROLL_INDEX["left"]] -= case.pelvis_lateral_target_delta * alpha
    return target


def _contact_metrics(env: SeedonStandingEnv) -> tuple[float, float, float]:
    """Return left/right world-z forces and max penetration."""
    left = 0.0
    right = 0.0
    max_penetration = 0.0
    wrench = np.zeros(6, dtype=np.float64)
    for contact_index in range(env.data.ncon):
        contact = env.data.contact[contact_index]
        geom1 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1))
        geom2 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2))
        pair = {geom1, geom2}
        if pair not in ({FLOOR_GEOM, LEFT_FOOT_GEOM}, {FLOOR_GEOM, RIGHT_FOOT_GEOM}):
            continue
        mujoco.mj_contactForce(env.model, env.data, contact_index, wrench)
        contact_frame = np.asarray(contact.frame, dtype=np.float64).reshape(3, 3)
        world_force = contact_frame.T @ wrench[:3]
        force_z = abs(float(world_force[2]))
        max_penetration = max(max_penetration, max(0.0, -float(contact.dist)))
        if pair == {FLOOR_GEOM, LEFT_FOOT_GEOM}:
            left += force_z
        else:
            right += force_z
    return left, right, max_penetration


def _ctrl_with_case(env: SeedonStandingEnv, target: np.ndarray, case: AuthorityCase, alpha: float) -> np.ndarray:
    """Return PD ctrl with case-specific actuator authority injection."""
    ctrl = env._pd_control(target).copy()
    if case.hip_roll_ctrl_delta:
        ctrl[HIP_ROLL_INDEX["right"]] += case.hip_roll_ctrl_delta * alpha
        ctrl[HIP_ROLL_INDEX["left"]] -= case.hip_roll_ctrl_delta * alpha
    if case.hip_yaw_ctrl_delta:
        ctrl[HIP_YAW_INDEX["right"]] += case.hip_yaw_ctrl_delta * alpha
        ctrl[HIP_YAW_INDEX["left"]] -= case.hip_yaw_ctrl_delta * alpha
    return np.clip(ctrl, env._ctrl_range[:, 0], env._ctrl_range[:, 1])


def _step_case(env: SeedonStandingEnv, target: np.ndarray, case: AuthorityCase, alpha: float) -> None:
    """Advance one RL control step with optional lateral force and actuator injection."""
    base_id = env._base_body_id
    for _ in range(env.frame_skip):
        env.data.ctrl[:] = _ctrl_with_case(env, target, case, alpha)
        if case.lateral_force_y:
            env.data.xfrc_applied[base_id, 1] = case.lateral_force_y
        mujoco.mj_step(env.model, env.data)
        env.data.xfrc_applied[base_id, :] = 0.0


def _torque_saturation(env: SeedonStandingEnv) -> float:
    """Return fraction of actuators near control limits."""
    lower_margin = np.abs(env.data.ctrl - env._ctrl_range[:, 0])
    upper_margin = np.abs(env.data.ctrl - env._ctrl_range[:, 1])
    span = np.maximum(env._ctrl_range[:, 1] - env._ctrl_range[:, 0], 1e-9)
    return float(np.count_nonzero(np.minimum(lower_margin, upper_margin) <= 0.02 * span) / env.data.ctrl.size)


def _evaluate_case(scene_path: Path, case: AuthorityCase) -> AuthorityAuditRow:
    """Evaluate one authority replacement case."""
    env = SeedonStandingEnv(
        scene_path=require_scene(scene_path),
        reset_noise_scale=0.0,
        reward_config=load_seedon_config_from_env(),
    )
    _apply_stance_width_scale(env, 0.9)
    env.reset(seed=0)
    base_id = env._base_body_id
    assist_position = env.data.xpos[base_id].copy()
    steps = max(1, int(round(0.2 / float(env.model.opt.timestep))))
    total_weight = float(np.sum(env.model.body_mass) * 9.81)
    peak_support_ratio = 0.0
    min_swing_ratio = 1.0
    max_total_force_norm = 0.0
    max_penetration = 0.0
    min_upright = 1.0
    max_abs_base_roll = 0.0
    max_abs_base_pitch = 0.0
    max_abs_ctrl = 0.0
    mean_abs_ctrl_samples: list[float] = []
    max_torque_saturation = 0.0
    for step_index in range(steps):
        alpha = _smoothstep(step_index / max(1, steps - 1))
        target = _build_target(env, alpha, case)
        _step_case(env, target, case, alpha)
        left_force, right_force, penetration = _contact_metrics(env)
        total_force = left_force + right_force
        support_ratio = float(right_force / (total_force + 1e-6))
        swing_ratio = float(left_force / (total_force + 1e-6))
        peak_support_ratio = max(peak_support_ratio, support_ratio)
        min_swing_ratio = min(min_swing_ratio, swing_ratio)
        max_total_force_norm = max(max_total_force_norm, total_force / max(total_weight, 1e-6))
        max_penetration = max(max_penetration, penetration)
        min_upright = min(min_upright, float(env._base_upright()))
        max_abs_base_roll = max(max_abs_base_roll, abs(float(env._base_roll())))
        max_abs_base_pitch = max(max_abs_base_pitch, abs(float(env._base_pitch())))
        max_abs_ctrl = max(max_abs_ctrl, float(np.max(np.abs(env.data.ctrl))))
        mean_abs_ctrl_samples.append(float(np.mean(np.abs(env.data.ctrl))))
        max_torque_saturation = max(max_torque_saturation, _torque_saturation(env))
    score = max(0.0, 0.58 - peak_support_ratio)
    diagnosis = _diagnose(case, peak_support_ratio)
    return AuthorityAuditRow(
        case=case.name,
        lateral_assist_force_y=case.lateral_force_y,
        lateral_assist_torque_norm=0.0,
        assist_body=BASE_BODY_NAME,
        assist_position_x=float(assist_position[0]),
        assist_position_y=float(assist_position[1]),
        assist_position_z=float(assist_position[2]),
        hip_roll_ctrl_delta=case.hip_roll_ctrl_delta,
        hip_yaw_ctrl_delta=case.hip_yaw_ctrl_delta,
        stance_knee_delta=case.stance_knee_delta,
        stance_ankle_delta=case.stance_ankle_delta,
        pelvis_lateral_target_delta=case.pelvis_lateral_target_delta,
        peak_support_ratio=float(peak_support_ratio),
        min_swing_ratio=float(min_swing_ratio),
        total_force_norm=float(max_total_force_norm),
        penetration=float(max_penetration),
        upright=float(min_upright),
        torque_saturation=float(max_torque_saturation),
        max_abs_base_roll=float(max_abs_base_roll),
        max_abs_base_pitch=float(max_abs_base_pitch),
        max_abs_ctrl=float(max_abs_ctrl),
        mean_abs_ctrl=float(np.mean(mean_abs_ctrl_samples)) if mean_abs_ctrl_samples else 0.0,
        diagnosis=diagnosis,
        score=float(score),
    )


def _diagnose(case: AuthorityCase, peak_ratio: float) -> str:
    """Return concise interpretation for one replacement case."""
    if peak_ratio < 0.58:
        return "not_enough_authority"
    if case.hip_roll_ctrl_delta:
        return "hip_roll_can_replace_lateral_assist"
    if case.hip_yaw_ctrl_delta:
        return "hip_yaw_can_help"
    if case.stance_knee_delta or case.stance_ankle_delta:
        return "knee_ankle_reference_can_help"
    if case.pelvis_lateral_target_delta:
        return "pelvis_lateral_target_can_help"
    if case.lateral_force_y:
        return "virtual_lateral_assist_baseline"
    return "baseline_reaches_target"


def _cases() -> list[AuthorityCase]:
    """Return the requested lateral authority audit cases."""
    return [
        AuthorityCase("baseline"),
        AuthorityCase("medium_lateral_assist", lateral_force_y=-MEDIUM_LATERAL_ASSIST_N),
        AuthorityCase("hip_roll_ctrl_2p5", hip_roll_ctrl_delta=2.5),
        AuthorityCase("hip_roll_ctrl_5", hip_roll_ctrl_delta=5.0),
        AuthorityCase("hip_roll_ctrl_10", hip_roll_ctrl_delta=10.0),
        AuthorityCase("hip_roll_ctrl_15", hip_roll_ctrl_delta=15.0),
        AuthorityCase("hip_yaw_ctrl_0p5", hip_yaw_ctrl_delta=0.5),
        AuthorityCase("hip_yaw_ctrl_1", hip_yaw_ctrl_delta=1.0),
        AuthorityCase("hip_yaw_ctrl_2", hip_yaw_ctrl_delta=2.0),
        AuthorityCase("stance_knee_ankle_soft", stance_knee_delta=0.01, stance_ankle_delta=-0.005),
        AuthorityCase("stance_knee_ankle_medium", stance_knee_delta=0.02, stance_ankle_delta=-0.01),
        AuthorityCase("stance_knee_ankle_reverse", stance_knee_delta=-0.01, stance_ankle_delta=0.005),
        AuthorityCase("pelvis_lateral_target_0p005", pelvis_lateral_target_delta=0.005),
        AuthorityCase("pelvis_lateral_target_0p010", pelvis_lateral_target_delta=0.010),
        AuthorityCase("pelvis_lateral_target_0p020", pelvis_lateral_target_delta=0.020),
    ]


def _write_csv(path: Path, rows: list[AuthorityAuditRow]) -> None:
    """Write audit rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def run_audit(scene_path: Path, out_csv: Path) -> list[AuthorityAuditRow]:
    """Run lateral authority audit."""
    rows = [_evaluate_case(scene_path, case) for case in _cases()]
    _write_csv(out_csv, rows)
    rows_sorted = sorted(rows, key=lambda row: row.score)
    print(f"wrote {len(rows)} rows to {out_csv}")
    for row in rows_sorted:
        print(
            f"{row.case:<30} peak={row.peak_support_ratio:.3f} swing={row.min_swing_ratio:.3f} "
            f"force_norm={row.total_force_norm:.3f} pen={row.penetration * 1000.0:.2f}mm "
            f"upright={row.upright:.3f} sat={row.torque_saturation:.2f} "
            f"roll={row.max_abs_base_roll:.3f} pitch={row.max_abs_base_pitch:.3f} "
            f"max_ctrl={row.max_abs_ctrl:.2f} mean_ctrl={row.mean_abs_ctrl:.2f} "
            f"diagnosis={row.diagnosis}"
        )
    actuator_rows = [row for row in rows if row.case != "medium_lateral_assist" and row.peak_support_ratio >= 0.58]
    if any("hip_roll" in row.diagnosis for row in actuator_rows):
        print("judgment: hip_roll torque assist can replace medium lateral assist; tune gear/forcerange/PD/residual scale.")
    elif actuator_rows:
        print("judgment: non-hip-roll actuator/reference changes can replace lateral assist; inspect listed diagnoses.")
    else:
        print("judgment: tested actuators did not replace medium lateral assist; prefer morphology or lower-threshold shuffle reference.")
    return rows


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE_PATH)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run lateral authority audit CLI."""
    args = build_parser().parse_args(argv)
    run_audit(require_scene(args.scene), args.out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
