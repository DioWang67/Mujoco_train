"""Phase W0-DuckRef conservative scripted walking smoke test for Seedon.

This Class C diagnostic runs a conservative, deterministic MuJoCo sweep after
Pre-W0 readiness passes. It does not train PPO, does not load Open Duck ONNX
weights or raw trajectories, and does not claim Blue-like dynamic gait success.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import mujoco
import numpy as np

from seedon_baseline.env import JOINT_NAMES, SeedonStandingEnv, load_seedon_config_from_env
from tools.seedon_debug_common import DEBUG_OUT_DIR, DEFAULT_SCENE_PATH, geom_name, require_scene


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V5A_SCENE = DEBUG_OUT_DIR / "blue_like_sole_experiments_v5" / "training_scene_v5_a.xml"
DEFAULT_DUCK_SCALED_REFERENCE = DEBUG_OUT_DIR / "phase_m0_duck_morphology_audit" / "seedon_duck_scaled_gait_reference.json"
DEFAULT_JOINT_SIGN_MAPPING = DEBUG_OUT_DIR / "phase_m0_duck_morphology_audit" / "seedon_joint_sign_mapping.json"
DEFAULT_READINESS_SUMMARY = (
    DEBUG_OUT_DIR / "phase_pre_w0_duckref_readiness_check" / "phase_pre_w0_duckref_readiness_summary.json"
)
DEFAULT_OUTPUT_DIR = DEBUG_OUT_DIR / "phase_w0_duckref_conservative_smoke_test"
PROGRESS_LOG = REPO_ROOT / "docs" / "seedon_blue_like_dynamic_gait_progress_log.md"
FOOT_FORCE_THRESHOLD_N = 0.1
UNKNOWN_CONTACT_FORCE_RATIO_LIMIT = 0.25

R_HIP_ROLL, R_HIP_PITCH, R_KNEE, R_ANKLE = 1, 2, 3, 4
L_HIP_ROLL, L_HIP_PITCH, L_KNEE, L_ANKLE = 6, 7, 8, 9


@dataclass(frozen=True)
class SmokeConfig:
    """Runtime configuration for the W0 smoke test.

    Parameters:
        scene_path: MuJoCo XML scene.
        duck_scaled_reference: Phase M0 scaled reference JSON.
        joint_sign_mapping: Phase M0 joint sign mapping JSON.
        readiness_summary: Pre-W0 readiness summary JSON.
        output_dir: Directory for CSV, JSON, and report outputs.
        steps: Physics-control steps per trial.
        disable_render: Accepted for script compatibility; this tool does not render.
    """

    scene_path: Path
    duck_scaled_reference: Path
    joint_sign_mapping: Path
    readiness_summary: Path
    output_dir: Path
    steps: int
    disable_render: bool


@dataclass(frozen=True)
class TrialParams:
    """One conservative W0 trial parameter set."""

    trial_id: int
    stage: int
    target_vx: float
    gait_period: float
    clearance: float
    action_scale: float


@dataclass(frozen=True)
class TrialSummary:
    """Aggregate metrics for one W0 trial."""

    trial_id: int
    stage: int
    target_vx: float
    gait_period: float
    clearance: float
    action_scale: float
    delta_base_x: float
    average_forward_velocity: float
    base_height_mean: float
    base_height_min: float
    base_height_max: float
    base_height_range: float
    max_abs_roll: float
    max_abs_pitch: float
    fall_detected: bool
    jump_count: int
    contact_none_ratio: float
    left_contact_ratio: float
    right_contact_ratio: float
    double_support_ratio: float
    single_support_ratio: float
    flight_ratio: float
    contact_switch_count: int
    estimated_gait_period_seconds: float
    left_foot_x_range: float
    right_foot_x_range: float
    left_foot_z_range: float
    right_foot_z_range: float
    approximate_left_clearance: float
    approximate_right_clearance: float
    left_swing_contact_force_mean: float
    right_swing_contact_force_mean: float
    left_unload_ratio_mean: float
    right_unload_ratio_mean: float
    left_slip_distance_while_contact: float
    right_slip_distance_while_contact: float
    support_force_ratio_peak: float
    support_force_ratio_mean: float
    action_abs_max: float
    action_saturation_ratio: float
    unknown_contact_force_ratio_mean: float
    toe_handoff_detected: bool
    center_contact_observed: bool
    gait_candidate_label: str
    failure_mode: str


def default_scene_path() -> Path:
    """Return preferred generated v5_a scene, otherwise the standard scene."""

    return DEFAULT_V5A_SCENE if DEFAULT_V5A_SCENE.is_file() else DEFAULT_SCENE_PATH


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON file with a useful missing-file error."""

    if not path.is_file():
        raise FileNotFoundError(f"Required JSON input not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def csv_write(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """Write CSV rows with a stable header."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def side_for_geom(name: str) -> str:
    """Infer left/right side from a Seedon foot geom name."""

    if name.startswith("R_") or name.lower().startswith("right"):
        return "right"
    if name.startswith("L_") or name.lower().startswith("left"):
        return "left"
    return "unknown"


def is_floor_name(name: str) -> bool:
    """Return whether a geom name should be treated as floor/ground."""

    lowered = name.lower()
    return lowered == "floor" or "floor" in lowered or "ground" in lowered


def is_foot_geom_name(name: str) -> bool:
    """Return whether a geom name should be treated as foot-related."""

    lowered = name.lower()
    return any(token in lowered for token in ("foot", "toe", "heel", "center", "sole", "bottom", "rocker"))


def region_for_geom(name: str, local_x: float | None) -> str:
    """Classify a foot contact region."""

    lowered = name.lower()
    if "toe" in lowered:
        return "toe"
    if "heel" in lowered:
        return "heel"
    if "center" in lowered or lowered in {"r_foot_collision", "l_foot_collision"}:
        return "center"
    if local_x is not None:
        if local_x > 0.055:
            return "toe"
        if local_x < -0.015:
            return "heel"
        return "center"
    return "unknown"


def quat_to_euler_wxyz(quat: np.ndarray) -> tuple[float, float, float]:
    """Convert a wxyz quaternion to roll, pitch, yaw."""

    w, x, y, z = [float(value) for value in quat]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def foot_contact_forces(env: SeedonStandingEnv) -> dict[str, float | bool | int]:
    """Return side/region contact forces and unknown-force ratio inputs."""

    forces: dict[str, float | bool | int] = {
        "left_force": 0.0,
        "right_force": 0.0,
        "left_center_force": 0.0,
        "left_toe_force": 0.0,
        "left_heel_force": 0.0,
        "right_center_force": 0.0,
        "right_toe_force": 0.0,
        "right_heel_force": 0.0,
        "unknown_left_foot_force": 0.0,
        "unknown_right_foot_force": 0.0,
        "left_center_contact": False,
        "right_center_contact": False,
        "left_toe_contact": False,
        "right_toe_contact": False,
        "contact_count": 0,
    }
    wrench = np.zeros(6, dtype=np.float64)
    for contact_index in range(env.data.ncon):
        contact = env.data.contact[contact_index]
        name_a = geom_name(env.model, int(contact.geom1))
        name_b = geom_name(env.model, int(contact.geom2))
        if not (is_floor_name(name_a) or is_floor_name(name_b)):
            continue
        foot_name = name_b if is_floor_name(name_a) else name_a
        if not is_foot_geom_name(foot_name):
            continue
        foot_geom_id = int(contact.geom2 if is_floor_name(name_a) else contact.geom1)
        foot_body_id = int(env.model.geom_bodyid[foot_geom_id])
        body_xmat = env.data.xmat[foot_body_id].reshape(3, 3)
        local_pos = body_xmat.T @ (np.asarray(contact.pos, dtype=np.float64) - env.data.xpos[foot_body_id])
        side = side_for_geom(foot_name)
        region = region_for_geom(foot_name, float(local_pos[0]))
        mujoco.mj_contactForce(env.model, env.data, contact_index, wrench)
        normal_force = abs(float(wrench[0]))
        if side not in {"left", "right"}:
            continue
        forces[f"{side}_force"] = float(forces[f"{side}_force"]) + normal_force
        if region in {"center", "toe", "heel"}:
            forces[f"{side}_{region}_force"] = float(forces[f"{side}_{region}_force"]) + normal_force
        else:
            forces[f"unknown_{side}_foot_force"] = float(forces[f"unknown_{side}_foot_force"]) + normal_force
        if region == "center":
            forces[f"{side}_center_contact"] = True
        if region == "toe":
            forces[f"{side}_toe_contact"] = True
        forces["contact_count"] = int(forces["contact_count"]) + 1
    return forces


def conservative_trial_grid() -> list[TrialParams]:
    """Build a staged 21-trial conservative W0 grid."""

    trials: list[TrialParams] = []
    trial_id = 1
    for clearance in (0.005, 0.015, 0.025):
        for action_scale in (0.10, 0.15, 0.20):
            trials.append(TrialParams(trial_id, 1, 0.05, 0.85, clearance, action_scale))
            trial_id += 1
    for target_vx in (0.08, 0.10):
        for clearance in (0.005, 0.015, 0.025):
            trials.append(TrialParams(trial_id, 2, target_vx, 0.85, clearance, 0.15))
            trial_id += 1
    for target_vx in (0.05, 0.08, 0.10):
        for action_scale in (0.15, 0.20):
            trials.append(TrialParams(trial_id, 3, target_vx, 0.70, 0.015, action_scale))
            trial_id += 1
    return trials


def validate_readiness(readiness: dict[str, Any]) -> None:
    """Fail fast unless the Pre-W0 readiness label allows conservative W0."""

    label = str(readiness.get("readiness_label", ""))
    if label != "READY_FOR_W0_DUCKREF_CONSERVATIVE":
        raise RuntimeError(
            "W0 smoke test is blocked: readiness_label must be "
            f"READY_FOR_W0_DUCKREF_CONSERVATIVE, got {label!r}"
        )


def action_from_phase(params: TrialParams, time_value: float) -> tuple[np.ndarray, float, float, str, str, float, float]:
    """Return normalized action offsets from a conservative phase oscillator."""

    phase = 2.0 * math.pi * time_value / max(params.gait_period, 1e-6)
    left_phase = phase
    right_phase = phase + math.pi
    left_swing_alpha = max(0.0, math.sin(left_phase))
    right_swing_alpha = max(0.0, math.sin(right_phase))
    if left_swing_alpha >= right_swing_alpha:
        swing_side = "left"
        support_side = "right"
    else:
        swing_side = "right"
        support_side = "left"

    action = np.zeros(len(JOINT_NAMES), dtype=np.float64)
    vx_gain = 1.0 + 2.0 * max(params.target_vx - 0.05, 0.0)
    swing_pitch = -params.action_scale * vx_gain
    swing_knee = -min(0.28, params.action_scale + params.clearance * 4.0)
    swing_ankle = -min(0.22, 0.75 * params.action_scale + params.clearance * 2.5)
    stance_pitch = 0.25 * params.action_scale
    support_roll = 0.55 * params.action_scale

    if swing_side == "right":
        swing_alpha = right_swing_alpha
        action[R_HIP_PITCH] += swing_pitch * swing_alpha
        action[R_KNEE] += swing_knee * swing_alpha
        action[R_ANKLE] += swing_ankle * swing_alpha
        action[L_HIP_PITCH] += stance_pitch * swing_alpha
        action[R_HIP_ROLL] += support_roll * swing_alpha
        action[L_HIP_ROLL] += -support_roll * swing_alpha
    else:
        swing_alpha = left_swing_alpha
        action[L_HIP_PITCH] += swing_pitch * swing_alpha
        action[L_KNEE] += swing_knee * swing_alpha
        action[L_ANKLE] += swing_ankle * swing_alpha
        action[R_HIP_PITCH] += stance_pitch * swing_alpha
        action[R_HIP_ROLL] += -support_roll * swing_alpha
        action[L_HIP_ROLL] += support_roll * swing_alpha
    return action, phase, left_phase, support_side, swing_side, left_swing_alpha, right_swing_alpha


def ctrl_saturation_ratio(ctrl: np.ndarray, ctrl_range: np.ndarray) -> float:
    """Return fraction of controls near actuator limits."""

    span = np.maximum(ctrl_range[:, 1] - ctrl_range[:, 0], 1e-9)
    margin = np.minimum(np.abs(ctrl - ctrl_range[:, 0]), np.abs(ctrl - ctrl_range[:, 1]))
    return float(np.count_nonzero(margin <= 0.02 * span) / max(ctrl.size, 1))


def foot_positions(env: SeedonStandingEnv) -> dict[str, tuple[float, float, float]]:
    """Return right/left main foot collision geom positions."""

    right = env.data.geom_xpos[env._foot_geom_ids[0]]
    left = env.data.geom_xpos[env._foot_geom_ids[1]]
    return {
        "right": (float(right[0]), float(right[1]), float(right[2] - env.model.geom_size[env._foot_geom_ids[0]][2])),
        "left": (float(left[0]), float(left[1]), float(left[2] - env.model.geom_size[env._foot_geom_ids[1]][2])),
    }


def contact_state(left_contact: bool, right_contact: bool) -> str:
    """Return none/left/right/both contact state."""

    if left_contact and right_contact:
        return "both"
    if left_contact:
        return "left"
    if right_contact:
        return "right"
    return "none"


def run_trial(config: SmokeConfig, params: TrialParams) -> tuple[list[dict[str, Any]], TrialSummary]:
    """Run one W0 trial and return timeline rows plus summary."""

    env = SeedonStandingEnv(scene_path=config.scene_path, reset_noise_scale=0.0, reward_config=load_seedon_config_from_env())
    rows: list[dict[str, Any]] = []
    try:
        env.reset(seed=1000 + params.trial_id)
        initial_x = float(env.data.qpos[0])
        robot_weight = float(np.sum(env.model.body_mass) * 9.81)
        prev_contact_state = "unknown"
        prev_left_x: float | None = None
        prev_right_x: float | None = None
        left_slip = 0.0
        right_slip = 0.0
        for step in range(1, config.steps + 1):
            action, phase, left_phase, support_side, swing_side, left_alpha, right_alpha = action_from_phase(
                params,
                float(env.data.time),
            )
            target = env._nominal_joint_qpos + action
            ctrl = env._pd_control(target)
            env.data.ctrl[:] = ctrl
            mujoco.mj_step(env.model, env.data)
            env._gait_step += 1

            positions = foot_positions(env)
            forces = foot_contact_forces(env)
            left_force = float(forces["left_force"])
            right_force = float(forces["right_force"])
            left_contact = left_force > FOOT_FORCE_THRESHOLD_N
            right_contact = right_force > FOOT_FORCE_THRESHOLD_N
            state = contact_state(left_contact, right_contact)
            if prev_left_x is not None and left_contact:
                left_slip += abs(positions["left"][0] - prev_left_x)
            if prev_right_x is not None and right_contact:
                right_slip += abs(positions["right"][0] - prev_right_x)
            prev_left_x = positions["left"][0]
            prev_right_x = positions["right"][0]

            total_force = left_force + right_force
            support_force = left_force if support_side == "left" else right_force
            swing_force = right_force if support_side == "left" else left_force
            support_ratio = support_force / max(total_force, 1e-9)
            left_unload = 1.0 - left_force / max(total_force, 1e-9)
            right_unload = 1.0 - right_force / max(total_force, 1e-9)
            unknown_force = float(forces["unknown_left_foot_force"]) + float(forces["unknown_right_foot_force"])
            unknown_ratio = unknown_force / max(
                unknown_force
                + float(forces["left_center_force"])
                + float(forces["left_toe_force"])
                + float(forces["left_heel_force"])
                + float(forces["right_center_force"])
                + float(forces["right_toe_force"])
                + float(forces["right_heel_force"]),
                1e-9,
            )
            roll, pitch, yaw = quat_to_euler_wxyz(np.asarray(env.data.qpos[3:7], dtype=np.float64))
            contact_switch = prev_contact_state not in {"unknown", state}
            prev_contact_state = state
            jump = bool(state == "none" or (float(env.data.qvel[2]) > 0.05 and total_force < 0.10 * robot_weight))
            rows.append(
                {
                    "trial_id": params.trial_id,
                    "stage": params.stage,
                    "target_vx": params.target_vx,
                    "gait_period": params.gait_period,
                    "clearance": params.clearance,
                    "action_scale": params.action_scale,
                    "step": step,
                    "time": float(env.data.time),
                    "phase": phase,
                    "intended_left_phase": left_phase,
                    "intended_right_phase": left_phase + math.pi,
                    "intended_support_side": support_side,
                    "intended_swing_side": swing_side,
                    "base_x": float(env.data.qpos[0]),
                    "base_y": float(env.data.qpos[1]),
                    "base_z": float(env.data.qpos[2]),
                    "base_roll": roll,
                    "base_pitch": pitch,
                    "base_yaw": yaw,
                    "base_vx": float(env.data.qvel[0]),
                    "base_vy": float(env.data.qvel[1]),
                    "base_vz": float(env.data.qvel[2]),
                    "left_foot_x": positions["left"][0],
                    "left_foot_y": positions["left"][1],
                    "left_foot_z": positions["left"][2],
                    "right_foot_x": positions["right"][0],
                    "right_foot_y": positions["right"][1],
                    "right_foot_z": positions["right"][2],
                    "left_contact": left_contact,
                    "right_contact": right_contact,
                    "left_contact_force": left_force,
                    "right_contact_force": right_force,
                    "left_center_force": float(forces["left_center_force"]),
                    "left_toe_force": float(forces["left_toe_force"]),
                    "left_heel_force": float(forces["left_heel_force"]),
                    "right_center_force": float(forces["right_center_force"]),
                    "right_toe_force": float(forces["right_toe_force"]),
                    "right_heel_force": float(forces["right_heel_force"]),
                    "support_force_ratio": support_ratio,
                    "double_support": state == "both",
                    "single_support": state in {"left", "right"},
                    "flight": state == "none",
                    "left_swing_contact_force": left_force if swing_side == "left" else 0.0,
                    "right_swing_contact_force": right_force if swing_side == "right" else 0.0,
                    "left_unload_ratio": left_unload,
                    "right_unload_ratio": right_unload,
                    "left_slip_distance_while_contact": left_slip,
                    "right_slip_distance_while_contact": right_slip,
                    "unknown_contact_force_ratio": unknown_ratio,
                    "action_abs_max": float(np.max(np.abs(action))),
                    "action_saturation": ctrl_saturation_ratio(ctrl, env._ctrl_range),
                    "center_contact_observed": bool(forces["left_center_contact"] or forces["right_center_contact"]),
                    "toe_handoff_detected": False,
                    "contact_state": state,
                    "contact_switch": contact_switch,
                    "jump": jump,
                    "left_swing_alpha": left_alpha,
                    "right_swing_alpha": right_alpha,
                }
            )
    finally:
        env.close()
    mark_toe_handoffs(rows)
    return rows, summarize_trial(rows, params)


def mark_toe_handoffs(rows: list[dict[str, Any]]) -> None:
    """Mark rows where a side has observed center contact before toe contact."""

    saw_center = {"left": False, "right": False}
    for row in rows:
        for side in ("left", "right"):
            if float(row[f"{side}_center_force"]) > FOOT_FORCE_THRESHOLD_N:
                saw_center[side] = True
            if saw_center[side] and float(row[f"{side}_toe_force"]) > FOOT_FORCE_THRESHOLD_N:
                row["toe_handoff_detected"] = True
                saw_center[side] = False


def count_bursts(rows: list[dict[str, Any]], key: str) -> int:
    """Count contiguous truthy bursts for a row key."""

    count = 0
    active = False
    for row in rows:
        value = bool(row[key])
        if value and not active:
            count += 1
            active = True
        elif not value:
            active = False
    return count


def summarize_trial(rows: list[dict[str, Any]], params: TrialParams) -> TrialSummary:
    """Build one trial summary from timeline rows."""

    if not rows:
        return TrialSummary(
            trial_id=params.trial_id,
            stage=params.stage,
            target_vx=params.target_vx,
            gait_period=params.gait_period,
            clearance=params.clearance,
            action_scale=params.action_scale,
            delta_base_x=0.0,
            average_forward_velocity=0.0,
            base_height_mean=0.0,
            base_height_min=0.0,
            base_height_max=0.0,
            base_height_range=0.0,
            max_abs_roll=0.0,
            max_abs_pitch=0.0,
            fall_detected=True,
            jump_count=0,
            contact_none_ratio=1.0,
            left_contact_ratio=0.0,
            right_contact_ratio=0.0,
            double_support_ratio=0.0,
            single_support_ratio=0.0,
            flight_ratio=1.0,
            contact_switch_count=0,
            estimated_gait_period_seconds=params.gait_period,
            left_foot_x_range=0.0,
            right_foot_x_range=0.0,
            left_foot_z_range=0.0,
            right_foot_z_range=0.0,
            approximate_left_clearance=0.0,
            approximate_right_clearance=0.0,
            left_swing_contact_force_mean=0.0,
            right_swing_contact_force_mean=0.0,
            left_unload_ratio_mean=0.0,
            right_unload_ratio_mean=0.0,
            left_slip_distance_while_contact=0.0,
            right_slip_distance_while_contact=0.0,
            support_force_ratio_peak=0.0,
            support_force_ratio_mean=0.0,
            action_abs_max=0.0,
            action_saturation_ratio=0.0,
            unknown_contact_force_ratio_mean=1.0,
            toe_handoff_detected=False,
            center_contact_observed=False,
            gait_candidate_label="NO_MEANINGFUL_MOTION",
            failure_mode="empty_rollout",
        )
    n = len(rows)
    base_z = [float(row["base_z"]) for row in rows]
    left_x = [float(row["left_foot_x"]) for row in rows]
    right_x = [float(row["right_foot_x"]) for row in rows]
    left_z = [float(row["left_foot_z"]) for row in rows]
    right_z = [float(row["right_foot_z"]) for row in rows]
    delta_base_x = float(rows[-1]["base_x"]) - float(rows[0]["base_x"])
    duration = max(float(rows[-1]["time"]) - float(rows[0]["time"]), 1e-9)
    summary_base = {
        "delta_base_x": delta_base_x,
        "average_forward_velocity": delta_base_x / duration,
        "base_height_mean": float(np.mean(base_z)),
        "base_height_min": min(base_z),
        "base_height_max": max(base_z),
        "base_height_range": max(base_z) - min(base_z),
        "max_abs_roll": max(abs(float(row["base_roll"])) for row in rows),
        "max_abs_pitch": max(abs(float(row["base_pitch"])) for row in rows),
        "fall_detected": min(base_z) < 0.25 or max(abs(float(row["base_roll"])) for row in rows) > 0.45,
        "jump_count": count_bursts(rows, "jump"),
        "contact_none_ratio": sum(bool(row["flight"]) for row in rows) / n,
        "left_contact_ratio": sum(bool(row["left_contact"]) for row in rows) / n,
        "right_contact_ratio": sum(bool(row["right_contact"]) for row in rows) / n,
        "double_support_ratio": sum(bool(row["double_support"]) for row in rows) / n,
        "single_support_ratio": sum(bool(row["single_support"]) for row in rows) / n,
        "flight_ratio": sum(bool(row["flight"]) for row in rows) / n,
        "contact_switch_count": sum(bool(row["contact_switch"]) for row in rows),
        "estimated_gait_period_seconds": params.gait_period,
        "left_foot_x_range": max(left_x) - min(left_x),
        "right_foot_x_range": max(right_x) - min(right_x),
        "left_foot_z_range": max(left_z) - min(left_z),
        "right_foot_z_range": max(right_z) - min(right_z),
        "approximate_left_clearance": max(left_z) - float(np.percentile(left_z, 10)),
        "approximate_right_clearance": max(right_z) - float(np.percentile(right_z, 10)),
        "left_swing_contact_force_mean": float(np.mean([float(row["left_swing_contact_force"]) for row in rows])),
        "right_swing_contact_force_mean": float(np.mean([float(row["right_swing_contact_force"]) for row in rows])),
        "left_unload_ratio_mean": float(np.mean([float(row["left_unload_ratio"]) for row in rows])),
        "right_unload_ratio_mean": float(np.mean([float(row["right_unload_ratio"]) for row in rows])),
        "left_slip_distance_while_contact": float(rows[-1]["left_slip_distance_while_contact"]),
        "right_slip_distance_while_contact": float(rows[-1]["right_slip_distance_while_contact"]),
        "support_force_ratio_peak": max(float(row["support_force_ratio"]) for row in rows),
        "support_force_ratio_mean": float(np.mean([float(row["support_force_ratio"]) for row in rows])),
        "action_abs_max": max(float(row["action_abs_max"]) for row in rows),
        "action_saturation_ratio": float(np.mean([float(row["action_saturation"]) for row in rows])),
        "unknown_contact_force_ratio_mean": float(np.mean([float(row["unknown_contact_force_ratio"]) for row in rows])),
        "toe_handoff_detected": any(bool(row["toe_handoff_detected"]) for row in rows),
        "center_contact_observed": any(bool(row["center_contact_observed"]) for row in rows),
    }
    label, failure = classify_trial(summary_base)
    return TrialSummary(
        trial_id=params.trial_id,
        stage=params.stage,
        target_vx=params.target_vx,
        gait_period=params.gait_period,
        clearance=params.clearance,
        action_scale=params.action_scale,
        gait_candidate_label=label,
        failure_mode=failure,
        **summary_base,
    )


def classify_trial(metrics: dict[str, Any]) -> tuple[str, str]:
    """Classify one W0 trial using conservative gates."""

    if metrics["unknown_contact_force_ratio_mean"] > UNKNOWN_CONTACT_FORCE_RATIO_LIMIT:
        return "CONTACT_CLASSIFICATION_UNRELIABLE", "unknown_contact_force_ratio_high"
    if metrics["fall_detected"]:
        return "POSTURE_INSTABILITY", "fall_or_large_tilt"
    if metrics["jump_count"] > 0 or metrics["flight_ratio"] > 0.05:
        return "UNSTABLE_JUMP", "jump_or_flight"
    if metrics["action_saturation_ratio"] >= 0.05:
        return "ACTION_SATURATED", "action_saturation"
    partial_unload = max(metrics["left_unload_ratio_mean"], metrics["right_unload_ratio_mean"]) > 0.42
    excessive_slip = max(metrics["left_slip_distance_while_contact"], metrics["right_slip_distance_while_contact"]) > 0.20
    if (
        metrics["contact_none_ratio"] < 0.02
        and metrics["delta_base_x"] > 0.05
        and 0.02 <= metrics["average_forward_velocity"] <= 0.12
        and metrics["contact_switch_count"] > 2
        and metrics["single_support_ratio"] > 0.10
        and metrics["max_abs_roll"] < 0.12
        and metrics["max_abs_pitch"] < 0.12
        and partial_unload
        and not excessive_slip
    ):
        return "DUCKREF_STABLE_GAIT_CANDIDATE", "none"
    periodic_foot = metrics["left_foot_x_range"] > 0.01 or metrics["right_foot_x_range"] > 0.01
    if (
        metrics["delta_base_x"] > 0.03
        and metrics["contact_switch_count"] > 2
        and periodic_foot
        and metrics["action_saturation_ratio"] < 0.10
        and partial_unload
    ):
        return "DUCKREF_LOW_SPEED_WADDLE_CANDIDATE", "low_speed_waddle"
    if metrics["delta_base_x"] > 0.03 and metrics["double_support_ratio"] > 0.85 and metrics["single_support_ratio"] < 0.10:
        return "GROUNDED_SHUFFLE", "double_support_dominant"
    if periodic_foot and max(metrics["left_swing_contact_force_mean"], metrics["right_swing_contact_force_mean"]) > 10.0:
        return "CONTACT_DRAGGING", "swing_contact_force_high"
    if metrics["support_force_ratio_peak"] < 0.58:
        return "NO_LOAD_TRANSFER", "support_force_ratio_low"
    if metrics["left_foot_x_range"] < 0.005 and metrics["right_foot_x_range"] < 0.005:
        return "NO_FOOT_ADVANCEMENT", "foot_x_range_low"
    if metrics["action_abs_max"] < 0.05:
        return "ACTION_LIMITED", "action_too_small"
    if abs(metrics["delta_base_x"]) < 0.01:
        return "NO_MEANINGFUL_MOTION", "base_delta_low"
    return "CONTACT_DRAGGING", "residual_contact_drag_or_unclassified"


def summary_score(summary: TrialSummary) -> tuple[int, float, float]:
    """Return sorting score for best-trial selection."""

    label_rank = {
        "DUCKREF_STABLE_GAIT_CANDIDATE": 6,
        "DUCKREF_LOW_SPEED_WADDLE_CANDIDATE": 5,
        "GROUNDED_SHUFFLE": 4,
        "CONTACT_DRAGGING": 3,
        "NO_LOAD_TRANSFER": 2,
        "NO_FOOT_ADVANCEMENT": 1,
        "NO_MEANINGFUL_MOTION": 0,
        "ACTION_LIMITED": 0,
        "ACTION_SATURATED": -1,
        "CONTACT_CLASSIFICATION_UNRELIABLE": -2,
        "POSTURE_INSTABILITY": -3,
        "FORWARD_FALL": -3,
        "UNSTABLE_JUMP": -4,
    }.get(summary.gait_candidate_label, -1)
    stability_score = (
        -float(summary.fall_detected)
        - summary.flight_ratio
        - summary.max_abs_roll
        - summary.max_abs_pitch
        - 0.01 * summary.jump_count
    )
    return (label_rank, stability_score, summary.delta_base_x)


def output_recommendation(best: TrialSummary, readiness: dict[str, Any]) -> str:
    """Return next recommendation from best W0 result."""

    if best.gait_candidate_label == "DUCKREF_STABLE_GAIT_CANDIDATE":
        return "PROCEED_TO_W1_SHORT_PPO_SMOKE_TEST"
    if best.gait_candidate_label == "DUCKREF_LOW_SPEED_WADDLE_CANDIDATE":
        return "RUN_SIMPLIFIED_FOOT_BOTTOM_COMPARISON"
    if best.gait_candidate_label == "CONTACT_CLASSIFICATION_UNRELIABLE":
        return "FIX_CONTACT_CLASSIFICATION"
    if best.gait_candidate_label in {"POSTURE_INSTABILITY", "FORWARD_FALL", "UNSTABLE_JUMP"}:
        return "FIX_POSTURE_STABILITY"
    if best.gait_candidate_label in {"NO_FOOT_ADVANCEMENT", "NO_LOAD_TRANSFER", "ACTION_LIMITED", "ACTION_SATURATED"}:
        return "FIX_CONTROL_MAPPING"
    if readiness.get("split_contact_style_risk") is True:
        return "RUN_SIMPLIFIED_FOOT_BOTTOM_COMPARISON"
    return "NO_GO"


def write_report(path: Path, summary_payload: dict[str, Any], summaries: list[TrialSummary]) -> None:
    """Write W0 Markdown report."""

    best = summary_payload["best_trial_metrics"]
    lines = [
        "# Phase W0-DuckRef Conservative Scripted Walking Smoke Test",
        "",
        "## A. Goal",
        "",
        "Run a conservative scripted smoke test using only gait-level DuckRef ranges.",
        "",
        "## B. Why This Is Not PPO / Not Walking Success",
        "",
        "This tool does not train, does not load Duck ONNX weights or raw trajectories, and does not claim Blue-like dynamic gait success.",
        "",
        "## C. Readiness Summary Used",
        "",
        f"- readiness_label_used: `{summary_payload['readiness_label_used']}`",
        "",
        "## D. Conservative Grid Used",
        "",
        f"- total_trials: `{summary_payload['total_trials']}`",
        f"- grid: `{summary_payload['conservative_grid_used']}`",
        "",
        "## E. Best Trial Table",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| best_trial_id | {summary_payload['best_trial_id']} |",
        f"| target_vx | {best['target_vx']:.6f} |",
        f"| gait_period | {best['gait_period']:.6f} |",
        f"| clearance | {best['clearance']:.6f} |",
        f"| action_scale | {best['action_scale']:.6f} |",
        f"| delta_base_x | {best['delta_base_x']:.6f} |",
        f"| average_forward_velocity | {best['average_forward_velocity']:.6f} |",
        f"| single_support_ratio | {best['single_support_ratio']:.6f} |",
        f"| support_force_ratio_peak | {best['support_force_ratio_peak']:.6f} |",
        f"| action_saturation_ratio | {best['action_saturation_ratio']:.6f} |",
        "",
        "## F. Gait Candidate Classification",
        "",
        f"- best_gait_candidate_label: `{summary_payload['best_gait_candidate_label']}`",
        f"- best_failure_mode: `{summary_payload['best_failure_mode']}`",
        "",
        "## G. Failure Mode Analysis",
        "",
        *(f"- trial {item.trial_id}: `{item.gait_candidate_label}` / `{item.failure_mode}`" for item in summaries[:8]),
        "",
        "## H. Contact Dragging / Unload Analysis",
        "",
        f"- left_unload_ratio_mean: `{best['left_unload_ratio_mean']:.6f}`",
        f"- right_unload_ratio_mean: `{best['right_unload_ratio_mean']:.6f}`",
        f"- left_slip_distance_while_contact: `{best['left_slip_distance_while_contact']:.6f}`",
        f"- right_slip_distance_while_contact: `{best['right_slip_distance_while_contact']:.6f}`",
        "",
        "## I. Whether Split Contact Style Appears To Block Gait",
        "",
        f"- unknown_contact_force_ratio_mean: `{best['unknown_contact_force_ratio_mean']:.6f}`",
        f"- center_contact_observed: `{str(best['center_contact_observed']).lower()}`",
        "",
        "## J. Whether Simplified Foot Bottom Comparison Is Recommended",
        "",
        f"- recommendation: `{summary_payload['recommendation']}`",
        "",
        "## K. Whether W1 Short PPO / Imitation Smoke Test Is Justified",
        "",
        "Only justified if the recommendation is `PROCEED_TO_W1_SHORT_PPO_SMOKE_TEST`.",
        "",
        "## L. Next Decision",
        "",
        summary_payload["next_decision"],
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_progress_log(path: Path, summary_payload: dict[str, Any]) -> None:
    """Append W0 result to the Seedon progress log."""

    header = "## 2026-06-01 - Phase W0-DuckRef Conservative Scripted Walking Smoke Test"
    best = summary_payload["best_trial_metrics"]
    result = (
        summary_payload["best_gait_candidate_label"]
        if summary_payload["best_gait_candidate_label"]
        in {"DUCKREF_STABLE_GAIT_CANDIDATE", "DUCKREF_LOW_SPEED_WADDLE_CANDIDATE"}
        else "FAIL"
    )
    lines = [
        "",
        header,
        "",
        "### Goal",
        "",
        "Run a conservative scripted W0 smoke test after Pre-W0 readiness, without training PPO or claiming Blue-like dynamic gait success.",
        "",
        "### Files",
        "",
        "- `tools/seedon_phase_w0_duckref_conservative_smoke_test.py`",
        "- `artifacts/seedon_debug/phase_w0_duckref_conservative_smoke_test/phase_w0_duckref_trials.csv`",
        "- `artifacts/seedon_debug/phase_w0_duckref_conservative_smoke_test/phase_w0_duckref_best_timeline.csv`",
        "- `artifacts/seedon_debug/phase_w0_duckref_conservative_smoke_test/phase_w0_duckref_summary.json`",
        "- `artifacts/seedon_debug/phase_w0_duckref_conservative_smoke_test/phase_w0_duckref_report.md`",
        "",
        "### Commands",
        "",
        "```text",
        "python -m py_compile tools/seedon_phase_w0_duckref_conservative_smoke_test.py",
        ".venv\\Scripts\\python.exe -m tools.seedon_phase_w0_duckref_conservative_smoke_test",
        "```",
        "",
        "### Key Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| total_trials | {summary_payload['total_trials']} |",
        f"| best_trial_id | {summary_payload['best_trial_id']} |",
        f"| delta_base_x | {best['delta_base_x']:.6f} |",
        f"| average_forward_velocity | {best['average_forward_velocity']:.6f} |",
        f"| contact_switch_count | {best['contact_switch_count']} |",
        f"| single_support_ratio | {best['single_support_ratio']:.6f} |",
        f"| support_force_ratio_peak | {best['support_force_ratio_peak']:.6f} |",
        f"| action_saturation_ratio | {best['action_saturation_ratio']:.6f} |",
        f"| gait_candidate_label | {summary_payload['best_gait_candidate_label']} |",
        "",
        "### Result",
        "",
        result,
        "",
        "### Engineering Interpretation",
        "",
        summary_payload["engineering_interpretation"],
        "",
        "This is not Blue-like dynamic gait success.",
        "",
        "### Next Decision",
        "",
        summary_payload["next_decision"],
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8") if path.is_file() else ""
    if header in existing:
        existing = existing[: existing.index(header)].rstrip() + "\n"
    path.write_text(existing + "\n".join(lines) + "\n", encoding="utf-8")


def run_smoke_test(config: SmokeConfig) -> dict[str, Any]:
    """Run all W0 trials and write outputs."""

    readiness = read_json(config.readiness_summary)
    validate_readiness(readiness)
    read_json(config.duck_scaled_reference)
    read_json(config.joint_sign_mapping)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    timeline_rows: list[dict[str, Any]] = []
    summaries: list[TrialSummary] = []
    for params in conservative_trial_grid():
        rows, summary = run_trial(config, params)
        timeline_rows.extend(rows)
        summaries.append(summary)

    best = max(summaries, key=summary_score)
    best_rows = [row for row in timeline_rows if int(row["trial_id"]) == best.trial_id]
    recommendation = output_recommendation(best, readiness)
    if recommendation == "PROCEED_TO_W1_SHORT_PPO_SMOKE_TEST":
        next_decision = "Proceed only to W1 short PPO / imitation smoke test with strict gates."
    elif recommendation == "RUN_SIMPLIFIED_FOOT_BOTTOM_COMPARISON":
        next_decision = "Run simplified foot_bottom_collision comparison before PPO."
    elif recommendation == "FIX_CONTACT_CLASSIFICATION":
        next_decision = "Fix contact classification before changing controller or training."
    elif recommendation == "FIX_POSTURE_STABILITY":
        next_decision = "Tune posture stability and conservative amplitude before any PPO."
    elif recommendation == "FIX_CONTROL_MAPPING":
        next_decision = "Fix control mapping or joint primitive signs before W1."
    else:
        next_decision = "No-go for W1; inspect W0 timelines manually."
    engineering = (
        f"Best trial classified as {best.gait_candidate_label} with failure mode {best.failure_mode}. "
        "The result is a scripted diagnostic only and must not be treated as a learned gait."
    )
    payload = {
        "readiness_label_used": readiness.get("readiness_label"),
        "conservative_grid_used": [asdict(params) for params in conservative_trial_grid()],
        "total_trials": len(summaries),
        "best_trial_id": best.trial_id,
        "best_trial_params": {
            "stage": best.stage,
            "target_vx": best.target_vx,
            "gait_period": best.gait_period,
            "clearance": best.clearance,
            "action_scale": best.action_scale,
        },
        "best_gait_candidate_label": best.gait_candidate_label,
        "best_failure_mode": best.failure_mode,
        "best_trial_metrics": asdict(best),
        "all_trial_summaries": [asdict(item) for item in summaries],
        "recommendation": recommendation,
        "engineering_interpretation": engineering,
        "next_decision": next_decision,
    }

    trial_fields = list(timeline_rows[0].keys()) if timeline_rows else []
    csv_write(config.output_dir / "phase_w0_duckref_trials.csv", timeline_rows, trial_fields)
    csv_write(config.output_dir / "phase_w0_duckref_best_timeline.csv", best_rows, trial_fields)
    (config.output_dir / "phase_w0_duckref_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(config.output_dir / "phase_w0_duckref_report.md", payload, summaries)
    append_progress_log(PROGRESS_LOG, payload)
    return payload


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-path", type=Path, default=default_scene_path())
    parser.add_argument("--duck-scaled-reference", type=Path, default=DEFAULT_DUCK_SCALED_REFERENCE)
    parser.add_argument("--joint-sign-mapping", type=Path, default=DEFAULT_JOINT_SIGN_MAPPING)
    parser.add_argument("--readiness-summary", type=Path, default=DEFAULT_READINESS_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--disable-render", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    """Run Phase W0-DuckRef conservative scripted smoke test."""

    args = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    config = SmokeConfig(
        scene_path=require_scene(args.scene_path),
        duck_scaled_reference=args.duck_scaled_reference,
        joint_sign_mapping=args.joint_sign_mapping,
        readiness_summary=args.readiness_summary,
        output_dir=args.output_dir,
        steps=int(args.steps),
        disable_render=bool(args.disable_render),
    )
    summary = run_smoke_test(config)
    print(f"trials={config.output_dir / 'phase_w0_duckref_trials.csv'}")
    print(f"best_timeline={config.output_dir / 'phase_w0_duckref_best_timeline.csv'}")
    print(f"summary={config.output_dir / 'phase_w0_duckref_summary.json'}")
    print(f"report={config.output_dir / 'phase_w0_duckref_report.md'}")
    print(f"best_trial_id={summary['best_trial_id']}")
    print(f"gait_candidate_label={summary['best_gait_candidate_label']}")
    print(f"failure_mode={summary['best_failure_mode']}")
    print(f"recommendation={summary['recommendation']}")


if __name__ == "__main__":
    main()
