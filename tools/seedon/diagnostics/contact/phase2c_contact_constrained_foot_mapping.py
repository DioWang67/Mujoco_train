"""Phase 2C contact-constrained foot mapping and rollover diagnostic.

This Class C diagnostic inspects why left-foot micro capture remains tiny under
the right-support force gate, and why toe handoff has not appeared. It does not
train PPO, does not run a walking controller, and does not modify scenes.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from seedon_baseline.env import SeedonStandingEnv, load_seedon_config_from_env
from tools.seedon_blue_like_phase1_rollover_diagnostic import _region_for_geom, _side_for_geom
from tools.seedon_debug_common import DEBUG_OUT_DIR, DEFAULT_SCENE_PATH, geom_name, require_scene


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_V5A_SCENE = DEBUG_OUT_DIR / "blue_like_sole_experiments_v5" / "training_scene_v5_a.xml"
DEFAULT_OUT_DIR = DEBUG_OUT_DIR / "phase2c_contact_constrained_foot_mapping"
PROGRESS_LOG = REPO_ROOT / "docs" / "seedon_blue_like_dynamic_gait_progress_log.md"

L_HIP_ROLL, L_HIP_PITCH, L_KNEE, L_ANKLE = 6, 7, 8, 9
R_HIP_ROLL = 1
FOOT_FORCE_THRESHOLD_N = 0.1

JOINT_CHANNELS: tuple[tuple[str, dict[int, float], str], ...] = (
    ("left_hip_pitch", {L_HIP_PITCH: 1.0}, "forward/backward foot x"),
    ("left_hip_roll", {L_HIP_ROLL: 1.0}, "lateral foot y"),
    ("left_knee_pitch", {L_KNEE: 1.0}, "vertical/fore-aft leg fold"),
    ("left_ankle_pitch", {L_ANKLE: 1.0}, "foot pitch/contact patch"),
    ("left_hip_pitch+left_knee_pitch", {L_HIP_PITCH: 1.0, L_KNEE: 1.0}, "coupled fore-aft/height"),
    ("left_hip_pitch+left_ankle_pitch", {L_HIP_PITCH: 1.0, L_ANKLE: 1.0}, "coupled fore-aft/ankle pitch"),
    ("left_hip_roll+left_hip_pitch", {L_HIP_ROLL: 1.0, L_HIP_PITCH: 1.0}, "side-forward intent"),
)


@dataclass(frozen=True)
class ContactForces:
    """Foot contact force summary."""

    left_total: float
    right_total: float
    right_center: float
    right_toe: float
    right_heel: float
    left_center: float
    left_toe: float
    left_heel: float
    right_center_contact: bool
    right_toe_contact: bool
    right_heel_contact: bool
    right_cop_x: float
    right_cop_y: float


def _default_scene() -> Path:
    return DEFAULT_V5A_SCENE if DEFAULT_V5A_SCENE.is_file() else DEFAULT_SCENE_PATH


def _parse_float_list(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one float")
    return values


def _left_foot_pos(env: SeedonStandingEnv, data: mujoco.MjData | None = None) -> np.ndarray:
    source = env.data if data is None else data
    geom_id = env._foot_geom_ids[1]
    pos = np.asarray(source.geom_xpos[geom_id], dtype=np.float64).copy()
    pos[2] -= float(env.model.geom_size[geom_id][2])
    return pos


def _set_joint_positions(env: SeedonStandingEnv, data: mujoco.MjData, joint_positions: np.ndarray) -> None:
    for joint_index, joint_id in enumerate(env._joint_ids):
        data.qpos[env.model.jnt_qposadr[joint_id]] = float(joint_positions[joint_index])


def _kinematic_left_foot_delta(env: SeedonStandingEnv, delta: np.ndarray) -> np.ndarray:
    data = mujoco.MjData(env.model)
    data.qpos[:] = env.data.qpos
    data.qvel[:] = env.data.qvel
    _set_joint_positions(env, data, env._joint_positions() + delta)
    mujoco.mj_forward(env.model, data)
    return _left_foot_pos(env, data) - _left_foot_pos(env)


def _contact_forces(env: SeedonStandingEnv) -> ContactForces:
    forces = {
        "left_center": 0.0,
        "left_toe": 0.0,
        "left_heel": 0.0,
        "right_center": 0.0,
        "right_toe": 0.0,
        "right_heel": 0.0,
    }
    weighted_right_x = 0.0
    weighted_right_y = 0.0
    right_force_for_cop = 0.0
    wrench = np.zeros(6, dtype=np.float64)
    for contact_index in range(env.data.ncon):
        contact = env.data.contact[contact_index]
        name_a = geom_name(env.model, int(contact.geom1))
        name_b = geom_name(env.model, int(contact.geom2))
        if "floor" not in {name_a, name_b}:
            continue
        foot_name = name_b if name_a == "floor" else name_a
        side = _side_for_geom(foot_name)
        if side not in {"left", "right"}:
            continue
        foot_geom_id = int(contact.geom2 if name_a == "floor" else contact.geom1)
        body_id = int(env.model.geom_bodyid[foot_geom_id])
        world_pos = np.asarray(contact.pos, dtype=np.float64)
        local_pos = env.data.xmat[body_id].reshape(3, 3).T @ (world_pos - env.data.xpos[body_id])
        region = _region_for_geom(foot_name, float(local_pos[0]))
        if region not in {"center", "toe", "heel"}:
            continue
        mujoco.mj_contactForce(env.model, env.data, contact_index, wrench)
        normal = abs(float(wrench[0]))
        forces[f"{side}_{region}"] += normal
        if side == "right":
            weighted_right_x += normal * float(world_pos[0])
            weighted_right_y += normal * float(world_pos[1])
            right_force_for_cop += normal
    right_total = forces["right_center"] + forces["right_toe"] + forces["right_heel"]
    left_total = forces["left_center"] + forces["left_toe"] + forces["left_heel"]
    return ContactForces(
        left_total=left_total,
        right_total=right_total,
        right_center=forces["right_center"],
        right_toe=forces["right_toe"],
        right_heel=forces["right_heel"],
        left_center=forces["left_center"],
        left_toe=forces["left_toe"],
        left_heel=forces["left_heel"],
        right_center_contact=forces["right_center"] > FOOT_FORCE_THRESHOLD_N,
        right_toe_contact=forces["right_toe"] > FOOT_FORCE_THRESHOLD_N,
        right_heel_contact=forces["right_heel"] > FOOT_FORCE_THRESHOLD_N,
        right_cop_x=weighted_right_x / max(right_force_for_cop, 1e-9),
        right_cop_y=weighted_right_y / max(right_force_for_cop, 1e-9),
    )


def _support_ratio(forces: ContactForces) -> float:
    return forces.right_total / max(forces.right_total + forces.left_total, 1e-9)


def _profile_alpha(step: int, ramp_steps: int, hold_steps: int, recover_steps: int) -> tuple[str, float]:
    if step <= ramp_steps:
        return "RAMP_IN", step / max(ramp_steps, 1)
    if step <= ramp_steps + hold_steps:
        return "HOLD", 1.0
    if step <= ramp_steps + hold_steps + recover_steps:
        local = step - ramp_steps - hold_steps
        return "RECOVER", 1.0 - local / max(recover_steps, 1)
    return "CENTER_SETTLE", 0.0


def _right_support_target(env: SeedonStandingEnv, alpha: float, magnitude: float) -> np.ndarray:
    target = env._nominal_joint_qpos.copy()
    target[R_HIP_ROLL] -= magnitude * alpha
    target[L_HIP_ROLL] -= magnitude * alpha * 2.0
    return env._apply_safe_joint_target_clamps(target)


def _settle_env(env: SeedonStandingEnv, seed: int, steps: int = 20) -> None:
    env.reset(seed=seed)
    target = env._nominal_joint_qpos.copy()
    for _ in range(steps):
        env._do_pd_simulation(target)
        env._gait_step += 1


def _apply_right_support_profile(
    env: SeedonStandingEnv,
    *,
    magnitude: float,
    ramp_steps: int,
    hold_steps: int,
) -> None:
    for step in range(1, ramp_steps + hold_steps + 1):
        phase, alpha = _profile_alpha(step, ramp_steps, hold_steps, 1)
        del phase
        env._do_pd_simulation(_right_support_target(env, alpha, magnitude))
        env._gait_step += 1


def _classify_mapping(dx: float, dy: float, dz: float) -> str:
    eps = 1e-6
    if max(abs(dx), abs(dy), abs(dz)) < eps:
        return "ineffective"
    if dx > eps:
        return "forward_effective"
    if dx < -eps:
        return "backward_effective"
    if abs(dz) > abs(dx) and abs(dz) > abs(dy):
        return "mostly_vertical"
    if abs(dy) > abs(dx):
        return "mostly_lateral"
    return "ineffective"


def run_joint_mapping(scene_path: Path, magnitudes: list[float], seed: int) -> list[dict[str, Any]]:
    env = SeedonStandingEnv(scene_path=scene_path, reset_noise_scale=0.0, reward_config=load_seedon_config_from_env())
    rows: list[dict[str, Any]] = []
    try:
        _settle_env(env, seed)
        base_pos = _left_foot_pos(env)
        base_x = float(env.data.qpos[0])
        base_roll = float(env._base_roll())
        base_pitch = float(env._base_pitch())
        base_force = _contact_forces(env).left_total
        for name, weights, expected in JOINT_CHANNELS:
            for sign in (-1.0, 1.0):
                for mag in magnitudes:
                    delta = np.zeros(len(env._joint_ids), dtype=np.float64)
                    for joint_index, weight in weights.items():
                        delta[joint_index] = sign * mag * weight
                    foot_delta = _kinematic_left_foot_delta(env, delta)
                    rows.append(
                        {
                            "joint_channel": name,
                            "sign": sign,
                            "magnitude": mag,
                            "expected_direction": expected,
                            "foot_x_delta": float(foot_delta[0]),
                            "foot_y_delta": float(foot_delta[1]),
                            "foot_z_delta": float(foot_delta[2]),
                            "base_x_delta": float(env.data.qpos[0] - base_x),
                            "base_roll_delta": float(env._base_roll() - base_roll),
                            "base_pitch_delta": float(env._base_pitch() - base_pitch),
                            "contact_force_delta": float(_contact_forces(env).left_total - base_force),
                            "classification": _classify_mapping(float(foot_delta[0]), float(foot_delta[1]), float(foot_delta[2])),
                        }
                    )
        del base_pos
    finally:
        env.close()
    return rows


def _dynamic_mapping_trial(
    env: SeedonStandingEnv,
    *,
    state_name: str,
    name: str,
    weights: dict[int, float],
    sign: float,
    magnitude: float,
    expected: np.ndarray,
    support_ratio_before: float,
    steps: int = 20,
) -> dict[str, Any]:
    before = _left_foot_pos(env)
    force_before = _contact_forces(env).left_total
    target = env._joint_positions()
    for joint_index, weight in weights.items():
        target[joint_index] += sign * magnitude * weight
    target = env._apply_safe_joint_target_clamps(target)
    forces: list[float] = []
    ratios: list[float] = []
    for _ in range(steps):
        env._do_pd_simulation(target)
        env._gait_step += 1
        current_forces = _contact_forces(env)
        forces.append(current_forces.left_total)
        ratios.append(_support_ratio(current_forces))
    after = _left_foot_pos(env)
    force_after = _contact_forces(env).left_total
    actual = after - before
    loss = 1.0 - np.divide(np.abs(actual), np.maximum(np.abs(expected), 1e-9))
    loss = np.clip(loss, -10.0, 10.0)
    if expected[0] > 0 and actual[0] < 0.00005:
        classification = "constraint_blocks_forward_motion"
    elif actual[0] > 0.00005:
        classification = "forward_motion_available"
    elif min(ratios or [support_ratio_before]) < 0.58 and actual[0] > 0:
        classification = "force_gate_breaks_before_motion"
    elif np.mean(forces or [0.0]) > 1.0 and max(abs(float(actual[0])), abs(float(actual[1])), abs(float(actual[2]))) < 0.00005:
        classification = "contact_locked"
    else:
        classification = "contact_locked"
    return {
        "state_name": state_name,
        "joint_channel": name,
        "sign": sign,
        "magnitude": magnitude,
        "left_foot_x_before": float(before[0]),
        "left_foot_x_after": float(after[0]),
        "left_foot_y_before": float(before[1]),
        "left_foot_y_after": float(after[1]),
        "left_foot_z_before": float(before[2]),
        "left_foot_z_after": float(after[2]),
        "left_foot_x_delta": float(actual[0]),
        "left_foot_y_delta": float(actual[1]),
        "left_foot_z_delta": float(actual[2]),
        "left_foot_contact_force_before": float(force_before),
        "left_foot_contact_force_after": float(force_after),
        "left_foot_contact_force_mean": float(np.mean(forces)) if forces else 0.0,
        "right_support_force_ratio": float(np.mean(ratios)) if ratios else support_ratio_before,
        "constraint_loss_ratio_x": float(loss[0]),
        "constraint_loss_ratio_y": float(loss[1]),
        "constraint_loss_ratio_z": float(loss[2]),
        "classification": classification,
    }


def run_contact_constrained_mapping(
    scene_path: Path,
    magnitudes: list[float],
    seed: int,
    support_profile_magnitude: float,
    support_profile_ramp_steps: int,
    support_profile_hold_steps: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    states = ("neutral_contact", "right_support_force_gate", "reduced_contact_or_light_unload")
    for state in states:
        for name, weights, _ in JOINT_CHANNELS:
            for sign in (-1.0, 1.0):
                for mag in magnitudes:
                    env = SeedonStandingEnv(scene_path=scene_path, reset_noise_scale=0.0, reward_config=load_seedon_config_from_env())
                    try:
                        _settle_env(env, seed)
                        if state in {"right_support_force_gate", "reduced_contact_or_light_unload"}:
                            _apply_right_support_profile(
                                env,
                                magnitude=support_profile_magnitude,
                                ramp_steps=support_profile_ramp_steps,
                                hold_steps=support_profile_hold_steps,
                            )
                        if state == "reduced_contact_or_light_unload":
                            # Keep the same force-gated posture; this explicitly avoids
                            # lift and tests whether lighter swing load changes mapping.
                            pass
                        delta = np.zeros(len(env._joint_ids), dtype=np.float64)
                        for joint_index, weight in weights.items():
                            delta[joint_index] = sign * mag * weight
                        expected = _kinematic_left_foot_delta(env, delta)
                        rows.append(
                            _dynamic_mapping_trial(
                                env,
                                state_name=state,
                                name=name,
                                weights=weights,
                                sign=sign,
                                magnitude=mag,
                                expected=expected,
                                support_ratio_before=_support_ratio(_contact_forces(env)),
                            )
                        )
                    finally:
                        env.close()
    return rows


def run_rollover_timeline(
    scene_path: Path,
    steps: int,
    seed: int,
    support_profile_magnitude: float,
    ramp_steps: int,
    hold_steps: int,
    recover_steps: int,
) -> list[dict[str, Any]]:
    env = SeedonStandingEnv(scene_path=scene_path, reset_noise_scale=0.0, reward_config=load_seedon_config_from_env())
    rows: list[dict[str, Any]] = []
    try:
        _settle_env(env, seed)
        saw_center = False
        for step in range(1, steps + 1):
            phase, alpha = _profile_alpha(step, ramp_steps, hold_steps, recover_steps)
            env._do_pd_simulation(_right_support_target(env, alpha, support_profile_magnitude))
            env._gait_step += 1
            forces = _contact_forces(env)
            support_ratio = _support_ratio(forces)
            saw_center = saw_center or forces.right_center_contact
            toe_handoff = saw_center and forces.right_toe_contact
            center_to_toe = forces.right_toe / max(forces.right_center, 1e-9)
            rows.append(
                {
                    "step": step,
                    "phase": phase,
                    "base_x": float(env.data.qpos[0]),
                    "base_vx": float(env.data.qvel[0]),
                    "base_roll": float(env._base_roll()),
                    "base_pitch": float(env._base_pitch()),
                    "upright": float(env._base_upright()),
                    "right_total_force": forces.right_total,
                    "right_center_force": forces.right_center,
                    "right_toe_force": forces.right_toe,
                    "right_heel_force": forces.right_heel,
                    "left_total_force": forces.left_total,
                    "support_force_ratio": support_ratio,
                    "right_center_contact": forces.right_center_contact,
                    "right_toe_contact": forces.right_toe_contact,
                    "right_heel_contact": forces.right_heel_contact,
                    "right_contact_cop_x": forces.right_cop_x,
                    "right_contact_cop_y": forces.right_cop_y,
                    "toe_handoff_detected": toe_handoff,
                    "center_to_toe_force_ratio": center_to_toe,
                }
            )
    finally:
        env.close()
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _best_channel(rows: list[dict[str, Any]], key: str) -> str:
    if not rows:
        return "none"
    best = max(rows, key=lambda row: abs(float(row[key])))
    return f"{best['joint_channel']} {float(best['sign']):+g}x {float(best['magnitude']):.4f}"


def _append_progress_log(command: str, summary: dict[str, Any], out_dir: Path) -> None:
    PROGRESS_LOG.parent.mkdir(parents=True, exist_ok=True)
    if not PROGRESS_LOG.exists():
        PROGRESS_LOG.write_text("# Seedon Blue-Like Dynamic Gait Progress Log\n\n", encoding="utf-8")
    lines = [
        f"## {date.today().isoformat()} - Phase 2C Contact-Constrained Foot Mapping + Rollover Diagnostic",
        "",
        "### Goal",
        "",
        "Diagnose why left-foot micro capture remains around 0.08 mm under the right-support force gate, and why toe handoff has not appeared despite stable load transfer.",
        "",
        "### Files",
        "",
        "- `tools/seedon_phase2c_contact_constrained_foot_mapping.py`",
        "- `docs/seedon_blue_like_dynamic_gait_progress_log.md`",
        "",
        "### Command",
        "",
        "```powershell",
        command,
        "```",
        "",
        "### Outputs",
        "",
        f"- `{out_dir / 'phase2c_joint_mapping.csv'}`",
        f"- `{out_dir / 'phase2c_contact_constrained_mapping.csv'}`",
        f"- `{out_dir / 'phase2c_rollover_timeline.csv'}`",
        f"- `{out_dir / 'phase2c_summary.json'}`",
        "",
        "### Key Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| any_forward_effective_joint_channel | {str(summary['any_forward_effective_joint_channel']).lower()} |",
        f"| joint_mapping_best_forward_channel | {summary['joint_mapping_best_forward_channel']} |",
        f"| contact_constraint_blocks_forward_motion | {str(summary['contact_constraint_blocks_forward_motion']).lower()} |",
        f"| constraint_loss_ratio_x_mean | {summary['constraint_loss_ratio_x_mean']:.6f} |",
        f"| right_support_rollover_detected | {str(summary['right_support_rollover_detected']).lower()} |",
        f"| toe_handoff_detected | {str(summary['toe_handoff_detected']).lower()} |",
        f"| toe_handoff_detection_reliable | {str(summary['toe_handoff_detection_reliable']).lower()} |",
        f"| max_right_toe_force | {summary['max_right_toe_force']:.6f} |",
        f"| max_right_center_force | {summary['max_right_center_force']:.6f} |",
        "",
        "### Result",
        "",
        "INCONCLUSIVE" if summary["any_forward_effective_joint_channel"] else "FAIL",
        "",
        "### Engineering Interpretation",
        "",
        str(summary["engineering_interpretation"]),
        "",
        "### Next Decision",
        "",
        str(summary["recommended_next_step"]),
        "",
    ]
    with PROGRESS_LOG.open("a", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")


def run_diagnostic(
    *,
    scene_path: Path,
    out_dir: Path,
    steps: int,
    seed: int,
    magnitudes: list[float],
    support_profile_magnitude: float,
    support_profile_ramp_steps: int,
    support_profile_hold_steps: int,
    support_profile_recover_steps: int,
    command: str,
    update_progress_log: bool = True,
) -> dict[str, Any]:
    scene = require_scene(scene_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    joint_rows = run_joint_mapping(scene, magnitudes, seed)
    contact_rows = run_contact_constrained_mapping(
        scene,
        magnitudes,
        seed,
        support_profile_magnitude,
        support_profile_ramp_steps,
        support_profile_hold_steps,
    )
    rollover_rows = run_rollover_timeline(
        scene,
        steps,
        seed,
        support_profile_magnitude,
        support_profile_ramp_steps,
        support_profile_hold_steps,
        support_profile_recover_steps,
    )
    _write_csv(out_dir / "phase2c_joint_mapping.csv", joint_rows)
    _write_csv(out_dir / "phase2c_contact_constrained_mapping.csv", contact_rows)
    _write_csv(out_dir / "phase2c_rollover_timeline.csv", rollover_rows)

    forward_rows = [row for row in joint_rows if row["classification"] == "forward_effective"]
    constraint_forward = [row for row in contact_rows if row["classification"] == "constraint_blocks_forward_motion"]
    loss_x = [float(row["constraint_loss_ratio_x"]) for row in contact_rows]
    right_center_steps = sum(bool(row["right_center_contact"]) for row in rollover_rows)
    right_toe_steps = sum(bool(row["right_toe_contact"]) for row in rollover_rows)
    right_heel_steps = sum(bool(row["right_heel_contact"]) for row in rollover_rows)
    max_toe = max((float(row["right_toe_force"]) for row in rollover_rows), default=0.0)
    max_center = max((float(row["right_center_force"]) for row in rollover_rows), default=0.0)
    max_ratio = max((float(row["center_to_toe_force_ratio"]) for row in rollover_rows), default=0.0)
    toe_handoff = any(bool(row["toe_handoff_detected"]) for row in rollover_rows)
    rollover = right_center_steps > 0 and right_toe_steps > 0 and max_ratio > 0.1
    reliable = right_center_steps + right_toe_steps + right_heel_steps > 0
    recommended = (
        "Fix toe/contact geometry or contact patch before further capture-controller work."
        if not toe_handoff
        else "Use observed rollover conditions to design the next guarded micro-capture diagnostic."
    )
    interpretation = (
        f"Left-foot kinematic forward mapping exists={bool(forward_rows)}. "
        f"Contact-constrained forward blocking rows={len(constraint_forward)}. "
        f"Best forward channel={_best_channel(joint_rows, 'foot_x_delta')}. "
        f"Toe handoff detected={toe_handoff}; detection reliable={reliable}. "
        f"Right center steps={right_center_steps}, toe steps={right_toe_steps}, max toe force={max_toe:.3f}, max center force={max_center:.3f}. "
        f"Controlled right-support profile rollover={rollover}. Next focus: {recommended}"
    )
    summary = {
        "joint_mapping_best_forward_channel": _best_channel(joint_rows, "foot_x_delta"),
        "joint_mapping_best_lateral_channel": _best_channel(joint_rows, "foot_y_delta"),
        "joint_mapping_best_vertical_channel": _best_channel(joint_rows, "foot_z_delta"),
        "any_forward_effective_joint_channel": bool(forward_rows),
        "contact_constraint_blocks_forward_motion": bool(constraint_forward),
        "constraint_loss_ratio_x_max": max(loss_x, default=0.0),
        "constraint_loss_ratio_x_mean": float(np.mean(loss_x)) if loss_x else 0.0,
        "right_support_rollover_detected": rollover,
        "toe_handoff_detected": toe_handoff,
        "toe_handoff_detection_reliable": reliable,
        "right_center_contact_steps": right_center_steps,
        "right_toe_contact_steps": right_toe_steps,
        "right_heel_contact_steps": right_heel_steps,
        "max_right_toe_force": max_toe,
        "max_right_center_force": max_center,
        "max_center_to_toe_force_ratio": max_ratio,
        "recommended_next_step": recommended,
        "engineering_interpretation": interpretation,
    }
    (out_dir / "phase2c_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if update_progress_log:
        _append_progress_log(command, summary, out_dir)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--scene-path", type=Path, default=_default_scene())
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--magnitudes", type=_parse_float_list, default="0.001,0.0025,0.005,0.0075,0.010,0.015,0.020")
    parser.add_argument("--support-profile-magnitude", type=float, default=0.03)
    parser.add_argument("--support-profile-ramp-steps", type=int, default=10)
    parser.add_argument("--support-profile-hold-steps", type=int, default=80)
    parser.add_argument("--support-profile-recover-steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-progress-log", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = f".venv\\Scripts\\python.exe -m tools.seedon_phase2c_contact_constrained_foot_mapping --steps {args.steps}"
    summary = run_diagnostic(
        scene_path=args.scene_path,
        out_dir=args.out_dir,
        steps=args.steps,
        seed=args.seed,
        magnitudes=args.magnitudes,
        support_profile_magnitude=args.support_profile_magnitude,
        support_profile_ramp_steps=args.support_profile_ramp_steps,
        support_profile_hold_steps=args.support_profile_hold_steps,
        support_profile_recover_steps=args.support_profile_recover_steps,
        command=command,
        update_progress_log=not args.no_progress_log,
    )
    print(
        "phase2c forward_mapping={forward} constraint_blocks={blocks} rollover={rollover} toe_handoff={toe}".format(
            forward=summary["any_forward_effective_joint_channel"],
            blocks=summary["contact_constraint_blocks_forward_motion"],
            rollover=summary["right_support_rollover_detected"],
            toe=summary["toe_handoff_detected"],
        )
    )
    print(f"summary={args.out_dir / 'phase2c_summary.json'}")
    print(f"joint_mapping={args.out_dir / 'phase2c_joint_mapping.csv'}")
    print(f"contact_mapping={args.out_dir / 'phase2c_contact_constrained_mapping.csv'}")
    print(f"rollover={args.out_dir / 'phase2c_rollover_timeline.csv'}")
    print(f"progress_log={PROGRESS_LOG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
