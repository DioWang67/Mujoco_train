"""Phase 2B force-gated micro-capture refinement for Seedon.

This Class C diagnostic refines the tiny left-leg capture intent introduced in
Phase 2A. It only runs after a right-support force gate is established, aborts
capture on force-gate drop by default, and does not train PPO or attempt full
capture stepping.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

import mujoco
import numpy as np

from seedon_baseline.env import SeedonStandingEnv, load_seedon_config_from_env
from tools.seedon_blue_like_phase1_rollover_diagnostic import (
    _foot_contact_forces,
    _region_for_geom,
    _side_for_geom,
)
from tools.seedon_debug_common import DEBUG_OUT_DIR, DEFAULT_SCENE_PATH, geom_name, require_scene


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V5A_SCENE = DEBUG_OUT_DIR / "blue_like_sole_experiments_v5" / "training_scene_v5_a.xml"
DEFAULT_OUT_DIR = DEBUG_OUT_DIR / "phase2b_micro_capture_refinement"
PROGRESS_LOG = REPO_ROOT / "docs" / "seedon_blue_like_dynamic_gait_progress_log.md"

R_HIP_ROLL, R_HIP_PITCH = 1, 2
L_HIP_ROLL, L_HIP_PITCH = 6, 7
FOOT_FORCE_THRESHOLD_N = 0.1


class Phase2BState(str, Enum):
    """Right-support gated micro-capture refinement phases."""

    SETTLE = "SETTLE"
    RAMP_IN_RIGHT_SUPPORT = "RAMP_IN_RIGHT_SUPPORT"
    HOLD_RIGHT_SUPPORT_FORCE_GATE = "HOLD_RIGHT_SUPPORT_FORCE_GATE"
    MICRO_CAPTURE_LEFT = "MICRO_CAPTURE_LEFT"
    RECOVER_CENTER = "RECOVER_CENTER"
    CENTER_SETTLE = "CENTER_SETTLE"


@dataclass(frozen=True)
class RefinementTrial:
    """One Phase 2B refinement trial."""

    trial_id: str
    capture_forward_bias: float
    capture_lateral_bias: float
    capture_duration_steps: int
    capture_profile_shape: str


@dataclass(frozen=True)
class RefinementResult:
    """Flat trial result for Phase 2B."""

    trial_id: str
    capture_forward_bias: float
    capture_lateral_bias: float
    capture_duration_steps: int
    capture_profile_shape: str
    steps: int
    completed_steps: int
    phase_reached: str
    micro_capture_started: bool
    micro_capture_completed: bool
    capture_aborted: bool
    capture_abort_reason: str
    capture_abort_step: int
    force_gate_reached: bool
    force_gate_reached_step: int
    force_gate_hold_steps_before_capture: int
    mean_forward_velocity: float
    forward_displacement: float
    min_upright: float
    max_abs_roll: float
    max_abs_pitch: float
    contact_none_ratio: float
    jump_count: int
    both_contact_ratio: float
    single_contact_ratio: float
    support_force_ratio_mean: float
    support_force_ratio_peak: float
    support_force_ratio_min_during_capture: float
    support_force_ratio_drop_during_capture: float
    support_force_ratio_hold_steps_058: int
    max_continuous_hold_058: int
    swing_force_ratio_min: float
    left_foot_x_before_capture: float
    left_foot_x_after_capture: float
    left_foot_y_before_capture: float
    left_foot_y_after_capture: float
    left_foot_z_max: float
    left_foot_forward_delta: float
    left_foot_lateral_delta: float
    left_foot_speed_peak_during_capture: float
    toe_handoff_detected: bool
    toe_handoff_left_count: int
    toe_handoff_right_count: int
    first_toe_handoff_step: int
    delayed_fall_after_recover: bool
    classification: str
    fail_reasons: str


def _default_scene() -> Path:
    """Return preferred v5_a scene when present."""

    return DEFAULT_V5A_SCENE if DEFAULT_V5A_SCENE.is_file() else DEFAULT_SCENE_PATH


def _parse_float_list(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one float")
    return values


def _parse_int_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("expected positive integers")
    return values


def _parse_shape_list(raw: str) -> list[str]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    allowed = {"linear", "smoothstep", "minimum_jerk"}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise argparse.ArgumentTypeError(f"unsupported profile shapes: {unknown}")
    return values


def _parse_bool(raw: str) -> bool:
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected true/false")


def _shape(shape: str, alpha: float) -> float:
    t = float(np.clip(alpha, 0.0, 1.0))
    if shape == "linear":
        return t
    if shape == "smoothstep":
        return t * t * (3.0 - 2.0 * t)
    if shape == "minimum_jerk":
        return t * t * t * (10.0 - 15.0 * t + 6.0 * t * t)
    raise ValueError(f"Unsupported shape: {shape}")


def _contact_state(left_force: float, right_force: float) -> str:
    left = left_force > FOOT_FORCE_THRESHOLD_N
    right = right_force > FOOT_FORCE_THRESHOLD_N
    if left and right:
        return "both"
    if left:
        return "left"
    if right:
        return "right"
    return "none"


def _foot_region_contacts(env: SeedonStandingEnv) -> dict[str, bool]:
    contacts = {
        "center_contact_left": False,
        "toe_contact_left": False,
        "center_contact_right": False,
        "toe_contact_right": False,
    }
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
        foot_body_id = int(env.model.geom_bodyid[foot_geom_id])
        world_pos = np.asarray(contact.pos, dtype=np.float64)
        local_pos = env.data.xmat[foot_body_id].reshape(3, 3).T @ (world_pos - env.data.xpos[foot_body_id])
        region = _region_for_geom(foot_name, float(local_pos[0]))
        if region in {"center", "toe"}:
            contacts[f"{region}_contact_{side}"] = True
    return contacts


def _left_foot_pos(env: SeedonStandingEnv) -> tuple[float, float, float]:
    geom_id = env._foot_geom_ids[1]
    pos = env.data.geom_xpos[geom_id]
    return float(pos[0]), float(pos[1]), float(pos[2] - env.model.geom_size[geom_id][2])


def _support_ratios(forces: dict[str, float | bool | int]) -> tuple[float, float, float, float]:
    support_force = float(forces["right_force"])
    swing_force = float(forces["left_force"])
    total = max(support_force + swing_force, 1e-9)
    return support_force, swing_force, support_force / total, swing_force / total


def _target(
    env: SeedonStandingEnv,
    *,
    phase: Phase2BState,
    phase_step: int,
    trial: RefinementTrial,
    max_roll: float,
    max_pitch: float,
    ramp_steps: int,
    recover_steps: int,
) -> np.ndarray:
    """Return joint target for one control step."""

    target = env._nominal_joint_qpos.copy()
    if phase in {Phase2BState.SETTLE, Phase2BState.CENTER_SETTLE}:
        load_alpha = 0.0
        capture_alpha = 0.0
    elif phase == Phase2BState.RAMP_IN_RIGHT_SUPPORT:
        load_alpha = phase_step / max(ramp_steps, 1)
        capture_alpha = 0.0
    elif phase == Phase2BState.HOLD_RIGHT_SUPPORT_FORCE_GATE:
        load_alpha = 1.0
        capture_alpha = 0.0
    elif phase == Phase2BState.MICRO_CAPTURE_LEFT:
        load_alpha = 1.0
        capture_alpha = _shape(trial.capture_profile_shape, phase_step / max(trial.capture_duration_steps, 1))
    elif phase == Phase2BState.RECOVER_CENTER:
        load_alpha = 1.0 - phase_step / max(recover_steps, 1)
        capture_alpha = 1.0 - phase_step / max(recover_steps, 1)
    else:
        load_alpha = 0.0
        capture_alpha = 0.0
    load_alpha = float(np.clip(load_alpha, 0.0, 1.0))
    capture_alpha = float(np.clip(capture_alpha, 0.0, 1.0))

    pelvis = float(np.clip(0.03 * load_alpha, -max_roll, max_roll))
    swing_roll = float(np.clip(0.03 * load_alpha, -max_roll, max_roll))
    target[R_HIP_ROLL] -= pelvis
    target[L_HIP_ROLL] -= pelvis + swing_roll
    target[L_HIP_PITCH] += float(np.clip(trial.capture_forward_bias * capture_alpha, -max_pitch, max_pitch))
    target[L_HIP_ROLL] += float(np.clip(trial.capture_lateral_bias * capture_alpha, -max_roll, max_roll))
    return env._apply_safe_joint_target_clamps(target)


def _safe_gate(row: dict[str, Any], threshold: float = 0.58) -> bool:
    return (
        float(row["support_force_ratio"]) >= threshold
        and float(row["swing_force_ratio"]) <= 1.0 - threshold
        and float(row["upright"]) >= 0.985
        and not bool(row["contact_none"])
        and not bool(row["jump"])
    )


def _count_bursts(rows: list[dict[str, Any]], key: str) -> int:
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


def _max_continuous_hold(rows: list[dict[str, Any]], threshold: float = 0.58) -> int:
    best = 0
    current = 0
    for row in rows:
        if _safe_gate(row, threshold):
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def _toe_handoff_counts(rows: list[dict[str, Any]]) -> tuple[int, int, int]:
    counts = {"left": 0, "right": 0}
    saw_center = {"left": False, "right": False}
    toe_active = {"left": False, "right": False}
    first_step = -1
    for row in rows:
        for side in ("left", "right"):
            if bool(row[f"center_contact_{side}"]):
                saw_center[side] = True
            toe = bool(row[f"toe_contact_{side}"])
            if saw_center[side] and toe and not toe_active[side]:
                counts[side] += 1
                if first_step < 0:
                    first_step = int(row["step"])
                toe_active[side] = True
                saw_center[side] = False
            if not toe:
                toe_active[side] = False
    return counts["left"], counts["right"], first_step


def _delayed_fall(rows: list[dict[str, Any]]) -> bool:
    pre = [
        row for row in rows
        if str(row["phase"]) in {"RAMP_IN_RIGHT_SUPPORT", "HOLD_RIGHT_SUPPORT_FORCE_GATE", "MICRO_CAPTURE_LEFT"}
    ]
    post = [row for row in rows if str(row["phase"]) in {"RECOVER_CENTER", "CENTER_SETTLE"}]
    pre_stable = bool(pre) and all(not row["contact_none"] and not row["jump"] and float(row["upright"]) >= 0.985 for row in pre)
    post_fail = any(row["contact_none"] or row["jump"] or float(row["upright"]) < 0.985 for row in post)
    return pre_stable and post_fail


def _classify(
    *,
    force_gate_reached: bool,
    micro_capture_started: bool,
    micro_capture_completed: bool,
    capture_aborted: bool,
    capture_abort_reason: str,
    contact_none_ratio: float,
    jump_count: int,
    min_upright: float,
    support_min_capture: float,
    left_forward_delta: float,
    toe_handoff: bool,
    min_support_ratio_during_capture: float,
) -> str:
    if micro_capture_started and (contact_none_ratio > 0.0 or jump_count > 0 or min_upright < 0.985):
        return "micro_capture_unstable"
    if capture_aborted and ("force_gate_drop" in capture_abort_reason or "support_ratio_drop" in capture_abort_reason):
        return "micro_capture_aborted_gate_drop"
    safe_common = (
        force_gate_reached
        and micro_capture_completed
        and not capture_aborted
        and contact_none_ratio == 0.0
        and jump_count == 0
        and min_upright >= 0.985
        and support_min_capture >= min_support_ratio_during_capture
    )
    if safe_common and left_forward_delta >= 0.001:
        return "micro_capture_strong_safe"
    if safe_common and left_forward_delta >= 0.0003:
        return "micro_capture_refined_safe"
    if safe_common and 0.0 < left_forward_delta < 0.0003:
        return "micro_capture_tiny_safe"
    if toe_handoff:
        return "toe_handoff_observed"
    if micro_capture_completed and left_forward_delta <= 0.0:
        return "no_capture_effect"
    return "gate_not_reached"


def _fail_reasons(
    classification: str,
    *,
    force_gate_reached: bool,
    capture_aborted: bool,
    left_forward_delta: float,
    support_min_capture: float,
    min_support_ratio_during_capture: float,
    contact_none_ratio: float,
    jump_count: int,
    min_upright: float,
) -> str:
    if classification in {"micro_capture_strong_safe", "micro_capture_refined_safe", "micro_capture_tiny_safe"}:
        return ""
    reasons: list[str] = []
    if not force_gate_reached:
        reasons.append("force_gate_not_reached")
    if capture_aborted:
        reasons.append("capture_aborted")
    if left_forward_delta <= 0.0:
        reasons.append("no_forward_delta")
    elif left_forward_delta < 0.0003:
        reasons.append("tiny_forward_delta")
    if support_min_capture < min_support_ratio_during_capture:
        reasons.append("support_ratio_drop")
    if contact_none_ratio > 0.0:
        reasons.append("contact_none")
    if jump_count > 0:
        reasons.append("jump")
    if min_upright < 0.985:
        reasons.append("upright")
    return ",".join(reasons)


def run_trial(
    *,
    scene_path: Path,
    trial: RefinementTrial,
    steps: int,
    seed: int,
    max_roll: float,
    max_pitch: float,
    min_support_ratio_during_capture: float,
    abort_on_force_gate_drop: bool,
    settle_steps: int = 20,
    ramp_steps: int = 10,
    hold_steps: int = 80,
    recover_steps: int = 40,
    center_settle_steps: int = 40,
) -> RefinementResult:
    """Run one Phase 2B refinement trial."""

    env = SeedonStandingEnv(scene_path=scene_path, reset_noise_scale=0.0, reward_config=load_seedon_config_from_env())
    rows: list[dict[str, Any]] = []
    phase = Phase2BState.SETTLE
    phase_step = 0
    gate_hold = 0
    force_gate_reached = False
    force_gate_reached_step = -1
    micro_capture_started = False
    micro_capture_completed = False
    capture_aborted = False
    capture_abort_reason = ""
    capture_abort_step = -1
    left_before = (float("nan"), float("nan"), float("nan"))
    left_after = (float("nan"), float("nan"), float("nan"))
    previous_left_pos: tuple[float, float, float] | None = None
    speed_peak = 0.0
    try:
        env.reset(seed=seed)
        initial_x = float(env.data.qpos[0])
        robot_weight = float(np.sum(env.model.body_mass) * 9.81)
        for step in range(1, steps + 1):
            if phase == Phase2BState.SETTLE and phase_step >= settle_steps:
                phase = Phase2BState.RAMP_IN_RIGHT_SUPPORT
                phase_step = 0
            elif phase == Phase2BState.RAMP_IN_RIGHT_SUPPORT and phase_step >= ramp_steps:
                phase = Phase2BState.HOLD_RIGHT_SUPPORT_FORCE_GATE
                phase_step = 0
            elif phase == Phase2BState.HOLD_RIGHT_SUPPORT_FORCE_GATE:
                if gate_hold >= 10:
                    phase = Phase2BState.MICRO_CAPTURE_LEFT
                    phase_step = 0
                    micro_capture_started = True
                    left_before = _left_foot_pos(env)
                    previous_left_pos = left_before
                elif phase_step >= hold_steps:
                    phase = Phase2BState.RECOVER_CENTER
                    phase_step = 0
            elif phase == Phase2BState.MICRO_CAPTURE_LEFT and phase_step >= trial.capture_duration_steps:
                micro_capture_completed = True
                left_after = _left_foot_pos(env)
                phase = Phase2BState.RECOVER_CENTER
                phase_step = 0
            elif phase == Phase2BState.RECOVER_CENTER and phase_step >= recover_steps:
                phase = Phase2BState.CENTER_SETTLE
                phase_step = 0

            target = _target(
                env,
                phase=phase,
                phase_step=phase_step,
                trial=trial,
                max_roll=max_roll,
                max_pitch=max_pitch,
                ramp_steps=ramp_steps,
                recover_steps=recover_steps,
            )
            env._do_pd_simulation(target)
            env._gait_step += 1
            left_pos = _left_foot_pos(env)
            if phase == Phase2BState.MICRO_CAPTURE_LEFT and previous_left_pos is not None:
                delta_xy = np.linalg.norm(np.asarray(left_pos[:2]) - np.asarray(previous_left_pos[:2]))
                speed_peak = max(speed_peak, float(delta_xy / max(env.dt, 1e-9)))
                previous_left_pos = left_pos
            forces = _foot_contact_forces(env)
            contacts = _foot_region_contacts(env)
            left_force = float(forces["left_force"])
            right_force = float(forces["right_force"])
            support_force, swing_force, support_ratio, swing_ratio = _support_ratios(forces)
            contact_state = _contact_state(left_force, right_force)
            total_force = left_force + right_force
            contact_none = contact_state == "none"
            jump = bool(contact_none or (float(env.data.qvel[2]) > 0.05 and total_force < 0.10 * robot_weight))
            row = {
                "step": step,
                "phase": phase.value,
                "base_x": float(env.data.qpos[0]),
                "base_vx": float(env.data.qvel[0]),
                "base_roll": float(env._base_roll()),
                "base_pitch": float(env._base_pitch()),
                "upright": float(env._base_upright()),
                "contact_state": contact_state,
                "contact_none": contact_none,
                "jump": jump,
                "support_force_ratio": support_ratio,
                "swing_force_ratio": swing_ratio,
                "center_contact_left": contacts["center_contact_left"],
                "toe_contact_left": contacts["toe_contact_left"],
                "center_contact_right": contacts["center_contact_right"],
                "toe_contact_right": contacts["toe_contact_right"],
                "left_foot_x": left_pos[0],
                "left_foot_y": left_pos[1],
                "left_foot_z": left_pos[2],
                "forward_displacement": float(env.data.qpos[0]) - initial_x,
            }
            rows.append(row)
            if phase == Phase2BState.HOLD_RIGHT_SUPPORT_FORCE_GATE:
                if _safe_gate(row, min_support_ratio_during_capture):
                    gate_hold += 1
                    if not force_gate_reached and gate_hold >= 10:
                        force_gate_reached = True
                        force_gate_reached_step = step
                else:
                    gate_hold = 0
            elif phase == Phase2BState.MICRO_CAPTURE_LEFT:
                if abort_on_force_gate_drop and not _safe_gate(row, min_support_ratio_during_capture):
                    capture_aborted = True
                    capture_abort_reason = "force_gate_drop_support_ratio_drop"
                    capture_abort_step = step
                    left_after = left_pos
                    phase = Phase2BState.RECOVER_CENTER
                    phase_step = 0
            phase_step += 1
            if phase == Phase2BState.CENTER_SETTLE and phase_step >= center_settle_steps:
                break
    finally:
        env.close()

    if micro_capture_started and not np.isfinite(left_after[0]):
        capture_rows = [row for row in rows if str(row["phase"]) == Phase2BState.MICRO_CAPTURE_LEFT.value]
        if capture_rows:
            left_after = (
                float(capture_rows[-1]["left_foot_x"]),
                float(capture_rows[-1]["left_foot_y"]),
                float(capture_rows[-1]["left_foot_z"]),
            )
    states = Counter(str(row["contact_state"]) for row in rows)
    support_ratios = [float(row["support_force_ratio"]) for row in rows]
    swing_ratios = [float(row["swing_force_ratio"]) for row in rows]
    capture_rows = [row for row in rows if str(row["phase"]) == Phase2BState.MICRO_CAPTURE_LEFT.value]
    support_min_capture = min((float(row["support_force_ratio"]) for row in capture_rows), default=1.0)
    support_drop_capture = max(0.0, max((float(row["support_force_ratio"]) for row in capture_rows), default=support_min_capture) - support_min_capture)
    left_forward_delta = left_after[0] - left_before[0] if np.isfinite(left_before[0]) and np.isfinite(left_after[0]) else 0.0
    left_lateral_delta = left_after[1] - left_before[1] if np.isfinite(left_before[1]) and np.isfinite(left_after[1]) else 0.0
    left_handoff, right_handoff, first_handoff = _toe_handoff_counts(rows)
    contact_none_ratio = states["none"] / max(1, len(rows))
    jump_count = _count_bursts(rows, "jump")
    min_upright = min((float(row["upright"]) for row in rows), default=0.0)
    max_hold = _max_continuous_hold(rows, min_support_ratio_during_capture)
    classification = _classify(
        force_gate_reached=force_gate_reached,
        micro_capture_started=micro_capture_started,
        micro_capture_completed=micro_capture_completed,
        capture_aborted=capture_aborted,
        capture_abort_reason=capture_abort_reason,
        contact_none_ratio=contact_none_ratio,
        jump_count=jump_count,
        min_upright=min_upright,
        support_min_capture=support_min_capture,
        left_forward_delta=left_forward_delta,
        toe_handoff=(left_handoff + right_handoff) > 0,
        min_support_ratio_during_capture=min_support_ratio_during_capture,
    )
    return RefinementResult(
        trial_id=trial.trial_id,
        capture_forward_bias=trial.capture_forward_bias,
        capture_lateral_bias=trial.capture_lateral_bias,
        capture_duration_steps=trial.capture_duration_steps,
        capture_profile_shape=trial.capture_profile_shape,
        steps=len(rows),
        completed_steps=len(rows),
        phase_reached=rows[-1]["phase"] if rows else "none",
        micro_capture_started=micro_capture_started,
        micro_capture_completed=micro_capture_completed,
        capture_aborted=capture_aborted,
        capture_abort_reason=capture_abort_reason,
        capture_abort_step=capture_abort_step,
        force_gate_reached=force_gate_reached,
        force_gate_reached_step=force_gate_reached_step,
        force_gate_hold_steps_before_capture=gate_hold,
        mean_forward_velocity=float(np.mean([float(row["base_vx"]) for row in rows])) if rows else 0.0,
        forward_displacement=(float(rows[-1]["base_x"]) - float(rows[0]["base_x"])) if rows else 0.0,
        min_upright=min_upright,
        max_abs_roll=max((abs(float(row["base_roll"])) for row in rows), default=0.0),
        max_abs_pitch=max((abs(float(row["base_pitch"])) for row in rows), default=0.0),
        contact_none_ratio=contact_none_ratio,
        jump_count=jump_count,
        both_contact_ratio=states["both"] / max(1, len(rows)),
        single_contact_ratio=(states["left"] + states["right"]) / max(1, len(rows)),
        support_force_ratio_mean=float(np.mean(support_ratios)) if support_ratios else 0.0,
        support_force_ratio_peak=max(support_ratios, default=0.0),
        support_force_ratio_min_during_capture=support_min_capture,
        support_force_ratio_drop_during_capture=support_drop_capture,
        support_force_ratio_hold_steps_058=sum(_safe_gate(row, min_support_ratio_during_capture) for row in rows),
        max_continuous_hold_058=max_hold,
        swing_force_ratio_min=min(swing_ratios, default=0.0),
        left_foot_x_before_capture=left_before[0],
        left_foot_x_after_capture=left_after[0],
        left_foot_y_before_capture=left_before[1],
        left_foot_y_after_capture=left_after[1],
        left_foot_z_max=max((float(row["left_foot_z"]) for row in rows), default=0.0),
        left_foot_forward_delta=left_forward_delta,
        left_foot_lateral_delta=left_lateral_delta,
        left_foot_speed_peak_during_capture=speed_peak,
        toe_handoff_detected=(left_handoff + right_handoff) > 0,
        toe_handoff_left_count=left_handoff,
        toe_handoff_right_count=right_handoff,
        first_toe_handoff_step=first_handoff,
        delayed_fall_after_recover=_delayed_fall(rows),
        classification=classification,
        fail_reasons=_fail_reasons(
            classification,
            force_gate_reached=force_gate_reached,
            capture_aborted=capture_aborted,
            left_forward_delta=left_forward_delta,
            support_min_capture=support_min_capture,
            min_support_ratio_during_capture=min_support_ratio_during_capture,
            contact_none_ratio=contact_none_ratio,
            jump_count=jump_count,
            min_upright=min_upright,
        ),
    )


def _class_rank(row: RefinementResult) -> int:
    ranks = {
        "micro_capture_strong_safe": 0,
        "micro_capture_refined_safe": 1,
        "micro_capture_tiny_safe": 2,
        "toe_handoff_observed": 3,
        "no_capture_effect": 4,
        "micro_capture_aborted_gate_drop": 5,
        "micro_capture_unstable": 6,
        "gate_not_reached": 7,
    }
    return ranks.get(row.classification, 99)


def _sort_key(row: RefinementResult) -> tuple[int, float, float, int, float, int, float, int]:
    return (
        _class_rank(row),
        -row.left_foot_forward_delta,
        -row.support_force_ratio_min_during_capture,
        -row.max_continuous_hold_058,
        row.contact_none_ratio,
        row.jump_count,
        -row.min_upright,
        0 if row.toe_handoff_detected else 1,
    )


def _best(rows: Iterable[RefinementResult], classification: str) -> dict[str, Any] | None:
    selected = [row for row in rows if row.classification == classification]
    if not selected:
        return None
    return asdict(sorted(selected, key=_sort_key)[0])


def build_trials(
    forward_biases: list[float],
    lateral_biases: list[float],
    durations: list[int],
    shapes: list[str],
) -> list[RefinementTrial]:
    trials: list[RefinementTrial] = []
    index = 1
    for forward in forward_biases:
        for lateral in lateral_biases:
            for duration in durations:
                for shape in shapes:
                    trials.append(
                        RefinementTrial(
                            trial_id=f"p2b_{index:04d}",
                            capture_forward_bias=forward,
                            capture_lateral_bias=lateral,
                            capture_duration_steps=duration,
                            capture_profile_shape=shape,
                        )
                    )
                    index += 1
    return trials


def _write_csv(path: Path, rows: list[RefinementResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _interpret(rows: list[RefinementResult]) -> tuple[str, str]:
    safe = [row for row in rows if row.classification in {"micro_capture_refined_safe", "micro_capture_strong_safe"}]
    strong = [row for row in rows if row.classification == "micro_capture_strong_safe"]
    best = sorted(safe or rows, key=_sort_key)[0] if rows else None
    any_refined = bool(safe)
    any_strong = bool(strong)
    if any_strong:
        recommended = "Consider Phase 2C only as a force-gated contact/rollover probe, not full gait."
    elif any_refined:
        recommended = "Consider Phase 2C as a narrow refinement toward rollover observation, with the same abort gate."
    else:
        recommended = "Do not progress. Inspect left leg joint mapping, toe rocker, and contact patch before increasing capture bias."
    interpretation = (
        f"Phase 2B {'did' if any_refined else 'did not'} reach >=0.3 mm safe forward delta. "
        f"Strong >=1.0 mm safe capture={any_strong}. "
        f"Best capture_forward_bias={best.capture_forward_bias if best else 'none'}, "
        f"duration={best.capture_duration_steps if best else 'none'}, "
        f"profile={best.capture_profile_shape if best else 'none'}, "
        f"classification={best.classification if best else 'none'}. "
        f"Force gate {'remained stable' if any_refined else 'limited the refinement or no refined candidate emerged'}. "
        f"Toe handoff observed={any(row.toe_handoff_detected for row in rows)}. "
        f"Phase 2C recommendation: {'yes' if any_refined else 'no'}. Next focus: {recommended}"
    )
    return recommended, interpretation


def _append_progress_log(command: str, summary: dict[str, Any], out_dir: Path) -> None:
    PROGRESS_LOG.parent.mkdir(parents=True, exist_ok=True)
    if not PROGRESS_LOG.exists():
        PROGRESS_LOG.write_text("# Seedon Blue-Like Dynamic Gait Progress Log\n\n", encoding="utf-8")
    best = (
        summary.get("best_micro_capture_strong_safe")
        or summary.get("best_micro_capture_refined_safe")
        or summary.get("best_micro_capture_tiny_safe")
        or {}
    )
    result = "PASS" if summary["any_micro_capture_refined_safe"] else "FAIL"
    lines = [
        f"## {date.today().isoformat()} - Phase 2B Force-Gated Micro Capture Refinement",
        "",
        "### Goal",
        "",
        "Refine the left-leg micro capture intent under the right-support force gate, aiming to increase measurable left-foot forward reposition without breaking support force ratio, contact safety, or upright stability.",
        "",
        "### Files",
        "",
        "- `tools/seedon_phase2b_micro_capture_refinement.py`",
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
        f"- `{out_dir / 'phase2b_trials.csv'}`",
        f"- `{out_dir / 'phase2b_summary.json'}`",
        f"- `{out_dir / 'phase2b_top_candidates.csv'}`",
        "",
        "### Key Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| total_trials | {summary['total_trials']} |",
        f"| any_micro_capture_refined_safe | {str(summary['any_micro_capture_refined_safe']).lower()} |",
        f"| any_micro_capture_strong_safe | {str(summary['any_micro_capture_strong_safe']).lower()} |",
        f"| max_left_foot_forward_delta_safe | {summary['max_left_foot_forward_delta_safe']:.6f} |",
        f"| max_support_force_ratio_min_during_capture_safe | {summary['max_support_force_ratio_min_during_capture_safe']:.6f} |",
        f"| max_continuous_hold_058_safe | {summary['max_continuous_hold_058_safe']} |",
        f"| any_toe_handoff | {str(summary['any_toe_handoff']).lower()} |",
        f"| best_capture_forward_bias | {best.get('capture_forward_bias', 'none')} |",
        f"| best_capture_duration_steps | {best.get('capture_duration_steps', 'none')} |",
        f"| best_capture_profile_shape | {best.get('capture_profile_shape', 'none')} |",
        "",
        "### Result",
        "",
        result,
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


def run_sweep(
    *,
    scene_path: Path,
    out_dir: Path,
    steps: int,
    seed: int,
    forward_biases: list[float],
    lateral_biases: list[float],
    durations: list[int],
    shapes: list[str],
    min_support_ratio_during_capture: float,
    abort_on_force_gate_drop: bool,
    max_roll: float,
    max_pitch: float,
    command: str,
    update_progress_log: bool = True,
) -> dict[str, Any]:
    scene = require_scene(scene_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        run_trial(
            scene_path=scene,
            trial=trial,
            steps=steps,
            seed=seed,
            max_roll=max_roll,
            max_pitch=max_pitch,
            min_support_ratio_during_capture=min_support_ratio_during_capture,
            abort_on_force_gate_drop=abort_on_force_gate_drop,
        )
        for trial in build_trials(forward_biases, lateral_biases, durations, shapes)
    ]
    sorted_rows = sorted(rows, key=_sort_key)
    _write_csv(out_dir / "phase2b_trials.csv", rows)
    _write_csv(out_dir / "phase2b_top_candidates.csv", sorted_rows[:20])
    safe = [row for row in rows if row.classification in {"micro_capture_refined_safe", "micro_capture_strong_safe"}]
    recommended, interpretation = _interpret(rows)
    summary = {
        "total_trials": len(rows),
        "classification_counts": dict(Counter(row.classification for row in rows)),
        "best_micro_capture_strong_safe": _best(rows, "micro_capture_strong_safe"),
        "best_micro_capture_refined_safe": _best(rows, "micro_capture_refined_safe"),
        "best_micro_capture_tiny_safe": _best(rows, "micro_capture_tiny_safe"),
        "best_micro_capture_aborted_gate_drop": _best(rows, "micro_capture_aborted_gate_drop"),
        "best_micro_capture_unstable": _best(rows, "micro_capture_unstable"),
        "best_toe_handoff_candidate": _best(rows, "toe_handoff_observed"),
        "max_left_foot_forward_delta_safe": max((row.left_foot_forward_delta for row in safe), default=0.0),
        "max_support_force_ratio_min_during_capture_safe": max((row.support_force_ratio_min_during_capture for row in safe), default=0.0),
        "max_continuous_hold_058_safe": max((row.max_continuous_hold_058 for row in safe), default=0),
        "any_micro_capture_refined_safe": any(row.classification in {"micro_capture_refined_safe", "micro_capture_strong_safe"} for row in rows),
        "any_micro_capture_strong_safe": any(row.classification == "micro_capture_strong_safe" for row in rows),
        "any_toe_handoff": any(row.toe_handoff_detected for row in rows),
        "recommended_next_step": recommended,
        "engineering_interpretation": interpretation,
    }
    (out_dir / "phase2b_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if update_progress_log:
        _append_progress_log(command, summary, out_dir)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=280)
    parser.add_argument("--scene-path", type=Path, default=_default_scene())
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--capture-forward-biases", type=_parse_float_list, default="0.001,0.002,0.003,0.004,0.005,0.0075,0.010")
    parser.add_argument("--capture-lateral-biases", type=_parse_float_list, default="0.000,0.001,0.002")
    parser.add_argument("--capture-duration-steps-list", type=_parse_int_list, default="10,20,30,40,60")
    parser.add_argument("--capture-profile-shapes", type=_parse_shape_list, default="linear,smoothstep,minimum_jerk")
    parser.add_argument("--min-support-ratio-during-capture", type=float, default=0.58)
    parser.add_argument("--abort-on-force-gate-drop", type=_parse_bool, default=True)
    parser.add_argument("--max-roll", type=float, default=0.08)
    parser.add_argument("--max-pitch", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-progress-log", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = f".venv\\Scripts\\python.exe -m tools.seedon_phase2b_micro_capture_refinement --steps {args.steps}"
    summary = run_sweep(
        scene_path=args.scene_path,
        out_dir=args.out_dir,
        steps=args.steps,
        seed=args.seed,
        forward_biases=args.capture_forward_biases,
        lateral_biases=args.capture_lateral_biases,
        durations=args.capture_duration_steps_list,
        shapes=args.capture_profile_shapes,
        min_support_ratio_during_capture=args.min_support_ratio_during_capture,
        abort_on_force_gate_drop=args.abort_on_force_gate_drop,
        max_roll=args.max_roll,
        max_pitch=args.max_pitch,
        command=command,
        update_progress_log=not args.no_progress_log,
    )
    print(
        "phase2b trials={trials} refined_safe={refined} strong_safe={strong} "
        "max_delta_safe={delta:.6f} any_toe={toe}".format(
            trials=summary["total_trials"],
            refined=summary["any_micro_capture_refined_safe"],
            strong=summary["any_micro_capture_strong_safe"],
            delta=summary["max_left_foot_forward_delta_safe"],
            toe=summary["any_toe_handoff"],
        )
    )
    print(f"summary={args.out_dir / 'phase2b_summary.json'}")
    print(f"trials={args.out_dir / 'phase2b_trials.csv'}")
    print(f"top={args.out_dir / 'phase2b_top_candidates.csv'}")
    print(f"progress_log={PROGRESS_LOG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
