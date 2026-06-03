"""Phase 2A right-support force-gated micro capture diagnostic for Seedon.

This Class C tool tests whether a tiny left-leg capture intent can be added
after the Phase 1.7 right-support force gate is established. It does not train,
does not enter full swing, and does not claim walking.
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
DEFAULT_OUT_DIR = DEBUG_OUT_DIR / "phase2a_right_support_micro_capture"
PROGRESS_LOG = REPO_ROOT / "docs" / "seedon_blue_like_dynamic_gait_progress_log.md"

R_HIP_ROLL, R_HIP_PITCH = 1, 2
L_HIP_ROLL, L_HIP_PITCH = 6, 7
FOOT_FORCE_THRESHOLD_N = 0.1


class Phase2AState(str, Enum):
    """Conservative right-support micro-capture FSM."""

    SETTLE = "SETTLE"
    RAMP_IN_RIGHT_SUPPORT = "RAMP_IN_RIGHT_SUPPORT"
    HOLD_RIGHT_SUPPORT_FORCE_GATE = "HOLD_RIGHT_SUPPORT_FORCE_GATE"
    MICRO_CAPTURE_LEFT = "MICRO_CAPTURE_LEFT"
    RECOVER_CENTER = "RECOVER_CENTER"
    CENTER_SETTLE = "CENTER_SETTLE"


@dataclass(frozen=True)
class CaptureTrial:
    """One Phase 2A micro-capture trial."""

    trial_id: str
    capture_forward_bias: float
    capture_lateral_bias: float
    capture_duration_steps: int
    capture_profile_shape: str


@dataclass(frozen=True)
class CaptureResult:
    """Flat trial result for Phase 2A."""

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
    toe_handoff_detected: bool
    toe_handoff_left_count: int
    toe_handoff_right_count: int
    first_toe_handoff_step: int
    delayed_fall_after_recover: bool
    classification: str
    fail_reasons: str


def _default_scene() -> Path:
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


def _shape(shape: str, alpha: float) -> float:
    t = float(np.clip(alpha, 0.0, 1.0))
    if shape == "linear":
        return t
    if shape == "smoothstep":
        return t * t * (3.0 - 2.0 * t)
    if shape == "minimum_jerk":
        return t * t * t * (10.0 - 15.0 * t + 6.0 * t * t)
    raise ValueError(f"Unsupported shape: {shape}")


def _profile_alpha(step: int, duration: int, shape: str) -> float:
    return _shape(shape, step / max(duration, 1))


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


def _target(
    env: SeedonStandingEnv,
    *,
    phase: Phase2AState,
    phase_step: int,
    trial: CaptureTrial,
    max_roll: float,
    max_pitch: float,
    ramp_steps: int,
    hold_steps: int,
    recover_steps: int,
) -> np.ndarray:
    """Return joint target for the conservative Phase 2A FSM."""

    target = env._nominal_joint_qpos.copy()
    if phase == Phase2AState.SETTLE or phase == Phase2AState.CENTER_SETTLE:
        load_alpha = 0.0
        capture_alpha = 0.0
    elif phase == Phase2AState.RAMP_IN_RIGHT_SUPPORT:
        load_alpha = phase_step / max(ramp_steps, 1)
        capture_alpha = 0.0
    elif phase == Phase2AState.HOLD_RIGHT_SUPPORT_FORCE_GATE:
        load_alpha = 1.0
        capture_alpha = 0.0
    elif phase == Phase2AState.MICRO_CAPTURE_LEFT:
        load_alpha = 1.0
        capture_alpha = _profile_alpha(phase_step, trial.capture_duration_steps, trial.capture_profile_shape)
    elif phase == Phase2AState.RECOVER_CENTER:
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


def _support_ratios(forces: dict[str, float | bool | int]) -> tuple[float, float, float, float]:
    support_force = float(forces["right_force"])
    swing_force = float(forces["left_force"])
    total = max(support_force + swing_force, 1e-9)
    return support_force, swing_force, support_force / total, swing_force / total


def _safe_gate(row: dict[str, Any]) -> bool:
    return (
        float(row["support_force_ratio"]) >= 0.58
        and float(row["swing_force_ratio"]) <= 0.42
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


def _max_continuous_hold(rows: list[dict[str, Any]]) -> int:
    best = 0
    current = 0
    for row in rows:
        if _safe_gate(row):
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
    pre = [row for row in rows if str(row["phase"]) in {"RAMP_IN_RIGHT_SUPPORT", "HOLD_RIGHT_SUPPORT_FORCE_GATE", "MICRO_CAPTURE_LEFT"}]
    post = [row for row in rows if str(row["phase"]) in {"RECOVER_CENTER", "CENTER_SETTLE"}]
    pre_stable = bool(pre) and all(not row["contact_none"] and not row["jump"] and float(row["upright"]) >= 0.985 for row in pre)
    post_fail = any(row["contact_none"] or row["jump"] or float(row["upright"]) < 0.985 for row in post)
    return pre_stable and post_fail


def _classify(
    *,
    force_gate_reached: bool,
    micro_capture_started: bool,
    micro_capture_completed: bool,
    contact_none_ratio: float,
    jump_count: int,
    min_upright: float,
    max_hold_058: int,
    left_forward_delta: float,
    support_min_capture: float,
    toe_handoff: bool,
) -> str:
    if micro_capture_started and (contact_none_ratio > 0.0 or jump_count > 0 or min_upright < 0.985):
        return "micro_capture_unstable"
    if force_gate_reached and micro_capture_started and support_min_capture < 0.55:
        return "micro_capture_disrupts_force_gate"
    if (
        force_gate_reached
        and micro_capture_completed
        and contact_none_ratio == 0.0
        and jump_count == 0
        and min_upright >= 0.985
        and max_hold_058 >= 10
        and left_forward_delta > 0.0
    ):
        return "micro_capture_safe"
    if toe_handoff:
        return "toe_handoff_observed"
    if micro_capture_completed and left_forward_delta <= 0.0:
        return "no_capture_effect"
    return "gate_not_reached"


def _fail_reasons(
    result_class: str,
    *,
    force_gate_reached: bool,
    left_forward_delta: float,
    support_min_capture: float,
    contact_none_ratio: float,
    jump_count: int,
    min_upright: float,
) -> str:
    reasons: list[str] = []
    if not force_gate_reached:
        reasons.append("force_gate_not_reached")
    if left_forward_delta <= 0.0:
        reasons.append("no_forward_delta")
    if support_min_capture < 0.55:
        reasons.append("force_gate_disrupted")
    if contact_none_ratio > 0.0:
        reasons.append("contact_none")
    if jump_count > 0:
        reasons.append("jump")
    if min_upright < 0.985:
        reasons.append("upright")
    if result_class == "micro_capture_safe":
        return ""
    return ",".join(reasons)


def run_trial(
    *,
    scene_path: Path,
    trial: CaptureTrial,
    steps: int,
    seed: int,
    max_roll: float,
    max_pitch: float,
    settle_steps: int = 20,
    ramp_steps: int = 10,
    hold_steps: int = 80,
    recover_steps: int = 40,
    center_settle_steps: int = 40,
) -> CaptureResult:
    """Run one right-support micro-capture trial."""

    env = SeedonStandingEnv(scene_path=scene_path, reset_noise_scale=0.0, reward_config=load_seedon_config_from_env())
    rows: list[dict[str, Any]] = []
    phase = Phase2AState.SETTLE
    phase_step = 0
    gate_hold = 0
    force_gate_reached = False
    force_gate_reached_step = -1
    micro_capture_started = False
    micro_capture_completed = False
    left_before = (float("nan"), float("nan"), float("nan"))
    left_after = (float("nan"), float("nan"), float("nan"))
    try:
        env.reset(seed=seed)
        initial_x = float(env.data.qpos[0])
        robot_weight = float(np.sum(env.model.body_mass) * 9.81)
        for step in range(1, steps + 1):
            if phase == Phase2AState.SETTLE and phase_step >= settle_steps:
                phase = Phase2AState.RAMP_IN_RIGHT_SUPPORT
                phase_step = 0
            elif phase == Phase2AState.RAMP_IN_RIGHT_SUPPORT and phase_step >= ramp_steps:
                phase = Phase2AState.HOLD_RIGHT_SUPPORT_FORCE_GATE
                phase_step = 0
            elif phase == Phase2AState.HOLD_RIGHT_SUPPORT_FORCE_GATE:
                if gate_hold >= 10:
                    phase = Phase2AState.MICRO_CAPTURE_LEFT
                    phase_step = 0
                    micro_capture_started = True
                    left_before = _left_foot_pos(env)
                elif phase_step >= hold_steps:
                    phase = Phase2AState.RECOVER_CENTER
                    phase_step = 0
            elif phase == Phase2AState.MICRO_CAPTURE_LEFT and phase_step >= trial.capture_duration_steps:
                micro_capture_completed = True
                left_after = _left_foot_pos(env)
                phase = Phase2AState.RECOVER_CENTER
                phase_step = 0
            elif phase == Phase2AState.RECOVER_CENTER and phase_step >= recover_steps:
                phase = Phase2AState.CENTER_SETTLE
                phase_step = 0

            target = _target(
                env,
                phase=phase,
                phase_step=phase_step,
                trial=trial,
                max_roll=max_roll,
                max_pitch=max_pitch,
                ramp_steps=ramp_steps,
                hold_steps=hold_steps,
                recover_steps=recover_steps,
            )
            env._do_pd_simulation(target)
            env._gait_step += 1
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
                "left_foot_x": _left_foot_pos(env)[0],
                "left_foot_y": _left_foot_pos(env)[1],
                "left_foot_z": _left_foot_pos(env)[2],
                "forward_displacement": float(env.data.qpos[0]) - initial_x,
            }
            rows.append(row)
            if phase == Phase2AState.HOLD_RIGHT_SUPPORT_FORCE_GATE:
                if _safe_gate(row):
                    gate_hold += 1
                    if not force_gate_reached and gate_hold >= 10:
                        force_gate_reached = True
                        force_gate_reached_step = step
                else:
                    gate_hold = 0
            phase_step += 1
            if phase == Phase2AState.CENTER_SETTLE and phase_step >= center_settle_steps:
                break
    finally:
        env.close()

    if micro_capture_started and not micro_capture_completed:
        left_after = _left_foot_pos(env) if rows else left_before
    states = Counter(str(row["contact_state"]) for row in rows)
    support_ratios = [float(row["support_force_ratio"]) for row in rows]
    swing_ratios = [float(row["swing_force_ratio"]) for row in rows]
    capture_rows = [row for row in rows if str(row["phase"]) == Phase2AState.MICRO_CAPTURE_LEFT.value]
    support_min_capture = min((float(row["support_force_ratio"]) for row in capture_rows), default=1.0)
    left_x_before = left_before[0]
    left_y_before = left_before[1]
    if micro_capture_completed:
        left_x_after = left_after[0]
        left_y_after = left_after[1]
    elif capture_rows:
        left_x_after = float(capture_rows[-1]["left_foot_x"])
        left_y_after = float(capture_rows[-1]["left_foot_y"])
    else:
        left_x_after = left_x_before
        left_y_after = left_y_before
    left_forward_delta = left_x_after - left_x_before if np.isfinite(left_x_before) else 0.0
    left_lateral_delta = left_y_after - left_y_before if np.isfinite(left_y_before) else 0.0
    left_handoff, right_handoff, first_handoff = _toe_handoff_counts(rows)
    contact_none_ratio = states["none"] / max(1, len(rows))
    jump_count = _count_bursts(rows, "jump")
    min_upright = min((float(row["upright"]) for row in rows), default=0.0)
    max_hold_058 = _max_continuous_hold(rows)
    classification = _classify(
        force_gate_reached=force_gate_reached,
        micro_capture_started=micro_capture_started,
        micro_capture_completed=micro_capture_completed,
        contact_none_ratio=contact_none_ratio,
        jump_count=jump_count,
        min_upright=min_upright,
        max_hold_058=max_hold_058,
        left_forward_delta=left_forward_delta,
        support_min_capture=support_min_capture,
        toe_handoff=(left_handoff + right_handoff) > 0,
    )
    return CaptureResult(
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
        support_force_ratio_hold_steps_058=sum(_safe_gate(row) for row in rows),
        max_continuous_hold_058=max_hold_058,
        swing_force_ratio_min=min(swing_ratios, default=0.0),
        left_foot_x_before_capture=left_x_before,
        left_foot_x_after_capture=left_x_after,
        left_foot_y_before_capture=left_y_before,
        left_foot_y_after_capture=left_y_after,
        left_foot_z_max=max((float(row["left_foot_z"]) for row in rows), default=0.0),
        left_foot_forward_delta=left_forward_delta,
        left_foot_lateral_delta=left_lateral_delta,
        toe_handoff_detected=(left_handoff + right_handoff) > 0,
        toe_handoff_left_count=left_handoff,
        toe_handoff_right_count=right_handoff,
        first_toe_handoff_step=first_handoff,
        delayed_fall_after_recover=_delayed_fall(rows),
        classification=classification,
        fail_reasons=_fail_reasons(
            classification,
            force_gate_reached=force_gate_reached,
            left_forward_delta=left_forward_delta,
            support_min_capture=support_min_capture,
            contact_none_ratio=contact_none_ratio,
            jump_count=jump_count,
            min_upright=min_upright,
        ),
    )


def _sort_key(row: CaptureResult) -> tuple[int, float, float, int, float, int, float, int]:
    return (
        0 if row.classification == "micro_capture_safe" else 1,
        -row.left_foot_forward_delta,
        -row.support_force_ratio_min_during_capture,
        -row.max_continuous_hold_058,
        row.contact_none_ratio,
        row.jump_count,
        -row.min_upright,
        0 if row.toe_handoff_detected else 1,
    )


def _best(rows: Iterable[CaptureResult], classification: str) -> dict[str, Any] | None:
    selected = [row for row in rows if row.classification == classification]
    if not selected:
        return None
    return asdict(sorted(selected, key=_sort_key)[0])


def build_trials(
    forward_biases: list[float],
    lateral_biases: list[float],
    durations: list[int],
    shapes: list[str],
) -> list[CaptureTrial]:
    trials: list[CaptureTrial] = []
    index = 1
    for forward in forward_biases:
        for lateral in lateral_biases:
            for duration in durations:
                for shape in shapes:
                    trials.append(
                        CaptureTrial(
                            trial_id=f"p2a_{index:04d}",
                            capture_forward_bias=forward,
                            capture_lateral_bias=lateral,
                            capture_duration_steps=duration,
                            capture_profile_shape=shape,
                        )
                    )
                    index += 1
    return trials


def _write_csv(path: Path, rows: list[CaptureResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _interpret(rows: list[CaptureResult]) -> tuple[str, str]:
    safe = [row for row in rows if row.classification == "micro_capture_safe"]
    best = sorted(safe or rows, key=_sort_key)[0] if rows else None
    any_safe = bool(safe)
    if any_safe:
        recommended = "Consider Phase 2B only as another force-gated micro probe, not full walking."
    else:
        recommended = "Do not enter Phase 2B. Shrink capture intent or adjust timing around force-gate hold."
    interpretation = (
        f"Phase 2A {'can' if any_safe else 'cannot'} add left-leg micro capture after right-support gate. "
        f"Best capture_forward_bias={best.capture_forward_bias if best else 'none'}, "
        f"capture_lateral_bias={best.capture_lateral_bias if best else 'none'}, "
        f"duration={best.capture_duration_steps if best else 'none'}, "
        f"classification={best.classification if best else 'none'}. "
        f"Micro capture {'preserved' if any_safe else 'did not preserve'} the force split. "
        f"Left foot forward delta best={best.left_foot_forward_delta if best else 0.0:.6f}. "
        f"Toe handoff observed={any(row.toe_handoff_detected for row in rows)}. "
        f"Phase 2B recommendation: {'yes' if any_safe else 'no'}. Next focus: {recommended}"
    )
    return recommended, interpretation


def _append_progress_log(command: str, summary: dict[str, Any], out_dir: Path) -> None:
    PROGRESS_LOG.parent.mkdir(parents=True, exist_ok=True)
    if not PROGRESS_LOG.exists():
        PROGRESS_LOG.write_text("# Seedon Blue-Like Dynamic Gait Progress Log\n\n", encoding="utf-8")
    best = summary.get("best_micro_capture_safe") or summary.get("best_no_capture_effect") or {}
    result = "PASS" if summary["any_micro_capture_safe"] else "FAIL"
    lines = [
        f"## {date.today().isoformat()} - Phase 2A Right-Support Force-Gated Micro Capture",
        "",
        "### Goal",
        "",
        "Test whether a very small left-leg micro capture intent can be safely added after the right-support force gate is established.",
        "",
        "### Files",
        "",
        "- `tools/seedon_phase2a_right_support_micro_capture.py`",
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
        f"- `{out_dir / 'phase2a_trials.csv'}`",
        f"- `{out_dir / 'phase2a_summary.json'}`",
        f"- `{out_dir / 'phase2a_top_candidates.csv'}`",
        "",
        "### Key Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| total_trials | {summary['total_trials']} |",
        f"| any_micro_capture_safe | {str(summary['any_micro_capture_safe']).lower()} |",
        f"| max_left_foot_forward_delta_safe | {summary['max_left_foot_forward_delta_safe']:.6f} |",
        f"| max_support_force_ratio_min_during_capture_safe | {summary['max_support_force_ratio_min_during_capture_safe']:.6f} |",
        f"| max_continuous_hold_058_safe | {summary['max_continuous_hold_058_safe']} |",
        f"| any_toe_handoff | {str(summary['any_toe_handoff']).lower()} |",
        f"| best_capture_forward_bias | {best.get('capture_forward_bias', 'none')} |",
        f"| best_capture_lateral_bias | {best.get('capture_lateral_bias', 'none')} |",
        f"| best_capture_duration_steps | {best.get('capture_duration_steps', 'none')} |",
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
        )
        for trial in build_trials(forward_biases, lateral_biases, durations, shapes)
    ]
    sorted_rows = sorted(rows, key=_sort_key)
    _write_csv(out_dir / "phase2a_trials.csv", rows)
    _write_csv(out_dir / "phase2a_top_candidates.csv", sorted_rows[:20])
    safe = [row for row in rows if row.classification == "micro_capture_safe"]
    recommended, interpretation = _interpret(rows)
    summary = {
        "total_trials": len(rows),
        "classification_counts": dict(Counter(row.classification for row in rows)),
        "best_micro_capture_safe": _best(rows, "micro_capture_safe"),
        "best_no_capture_effect": _best(rows, "no_capture_effect"),
        "best_micro_capture_disrupts_force_gate": _best(rows, "micro_capture_disrupts_force_gate"),
        "best_micro_capture_unstable": _best(rows, "micro_capture_unstable"),
        "best_toe_handoff_candidate": _best(rows, "toe_handoff_observed"),
        "max_left_foot_forward_delta_safe": max((row.left_foot_forward_delta for row in safe), default=0.0),
        "max_support_force_ratio_min_during_capture_safe": max((row.support_force_ratio_min_during_capture for row in safe), default=0.0),
        "max_continuous_hold_058_safe": max((row.max_continuous_hold_058 for row in safe), default=0),
        "any_micro_capture_safe": bool(safe),
        "any_toe_handoff": any(row.toe_handoff_detected for row in rows),
        "recommended_next_step": recommended,
        "engineering_interpretation": interpretation,
    }
    (out_dir / "phase2a_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if update_progress_log:
        _append_progress_log(command, summary, out_dir)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=260)
    parser.add_argument("--scene-path", type=Path, default=_default_scene())
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--capture-forward-biases", type=_parse_float_list, default="0.001,0.002,0.003,0.004,0.005")
    parser.add_argument("--capture-lateral-biases", type=_parse_float_list, default="0.000,0.001,0.002")
    parser.add_argument("--capture-duration-steps-list", type=_parse_int_list, default="5,10,15,20")
    parser.add_argument("--capture-profile-shapes", type=_parse_shape_list, default="linear,smoothstep,minimum_jerk")
    parser.add_argument("--max-roll", type=float, default=0.08)
    parser.add_argument("--max-pitch", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-progress-log", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = f".venv\\Scripts\\python.exe -m tools.seedon_phase2a_right_support_micro_capture --steps {args.steps}"
    summary = run_sweep(
        scene_path=args.scene_path,
        out_dir=args.out_dir,
        steps=args.steps,
        seed=args.seed,
        forward_biases=args.capture_forward_biases,
        lateral_biases=args.capture_lateral_biases,
        durations=args.capture_duration_steps_list,
        shapes=args.capture_profile_shapes,
        max_roll=args.max_roll,
        max_pitch=args.max_pitch,
        command=command,
        update_progress_log=not args.no_progress_log,
    )
    print(
        "phase2a trials={trials} any_safe={safe} max_forward_delta_safe={delta:.6f} any_toe={toe}".format(
            trials=summary["total_trials"],
            safe=summary["any_micro_capture_safe"],
            delta=summary["max_left_foot_forward_delta_safe"],
            toe=summary["any_toe_handoff"],
        )
    )
    print(f"summary={args.out_dir / 'phase2a_summary.json'}")
    print(f"trials={args.out_dir / 'phase2a_trials.csv'}")
    print(f"top={args.out_dir / 'phase2a_top_candidates.csv'}")
    print(f"progress_log={PROGRESS_LOG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
