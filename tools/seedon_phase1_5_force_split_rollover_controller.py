"""Phase 1.5 force-split and rollover controller for Seedon.

This Class C tool is a deterministic MuJoCo controller/diagnostic. It does not
train PPO and does not attempt visible stepping. The goal is to bridge Phase 1
controlled falling and later capture stepping by testing bounded forward
commit, stable support/swing force split, contact continuity, and natural
center-to-toe rollover evidence.
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
from typing import Any

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
DEFAULT_OUT_DIR = DEBUG_OUT_DIR / "phase1_5_force_split_rollover"
PROGRESS_LOG = REPO_ROOT / "docs" / "seedon_blue_like_dynamic_gait_progress_log.md"

R_HIP_ROLL, R_HIP_PITCH = 1, 2
L_HIP_ROLL, L_HIP_PITCH = 6, 7
FOOT_FORCE_THRESHOLD_N = 0.1


class Phase15State(str, Enum):
    """Simple Phase 1.5 FSM without any full swing phase."""

    SETTLE = "SETTLE"
    COMMIT_FORWARD_LEFT_SUPPORT = "COMMIT_FORWARD_LEFT_SUPPORT"
    HOLD_FORCE_SPLIT_LEFT = "HOLD_FORCE_SPLIT_LEFT"
    RECOVER_CENTER = "RECOVER_CENTER"
    COMMIT_FORWARD_RIGHT_SUPPORT = "COMMIT_FORWARD_RIGHT_SUPPORT"
    HOLD_FORCE_SPLIT_RIGHT = "HOLD_FORCE_SPLIT_RIGHT"


@dataclass
class Phase15Runtime:
    """Mutable Phase 1.5 controller state."""

    phase: Phase15State = Phase15State.SETTLE
    phase_elapsed_steps: int = 0
    support_side: str = "left"
    swing_side: str = "right"
    gate_hold_steps: int = 0
    force_gate_reached_count: int = 0
    recover_count: int = 0
    completed_steps: int = 0
    previous_support_side: str = ""


@dataclass(frozen=True)
class Phase15Config:
    """Controller constants for one Phase 1.5 diagnostic run.

    Args:
        max_forward_bias: Maximum bounded forward base force in Newtons.
        forward_ramp_rate: Per-control-step ramp for forward bias.
        support_ratio_enter: Support-force ratio required to count force gate.
        support_ratio_exit: Lower support-force ratio tolerated while holding.
        max_roll: Maximum hip-roll load-transfer bias in radians.
        max_pitch: Maximum forward commit hip-pitch bias in radians.
        settle_steps: Initial neutral steps.
        min_hold_steps: Consecutive force-gated hold steps required.
        min_commit_steps: Minimum commit time before entering force hold.
        max_commit_steps: Timeout before recover if force split does not open.
        max_hold_steps: Timeout for hold before recover.
        recover_steps: Neutral recovery duration before switching side.
        hip_roll_ramp_rate: Per-step hip-roll ramp.
        pelvis_lean_fraction: Extra same-sign hip-roll lean fraction.
    """

    max_forward_bias: float
    forward_ramp_rate: float
    support_ratio_enter: float
    support_ratio_exit: float
    max_roll: float
    max_pitch: float
    settle_steps: int = 40
    min_hold_steps: int = 10
    min_commit_steps: int = 20
    max_commit_steps: int = 120
    max_hold_steps: int = 120
    recover_steps: int = 55
    hip_roll_ramp_rate: float = 0.001
    pelvis_lean_fraction: float = 0.35


@dataclass(frozen=True)
class Phase15Summary:
    """Aggregate Phase 1.5 controller metrics."""

    steps: int
    completed_steps: int
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
    support_force_ratio_hold_steps: int
    swing_force_ratio_min: float
    left_right_phase_switch_count: int
    toe_handoff_detected: bool
    toe_handoff_left_count: int
    toe_handoff_right_count: int
    force_gate_reached_count: int
    recover_count: int
    still_grounded_shuffle: bool
    phase1_5_passed: bool
    fail_reasons: str
    csv_path: str
    json_path: str


def _default_scene() -> Path:
    """Return preferred v5_a scene when present, otherwise standard Seedon scene."""

    return DEFAULT_V5A_SCENE if DEFAULT_V5A_SCENE.is_file() else DEFAULT_SCENE_PATH


def _other_side(side: str) -> str:
    """Return the opposite side."""

    if side == "left":
        return "right"
    if side == "right":
        return "left"
    raise ValueError(f"Unsupported side: {side}")


def _commit_phase(side: str) -> Phase15State:
    """Return the commit phase for a designated support side."""

    return (
        Phase15State.COMMIT_FORWARD_LEFT_SUPPORT
        if side == "left"
        else Phase15State.COMMIT_FORWARD_RIGHT_SUPPORT
    )


def _hold_phase(side: str) -> Phase15State:
    """Return the hold phase for a designated support side."""

    return (
        Phase15State.HOLD_FORCE_SPLIT_LEFT
        if side == "left"
        else Phase15State.HOLD_FORCE_SPLIT_RIGHT
    )


def _enter_phase(runtime: Phase15Runtime, phase: Phase15State, reason: str) -> str:
    """Switch phase and return the transition reason for logging."""

    if phase == Phase15State.RECOVER_CENTER:
        runtime.recover_count += 1
        runtime.gate_hold_steps = 0
    if phase in {
        Phase15State.COMMIT_FORWARD_LEFT_SUPPORT,
        Phase15State.HOLD_FORCE_SPLIT_LEFT,
    }:
        runtime.support_side = "left"
        runtime.swing_side = "right"
    elif phase in {
        Phase15State.COMMIT_FORWARD_RIGHT_SUPPORT,
        Phase15State.HOLD_FORCE_SPLIT_RIGHT,
    }:
        runtime.support_side = "right"
        runtime.swing_side = "left"
    runtime.phase = phase
    runtime.phase_elapsed_steps = 0
    return reason


def _contact_state(left_force: float, right_force: float) -> str:
    """Return force-thresholded contact state."""

    left = left_force > FOOT_FORCE_THRESHOLD_N
    right = right_force > FOOT_FORCE_THRESHOLD_N
    if left and right:
        return "both"
    if left:
        return "left"
    if right:
        return "right"
    return "none"


def _support_force_values(forces: dict[str, float | bool | int], support_side: str) -> tuple[float, float, float, float]:
    """Return support/swing forces and normalized ratios."""

    swing_side = _other_side(support_side)
    support_force = float(forces[f"{support_side}_force"])
    swing_force = float(forces[f"{swing_side}_force"])
    total = max(support_force + swing_force, 1e-9)
    return support_force, swing_force, support_force / total, swing_force / total


def _foot_region_contacts(env: SeedonStandingEnv) -> dict[str, bool]:
    """Return center/toe/heel contact booleans for both feet."""

    contacts = {
        "center_contact_left": False,
        "toe_contact_left": False,
        "heel_contact_left": False,
        "center_contact_right": False,
        "toe_contact_right": False,
        "heel_contact_right": False,
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
        body_pos = env.data.xpos[foot_body_id]
        body_xmat = env.data.xmat[foot_body_id].reshape(3, 3)
        local_pos = body_xmat.T @ (world_pos - body_pos)
        region = _region_for_geom(foot_name, float(local_pos[0]))
        key = f"{region}_contact_{side}"
        if key in contacts:
            contacts[key] = True
    return contacts


def _force_gate_ok(
    *,
    support_ratio: float,
    swing_ratio: float,
    contact_none: bool,
    jump: bool,
    upright: float,
    config: Phase15Config,
) -> bool:
    """Return whether the conservative force gate is currently satisfied."""

    return (
        support_ratio >= config.support_ratio_enter
        and swing_ratio <= (1.0 - config.support_ratio_enter)
        and not contact_none
        and not jump
        and upright >= 0.985
    )


def _stable_enough(row: dict[str, Any], config: Phase15Config) -> bool:
    """Return whether the current row is safe enough to continue commit/hold."""

    return (
        not bool(row["contact_none"])
        and not bool(row["jump"])
        and float(row["upright"]) >= 0.985
        and abs(float(row["base_roll"])) <= max(config.max_roll * 2.0, 0.08)
        and abs(float(row["base_pitch"])) <= max(config.max_pitch * 4.0, 0.16)
    )


def _biases(runtime: Phase15Runtime, config: Phase15Config) -> tuple[float, float, float]:
    """Return forward, hip-roll, and pelvis-lean biases for current phase."""

    if runtime.phase == Phase15State.SETTLE:
        return 0.0, 0.0, 0.0
    if runtime.phase == Phase15State.RECOVER_CENTER:
        alpha = max(0.0, 1.0 - runtime.phase_elapsed_steps / max(config.recover_steps, 1))
        return 0.0, config.max_roll * 0.25 * alpha, config.max_roll * config.pelvis_lean_fraction * 0.25 * alpha
    forward_bias = min(config.max_forward_bias, config.forward_ramp_rate * runtime.phase_elapsed_steps)
    hip_roll_bias = min(config.max_roll, config.hip_roll_ramp_rate * runtime.phase_elapsed_steps)
    pelvis_lean_bias = min(config.max_roll * config.pelvis_lean_fraction, hip_roll_bias * config.pelvis_lean_fraction)
    if runtime.phase in {Phase15State.HOLD_FORCE_SPLIT_LEFT, Phase15State.HOLD_FORCE_SPLIT_RIGHT}:
        hip_roll_bias = config.max_roll
        pelvis_lean_bias = config.max_roll * config.pelvis_lean_fraction
    return forward_bias, hip_roll_bias, pelvis_lean_bias


def _target(env: SeedonStandingEnv, runtime: Phase15Runtime, config: Phase15Config) -> tuple[np.ndarray, float, float, float]:
    """Return joint target and debug biases for the current controller step."""

    target = env._nominal_joint_qpos.copy()
    forward_bias, hip_roll_bias, pelvis_lean_bias = _biases(runtime, config)
    if runtime.phase != Phase15State.SETTLE:
        pitch_alpha = min(1.0, runtime.phase_elapsed_steps / max(config.max_commit_steps, 1))
        target[R_HIP_PITCH] += config.max_pitch * pitch_alpha
        target[L_HIP_PITCH] += config.max_pitch * pitch_alpha
    if runtime.support_side == "left":
        target[R_HIP_ROLL] += 0.45 * hip_roll_bias + pelvis_lean_bias
        target[L_HIP_ROLL] += hip_roll_bias + pelvis_lean_bias
    else:
        target[R_HIP_ROLL] -= hip_roll_bias + pelvis_lean_bias
        target[L_HIP_ROLL] -= 0.45 * hip_roll_bias + pelvis_lean_bias
    return env._apply_safe_joint_target_clamps(target), forward_bias, hip_roll_bias, pelvis_lean_bias


def _advance(
    runtime: Phase15Runtime,
    row: dict[str, Any] | None,
    config: Phase15Config,
) -> str:
    """Advance the simple force-gated FSM."""

    reason = ""
    if runtime.phase == Phase15State.SETTLE:
        if runtime.phase_elapsed_steps >= config.settle_steps:
            reason = _enter_phase(runtime, _commit_phase("left"), "settle_complete")
        else:
            runtime.phase_elapsed_steps += 1
        return reason

    if row is None:
        runtime.phase_elapsed_steps += 1
        return reason

    stable = _stable_enough(row, config)
    support_ratio = float(row["support_force_ratio"])
    swing_ratio = float(row["swing_force_ratio"])
    force_gate_ok = _force_gate_ok(
        support_ratio=support_ratio,
        swing_ratio=swing_ratio,
        contact_none=bool(row["contact_none"]),
        jump=bool(row["jump"]),
        upright=float(row["upright"]),
        config=config,
    )

    if runtime.phase in {
        Phase15State.COMMIT_FORWARD_LEFT_SUPPORT,
        Phase15State.COMMIT_FORWARD_RIGHT_SUPPORT,
    }:
        if not stable:
            return _enter_phase(runtime, Phase15State.RECOVER_CENTER, "safety_recover")
        if runtime.phase_elapsed_steps >= config.min_commit_steps and force_gate_ok:
            return _enter_phase(runtime, _hold_phase(runtime.support_side), "force_gate_enter_hold")
        if runtime.phase_elapsed_steps >= config.max_commit_steps:
            return _enter_phase(runtime, Phase15State.RECOVER_CENTER, "commit_timeout_recover")

    elif runtime.phase in {
        Phase15State.HOLD_FORCE_SPLIT_LEFT,
        Phase15State.HOLD_FORCE_SPLIT_RIGHT,
    }:
        if not stable:
            return _enter_phase(runtime, Phase15State.RECOVER_CENTER, "safety_recover")
        if support_ratio < config.support_ratio_exit:
            runtime.gate_hold_steps = 0
        elif force_gate_ok:
            runtime.gate_hold_steps += 1
        if runtime.gate_hold_steps >= config.min_hold_steps:
            runtime.force_gate_reached_count += 1
            runtime.completed_steps += 1
            return _enter_phase(runtime, Phase15State.RECOVER_CENTER, "force_gate_reached_recover")
        if runtime.phase_elapsed_steps >= config.max_hold_steps:
            return _enter_phase(runtime, Phase15State.RECOVER_CENTER, "hold_timeout_recover")

    elif runtime.phase == Phase15State.RECOVER_CENTER:
        if runtime.phase_elapsed_steps >= config.recover_steps:
            next_side = _other_side(runtime.support_side)
            return _enter_phase(runtime, _commit_phase(next_side), "recover_complete_switch_support")

    runtime.phase_elapsed_steps += 1
    return reason


def _count_bursts(rows: list[dict[str, Any]], key: str) -> int:
    """Count contiguous truthy bursts for a row key."""

    count = 0
    in_burst = False
    for row in rows:
        active = bool(row[key])
        if active and not in_burst:
            count += 1
            in_burst = True
        elif not active:
            in_burst = False
    return count


def _support_switches(rows: list[dict[str, Any]]) -> int:
    """Count left/right designated support side switches."""

    switches = 0
    previous = ""
    for row in rows:
        side = str(row["support_side"])
        if previous and side != previous:
            switches += 1
        previous = side
    return switches


def _toe_handoff_counts(rows: list[dict[str, Any]]) -> tuple[int, int]:
    """Count center-to-toe handoff events per side."""

    counts = {"left": 0, "right": 0}
    saw_center = {"left": False, "right": False}
    toe_active = {"left": False, "right": False}
    for row in rows:
        for side in ("left", "right"):
            center = bool(row[f"center_contact_{side}"])
            toe = bool(row[f"toe_contact_{side}"])
            if center:
                saw_center[side] = True
            if saw_center[side] and toe and not toe_active[side]:
                counts[side] += 1
                toe_active[side] = True
                saw_center[side] = False
            if not toe:
                toe_active[side] = False
    return counts["left"], counts["right"]


def _summarize(
    rows: list[dict[str, Any]],
    runtime: Phase15Runtime,
    csv_path: Path,
    json_path: Path,
) -> Phase15Summary:
    """Aggregate timeline rows into the requested summary JSON payload."""

    if not rows:
        return Phase15Summary(
            steps=0,
            completed_steps=0,
            mean_forward_velocity=0.0,
            forward_displacement=0.0,
            min_upright=0.0,
            max_abs_roll=0.0,
            max_abs_pitch=0.0,
            contact_none_ratio=1.0,
            jump_count=0,
            both_contact_ratio=0.0,
            single_contact_ratio=0.0,
            support_force_ratio_mean=0.0,
            support_force_ratio_peak=0.0,
            support_force_ratio_hold_steps=0,
            swing_force_ratio_min=0.0,
            left_right_phase_switch_count=0,
            toe_handoff_detected=False,
            toe_handoff_left_count=0,
            toe_handoff_right_count=0,
            force_gate_reached_count=0,
            recover_count=0,
            still_grounded_shuffle=True,
            phase1_5_passed=False,
            fail_reasons="empty_rollout",
            csv_path=str(csv_path),
            json_path=str(json_path),
        )

    states = Counter(str(row["contact_state"]) for row in rows)
    support_ratios = [float(row["support_force_ratio"]) for row in rows]
    swing_ratios = [float(row["swing_force_ratio"]) for row in rows]
    contact_none_ratio = states["none"] / len(rows)
    jump_count = _count_bursts(rows, "jump")
    support_hold_steps = sum(float(row["support_force_ratio"]) >= 0.58 for row in rows)
    toe_left, toe_right = _toe_handoff_counts(rows)
    mean_vx = float(np.mean([float(row["base_vx"]) for row in rows]))
    forward_displacement = float(rows[-1]["base_x"]) - float(rows[0]["base_x"])
    min_upright = min(float(row["upright"]) for row in rows)
    switches = _support_switches(rows)
    fail_reasons: list[str] = []
    if contact_none_ratio != 0.0:
        fail_reasons.append("contact_none")
    if jump_count != 0:
        fail_reasons.append("jump")
    if min_upright < 0.985:
        fail_reasons.append("upright")
    if mean_vx <= 0.0:
        fail_reasons.append("mean_forward_velocity")
    if forward_displacement <= 0.0:
        fail_reasons.append("forward_displacement")
    if max(support_ratios) < 0.58:
        fail_reasons.append("support_force_ratio_peak")
    if support_hold_steps < 10:
        fail_reasons.append("support_force_ratio_hold_steps")
    if switches < 1:
        fail_reasons.append("left_right_phase_switch")

    single_contact_ratio = (states["left"] + states["right"]) / len(rows)
    return Phase15Summary(
        steps=len(rows),
        completed_steps=runtime.completed_steps,
        mean_forward_velocity=mean_vx,
        forward_displacement=forward_displacement,
        min_upright=min_upright,
        max_abs_roll=max(abs(float(row["base_roll"])) for row in rows),
        max_abs_pitch=max(abs(float(row["base_pitch"])) for row in rows),
        contact_none_ratio=contact_none_ratio,
        jump_count=jump_count,
        both_contact_ratio=states["both"] / len(rows),
        single_contact_ratio=single_contact_ratio,
        support_force_ratio_mean=float(np.mean(support_ratios)),
        support_force_ratio_peak=max(support_ratios),
        support_force_ratio_hold_steps=support_hold_steps,
        swing_force_ratio_min=min(swing_ratios),
        left_right_phase_switch_count=switches,
        toe_handoff_detected=(toe_left + toe_right) > 0,
        toe_handoff_left_count=toe_left,
        toe_handoff_right_count=toe_right,
        force_gate_reached_count=runtime.force_gate_reached_count,
        recover_count=runtime.recover_count,
        still_grounded_shuffle=single_contact_ratio < 0.05,
        phase1_5_passed=not fail_reasons,
        fail_reasons=",".join(fail_reasons),
        csv_path=str(csv_path),
        json_path=str(json_path),
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write timeline rows to CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _append_progress_log(command: str, summary: Phase15Summary, output_dir: Path) -> None:
    """Append this run to the human-readable Blue-like gait progress log."""

    PROGRESS_LOG.parent.mkdir(parents=True, exist_ok=True)
    if not PROGRESS_LOG.exists():
        PROGRESS_LOG.write_text(
            "# Seedon Blue-Like Dynamic Gait Progress Log\n\n"
            "This append-only log tracks Seedon Blue / BDX-like dynamic gait experiments, "
            "diagnostics, controller results, and engineering decisions.\n\n"
            "Core decisions:\n\n"
            "- Grounded shuffle is only the Phase 0 baseline, not final walking.\n"
            "- The target remains Blue / BDX-like dynamic gait.\n"
            "- Visible stepping cannot be solved by lift scale or landing interpolation alone.\n"
            "- The current bridge is controlled forward commit, support/swing force split, and toe rollover.\n"
            "- Force split must be established before any full SWING_FORWARD phase is allowed.\n"
            "- Toe handoff must be observed from contact / force, not hard-coded.\n"
            "- Base x displacement alone is never sufficient to claim gait success.\n\n"
            "## 2026-05-29 - Phase 0 v5_a Geometry And Teacher Baseline\n\n"
            "### Goal\n\n"
            "Establish the current grounded / slow forward shuffle baseline and best known sole geometry.\n\n"
            "### Files\n\n"
            "- `artifacts/seedon_debug/blue_like_sole_experiments_v5/training_scene_v5_a.xml`\n"
            "- `models/seedon/teacher_safe_baseline/`\n\n"
            "### Key Metrics\n\n"
            "| Metric | Value |\n"
            "|---|---:|\n"
            "| v5_a center-first standing | observed |\n"
            "| dynamic push center-to-toe handoff | observed previously |\n"
            "| teacher pipeline | grounded shuffle |\n\n"
            "### Result\n\n"
            "Baseline only. Not walking.\n\n"
            "### Engineering Interpretation\n\n"
            "v5_a remains the best candidate geometry, but grounded shuffle must not be promoted to dynamic gait.\n\n"
            "### Next Decision\n\n"
            "Use v5_a for controlled forward commit and force split diagnostics.\n\n"
            "## 2026-05-29 - Phase 1 Rollover Diagnostic\n\n"
            "### Goal\n\n"
            "Check whether forward momentum can produce controlled falling, contact continuity, rollover, and support alternation without requiring visible clearance.\n\n"
            "### Files\n\n"
            "- `tools/seedon_blue_like_phase1_rollover_diagnostic.py`\n\n"
            "### Command\n\n"
            "```powershell\n"
            ".venv\\Scripts\\python.exe -m tools.seedon_blue_like_phase1_rollover_diagnostic --steps 600\n"
            "```\n\n"
            "### Outputs\n\n"
            "- `artifacts/seedon_debug/blue_like_phase1_rollover/phase1_rollover_timeline.csv`\n"
            "- `artifacts/seedon_debug/blue_like_phase1_rollover/phase1_rollover_summary.json`\n\n"
            "### Key Metrics\n\n"
            "| Metric | Value |\n"
            "|---|---:|\n"
            "| steps | 105 |\n"
            "| mean_forward_velocity | 0.1126 |\n"
            "| forward_displacement | 0.1556 |\n"
            "| min_upright | 0.7160 |\n"
            "| contact_none_ratio | 0.0952 |\n"
            "| jump_count | 2 |\n"
            "| support_force_ratio_peak | 1.0 |\n"
            "| toe_handoff_detected | false |\n"
            "| left_right_phase_switch_count | 4 |\n\n"
            "### Result\n\n"
            "FAIL: `contact_none,jump,upright,toe_handoff`.\n\n"
            "### Engineering Interpretation\n\n"
            "The probe creates forward momentum, but it is too uncontrolled and collapses into fall/no-contact. This is not a gait success.\n\n"
            "### Next Decision\n\n"
            "Replace raw dynamic push with bounded forward commit and force-gated recovery.\n\n"
            "## 2026-05-29 - Capture V1 Controller Skeleton\n\n"
            "### Goal\n\n"
            "Create a first capture-step debug skeleton without claiming walking success.\n\n"
            "### Files\n\n"
            "- `tools/seedon_capture_step_controller_v1.py`\n\n"
            "### Command\n\n"
            "```powershell\n"
            ".venv\\Scripts\\python.exe -m tools.seedon_capture_step_controller_v1 --steps 600\n"
            "```\n\n"
            "### Outputs\n\n"
            "- `artifacts/seedon_debug/capture_step_controller_v1/capture_step_controller_v1.csv`\n"
            "- `artifacts/seedon_debug/capture_step_controller_v1/capture_step_controller_v1_summary.json`\n\n"
            "### Key Metrics\n\n"
            "| Metric | Value |\n"
            "|---|---:|\n"
            "| forward_displacement | 0.0355 |\n"
            "| mean_forward_velocity | 0.0031 |\n"
            "| min_upright | 0.9927 |\n"
            "| contact_none_ratio | 0.0 |\n"
            "| jump_count | 0 |\n"
            "| support_force_ratio_mean | 0.5019 |\n"
            "| swing_force_ratio_min | 0.4647 |\n"
            "| toe_handoff_detected | false |\n"
            "| still_grounded_shuffle | true |\n\n"
            "### Result\n\n"
            "FAIL as dynamic gait. Safe but too conservative.\n\n"
            "### Engineering Interpretation\n\n"
            "The FSM is stable but does not open support/swing force split. It remains grounded shuffle.\n\n"
            "### Next Decision\n\n"
            "Do not enter full capture step. Build a Phase 1.5 force-split + rollover controller first.\n\n",
            encoding="utf-8",
        )

    lines = [
        f"## {date.today().isoformat()} - Phase 1.5 Force Split + Rollover Controller",
        "",
        "### Goal",
        "",
        "Test bounded forward commit, stable support/swing force split, contact continuity, and natural rollover before any visible stepping.",
        "",
        "### Files",
        "",
        "- `tools/seedon_phase1_5_force_split_rollover_controller.py`",
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
        f"- `{output_dir / 'phase1_5_timeline.csv'}`",
        f"- `{output_dir / 'phase1_5_summary.json'}`",
        "",
        "### Key Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| contact_none_ratio | {summary.contact_none_ratio:.6f} |",
        f"| jump_count | {summary.jump_count} |",
        f"| min_upright | {summary.min_upright:.6f} |",
        f"| mean_forward_velocity | {summary.mean_forward_velocity:.6f} |",
        f"| forward_displacement | {summary.forward_displacement:.6f} |",
        f"| support_force_ratio_peak | {summary.support_force_ratio_peak:.6f} |",
        f"| support_force_ratio_hold_steps | {summary.support_force_ratio_hold_steps} |",
        f"| swing_force_ratio_min | {summary.swing_force_ratio_min:.6f} |",
        f"| force_gate_reached_count | {summary.force_gate_reached_count} |",
        f"| left_right_phase_switch_count | {summary.left_right_phase_switch_count} |",
        f"| toe_handoff_detected | {str(summary.toe_handoff_detected).lower()} |",
        f"| toe_handoff_left_count | {summary.toe_handoff_left_count} |",
        f"| toe_handoff_right_count | {summary.toe_handoff_right_count} |",
        f"| still_grounded_shuffle | {str(summary.still_grounded_shuffle).lower()} |",
        "",
        "### Result",
        "",
        "PASS" if summary.phase1_5_passed else f"FAIL: `{summary.fail_reasons}`",
        "",
        "### Engineering Interpretation",
        "",
        (
            "Phase 1.5 opened the force-split gate without no-contact or jump. This is still not walking, but it is enough to consider a force-gated micro-swing next."
            if summary.phase1_5_passed
            else "Phase 1.5 did not satisfy the force-split bridge. Do not proceed to capture stepping until support ratio can hold above the gate safely."
        ),
        "",
        "### Next Decision",
        "",
        (
            "Consider Phase 2 force-gated capture-step micro swing, still with strict contact and impact diagnostics."
            if summary.phase1_5_passed
            else "Debug hip-roll direction, pelvis lean authority, support hip-roll contact-force effect, contact patch rollover, stance width, COM height, and v5_a controlled-commit rollover."
        ),
        "",
    ]
    with PROGRESS_LOG.open("a", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")


def run_controller(
    *,
    scene_path: Path,
    out_dir: Path,
    steps: int,
    seed: int,
    config: Phase15Config,
    command: str,
    update_progress_log: bool = True,
) -> Phase15Summary:
    """Run deterministic Phase 1.5 force-split controller.

    Args:
        scene_path: MuJoCo XML scene to load.
        out_dir: Directory for CSV and JSON outputs.
        steps: Number of control steps.
        seed: Deterministic reset seed.
        config: Controller constants.
        command: User-facing command string for progress-log recording.
        update_progress_log: Whether to append the markdown progress log.

    Returns:
        Aggregate Phase 1.5 summary.

    Raises:
        FileNotFoundError: If the scene is missing.
        ValueError: If inputs are outside supported ranges.
    """

    if steps <= 0:
        raise ValueError("steps must be positive.")
    if config.support_ratio_enter <= config.support_ratio_exit:
        raise ValueError("support_ratio_enter must be greater than support_ratio_exit.")
    scene = require_scene(scene_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "phase1_5_timeline.csv"
    json_path = out_dir / "phase1_5_summary.json"
    env = SeedonStandingEnv(
        scene_path=scene,
        reset_noise_scale=0.0,
        reward_config=load_seedon_config_from_env(),
    )
    runtime = Phase15Runtime()
    rows: list[dict[str, Any]] = []
    try:
        env.reset(seed=seed)
        initial_x = float(env.data.qpos[0])
        robot_weight = float(np.sum(env.model.body_mass) * 9.81)
        previous_row: dict[str, Any] | None = None
        for step in range(1, steps + 1):
            transition_reason = _advance(runtime, previous_row, config)
            target, forward_bias, hip_roll_bias, pelvis_lean_bias = _target(env, runtime, config)
            env.data.xfrc_applied[:] = 0.0
            env.data.xfrc_applied[env._base_body_id, 0] = forward_bias
            env._do_pd_simulation(target)
            env.data.xfrc_applied[:] = 0.0
            env._gait_step += 1

            forces = _foot_contact_forces(env)
            contacts = _foot_region_contacts(env)
            left_force = float(forces["left_force"])
            right_force = float(forces["right_force"])
            support_force, swing_force, support_ratio, swing_ratio = _support_force_values(
                forces,
                runtime.support_side,
            )
            contact_state = _contact_state(left_force, right_force)
            total_force = left_force + right_force
            contact_none = contact_state == "none"
            jump = bool(contact_none or (float(env.data.qvel[2]) > 0.05 and total_force < 0.10 * robot_weight))
            row = {
                "step": step,
                "time": float(env.data.time),
                "phase": runtime.phase.value,
                "support_side": runtime.support_side,
                "swing_side": runtime.swing_side,
                "base_x": float(env.data.qpos[0]),
                "base_y": float(env.data.qpos[1]),
                "base_z": float(env._base_height()),
                "base_vx": float(env.data.qvel[0]),
                "base_vy": float(env.data.qvel[1]),
                "base_roll": float(env._base_roll()),
                "base_pitch": float(env._base_pitch()),
                "upright": float(env._base_upright()),
                "left_force": left_force,
                "right_force": right_force,
                "support_force": support_force,
                "swing_force": swing_force,
                "support_force_ratio": support_ratio,
                "swing_force_ratio": swing_ratio,
                "center_contact_left": contacts["center_contact_left"],
                "toe_contact_left": contacts["toe_contact_left"],
                "heel_contact_left": contacts["heel_contact_left"],
                "center_contact_right": contacts["center_contact_right"],
                "toe_contact_right": contacts["toe_contact_right"],
                "heel_contact_right": contacts["heel_contact_right"],
                "toe_handoff_left": False,
                "toe_handoff_right": False,
                "contact_state": contact_state,
                "contact_none": contact_none,
                "jump": jump,
                "forward_bias": forward_bias,
                "hip_roll_bias": hip_roll_bias,
                "pelvis_lean_bias": pelvis_lean_bias,
                "phase_elapsed_steps": runtime.phase_elapsed_steps,
                "phase_transition_reason": transition_reason,
                "forward_displacement": float(env.data.qpos[0]) - initial_x,
            }
            rows.append(row)
            previous_row = row
    finally:
        env.data.xfrc_applied[:] = 0.0
        env.close()

    left_count, right_count = _toe_handoff_counts(rows)
    # Mark the first detected handoff rows for easier CSV scanning.
    for side, count in (("left", left_count), ("right", right_count)):
        if count <= 0:
            continue
        saw_center = False
        marked = 0
        for row in rows:
            if bool(row[f"center_contact_{side}"]):
                saw_center = True
            if saw_center and bool(row[f"toe_contact_{side}"]):
                row[f"toe_handoff_{side}"] = True
                marked += 1
                saw_center = False
                if marked >= count:
                    break
    _write_csv(csv_path, rows)
    summary = _summarize(rows, runtime, csv_path, json_path)
    json_path.write_text(json.dumps(asdict(summary), indent=2) + "\n", encoding="utf-8")
    if update_progress_log:
        _append_progress_log(command, summary, out_dir)
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-path", type=Path, default=_default_scene())
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-forward-bias", type=float, default=2.0)
    parser.add_argument("--forward-ramp-rate", type=float, default=0.04)
    parser.add_argument("--support-ratio-enter", type=float, default=0.58)
    parser.add_argument("--support-ratio-exit", type=float, default=0.54)
    parser.add_argument("--max-roll", type=float, default=0.055)
    parser.add_argument("--max-pitch", type=float, default=0.012)
    parser.add_argument("--no-progress-log", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the Phase 1.5 diagnostic from CLI."""

    args = build_parser().parse_args(argv)
    config = Phase15Config(
        max_forward_bias=args.max_forward_bias,
        forward_ramp_rate=args.forward_ramp_rate,
        support_ratio_enter=args.support_ratio_enter,
        support_ratio_exit=args.support_ratio_exit,
        max_roll=args.max_roll,
        max_pitch=args.max_pitch,
    )
    command = (
        ".venv\\Scripts\\python.exe -m tools.seedon_phase1_5_force_split_rollover_controller "
        f"--steps {args.steps} --max-forward-bias {args.max_forward_bias} "
        f"--forward-ramp-rate {args.forward_ramp_rate} --support-ratio-enter {args.support_ratio_enter} "
        f"--support-ratio-exit {args.support_ratio_exit} --max-roll {args.max_roll} --max-pitch {args.max_pitch}"
    )
    summary = run_controller(
        scene_path=args.scene_path,
        out_dir=args.out_dir,
        steps=args.steps,
        seed=args.seed,
        config=config,
        command=command,
        update_progress_log=not args.no_progress_log,
    )
    print(
        "phase1_5_pass={passed} forward={forward:.4f} mean_vx={vx:.4f} "
        "upright={upright:.4f} support_peak={peak:.4f} hold_steps={hold} "
        "gates={gates} switches={switches} toe_handoff={toe} reasons={reasons}".format(
            passed=summary.phase1_5_passed,
            forward=summary.forward_displacement,
            vx=summary.mean_forward_velocity,
            upright=summary.min_upright,
            peak=summary.support_force_ratio_peak,
            hold=summary.support_force_ratio_hold_steps,
            gates=summary.force_gate_reached_count,
            switches=summary.left_right_phase_switch_count,
            toe=summary.toe_handoff_detected,
            reasons=summary.fail_reasons or "-",
        )
    )
    print(f"csv={summary.csv_path}")
    print(f"json={summary.json_path}")
    print(f"progress_log={PROGRESS_LOG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
