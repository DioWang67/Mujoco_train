"""Phase 1.7 load-transfer profile shaping for Sedon.

This Class C diagnostic shapes the best Phase 1.6 channel
``pelvis_lean+swing_hip_roll`` with deterministic ramp/hold/recover profiles.
It does not train PPO, does not enter capture stepping, and does not claim
walking from grounded displacement.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable, Iterable

import mujoco
import numpy as np

from sedon_baseline.env import SedonStandingEnv, load_sedon_config_from_env
from tools.sedon_blue_like_phase1_rollover_diagnostic import (
    _foot_contact_forces,
    _region_for_geom,
    _side_for_geom,
)
from tools.sedon_debug_common import DEBUG_OUT_DIR, DEFAULT_SCENE_PATH, geom_name, require_scene


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V5A_SCENE = DEBUG_OUT_DIR / "blue_like_sole_experiments_v5" / "training_scene_v5_a.xml"
DEFAULT_OUT_DIR = DEBUG_OUT_DIR / "phase1_7_load_transfer_profile_shaping"
PROGRESS_LOG = REPO_ROOT / "docs" / "sedon_blue_like_dynamic_gait_progress_log.md"

R_HIP_ROLL = 1
L_HIP_ROLL = 6
FOOT_FORCE_THRESHOLD_N = 0.1


@dataclass(frozen=True)
class ProfileTrial:
    """One deterministic Phase 1.7 profile trial."""

    trial_id: str
    support_side: str
    magnitude: float
    pelvis_scale: float
    swing_hip_roll_scale: float
    ramp_steps: int
    hold_steps: int
    recover_steps: int
    profile_shape: str


@dataclass(frozen=True)
class ProfileResult:
    """Flat trial-level result row."""

    trial_id: str
    support_side: str
    magnitude: float
    pelvis_scale: float
    swing_hip_roll_scale: float
    ramp_steps: int
    hold_steps: int
    recover_steps: int
    profile_shape: str
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
    support_force_ratio_hold_steps_055: int
    support_force_ratio_hold_steps_058: int
    support_force_ratio_hold_steps_060: int
    max_continuous_hold_055: int
    max_continuous_hold_058: int
    max_continuous_hold_060: int
    swing_force_ratio_min: float
    force_gate_055_reached: bool
    force_gate_058_reached: bool
    force_gate_060_reached: bool
    toe_handoff_detected: bool
    toe_handoff_left_count: int
    toe_handoff_right_count: int
    first_toe_handoff_step: int
    delayed_fall_after_recover: bool
    classification: str
    fail_reasons: str


def _default_scene() -> Path:
    """Return preferred v5_a scene when available."""

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


def _parse_str_list(raw: str) -> list[str]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one value")
    allowed = {"linear", "smoothstep", "minimum_jerk"}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise argparse.ArgumentTypeError(f"unsupported profile shapes: {unknown}")
    return values


def _other_side(side: str) -> str:
    if side == "left":
        return "right"
    if side == "right":
        return "left"
    raise ValueError(f"Unsupported side: {side}")


def _shape_alpha(shape: str, alpha: float) -> float:
    t = float(np.clip(alpha, 0.0, 1.0))
    if shape == "linear":
        return t
    if shape == "smoothstep":
        return t * t * (3.0 - 2.0 * t)
    if shape == "minimum_jerk":
        return t * t * t * (10.0 - 15.0 * t + 6.0 * t * t)
    raise ValueError(f"Unsupported profile shape: {shape}")


def _phase_and_alpha(trial: ProfileTrial, step: int, settle_steps: int, center_settle_steps: int) -> tuple[str, float]:
    local = step - settle_steps
    if local <= 0:
        return "SETTLE", 0.0
    if local <= trial.ramp_steps:
        return "RAMP_IN", _shape_alpha(trial.profile_shape, local / max(trial.ramp_steps, 1))
    local -= trial.ramp_steps
    if local <= trial.hold_steps:
        return "HOLD", 1.0
    local -= trial.hold_steps
    if local <= trial.recover_steps:
        return "RECOVER", 1.0 - _shape_alpha(trial.profile_shape, local / max(trial.recover_steps, 1))
    if local <= trial.recover_steps + center_settle_steps:
        return "CENTER_SETTLE", 0.0
    return "CENTER_SETTLE", 0.0


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


def _foot_region_contacts(env: SedonStandingEnv) -> dict[str, bool]:
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
        body_pos = env.data.xpos[foot_body_id]
        body_xmat = env.data.xmat[foot_body_id].reshape(3, 3)
        local_pos = body_xmat.T @ (world_pos - body_pos)
        region = _region_for_geom(foot_name, float(local_pos[0]))
        if region in {"center", "toe"}:
            contacts[f"{region}_contact_{side}"] = True
    return contacts


def _target(
    env: SedonStandingEnv,
    trial: ProfileTrial,
    alpha: float,
    max_roll: float,
) -> tuple[np.ndarray, float, float]:
    pelvis = float(np.clip(trial.magnitude * trial.pelvis_scale * alpha, -max_roll, max_roll))
    swing_roll = float(np.clip(trial.magnitude * trial.swing_hip_roll_scale * alpha, -max_roll, max_roll))
    target = env._nominal_joint_qpos.copy()
    swing_side = _other_side(trial.support_side)
    if trial.support_side == "right":
        target[R_HIP_ROLL] -= pelvis
        target[L_HIP_ROLL] -= pelvis
    else:
        target[R_HIP_ROLL] += pelvis
        target[L_HIP_ROLL] += pelvis
    if swing_side == "right":
        target[R_HIP_ROLL] += swing_roll
    else:
        target[L_HIP_ROLL] -= swing_roll
    return env._apply_safe_joint_target_clamps(target), pelvis, swing_roll


def _support_ratios(forces: dict[str, float | bool | int], support_side: str) -> tuple[float, float, float, float]:
    swing_side = _other_side(support_side)
    support_force = float(forces[f"{support_side}_force"])
    swing_force = float(forces[f"{swing_side}_force"])
    total = max(support_force + swing_force, 1e-9)
    return support_force, swing_force, support_force / total, swing_force / total


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


def _safe_hold(row: dict[str, Any], threshold: float) -> bool:
    return (
        float(row["support_force_ratio"]) >= threshold
        and float(row["upright"]) >= 0.985
        and not bool(row["contact_none"])
        and not bool(row["jump"])
    )


def _cumulative_hold(rows: list[dict[str, Any]], threshold: float) -> int:
    return sum(_safe_hold(row, threshold) for row in rows)


def _max_continuous_hold(rows: list[dict[str, Any]], threshold: float) -> int:
    best = 0
    current = 0
    for row in rows:
        if _safe_hold(row, threshold):
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
    pre = [row for row in rows if str(row["profile_phase"]) in {"RAMP_IN", "HOLD"}]
    post = [row for row in rows if str(row["profile_phase"]) in {"RECOVER", "CENTER_SETTLE"}]
    pre_stable = bool(pre) and all(
        not bool(row["contact_none"]) and not bool(row["jump"]) and float(row["upright"]) >= 0.985
        for row in pre
    )
    post_fail = any(
        bool(row["contact_none"]) or bool(row["jump"]) or float(row["upright"]) < 0.985
        for row in post
    )
    return pre_stable and post_fail


def _classify(
    *,
    delayed_fall_after_recover: bool,
    contact_none_ratio: float,
    jump_count: int,
    min_upright: float,
    support_peak: float,
    max_hold_055: int,
    max_hold_058: int,
    toe_handoff: bool,
) -> str:
    fall = contact_none_ratio > 0.0 or jump_count > 0 or min_upright < 0.985
    if delayed_fall_after_recover:
        return "delayed_fall"
    if fall and support_peak >= 0.58:
        return "unstable_profile_split"
    if max_hold_058 >= 10 and not fall:
        return "stable_profile_split"
    if not fall and support_peak >= 0.58 and 3 <= max_hold_058 < 10:
        return "stable_near_profile_split"
    if toe_handoff and max_hold_058 < 10:
        return "toe_handoff_only"
    if not fall and max_hold_055 >= 20 and support_peak < 0.58:
        return "stable_055_hold"
    return "stable_no_split" if not fall else "fall_or_jump"


def _fail_reasons(
    *,
    contact_none_ratio: float,
    jump_count: int,
    min_upright: float,
    max_hold_058: int,
    delayed_fall_after_recover: bool,
) -> str:
    reasons: list[str] = []
    if contact_none_ratio > 0.0:
        reasons.append("contact_none")
    if jump_count > 0:
        reasons.append("jump")
    if min_upright < 0.985:
        reasons.append("upright")
    if max_hold_058 < 10:
        reasons.append("max_continuous_hold_058")
    if delayed_fall_after_recover:
        reasons.append("delayed_fall_after_recover")
    return ",".join(reasons)


def run_trial(
    *,
    scene_path: Path,
    trial: ProfileTrial,
    steps: int,
    seed: int,
    max_roll: float,
    settle_steps: int = 20,
    center_settle_steps: int = 40,
) -> ProfileResult:
    """Run one deterministic profile-shaping trial."""

    env = SedonStandingEnv(scene_path=scene_path, reset_noise_scale=0.0, reward_config=load_sedon_config_from_env())
    rows: list[dict[str, Any]] = []
    try:
        env.reset(seed=seed)
        initial_x = float(env.data.qpos[0])
        robot_weight = float(np.sum(env.model.body_mass) * 9.81)
        for step in range(1, steps + 1):
            phase, alpha = _phase_and_alpha(trial, step, settle_steps, center_settle_steps)
            target, pelvis_bias, swing_bias = _target(env, trial, alpha, max_roll)
            env._do_pd_simulation(target)
            env._gait_step += 1
            forces = _foot_contact_forces(env)
            contacts = _foot_region_contacts(env)
            left_force = float(forces["left_force"])
            right_force = float(forces["right_force"])
            support_force, swing_force, support_ratio, swing_ratio = _support_ratios(forces, trial.support_side)
            contact_state = _contact_state(left_force, right_force)
            total_force = left_force + right_force
            contact_none = contact_state == "none"
            jump = bool(contact_none or (float(env.data.qvel[2]) > 0.05 and total_force < 0.10 * robot_weight))
            rows.append(
                {
                    "step": step,
                    "profile_phase": phase,
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
                    "pelvis_lean_bias": pelvis_bias,
                    "swing_hip_roll_bias": swing_bias,
                    "forward_displacement": float(env.data.qpos[0]) - initial_x,
                }
            )
    finally:
        env.close()

    states = Counter(str(row["contact_state"]) for row in rows)
    support_ratios = [float(row["support_force_ratio"]) for row in rows]
    swing_ratios = [float(row["swing_force_ratio"]) for row in rows]
    left_handoff, right_handoff, first_handoff = _toe_handoff_counts(rows)
    contact_none_ratio = states["none"] / max(1, len(rows))
    jump_count = _count_bursts(rows, "jump")
    min_upright = min((float(row["upright"]) for row in rows), default=0.0)
    hold_055 = _cumulative_hold(rows, 0.55)
    hold_058 = _cumulative_hold(rows, 0.58)
    hold_060 = _cumulative_hold(rows, 0.60)
    max_hold_055 = _max_continuous_hold(rows, 0.55)
    max_hold_058 = _max_continuous_hold(rows, 0.58)
    max_hold_060 = _max_continuous_hold(rows, 0.60)
    delayed = _delayed_fall(rows)
    toe_handoff = (left_handoff + right_handoff) > 0
    support_peak = max(support_ratios, default=0.0)
    classification = _classify(
        delayed_fall_after_recover=delayed,
        contact_none_ratio=contact_none_ratio,
        jump_count=jump_count,
        min_upright=min_upright,
        support_peak=support_peak,
        max_hold_055=max_hold_055,
        max_hold_058=max_hold_058,
        toe_handoff=toe_handoff,
    )
    return ProfileResult(
        trial_id=trial.trial_id,
        support_side=trial.support_side,
        magnitude=trial.magnitude,
        pelvis_scale=trial.pelvis_scale,
        swing_hip_roll_scale=trial.swing_hip_roll_scale,
        ramp_steps=trial.ramp_steps,
        hold_steps=trial.hold_steps,
        recover_steps=trial.recover_steps,
        profile_shape=trial.profile_shape,
        steps=len(rows),
        completed_steps=len(rows),
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
        support_force_ratio_peak=support_peak,
        support_force_ratio_hold_steps_055=hold_055,
        support_force_ratio_hold_steps_058=hold_058,
        support_force_ratio_hold_steps_060=hold_060,
        max_continuous_hold_055=max_hold_055,
        max_continuous_hold_058=max_hold_058,
        max_continuous_hold_060=max_hold_060,
        swing_force_ratio_min=min(swing_ratios, default=0.0),
        force_gate_055_reached=max_hold_055 >= 10,
        force_gate_058_reached=max_hold_058 >= 10,
        force_gate_060_reached=max_hold_060 >= 10,
        toe_handoff_detected=toe_handoff,
        toe_handoff_left_count=left_handoff,
        toe_handoff_right_count=right_handoff,
        first_toe_handoff_step=first_handoff,
        delayed_fall_after_recover=delayed,
        classification=classification,
        fail_reasons=_fail_reasons(
            contact_none_ratio=contact_none_ratio,
            jump_count=jump_count,
            min_upright=min_upright,
            max_hold_058=max_hold_058,
            delayed_fall_after_recover=delayed,
        ),
    )


def _classification_rank(row: ProfileResult) -> int:
    order = {
        "stable_profile_split": 0,
        "stable_near_profile_split": 1,
        "stable_055_hold": 2,
        "toe_handoff_only": 3,
        "stable_no_split": 4,
        "unstable_profile_split": 5,
        "delayed_fall": 6,
        "fall_or_jump": 7,
    }
    return order.get(row.classification, 99)


def _sort_key(row: ProfileResult) -> tuple[int, int, int, float, float, int, float, int]:
    return (
        _classification_rank(row),
        -row.max_continuous_hold_058,
        -row.support_force_ratio_hold_steps_058,
        -row.support_force_ratio_peak,
        row.contact_none_ratio,
        row.jump_count,
        -row.min_upright,
        0 if row.toe_handoff_detected else 1,
    )


def _best(rows: Iterable[ProfileResult], classification: str) -> dict[str, Any] | None:
    selected = [row for row in rows if row.classification == classification]
    if not selected:
        return None
    return asdict(sorted(selected, key=_sort_key)[0])


def build_trials(
    *,
    magnitudes: list[float],
    ramp_steps_list: list[int],
    hold_steps_list: list[int],
    recover_steps_list: list[int],
    profile_shapes: list[str],
    pelvis_scales: list[float],
    swing_hip_roll_scales: list[float],
) -> list[ProfileTrial]:
    trials: list[ProfileTrial] = []
    index = 1
    for magnitude in magnitudes:
        for ramp_steps in ramp_steps_list:
            for hold_steps in hold_steps_list:
                for recover_steps in recover_steps_list:
                    for shape in profile_shapes:
                        for pelvis_scale in pelvis_scales:
                            for swing_scale in swing_hip_roll_scales:
                                for support_side in ("left", "right"):
                                    trials.append(
                                        ProfileTrial(
                                            trial_id=f"p17_{index:04d}",
                                            support_side=support_side,
                                            magnitude=magnitude,
                                            pelvis_scale=pelvis_scale,
                                            swing_hip_roll_scale=swing_scale,
                                            ramp_steps=ramp_steps,
                                            hold_steps=hold_steps,
                                            recover_steps=recover_steps,
                                            profile_shape=shape,
                                        )
                                    )
                                    index += 1
    return trials


def _write_csv(path: Path, rows: list[ProfileResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _mode(rows: list[ProfileResult], getter: Callable[[ProfileResult], Any]) -> Any:
    if not rows:
        return "none"
    return Counter(getter(row) for row in rows).most_common(1)[0][0]


def _interpret(rows: list[ProfileResult]) -> tuple[str, str]:
    stable = [row for row in rows if row.classification == "stable_profile_split"]
    near = [row for row in rows if row.classification == "stable_near_profile_split"]
    safe_rows = [row for row in rows if row.contact_none_ratio == 0.0 and row.jump_count == 0 and row.min_upright >= 0.985]
    best = sorted(stable or near or safe_rows or rows, key=_sort_key)[0] if rows else None
    max_hold = max((row.max_continuous_hold_058 for row in rows), default=0)
    max_cum = max((row.support_force_ratio_hold_steps_058 for row in rows), default=0)
    any_stable = bool(stable)
    if any_stable:
        recommended = "Consider Phase 2 force-gated micro capture only with this profile and strict safety gates."
    elif near:
        recommended = "Do not enter Phase 2. Refine around the best near profile to reach continuous hold_058 >= 10."
    else:
        recommended = "Do not enter Phase 2. Continue controller shaping and inspect toe rocker/contact patch if rollover remains absent."
    best_shape = best.profile_shape if best else "none"
    best_mag = best.magnitude if best else 0.0
    left_best = max((row.max_continuous_hold_058 for row in safe_rows if row.support_side == "left"), default=0)
    right_best = max((row.max_continuous_hold_058 for row in safe_rows if row.support_side == "right"), default=0)
    symmetry = "roughly symmetric" if abs(left_best - right_best) <= 2 else f"asymmetric left={left_best}, right={right_best}"
    interpretation = (
        f"Profile shaping {'improved hold' if max_hold > 2 else 'did not materially improve hold'}: "
        f"max_continuous_hold_058={max_hold}, max_cumulative_hold_058={max_cum}. "
        f"Best profile shape={best_shape}, magnitude={best_mag}. "
        f"Ramp/hold/recover trend from best candidate: ramp={best.ramp_steps if best else 'n/a'}, "
        f"hold={best.hold_steps if best else 'n/a'}, recover={best.recover_steps if best else 'n/a'}. "
        f"Magnitude is bounded by stability and hold continuity, not simply larger-is-better. "
        f"Support symmetry is {symmetry}. Phase 2 recommendation: {'yes' if any_stable else 'no'}. "
        f"Next focus: {recommended}"
    )
    return recommended, interpretation


def _append_progress_log(command: str, summary: dict[str, Any], out_dir: Path) -> None:
    PROGRESS_LOG.parent.mkdir(parents=True, exist_ok=True)
    if not PROGRESS_LOG.exists():
        PROGRESS_LOG.write_text("# Sedon Blue-Like Dynamic Gait Progress Log\n\n", encoding="utf-8")
    best = summary.get("best_stable_profile_split") or summary.get("best_stable_near_profile_split") or summary.get("best_stable_055_hold") or {}
    result = "PASS" if summary["any_stable_profile_split"] else ("INCONCLUSIVE" if summary["max_continuous_hold_058"] >= 3 else "FAIL")
    lines = [
        f"## {date.today().isoformat()} - Phase 1.7 Load Transfer Profile Shaping",
        "",
        "### Goal",
        "",
        "Shape the pelvis_lean+swing_hip_roll load-transfer profile so that support_force_ratio >= 0.58 can be held continuously without contact-none, jump, or upright failure.",
        "",
        "### Files",
        "",
        "- `tools/sedon_phase1_7_load_transfer_profile_shaping.py`",
        "- `docs/sedon_blue_like_dynamic_gait_progress_log.md`",
        "",
        "### Command",
        "",
        "```powershell",
        command,
        "```",
        "",
        "### Outputs",
        "",
        f"- `{out_dir / 'phase1_7_trials.csv'}`",
        f"- `{out_dir / 'phase1_7_summary.json'}`",
        f"- `{out_dir / 'phase1_7_top_candidates.csv'}`",
        "",
        "### Key Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| total_trials | {summary['total_trials']} |",
        f"| any_stable_profile_split | {str(summary['any_stable_profile_split']).lower()} |",
        f"| max_stable_support_force_ratio_peak | {summary['max_stable_support_force_ratio_peak']:.6f} |",
        f"| max_continuous_hold_058 | {summary['max_continuous_hold_058']} |",
        f"| max_cumulative_hold_058 | {summary['max_cumulative_hold_058']} |",
        f"| best_profile_shape | {best.get('profile_shape', 'none')} |",
        f"| best_magnitude | {best.get('magnitude', 'none')} |",
        f"| any_toe_handoff | {str(summary['any_toe_handoff']).lower()} |",
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
    magnitudes: list[float],
    ramp_steps_list: list[int],
    hold_steps_list: list[int],
    recover_steps_list: list[int],
    profile_shapes: list[str],
    pelvis_scales: list[float],
    swing_hip_roll_scales: list[float],
    max_roll: float,
    command: str,
    update_progress_log: bool = True,
) -> dict[str, Any]:
    if steps <= 0:
        raise ValueError("steps must be positive.")
    scene = require_scene(scene_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    trials = build_trials(
        magnitudes=magnitudes,
        ramp_steps_list=ramp_steps_list,
        hold_steps_list=hold_steps_list,
        recover_steps_list=recover_steps_list,
        profile_shapes=profile_shapes,
        pelvis_scales=pelvis_scales,
        swing_hip_roll_scales=swing_hip_roll_scales,
    )
    rows = [
        run_trial(
            scene_path=scene,
            trial=trial,
            steps=steps,
            seed=seed,
            max_roll=max_roll,
        )
        for trial in trials
    ]
    sorted_rows = sorted(rows, key=_sort_key)
    _write_csv(out_dir / "phase1_7_trials.csv", rows)
    _write_csv(out_dir / "phase1_7_top_candidates.csv", sorted_rows[:20])
    stable_safe = [
        row
        for row in rows
        if row.contact_none_ratio == 0.0 and row.jump_count == 0 and row.min_upright >= 0.985
    ]
    recommended, interpretation = _interpret(rows)
    summary = {
        "total_trials": len(rows),
        "classification_counts": dict(Counter(row.classification for row in rows)),
        "best_stable_profile_split": _best(rows, "stable_profile_split"),
        "best_stable_near_profile_split": _best(rows, "stable_near_profile_split"),
        "best_stable_055_hold": _best(rows, "stable_055_hold"),
        "best_unstable_profile_split": _best(rows, "unstable_profile_split"),
        "best_delayed_fall": _best(rows, "delayed_fall"),
        "best_toe_handoff_candidate": _best(rows, "toe_handoff_only"),
        "max_stable_support_force_ratio_peak": max((row.support_force_ratio_peak for row in stable_safe), default=0.0),
        "max_continuous_hold_058": max((row.max_continuous_hold_058 for row in rows), default=0),
        "max_cumulative_hold_058": max((row.support_force_ratio_hold_steps_058 for row in rows), default=0),
        "any_stable_profile_split": any(row.classification == "stable_profile_split" for row in rows),
        "any_toe_handoff": any(row.toe_handoff_detected for row in rows),
        "recommended_next_step": recommended,
        "engineering_interpretation": interpretation,
    }
    (out_dir / "phase1_7_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if update_progress_log:
        _append_progress_log(command, summary, out_dir)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--scene-path", type=Path, default=_default_scene())
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--magnitudes", type=_parse_float_list, default="0.015,0.020,0.0225")
    parser.add_argument("--ramp-steps-list", type=_parse_int_list, default="20,40")
    parser.add_argument("--hold-steps-list", type=_parse_int_list, default="40,80")
    parser.add_argument("--recover-steps-list", type=_parse_int_list, default="40")
    parser.add_argument("--profile-shapes", type=_parse_str_list, default="linear,smoothstep")
    parser.add_argument("--pelvis-scales", type=_parse_float_list, default="1.0")
    parser.add_argument("--swing-hip-roll-scales", type=_parse_float_list, default="1.0")
    parser.add_argument("--max-roll", type=float, default=0.08)
    parser.add_argument("--max-pitch", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-progress-log", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    del args.max_pitch
    command = f".venv\\Scripts\\python.exe -m tools.sedon_phase1_7_load_transfer_profile_shaping --steps {args.steps}"
    summary = run_sweep(
        scene_path=args.scene_path,
        out_dir=args.out_dir,
        steps=args.steps,
        seed=args.seed,
        magnitudes=args.magnitudes,
        ramp_steps_list=args.ramp_steps_list,
        hold_steps_list=args.hold_steps_list,
        recover_steps_list=args.recover_steps_list,
        profile_shapes=args.profile_shapes,
        pelvis_scales=args.pelvis_scales,
        swing_hip_roll_scales=args.swing_hip_roll_scales,
        max_roll=args.max_roll,
        command=command,
        update_progress_log=not args.no_progress_log,
    )
    print(
        "phase1_7 trials={trials} any_stable_profile_split={stable} "
        "max_cont_hold_058={cont} max_cum_hold_058={cum} any_toe={toe}".format(
            trials=summary["total_trials"],
            stable=summary["any_stable_profile_split"],
            cont=summary["max_continuous_hold_058"],
            cum=summary["max_cumulative_hold_058"],
            toe=summary["any_toe_handoff"],
        )
    )
    print(f"summary={args.out_dir / 'phase1_7_summary.json'}")
    print(f"trials={args.out_dir / 'phase1_7_trials.csv'}")
    print(f"top={args.out_dir / 'phase1_7_top_candidates.csv'}")
    print(f"progress_log={PROGRESS_LOG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
