"""Phase 1.6 load-transfer attribution sweep for Sedon.

This Class C diagnostic runs deterministic short rollouts to attribute which
control channels can safely increase support/swing force split. It does not
train PPO, does not enter full swing, and does not claim walking success from
grounded displacement.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable

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
DEFAULT_OUT_DIR = DEBUG_OUT_DIR / "phase1_6_load_transfer_attribution"
PROGRESS_LOG = REPO_ROOT / "docs" / "sedon_blue_like_dynamic_gait_progress_log.md"

R_HIP_ROLL, R_HIP_PITCH = 1, 2
L_HIP_ROLL, L_HIP_PITCH = 6, 7
FOOT_FORCE_THRESHOLD_N = 0.1

CHANNEL_SETS = (
    ("pelvis_lean", ("pelvis_lean",)),
    ("support_hip_roll", ("support_hip_roll",)),
    ("swing_hip_roll", ("swing_hip_roll",)),
    ("hip_pitch_forward", ("hip_pitch_forward",)),
    ("pelvis_lean+support_hip_roll", ("pelvis_lean", "support_hip_roll")),
    ("pelvis_lean+swing_hip_roll", ("pelvis_lean", "swing_hip_roll")),
    ("support_hip_roll+swing_hip_roll", ("support_hip_roll", "swing_hip_roll")),
    (
        "pelvis_lean+support_hip_roll+swing_hip_roll",
        ("pelvis_lean", "support_hip_roll", "swing_hip_roll"),
    ),
)
CLASSIFICATION_PRIORITY = {
    "fall_or_jump": 0,
    "unstable_split": 1,
    "stable_split": 2,
    "toe_handoff_only": 3,
    "near_split": 4,
    "stable_no_split": 5,
}


@dataclass(frozen=True)
class TrialConfig:
    """One deterministic attribution trial."""

    trial_id: str
    channel_set: str
    channels: tuple[str, ...]
    sign: float
    magnitude: float
    support_side: str


@dataclass(frozen=True)
class TrialResult:
    """Flat CSV row for one Phase 1.6 attribution trial."""

    trial_id: str
    channel_set: str
    sign_pattern: str
    pelvis_lean_bias: float
    support_hip_roll_bias: float
    swing_hip_roll_bias: float
    hip_pitch_forward_bias: float
    support_side: str
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
    swing_force_ratio_min: float
    force_gate_055_reached: bool
    force_gate_058_reached: bool
    force_gate_060_reached: bool
    toe_handoff_detected: bool
    toe_handoff_left_count: int
    toe_handoff_right_count: int
    first_toe_handoff_step: int
    classification: str
    fail_reasons: str


def _default_scene() -> Path:
    """Return preferred v5_a scene when present, otherwise standard Sedon scene."""

    return DEFAULT_V5A_SCENE if DEFAULT_V5A_SCENE.is_file() else DEFAULT_SCENE_PATH


def _parse_float_list(raw: str) -> list[float]:
    """Parse comma-separated floats for CLI sweeps."""

    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one float")
    return values


def _other_side(side: str) -> str:
    """Return the opposite side."""

    if side == "left":
        return "right"
    if side == "right":
        return "left"
    raise ValueError(f"Unsupported side: {side}")


def _contact_state(left_force: float, right_force: float) -> str:
    """Return force-thresholded foot contact state."""

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
    """Return center/toe/heel contact flags for both feet."""

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


def _bias_values(trial: TrialConfig) -> tuple[float, float, float, float]:
    """Return signed channel values for one trial."""

    value = trial.sign * trial.magnitude
    pelvis = value if "pelvis_lean" in trial.channels else 0.0
    support_roll = value if "support_hip_roll" in trial.channels else 0.0
    swing_roll = value if "swing_hip_roll" in trial.channels else 0.0
    pitch = value if "hip_pitch_forward" in trial.channels else 0.0
    return pelvis, support_roll, swing_roll, pitch


def _target(
    env: SedonStandingEnv,
    trial: TrialConfig,
    *,
    max_roll: float,
    max_pitch: float,
) -> tuple[np.ndarray, float, float, float, float]:
    """Return clamped joint target and actual applied channel values."""

    pelvis, support_roll, swing_roll, pitch = _bias_values(trial)
    pelvis = float(np.clip(pelvis, -max_roll, max_roll))
    support_roll = float(np.clip(support_roll, -max_roll, max_roll))
    swing_roll = float(np.clip(swing_roll, -max_roll, max_roll))
    pitch = float(np.clip(pitch, -max_pitch, max_pitch))

    target = env._nominal_joint_qpos.copy()
    target[R_HIP_PITCH] += pitch
    target[L_HIP_PITCH] += pitch

    if trial.support_side == "left":
        target[L_HIP_ROLL] += support_roll + pelvis
        target[R_HIP_ROLL] += swing_roll + pelvis
    else:
        target[R_HIP_ROLL] -= support_roll + pelvis
        target[L_HIP_ROLL] -= swing_roll + pelvis
    return env._apply_safe_joint_target_clamps(target), pelvis, support_roll, swing_roll, pitch


def _support_ratios(forces: dict[str, float | bool | int], support_side: str) -> tuple[float, float, float, float]:
    """Return support/swing force and normalized ratios."""

    swing_side = _other_side(support_side)
    support_force = float(forces[f"{support_side}_force"])
    swing_force = float(forces[f"{swing_side}_force"])
    total = max(support_force + swing_force, 1e-9)
    return support_force, swing_force, support_force / total, swing_force / total


def _count_bursts(rows: list[dict[str, Any]], key: str) -> int:
    """Count contiguous truthy bursts."""

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


def _toe_handoff_counts(rows: list[dict[str, Any]]) -> tuple[int, int, int]:
    """Return left/right handoff counts and first handoff step."""

    counts = {"left": 0, "right": 0}
    saw_center = {"left": False, "right": False}
    toe_active = {"left": False, "right": False}
    first_step = -1
    for row in rows:
        for side in ("left", "right"):
            center = bool(row[f"center_contact_{side}"])
            toe = bool(row[f"toe_contact_{side}"])
            if center:
                saw_center[side] = True
            if saw_center[side] and toe and not toe_active[side]:
                counts[side] += 1
                if first_step < 0:
                    first_step = int(row["step"])
                toe_active[side] = True
                saw_center[side] = False
            if not toe:
                toe_active[side] = False
    return counts["left"], counts["right"], first_step


def _hold_steps(rows: list[dict[str, Any]], threshold: float) -> int:
    """Return cumulative safe hold steps above support-ratio threshold."""

    return sum(
        float(row["support_force_ratio"]) >= threshold
        and float(row["upright"]) >= 0.985
        and not bool(row["contact_none"])
        and not bool(row["jump"])
        for row in rows
    )


def _classify(
    *,
    contact_none_ratio: float,
    jump_count: int,
    min_upright: float,
    support_peak: float,
    hold_058: int,
    toe_handoff: bool,
) -> str:
    """Classify one trial by the requested priority order."""

    fall = contact_none_ratio > 0.0 or jump_count > 0 or min_upright < 0.985
    if fall:
        return "unstable_split" if support_peak >= 0.58 else "fall_or_jump"
    if support_peak >= 0.58 and hold_058 >= 10:
        return "stable_split"
    if toe_handoff and support_peak < 0.58:
        return "toe_handoff_only"
    if support_peak >= 0.55:
        return "near_split"
    return "stable_no_split"


def _fail_reasons(
    *,
    contact_none_ratio: float,
    jump_count: int,
    min_upright: float,
    support_peak: float,
    hold_058: int,
) -> str:
    """Return compact failure labels for one trial."""

    reasons: list[str] = []
    if contact_none_ratio > 0.0:
        reasons.append("contact_none")
    if jump_count > 0:
        reasons.append("jump")
    if min_upright < 0.985:
        reasons.append("upright")
    if support_peak < 0.58:
        reasons.append("support_force_ratio_peak")
    if hold_058 < 10:
        reasons.append("support_force_ratio_hold_steps_058")
    return ",".join(reasons)


def run_trial(
    *,
    scene_path: Path,
    trial: TrialConfig,
    steps: int,
    seed: int,
    max_roll: float,
    max_pitch: float,
) -> TrialResult:
    """Run one deterministic load-transfer attribution trial."""

    env = SedonStandingEnv(
        scene_path=scene_path,
        reset_noise_scale=0.0,
        reward_config=load_sedon_config_from_env(),
    )
    rows: list[dict[str, Any]] = []
    applied = (0.0, 0.0, 0.0, 0.0)
    try:
        env.reset(seed=seed)
        initial_x = float(env.data.qpos[0])
        robot_weight = float(np.sum(env.model.body_mass) * 9.81)
        for step in range(1, steps + 1):
            target, pelvis, support_roll, swing_roll, pitch = _target(
                env,
                trial,
                max_roll=max_roll,
                max_pitch=max_pitch,
            )
            applied = (pelvis, support_roll, swing_roll, pitch)
            env._do_pd_simulation(target)
            env._gait_step += 1
            forces = _foot_contact_forces(env)
            contacts = _foot_region_contacts(env)
            left_force = float(forces["left_force"])
            right_force = float(forces["right_force"])
            support_force, swing_force, support_ratio, swing_ratio = _support_ratios(
                forces,
                trial.support_side,
            )
            contact_state = _contact_state(left_force, right_force)
            total_force = left_force + right_force
            contact_none = contact_state == "none"
            jump = bool(contact_none or (float(env.data.qvel[2]) > 0.05 and total_force < 0.10 * robot_weight))
            rows.append(
                {
                    "step": step,
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
    support_peak = max(support_ratios, default=0.0)
    hold_055 = _hold_steps(rows, 0.55)
    hold_058 = _hold_steps(rows, 0.58)
    hold_060 = _hold_steps(rows, 0.60)
    classification = _classify(
        contact_none_ratio=contact_none_ratio,
        jump_count=jump_count,
        min_upright=min_upright,
        support_peak=support_peak,
        hold_058=hold_058,
        toe_handoff=(left_handoff + right_handoff) > 0,
    )
    return TrialResult(
        trial_id=trial.trial_id,
        channel_set=trial.channel_set,
        sign_pattern=f"{trial.sign:+.1f}x",
        pelvis_lean_bias=applied[0],
        support_hip_roll_bias=applied[1],
        swing_hip_roll_bias=applied[2],
        hip_pitch_forward_bias=applied[3],
        support_side=trial.support_side,
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
        swing_force_ratio_min=min(swing_ratios, default=0.0),
        force_gate_055_reached=hold_055 >= 10,
        force_gate_058_reached=hold_058 >= 10,
        force_gate_060_reached=hold_060 >= 10,
        toe_handoff_detected=(left_handoff + right_handoff) > 0,
        toe_handoff_left_count=left_handoff,
        toe_handoff_right_count=right_handoff,
        first_toe_handoff_step=first_handoff,
        classification=classification,
        fail_reasons=_fail_reasons(
            contact_none_ratio=contact_none_ratio,
            jump_count=jump_count,
            min_upright=min_upright,
            support_peak=support_peak,
            hold_058=hold_058,
        ),
    )


def _trial_sort_key(row: TrialResult) -> tuple[int, int, float, float, int, float, int]:
    """Sort key for top candidates."""

    return (
        0 if row.classification == "stable_split" else 1,
        -row.support_force_ratio_hold_steps_058,
        -row.support_force_ratio_peak,
        row.contact_none_ratio,
        row.jump_count,
        -row.min_upright,
        0 if row.toe_handoff_detected else 1,
    )


def _best(rows: Iterable[TrialResult], classification: str | None = None) -> dict[str, Any] | None:
    """Return best row as a JSON object, optionally filtered by classification."""

    filtered = [row for row in rows if classification is None or row.classification == classification]
    if not filtered:
        return None
    return asdict(sorted(filtered, key=_trial_sort_key)[0])


def _best_toe(rows: list[TrialResult]) -> dict[str, Any] | None:
    """Return best toe-handoff candidate."""

    candidates = [row for row in rows if row.toe_handoff_detected]
    if not candidates:
        return None
    return asdict(sorted(candidates, key=_trial_sort_key)[0])


def _interpret(rows: list[TrialResult]) -> tuple[str, str]:
    """Return recommended next step and engineering interpretation."""

    stable_split = [row for row in rows if row.classification == "stable_split"]
    near_split = [row for row in rows if row.classification == "near_split"]
    stable_rows = [
        row
        for row in rows
        if row.contact_none_ratio == 0.0 and row.jump_count == 0 and row.min_upright >= 0.985
    ]
    max_stable = max((row.support_force_ratio_peak for row in stable_rows), default=0.0)
    max_any = max((row.support_force_ratio_peak for row in rows), default=0.0)
    best_ranked = sorted(rows, key=_trial_sort_key)[0] if rows else None
    best_safe = sorted(stable_split or near_split or stable_rows, key=_trial_sort_key)[0] if (stable_split or near_split or stable_rows) else None
    best_raw = max(rows, key=lambda row: row.support_force_ratio_peak) if rows else None
    best_channel = best_safe.channel_set if best_safe else "none"
    best_sign = best_safe.sign_pattern if best_safe else "none"
    unstable_split = any(row.classification == "unstable_split" for row in rows)

    if stable_split:
        recommended = "Proceed only to Phase 2 force-gated capture-step micro swing; keep strict no-contact and impact gates."
        split_state = "Force split can be opened and held safely above 0.58 in at least one deterministic channel setting."
    elif max_stable >= 0.55:
        recommended = "Do not enter Phase 2. Tune controller load-transfer shaping around the best near-split channel before changing morphology."
        split_state = "Force split is not fully open, but it is close to threshold under stable conditions."
    elif unstable_split:
        recommended = "Do not enter Phase 2. Reduce abrupt gain and inspect stability limits before any swing command."
        split_state = "Force split appears possible only after crossing the stability boundary."
    else:
        recommended = "Do not enter Phase 2. Inspect morphology/contact patch, stance width, COM height, and hip-roll authority."
        split_state = "Force split is effectively blocked under this deterministic bias sweep."

    stability = (
        "Gain increase reaches unstable split before stable split."
        if unstable_split and not stable_split
        else "No unstable split dominated the best stable candidate set."
    )
    raw_text = (
        f"Raw strongest channel_set={best_raw.channel_set}, sign={best_raw.sign_pattern}, "
        f"classification={best_raw.classification}"
        if best_raw
        else "Raw strongest channel_set=none"
    )
    ranked_text = (
        f"Top sorted candidate={best_ranked.channel_set}, sign={best_ranked.sign_pattern}, "
        f"classification={best_ranked.classification}"
        if best_ranked
        else "Top sorted candidate=none"
    )
    interpretation = (
        f"{split_state} Safest effective channel_set: {best_channel}. "
        f"Safest effective sign_pattern: {best_sign}. {ranked_text}. {raw_text}. {stability} "
        f"Max support_force_ratio_peak overall={max_any:.4f}, stable={max_stable:.4f}. "
        f"Phase 2 recommendation: {'yes' if stable_split else 'no'}. "
        f"Next focus: {recommended}"
    )
    return recommended, interpretation


def build_trials(magnitudes: list[float]) -> list[TrialConfig]:
    """Build deterministic left/right, sign, magnitude trials."""

    trials: list[TrialConfig] = []
    index = 1
    for channel_set, channels in CHANNEL_SETS:
        for sign in (-1.0, 1.0):
            for magnitude in magnitudes:
                for support_side in ("left", "right"):
                    trials.append(
                        TrialConfig(
                            trial_id=f"p16_{index:04d}",
                            channel_set=channel_set,
                            channels=tuple(channels),
                            sign=sign,
                            magnitude=float(magnitude),
                            support_side=support_side,
                        )
                    )
                    index += 1
    return trials


def _write_csv(path: Path, rows: list[TrialResult]) -> None:
    """Write trial result rows to CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _append_progress_log(command: str, summary: dict[str, Any], out_dir: Path) -> None:
    """Append Phase 1.6 result to the human-readable progress log."""

    PROGRESS_LOG.parent.mkdir(parents=True, exist_ok=True)
    if not PROGRESS_LOG.exists():
        PROGRESS_LOG.write_text("# Sedon Blue-Like Dynamic Gait Progress Log\n\n", encoding="utf-8")
    best = summary.get("best_stable_split") or summary.get("best_near_split") or summary.get("best_unstable_split") or {}
    result = "PASS" if summary["any_stable_split"] else ("INCONCLUSIVE" if summary["max_stable_support_force_ratio_peak"] >= 0.55 else "FAIL")
    lines = [
        f"## {date.today().isoformat()} - Phase 1.6 Load Transfer Attribution Sweep",
        "",
        "### Goal",
        "",
        "Identify which control channels can increase support/swing force split safely before attempting capture-step or visible stepping.",
        "",
        "### Files",
        "",
        "- `tools/sedon_phase1_6_load_transfer_attribution.py`",
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
        f"- `{out_dir / 'phase1_6_trials.csv'}`",
        f"- `{out_dir / 'phase1_6_summary.json'}`",
        f"- `{out_dir / 'phase1_6_top_candidates.csv'}`",
        "",
        "### Key Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| total_trials | {summary['total_trials']} |",
        f"| any_stable_split | {str(summary['any_stable_split']).lower()} |",
        f"| max_stable_support_force_ratio_peak | {summary['max_stable_support_force_ratio_peak']:.6f} |",
        f"| max_support_force_ratio_hold_steps_058 | {summary['max_support_force_ratio_hold_steps_058']} |",
        f"| any_toe_handoff | {str(summary['any_toe_handoff']).lower()} |",
        f"| best_channel_set | {best.get('channel_set', 'none')} |",
        f"| best_sign_pattern | {best.get('sign_pattern', 'none')} |",
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
    max_roll: float,
    max_pitch: float,
    command: str,
    update_progress_log: bool = True,
) -> dict[str, Any]:
    """Run all deterministic Phase 1.6 attribution trials."""

    if steps <= 0:
        raise ValueError("steps must be positive.")
    scene = require_scene(scene_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    trials = build_trials(magnitudes)
    rows = [
        run_trial(
            scene_path=scene,
            trial=trial,
            steps=steps,
            seed=seed,
            max_roll=max_roll,
            max_pitch=max_pitch,
        )
        for trial in trials
    ]
    sorted_rows = sorted(rows, key=_trial_sort_key)
    _write_csv(out_dir / "phase1_6_trials.csv", rows)
    _write_csv(out_dir / "phase1_6_top_candidates.csv", sorted_rows[:20])

    classification_counts = dict(Counter(row.classification for row in rows))
    stable_rows = [
        row
        for row in rows
        if row.contact_none_ratio == 0.0 and row.jump_count == 0 and row.min_upright >= 0.985
    ]
    recommended, interpretation = _interpret(rows)
    summary = {
        "total_trials": len(rows),
        "classification_counts": classification_counts,
        "best_stable_split": _best(rows, "stable_split"),
        "best_near_split": _best(rows, "near_split"),
        "best_unstable_split": _best(rows, "unstable_split"),
        "best_toe_handoff_candidate": _best_toe(rows),
        "max_support_force_ratio_peak": max((row.support_force_ratio_peak for row in rows), default=0.0),
        "max_stable_support_force_ratio_peak": max((row.support_force_ratio_peak for row in stable_rows), default=0.0),
        "max_support_force_ratio_hold_steps_058": max((row.support_force_ratio_hold_steps_058 for row in rows), default=0),
        "any_stable_split": any(row.classification == "stable_split" for row in rows),
        "any_toe_handoff": any(row.toe_handoff_detected for row in rows),
        "recommended_next_step": recommended,
        "engineering_interpretation": interpretation,
    }
    (out_dir / "phase1_6_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if update_progress_log:
        _append_progress_log(command, summary, out_dir)
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--scene-path", type=Path, default=_default_scene())
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--magnitudes", type=_parse_float_list, default="0.0025,0.005,0.0075,0.010,0.0125,0.015,0.020")
    parser.add_argument("--support-thresholds", type=_parse_float_list, default="0.55,0.58,0.60")
    parser.add_argument("--max-roll", type=float, default=0.08)
    parser.add_argument("--max-pitch", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-progress-log", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run Phase 1.6 sweep from CLI."""

    args = build_parser().parse_args(argv)
    # The thresholds are accepted for CLI reproducibility; the requested report
    # currently standardizes on 0.55, 0.58, and 0.60 columns.
    del args.support_thresholds
    command = (
        ".venv\\Scripts\\python.exe -m tools.sedon_phase1_6_load_transfer_attribution "
        f"--steps {args.steps}"
    )
    summary = run_sweep(
        scene_path=args.scene_path,
        out_dir=args.out_dir,
        steps=args.steps,
        seed=args.seed,
        magnitudes=args.magnitudes,
        max_roll=args.max_roll,
        max_pitch=args.max_pitch,
        command=command,
        update_progress_log=not args.no_progress_log,
    )
    print(
        "phase1_6 trials={trials} any_stable_split={stable} "
        "max_stable_peak={stable_peak:.4f} max_hold_058={hold} any_toe={toe}".format(
            trials=summary["total_trials"],
            stable=summary["any_stable_split"],
            stable_peak=summary["max_stable_support_force_ratio_peak"],
            hold=summary["max_support_force_ratio_hold_steps_058"],
            toe=summary["any_toe_handoff"],
        )
    )
    print(f"summary={args.out_dir / 'phase1_6_summary.json'}")
    print(f"trials={args.out_dir / 'phase1_6_trials.csv'}")
    print(f"top={args.out_dir / 'phase1_6_top_candidates.csv'}")
    print(f"progress_log={PROGRESS_LOG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
