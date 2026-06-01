"""Phase Pre-W0 Sedon actuator semantics audit.

This Class C diagnostic inspects MuJoCo actuator metadata before any
W0-DuckRef scripted walking smoke test. It answers whether Sedon controls look
like position targets, torque motors, velocity commands, or generic actuators,
and whether the DuckRef `action_scale=0.25` transfer is safe enough to try.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import mujoco

from tools.sedon_debug_common import DEBUG_OUT_DIR, require_scene


DEFAULT_SCENE_PATH = DEBUG_OUT_DIR / "blue_like_sole_experiments_v5" / "training_scene_v5_a.xml"
DEFAULT_OUTPUT_DIR = DEBUG_OUT_DIR / "phase_pre_w0_actuator_semantics_audit"


@dataclass(frozen=True)
class AuditConfig:
    """Runtime configuration for the actuator semantics audit.

    Parameters:
        scene_path: MuJoCo XML scene to inspect.
        output_dir: Directory where CSV, JSON, and Markdown outputs are written.
    """

    scene_path: Path
    output_dir: Path


def mj_name(model: mujoco.MjModel, obj_type: mujoco.mjtObj, index: int, fallback: str) -> str:
    """Return a stable MuJoCo object name.

    Parameters:
        model: Loaded MuJoCo model.
        obj_type: MuJoCo object type.
        index: Object index.
        fallback: Name prefix used when the object is unnamed.

    Returns:
        A MuJoCo name or deterministic fallback.
    """

    name = mujoco.mj_id2name(model, obj_type, int(index))
    return name or f"{fallback}_{index}"


def csv_vec(values: Iterable[Any]) -> str:
    """Format vector-like metadata for compact CSV cells.

    Parameters:
        values: Numeric values from MuJoCo arrays.

    Returns:
        Space-separated scalar string.
    """

    return " ".join(f"{float(value):.9g}" for value in values)


def enum_name(enum_cls: Any, value: int, prefix: str) -> str:
    """Return a MuJoCo enum name without its common prefix.

    Parameters:
        enum_cls: MuJoCo enum class.
        value: Raw enum integer.
        prefix: Prefix to remove.

    Returns:
        Lowercase enum name.
    """

    try:
        return enum_cls(int(value)).name.replace(prefix, "").lower()
    except ValueError:
        return f"unknown_{value}"


def actuator_name(model: mujoco.MjModel, actuator_id: int) -> str:
    """Return actuator name for an id."""

    return mj_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id, "actuator")


def joint_name(model: mujoco.MjModel, joint_id: int) -> str:
    """Return joint name for an id."""

    return mj_name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id, "joint")


def joint_type_name(model: mujoco.MjModel, joint_id: int) -> str:
    """Return a compact joint type name."""

    return enum_name(mujoco.mjtJoint, int(model.jnt_type[joint_id]), "mjJNT_")


def infer_semantics(
    *,
    actuator_name_value: str,
    gain_type: str,
    bias_type: str,
    dyn_type: str,
    trn_type: str,
    gainprm: list[float],
    biasprm: list[float],
) -> tuple[str, str, str]:
    """Infer actuator control semantics from MuJoCo metadata.

    Parameters:
        actuator_name_value: Actuator name from the model.
        gain_type: MuJoCo gain type.
        bias_type: MuJoCo bias type.
        dyn_type: MuJoCo dynamics type.
        trn_type: MuJoCo transmission type.
        gainprm: Gain parameter vector.
        biasprm: Bias parameter vector.

    Returns:
        Tuple of `(semantics, confidence, notes)`.
    """

    lowered = actuator_name_value.lower()
    kp = abs(gainprm[0]) if gainprm else 0.0
    bias_qpos = biasprm[1] if len(biasprm) > 1 else 0.0
    bias_qvel = biasprm[2] if len(biasprm) > 2 else 0.0

    if trn_type != "joint":
        return "general_like", "medium", f"non-joint transmission `{trn_type}`"

    if "velocity" in lowered or lowered.endswith("_vel"):
        return "velocity_like", "medium", "name suggests velocity command"

    if gain_type == "fixed" and bias_type == "affine" and kp > 0.0 and bias_qpos < 0.0:
        if bias_qvel < 0.0:
            return "position_like", "high", "fixed gain + affine qpos/qvel feedback matches position servo"
        return "position_like", "medium", "fixed gain + affine qpos feedback matches position servo"

    if gain_type == "fixed" and bias_type == "none" and dyn_type == "none":
        return "motor_like", "high", "fixed gain without bias feedback matches torque/motor command"

    if gain_type == "fixed" and bias_type == "affine" and abs(bias_qvel) > 0.0 and abs(bias_qpos) <= 1e-12:
        return "velocity_like", "medium", "affine velocity feedback without qpos feedback"

    if gain_type == "user" or bias_type == "user" or dyn_type == "user":
        return "general_like", "medium", "user-defined actuator metadata"

    return "unknown_like", "low", f"gain={gain_type}, bias={bias_type}, dyn={dyn_type}"


def actuator_rows(model: mujoco.MjModel) -> list[dict[str, Any]]:
    """Build actuator inventory rows.

    Parameters:
        model: Loaded MuJoCo model.

    Returns:
        Rows for `sedon_actuator_semantics_inventory.csv`.
    """

    rows: list[dict[str, Any]] = []
    for actuator_id in range(model.nu):
        name = actuator_name(model, actuator_id)
        trnid = [int(value) for value in model.actuator_trnid[actuator_id]]
        joint_id = trnid[0]
        has_joint = 0 <= joint_id < model.njnt
        gainprm = [float(value) for value in model.actuator_gainprm[actuator_id]]
        biasprm = [float(value) for value in model.actuator_biasprm[actuator_id]]
        dynprm = [float(value) for value in model.actuator_dynprm[actuator_id]]
        gain_type = enum_name(mujoco.mjtGain, int(model.actuator_gaintype[actuator_id]), "mjGAIN_")
        bias_type = enum_name(mujoco.mjtBias, int(model.actuator_biastype[actuator_id]), "mjBIAS_")
        dyn_type = enum_name(mujoco.mjtDyn, int(model.actuator_dyntype[actuator_id]), "mjDYN_")
        trn_type = enum_name(mujoco.mjtTrn, int(model.actuator_trntype[actuator_id]), "mjTRN_")
        semantics, confidence, notes = infer_semantics(
            actuator_name_value=name,
            gain_type=gain_type,
            bias_type=bias_type,
            dyn_type=dyn_type,
            trn_type=trn_type,
            gainprm=gainprm,
            biasprm=biasprm,
        )
        rows.append(
            {
                "actuator_id": actuator_id,
                "actuator_name": name,
                "actuator_type_or_inferred_type": f"{trn_type}:{gain_type}:{bias_type}:{dyn_type}",
                "trnid": " ".join(str(value) for value in trnid),
                "joint_id": joint_id if has_joint else "",
                "joint_name": joint_name(model, joint_id) if has_joint else "",
                "joint_type": joint_type_name(model, joint_id) if has_joint else "",
                "ctrlrange": csv_vec(model.actuator_ctrlrange[actuator_id]),
                "forcerange": csv_vec(model.actuator_forcerange[actuator_id]),
                "gear": csv_vec(model.actuator_gear[actuator_id]),
                "gainprm": csv_vec(gainprm),
                "biasprm": csv_vec(biasprm),
                "dynprm": csv_vec(dynprm),
                "actrange": csv_vec(model.actuator_actrange[actuator_id]),
                "joint_range": csv_vec(model.jnt_range[joint_id]) if has_joint else "",
                "inferred_control_semantics": semantics,
                "confidence": confidence,
                "notes": notes,
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize actuator semantics.

    Parameters:
        rows: Actuator inventory rows.

    Returns:
        JSON-serializable summary payload.
    """

    total = len(rows)
    counts = {
        "position_like_count": sum(row["inferred_control_semantics"] == "position_like" for row in rows),
        "motor_like_count": sum(row["inferred_control_semantics"] == "motor_like" for row in rows),
        "velocity_like_count": sum(row["inferred_control_semantics"] == "velocity_like" for row in rows),
        "general_like_count": sum(row["inferred_control_semantics"] == "general_like" for row in rows),
        "unknown_like_count": sum(row["inferred_control_semantics"] == "unknown_like" for row in rows),
    }
    high_confidence = sum(row["confidence"] == "high" for row in rows)
    dominant = max(counts, key=lambda key: counts[key]) if rows else "unknown_like_count"
    blocking: list[str] = []
    if total == 0:
        blocking.append("no_actuators_found")
    if counts["unknown_like_count"] > 0:
        blocking.append("unknown_actuator_semantics_present")
    if counts["position_like_count"] and (counts["motor_like_count"] or counts["velocity_like_count"]):
        blocking.append("mixed_position_and_non_position_semantics")
    if high_confidence < total:
        blocking.append("not_all_actuators_high_confidence")

    if dominant == "position_like_count" and counts["unknown_like_count"] == 0:
        ctrl_units = "ctrl values are likely absolute joint position targets in joint units"
        safe_scale = 0.10
        transfer_safe: bool | str = False
        blocking.append("duck_action_scale_025_too_large_for_direct_absolute_position_targets")
    elif dominant == "motor_like_count" and counts["unknown_like_count"] == 0:
        ctrl_units = "ctrl values are likely normalized or gear-scaled motor/torque commands"
        safe_scale = 0.25
        transfer_safe = "inconclusive"
    else:
        ctrl_units = "ctrl units are mixed or unclear"
        safe_scale = 0.10
        transfer_safe = "inconclusive"

    return {
        "total_actuators": total,
        **counts,
        "ctrl_units_interpretation": ctrl_units,
        "safe_initial_action_scale_recommendation": safe_scale,
        "duck_action_scale_025_transfer_safe": transfer_safe,
        "blocking_issues_before_w0": sorted(set(blocking)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """Write CSV rows with a stable header."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, config: AuditConfig, summary: dict[str, Any]) -> None:
    """Write Markdown actuator semantics report."""

    lines = [
        "# Phase Pre-W0 Actuator Semantics Audit",
        "",
        f"- scene_path: `{config.scene_path}`",
        f"- total_actuators: `{summary['total_actuators']}`",
        f"- ctrl_units_interpretation: `{summary['ctrl_units_interpretation']}`",
        f"- duck_action_scale_025_transfer_safe: `{summary['duck_action_scale_025_transfer_safe']}`",
        f"- safe_initial_action_scale_recommendation: `{summary['safe_initial_action_scale_recommendation']}`",
        "",
        "## Counts",
        "",
        f"- position_like_count: `{summary['position_like_count']}`",
        f"- motor_like_count: `{summary['motor_like_count']}`",
        f"- velocity_like_count: `{summary['velocity_like_count']}`",
        f"- general_like_count: `{summary['general_like_count']}`",
        f"- unknown_like_count: `{summary['unknown_like_count']}`",
        "",
        "## Blocking Issues Before W0",
        "",
        *(f"- `{issue}`" for issue in summary["blocking_issues_before_w0"]),
        "",
        "## Engineering Interpretation",
        "",
        "This audit only classifies actuator metadata. It does not prove that a W0 controller convention is safe by itself.",
        "If the model is position-like, W0 should emit conservative delta targets around the current neutral pose rather than raw Duck actions.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-path", type=Path, default=DEFAULT_SCENE_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    """Run the actuator semantics audit."""

    args = parse_args()
    config = AuditConfig(scene_path=require_scene(args.scene_path), output_dir=args.output_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(config.scene_path))
    rows = actuator_rows(model)
    summary = summarize(rows)

    write_csv(
        config.output_dir / "sedon_actuator_semantics_inventory.csv",
        rows,
        [
            "actuator_id",
            "actuator_name",
            "actuator_type_or_inferred_type",
            "trnid",
            "joint_id",
            "joint_name",
            "joint_type",
            "ctrlrange",
            "forcerange",
            "gear",
            "gainprm",
            "biasprm",
            "dynprm",
            "actrange",
            "joint_range",
            "inferred_control_semantics",
            "confidence",
            "notes",
        ],
    )
    (config.output_dir / "sedon_actuator_semantics_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(config.output_dir / "sedon_actuator_semantics_report.md", config, summary)

    print(f"inventory={config.output_dir / 'sedon_actuator_semantics_inventory.csv'}")
    print(f"summary={config.output_dir / 'sedon_actuator_semantics_summary.json'}")
    print(f"report={config.output_dir / 'sedon_actuator_semantics_report.md'}")
    print(f"duck_action_scale_025_transfer_safe={summary['duck_action_scale_025_transfer_safe']}")


if __name__ == "__main__":
    main()
