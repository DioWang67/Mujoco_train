"""Build the Seedon v5_22 actuator envelope.

This Class C diagnostic aggregates v5_22 MJCF actuator data, source URDF joint
limits, Python controller semantics, and team-provided torque data. It does not
modify source XML/URDF, training code, evaluation code, or environment runtime
behavior.
"""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

try:
    import mujoco
except ModuleNotFoundError as exc:  # pragma: no cover - only when MuJoCo is unavailable.
    mujoco = None
    _MUJOCO_IMPORT_ERROR: ModuleNotFoundError | None = exc
else:
    _MUJOCO_IMPORT_ERROR = None


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "seedon" / "seedon_v5_22_actuator_envelope.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "seedon_debug" / "v5_22_actuator_envelope" / "envelope.json"
DEFAULT_REPORT = REPO_ROOT / "docs" / "seedon_v5_22_actuator_envelope_report.md"

MISSING_FIELDS = (
    "forcerange",
    "verified_gear_ratio",
    "confirmed_kp",
    "confirmed_kd",
    "max_velocity",
    "control_mode",
    "current_limit",
    "encoder_resolution",
    "backlash",
)


def rel_path(path: Path) -> str:
    """Return repository-relative path text."""

    try:
        return str(path.resolve().relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def load_config(path: Path) -> dict[str, Any]:
    """Load JSON-compatible YAML config."""

    if not path.is_file():
        raise FileNotFoundError(f"Actuator envelope config not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Config must decode to a JSON object: {path}")
    return data


def resolve_repo_path(raw_path: str) -> Path:
    """Resolve a repo-relative path."""

    path = Path(raw_path)
    return path if path.is_absolute() else REPO_ROOT / path


def load_mujoco_model(path: Path) -> Any:
    """Load MuJoCo model."""

    if mujoco is None:
        raise RuntimeError(f"MuJoCo is not importable: {_MUJOCO_IMPORT_ERROR}")
    return mujoco.MjModel.from_xml_path(str(path))


def model_name(model: Any, obj_type: Any, index: int) -> str:
    """Return MuJoCo object name."""

    return mujoco.mj_id2name(model, obj_type, int(index)) or f"<unnamed:{index}>"


def actuator_rows(model: Any) -> list[dict[str, Any]]:
    """Return v5_22 actuator mapping rows."""

    rows: list[dict[str, Any]] = []
    for actuator_id in range(model.nu):
        actuator_name = model_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        joint_name = model_name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) if joint_id >= 0 else "<none>"
        ctrlrange = [float(value) for value in model.actuator_ctrlrange[actuator_id]]
        forcerange = [float(value) for value in model.actuator_forcerange[actuator_id]]
        gear = [float(value) for value in model.actuator_gear[actuator_id]]
        rows.append(
            {
                "actuator_index": actuator_id,
                "actuator_name": actuator_name,
                "joint_name": joint_name,
                "joint_group": classify_joint_group(joint_name),
                "ctrlrange": ctrlrange,
                "ctrlrange_status": "FOUND" if any(abs(value) > 0 for value in ctrlrange) else "manual_required",
                "forcerange": None if all(abs(value) <= 1e-12 for value in forcerange) else forcerange,
                "forcerange_status": "manual_required" if all(abs(value) <= 1e-12 for value in forcerange) else "FOUND",
                "gear": gear,
                "gear_status": "manual_required" if is_default_mujoco_gear(gear) else "FOUND",
            }
        )
    return rows


def is_default_mujoco_gear(gear: list[float]) -> bool:
    """Return whether gear looks like default MuJoCo motor gear."""

    return len(gear) >= 1 and abs(gear[0] - 1.0) <= 1e-12 and all(abs(value) <= 1e-12 for value in gear[1:])


def classify_joint_group(joint_name: str) -> str:
    """Map joint name to team-provided torque group."""

    lowered = joint_name.lower()
    if "hip_pitch" in lowered:
        return "hip_pitch"
    if "ankle_pitch" in lowered:
        return "ankle_pitch"
    return "other_leg_joints"


def parse_urdf_joints(path: Path) -> dict[str, dict[str, Any]]:
    """Parse source URDF joint fields."""

    if not path.is_file():
        return {}
    root = ET.parse(path).getroot()
    rows: dict[str, dict[str, Any]] = {}
    for joint in root.findall("joint"):
        name = joint.attrib.get("name", "")
        axis = joint.find("axis")
        limit = joint.find("limit")
        if not name:
            continue
        rows[name] = {
            "source_joint_name": name,
            "axis": axis.attrib.get("xyz") if axis is not None else None,
            "lower": limit.attrib.get("lower") if limit is not None else None,
            "upper": limit.attrib.get("upper") if limit is not None else None,
            "effort": limit.attrib.get("effort") if limit is not None else None,
            "velocity": limit.attrib.get("velocity") if limit is not None else None,
            "source": rel_path(path),
        }
    return rows


def urdf_record_for_mjcf_joint(joint_name: str, urdf_joints: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """Resolve source URDF joint record for generated v5_22 MJCF joint names."""

    if joint_name in urdf_joints:
        return urdf_joints[joint_name]
    reverse_adapter = {
        "R_joint_knee_pitch": "R_joint_knee",
        "R_joint_ankle_pitch": "R_joint_knee_pitch",
    }
    source_name = reverse_adapter.get(joint_name)
    return urdf_joints.get(source_name) if source_name else None


def read_env_lines(path: Path) -> list[str]:
    """Read env.py lines."""

    if not path.is_file():
        return []
    return path.read_text(encoding="utf-8").splitlines()


def find_line(lines: list[str], pattern: str) -> dict[str, Any] | None:
    """Find first regex line."""

    regex = re.compile(pattern)
    for index, line in enumerate(lines, start=1):
        if regex.search(line):
            return {"line": index, "text": line.strip()}
    return None


def controller_semantics(path: Path) -> dict[str, Any]:
    """Extract Python controller semantics from seedon_baseline/env.py."""

    lines = read_env_lines(path)
    return {
        "source_file": rel_path(path),
        "_do_pd_simulation": find_line(lines, r"def _do_pd_simulation\b"),
        "_ctrl_range": find_line(lines, r"self\._ctrl_range\s*=\s*self\.model\.actuator_ctrlrange\.copy\(\)"),
        "np_clip_saturation": find_line(lines, r"return np\.clip\(scaled_ctrl, self\._ctrl_range"),
        "action_joint_delta_scale": find_line(lines, r"action_joint_delta_scale:\s*float\s*="),
        "pd_stiffness": find_line(lines, r"pd_stiffness:\s*float\s*="),
        "pd_damping": find_line(lines, r"pd_damping:\s*float\s*="),
        "safe_target_clamp": find_line(lines, r"def _apply_safe_joint_target_clamps\b"),
        "status": "FOUND" if lines else "NOT_FOUND",
    }


def torque_for_joint(joint_group: str, config: dict[str, Any]) -> dict[str, Any]:
    """Return team-provided torque envelope for one joint group."""

    torque_data = config["torque_data"]
    group = torque_data["groups"][joint_group]
    return {
        "rated_nm": float(group["rated_nm"]),
        "peak_nm": float(group["peak_nm"]),
        "torque_side": torque_data["torque_side"],
        "source": torque_data["source"],
        "confidence": torque_data["confidence"],
    }


def build_profiles(config: dict[str, Any], actuator_envelopes: list[dict[str, Any]]) -> dict[str, Any]:
    """Build diagnostic profiles."""

    rated = []
    peak = []
    for item in actuator_envelopes:
        rated.append(
            {
                "joint_name": item["joint_name"],
                "limit_nm": item["torque_envelope"]["rated_nm"],
                "torque_side": item["torque_envelope"]["torque_side"],
            }
        )
        peak.append(
            {
                "joint_name": item["joint_name"],
                "limit_nm": item["torque_envelope"]["peak_nm"],
                "torque_side": item["torque_envelope"]["torque_side"],
            }
        )
    profiles = config["profiles"]
    return {
        "rated_safe": {
            **profiles["rated_safe"],
            "joint_limits": rated,
        },
        "peak_upper_bound": {
            **profiles["peak_upper_bound"],
            "joint_limits": peak,
        },
        "ankle_risk_sweep": profiles["ankle_risk_sweep"],
    }


def build_payload(config: dict[str, Any]) -> dict[str, Any]:
    """Build full actuator envelope payload."""

    model_path = resolve_repo_path(config["model_path"])
    urdf_path = resolve_repo_path(config["source_urdf"])
    controller_path = resolve_repo_path(config["controller_source"])
    model = load_mujoco_model(model_path)
    urdf_joints = parse_urdf_joints(urdf_path)
    actuators = actuator_rows(model)
    envelopes: list[dict[str, Any]] = []
    for row in actuators:
        urdf = urdf_record_for_mjcf_joint(row["joint_name"], urdf_joints)
        torque = torque_for_joint(row["joint_group"], config)
        envelopes.append(
            {
                **row,
                "urdf_joint": urdf,
                "torque_envelope": torque,
                "torque_can_be_used_as_verified_forcerange": False,
                "reason_not_verified_forcerange": "torque_side is unknown_motor_side_or_joint_output",
            }
        )
    missing_fields = {field: "manual_required" for field in MISSING_FIELDS}
    missing_fields["rated_torque"] = "PARTIAL_TEAM_PROVIDED_TORQUE_SIDE_UNKNOWN"
    missing_fields["peak_torque"] = "PARTIAL_TEAM_PROVIDED_TORQUE_SIDE_UNKNOWN"
    payload = {
        "schema_version": 1,
        "version": config["version"],
        "status": "PARTIAL_ACTUATOR_ENVELOPE",
        "valid_for": "bounded_diagnostic_only",
        "invalid_for": ["sim2real_claim", "walking_success_claim", "verified_joint_forcerange_claim"],
        "source_inputs": {
            "model_path": rel_path(model_path),
            "source_urdf": rel_path(urdf_path),
            "controller_source": rel_path(controller_path),
            "torque_data": config["torque_data"],
        },
        "model_actuator_summary": {
            "actuator_count": int(model.nu),
            "actuator_order": [item["actuator_name"] for item in actuators],
            "ctrlrange_count": sum(1 for item in actuators if item["ctrlrange_status"] == "FOUND"),
            "forcerange_count": sum(1 for item in actuators if item["forcerange_status"] == "FOUND"),
            "verified_gear_count": sum(1 for item in actuators if item["gear_status"] == "FOUND"),
        },
        "actuator_mapping": actuators,
        "actuator_envelopes": envelopes,
        "controller_semantics": controller_semantics(controller_path),
        "profiles": build_profiles(config, envelopes),
        "known_partial_manual_required": {
            "known": ["actuator_count", "actuator_order", "actuator_name", "joint_mapping", "ctrlrange", "joint_axis", "joint_range"],
            "partial": ["urdf_effort_velocity", "rated_torque", "peak_torque", "python_pd_controller_semantics"],
            "manual_required": missing_fields,
        },
        "recommendation": {
            "can_proceed_to_foot_x_actuator_controller_sensitivity": True,
            "mode": "bounded_diagnostic_only",
            "reason": "v5_22 mechanical sanity gate is PARTIAL_PASS and torque data is available, but torque side and controller/motor specs remain unresolved.",
        },
    }
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON payload."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_report(path: Path, payload: dict[str, Any]) -> None:
    """Write Markdown report."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Seedon v5_22 Actuator Envelope Report",
        "",
        "Task class: Class C actuator-envelope diagnostic. This report does not claim walking success and does not convert team-provided torque into verified joint forcerange.",
        "",
        "## Summary",
        "",
        f"- Status: `{payload['status']}`",
        f"- Valid for: `{payload['valid_for']}`",
        f"- Invalid for: `{payload['invalid_for']}`",
        f"- Torque side: `{payload['source_inputs']['torque_data']['torque_side']}`",
        f"- Can proceed to foot x actuator/controller sensitivity: `{payload['recommendation']['can_proceed_to_foot_x_actuator_controller_sensitivity']}`",
        "",
        "## Source Inputs",
        "",
    ]
    for key, value in payload["source_inputs"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Actuator Mapping Table", ""])
    lines.extend(["| idx | actuator | joint | group | ctrlrange | forcerange | gear status |", "|---:|---|---|---|---|---|---|"])
    for item in payload["actuator_mapping"]:
        lines.append(
            f"| {item['actuator_index']} | `{item['actuator_name']}` | `{item['joint_name']}` | "
            f"`{item['joint_group']}` | `{item['ctrlrange']}` | `{item['forcerange']}` | `{item['gear_status']}` |"
        )
    lines.extend(["", "## Torque Envelope Table", ""])
    lines.extend(["| joint | group | rated Nm | peak Nm | torque side | source | confidence | verified forcerange? |", "|---|---|---:|---:|---|---|---|---|"])
    for item in payload["actuator_envelopes"]:
        torque = item["torque_envelope"]
        lines.append(
            f"| `{item['joint_name']}` | `{item['joint_group']}` | {torque['rated_nm']:.1f} | {torque['peak_nm']:.1f} | "
            f"`{torque['torque_side']}` | `{torque['source']}` | `{torque['confidence']}` | `False` |"
        )
    lines.extend(["", "## Controller Semantics", ""])
    for key, value in payload["controller_semantics"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Known / Partial / Manual Required Fields", ""])
    kpm = payload["known_partial_manual_required"]
    lines.append(f"- Known: `{kpm['known']}`")
    lines.append(f"- Partial: `{kpm['partial']}`")
    for field, status in kpm["manual_required"].items():
        lines.append(f"- `{field}`: `{status}`")
    lines.extend(["", "## Why Ankle Pitch Is High Risk", ""])
    lines.append("- Ankle pitch has the lowest provided rated torque: `5 Nm`.")
    lines.append("- Ankle pitch is directly involved in toe handoff / rollover authority.")
    lines.append("- Current foot contact geoms cannot classify center/toe/heel, so ankle authority sensitivity must stay bounded.")
    lines.append("- Torque side is unknown, so values cannot be treated as verified joint-output torque.")
    lines.extend(["", "## What Can Be Used For Bounded Diagnostic", ""])
    lines.append("- `rated_safe` profile for conservative bounded diagnostic.")
    lines.append("- `peak_upper_bound` profile for short-burst upper-bound diagnostic only.")
    lines.append("- `ankle_risk_sweep` values: `5, 10, 14, 20 Nm`.")
    lines.append("- MJCF actuator mapping and ctrlrange can be used as simulation metadata, not verified motor spec.")
    lines.extend(["", "## What Must Not Be Claimed", ""])
    lines.append("- Do not claim walking success.")
    lines.append("- Do not claim sim2real validity.")
    lines.append("- Do not claim provided torque is verified joint forcerange.")
    lines.append("- Do not claim continuous gait is safe under `peak_upper_bound`.")
    lines.extend(["", "## Next Step Recommendation", ""])
    recommendation = payload["recommendation"]
    lines.append(f"- Proceed: `{recommendation['can_proceed_to_foot_x_actuator_controller_sensitivity']}`")
    lines.append(f"- Mode: `{recommendation['mode']}`")
    lines.append(f"- Reason: {recommendation['reason']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> int:
    """Build Seedon v5_22 actuator envelope."""

    args = parse_args()
    config = load_config(args.config)
    payload = build_payload(config)
    artifacts_dir = resolve_repo_path(config["artifacts_dir"])
    output_path = artifacts_dir / "envelope.json"
    model_summary_path = artifacts_dir / "model_actuator_summary.json"
    report_path = resolve_repo_path(config["report_path"])
    write_json(output_path, payload)
    write_json(model_summary_path, payload["model_actuator_summary"])
    write_report(report_path, payload)
    print(f"status={payload['status']}")
    print(f"output={output_path}")
    print(f"report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
