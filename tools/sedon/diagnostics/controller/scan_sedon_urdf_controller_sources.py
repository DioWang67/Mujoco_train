"""Scan Sedon URDF and controller source candidates.

This Class C diagnostic is read-only. It does not call existing extraction
pipelines, does not run MuJoCo simulation, does not modify source XML/YAML/Python
files, and does not enter PPO or reward/training code.
"""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT = REPO_ROOT / "configs" / "sedon" / "sedon_control_source_candidates.yaml"
DEFAULT_REPORT = REPO_ROOT / "docs" / "sedon_urdf_controller_source_scan.md"
SCAN_ROOTS = (
    "private_assets",
    "configs",
    "sedon_baseline",
    "tools",
    "docs",
    "references",
)
ALLOWED_SUFFIXES = {".urdf", ".xacro", ".xml", ".yaml", ".yml", ".json", ".toml", ".py", ".md", ".txt"}
CONTROLLER_KEYWORDS = (
    "ros2_control",
    "controller_manager",
    "joint_trajectory_controller",
    "effort_controller",
    "position_controller",
    "velocity_controller",
    "command_interfaces",
    "state_interfaces",
    "pid",
    "update_rate",
)
PYTHON_CONTROL_KEYWORDS = (
    "data.ctrl",
    "ctrlrange",
    "forcerange",
    "gear",
    "stiffness",
    "damping",
    "action_scale",
    "action_joint_delta_scale",
    "torque_saturation",
    "default_joint_pos",
    "default_pose",
    "nominal_joint_qpos",
    "target_q",
    "np.clip",
    "clip_action",
    "saturation",
    "_do_pd_simulation",
    "_apply_safe_joint_target_clamps",
    "_ctrl_range",
    "armature",
    "frictionloss",
)
MOTOR_SPEC_KEYWORDS = (
    "motor model",
    "max torque",
    "max velocity",
    "gear ratio",
    "current limit",
    "encoder",
    "backlash",
    "servo mode",
)
MAX_CONTROLLER_FINDINGS = 200
MAX_PYTHON_FINDINGS = 600
MAX_MOTOR_FINDINGS = 200


@dataclass(frozen=True)
class LineFinding:
    """One source-line finding.

    Args:
        source_file: Repository-relative source path.
        line_number: One-based line number.
        field: Semantic field name.
        value: Compact value or source line.
        confidence: Confidence level.
        category: Finding category.
    """

    source_file: str
    line_number: int
    field: str
    value: str
    confidence: str
    category: str


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def rel_path(path: Path) -> str:
    """Return repository-relative path text."""

    try:
        return str(path.resolve().relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def iter_scan_files() -> list[Path]:
    """Return candidate files under the requested scan roots."""

    excluded = {DEFAULT_OUTPUT.resolve(), DEFAULT_REPORT.resolve()}
    files: list[Path] = []
    for root_name in SCAN_ROOTS:
        root = REPO_ROOT / root_name
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.resolve() in excluded:
                continue
            if path.is_file() and path.suffix.lower() in ALLOWED_SUFFIXES:
                files.append(path)
    return sorted(files)


def safe_read_lines(path: Path) -> list[str]:
    """Read text lines with tolerant decoding."""

    try:
        return path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()


def line_number_for_pattern(lines: list[str], pattern: str, start: int = 0) -> int:
    """Return one-based line number for a regex pattern."""

    regex = re.compile(pattern)
    for index in range(max(0, start), len(lines)):
        if regex.search(lines[index]):
            return index + 1
    return 0


def line_number_for_text(lines: list[str], text: str, start: int = 0) -> int:
    """Return one-based line number for a literal text fragment."""

    for index in range(max(0, start), len(lines)):
        if text in lines[index]:
            return index + 1
    return 0


def append_line_finding(
    findings: list[dict[str, Any]],
    *,
    path: Path,
    line_number: int,
    field: str,
    value: Any,
    confidence: str,
    category: str,
) -> None:
    """Append a normalized finding record."""

    findings.append(
        {
            "source_file": rel_path(path),
            "line_number": int(line_number),
            "field": field,
            "value": str(value),
            "confidence": confidence,
            "category": category,
        }
    )


def parse_xml(path: Path) -> ET.Element | None:
    """Parse XML-like files when possible."""

    try:
        return ET.parse(path).getroot()
    except ET.ParseError:
        return None


def scan_urdf_file(path: Path) -> dict[str, Any] | None:
    """Scan a URDF/Xacro/XML robot-description candidate."""

    root = parse_xml(path)
    if root is None or root.tag != "robot":
        return None
    lines = safe_read_lines(path)
    joints: list[dict[str, Any]] = []
    findings: list[dict[str, Any]] = []
    inertial_count = 0
    collision_count = 0
    for link in root.findall("link"):
        link_name = link.attrib.get("name", "unknown")
        inertial = link.find("inertial")
        if inertial is not None:
            inertial_count += 1
            line = line_number_for_text(lines, f'name="{link_name}"')
            append_line_finding(
                findings,
                path=path,
                line_number=line,
                field="inertial_mass_com_inertia",
                value=f"link={link_name}",
                confidence="high",
                category="urdf_xacro",
            )
        collisions = link.findall("collision")
        collision_count += len(collisions)
        if collisions:
            line = line_number_for_text(lines, f'name="{link_name}"')
            append_line_finding(
                findings,
                path=path,
                line_number=line,
                field="collision_geometry",
                value=f"link={link_name}, collision_count={len(collisions)}",
                confidence="high",
                category="urdf_xacro",
            )

    for joint in root.findall("joint"):
        name = joint.attrib.get("name", "unknown")
        axis = joint.find("axis")
        limit = joint.find("limit")
        joint_line = line_number_for_text(lines, f'name="{name}"')
        record = {
            "name": name,
            "type": joint.attrib.get("type"),
            "axis": axis.attrib.get("xyz") if axis is not None else None,
            "lower": limit.attrib.get("lower") if limit is not None else None,
            "upper": limit.attrib.get("upper") if limit is not None else None,
            "effort": limit.attrib.get("effort") if limit is not None else None,
            "velocity": limit.attrib.get("velocity") if limit is not None else None,
            "line_number": joint_line,
        }
        joints.append(record)
        if axis is not None:
            append_line_finding(
                findings,
                path=path,
                line_number=line_number_for_pattern(lines, r"<axis\b", max(0, joint_line - 1)),
                field="joint_axis",
                value=f"{name}: {record['axis']}",
                confidence="high",
                category="urdf_xacro",
            )
        if limit is not None:
            append_line_finding(
                findings,
                path=path,
                line_number=line_number_for_pattern(lines, r"<limit\b", max(0, joint_line - 1)),
                field="joint_limit_range_effort_velocity",
                value=(
                    f"{name}: lower={record['lower']} upper={record['upper']} "
                    f"effort={record['effort']} velocity={record['velocity']}"
                ),
                confidence="high",
                category="urdf_xacro",
            )

    transmissions = root.findall("transmission")
    for transmission in transmissions:
        name = transmission.attrib.get("name", "unnamed")
        line = line_number_for_pattern(lines, rf"<transmission\b.*{re.escape(name)}|<transmission\b")
        append_line_finding(
            findings,
            path=path,
            line_number=line,
            field="transmission",
            value=name,
            confidence="high",
            category="urdf_xacro",
        )
        for tag in ("mechanicalReduction", "hardwareInterface"):
            for element in transmission.findall(f".//{tag}"):
                append_line_finding(
                    findings,
                    path=path,
                    line_number=line_number_for_pattern(lines, rf"<{tag}\b", line - 1),
                    field=tag,
                    value=(element.text or "").strip(),
                    confidence="high",
                    category="urdf_xacro",
                )

    return {
        "source_file": rel_path(path),
        "robot_name": root.attrib.get("name"),
        "joint_count": len(joints),
        "joints": joints,
        "inertial_link_count": inertial_count,
        "collision_count": collision_count,
        "transmission_count": len(transmissions),
        "findings": findings,
        "status": "FOUND",
    }


def keyword_findings(path: Path, keywords: Iterable[str], category: str, confidence: str = "medium") -> list[dict[str, Any]]:
    """Find keyword hits in one file."""

    findings: list[dict[str, Any]] = []
    lowered_keywords = tuple(keyword.lower() for keyword in keywords)
    for index, line in enumerate(safe_read_lines(path), start=1):
        lowered = line.lower()
        for keyword, lowered_keyword in zip(keywords, lowered_keywords):
            if lowered_keyword in lowered:
                if lowered_keyword == "np.clip" and not any(
                    hint in lowered for hint in ("ctrl", "action", "target", "torque", "joint")
                ):
                    continue
                append_line_finding(
                    findings,
                    path=path,
                    line_number=index,
                    field=keyword,
                    value=line.strip()[:240],
                    confidence=confidence,
                    category=category,
                )
    return findings


def python_control_priority(finding: dict[str, Any]) -> tuple[int, str, int]:
    """Order Python/MuJoCo findings by diagnostic value."""

    source = finding["source_file"]
    field = finding["field"]
    if source == "sedon_baseline/env.py":
        source_priority = 0
    elif source.startswith("private_assets/sedon/"):
        source_priority = 1
    elif source.startswith("sedon_baseline/"):
        source_priority = 2
    elif source.startswith("configs/sedon/"):
        source_priority = 3
    else:
        source_priority = 4

    field_priority = {
        "data.ctrl": 0,
        "_do_pd_simulation": 1,
        "_ctrl_range": 2,
        "ctrlrange": 3,
        "forcerange": 4,
        "gear": 5,
        "np.clip": 6,
        "action_joint_delta_scale": 7,
        "action_scale": 8,
        "nominal_joint_qpos": 9,
        "default_joint_pos": 10,
        "default_pose": 11,
        "target_q": 12,
        "torque_saturation": 13,
        "saturation": 14,
    }.get(field, 20)
    return (source_priority * 100 + field_priority, source, int(finding["line_number"]))


def scan_controller_configs(files: list[Path]) -> list[dict[str, Any]]:
    """Scan YAML/JSON/TOML/XML/MD/TXT files for ROS/controller config candidates."""

    findings: list[dict[str, Any]] = []
    for path in files:
        if path.suffix.lower() not in {".yaml", ".yml", ".json", ".toml", ".xml"}:
            continue
        findings.extend(keyword_findings(path, CONTROLLER_KEYWORDS, "controller_config", "medium"))
    return findings


def scan_python_control(files: list[Path]) -> list[dict[str, Any]]:
    """Scan Python/MuJoCo control semantics candidates."""

    findings: list[dict[str, Any]] = []
    for path in files:
        source = rel_path(path)
        if not (
            source.startswith("sedon_baseline/")
            or source.startswith("tools/")
            or source.startswith("configs/sedon/")
            or source.startswith("private_assets/sedon/")
        ):
            continue
        if path.suffix.lower() not in {".py", ".xml", ".json", ".yaml", ".yml", ".md"}:
            continue
        findings.extend(keyword_findings(path, PYTHON_CONTROL_KEYWORDS, "python_mujoco_control", "medium"))
    return sorted(findings, key=python_control_priority)


def scan_motor_specs(files: list[Path]) -> list[dict[str, Any]]:
    """Scan for motor specification candidates."""

    findings: list[dict[str, Any]] = []
    for path in files:
        source = rel_path(path)
        if not (
            source.startswith("private_assets/sedon/")
            or source.startswith("configs/sedon/")
            or source.startswith("sedon_baseline/")
        ):
            continue
        findings.extend(keyword_findings(path, MOTOR_SPEC_KEYWORDS, "motor_spec", "low"))
    return findings


def classify_summary(
    urdf_records: list[dict[str, Any]],
    controller_findings: list[dict[str, Any]],
    python_findings: list[dict[str, Any]],
    motor_findings: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build high-level FOUND/PARTIAL/NOT_FOUND status."""

    has_effort_velocity = any(
        joint.get("effort") is not None and joint.get("velocity") is not None
        for record in urdf_records
        for joint in record["joints"]
    )
    has_transmission = any(record["transmission_count"] > 0 for record in urdf_records)
    has_motor_specs = bool(motor_findings)
    actuator_status = "PARTIAL" if has_effort_velocity and not has_transmission else "NOT_FOUND"
    if has_effort_velocity and has_transmission and has_motor_specs:
        actuator_status = "FOUND"
    elif has_effort_velocity:
        actuator_status = "PARTIAL"

    return {
        "overall_status": "PARTIAL" if urdf_records or python_findings else "NOT_FOUND",
        "urdf_xacro_status": "FOUND" if urdf_records else "NOT_FOUND",
        "controller_yaml_status": "FOUND" if controller_findings else "NOT_FOUND",
        "motor_spec_status": "FOUND" if has_motor_specs else "NOT_FOUND",
        "python_control_semantics_status": "FOUND" if python_findings else "NOT_FOUND",
        "actuator_envelope_status": actuator_status,
        "partial_actuator_envelope": bool(has_effort_velocity and not has_transmission),
        "notes": (
            "Joint effort/velocity limits are present, but this is only a partial actuator envelope; "
            "it is not a complete motor model without gear/reduction/control mode/PID or motor spec."
            if has_effort_velocity
            else "No joint effort/velocity actuator envelope source found."
        ),
    }


def build_payload() -> dict[str, Any]:
    """Run the read-only source scan and return output payload."""

    files = iter_scan_files()
    urdf_records = [
        record
        for path in files
        if path.suffix.lower() in {".urdf", ".xacro", ".xml"}
        for record in [scan_urdf_file(path)]
        if record is not None
    ]
    controller_findings = scan_controller_configs(files)[:MAX_CONTROLLER_FINDINGS]
    python_findings = scan_python_control(files)[:MAX_PYTHON_FINDINGS]
    motor_findings = scan_motor_specs(files)[:MAX_MOTOR_FINDINGS]
    all_findings: list[dict[str, Any]] = []
    for record in urdf_records:
        all_findings.extend(record["findings"])
    all_findings.extend(controller_findings)
    all_findings.extend(python_findings)
    all_findings.extend(motor_findings)
    summary = classify_summary(urdf_records, controller_findings, python_findings, motor_findings)

    return {
        "schema_version": 1,
        "generated_at": utc_now_iso(),
        "task_class": "Class C read-only diagnostic/source inventory",
        "scan_roots": list(SCAN_ROOTS),
        "allowed_suffixes": sorted(ALLOWED_SUFFIXES),
        "summary": summary,
        "urdf_xacro_findings": urdf_records,
        "controller_config_findings": controller_findings,
        "python_mujoco_control_findings": python_findings,
        "motor_spec_findings": motor_findings,
        "all_findings": all_findings,
        "what_this_resolves": {
            "joint_axis": "FOUND" if any(f["field"] == "joint_axis" for f in all_findings) else "NOT_FOUND",
            "joint_range": "FOUND" if any("range" in f["field"] for f in all_findings) else "NOT_FOUND",
            "joint_effort": "PARTIAL" if summary["partial_actuator_envelope"] else "NOT_FOUND",
            "joint_velocity": "PARTIAL" if summary["partial_actuator_envelope"] else "NOT_FOUND",
            "inertial_mass_com_inertia": "FOUND" if any(f["field"] == "inertial_mass_com_inertia" for f in all_findings) else "NOT_FOUND",
            "collision_geometry": "FOUND" if any(f["field"] == "collision_geometry" for f in all_findings) else "NOT_FOUND",
            "python_control_path": summary["python_control_semantics_status"],
        },
        "still_missing": {
            "motor_max_torque": "manual_required",
            "motor_max_velocity": "manual_required",
            "gear_ratio": "manual_required",
            "control_mode": "manual_required",
            "pid_gains": "manual_required",
            "encoder_imu_sensor_data": "manual_required",
            "external_sedon_mechanical_or_motor_spec_path": "not_found_user_path_required",
        },
        "recommendation": {
            "foot_x_actuator_sensitivity": "PARTIAL_READY_FOR_BOUNDED_DIAGNOSTIC_ONLY",
            "blocked_for_full_actuator_model": True,
            "needs_user_external_paths": True,
            "reason": (
                "URDF joint limits and Python/MuJoCo control path are available, but motor model, gear ratio, "
                "PID gains, control mode, and sensor specs remain manual_required."
            ),
        },
    }


def write_json_yaml(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON-compatible YAML output."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def format_bool_status(value: str) -> str:
    """Format a status for Markdown."""

    return f"`{value}`"


def write_report(path: Path, payload: dict[str, Any]) -> None:
    """Write Markdown scan report."""

    path.parent.mkdir(parents=True, exist_ok=True)
    summary = payload["summary"]
    lines = [
        "# Sedon URDF / Controller Source Scan",
        "",
        "Task class: Class C read-only source inventory. This scan does not rerun extraction/readiness/foot prototype pipelines, does not run simulation, and does not modify training or evaluation code.",
        "",
        "## Summary",
        "",
        f"- Overall: {format_bool_status(summary['overall_status'])}",
        f"- URDF/Xacro: {format_bool_status(summary['urdf_xacro_status'])}",
        f"- Controller YAML/config: {format_bool_status(summary['controller_yaml_status'])}",
        f"- Motor spec: {format_bool_status(summary['motor_spec_status'])}",
        f"- Python/MuJoCo control semantics: {format_bool_status(summary['python_control_semantics_status'])}",
        f"- Actuator envelope: {format_bool_status(summary['actuator_envelope_status'])}",
        f"- Note: {summary['notes']}",
        "",
        "## URDF/Xacro Findings",
        "",
    ]
    if not payload["urdf_xacro_findings"]:
        lines.append("NOT FOUND: no URDF/Xacro robot description was found in the scan scope.")
    for record in payload["urdf_xacro_findings"]:
        lines.extend(
            [
                f"### `{record['source_file']}`",
                "",
                f"- robot name: `{record['robot_name']}`",
                f"- joints: `{record['joint_count']}`",
                f"- inertial links: `{record['inertial_link_count']}`",
                f"- collisions: `{record['collision_count']}`",
                f"- transmissions: `{record['transmission_count']}`",
                "",
                "| joint | line | axis | lower | upper | effort | velocity |",
                "|---|---:|---|---:|---:|---:|---:|",
            ]
        )
        for joint in record["joints"]:
            lines.append(
                f"| `{joint['name']}` | {joint['line_number']} | `{joint['axis']}` | "
                f"`{joint['lower']}` | `{joint['upper']}` | `{joint['effort']}` | `{joint['velocity']}` |"
            )
        lines.append("")
        if record["transmission_count"] == 0:
            lines.append("- transmission / mechanicalReduction / hardwareInterface: `NOT_FOUND`")
            lines.append("")

    lines.extend(["## Controller Config Findings", ""])
    if not payload["controller_config_findings"]:
        lines.append("NOT FOUND: no ROS controller YAML/config source was found in the scan scope.")
    else:
        lines.extend(["| file | line | field | confidence | value |", "|---|---:|---|---|---|"])
        for item in payload["controller_config_findings"][:80]:
            lines.append(
                f"| `{item['source_file']}` | {item['line_number']} | `{item['field']}` | "
                f"`{item['confidence']}` | `{item['value']}` |"
            )
    lines.append("")

    lines.extend(["## Python/MuJoCo Control Findings", ""])
    if not payload["python_mujoco_control_findings"]:
        lines.append("NOT FOUND: no Python/MuJoCo control semantics candidates were found.")
    else:
        lines.extend(["| file | line | field | confidence | value |", "|---|---:|---|---|---|"])
        for item in payload["python_mujoco_control_findings"][:120]:
            lines.append(
                f"| `{item['source_file']}` | {item['line_number']} | `{item['field']}` | "
                f"`{item['confidence']}` | `{item['value']}` |"
            )
    lines.append("")

    lines.extend(["## What This Resolves", ""])
    for field, status in payload["what_this_resolves"].items():
        lines.append(f"- `{field}`: `{status}`")
    lines.append("")

    lines.extend(["## Still Missing", ""])
    for field, status in payload["still_missing"].items():
        lines.append(f"- `{field}`: `{status}`")
    lines.append("")

    lines.extend(["## Recommendation", ""])
    recommendation = payload["recommendation"]
    lines.append(f"- Foot x actuator sensitivity: `{recommendation['foot_x_actuator_sensitivity']}`")
    lines.append(f"- Blocked for full actuator model: `{recommendation['blocked_for_full_actuator_model']}`")
    lines.append(f"- Needs user external paths: `{recommendation['needs_user_external_paths']}`")
    lines.append(f"- Reason: {recommendation['reason']}")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> int:
    """Run the read-only source scan."""

    args = parse_args()
    payload = build_payload()
    write_json_yaml(args.output, payload)
    write_report(args.report, payload)
    print(f"summary={payload['summary']['overall_status']}")
    print(f"output={args.output}")
    print(f"report={args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
