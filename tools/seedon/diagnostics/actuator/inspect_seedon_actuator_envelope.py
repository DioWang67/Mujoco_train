"""Inspect Seedon actuator envelope fields from extracted robot parameters.

This is a read-only diagnostic. It records explicit actuator fields from
`configs/seedon/seedon_robot_parameters.yaml` and marks missing actuator envelope
fields as unknown/manual required instead of inferring control semantics.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TOOL_VERSION = "seedon-actuator-envelope-inspector-v1"
UNKNOWN = "unknown"
MANUAL_REQUIRED = "manual_required"

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SEEDON_YAML = REPO_ROOT / "configs" / "seedon" / "seedon_robot_parameters.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "configs" / "seedon" / "seedon_actuator_envelope.yaml"
DEFAULT_REPORT = REPO_ROOT / "docs" / "seedon_actuator_envelope_report.md"


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_json_yaml(path: Path) -> dict[str, Any]:
    """Load a JSON-compatible YAML file."""

    if not path.is_file():
        raise ValueError(f"Input file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        loaded = json.load(file)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return loaded


def write_json_yaml(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON-compatible YAML output."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=False)
        file.write("\n")


def field_status(value: Any) -> str:
    """Return explicit/unknown status for an extracted field."""

    return "explicit" if value is not None else UNKNOWN


def side_from_joint(joint_name: str | None) -> str:
    """Infer side from Seedon joint naming convention."""

    if not joint_name:
        return UNKNOWN
    if joint_name.startswith("L_"):
        return "left"
    if joint_name.startswith("R_"):
        return "right"
    return UNKNOWN


def build_envelope(robot: dict[str, Any], source_path: Path) -> dict[str, Any]:
    """Build actuator envelope inspection output."""

    actuator_records = []
    missing_kp = 0
    missing_forcerange = 0
    missing_gear = 0
    explicit_ctrlrange = 0

    for actuator in robot.get("actuators", []):
        kp = actuator.get("kp")
        ctrlrange = actuator.get("ctrlrange")
        forcerange = actuator.get("forcerange")
        gear = actuator.get("gear")
        if kp is None:
            missing_kp += 1
        if forcerange is None:
            missing_forcerange += 1
        if gear is None:
            missing_gear += 1
        if ctrlrange is not None:
            explicit_ctrlrange += 1
        actuator_records.append(
            {
                "name": actuator.get("name"),
                "joint": actuator.get("joint"),
                "side": side_from_joint(actuator.get("joint")),
                "kind": actuator.get("kind"),
                "ctrlrange": ctrlrange,
                "kp": kp,
                "forcerange": forcerange,
                "gear": gear,
                "field_status": {
                    "kind": field_status(actuator.get("kind")),
                    "joint": field_status(actuator.get("joint")),
                    "ctrlrange": field_status(ctrlrange),
                    "kp": field_status(kp),
                    "forcerange": field_status(forcerange),
                    "gear": field_status(gear),
                },
                "confidence": {
                    "kind": "high" if actuator.get("kind") else MANUAL_REQUIRED,
                    "ctrlrange": "high" if ctrlrange is not None else MANUAL_REQUIRED,
                    "kp": MANUAL_REQUIRED if kp is None else "high",
                    "forcerange": MANUAL_REQUIRED if forcerange is None else "high",
                    "gear": MANUAL_REQUIRED if gear is None else "high",
                },
                "source": "Explicit actuator fields extracted from Seedon MJCF XML.",
                "notes": "Envelope is not a torque limit unless Seedon control semantics confirm it.",
            }
        )

    total = len(actuator_records)
    return {
        "schema_version": 1,
        "tool_version": TOOL_VERSION,
        "generated_at": utc_now_iso(),
        "purpose": "Read-only Seedon actuator envelope inspection for Duck-like gait readiness.",
        "sources": {
            "seedon_parameters": str(source_path),
            "source_xml": robot.get("source_xml"),
        },
        "summary": {
            "status": "NOT_READY",
            "actuator_count": total,
            "actuator_kind_counts": robot.get("summary", {}).get("actuator_kinds", {}),
            "explicit_ctrlrange_count": explicit_ctrlrange,
            "missing_kp_count": missing_kp,
            "missing_forcerange_count": missing_forcerange,
            "missing_gear_count": missing_gear,
            "direct_duck_actuator_transfer_allowed": False,
            "reason": "Seedon actuator ctrlrange is explicit, but kp/forcerange/gear and control semantics require clarification.",
        },
        "actuators": actuator_records,
        "manual_required": [
            {
                "field": "seedon_actuator_kp",
                "count": missing_kp,
                "reason": "Seedon XML exposes motor actuators without explicit kp.",
            },
            {
                "field": "seedon_actuator_forcerange",
                "count": missing_forcerange,
                "reason": "Seedon XML exposes no explicit actuator force range.",
            },
            {
                "field": "seedon_actuator_gear",
                "count": missing_gear,
                "reason": "Seedon XML exposes no explicit actuator gear in extracted fields.",
            },
            {
                "field": "control_semantics",
                "count": total,
                "reason": "Need environment/control-path inspection before interpreting ctrlrange as torque, position, or normalized action semantics.",
            },
        ],
        "safe_reference_fields": [
            "actuator name",
            "actuator kind",
            "joint binding",
            "explicit ctrlrange as source metadata",
        ],
        "not_safe_to_transfer": [
            "Duck kp",
            "Duck forcerange",
            "Duck position-actuator semantics",
            "Duck joint trajectory",
        ],
    }


def value_text(value: Any) -> str:
    """Format a Markdown value."""

    if value is None:
        return "`unknown`"
    return f"`{value}`"


def write_report(path: Path, payload: dict[str, Any]) -> None:
    """Write the actuator envelope report."""

    path.parent.mkdir(parents=True, exist_ok=True)
    summary = payload["summary"]
    lines = [
        "# Seedon Actuator Envelope Report",
        "",
        "Generated by `tools/seedon/diagnostics/actuator/inspect_seedon_actuator_envelope.py`.",
        "",
        "## Summary",
        "",
        f"- Status: `{summary['status']}`",
        f"- Actuator count: `{summary['actuator_count']}`",
        f"- Actuator kinds: `{summary['actuator_kind_counts']}`",
        f"- Explicit ctrlrange count: `{summary['explicit_ctrlrange_count']}`",
        f"- Missing kp count: `{summary['missing_kp_count']}`",
        f"- Missing forcerange count: `{summary['missing_forcerange_count']}`",
        f"- Missing gear count: `{summary['missing_gear_count']}`",
        "- Direct Duck actuator transfer allowed: `false`",
        f"- Reason: {summary['reason']}",
        "",
        "## Actuator Fields",
        "",
        "| Actuator | Joint | Kind | Ctrlrange | kp | Forcerange | Gear | Confidence |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for item in payload["actuators"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{item['name']}`",
                    f"`{item['joint']}`",
                    f"`{item['kind']}`",
                    value_text(item["ctrlrange"]),
                    value_text(item["kp"]),
                    value_text(item["forcerange"]),
                    value_text(item["gear"]),
                    f"`ctrlrange:{item['confidence']['ctrlrange']}; kp:{item['confidence']['kp']}; forcerange:{item['confidence']['forcerange']}`",
                ]
            )
            + " |"
        )

    lines.extend(["", "## Manual Required / Unknown", ""])
    for item in payload["manual_required"]:
        lines.append(f"- `{item['field']}` (`count={item['count']}`): {item['reason']}")

    lines.extend(["", "## Safe Reference Fields", ""])
    for field in payload["safe_reference_fields"]:
        lines.append(f"- {field}")

    lines.extend(["", "## Not Safe To Transfer", ""])
    for field in payload["not_safe_to_transfer"]:
        lines.append(f"- {field}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seedon-yaml", type=Path, default=DEFAULT_SEEDON_YAML)
    parser.add_argument("--output-yaml", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> int:
    """Run actuator envelope inspection."""

    args = parse_args()
    try:
        robot = load_json_yaml(args.seedon_yaml)
        payload = build_envelope(robot, args.seedon_yaml)
        write_json_yaml(args.output_yaml, payload)
        write_report(args.report, payload)
    except ValueError as exc:
        print(f"Actuator envelope inspection failed: {exc}")
        return 1
    print(f"Wrote {args.output_yaml}")
    print(f"Wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
