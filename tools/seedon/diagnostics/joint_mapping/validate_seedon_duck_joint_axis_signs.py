"""Generate a conservative Seedon/Duck joint axis and sign readiness report.

The selected Duck XML does not expose explicit leg-joint axes. This diagnostic
therefore does not infer signs or trajectories. It records what is known from
the semantic mapping and marks sign validation as manual/simulation required.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TOOL_VERSION = "seedon-duck-joint-axis-sign-validation-v1"
UNKNOWN = "unknown"
MANUAL_REQUIRED = "manual_required"

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MAPPING = REPO_ROOT / "references" / "open_duck_mini" / "seedon_duck_joint_mapping.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "configs" / "seedon" / "seedon_duck_joint_sign_mapping.yaml"
DEFAULT_REPORT = REPO_ROOT / "docs" / "seedon_duck_joint_axis_sign_report.md"


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
    """Write a stable JSON-compatible YAML file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=False)
        file.write("\n")


def axis_status(axis: Any) -> str:
    """Classify whether a joint axis is explicitly available."""

    if isinstance(axis, list) and len(axis) == 3 and all(isinstance(value, (int, float)) for value in axis):
        return "explicit"
    return UNKNOWN


def range_status(joint_range: Any) -> str:
    """Classify whether a joint range is explicitly available."""

    if isinstance(joint_range, list) and len(joint_range) == 2:
        return "explicit"
    return UNKNOWN


def build_sign_mapping(mapping: dict[str, Any], source_path: Path) -> dict[str, Any]:
    """Build conservative sign validation output from semantic mapping data."""

    records: list[dict[str, Any]] = []
    blocking = 0
    for item in mapping.get("joint_mappings", []):
        seedon_axis_state = axis_status(item.get("seedon_axis"))
        duck_axis_state = axis_status(item.get("duck_axis"))
        sign_validated = seedon_axis_state == "explicit" and duck_axis_state == "explicit"
        if not sign_validated:
            blocking += 1
        records.append(
            {
                "semantic_joint": item.get("semantic_joint"),
                "side": item.get("side", UNKNOWN),
                "seedon_joint": item.get("seedon_joint"),
                "duck_joint": item.get("duck_joint"),
                "seedon_axis": item.get("seedon_axis"),
                "duck_axis": item.get("duck_axis"),
                "seedon_axis_status": seedon_axis_state,
                "duck_axis_status": duck_axis_state,
                "seedon_range": item.get("seedon_range"),
                "duck_range": item.get("duck_range"),
                "seedon_range_status": range_status(item.get("seedon_range")),
                "duck_range_status": range_status(item.get("duck_range")),
                "semantic_mapping_confidence": item.get("mapping_confidence", UNKNOWN),
                "sign_relation": UNKNOWN,
                "sign_validated": sign_validated,
                "validation_status": "ready_for_simulation_check" if sign_validated else MANUAL_REQUIRED,
                "source": {
                    "mapping_file": str(source_path),
                    "seedon_axis": "semantic mapping extracted from Seedon XML" if seedon_axis_state == "explicit" else UNKNOWN,
                    "duck_axis": "not explicit in selected Duck XML" if duck_axis_state == UNKNOWN else "semantic mapping extracted from Duck XML",
                },
                "confidence": "low" if not sign_validated else "medium",
                "notes": "Do not transfer sign or trajectory until simulation perturbation validates this mapping.",
            }
        )

    total = len(records)
    return {
        "schema_version": 1,
        "tool_version": TOOL_VERSION,
        "generated_at": utc_now_iso(),
        "purpose": "Read-only Seedon/Duck semantic joint axis and sign readiness gate.",
        "sources": {
            "semantic_mapping": str(source_path),
            "seedon_parameters": mapping.get("source_files", {}).get("seedon_parameters"),
            "duck_parameters": mapping.get("source_files", {}).get("duck_parameters"),
        },
        "summary": {
            "status": "NOT_READY" if blocking else "INCONCLUSIVE",
            "mapped_joint_count": total,
            "sign_validated_count": total - blocking,
            "manual_required_count": blocking,
            "direct_joint_trajectory_transfer_allowed": False,
            "reason": "Duck leg-joint axes are unknown/null in the selected XML, so sign relation remains unknown.",
        },
        "joint_sign_mappings": records,
        "required_next_diagnostics": [
            "Load compiled Seedon and Duck models or equivalent simulation state.",
            "Apply small positive and negative perturbations to each mapped joint.",
            "Record semantic motion direction for hip yaw/roll/pitch, knee pitch, and ankle pitch.",
            "Only then fill sign_relation with same/opposite and raise confidence.",
        ],
        "safety_notes": [
            "This file is a validation gate, not a controller.",
            "Do not apply Duck joint trajectories to Seedon.",
            "Do not infer Duck joint axes from XML omissions in this report.",
        ],
    }


def value_text(value: Any) -> str:
    """Format Markdown values."""

    if value is None:
        return "`unknown`"
    return f"`{value}`"


def write_report(path: Path, payload: dict[str, Any]) -> None:
    """Write a Markdown report for the sign mapping gate."""

    path.parent.mkdir(parents=True, exist_ok=True)
    summary = payload["summary"]
    lines = [
        "# Seedon Duck Joint Axis Sign Report",
        "",
        "Generated by `tools/seedon/diagnostics/joint_mapping/validate_seedon_duck_joint_axis_signs.py`.",
        "",
        "## Summary",
        "",
        f"- Status: `{summary['status']}`",
        f"- Mapped joints: `{summary['mapped_joint_count']}`",
        f"- Sign validated: `{summary['sign_validated_count']}`",
        f"- Manual required: `{summary['manual_required_count']}`",
        "- Direct Duck joint trajectory transfer allowed: `false`",
        f"- Reason: {summary['reason']}",
        "",
        "## Joint Axis / Sign Gate",
        "",
        "| Semantic joint | Seedon joint | Duck joint | Seedon axis | Duck axis | Sign relation | Status | Confidence |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for item in payload["joint_sign_mappings"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{item['semantic_joint']}`",
                    f"`{item['seedon_joint']}`",
                    f"`{item['duck_joint']}`",
                    value_text(item["seedon_axis"]),
                    value_text(item["duck_axis"]),
                    f"`{item['sign_relation']}`",
                    f"`{item['validation_status']}`",
                    f"`{item['confidence']}`",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Blocking Gap",
            "",
            "Duck leg-joint axes are `unknown/null` in the selected XML. Seedon axes are explicit, but sign relation cannot be validated from names or ranges alone.",
            "",
            "## Required Next Diagnostics",
            "",
        ]
    )
    for diagnostic in payload["required_next_diagnostics"]:
        lines.append(f"- {diagnostic}")
    lines.extend(["", "## Safety Notes", ""])
    for note in payload["safety_notes"]:
        lines.append(f"- {note}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--output-yaml", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> int:
    """Run the conservative joint axis/sign validation gate."""

    args = parse_args()
    try:
        mapping = load_json_yaml(args.mapping)
        payload = build_sign_mapping(mapping, args.mapping)
        write_json_yaml(args.output_yaml, payload)
        write_report(args.report, payload)
    except ValueError as exc:
        print(f"Joint sign validation failed: {exc}")
        return 1
    print(f"Wrote {args.output_yaml}")
    print(f"Wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
