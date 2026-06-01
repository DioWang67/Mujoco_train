"""Generate a gated Duck-like task-space reference skeleton for Sedon.

The generator refuses to invent gait timing, clearance, or trajectories when a
validated Duck gait/reference motion source is missing. It writes a structured
reference file that downstream diagnostics can read as a blocked readiness gate.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TOOL_VERSION = "sedon-duck-like-task-space-reference-v1"
UNKNOWN = "unknown"
MANUAL_REQUIRED = "manual_required"

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SCALE = REPO_ROOT / "configs" / "sedon" / "sedon_duck_scale_mapping.yaml"
DEFAULT_SIGN = REPO_ROOT / "configs" / "sedon" / "sedon_duck_joint_sign_mapping.yaml"
DEFAULT_CONTACT = REPO_ROOT / "configs" / "sedon" / "sedon_contact_patch_status.yaml"
DEFAULT_DUCK_GAIT = REPO_ROOT / "references" / "open_duck_mini" / "duck_gait_reference_metadata.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "configs" / "sedon" / "sedon_duck_like_gait_reference.yaml"
DEFAULT_REPORT = REPO_ROOT / "docs" / "sedon_duck_like_task_space_reference.md"


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


def scale_ratio(scale: dict[str, Any], field_name: str) -> Any:
    """Return a scale ratio by field name."""

    for field in scale.get("scale_fields", []):
        if field.get("field") == field_name:
            return field.get("sedon_to_duck_ratio")
    return None


def build_reference(scale: dict[str, Any], sign: dict[str, Any], contact: dict[str, Any], duck_gait: dict[str, Any]) -> dict[str, Any]:
    """Build a gated task-space reference skeleton."""

    gait_status = duck_gait.get("summary", {}).get("status", UNKNOWN)
    sign_ready = sign.get("summary", {}).get("sign_validated_count", 0) == sign.get("summary", {}).get("mapped_joint_count", -1)
    contact_validated = bool(contact.get("summary", {}).get("contact_simulation_validated", False))
    blockers = []
    if gait_status != "FOUND_CANDIDATES":
        blockers.append("duck_gait_reference_source_missing")
    if not sign_ready:
        blockers.append("joint_sign_validation_missing")
    if not contact_validated:
        blockers.append("sedon_contact_simulation_validation_missing")

    status = "BLOCKED" if blockers else "REFERENCE_DRAFT"
    return {
        "schema_version": 1,
        "tool_version": TOOL_VERSION,
        "generated_at": utc_now_iso(),
        "purpose": "Duck-like task-space gait reference skeleton for Sedon readiness workflow.",
        "sources": {
            "scale_mapping": str(DEFAULT_SCALE),
            "joint_sign_mapping": str(DEFAULT_SIGN),
            "contact_patch_status": str(DEFAULT_CONTACT),
            "duck_gait_reference_metadata": str(DEFAULT_DUCK_GAIT),
        },
        "summary": {
            "status": status,
            "direct_joint_trajectory_transfer_allowed": False,
            "training_or_reward_change_allowed": False,
            "blockers": blockers,
            "reason": "Reference remains a skeleton because required Duck gait source/sign/contact validations are incomplete."
            if blockers
            else "Required readiness gates are available for a draft task-space reference.",
        },
        "normalization_references": {
            "explicit_body_mass_ratio_sedon_to_duck": {
                "value": scale_ratio(scale, "explicit_body_mass"),
                "source": "configs/sedon/sedon_duck_scale_mapping.yaml",
                "confidence": "high",
            },
            "base_height_ratio_sedon_to_duck": {
                "value": scale_ratio(scale, "initial_base_height_z"),
                "source": "configs/sedon/sedon_duck_scale_mapping.yaml",
                "confidence": "medium",
            },
            "hip_lateral_spacing_ratio_sedon_to_duck": {
                "value": scale_ratio(scale, "hip_lateral_spacing"),
                "source": "configs/sedon/sedon_duck_scale_mapping.yaml",
                "confidence": "medium",
            },
            "local_leg_chain_path_ratio_sedon_to_duck": {
                "value": scale_ratio(scale, "local_leg_chain_path_length_average"),
                "source": "configs/sedon/sedon_duck_scale_mapping.yaml",
                "confidence": "low",
            },
        },
        "task_space_reference": {
            "gait_period_s": {
                "value": None,
                "source": None,
                "confidence": MANUAL_REQUIRED,
                "reason": "No validated Duck gait/reference motion source found.",
            },
            "stance_ratio": {
                "value": None,
                "source": None,
                "confidence": MANUAL_REQUIRED,
                "reason": "No validated Duck gait/reference motion source found.",
            },
            "swing_clearance_m": {
                "value": None,
                "source": None,
                "confidence": MANUAL_REQUIRED,
                "reason": "No validated Duck gait/reference motion source found.",
            },
            "foot_path": {
                "value": None,
                "source": None,
                "confidence": MANUAL_REQUIRED,
                "reason": "No validated Duck task-space foot path source found.",
            },
            "body_motion_targets": {
                "value": None,
                "source": None,
                "confidence": MANUAL_REQUIRED,
                "reason": "No validated Duck body motion source found.",
            },
        },
        "required_before_use": [
            "validated Duck gait/reference motion source",
            "joint sign validation",
            "Sedon contact simulation validation",
            "scripted smoke test",
        ],
        "safety_notes": [
            "This file is not a controller.",
            "Do not use this skeleton as a joint trajectory.",
            "Do not modify PPO/reward/training from this skeleton.",
        ],
    }


def value_text(value: Any) -> str:
    """Format Markdown values."""

    if value is None:
        return "`unknown`"
    return f"`{value}`"


def write_report(path: Path, payload: dict[str, Any]) -> None:
    """Write the task-space reference Markdown report."""

    path.parent.mkdir(parents=True, exist_ok=True)
    summary = payload["summary"]
    lines = [
        "# Sedon Duck-Like Task-Space Reference",
        "",
        "Generated by `tools/sedon/gait/generate_duck_like_task_space_reference.py`.",
        "",
        "## Summary",
        "",
        f"- Status: `{summary['status']}`",
        "- Direct joint trajectory transfer allowed: `false`",
        "- Training or reward change allowed: `false`",
        f"- Blockers: `{summary['blockers']}`",
        f"- Reason: {summary['reason']}",
        "",
        "## Normalization References",
        "",
        "| Field | Value | Confidence | Source |",
        "|---|---:|---|---|",
    ]
    for name, item in payload["normalization_references"].items():
        lines.append(f"| `{name}` | {value_text(item['value'])} | `{item['confidence']}` | `{item['source']}` |")

    lines.extend(
        [
            "",
            "## Task-Space Reference Fields",
            "",
            "| Field | Value | Confidence | Reason |",
            "|---|---|---|---|",
        ]
    )
    for name, item in payload["task_space_reference"].items():
        lines.append(f"| `{name}` | {value_text(item['value'])} | `{item['confidence']}` | {item['reason']} |")

    lines.extend(["", "## Required Before Use", ""])
    for item in payload["required_before_use"]:
        lines.append(f"- {item}")

    lines.extend(["", "## Safety Notes", ""])
    for note in payload["safety_notes"]:
        lines.append(f"- {note}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale-yaml", type=Path, default=DEFAULT_SCALE)
    parser.add_argument("--sign-yaml", type=Path, default=DEFAULT_SIGN)
    parser.add_argument("--contact-yaml", type=Path, default=DEFAULT_CONTACT)
    parser.add_argument("--duck-gait-yaml", type=Path, default=DEFAULT_DUCK_GAIT)
    parser.add_argument("--output-yaml", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> int:
    """Generate a gated task-space reference skeleton."""

    args = parse_args()
    try:
        scale = load_json_yaml(args.scale_yaml)
        sign = load_json_yaml(args.sign_yaml)
        contact = load_json_yaml(args.contact_yaml)
        duck_gait = load_json_yaml(args.duck_gait_yaml)
        payload = build_reference(scale, sign, contact, duck_gait)
        write_json_yaml(args.output_yaml, payload)
        write_report(args.report, payload)
    except ValueError as exc:
        print(f"Task-space reference generation failed: {exc}")
        return 1
    print(f"Wrote {args.output_yaml}")
    print(f"Wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
