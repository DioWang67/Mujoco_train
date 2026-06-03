"""Generate Seedon contact patch readiness data from extracted parameters.

This read-only diagnostic classifies foot-related geoms from the existing
Seedon parameter snapshot. It does not run MuJoCo, move artifacts, or modify
training/evaluation logic.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TOOL_VERSION = "seedon-contact-patch-readiness-v1"
UNKNOWN = "unknown"
MANUAL_REQUIRED = "manual_required"

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SEEDON_YAML = REPO_ROOT / "configs" / "seedon" / "seedon_robot_parameters.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "configs" / "seedon" / "seedon_contact_patch_status.yaml"
DEFAULT_REPORT = REPO_ROOT / "docs" / "seedon_contact_patch_readiness_report.md"


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


def full_box_extents(size: Any) -> list[float] | None:
    """Convert MuJoCo box half-extents to full extents."""

    if isinstance(size, list) and len(size) == 3 and all(isinstance(value, (int, float)) for value in size):
        return [round(float(value) * 2.0, 9) for value in size]
    return None


def classify_geom(geom: dict[str, Any]) -> dict[str, Any]:
    """Classify one Seedon geom for contact readiness."""

    name = geom.get("name")
    geom_type = geom.get("type")
    is_foot = bool(geom.get("foot_related"))
    is_collision_name = isinstance(name, str) and "foot_collision" in name
    is_box = geom_type == "box"
    if is_foot and is_collision_name and is_box:
        role = "candidate_support_patch"
        confidence = "medium"
        readiness = "needs_sim_contact_validation"
        reason = "Explicit foot collision box is likely support contact, but contact behavior has not been simulated in this phase."
    elif is_foot:
        role = "foot_related_non_support_or_visual_candidate"
        confidence = "low"
        readiness = MANUAL_REQUIRED
        reason = "Name heuristic marks this foot-related, but it is not an explicit foot collision box."
    else:
        role = "non_foot"
        confidence = "high"
        readiness = "not_applicable"
        reason = "Not selected by foot-related name heuristic."

    return {
        "name": name,
        "body": geom.get("body"),
        "side": geom.get("side", UNKNOWN),
        "type": geom_type,
        "size": geom.get("size"),
        "box_full_extents_m": full_box_extents(geom.get("size")) if is_box else None,
        "pos": geom.get("pos"),
        "contype": geom.get("contype"),
        "conaffinity": geom.get("conaffinity"),
        "foot_related": is_foot,
        "role": role,
        "readiness": readiness,
        "confidence": confidence,
        "source": "Seedon extracted geom fields from XML.",
        "reason": reason,
    }


def build_contact_status(robot: dict[str, Any], source_path: Path) -> dict[str, Any]:
    """Build contact patch readiness payload."""

    geoms = [classify_geom(geom) for geom in robot.get("geoms", [])]
    foot_geoms = [geom for geom in geoms if geom["foot_related"]]
    support_candidates = [geom for geom in foot_geoms if geom["role"] == "candidate_support_patch"]
    sides = sorted({geom["side"] for geom in support_candidates if geom["side"] in {"left", "right"}})
    has_bilateral_support = sides == ["left", "right"]

    return {
        "schema_version": 1,
        "tool_version": TOOL_VERSION,
        "generated_at": utc_now_iso(),
        "purpose": "Read-only Seedon foot contact patch readiness for Duck-like gait preparation.",
        "sources": {
            "seedon_parameters": str(source_path),
            "source_xml": robot.get("source_xml"),
        },
        "summary": {
            "status": "INCONCLUSIVE" if has_bilateral_support else "NOT_READY",
            "total_geoms": len(geoms),
            "foot_related_geoms": len(foot_geoms),
            "candidate_support_patches": len(support_candidates),
            "candidate_support_sides": sides,
            "bilateral_candidate_support": has_bilateral_support,
            "contact_simulation_validated": False,
            "reason": "Bilateral explicit foot collision boxes exist, but actual support/contact patch behavior is not validated by XML-only inspection.",
        },
        "geoms": geoms,
        "manual_required": [
            {
                "field": "support_contact_behavior",
                "reason": "Requires MuJoCo contact inspection under stance/swing states.",
            },
            {
                "field": "contact_patch_extent_world_frame",
                "reason": "XML local box dimensions do not prove world-frame contact footprint under pose.",
            },
            {
                "field": "duck_contact_patch_equivalence",
                "reason": "Duck foot is mesh based; Seedon box support cannot be assumed equivalent.",
            },
        ],
        "safe_reference_fields": [
            "Seedon foot collision geom names",
            "Seedon foot collision box full extents",
            "left/right candidate support patch existence",
        ],
        "not_safe_to_claim": [
            "stable contact",
            "Duck-equivalent contact patch",
            "walking success",
        ],
    }


def value_text(value: Any) -> str:
    """Format values for Markdown."""

    if value is None:
        return "`unknown`"
    return f"`{value}`"


def write_report(path: Path, payload: dict[str, Any]) -> None:
    """Write contact readiness Markdown report."""

    path.parent.mkdir(parents=True, exist_ok=True)
    summary = payload["summary"]
    lines = [
        "# Seedon Contact Patch Readiness Report",
        "",
        "Generated by `tools/seedon/diagnostics/contact/generate_contact_readiness_report.py`.",
        "",
        "## Summary",
        "",
        f"- Status: `{summary['status']}`",
        f"- Total geoms: `{summary['total_geoms']}`",
        f"- Foot-related geoms: `{summary['foot_related_geoms']}`",
        f"- Candidate support patches: `{summary['candidate_support_patches']}`",
        f"- Candidate support sides: `{summary['candidate_support_sides']}`",
        "- Contact simulation validated: `false`",
        f"- Reason: {summary['reason']}",
        "",
        "## Geom Classification",
        "",
        "| Geom | Body | Side | Type | Full box extents m | Role | Readiness | Confidence |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for geom in payload["geoms"]:
        if not geom["foot_related"]:
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{geom['name']}`",
                    f"`{geom['body']}`",
                    f"`{geom['side']}`",
                    f"`{geom['type']}`",
                    value_text(geom["box_full_extents_m"]),
                    f"`{geom['role']}`",
                    f"`{geom['readiness']}`",
                    f"`{geom['confidence']}`",
                ]
            )
            + " |"
        )

    lines.extend(["", "## Manual Required / Unknown", ""])
    for item in payload["manual_required"]:
        lines.append(f"- `{item['field']}`: {item['reason']}")

    lines.extend(["", "## Safe Reference Fields", ""])
    for item in payload["safe_reference_fields"]:
        lines.append(f"- {item}")

    lines.extend(["", "## Not Safe To Claim", ""])
    for item in payload["not_safe_to_claim"]:
        lines.append(f"- {item}")
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
    """Run the contact patch readiness report generator."""

    args = parse_args()
    try:
        robot = load_json_yaml(args.seedon_yaml)
        payload = build_contact_status(robot, args.seedon_yaml)
        write_json_yaml(args.output_yaml, payload)
        write_report(args.report, payload)
    except ValueError as exc:
        print(f"Contact readiness generation failed: {exc}")
        return 1
    print(f"Wrote {args.output_yaml}")
    print(f"Wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
