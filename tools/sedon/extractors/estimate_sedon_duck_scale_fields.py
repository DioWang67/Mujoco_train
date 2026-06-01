"""Estimate conservative Sedon/Open Duck Mini scale reference fields.

This tool reads existing extracted parameter snapshots and writes a read-only
scale mapping table. It does not apply Duck parameters to Sedon and does not
modify training, reward, PPO, or evaluation logic.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TOOL_VERSION = "sedon-duck-scale-fields-v1"
UNKNOWN = "unknown"
MANUAL_REQUIRED = "manual_required"

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SEDON_YAML = REPO_ROOT / "configs" / "sedon" / "sedon_robot_parameters.yaml"
DEFAULT_DUCK_YAML = REPO_ROOT / "references" / "open_duck_mini" / "duck_robot_parameters.yaml"
DEFAULT_OUTPUT_YAML = REPO_ROOT / "configs" / "sedon" / "sedon_duck_scale_mapping.yaml"
DEFAULT_REPORT = REPO_ROOT / "docs" / "sedon_duck_scale_table.md"


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp.

    Returns:
        Timestamp string without microseconds.
    """

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_json_yaml(path: Path) -> dict[str, Any]:
    """Load the repo's JSON-compatible YAML files.

    Args:
        path: Input file path.

    Returns:
        Parsed object.

    Raises:
        ValueError: If the file is missing or not a JSON object.
    """

    if not path.is_file():
        raise ValueError(f"Input file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        loaded = json.load(file)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return loaded


def write_json_yaml(path: Path, payload: dict[str, Any]) -> None:
    """Write a stable JSON-compatible YAML file.

    Args:
        path: Output path.
        payload: Serializable mapping.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=False)
        file.write("\n")


def vector_norm(values: Any) -> float | None:
    """Return Euclidean norm for a numeric vector.

    Args:
        values: Candidate vector.

    Returns:
        Norm, or ``None`` if the input is not a numeric list.
    """

    if not isinstance(values, list) or not values:
        return None
    if not all(isinstance(value, (int, float)) for value in values):
        return None
    return math.sqrt(sum(float(value) ** 2 for value in values))


def ratio(sedon_value: float | int | None, duck_value: float | int | None) -> float | None:
    """Return Sedon/Duck ratio when both values are numeric.

    Args:
        sedon_value: Sedon scalar.
        duck_value: Duck scalar.

    Returns:
        Rounded ratio, or ``None`` when unavailable.
    """

    if not isinstance(sedon_value, (int, float)) or not isinstance(duck_value, (int, float)):
        return None
    if duck_value == 0:
        return None
    return round(float(sedon_value) / float(duck_value), 6)


def body_by_name(robot: dict[str, Any], name: str) -> dict[str, Any] | None:
    """Find a body record by name."""

    for body in robot.get("bodies", []):
        if body.get("name") == name:
            return body
    return None


def first_body_pos_z(robot: dict[str, Any]) -> float | None:
    """Return the first body z position from the extracted world body list."""

    bodies = robot.get("bodies", [])
    if not bodies:
        return None
    pos = bodies[0].get("pos")
    if isinstance(pos, list) and len(pos) >= 3 and isinstance(pos[2], (int, float)):
        return float(pos[2])
    return None


def lateral_spacing(robot: dict[str, Any], left_name: str, right_name: str) -> float | None:
    """Estimate lateral spacing from two explicit body local positions."""

    left = body_by_name(robot, left_name)
    right = body_by_name(robot, right_name)
    if left is None or right is None:
        return None
    left_pos = left.get("pos")
    right_pos = right.get("pos")
    if not isinstance(left_pos, list) or not isinstance(right_pos, list):
        return None
    if len(left_pos) < 2 or len(right_pos) < 2:
        return None
    if not isinstance(left_pos[1], (int, float)) or not isinstance(right_pos[1], (int, float)):
        return None
    return round(abs(float(left_pos[1]) - float(right_pos[1])), 9)


def chain_path_length(robot: dict[str, Any], chain: list[str]) -> float | None:
    """Estimate local kinematic chain path length from explicit body offsets.

    This is not a world-frame leg length. It is a conservative geometric
    reference because the extractor intentionally does not compile MJCF.
    """

    total = 0.0
    for name in chain:
        body = body_by_name(robot, name)
        if body is None:
            return None
        norm = vector_norm(body.get("pos"))
        if norm is None:
            return None
        total += norm
    return round(total, 9)


def average_numeric(values: list[float | None]) -> float | None:
    """Average available numeric values."""

    numeric = [value for value in values if isinstance(value, (int, float))]
    if not numeric:
        return None
    return round(sum(float(value) for value in numeric) / len(numeric), 9)


def foot_box_full_extents(robot: dict[str, Any]) -> dict[str, Any]:
    """Extract explicit foot box full extents when MuJoCo box sizes are present."""

    extents: list[list[float]] = []
    for geom in robot.get("geoms", []):
        if not geom.get("foot_related") or geom.get("type") != "box":
            continue
        size = geom.get("size")
        if isinstance(size, list) and len(size) == 3 and all(isinstance(value, (int, float)) for value in size):
            extents.append([round(float(value) * 2.0, 9) for value in size])
    if not extents:
        return {
            "value": None,
            "source": "No explicit foot-related box geom size in extracted XML.",
            "confidence": MANUAL_REQUIRED,
        }
    unique_extents = []
    for extent in extents:
        if extent not in unique_extents:
            unique_extents.append(extent)
    return {
        "value": unique_extents,
        "source": "Explicit MuJoCo box geom size fields; box size is half-extent, output is full extent.",
        "confidence": "high",
    }


def foot_geom_position_bbox(robot: dict[str, Any]) -> dict[str, Any]:
    """Compute a low-confidence local-position bbox for foot-related geoms."""

    positions = []
    for geom in robot.get("geoms", []):
        if not geom.get("foot_related"):
            continue
        pos = geom.get("pos")
        if isinstance(pos, list) and len(pos) == 3 and all(isinstance(value, (int, float)) for value in pos):
            positions.append([float(value) for value in pos])
    if not positions:
        return {
            "value": None,
            "source": "No explicit foot-related geom local positions available.",
            "confidence": MANUAL_REQUIRED,
        }
    mins = [min(position[index] for position in positions) for index in range(3)]
    maxs = [max(position[index] for position in positions) for index in range(3)]
    span = [round(maxs[index] - mins[index], 9) for index in range(3)]
    return {
        "value": {"min": [round(value, 9) for value in mins], "max": [round(value, 9) for value in maxs], "span": span},
        "source": "Foot-related geom local positions selected by name heuristic; not physical foot size.",
        "confidence": "low",
    }


def scalar_field(
    name: str,
    sedon_value: float | int | None,
    duck_value: float | int | None,
    unit: str,
    source: str,
    confidence: str,
    note: str,
) -> dict[str, Any]:
    """Build a scalar scale field record."""

    return {
        "field": name,
        "unit": unit,
        "sedon_value": sedon_value,
        "duck_value": duck_value,
        "sedon_to_duck_ratio": ratio(sedon_value, duck_value),
        "source": source,
        "confidence": confidence,
        "notes": note,
    }


def build_scale_mapping(sedon: dict[str, Any], duck: dict[str, Any], sedon_path: Path, duck_path: Path) -> dict[str, Any]:
    """Build the scale mapping payload from extracted parameter snapshots."""

    sedon_mass = sedon.get("summary", {}).get("total_explicit_body_mass")
    duck_mass = duck.get("summary", {}).get("total_explicit_body_mass")
    sedon_counts = sedon.get("counts", {})
    duck_counts = duck.get("counts", {})

    sedon_left_chain = [
        "L_link_hip_yaw",
        "L_link_hip_roll",
        "L_link_hip_pitch",
        "L_link_knee_pitch",
        "L_link_ankle_pitch",
    ]
    sedon_right_chain = [
        "R_link_hip_yaw",
        "R_link_hip_roll",
        "R_link_hip_pitch",
        "R_link_knee_pitch",
        "R_link_ankle_pitch",
    ]
    duck_left_chain = [
        "hip_roll_assembly",
        "left_roll_to_pitch_assembly",
        "knee_and_ankle_assembly",
        "knee_and_ankle_assembly_2",
        "foot_assembly",
    ]
    duck_right_chain = [
        "hip_roll_assembly_2",
        "right_roll_to_pitch_assembly",
        "knee_and_ankle_assembly_3",
        "knee_and_ankle_assembly_4",
        "foot_assembly_2",
    ]

    sedon_chain_lengths = {
        "left": chain_path_length(sedon, sedon_left_chain),
        "right": chain_path_length(sedon, sedon_right_chain),
    }
    duck_chain_lengths = {
        "left": chain_path_length(duck, duck_left_chain),
        "right": chain_path_length(duck, duck_right_chain),
    }
    sedon_avg_chain = average_numeric(list(sedon_chain_lengths.values()))
    duck_avg_chain = average_numeric(list(duck_chain_lengths.values()))

    scale_fields = [
        scalar_field(
            "explicit_body_mass",
            sedon_mass,
            duck_mass,
            "kg",
            "summary.total_explicit_body_mass from extracted XML inertial fields.",
            "high",
            "Explicit inertial masses only; compiled/default masses are not inferred.",
        ),
        scalar_field(
            "initial_base_height_z",
            first_body_pos_z(sedon),
            first_body_pos_z(duck),
            "m",
            "First extracted world body pos.z.",
            "medium",
            "Initial XML placement reference, not a measured standing height.",
        ),
        scalar_field(
            "hip_lateral_spacing",
            lateral_spacing(sedon, "L_link_hip_yaw", "R_link_hip_yaw"),
            lateral_spacing(duck, "hip_roll_assembly", "hip_roll_assembly_2"),
            "m",
            "Explicit left/right hip body local pos.y spacing.",
            "medium",
            "Useful as a morphology reference; not a full pelvis width measurement.",
        ),
        scalar_field(
            "local_leg_chain_path_length_average",
            sedon_avg_chain,
            duck_avg_chain,
            "m",
            "Sum of explicit local body offset norms along named leg chains.",
            "low",
            "Not world-frame leg length; requires compiled kinematic validation.",
        ),
        scalar_field(
            "body_count",
            sedon_counts.get("bodies"),
            duck_counts.get("bodies"),
            "count",
            "counts.bodies from extracted XML.",
            "high",
            "Structural count, not a physical scale.",
        ),
        scalar_field(
            "foot_related_geom_count",
            sedon_counts.get("foot_related_geoms"),
            duck_counts.get("foot_related_geoms"),
            "count",
            "counts.foot_related_geoms from name heuristic.",
            "low",
            "Name heuristic count; contact semantics require simulation validation.",
        ),
    ]

    sedon_foot_box = foot_box_full_extents(sedon)
    duck_foot_box = foot_box_full_extents(duck)
    sedon_foot_bbox = foot_geom_position_bbox(sedon)
    duck_foot_bbox = foot_geom_position_bbox(duck)

    return {
        "schema_version": 1,
        "tool_version": TOOL_VERSION,
        "generated_at": utc_now_iso(),
        "purpose": "Read-only normalized scale references for Sedon/Duck gait mapping preparation.",
        "sources": {
            "sedon_parameters": str(sedon_path),
            "duck_parameters": str(duck_path),
        },
        "summary": {
            "readiness": "scale_reference_only",
            "direct_parameter_transfer_allowed": False,
            "primary_ratio": {
                "field": "explicit_body_mass",
                "sedon_to_duck_ratio": ratio(sedon_mass, duck_mass),
                "confidence": "high",
            },
            "usable_for": [
                "normalization discussion",
                "task-space reference scaling preparation",
                "manual review checklist",
            ],
            "not_usable_for": [
                "direct Duck parameter transfer",
                "direct joint trajectory transfer",
                "PPO reward or policy settings",
            ],
        },
        "scale_fields": scale_fields,
        "shape_fields": {
            "sedon_foot_box_full_extents_m": sedon_foot_box,
            "duck_foot_box_full_extents_m": duck_foot_box,
            "sedon_foot_geom_local_position_bbox_m": sedon_foot_bbox,
            "duck_foot_geom_local_position_bbox_m": duck_foot_bbox,
            "local_leg_chain_path_lengths_m": {
                "sedon": sedon_chain_lengths,
                "duck": duck_chain_lengths,
                "source": "Explicit body local pos chain norm sums.",
                "confidence": "low",
            },
        },
        "manual_required": [
            {
                "field": "duck_foot_physical_extents",
                "reason": "Duck foot geoms are mesh based and extracted XML does not include mesh dimensions.",
            },
            {
                "field": "world_frame_leg_length",
                "reason": "XML-only extraction does not compile transforms or account for joint orientations.",
            },
            {
                "field": "world_frame_foot_contact_patch",
                "reason": "Foot-related geoms were selected by name heuristic; contact behavior is not validated.",
            },
            {
                "field": "dynamic_gait_scale",
                "reason": "Duck gait period, clearance, and reference motion are not present in robot XML.",
            },
        ],
        "safety_notes": [
            "All ratios are references only.",
            "Do not apply Duck kp, forcerange, or joint trajectories to Sedon.",
            "Unknown fields are retained instead of inferred from mesh names.",
        ],
    }


def format_value(value: Any) -> str:
    """Format a table value for Markdown."""

    if value is None:
        return "`unknown`"
    if isinstance(value, float):
        return f"`{value:.6g}`"
    return f"`{value}`"


def write_report(path: Path, mapping: dict[str, Any]) -> None:
    """Write the Markdown scale table report."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Sedon Duck Normalized Scale Table",
        "",
        "Generated by `tools/sedon/extractors/estimate_sedon_duck_scale_fields.py`.",
        "",
        "## Summary",
        "",
        "- Status: `REFERENCE_ONLY`",
        "- Direct Duck parameter transfer allowed: `false`",
        f"- Primary explicit mass ratio Sedon/Duck: `{mapping['summary']['primary_ratio']['sedon_to_duck_ratio']}`",
        "- This table is for normalization planning and manual diagnostics only.",
        "",
        "## Sources",
        "",
        f"- Sedon parameters: `{mapping['sources']['sedon_parameters']}`",
        f"- Duck parameters: `{mapping['sources']['duck_parameters']}`",
        f"- Generated at: `{mapping['generated_at']}`",
        "",
        "## Scale Fields",
        "",
        "| Field | Sedon | Duck | Sedon/Duck | Unit | Confidence | Source | Notes |",
        "|---|---:|---:|---:|---|---|---|---|",
    ]
    for field in mapping["scale_fields"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{field['field']}`",
                    format_value(field["sedon_value"]),
                    format_value(field["duck_value"]),
                    format_value(field["sedon_to_duck_ratio"]),
                    f"`{field['unit']}`",
                    f"`{field['confidence']}`",
                    field["source"],
                    field["notes"],
                ]
            )
            + " |"
        )

    shape = mapping["shape_fields"]
    lines.extend(
        [
            "",
            "## Shape Fields",
            "",
            "| Field | Value | Confidence | Source |",
            "|---|---|---|---|",
            f"| `sedon_foot_box_full_extents_m` | {format_value(shape['sedon_foot_box_full_extents_m']['value'])} | `{shape['sedon_foot_box_full_extents_m']['confidence']}` | {shape['sedon_foot_box_full_extents_m']['source']} |",
            f"| `duck_foot_box_full_extents_m` | {format_value(shape['duck_foot_box_full_extents_m']['value'])} | `{shape['duck_foot_box_full_extents_m']['confidence']}` | {shape['duck_foot_box_full_extents_m']['source']} |",
            f"| `sedon_foot_geom_local_position_bbox_m` | {format_value(shape['sedon_foot_geom_local_position_bbox_m']['value'])} | `{shape['sedon_foot_geom_local_position_bbox_m']['confidence']}` | {shape['sedon_foot_geom_local_position_bbox_m']['source']} |",
            f"| `duck_foot_geom_local_position_bbox_m` | {format_value(shape['duck_foot_geom_local_position_bbox_m']['value'])} | `{shape['duck_foot_geom_local_position_bbox_m']['confidence']}` | {shape['duck_foot_geom_local_position_bbox_m']['source']} |",
            f"| `local_leg_chain_path_lengths_m` | {format_value(shape['local_leg_chain_path_lengths_m'])} | `low` | Explicit body local pos chain norm sums. |",
            "",
            "## Manual Required / Unknown",
            "",
        ]
    )
    for item in mapping["manual_required"]:
        lines.append(f"- `{item['field']}`: {item['reason']}")

    lines.extend(
        [
            "",
            "## Safety Notes",
            "",
        ]
    )
    for note in mapping["safety_notes"]:
        lines.append(f"- {note}")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sedon-yaml", type=Path, default=DEFAULT_SEDON_YAML)
    parser.add_argument("--duck-yaml", type=Path, default=DEFAULT_DUCK_YAML)
    parser.add_argument("--output-yaml", type=Path, default=DEFAULT_OUTPUT_YAML)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> int:
    """Run the scale field estimator."""

    args = parse_args()
    try:
        sedon = load_json_yaml(args.sedon_yaml)
        duck = load_json_yaml(args.duck_yaml)
        mapping = build_scale_mapping(sedon, duck, args.sedon_yaml, args.duck_yaml)
        write_json_yaml(args.output_yaml, mapping)
        write_report(args.report, mapping)
    except ValueError as exc:
        print(f"Scale estimation failed: {exc}")
        return 1

    print(f"Wrote {args.output_yaml}")
    print(f"Wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
