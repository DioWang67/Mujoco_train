"""Extract Open Duck Mini foot contact geometry reference.

This Class C extractor reads local Duck MJCF XML only. It reports foot-related
geoms and conservative normalized reference fields without treating mesh
geometry as directly measurable contact-patch dimensions.
"""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DUCK_XML = (
    REPO_ROOT
    / "references"
    / "open_duck_mini"
    / "source"
    / "playground_open_duck_mini_v2"
    / "xmls"
    / "open_duck_mini_v2.xml"
)
DEFAULT_OUTPUT = REPO_ROOT / "references" / "open_duck_mini" / "duck_foot_contact_geometry_reference.yaml"
DEFAULT_REPORT = REPO_ROOT / "docs" / "open_duck_foot_contact_geometry_reference.md"
DEFAULT_SEEDON_TUNING = REPO_ROOT / "configs" / "seedon" / "seedon_v5_22_duck_guided_foot_tuning.yaml"
DEFAULT_SEEDON_PLAN = REPO_ROOT / "docs" / "seedon_v5_22_duck_guided_foot_tuning_plan.md"

FOOT_TOKENS = ("foot", "toe", "heel", "sole", "ankle")
SIDE_PATTERNS = {
    "left": re.compile(r"(^|[_-])left([_-]|$)|(^|[_-])l([_-]|$)", re.IGNORECASE),
    "right": re.compile(r"(^|[_-])right([_-]|$)|(^|[_-])r([_-]|$)", re.IGNORECASE),
}
CATEGORY_TOKENS = (
    ("toe", ("toe",)),
    ("heel", ("heel",)),
    ("sole", ("sole", "bottom", "tpu")),
    ("inner", ("inner", "inside", "medial")),
    ("outer", ("outer", "outside", "lateral")),
    ("foot", ("foot",)),
    ("center", ("center", "middle")),
)


@dataclass(frozen=True)
class BodyContext:
    """Current XML body traversal context."""

    name: str
    parent: str
    foot_related_ancestor: bool


def rel_path(path: Path) -> str:
    """Return repository-relative path when possible."""

    try:
        return str(path.resolve().relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def parse_vec(raw_value: str | None) -> list[float] | None:
    """Parse an MJCF numeric vector field."""

    if raw_value is None:
        return None
    parts = raw_value.replace(",", " ").split()
    if not parts:
        return None
    return [float(part) for part in parts]


def resolve_duck_xml(preferred_path: Path) -> Path:
    """Resolve Duck XML from preferred path, manifest, or source tree."""

    if preferred_path.is_file():
        return preferred_path
    manifest = REPO_ROOT / "references" / "open_duck_mini" / "source_manifest.yaml"
    if manifest.is_file():
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            raw_path = payload.get("duck_xml_source_path")
            if isinstance(raw_path, str):
                candidate = REPO_ROOT / raw_path
                if candidate.is_file():
                    return candidate
        except json.JSONDecodeError:
            pass
    source_root = REPO_ROOT / "references" / "open_duck_mini" / "source"
    candidates = sorted(source_root.rglob("*duck*.xml")) if source_root.is_dir() else []
    if candidates:
        return candidates[0]
    raise FileNotFoundError("Open Duck Mini XML not found under references/open_duck_mini/source.")


def contains_foot_token(value: str | None) -> bool:
    """Return whether a string is foot-related by name heuristic."""

    if not value:
        return False
    lowered = value.lower()
    return any(token in lowered for token in FOOT_TOKENS)


def classify_side(*values: str | None) -> str:
    """Classify side from names."""

    text = " ".join(value or "" for value in values)
    for side, pattern in SIDE_PATTERNS.items():
        if pattern.search(text):
            return side
    return "unknown"


def classify_category(*values: str | None) -> str:
    """Classify foot geom category from name/body/mesh text."""

    text = " ".join(value or "" for value in values).lower()
    for category, tokens in CATEGORY_TOKENS:
        if any(token in text for token in tokens):
            return category
    return "unknown"


def inherited_default_class(body: ET.Element, geom: ET.Element) -> str | None:
    """Return visible class/default class for one geom."""

    return geom.attrib.get("class") or body.attrib.get("childclass")


def is_active_contact_geom(row: dict[str, Any]) -> bool | str:
    """Classify whether a geom is visibly active for contact."""

    contype = row["contype"]
    conaffinity = row["conaffinity"]
    geom_class = row["class_or_default"]
    if contype == "0" or conaffinity == "0" or geom_class == "visual":
        return False
    if geom_class == "collision":
        return True
    return "unknown"


def collect_geoms(root: ET.Element) -> list[dict[str, Any]]:
    """Collect all foot-related geom rows."""

    rows: list[dict[str, Any]] = []

    def walk_body(body: ET.Element, parent: str, ancestor_foot_related: bool, ancestor_side: str) -> None:
        body_name = body.attrib.get("name", "unknown_body")
        body_foot_related = ancestor_foot_related or contains_foot_token(body_name)
        branch_side = classify_branch_side(body_name, parent, ancestor_side)
        for geom in body.findall("geom"):
            name = geom.attrib.get("name")
            mesh = geom.attrib.get("mesh")
            category = classify_category(name, body_name, mesh)
            foot_related = body_foot_related or contains_foot_token(name) or contains_foot_token(mesh)
            if not foot_related:
                continue
            row = {
                "name": name or f"<unnamed:{body_name}:{len(rows)}>",
                "parent_body": body_name,
                "parent_body_parent": parent,
                "side": classify_geom_side(branch_side, name, body_name, mesh),
                "category": category,
                "type": geom.attrib.get("type", "unknown"),
                "pos": parse_vec(geom.attrib.get("pos")),
                "size": parse_vec(geom.attrib.get("size")),
                "friction": geom.attrib.get("friction"),
                "contype": geom.attrib.get("contype", "unknown"),
                "conaffinity": geom.attrib.get("conaffinity", "unknown"),
                "class_or_default": inherited_default_class(body, geom),
                "mesh": mesh,
                "source_type": "explicit_xml",
                "category_source_type": "estimated_from_name_heuristic",
                "size_source_type": "explicit_xml" if geom.attrib.get("size") else "unknown",
                "active_contact_visible": "unknown",
            }
            row["active_contact_visible"] = is_active_contact_geom(row)
            rows.append(row)
        for child in body.findall("body"):
            walk_body(child, body_name, body_foot_related, branch_side)

    worldbody = root.find("worldbody")
    if worldbody is None:
        return rows
    for body in worldbody.findall("body"):
        walk_body(body, "worldbody", False, "unknown")
    return rows


def classify_branch_side(body_name: str, parent_name: str, ancestor_side: str) -> str:
    """Classify robot side from body branch names."""

    side = classify_side(body_name, parent_name)
    return side if side != "unknown" else ancestor_side


def classify_geom_side(branch_side: str, *values: str | None) -> str:
    """Classify geom side, preferring robot branch side over part-side names."""

    if branch_side != "unknown":
        return branch_side
    return classify_side(*values)


def contact_candidate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return rows that visibly look like foot contact candidates."""

    candidates: list[dict[str, Any]] = []
    for row in rows:
        if row["active_contact_visible"] is True:
            candidates.append(row)
    return candidates


def normalized_reference(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build normalized Duck foot contact reference fields."""

    candidates = contact_candidate_rows(rows)
    by_category: dict[str, list[dict[str, Any]]] = {}
    for row in candidates:
        by_category.setdefault(str(row["category"]), []).append(row)

    explicit_patch_categories = {category for category in ("toe", "center", "heel") if by_category.get(category)}
    if len(explicit_patch_categories) < 3:
        return {
            "status": "PARTIAL_REFERENCE",
            "foot_length_estimate": {
                "value": None,
                "source_type": "manual_required",
                "confidence": "low",
                "reason": "Duck XML exposes mesh foot collision geoms but no primitive size/pos toe-center-heel patches.",
            },
            "toe_x_ratio": unknown_ratio("toe patch is not explicit in XML."),
            "center_x_ratio": unknown_ratio("center patch is not explicit in XML."),
            "heel_x_ratio": unknown_ratio("heel patch is not explicit in XML."),
            "toe_z_relative_to_center": unknown_ratio("toe and center z offsets are not explicit primitive patch fields."),
            "heel_z_relative_to_center": unknown_ratio("heel and center z offsets are not explicit primitive patch fields."),
            "toe_patch_size_ratio": unknown_ratio("mesh dimensions are not treated as measurable patch size."),
            "heel_patch_size_ratio": unknown_ratio("mesh dimensions are not treated as measurable patch size."),
            "inner_outer_width_ratio": unknown_ratio("inner/outer patches are not explicit in XML."),
            "reference_basis": {
                "source_type": "explicit_xml",
                "confidence": "medium",
                "active_contact_candidate_count": len(candidates),
                "active_contact_candidate_names": [row["name"] for row in candidates],
                "note": "Active candidates are visible XML collision geoms, not decomposed toe/center/heel contact patches.",
            },
        }
    return estimated_ratios_from_primitives(by_category)


def unknown_ratio(reason: str) -> dict[str, Any]:
    """Return a normalized-reference unknown field."""

    return {
        "value": None,
        "source_type": "manual_required",
        "confidence": "low",
        "reason": reason,
    }


def estimated_ratios_from_primitives(by_category: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Estimate normalized ratios when explicit primitive patches exist."""

    centers = by_category.get("center", [])
    toes = by_category.get("toe", [])
    heels = by_category.get("heel", [])
    if not centers or not toes or not heels:
        raise ValueError("Explicit primitive ratio estimation requires toe/center/heel rows.")
    center_pos = average_pos(centers)
    toe_pos = average_pos(toes)
    heel_pos = average_pos(heels)
    length = abs(toe_pos[0] - heel_pos[0])
    if length <= 1e-12:
        return normalized_reference([])
    return {
        "status": "REFERENCE_ESTIMATED_FROM_EXPLICIT_PRIMITIVES",
        "foot_length_estimate": {
            "value": length,
            "source_type": "estimated_from_geom_positions",
            "confidence": "medium",
        },
        "toe_x_ratio": ratio_field((toe_pos[0] - center_pos[0]) / length),
        "center_x_ratio": ratio_field(0.0),
        "heel_x_ratio": ratio_field((heel_pos[0] - center_pos[0]) / length),
        "toe_z_relative_to_center": ratio_field(toe_pos[2] - center_pos[2]),
        "heel_z_relative_to_center": ratio_field(heel_pos[2] - center_pos[2]),
        "toe_patch_size_ratio": unknown_ratio("primitive patch sizes were not available for every toe patch."),
        "heel_patch_size_ratio": unknown_ratio("primitive patch sizes were not available for every heel patch."),
        "inner_outer_width_ratio": unknown_ratio("inner/outer primitive patches were not available."),
        "reference_basis": {
            "source_type": "estimated_from_geom_positions",
            "confidence": "medium",
        },
    }


def ratio_field(value: float) -> dict[str, Any]:
    """Return an estimated ratio field."""

    return {
        "value": float(value),
        "source_type": "estimated_from_geom_positions",
        "confidence": "medium",
    }


def average_pos(rows: list[dict[str, Any]]) -> list[float]:
    """Average explicit pos values."""

    values = [row["pos"] for row in rows if row.get("pos") and len(row["pos"]) >= 3]
    if not values:
        raise ValueError("Cannot average rows without explicit pos.")
    return [sum(value[index] for value in values) / len(values) for index in range(3)]


def build_payload(xml_path: Path) -> dict[str, Any]:
    """Build the full reference payload."""

    root = ET.parse(xml_path).getroot()
    rows = collect_geoms(root)
    active = contact_candidate_rows(rows)
    return {
        "schema_version": 1,
        "source_xml": rel_path(xml_path),
        "source_type": "explicit_xml",
        "safety": {
            "network_used": False,
            "repo_refetched": False,
            "duck_result_directly_applied_to_seedon": False,
            "mesh_dimensions_treated_as_measurable": False,
        },
        "summary": {
            "foot_related_geom_count": len(rows),
            "visible_active_contact_candidate_count": len(active),
            "visible_active_contact_candidate_names": [row["name"] for row in active],
            "normalized_reference_status": "PARTIAL_REFERENCE",
        },
        "foot_related_geoms": rows,
        "normalized_geometry_reference": normalized_reference(rows),
        "notes": [
            "Foot-related geoms use name and ancestor-body heuristics.",
            "Mesh geoms are listed but their dimensions are not inferred from mesh files.",
            "Visible active contact candidates are based on explicit XML class/contype/conaffinity only.",
            "Duck reference is for Seedon prototype guidance only, not final mechanical design.",
        ],
    }


def seedon_tuning_config(reference_payload: dict[str, Any]) -> dict[str, Any]:
    """Build Seedon v5_22 Duck-guided foot tuning plan config."""

    ref = reference_payload["normalized_geometry_reference"]
    return {
        "schema_version": 1,
        "version": "v5_22",
        "source": "duck_reference_guided_prototype",
        "confidence": "low",
        "valid_for": "contact_persistence_tuning_only",
        "invalid_for": ["final_mechanical_design_claim", "walking_success_claim", "sim2real_claim"],
        "duck_reference": {
            "reference_file": rel_path(DEFAULT_OUTPUT),
            "status": ref["status"],
            "directly_applied_to_seedon": False,
            "note": "Use Duck normalized ratios only as initial centers when explicit/estimated values are available.",
        },
        "seedon_source_model": "private_assets/seedon_v5_22/training_scene.xml",
        "search_space": {
            "toe_x_ratio": search_axis(ref["toe_x_ratio"], delta=0.08, step=0.02),
            "toe_z_relative_to_center": search_axis(ref["toe_z_relative_to_center"], delta=0.006, step=0.003),
            "toe_patch_scale": search_axis(ref["toe_patch_size_ratio"], delta=0.12, step=0.04),
            "center_z_offset": {
                "center": 0.0,
                "values": [-0.004, -0.002, 0.0, 0.002, 0.004],
                "source": "assumption",
                "confidence": "low",
            },
            "heel_x_ratio": search_axis(ref["heel_x_ratio"], delta=0.06, step=0.02),
            "heel_z_relative_to_center": search_axis(ref["heel_z_relative_to_center"], delta=0.004, step=0.002),
        },
        "variant_metadata": {
            "source": "duck_reference_guided_prototype",
            "confidence": "low/medium",
            "valid_for": "contact_persistence_tuning_only",
            "invalid_for": "final_mechanical_design_claim",
        },
        "execution_policy": {
            "run_tuning_now": False,
            "requires_manual_duck_patch_review": ref["status"] == "PARTIAL_REFERENCE",
            "do_not_modify_source_xml": True,
            "do_not_run_ppo": True,
        },
    }


def search_axis(reference_field: dict[str, Any], *, delta: float, step: float) -> dict[str, Any]:
    """Build one Seedon tuning search axis."""

    value = reference_field.get("value")
    if value is None:
        return {
            "center": None,
            "values": [],
            "source": "duck_reference_guided_prototype",
            "confidence": "low",
            "status": "manual_required",
            "reason": reference_field.get("reason", "Duck reference value unavailable."),
            "planned_delta_if_available": delta,
            "planned_step_if_available": step,
        }
    center = float(value)
    count = int(round(delta / step))
    values = [round(center + index * step, 9) for index in range(-count, count + 1)]
    return {
        "center": center,
        "values": values,
        "source": "duck_reference_guided_prototype",
        "confidence": reference_field.get("confidence", "low"),
        "status": "READY_FOR_CONTACT_PERSISTENCE_TUNING",
    }


def write_json(path: Path, payload: Any) -> None:
    """Write JSON-compatible YAML payload."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_reference_report(path: Path, payload: dict[str, Any]) -> None:
    """Write Duck reference Markdown report."""

    path.parent.mkdir(parents=True, exist_ok=True)
    summary = payload["summary"]
    ref = payload["normalized_geometry_reference"]
    lines = [
        "# Open Duck Foot Contact Geometry Reference",
        "",
        "Task class: Class C local XML reference extraction. No network was used and Duck geometry is not directly applied to Seedon.",
        "",
        "## Summary",
        "",
        f"- Source XML: `{payload['source_xml']}`",
        f"- Foot-related geom count: `{summary['foot_related_geom_count']}`",
        f"- Visible active contact candidates: `{summary['visible_active_contact_candidate_count']}`",
        f"- Candidate names: `{summary['visible_active_contact_candidate_names']}`",
        f"- Normalized reference status: `{ref['status']}`",
        "",
        "## Normalized Geometry Reference",
        "",
        "| field | value | source type | confidence |",
        "|---|---:|---|---|",
    ]
    for field in (
        "foot_length_estimate",
        "toe_x_ratio",
        "center_x_ratio",
        "heel_x_ratio",
        "toe_z_relative_to_center",
        "heel_z_relative_to_center",
        "toe_patch_size_ratio",
        "heel_patch_size_ratio",
        "inner_outer_width_ratio",
    ):
        item = ref[field]
        lines.append(
            f"| `{field}` | `{item.get('value')}` | `{item.get('source_type')}` | `{item.get('confidence')}` |"
        )
    lines.extend(["", "## Foot-Related Geoms", ""])
    lines.extend(["| name | parent body | side | category | type | pos | size | class/default | contype | conaffinity | active? |", "|---|---|---|---|---|---|---|---|---|---|---|"])
    for row in payload["foot_related_geoms"]:
        lines.append(
            f"| `{row['name']}` | `{row['parent_body']}` | `{row['side']}` | `{row['category']}` | "
            f"`{row['type']}` | `{row['pos']}` | `{row['size']}` | `{row['class_or_default']}` | "
            f"`{row['contype']}` | `{row['conaffinity']}` | `{row['active_contact_visible']}` |"
        )
    lines.extend(["", "## Limitations", ""])
    for note in payload["notes"]:
        lines.append(f"- {note}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_seedon_plan(path: Path, tuning: dict[str, Any]) -> None:
    """Write Seedon Duck-guided tuning plan Markdown."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Seedon v5_22 Duck-Guided Foot Tuning Plan",
        "",
        "Task class: Class C planning artifact. This plan does not run tuning, modify source XML, run PPO, or claim walking success.",
        "",
        "## Summary",
        "",
        f"- Source: `{tuning['source']}`",
        f"- Confidence: `{tuning['confidence']}`",
        f"- Valid for: `{tuning['valid_for']}`",
        f"- Invalid for: `{tuning['invalid_for']}`",
        f"- Duck reference status: `{tuning['duck_reference']['status']}`",
        "",
        "## Search Space",
        "",
        "| axis | center | values | status |",
        "|---|---:|---|---|",
    ]
    for axis, item in tuning["search_space"].items():
        lines.append(f"| `{axis}` | `{item.get('center')}` | `{item.get('values')}` | `{item.get('status', 'READY')}` |")
    lines.extend(["", "## Variant Metadata", ""])
    for key, value in tuning["variant_metadata"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Execution Policy", ""])
    for key, value in tuning["execution_policy"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Recommendation", ""])
    if tuning["execution_policy"]["requires_manual_duck_patch_review"]:
        lines.append("- Duck XML does not expose explicit toe/center/heel primitive patches, so manual Duck foot contact review is required before running Duck-guided Seedon tuning.")
    else:
        lines.append("- Use the Duck normalized centers as initial Seedon contact persistence tuning centers, with small bounded sweeps only.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duck-xml", type=Path, default=DEFAULT_DUCK_XML)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--seedon-tuning", type=Path, default=DEFAULT_SEEDON_TUNING)
    parser.add_argument("--seedon-plan", type=Path, default=DEFAULT_SEEDON_PLAN)
    return parser.parse_args()


def main() -> int:
    """Run Duck foot contact geometry extraction."""

    args = parse_args()
    duck_xml = resolve_duck_xml(args.duck_xml)
    payload = build_payload(duck_xml)
    tuning = seedon_tuning_config(payload)
    write_json(args.output, payload)
    write_reference_report(args.report, payload)
    write_json(args.seedon_tuning, tuning)
    write_seedon_plan(args.seedon_plan, tuning)
    print(f"duck_xml={duck_xml}")
    print(f"foot_related_geom_count={payload['summary']['foot_related_geom_count']}")
    print(f"active_contact_candidate_count={payload['summary']['visible_active_contact_candidate_count']}")
    print(f"reference_status={payload['normalized_geometry_reference']['status']}")
    print(f"output={args.output}")
    print(f"report={args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
