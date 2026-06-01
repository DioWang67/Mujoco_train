"""Shared MJCF parameter extraction helpers.

The helpers intentionally parse the XML source directly instead of compiling a
MuJoCo model. That keeps this inventory-style extractor simple and preserves
which fields were explicitly present in the MJCF versus inferred by MuJoCo
defaults.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TOOL_VERSION = "sedon-parameter-extractor-v1"
UNKNOWN = "unknown"
FOOT_NAME_TOKENS = ("foot", "toe", "heel", "sole", "ankle", "bottom")


class ExtractionError(RuntimeError):
    """Raised when an MJCF source cannot be extracted cleanly."""


def utc_now_iso() -> str:
    """Return a stable UTC timestamp string.

    Returns:
        ISO-8601 UTC timestamp.
    """

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_numeric_list(raw: str | None) -> list[float] | None:
    """Parse a MuJoCo numeric attribute.

    Args:
        raw: Attribute string from XML.

    Returns:
        List of floats, or ``None`` when the source field is absent or not
        numeric.
    """

    if raw is None:
        return None
    try:
        return [float(part) for part in raw.split()]
    except ValueError:
        return None


def scalar_or_list(raw: str | None) -> float | list[float] | str | None:
    """Parse a numeric MuJoCo field while preserving unknown strings.

    Args:
        raw: Attribute string from XML.

    Returns:
        Float, list of floats, raw string, or ``None`` for absent fields.
    """

    if raw is None:
        return None
    values = parse_numeric_list(raw)
    if values is None:
        return raw
    if len(values) == 1:
        return values[0]
    return values


def attr(element: ET.Element, name: str) -> Any:
    """Return a parsed XML attribute or ``None`` when absent."""

    return scalar_or_list(element.attrib.get(name))


def is_foot_related(name: str | None) -> bool:
    """Return whether a name looks foot-related by a conservative heuristic."""

    if not name:
        return False
    lowered = name.lower()
    return any(token in lowered for token in FOOT_NAME_TOKENS)


def side_from_name(name: str | None) -> str:
    """Infer left/right side from a conventional robot link or geom name."""

    if not name:
        return UNKNOWN
    lowered = name.lower()
    if lowered.startswith(("l_", "left_")) or "_l_" in lowered or "left" in lowered:
        return "left"
    if lowered.startswith(("r_", "right_")) or "_r_" in lowered or "right" in lowered:
        return "right"
    return UNKNOWN


def load_xml(path: Path) -> ET.Element:
    """Load an MJCF XML root.

    Args:
        path: MJCF XML path.

    Returns:
        Root XML element.

    Raises:
        ExtractionError: If the path is missing or XML cannot be parsed.
    """

    if not path.is_file():
        raise ExtractionError(f"MJCF XML not found: {path}")
    try:
        return ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise ExtractionError(f"Failed to parse MJCF XML {path}: {exc}") from exc


def body_record(body: ET.Element, parent: str | None, depth: int) -> dict[str, Any]:
    """Build a body record from an MJCF body element."""

    name = body.attrib.get("name") or f"unnamed_body_depth_{depth}"
    inertial = body.find("inertial")
    return {
        "name": name,
        "parent": parent,
        "depth": depth,
        "pos": attr(body, "pos"),
        "quat": attr(body, "quat"),
        "euler": attr(body, "euler"),
        "mocap": body.attrib.get("mocap"),
        "inertial": {
            "pos": attr(inertial, "pos") if inertial is not None else None,
            "mass": attr(inertial, "mass") if inertial is not None else None,
            "diaginertia": attr(inertial, "diaginertia") if inertial is not None else None,
            "fullinertia": attr(inertial, "fullinertia") if inertial is not None else None,
            "quat": attr(inertial, "quat") if inertial is not None else None,
        },
        "foot_related": is_foot_related(name),
        "side": side_from_name(name),
    }


def joint_record(joint: ET.Element, body_name: str | None, joint_type: str) -> dict[str, Any]:
    """Build a joint or freejoint record."""

    name = joint.attrib.get("name") or f"unnamed_{joint_type}"
    return {
        "name": name,
        "body": body_name,
        "kind": joint_type,
        "type": joint.attrib.get("type", "hinge" if joint_type == "joint" else joint_type),
        "axis": attr(joint, "axis"),
        "range": attr(joint, "range"),
        "pos": attr(joint, "pos"),
        "limited": joint.attrib.get("limited"),
        "actuatorfrcrange": attr(joint, "actuatorfrcrange"),
        "damping": attr(joint, "damping"),
        "stiffness": attr(joint, "stiffness"),
        "armature": attr(joint, "armature"),
        "foot_related": is_foot_related(name) or is_foot_related(body_name),
        "side": side_from_name(name) if side_from_name(name) != UNKNOWN else side_from_name(body_name),
    }


def geom_record(geom: ET.Element, body_name: str | None) -> dict[str, Any]:
    """Build a geom record."""

    name = geom.attrib.get("name") or geom.attrib.get("mesh") or "unnamed_geom"
    return {
        "name": name,
        "body": body_name,
        "type": geom.attrib.get("type"),
        "mesh": geom.attrib.get("mesh"),
        "pos": attr(geom, "pos"),
        "quat": attr(geom, "quat"),
        "size": attr(geom, "size"),
        "mass": attr(geom, "mass"),
        "density": attr(geom, "density"),
        "friction": attr(geom, "friction"),
        "contype": attr(geom, "contype"),
        "conaffinity": attr(geom, "conaffinity"),
        "group": attr(geom, "group"),
        "rgba": attr(geom, "rgba"),
        "foot_related": is_foot_related(name) or is_foot_related(body_name),
        "side": side_from_name(name) if side_from_name(name) != UNKNOWN else side_from_name(body_name),
    }


def default_actuator_fields(root: ET.Element) -> dict[str, dict[str, dict[str, Any]]]:
    """Collect explicit actuator defaults by MuJoCo default class.

    Args:
        root: MJCF root.

    Returns:
        Mapping of class name to actuator tag to parsed default attributes.
    """

    defaults: dict[str, dict[str, dict[str, Any]]] = {}
    actuator_tags = {"motor", "position", "velocity", "intvelocity", "damper", "cylinder", "muscle", "adhesion"}
    for default in root.iter("default"):
        class_name = default.attrib.get("class")
        if not class_name:
            continue
        class_defaults = defaults.setdefault(class_name, {})
        for child in list(default):
            if child.tag not in actuator_tags:
                continue
            class_defaults[child.tag] = {key: scalar_or_list(value) for key, value in child.attrib.items()}
    return defaults


def inherited_actuator_value(
    actuator: ET.Element,
    field: str,
    defaults: dict[str, dict[str, dict[str, Any]]],
) -> Any:
    """Return an actuator field from explicit XML or default class XML."""

    explicit = attr(actuator, field)
    if explicit is not None:
        return explicit
    class_name = actuator.attrib.get("class")
    if not class_name:
        return None
    return defaults.get(class_name, {}).get(actuator.tag, {}).get(field)


def actuator_record(actuator: ET.Element, defaults: dict[str, dict[str, dict[str, Any]]] | None = None) -> dict[str, Any]:
    """Build an actuator record."""

    default_values = defaults or {}
    name = actuator.attrib.get("name") or f"unnamed_{actuator.tag}"
    class_name = actuator.attrib.get("class")
    return {
        "name": name,
        "kind": actuator.tag,
        "class": class_name,
        "joint": actuator.attrib.get("joint"),
        "gear": inherited_actuator_value(actuator, "gear", default_values),
        "kp": inherited_actuator_value(actuator, "kp", default_values),
        "kv": inherited_actuator_value(actuator, "kv", default_values),
        "ctrlrange": inherited_actuator_value(actuator, "ctrlrange", default_values),
        "forcerange": inherited_actuator_value(actuator, "forcerange", default_values),
        "ctrllimited": actuator.attrib.get("ctrllimited"),
        "forcelimited": actuator.attrib.get("forcelimited"),
        "biastype": actuator.attrib.get("biastype"),
        "gaintype": actuator.attrib.get("gaintype"),
        "dynprm": inherited_actuator_value(actuator, "dynprm", default_values),
        "gainprm": inherited_actuator_value(actuator, "gainprm", default_values),
        "biasprm": inherited_actuator_value(actuator, "biasprm", default_values),
        "default_source": f"class:{class_name}" if class_name and actuator.tag in default_values.get(class_name, {}) else None,
        "foot_related": is_foot_related(name) or is_foot_related(actuator.attrib.get("joint")),
        "side": side_from_name(name)
        if side_from_name(name) != UNKNOWN
        else side_from_name(actuator.attrib.get("joint")),
    }


def collect_body_tree(
    body: ET.Element,
    parent: str | None,
    depth: int,
    bodies: list[dict[str, Any]],
    joints: list[dict[str, Any]],
    geoms: list[dict[str, Any]],
) -> None:
    """Collect nested body, joint, and geom records recursively."""

    current = body.attrib.get("name") or f"unnamed_body_depth_{depth}"
    bodies.append(body_record(body, parent, depth))
    for freejoint in body.findall("freejoint"):
        joints.append(joint_record(freejoint, current, "freejoint"))
    for joint in body.findall("joint"):
        joints.append(joint_record(joint, current, "joint"))
    for geom in body.findall("geom"):
        geoms.append(geom_record(geom, current))
    for child in body.findall("body"):
        collect_body_tree(child, current, depth + 1, bodies, joints, geoms)


def extract_robot_parameters(xml_path: Path, robot_name: str) -> dict[str, Any]:
    """Extract robot parameters from an MJCF XML source.

    Args:
        xml_path: MJCF XML path.
        robot_name: Human-readable robot label.

    Returns:
        Structured parameter dictionary.

    Raises:
        ExtractionError: If the XML source is missing or invalid.
    """

    root = load_xml(xml_path)
    bodies: list[dict[str, Any]] = []
    joints: list[dict[str, Any]] = []
    geoms: list[dict[str, Any]] = []

    worldbody = root.find("worldbody")
    if worldbody is not None:
        for geom in worldbody.findall("geom"):
            geoms.append(geom_record(geom, "worldbody"))
        for body in worldbody.findall("body"):
            collect_body_tree(body, "worldbody", 0, bodies, joints, geoms)

    actuators: list[dict[str, Any]] = []
    actuator_root = root.find("actuator")
    if actuator_root is not None:
        actuator_defaults = default_actuator_fields(root)
        actuators = [actuator_record(child, actuator_defaults) for child in list(actuator_root)]
    else:
        actuator_defaults = default_actuator_fields(root)

    foot_geoms = [geom for geom in geoms if geom["foot_related"]]
    foot_bodies = [body for body in bodies if body["foot_related"]]
    foot_joints = [joint for joint in joints if joint["foot_related"]]
    body_mass_values = [
        body["inertial"]["mass"]
        for body in bodies
        if isinstance(body.get("inertial"), dict) and isinstance(body["inertial"].get("mass"), (int, float))
    ]

    actuator_kinds = Counter(record["kind"] for record in actuators)
    geom_types = Counter(str(record["type"] or UNKNOWN) for record in geoms)
    joint_types = Counter(str(record["type"] or UNKNOWN) for record in joints)

    return {
        "schema_version": 1,
        "tool_version": TOOL_VERSION,
        "extracted_at": utc_now_iso(),
        "robot_name": robot_name,
        "source_xml": str(xml_path),
        "model": root.attrib.get("model"),
        "compiler": dict(root.find("compiler").attrib) if root.find("compiler") is not None else None,
        "option": dict(root.find("option").attrib) if root.find("option") is not None else None,
        "defaults": {
            "actuators": actuator_defaults,
        },
        "counts": {
            "bodies": len(bodies),
            "joints": len(joints),
            "geoms": len(geoms),
            "actuators": len(actuators),
            "foot_related_bodies": len(foot_bodies),
            "foot_related_joints": len(foot_joints),
            "foot_related_geoms": len(foot_geoms),
        },
        "summary": {
            "total_explicit_body_mass": round(sum(body_mass_values), 9) if body_mass_values else None,
            "body_mass_count": len(body_mass_values),
            "joint_types": dict(sorted(joint_types.items())),
            "geom_types": dict(sorted(geom_types.items())),
            "actuator_kinds": dict(sorted(actuator_kinds.items())),
            "unknown_notes": [
                "Values omitted by the source MJCF are null.",
                "MuJoCo defaults and compiled inertias are not inferred by this XML extractor.",
                "Foot-related fields use a name heuristic and should be manually reviewed.",
            ],
        },
        "bodies": bodies,
        "joints": joints,
        "geoms": geoms,
        "actuators": actuators,
        "foot_related": {
            "bodies": foot_bodies,
            "joints": foot_joints,
            "geoms": foot_geoms,
        },
    }


def write_yaml_compatible_json(path: Path, payload: dict[str, Any]) -> None:
    """Write deterministic YAML-compatible JSON to a ``.yaml`` path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def read_yaml_compatible_json(path: Path) -> dict[str, Any]:
    """Read a YAML-compatible JSON file written by this extractor.

    Args:
        path: Parameter file path.

    Returns:
        Parsed mapping.

    Raises:
        ExtractionError: If the file is missing or cannot be parsed.
    """

    if not path.is_file():
        raise ExtractionError(f"Parameter YAML not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ExtractionError(
            f"Could not parse {path}. This comparison tool expects the YAML-compatible JSON emitted by the extractors."
        ) from exc
    if not isinstance(payload, dict):
        raise ExtractionError(f"Expected a mapping at top level: {path}")
    return payload


def compact_names(records: list[dict[str, Any]]) -> list[str]:
    """Return names from record dictionaries."""

    return [str(record.get("name", UNKNOWN)) for record in records]


def missing_field_names(records: list[dict[str, Any]], field: str) -> list[str]:
    """Return record names where a field is missing or null."""

    return [str(record.get("name", UNKNOWN)) for record in records if record.get(field) is None]
