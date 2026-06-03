"""Generate the Seedon v5_22 MuJoCo scene from the updated URDF package.

This Class C migration helper creates a versioned MJCF scene without modifying
the source URDF, existing baseline XML, training code, or evaluation code.
Prototype-only fields such as actuator stubs and joint-name adapters are marked
as assumptions in the generated XML and companion metadata.
"""

from __future__ import annotations

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from xml.dom import minidom


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_URDF = REPO_ROOT / "private_assets" / "SEEDON_URDF_5_22" / "urdf" / "SEEDON_URDF_5_21.urdf"
DEFAULT_MESH_SOURCE_DIR = REPO_ROOT / "private_assets" / "SEEDON_URDF_5_22" / "meshes"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "private_assets" / "seedon_v5_22"
DEFAULT_SCENE_PATH = DEFAULT_OUTPUT_ROOT / "training_scene.xml"
DEFAULT_METADATA_PATH = DEFAULT_OUTPUT_ROOT / "seedon_v5_22_scene_metadata.json"

JOINT_ORDER = (
    "R_joint_hip_yaw",
    "R_joint_hip_roll",
    "R_joint_hip_pitch",
    "R_joint_knee_pitch",
    "R_joint_ankle_pitch",
    "L_joint_hip_yaw",
    "L_joint_hip_roll",
    "L_joint_hip_pitch",
    "L_joint_knee_pitch",
    "L_joint_ankle_pitch",
)
JOINT_NAME_MAP = {
    "R_joint_knee": "R_joint_knee_pitch",
    "R_joint_knee_pitch": "R_joint_ankle_pitch",
}
FALLBACK_EFFORT_BY_JOINT = {
    "R_joint_hip_yaw": "300",
}


@dataclass(frozen=True)
class LinkRecord:
    """URDF link fields needed for MJCF generation.

    Args:
        name: URDF link name.
        mass: Link mass from URDF inertial fields.
        inertial_pos: Link COM position in link frame.
        inertia: Full inertia fields from URDF.
        has_mesh: Whether a visual mesh exists for the link.
    """

    name: str
    mass: str
    inertial_pos: str
    inertia: dict[str, str]
    has_mesh: bool


@dataclass(frozen=True)
class JointRecord:
    """URDF joint fields needed for MJCF generation.

    Args:
        source_name: Joint name from the source URDF.
        mjcf_name: Joint name used in generated MJCF.
        parent: Parent link name.
        child: Child link name.
        origin_xyz: Joint origin relative to parent.
        origin_rpy: Joint orientation.
        axis: Joint axis.
        lower: Lower joint range.
        upper: Upper joint range.
        effort: URDF effort limit after prototype fallback if required.
        velocity: URDF velocity limit.
        effort_source: Source label for effort value.
    """

    source_name: str
    mjcf_name: str
    parent: str
    child: str
    origin_xyz: str
    origin_rpy: str
    axis: str
    lower: str
    upper: str
    effort: str
    velocity: str
    effort_source: str


def parse_urdf(urdf_path: Path) -> tuple[dict[str, LinkRecord], dict[str, JointRecord], dict[str, list[str]]]:
    """Parse the Seedon v5_22 URDF.

    Args:
        urdf_path: Source URDF path.

    Returns:
        Tuple of link records, child-link joint records, and parent-to-children map.

    Raises:
        FileNotFoundError: If the URDF does not exist.
        ValueError: If required link/joint fields are missing.
    """

    if not urdf_path.is_file():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")
    root = ET.parse(urdf_path).getroot()
    links: dict[str, LinkRecord] = {}
    for link in root.findall("link"):
        name = require_attr(link, "name")
        inertial = link.find("inertial")
        if inertial is None:
            raise ValueError(f"Missing inertial for link: {name}")
        mass = inertial.find("mass")
        origin = inertial.find("origin")
        inertia = inertial.find("inertia")
        if mass is None or origin is None or inertia is None:
            raise ValueError(f"Incomplete inertial fields for link: {name}")
        mesh = link.find("visual/geometry/mesh")
        links[name] = LinkRecord(
            name=name,
            mass=require_attr(mass, "value"),
            inertial_pos=origin.attrib.get("xyz", "0 0 0"),
            inertia={key: require_attr(inertia, key) for key in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")},
            has_mesh=mesh is not None,
        )

    joints_by_child: dict[str, JointRecord] = {}
    children_by_parent: dict[str, list[str]] = {}
    for joint in root.findall("joint"):
        source_name = require_attr(joint, "name")
        parent = require_attr(require_child(joint, "parent"), "link")
        child = require_attr(require_child(joint, "child"), "link")
        axis = require_child(joint, "axis")
        limit = require_child(joint, "limit")
        origin = require_child(joint, "origin")
        effort = require_attr(limit, "effort")
        effort_source = "urdf"
        if float(effort) <= 0.0 and source_name in FALLBACK_EFFORT_BY_JOINT:
            effort = FALLBACK_EFFORT_BY_JOINT[source_name]
            effort_source = "source=assumption, confidence=low, reason=source_urdf_effort_velocity_zero"
        joints_by_child[child] = JointRecord(
            source_name=source_name,
            mjcf_name=JOINT_NAME_MAP.get(source_name, source_name),
            parent=parent,
            child=child,
            origin_xyz=origin.attrib.get("xyz", "0 0 0"),
            origin_rpy=origin.attrib.get("rpy", "0 0 0"),
            axis=require_attr(axis, "xyz"),
            lower=require_attr(limit, "lower"),
            upper=require_attr(limit, "upper"),
            effort=effort,
            velocity=require_attr(limit, "velocity"),
            effort_source=effort_source,
        )
        children_by_parent.setdefault(parent, []).append(child)
    return links, joints_by_child, children_by_parent


def require_child(element: ET.Element, tag: str) -> ET.Element:
    """Return a required XML child."""

    child = element.find(tag)
    if child is None:
        raise ValueError(f"Missing <{tag}> under <{element.tag}> {element.attrib}")
    return child


def require_attr(element: ET.Element, name: str) -> str:
    """Return a required XML attribute."""

    if name not in element.attrib:
        raise ValueError(f"Missing attribute '{name}' on <{element.tag}>")
    return element.attrib[name]


def copy_meshes(mesh_source_dir: Path, mesh_output_dir: Path) -> None:
    """Copy STL meshes into the versioned MJCF source directory."""

    if not mesh_source_dir.is_dir():
        raise FileNotFoundError(f"Mesh source directory not found: {mesh_source_dir}")
    mesh_output_dir.mkdir(parents=True, exist_ok=True)
    for mesh_path in sorted(mesh_source_dir.glob("*.STL")):
        shutil.copy2(mesh_path, mesh_output_dir / mesh_path.name)


def add_assumption_comment(parent: ET.Element, text: str) -> None:
    """Append a source/confidence annotation comment."""

    parent.append(ET.Comment(f" source=assumption, confidence=low, valid_for=simulation_prototype_only, {text} "))


def build_mjcf(
    *,
    links: dict[str, LinkRecord],
    joints_by_child: dict[str, JointRecord],
    children_by_parent: dict[str, list[str]],
) -> ET.Element:
    """Build the Seedon v5_22 MJCF tree."""

    mjcf = ET.Element("mujoco", {"model": "seedon_v5_22"})
    ET.SubElement(mjcf, "compiler", {"angle": "radian", "autolimits": "true", "meshdir": "mjcf_source"})
    ET.SubElement(mjcf, "option", {"timestep": "0.002", "integrator": "RK4", "gravity": "0 0 -9.81"})
    asset = ET.SubElement(mjcf, "asset")
    for link_name in links:
        if links[link_name].has_mesh:
            ET.SubElement(asset, "mesh", {"name": link_name, "file": f"{link_name}.STL"})

    world = ET.SubElement(mjcf, "worldbody")
    ET.SubElement(
        world,
        "geom",
        {
            "name": "floor",
            "type": "plane",
            "size": "2 2 0.05",
            "rgba": "0.55 0.55 0.55 1",
            "friction": "1.0 0.005 0.0001",
        },
    )
    ET.SubElement(world, "light", {"name": "key_light", "pos": "0 -1.5 2.5", "dir": "0 1 -1"})
    add_link_body(world, "base_link", links, joints_by_child, children_by_parent, is_root=True)

    actuator = ET.SubElement(mjcf, "actuator")
    add_assumption_comment(
        actuator,
        "MuJoCo motor stubs copied from prior Seedon simulation convention; not verified motor specs",
    )
    for joint_name in JOINT_ORDER:
        ET.SubElement(
            actuator,
            "motor",
            {
                "name": f"{joint_name}_motor",
                "joint": joint_name,
                "ctrlrange": "-100 100",
                "ctrllimited": "true",
            },
        )
    return mjcf


def add_link_body(
    parent: ET.Element,
    link_name: str,
    links: dict[str, LinkRecord],
    joints_by_child: dict[str, JointRecord],
    children_by_parent: dict[str, list[str]],
    *,
    is_root: bool = False,
) -> ET.Element:
    """Append one link body and its descendants."""

    if is_root:
        body = ET.SubElement(parent, "body", {"name": link_name, "pos": "0 0 0.62"})
        ET.SubElement(body, "freejoint", {"name": "floating_base"})
    else:
        joint = joints_by_child[link_name]
        body_attrs = {"name": link_name, "pos": joint.origin_xyz}
        if joint.origin_rpy != "0 0 0":
            body_attrs["euler"] = joint.origin_rpy
        body = ET.SubElement(parent, "body", body_attrs)
        if joint.source_name != joint.mjcf_name:
            add_assumption_comment(
                body,
                f"joint_name_adapter source_joint={joint.source_name} mjcf_joint={joint.mjcf_name}",
            )
        if joint.effort_source != "urdf":
            add_assumption_comment(body, f"joint_effort_fallback joint={joint.source_name} effort={joint.effort}")
        ET.SubElement(
            body,
            "joint",
            {
                "name": joint.mjcf_name,
                "pos": "0 0 0",
                "axis": joint.axis,
                "range": f"{joint.lower} {joint.upper}",
                "actuatorfrcrange": f"-{joint.effort} {joint.effort}",
            },
        )

    link = links[link_name]
    inertia = link.inertia
    ET.SubElement(
        body,
        "inertial",
        {
            "pos": link.inertial_pos,
            "mass": link.mass,
            "fullinertia": (
                f"{inertia['ixx']} {inertia['iyy']} {inertia['izz']} "
                f"{inertia['ixy']} {inertia['ixz']} {inertia['iyz']}"
            ),
        },
    )
    if link.has_mesh:
        ET.SubElement(
            body,
            "geom",
            {
                "type": "mesh",
                "mesh": link_name,
                "rgba": "0.752941 0.752941 0.752941 1",
                "contype": "0",
                "conaffinity": "0",
                "group": "1",
            },
        )
    add_eval_collision_proxy(body, link_name)
    for child in children_by_parent.get(link_name, []):
        add_link_body(body, child, links, joints_by_child, children_by_parent)
    return body


def add_eval_collision_proxy(body: ET.Element, link_name: str) -> None:
    """Add invisible env-compatible collision proxies."""

    if link_name == "base_link":
        add_assumption_comment(body, "base_proxy added for existing Seedon eval contact/fall semantics")
        ET.SubElement(
            body,
            "geom",
            {
                "name": "base_proxy",
                "type": "ellipsoid",
                "pos": "-0.02 0 -0.08",
                "size": "0.17 0.11 0.10",
                "rgba": "0 0 0 0",
                "friction": "0.8 0.005 0.0001",
            },
        )
    if link_name == "R_link_ankle_pitch":
        add_foot_collision(body, "R_foot_collision", "0.025 0.025 -0.054")
    if link_name == "L_link_ankle_pitch":
        add_foot_collision(body, "L_foot_collision", "0.025 -0.025 -0.054")


def add_foot_collision(body: ET.Element, name: str, pos: str) -> None:
    """Add one invisible foot contact proxy."""

    add_assumption_comment(body, f"{name} box copied from prior Seedon simulation convention")
    ET.SubElement(
        body,
        "geom",
        {
            "name": name,
            "type": "box",
            "pos": pos,
            "size": "0.07 0.04 0.025",
            "rgba": "0 0 0 0",
            "friction": "1.0 0.005 0.0001",
        },
    )


def write_scene(path: Path, root: ET.Element) -> None:
    """Write pretty MJCF XML."""

    path.parent.mkdir(parents=True, exist_ok=True)
    rough = ET.tostring(root, encoding="utf-8")
    path.write_text(minidom.parseString(rough).toprettyxml(indent="  "), encoding="utf-8")


def write_metadata(path: Path, *, urdf_path: Path, scene_path: Path) -> None:
    """Write companion metadata for the generated scene."""

    payload = {
        "schema_version": 1,
        "version": "v5_22",
        "source_urdf": str(urdf_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "scene_path": str(scene_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "source": "prototype_generator",
        "valid_for": "simulation_prototype_only",
        "assumptions": [
            {
                "field": "motor_ctrlrange",
                "value": "-100 100",
                "source": "assumption",
                "confidence": "low",
                "reason": "No verified motor/controller spec found for Seedon v5_22.",
            },
            {
                "field": "R_joint_hip_yaw_effort",
                "value": "300",
                "source": "assumption",
                "confidence": "low",
                "reason": "Source URDF has effort=0 velocity=0; fallback keeps MuJoCo prototype controllable.",
            },
            {
                "field": "right_leg_joint_name_adapter",
                "value": JOINT_NAME_MAP,
                "source": "assumption",
                "confidence": "low",
                "reason": "Existing Seedon env expects R_joint_knee_pitch and R_joint_ankle_pitch.",
            },
            {
                "field": "foot_collision_boxes",
                "value": "prior Seedon MuJoCo convention",
                "source": "assumption",
                "confidence": "low",
                "reason": "Source URDF collision meshes are not suitable as verified rollover contact geometry.",
            },
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--mesh-source-dir", type=Path, default=DEFAULT_MESH_SOURCE_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> int:
    """Generate the Seedon v5_22 MJCF scene."""

    args = parse_args()
    output_root = args.output_root
    scene_path = output_root / "training_scene.xml"
    metadata_path = output_root / "seedon_v5_22_scene_metadata.json"
    mesh_output_dir = output_root / "mjcf_source"
    copy_meshes(args.mesh_source_dir, mesh_output_dir)
    links, joints_by_child, children_by_parent = parse_urdf(args.urdf)
    mjcf = build_mjcf(
        links=links,
        joints_by_child=joints_by_child,
        children_by_parent=children_by_parent,
    )
    write_scene(scene_path, mjcf)
    write_metadata(metadata_path, urdf_path=args.urdf, scene_path=scene_path)
    print(f"scene={scene_path}")
    print(f"metadata={metadata_path}")
    print(f"mesh_dir={mesh_output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
