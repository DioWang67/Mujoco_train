"""Build a minimal MuJoCo training scene from the converted Sedon MJCF."""

from __future__ import annotations

import argparse
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

BASE_LINK_NAME = "base_link"
DEFAULT_BASE_HEIGHT = 0.46
DEFAULT_MOTOR_CTRL_RANGE = "-100 100"


def _set_compiler_defaults(root: ET.Element, mesh_dir: Path | None = None) -> None:
    """Ensure compiler settings are suitable for generated limits."""
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    compiler.set("angle", "radian")
    compiler.set("autolimits", "true")
    if mesh_dir is not None:
        compiler.set("meshdir", str(mesh_dir))


def _add_option(root: ET.Element) -> None:
    """Add conservative simulation defaults if missing."""
    if root.find("option") is not None:
        return
    option = ET.Element(
        "option",
        {
            "timestep": "0.002",
            "integrator": "RK4",
            "gravity": "0 0 -9.81",
        },
    )
    insert_index = 1 if root.find("compiler") is not None else 0
    root.insert(insert_index, option)


def _extract_base_inertial(urdf_path: Path) -> ET.Element:
    """Return a MJCF inertial element for ``base_link`` from the source URDF.

    Args:
        urdf_path: Prepared URDF used for the MuJoCo conversion.

    Returns:
        MJCF ``inertial`` element.

    Raises:
        ValueError: If the URDF has no usable ``base_link`` inertial data.
    """
    root = ET.parse(urdf_path).getroot()
    link = root.find(f"link[@name='{BASE_LINK_NAME}']")
    inertial = link.find("inertial") if link is not None else None
    if inertial is None:
        raise ValueError(f"URDF link '{BASE_LINK_NAME}' has no inertial element.")

    origin = inertial.find("origin")
    mass = inertial.find("mass")
    inertia = inertial.find("inertia")
    if mass is None or inertia is None:
        raise ValueError(f"URDF link '{BASE_LINK_NAME}' has incomplete inertial data.")

    ixx = inertia.attrib["ixx"]
    iyy = inertia.attrib["iyy"]
    izz = inertia.attrib["izz"]
    ixy = inertia.attrib["ixy"]
    ixz = inertia.attrib["ixz"]
    iyz = inertia.attrib["iyz"]
    return ET.Element(
        "inertial",
        {
            "pos": origin.attrib.get("xyz", "0 0 0") if origin is not None else "0 0 0",
            "mass": mass.attrib["value"],
            "fullinertia": f"{ixx} {iyy} {izz} {ixy} {ixz} {iyz}",
        },
    )


def _find_worldbody(root: ET.Element) -> ET.Element:
    """Return the MJCF worldbody element."""
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("MJCF has no worldbody.")
    return worldbody


def _take_world_children(worldbody: ET.Element) -> list[ET.Element]:
    """Remove and return all current worldbody children."""
    children = list(worldbody)
    for child in children:
        worldbody.remove(child)
    return children


def _build_base_body(
    world_children: list[ET.Element],
    base_inertial: ET.Element,
    base_height: float,
) -> ET.Element:
    """Wrap converted fixed-base bodies/geoms in a floating base body."""
    base_body = ET.Element(
        "body",
        {
            "name": BASE_LINK_NAME,
            "pos": f"0 0 {base_height:g}",
        },
    )
    base_body.append(base_inertial)
    base_body.append(ET.Element("freejoint", {"name": "floating_base"}))

    for child in world_children:
        base_body.append(child)
    return base_body


def _add_floor_and_lighting(worldbody: ET.Element) -> None:
    """Add a simple floor and light for smoke testing and visualization."""
    worldbody.append(
        ET.Element(
            "geom",
            {
                "name": "floor",
                "type": "plane",
                "size": "2 2 0.05",
                "rgba": "0.55 0.55 0.55 1",
                "friction": "1.0 0.005 0.0001",
            },
        )
    )
    worldbody.append(
        ET.Element(
            "light",
            {
                "name": "key_light",
                "pos": "0 -1.5 2.5",
                "dir": "0 1 -1",
                "diffuse": "0.8 0.8 0.8",
            },
        )
    )


def _joint_names(root: ET.Element) -> list[str]:
    """Return controllable hinge joint names from the model."""
    names: list[str] = []
    for joint in root.findall(".//joint"):
        name = joint.attrib.get("name")
        if not name:
            continue
        if joint.attrib.get("type", "hinge") != "hinge":
            continue
        names.append(name)
    return names


def _replace_actuators(root: ET.Element, ctrl_range: str) -> list[str]:
    """Create one position-like motor per hinge joint.

    Args:
        root: MJCF root.
        ctrl_range: Motor control range as ``"low high"``.

    Returns:
        Names of joints that received motors.
    """
    for actuator in root.findall("actuator"):
        root.remove(actuator)

    joint_names = _joint_names(root)
    actuator = ET.Element("actuator")
    for joint_name in joint_names:
        actuator.append(
            ET.Element(
                "motor",
                {
                    "name": f"{joint_name}_motor",
                    "joint": joint_name,
                    "ctrlrange": ctrl_range,
                    "ctrllimited": "true",
                },
            )
        )
    root.append(actuator)
    return joint_names


def build_training_scene(
    source_scene: Path,
    source_urdf: Path,
    output_scene: Path,
    base_height: float,
    motor_ctrl_range: str,
) -> tuple[Path, mujoco.MjModel, list[str]]:
    """Build and compile a minimal Sedon training scene.

    Args:
        source_scene: Converted MJCF scene from ``convert_urdf_to_mjcf``.
        source_urdf: Prepared URDF containing base inertial data.
        output_scene: Destination MJCF scene.
        base_height: Initial floating-base height above the floor.
        motor_ctrl_range: Control range for all motors.

    Returns:
        Output scene path, compiled MuJoCo model, and actuated joint names.
    """
    tree = ET.parse(source_scene)
    root = tree.getroot()
    root.set("model", "sedon_training")
    relative_mesh_dir = Path(
        os.path.relpath(source_scene.parent.resolve(), output_scene.parent.resolve())
    )
    _set_compiler_defaults(root, relative_mesh_dir)
    _add_option(root)

    worldbody = _find_worldbody(root)
    children = _take_world_children(worldbody)
    _add_floor_and_lighting(worldbody)
    worldbody.append(
        _build_base_body(
            world_children=children,
            base_inertial=_extract_base_inertial(source_urdf),
            base_height=base_height,
        )
    )

    actuated_joints = _replace_actuators(root, motor_ctrl_range)
    ET.indent(tree, space="  ")
    output_scene.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_scene, encoding="utf-8", xml_declaration=True)

    model = mujoco.MjModel.from_xml_path(str(output_scene))
    return output_scene, model, actuated_joints


def build_parser() -> argparse.ArgumentParser:
    """Build the command line parser."""
    parser = argparse.ArgumentParser(
        description="Build a minimal floating-base Sedon MuJoCo training scene."
    )
    parser.add_argument(
        "--source-scene",
        type=Path,
        default=Path("private_assets/sedon/mjcf_source/scene.xml"),
        help="Converted source MJCF scene.",
    )
    parser.add_argument(
        "--source-urdf",
        type=Path,
        default=Path("private_assets/sedon/mjcf_source/sedon.urdf"),
        help="Prepared URDF with base inertial data.",
    )
    parser.add_argument(
        "--output-scene",
        type=Path,
        default=Path("private_assets/sedon/training_scene.xml"),
        help="Output training MJCF scene.",
    )
    parser.add_argument(
        "--base-height",
        type=float,
        default=DEFAULT_BASE_HEIGHT,
        help="Initial base height above the floor.",
    )
    parser.add_argument(
        "--motor-ctrl-range",
        default=DEFAULT_MOTOR_CTRL_RANGE,
        help="Control range used for generated motors.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the training scene build command."""
    args = build_parser().parse_args(argv)
    output_scene, model, joint_names = build_training_scene(
        source_scene=args.source_scene,
        source_urdf=args.source_urdf,
        output_scene=args.output_scene,
        base_height=args.base_height,
        motor_ctrl_range=args.motor_ctrl_range,
    )
    print(f"Saved training scene: {output_scene}")
    print(
        "Model: "
        f"nbody={model.nbody} njnt={model.njnt} ngeom={model.ngeom} "
        f"nq={model.nq} nv={model.nv} nu={model.nu}"
    )
    print("Actuated joints:")
    for joint_name in joint_names:
        print(f"- {joint_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
