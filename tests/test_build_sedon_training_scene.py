from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from tools.build_sedon_training_scene import (
    _build_base_body,
    _replace_actuators,
)


def test_build_base_body_wraps_existing_children_with_freejoint() -> None:
    base_geom = ET.Element("geom", {"mesh": "base_link"})
    leg_body = ET.Element("body", {"name": "R_link_hip_yaw"})
    inertial = ET.Element("inertial", {"mass": "1", "pos": "0 0 0"})

    base_body = _build_base_body([base_geom, leg_body], inertial, base_height=0.55)

    assert base_body.tag == "body"
    assert base_body.attrib["name"] == "base_link"
    assert base_body.attrib["pos"] == "0 0 0.55"
    assert base_body.find("freejoint").attrib["name"] == "floating_base"
    assert base_body.find("geom").attrib["mesh"] == "base_link"
    assert base_body.find("body").attrib["name"] == "R_link_hip_yaw"


def test_replace_actuators_creates_one_motor_per_hinge_joint() -> None:
    root = ET.Element("mujoco")
    worldbody = ET.SubElement(root, "worldbody")
    body = ET.SubElement(worldbody, "body", {"name": "link"})
    ET.SubElement(body, "joint", {"name": "joint_a", "type": "hinge"})
    ET.SubElement(body, "joint", {"name": "joint_b"})

    joint_names = _replace_actuators(root, "-10 10")

    actuator = root.find("actuator")
    motors = actuator.findall("motor")
    assert joint_names == ["joint_a", "joint_b"]
    assert [motor.attrib["joint"] for motor in motors] == ["joint_a", "joint_b"]
    assert all(motor.attrib["ctrlrange"] == "-10 10" for motor in motors)
