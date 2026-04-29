from __future__ import annotations

import xml.etree.ElementTree as ET

from tools.build_sedon_training_scene import (
    _add_training_proxy_geoms,
    _build_base_body,
    _replace_actuators,
    _set_mesh_geoms_visual_only,
)


def test_build_base_body_wraps_existing_children_with_freejoint() -> None:
    base_geom = ET.Element("geom", {"mesh": "base_link"})
    leg_body = ET.Element("body", {"name": "R_link_hip_yaw"})
    inertial = ET.Element("inertial", {"mass": "1", "pos": "0 0 0"})

    base_body = _build_base_body([base_geom, leg_body], inertial, base_height=0.46)

    assert base_body.tag == "body"
    assert base_body.attrib["name"] == "base_link"
    assert base_body.attrib["pos"] == "0 0 0.46"
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


def test_training_proxy_geoms_hide_base_mesh_and_add_stable_contacts() -> None:
    root = ET.Element("mujoco")
    worldbody = ET.SubElement(root, "worldbody")
    base = ET.SubElement(worldbody, "body", {"name": "base_link"})
    ET.SubElement(base, "geom", {"type": "mesh", "mesh": "base_link"})
    right_foot = ET.SubElement(base, "body", {"name": "R_link_ankle_pitch"})
    left_foot = ET.SubElement(base, "body", {"name": "L_link_ankle_pitch"})
    ET.SubElement(right_foot, "geom", {"type": "mesh", "mesh": "R_link_ankle_pitch"})
    ET.SubElement(left_foot, "geom", {"type": "mesh", "mesh": "L_link_ankle_pitch"})

    _set_mesh_geoms_visual_only(root)
    _add_training_proxy_geoms(root)

    assert base.find("geom[@mesh='base_link']") is None
    assert base.find("geom[@name='base_proxy']").attrib["type"] == "ellipsoid"
    assert right_foot.find("geom[@name='R_foot_collision']").attrib["type"] == "box"
    assert left_foot.find("geom[@name='L_foot_collision']").attrib["type"] == "box"
    assert right_foot.find("geom[@name='R_foot_collision']").attrib["rgba"].endswith(" 0")
    assert right_foot.find("geom[@mesh='R_link_ankle_pitch']").attrib["contype"] == "0"
