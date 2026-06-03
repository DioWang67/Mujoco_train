"""Validate Seedon hip-roll model structure, kinematics, and dynamics."""

from __future__ import annotations

import argparse
import csv
import math
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np

from seedon_baseline.env import SeedonStandingEnv, load_seedon_config_from_env
from tools.seedon_debug_common import DEBUG_OUT_DIR, contact_pairs


DEFAULT_OUT_DIR = DEBUG_OUT_DIR / "hip_roll_model_validation"
DEFAULT_REPORT_PATH = DEFAULT_OUT_DIR / "report.md"
DEFAULT_METADATA_CSV = DEFAULT_OUT_DIR / "joint_metadata.csv"
DEFAULT_KINEMATIC_CSV = DEFAULT_OUT_DIR / "kinematic_sweep.csv"
DEFAULT_DYNAMICS_CSV = DEFAULT_OUT_DIR / "dynamics_sweep.csv"
URDF_PATH = Path("private_assets/seedon/mjcf_source/seedon.urdf")
MJCF_PATH = Path("private_assets/seedon/training_scene.xml")
RIGHT_HIP_ROLL_NAME = "R_joint_hip_roll"
LEFT_HIP_ROLL_NAME = "L_joint_hip_roll"
RIGHT_HIP_ROLL_ACTUATOR = "R_joint_hip_roll_motor"
LEFT_HIP_ROLL_ACTUATOR = "L_joint_hip_roll_motor"
RIGHT_FOOT_GEOM = "R_foot_collision"
LEFT_FOOT_GEOM = "L_foot_collision"
BASE_BODY = "base_link"
RIGHT_CHAIN_BODIES = ("R_link_knee_pitch", "R_link_ankle_pitch")
LEFT_CHAIN_BODIES = ("L_link_knee_pitch", "L_link_ankle_pitch")
MODES = ("right_only", "left_only", "symmetric")


@dataclass(frozen=True)
class HipRollJointMetadata:
    """Structural metadata for one hip-roll joint across URDF and MuJoCo."""

    joint_name: str
    urdf_axis: str
    mujoco_axis: str
    urdf_range: str
    mujoco_range: str
    urdf_origin_xyz: str
    mujoco_joint_frame_pos: str
    qpos_address: int
    parent_body: str
    child_body: str
    hip_roll_visual_origin: str
    hip_roll_collision_origin: str


@dataclass(frozen=True)
class HipRollKinematicResult:
    """Kinematic-only forward result for one offset candidate."""

    mode: str
    offset: float
    target_right_hip_roll: float
    target_left_hip_roll: float
    actual_right_hip_roll: float
    actual_left_hip_roll: float
    base_y: float
    com_y: float
    right_foot_y: float
    left_foot_y: float
    right_knee_pos: tuple[float, float, float]
    right_ankle_pos: tuple[float, float, float]
    right_foot_pos: tuple[float, float, float]
    left_knee_pos: tuple[float, float, float]
    left_ankle_pos: tuple[float, float, float]
    left_foot_pos: tuple[float, float, float]


@dataclass(frozen=True)
class HipRollDynamicsResult:
    """Dynamics target-tracking result for one hip-roll candidate."""

    mode: str
    offset: float
    support_side: str
    target_right_hip_roll: float
    target_left_hip_roll: float
    actual_right_hip_roll: float
    actual_left_hip_roll: float
    right_hip_roll_error: float
    left_hip_roll_error: float
    base_dy: float
    com_dy: float
    base_z: float
    upright: float
    support_contact_ratio: float
    swing_contact_ratio: float
    right_foot_bottom_z: float
    left_foot_bottom_z: float
    right_ctrl_max_abs: float
    left_ctrl_max_abs: float
    overall_ctrl_max_abs: float
    right_ctrl_saturation_ratio: float
    left_ctrl_saturation_ratio: float
    contact_pair_summary: str
    terminated: bool
    steps: int


def _parse_offsets(raw_value: str) -> list[float]:
    """Parse comma-separated float offsets."""
    offsets = [float(part.strip()) for part in raw_value.split(",") if part.strip()]
    if not offsets:
        raise argparse.ArgumentTypeError("Expected at least one offset.")
    return offsets


def _format_vector(values: np.ndarray | tuple[float, ...] | list[float]) -> str:
    """Return a stable human-readable vector string."""
    array = np.asarray(values, dtype=np.float64)
    return "[" + ", ".join(f"{float(value):.6f}" for value in array.tolist()) + "]"


def _normalize_pair(name_a: str, name_b: str) -> tuple[str, str]:
    """Return a stable contact-pair key."""
    return tuple(sorted((name_a, name_b)))


def _overall_com(env: SeedonStandingEnv) -> np.ndarray:
    """Return mass-weighted whole-model COM in world coordinates."""
    masses = env.model.body_mass
    total_mass = float(np.sum(masses))
    if total_mass <= 0.0:
        raise ValueError("Model has no positive body mass.")
    return np.sum(env.data.xipos * masses[:, None], axis=0) / total_mass


def _floor_contact_flags(env: SeedonStandingEnv) -> dict[str, bool]:
    """Return floor contact flags for feet and base proxy."""
    flags = {"right": False, "left": False, "base_proxy": False}
    for name_a, name_b, _ in contact_pairs(env.model, env.data):
        pair = {name_a, name_b}
        if pair == {"floor", RIGHT_FOOT_GEOM}:
            flags["right"] = True
        elif pair == {"floor", LEFT_FOOT_GEOM}:
            flags["left"] = True
        elif pair == {"floor", "base_proxy"}:
            flags["base_proxy"] = True
    return flags


def _mode_support_side(mode: str, offset: float) -> str:
    """Return the support side label used for contact-ratio reporting."""
    if mode == "right_only":
        return "right"
    if mode == "left_only":
        return "left"
    if mode == "symmetric":
        return "right" if offset > 0.0 else "left"
    raise ValueError(f"Unsupported mode: {mode}")


def _support_and_swing_contact(flags: dict[str, bool], support_side: str) -> tuple[bool, bool]:
    """Return support/swing contact booleans for the selected support side."""
    if support_side == "right":
        return flags["right"], flags["left"]
    if support_side == "left":
        return flags["left"], flags["right"]
    raise ValueError(f"Unsupported support side: {support_side}")


def _resolve_joint_targets(
    *,
    mode: str,
    offset: float,
    baseline_right: float,
    baseline_left: float,
) -> tuple[float, float]:
    """Return absolute right/left hip-roll targets for one candidate."""
    if mode == "right_only":
        return baseline_right + offset, baseline_left
    if mode == "left_only":
        return baseline_right, baseline_left + offset
    if mode == "symmetric":
        return baseline_right + offset, baseline_left - offset
    raise ValueError(f"Unsupported mode: {mode}")


def _body_world_position(env: SeedonStandingEnv, body_name: str) -> tuple[float, float, float]:
    """Return one body world position as a tuple."""
    body_id = env._body_id(body_name)
    return tuple(float(value) for value in env.data.xpos[body_id])


def _geom_world_position(env: SeedonStandingEnv, geom_name: str) -> tuple[float, float, float]:
    """Return one geom world position as a tuple."""
    geom_id = env._geom_id(geom_name)
    return tuple(float(value) for value in env.data.geom_xpos[geom_id])


def _hip_roll_actuator_ids(env: SeedonStandingEnv) -> tuple[int, int]:
    """Resolve the hip-roll actuator ids."""
    right_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, RIGHT_HIP_ROLL_ACTUATOR)
    left_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, LEFT_HIP_ROLL_ACTUATOR)
    if right_id < 0 or left_id < 0:
        raise ValueError("Hip-roll actuator ids could not be resolved.")
    return int(right_id), int(left_id)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write dictionaries to CSV using keys from the first row."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _load_xml_root(path: Path) -> ET.Element:
    """Load one XML file and return its root element."""
    tree = ET.parse(path)
    return tree.getroot()


def _find_urdf_joint(root: ET.Element, joint_name: str) -> ET.Element:
    """Return one URDF joint element by name."""
    for joint in root.findall("joint"):
        if joint.attrib.get("name") == joint_name:
            return joint
    raise ValueError(f"URDF joint not found: {joint_name}")


def _find_urdf_link(root: ET.Element, link_name: str) -> ET.Element:
    """Return one URDF link element by name."""
    for link in root.findall("link"):
        if link.attrib.get("name") == link_name:
            return link
    raise ValueError(f"URDF link not found: {link_name}")


def _find_urdf_origin(link: ET.Element, tag_name: str) -> str:
    """Return origin xyz for a URDF visual/collision element."""
    node = link.find(tag_name)
    if node is None:
        return "missing"
    origin = node.find("origin")
    if origin is None:
        return "missing"
    return origin.attrib.get("xyz", "missing")


def _joint_metadata(
    env: SeedonStandingEnv,
    urdf_root: ET.Element,
    joint_name: str,
) -> HipRollJointMetadata:
    """Collect cross-format metadata for one hip-roll joint."""
    urdf_joint = _find_urdf_joint(urdf_root, joint_name)
    child_body = urdf_joint.find("child")
    parent_body = urdf_joint.find("parent")
    axis = urdf_joint.find("axis")
    origin = urdf_joint.find("origin")
    limit = urdf_joint.find("limit")
    if child_body is None or parent_body is None or axis is None or origin is None or limit is None:
        raise ValueError(f"URDF joint is missing required fields: {joint_name}")

    child_name = child_body.attrib["link"]
    child_link = _find_urdf_link(urdf_root, child_name)

    joint_id = env._joint_id(joint_name)
    child_body_id = int(env.model.jnt_bodyid[joint_id])
    parent_body_id = int(env.model.body_parentid[child_body_id])
    parent_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, parent_body_id) or f"body_{parent_body_id}"
    child_name_mj = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, child_body_id) or f"body_{child_body_id}"
    mujoco_range = env.model.jnt_range[joint_id]
    mujoco_axis = env.model.jnt_axis[joint_id]
    mujoco_frame_pos = env.model.jnt_pos[joint_id]

    return HipRollJointMetadata(
        joint_name=joint_name,
        urdf_axis=axis.attrib.get("xyz", "missing"),
        mujoco_axis=_format_vector(mujoco_axis),
        urdf_range=f"[{limit.attrib.get('lower', 'missing')}, {limit.attrib.get('upper', 'missing')}]",
        mujoco_range=_format_vector(mujoco_range),
        urdf_origin_xyz=origin.attrib.get("xyz", "missing"),
        mujoco_joint_frame_pos=_format_vector(mujoco_frame_pos),
        qpos_address=int(env.model.jnt_qposadr[joint_id]),
        parent_body=parent_name,
        child_body=child_name_mj,
        hip_roll_visual_origin=_find_urdf_origin(child_link, "visual"),
        hip_roll_collision_origin=_find_urdf_origin(child_link, "collision"),
    )


def _foot_collision_mirror_summary(env: SeedonStandingEnv) -> dict[str, str]:
    """Return structural comparison for left/right foot collision geoms."""
    right_geom_id = env._geom_id(RIGHT_FOOT_GEOM)
    left_geom_id = env._geom_id(LEFT_FOOT_GEOM)
    right_local_pos = env.model.geom_pos[right_geom_id].copy()
    left_local_pos = env.model.geom_pos[left_geom_id].copy()
    right_size = env.model.geom_size[right_geom_id].copy()
    left_size = env.model.geom_size[left_geom_id].copy()
    mirrored_local_y = math.isclose(float(right_local_pos[0]), float(left_local_pos[0]), abs_tol=1e-9) and math.isclose(
        float(right_local_pos[1]), -float(left_local_pos[1]), abs_tol=1e-9
    ) and math.isclose(float(right_local_pos[2]), float(left_local_pos[2]), abs_tol=1e-9)
    mirrored_size = np.allclose(right_size, left_size)
    return {
        "right_local_pos": _format_vector(right_local_pos),
        "left_local_pos": _format_vector(left_local_pos),
        "right_size": _format_vector(right_size),
        "left_size": _format_vector(left_size),
        "mirrored_local_position": str(mirrored_local_y),
        "mirrored_size": str(bool(mirrored_size)),
    }


def _kinematic_result(
    env: SeedonStandingEnv,
    *,
    mode: str,
    offset: float,
    baseline_qpos: np.ndarray,
    baseline_qvel: np.ndarray,
    right_qpos_adr: int,
    left_qpos_adr: int,
) -> HipRollKinematicResult:
    """Run one kinematic-only forward pass without stepping dynamics."""
    qpos = baseline_qpos.copy()
    target_right, target_left = _resolve_joint_targets(
        mode=mode,
        offset=offset,
        baseline_right=float(qpos[right_qpos_adr]),
        baseline_left=float(qpos[left_qpos_adr]),
    )
    qpos[right_qpos_adr] = target_right
    qpos[left_qpos_adr] = target_left
    env.set_state(qpos, baseline_qvel)
    mujoco.mj_forward(env.model, env.data)

    return HipRollKinematicResult(
        mode=mode,
        offset=offset,
        target_right_hip_roll=target_right,
        target_left_hip_roll=target_left,
        actual_right_hip_roll=float(env.data.qpos[right_qpos_adr]),
        actual_left_hip_roll=float(env.data.qpos[left_qpos_adr]),
        base_y=float(env.data.qpos[1]),
        com_y=float(_overall_com(env)[1]),
        right_foot_y=float(env.data.geom_xpos[env._geom_id(RIGHT_FOOT_GEOM)][1]),
        left_foot_y=float(env.data.geom_xpos[env._geom_id(LEFT_FOOT_GEOM)][1]),
        right_knee_pos=_body_world_position(env, RIGHT_CHAIN_BODIES[0]),
        right_ankle_pos=_body_world_position(env, RIGHT_CHAIN_BODIES[1]),
        right_foot_pos=_geom_world_position(env, RIGHT_FOOT_GEOM),
        left_knee_pos=_body_world_position(env, LEFT_CHAIN_BODIES[0]),
        left_ankle_pos=_body_world_position(env, LEFT_CHAIN_BODIES[1]),
        left_foot_pos=_geom_world_position(env, LEFT_FOOT_GEOM),
    )


def _contact_pair_summary(counter: Counter[tuple[str, str]], limit: int = 4) -> str:
    """Return the most common contact pairs as a compact string."""
    if not counter:
        return "none"
    items = []
    for (name_a, name_b), count in counter.most_common(limit):
        items.append(f"{name_a}<->{name_b}:{count}")
    return "; ".join(items)


def _run_dynamics_candidate(
    env: SeedonStandingEnv,
    *,
    mode: str,
    offset: float,
    steps: int,
    seed: int,
    right_joint_index: int,
    left_joint_index: int,
    right_actuator_id: int,
    left_actuator_id: int,
) -> HipRollDynamicsResult:
    """Track one fixed hip-roll target and collect dynamics diagnostics."""
    env.reset(seed=seed)
    baseline_joints = env._joint_positions().copy()
    target = baseline_joints.copy()
    target_right, target_left = _resolve_joint_targets(
        mode=mode,
        offset=offset,
        baseline_right=float(target[right_joint_index]),
        baseline_left=float(target[left_joint_index]),
    )
    target[right_joint_index] = target_right
    target[left_joint_index] = target_left

    initial_base_y = float(env.data.qpos[1])
    initial_com_y = float(_overall_com(env)[1])
    support_side = _mode_support_side(mode, offset)
    support_contact_steps = 0
    swing_contact_steps = 0
    terminated = False
    step_count = 0
    final_base_z = float("nan")
    final_upright = float("nan")
    max_right_ctrl = 0.0
    max_left_ctrl = 0.0
    max_overall_ctrl = 0.0
    right_saturated_steps = 0
    left_saturated_steps = 0
    pair_counter: Counter[tuple[str, str]] = Counter()
    ctrl_lower = env.model.actuator_ctrlrange[:, 0]
    ctrl_upper = env.model.actuator_ctrlrange[:, 1]

    for step_index in range(steps):
        env._do_pd_simulation(target)
        obs = env._get_obs()
        final_base_z = env._base_height()
        final_upright = env._base_upright()
        terminated = env._is_terminated(final_base_z, final_upright, obs)
        flags = _floor_contact_flags(env)
        support_contact, swing_contact = _support_and_swing_contact(flags, support_side)
        support_contact_steps += int(support_contact)
        swing_contact_steps += int(swing_contact)

        right_ctrl = float(env.data.ctrl[right_actuator_id])
        left_ctrl = float(env.data.ctrl[left_actuator_id])
        max_right_ctrl = max(max_right_ctrl, abs(right_ctrl))
        max_left_ctrl = max(max_left_ctrl, abs(left_ctrl))
        max_overall_ctrl = max(max_overall_ctrl, float(np.max(np.abs(env.data.ctrl))))
        right_saturated_steps += int(
            abs(right_ctrl - float(ctrl_lower[right_actuator_id])) <= 1e-6
            or abs(right_ctrl - float(ctrl_upper[right_actuator_id])) <= 1e-6
        )
        left_saturated_steps += int(
            abs(left_ctrl - float(ctrl_lower[left_actuator_id])) <= 1e-6
            or abs(left_ctrl - float(ctrl_upper[left_actuator_id])) <= 1e-6
        )
        for name_a, name_b, _ in contact_pairs(env.model, env.data):
            pair_counter[_normalize_pair(name_a, name_b)] += 1

        step_count = step_index + 1
        if terminated:
            break

    joint_positions = env._joint_positions()
    foot_bottoms = env._foot_bottom_heights()
    return HipRollDynamicsResult(
        mode=mode,
        offset=offset,
        support_side=support_side,
        target_right_hip_roll=target_right,
        target_left_hip_roll=target_left,
        actual_right_hip_roll=float(joint_positions[right_joint_index]),
        actual_left_hip_roll=float(joint_positions[left_joint_index]),
        right_hip_roll_error=float(joint_positions[right_joint_index] - target_right),
        left_hip_roll_error=float(joint_positions[left_joint_index] - target_left),
        base_dy=float(env.data.qpos[1]) - initial_base_y,
        com_dy=float(_overall_com(env)[1]) - initial_com_y,
        base_z=final_base_z,
        upright=final_upright,
        support_contact_ratio=support_contact_steps / max(step_count, 1),
        swing_contact_ratio=swing_contact_steps / max(step_count, 1),
        right_foot_bottom_z=float(foot_bottoms[0]),
        left_foot_bottom_z=float(foot_bottoms[1]),
        right_ctrl_max_abs=max_right_ctrl,
        left_ctrl_max_abs=max_left_ctrl,
        overall_ctrl_max_abs=max_overall_ctrl,
        right_ctrl_saturation_ratio=right_saturated_steps / max(step_count, 1),
        left_ctrl_saturation_ratio=left_saturated_steps / max(step_count, 1),
        contact_pair_summary=_contact_pair_summary(pair_counter),
        terminated=terminated,
        steps=step_count,
    )


def _metadata_rows(metadata: list[HipRollJointMetadata]) -> list[dict[str, object]]:
    """Convert metadata dataclasses to flat CSV rows."""
    return [asdict(item) for item in metadata]


def _kinematic_rows(results: list[HipRollKinematicResult]) -> list[dict[str, object]]:
    """Convert kinematic dataclasses to flat CSV rows."""
    rows: list[dict[str, object]] = []
    for result in results:
        row = {
            "mode": result.mode,
            "offset": result.offset,
            "target_right_hip_roll": result.target_right_hip_roll,
            "target_left_hip_roll": result.target_left_hip_roll,
            "actual_right_hip_roll": result.actual_right_hip_roll,
            "actual_left_hip_roll": result.actual_left_hip_roll,
            "base_y": result.base_y,
            "com_y": result.com_y,
            "right_foot_y": result.right_foot_y,
            "left_foot_y": result.left_foot_y,
        }
        for prefix, position in (
            ("right_knee", result.right_knee_pos),
            ("right_ankle", result.right_ankle_pos),
            ("right_foot", result.right_foot_pos),
            ("left_knee", result.left_knee_pos),
            ("left_ankle", result.left_ankle_pos),
            ("left_foot", result.left_foot_pos),
        ):
            row[f"{prefix}_x"] = position[0]
            row[f"{prefix}_y"] = position[1]
            row[f"{prefix}_z"] = position[2]
        rows.append(row)
    return rows


def _dynamics_rows(results: list[HipRollDynamicsResult]) -> list[dict[str, object]]:
    """Convert dynamics dataclasses to flat CSV rows."""
    return [asdict(item) for item in results]


def _find_result(
    results: list[HipRollKinematicResult] | list[HipRollDynamicsResult],
    *,
    mode: str,
    offset: float,
) -> HipRollKinematicResult | HipRollDynamicsResult | None:
    """Return the first matching result."""
    for result in results:
        if result.mode == mode and math.isclose(result.offset, offset, abs_tol=1e-9):
            return result
    return None


def _vector_mirror_metrics(
    right_position: tuple[float, float, float],
    left_position: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Return x diff, mirrored y sum, and z diff for one mirrored body pair."""
    return (
        abs(right_position[0] - left_position[0]),
        abs(right_position[1] + left_position[1]),
        abs(right_position[2] - left_position[2]),
    )


def _kinematic_mirror_lines(results: list[HipRollKinematicResult]) -> list[str]:
    """Return summary lines for right/left mirrored kinematic candidates."""
    lines: list[str] = []
    symmetric_results = [result for result in results if result.mode == "symmetric"]
    baseline_com_y = (
        float(np.mean([result.com_y for result in symmetric_results]))
        if symmetric_results
        else 0.0
    )
    for magnitude in sorted({abs(result.offset) for result in results if result.mode != "symmetric"}):
        right = _find_result(results, mode="right_only", offset=magnitude)
        left = _find_result(results, mode="left_only", offset=-magnitude)
        if not isinstance(right, HipRollKinematicResult) or not isinstance(left, HipRollKinematicResult):
            continue
        knee_metrics = _vector_mirror_metrics(right.right_knee_pos, left.left_knee_pos)
        ankle_metrics = _vector_mirror_metrics(right.right_ankle_pos, left.left_ankle_pos)
        foot_metrics = _vector_mirror_metrics(right.right_foot_pos, left.left_foot_pos)
        lines.append(
            f"|offset|={magnitude:.2f} "
            f"mirror_com_delta_sum={(right.com_y - baseline_com_y) + (left.com_y - baseline_com_y):+.6f} "
            f"foot_y_sum={right.right_foot_y + left.left_foot_y:+.6f} "
            f"knee(x/y/z)=({knee_metrics[0]:.6f}, {knee_metrics[1]:.6f}, {knee_metrics[2]:.6f}) "
            f"ankle(x/y/z)=({ankle_metrics[0]:.6f}, {ankle_metrics[1]:.6f}, {ankle_metrics[2]:.6f}) "
            f"foot(x/y/z)=({foot_metrics[0]:.6f}, {foot_metrics[1]:.6f}, {foot_metrics[2]:.6f})"
        )
    return lines


def _dynamics_mirror_lines(results: list[HipRollDynamicsResult]) -> list[str]:
    """Return summary lines for right/left mirrored dynamics candidates."""
    lines: list[str] = []
    for magnitude in sorted({abs(result.offset) for result in results if result.mode != "symmetric"}):
        right = _find_result(results, mode="right_only", offset=magnitude)
        left = _find_result(results, mode="left_only", offset=-magnitude)
        if not isinstance(right, HipRollDynamicsResult) or not isinstance(left, HipRollDynamicsResult):
            continue
        lines.append(
            f"|offset|={magnitude:.2f} "
            f"base_dy_sum={right.base_dy + left.base_dy:+.6f} "
            f"com_dy_sum={right.com_dy + left.com_dy:+.6f} "
            f"right_qerr={abs(right.right_hip_roll_error):.6f} "
            f"left_qerr={abs(left.left_hip_roll_error):.6f} "
            f"right_sat={right.right_ctrl_saturation_ratio:.2f} "
            f"left_sat={left.left_ctrl_saturation_ratio:.2f}"
        )
    return lines


def _structural_difference_lines(
    metadata: list[HipRollJointMetadata],
    foot_collision_summary: dict[str, str],
) -> list[str]:
    """Return concrete structural differences or confirm symmetry."""
    right = next(item for item in metadata if item.joint_name == RIGHT_HIP_ROLL_NAME)
    left = next(item for item in metadata if item.joint_name == LEFT_HIP_ROLL_NAME)
    lines: list[str] = []

    if right.urdf_axis != left.urdf_axis or right.mujoco_axis != left.mujoco_axis:
        lines.append(
            f"axis mismatch: URDF {right.urdf_axis} vs {left.urdf_axis}, "
            f"MuJoCo {right.mujoco_axis} vs {left.mujoco_axis}"
        )
    else:
        lines.append(f"axis sign: both sides use the same local axis ({right.urdf_axis}).")

    if right.urdf_origin_xyz != left.urdf_origin_xyz or right.mujoco_joint_frame_pos != left.mujoco_joint_frame_pos:
        lines.append(
            f"joint origin mismatch: URDF {right.urdf_origin_xyz} vs {left.urdf_origin_xyz}, "
            f"MuJoCo {right.mujoco_joint_frame_pos} vs {left.mujoco_joint_frame_pos}"
        )
    else:
        lines.append(f"joint origin: matched on both sides ({right.urdf_origin_xyz}).")

    right_parent_child = (right.parent_body, right.child_body)
    left_parent_child = (left.parent_body, left.child_body)
    normalized_right = tuple(name.replace("R_", "{side}_") for name in right_parent_child)
    normalized_left = tuple(name.replace("L_", "{side}_") for name in left_parent_child)
    if normalized_right != normalized_left:
        lines.append(
            f"parent/child mismatch: right {right_parent_child} vs left {left_parent_child}"
        )
    else:
        lines.append(
            f"parent/child chain: mirrored naming only, right {right_parent_child}, left {left_parent_child}."
        )

    if foot_collision_summary["mirrored_local_position"] != "True" or foot_collision_summary["mirrored_size"] != "True":
        lines.append(
            "foot collision mismatch: "
            f"right pos {foot_collision_summary['right_local_pos']} size {foot_collision_summary['right_size']} "
            f"vs left pos {foot_collision_summary['left_local_pos']} size {foot_collision_summary['left_size']}"
        )
    else:
        lines.append(
            "foot collision: local position is mirrored on Y and box size matches left/right."
        )

    return lines


def _write_report(
    path: Path,
    *,
    metadata: list[HipRollJointMetadata],
    foot_collision_summary: dict[str, str],
    kinematic_results: list[HipRollKinematicResult],
    dynamics_results: list[HipRollDynamicsResult],
    kinematic_mirror_lines: list[str],
    dynamics_mirror_lines: list[str],
) -> None:
    """Write a markdown report summarizing the validation result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    structural_lines = _structural_difference_lines(metadata, foot_collision_summary)
    lines: list[str] = []
    lines.append("# Seedon Hip-Roll Model Validation")
    lines.append("")
    lines.append("## Joint Metadata")
    lines.append("")
    lines.append("| Joint | URDF axis | MuJoCo axis | URDF range | MuJoCo range | URDF origin | MuJoCo frame pos | qpos adr | Parent | Child |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | ---: | --- | --- |")
    for item in metadata:
        lines.append(
            f"| {item.joint_name} | `{item.urdf_axis}` | `{item.mujoco_axis}` | "
            f"`{item.urdf_range}` | `{item.mujoco_range}` | `{item.urdf_origin_xyz}` | "
            f"`{item.mujoco_joint_frame_pos}` | {item.qpos_address} | `{item.parent_body}` | `{item.child_body}` |"
        )
    lines.append("")
    lines.append("Hip-roll link visual/collision origins:")
    for item in metadata:
        lines.append(
            f"- `{item.joint_name}` visual `{item.hip_roll_visual_origin}`, collision `{item.hip_roll_collision_origin}`"
        )
    lines.append("")
    lines.append("Foot collision mirror check:")
    lines.append(
        f"- right local pos `{foot_collision_summary['right_local_pos']}`, left local pos `{foot_collision_summary['left_local_pos']}`"
    )
    lines.append(
        f"- right size `{foot_collision_summary['right_size']}`, left size `{foot_collision_summary['left_size']}`"
    )
    lines.append(
        f"- mirrored local position `{foot_collision_summary['mirrored_local_position']}`, mirrored size `{foot_collision_summary['mirrored_size']}`"
    )
    lines.append("")
    lines.append("Concrete structural differences:")
    for line in structural_lines:
        lines.append(f"- {line}")

    lines.append("")
    lines.append("## Kinematic-Only Sweep")
    lines.append("")
    lines.append("Mirror summary (`right_only +a` vs `left_only -a`):")
    for line in kinematic_mirror_lines:
        lines.append(f"- {line}")

    lines.append("")
    lines.append("Representative kinematic rows:")
    lines.append("")
    lines.append("| mode | offset | target_R | target_L | actual_R | actual_L | com_y | right_foot_y | left_foot_y |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for item in kinematic_results:
        lines.append(
            f"| {item.mode} | {item.offset:+.2f} | {item.target_right_hip_roll:+.3f} | {item.target_left_hip_roll:+.3f} | "
            f"{item.actual_right_hip_roll:+.3f} | {item.actual_left_hip_roll:+.3f} | {item.com_y:+.6f} | "
            f"{item.right_foot_y:+.6f} | {item.left_foot_y:+.6f} |"
        )

    lines.append("")
    lines.append("## Dynamics Sweep")
    lines.append("")
    lines.append("Mirror summary (`right_only +a` vs `left_only -a`):")
    for line in dynamics_mirror_lines:
        lines.append(f"- {line}")

    lines.append("")
    lines.append("| mode | offset | target_R | target_L | actual_R | actual_L | err_R | err_L | base_dy | com_dy | ctrl_R | ctrl_L | sat_R | sat_L | terminated | contact pairs |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |")
    for item in dynamics_results:
        lines.append(
            f"| {item.mode} | {item.offset:+.2f} | {item.target_right_hip_roll:+.3f} | {item.target_left_hip_roll:+.3f} | "
            f"{item.actual_right_hip_roll:+.3f} | {item.actual_left_hip_roll:+.3f} | "
            f"{item.right_hip_roll_error:+.3f} | {item.left_hip_roll_error:+.3f} | "
            f"{item.base_dy:+.5f} | {item.com_dy:+.5f} | {item.right_ctrl_max_abs:.2f} | {item.left_ctrl_max_abs:.2f} | "
            f"{item.right_ctrl_saturation_ratio:.2f} | {item.left_ctrl_saturation_ratio:.2f} | {item.terminated} | `{item.contact_pair_summary}` |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--offsets",
        type=_parse_offsets,
        default=[-0.3, -0.2, -0.1, 0.1, 0.2, 0.3],
        help="Comma-separated hip-roll offsets in radians.",
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the hip-roll model validation."""
    args = build_parser().parse_args(argv)
    if args.steps <= 0:
        raise ValueError("--steps must be positive.")

    reward_config = load_seedon_config_from_env()
    urdf_root = _load_xml_root(URDF_PATH)
    out_dir = args.out_dir.resolve()
    report_path = out_dir / DEFAULT_REPORT_PATH.name
    metadata_csv = out_dir / DEFAULT_METADATA_CSV.name
    kinematic_csv = out_dir / DEFAULT_KINEMATIC_CSV.name
    dynamics_csv = out_dir / DEFAULT_DYNAMICS_CSV.name

    env = SeedonStandingEnv(reset_noise_scale=0.0, reward_config=reward_config)
    metadata: list[HipRollJointMetadata] = []
    kinematic_results: list[HipRollKinematicResult] = []
    dynamics_results: list[HipRollDynamicsResult] = []
    try:
        env.reset(seed=args.seed)
        metadata = [
            _joint_metadata(env, urdf_root, RIGHT_HIP_ROLL_NAME),
            _joint_metadata(env, urdf_root, LEFT_HIP_ROLL_NAME),
        ]
        foot_collision_summary = _foot_collision_mirror_summary(env)
        baseline_qpos = env.data.qpos.copy()
        baseline_qvel = np.zeros_like(env.data.qvel)
        right_joint_id = env._joint_id(RIGHT_HIP_ROLL_NAME)
        left_joint_id = env._joint_id(LEFT_HIP_ROLL_NAME)
        right_qpos_adr = int(env.model.jnt_qposadr[right_joint_id])
        left_qpos_adr = int(env.model.jnt_qposadr[left_joint_id])
        right_joint_index = env._joint_ids.index(right_joint_id)
        left_joint_index = env._joint_ids.index(left_joint_id)
        right_actuator_id, left_actuator_id = _hip_roll_actuator_ids(env)

        for mode in MODES:
            for offset in args.offsets:
                kinematic_results.append(
                    _kinematic_result(
                        env,
                        mode=mode,
                        offset=offset,
                        baseline_qpos=baseline_qpos,
                        baseline_qvel=baseline_qvel,
                        right_qpos_adr=right_qpos_adr,
                        left_qpos_adr=left_qpos_adr,
                    )
                )

        for mode in MODES:
            for offset in args.offsets:
                dynamics_results.append(
                    _run_dynamics_candidate(
                        env,
                        mode=mode,
                        offset=offset,
                        steps=args.steps,
                        seed=args.seed,
                        right_joint_index=right_joint_index,
                        left_joint_index=left_joint_index,
                        right_actuator_id=right_actuator_id,
                        left_actuator_id=left_actuator_id,
                    )
                )
    finally:
        env.close()

    _write_csv(metadata_csv, _metadata_rows(metadata))
    _write_csv(kinematic_csv, _kinematic_rows(kinematic_results))
    _write_csv(dynamics_csv, _dynamics_rows(dynamics_results))
    kinematic_mirror_lines = _kinematic_mirror_lines(kinematic_results)
    dynamics_mirror_lines = _dynamics_mirror_lines(dynamics_results)
    _write_report(
        report_path,
        metadata=metadata,
        foot_collision_summary=foot_collision_summary,
        kinematic_results=kinematic_results,
        dynamics_results=dynamics_results,
        kinematic_mirror_lines=kinematic_mirror_lines,
        dynamics_mirror_lines=dynamics_mirror_lines,
    )

    print(f"report: {report_path}")
    print(f"metadata_csv: {metadata_csv}")
    print(f"kinematic_csv: {kinematic_csv}")
    print(f"dynamics_csv: {dynamics_csv}")
    print("\nkinematic mirror summary")
    for line in kinematic_mirror_lines:
        print(line)
    print("\ndynamics mirror summary")
    for line in dynamics_mirror_lines:
        print(line)
    print("\nstructural summary")
    for line in _structural_difference_lines(metadata, foot_collision_summary):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
