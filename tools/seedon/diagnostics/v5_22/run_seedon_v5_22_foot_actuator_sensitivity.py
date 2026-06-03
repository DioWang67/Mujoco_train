"""Run Seedon v5_22 foot x actuator/controller bounded sensitivity.

This Class C diagnostic creates artifact-only v5_22 foot contact variants and
combines them with bounded actuator profiles. It does not modify the source
v5_22 XML/URDF, train.py, eval.py, env.py runtime behavior, and does not run
PPO or claim walking success.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import mujoco
except ModuleNotFoundError as exc:  # pragma: no cover - only when MuJoCo is unavailable.
    mujoco = None
    _MUJOCO_IMPORT_ERROR: ModuleNotFoundError | None = exc
else:
    _MUJOCO_IMPORT_ERROR = None


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "seedon" / "seedon_v5_22_foot_actuator_sensitivity.yaml"
PATCH_METADATA = {
    "source": "assumption/prototype",
    "confidence": "low",
    "valid_for": "simulation_prototype_only",
}
SIDES = ("left", "right")
REGIONS = ("center", "toe", "heel", "inner", "outer", "unknown")
SCENARIOS = {
    "neutral": (0.0, 0.0),
    "forward_pitch": (0.0, math.radians(8.0)),
    "backward_pitch": (0.0, math.radians(-8.0)),
}
FOOT_SPECS = {
    "right": {"body": "R_link_ankle_pitch", "original_geom": "R_foot_collision", "prefix": "R"},
    "left": {"body": "L_link_ankle_pitch", "original_geom": "L_foot_collision", "prefix": "L"},
}
FORCE_EPS = 1e-5


@dataclass(frozen=True)
class FootGeom:
    """Resolved source foot collision geometry.

    Args:
        side: Semantic side label.
        body_name: Body containing the foot collision geom.
        geom_name: Source collision geom name.
        pos: Local geom position.
        size: MuJoCo box half-size.
        friction: MJCF friction triplet.
    """

    side: str
    body_name: str
    geom_name: str
    pos: tuple[float, float, float]
    size: tuple[float, float, float]
    friction: str

    @property
    def length(self) -> float:
        """Return full local x length."""

        return 2.0 * self.size[0]

    @property
    def width(self) -> float:
        """Return full local y width."""

        return 2.0 * self.size[1]

    @property
    def bottom_z(self) -> float:
        """Return local bottom z."""

        return self.pos[2] - self.size[2]


@dataclass(frozen=True)
class Variant:
    """Artifact-only v5_22 foot variant."""

    name: str
    scene_path: Path
    added_patches: list[dict[str, Any]]
    status: str


def rel_path(path: Path) -> str:
    """Return repository-relative path text when possible."""

    try:
        return str(path.resolve().relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def resolve_repo_path(raw_path: str) -> Path:
    """Resolve a repo-relative path."""

    path = Path(raw_path)
    return path if path.is_absolute() else REPO_ROOT / path


def load_json_object(path: Path, label: str) -> dict[str, Any]:
    """Load a JSON-compatible object file.

    Args:
        path: File path.
        label: Human-readable label used in errors.

    Returns:
        Parsed JSON object.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the payload is not an object.
    """

    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must decode to a JSON object: {path}")
    return payload


def require_mujoco() -> None:
    """Raise if MuJoCo cannot be imported."""

    if mujoco is None:
        raise RuntimeError(f"MuJoCo is not importable: {_MUJOCO_IMPORT_ERROR}")


def parse_vec3(raw_value: str, field_name: str) -> tuple[float, float, float]:
    """Parse an MJCF vec3 field."""

    parts = raw_value.replace(",", " ").split()
    if len(parts) != 3:
        raise ValueError(f"{field_name} must contain exactly three numeric values.")
    return (float(parts[0]), float(parts[1]), float(parts[2]))


def fmt_vec(values: tuple[float, float, float]) -> str:
    """Format a compact MJCF vector."""

    return " ".join(f"{value:.9g}" for value in values)


def find_required(root: ET.Element, xpath: str, label: str) -> ET.Element:
    """Find a required XML element."""

    node = root.find(xpath)
    if node is None:
        raise ValueError(f"Missing required {label}: {xpath}")
    return node


def resolve_source_feet(root: ET.Element) -> dict[str, FootGeom]:
    """Resolve v5_22 source foot collision boxes."""

    feet: dict[str, FootGeom] = {}
    for side, spec in FOOT_SPECS.items():
        body = find_required(root, f".//body[@name='{spec['body']}']", spec["body"])
        geom = body.find(f"geom[@name='{spec['original_geom']}']")
        if geom is None:
            raise ValueError(f"Missing source foot geom: {spec['original_geom']}")
        if geom.attrib.get("type") != "box":
            raise ValueError(f"{spec['original_geom']} must be a box for normalized patch placement.")
        pos = parse_vec3(geom.attrib.get("pos", ""), f"{spec['original_geom']}.pos")
        size = parse_vec3(geom.attrib.get("size", ""), f"{spec['original_geom']}.size")
        if min(size) <= 0.0:
            raise ValueError(f"{spec['original_geom']} size must be positive.")
        feet[side] = FootGeom(
            side=side,
            body_name=spec["body"],
            geom_name=spec["original_geom"],
            pos=pos,
            size=size,
            friction=geom.attrib.get("friction", "1.0 0.005 0.0001"),
        )
    return feet


def foot_length_status(feet: dict[str, FootGeom]) -> tuple[bool, str]:
    """Return whether foot box dimensions are usable for prototype patches."""

    lengths = [foot.length for foot in feet.values()]
    widths = [foot.width for foot in feet.values()]
    if len(lengths) != 2 or any(value <= 0.0 for value in lengths + widths):
        return False, "manual_required: missing or non-positive v5_22 foot size."
    relative_delta = abs(lengths[0] - lengths[1]) / max(lengths)
    if relative_delta > 0.05:
        return False, "manual_required: left/right v5_22 foot length mismatch exceeds 5%."
    return True, "resolved_from_v5_22_seedon_foot_collision_box_size"


def patch_position(foot: FootGeom, patch: dict[str, Any]) -> tuple[float, float, float]:
    """Compute local patch center from normalized ratios."""

    normalized_x = float(patch["normalized_x"])
    normalized_y = float(patch.get("normalized_y", 0.0))
    size_scale = patch["size_scale"]
    half_z = max(0.003, foot.size[2] * float(size_scale[2]))
    return (
        foot.pos[0] + normalized_x * foot.size[0],
        foot.pos[1] + normalized_y * foot.size[1],
        foot.bottom_z + half_z,
    )


def patch_size(foot: FootGeom, patch: dict[str, Any]) -> tuple[float, float, float]:
    """Compute prototype patch half-size from v5_22 source foot size."""

    scale = patch["size_scale"]
    if not isinstance(scale, list) or len(scale) != 3:
        raise ValueError("patch size_scale must contain three values.")
    return (
        max(0.004, foot.size[0] * float(scale[0])),
        max(0.004, foot.size[1] * float(scale[1])),
        max(0.003, foot.size[2] * float(scale[2])),
    )


def mark_original_foot_non_contact(root: ET.Element) -> None:
    """Disable original foot collision contacts in generated variants."""

    for spec in FOOT_SPECS.values():
        geom = find_required(root, f".//geom[@name='{spec['original_geom']}']", spec["original_geom"])
        geom.set("contype", "0")
        geom.set("conaffinity", "0")
        geom.set("group", "1")
        geom.set("rgba", "0.12 0.12 0.12 0.12")


def add_profile_patches(root: ET.Element, profile: dict[str, Any], feet: dict[str, FootGeom]) -> list[dict[str, Any]]:
    """Add low-confidence prototype contact patches to one v5_22 variant."""

    added: list[dict[str, Any]] = []
    for side, foot in feet.items():
        body = find_required(root, f".//body[@name='{foot.body_name}']", foot.body_name)
        for patch in profile["patches"]:
            region = str(patch["region"])
            geom_name = f"{FOOT_SPECS[side]['prefix']}_foot_contact_{profile['name']}_{region}"
            pos = patch_position(foot, patch)
            size = patch_size(foot, patch)
            ET.SubElement(
                body,
                "geom",
                {
                    "name": geom_name,
                    "type": "box",
                    "pos": fmt_vec(pos),
                    "size": fmt_vec(size),
                    "rgba": "0.1 0.5 0.9 0.35",
                    "friction": foot.friction,
                    "contype": "1",
                    "conaffinity": "1",
                    "group": "3",
                    "user": "1",
                },
            )
            added.append(
                {
                    "side": side,
                    "region": region,
                    "geom_name": geom_name,
                    "body_name": foot.body_name,
                    "pos": fmt_vec(pos),
                    "size": fmt_vec(size),
                    **PATCH_METADATA,
                }
            )
    return added


def copy_mesh_assets(source_scene: Path, variant_dir: Path) -> None:
    """Copy mesh assets needed by artifact-only variants."""

    source_mesh_dir = source_scene.parent / "mjcf_source"
    if not source_mesh_dir.is_dir():
        return
    target_mesh_dir = variant_dir / "mjcf_source"
    target_mesh_dir.mkdir(parents=True, exist_ok=True)
    for source_file in source_mesh_dir.iterdir():
        if source_file.is_file():
            shutil.copy2(source_file, target_mesh_dir / source_file.name)


def create_v5_22_variants(config: dict[str, Any]) -> list[Variant]:
    """Create artifact-only v5_22 foot contact variants."""

    source_scene = resolve_repo_path(config["model_path"])
    foot_profile = load_json_object(resolve_repo_path(config["foot_profile_path"]), "foot profile")
    requested_names = set(config["foot_variants"])
    profiles = [item for item in foot_profile["profiles"] if item["name"] in requested_names]
    if len(profiles) != len(requested_names):
        found = {item["name"] for item in profiles}
        raise ValueError(f"Missing requested foot profiles: {sorted(requested_names - found)}")

    variants_dir = resolve_repo_path(config["artifacts_dir"]) / "variants"
    variants: list[Variant] = []
    for profile in profiles:
        tree = ET.parse(source_scene)
        root = tree.getroot()
        feet = resolve_source_feet(root)
        foot_length_ok, status = foot_length_status(feet)
        added: list[dict[str, Any]] = []
        if profile.get("disable_original_foot_collision"):
            mark_original_foot_non_contact(root)
        if profile.get("patches"):
            if not foot_length_ok and profile.get("manual_required_if_foot_length_unknown"):
                status = f"manual_required: {status}"
            else:
                added = add_profile_patches(root, profile, feet)

        variant_dir = variants_dir / profile["name"]
        variant_dir.mkdir(parents=True, exist_ok=True)
        copy_mesh_assets(source_scene, variant_dir)
        scene_path = variant_dir / f"training_scene_{profile['name']}.xml"
        tree.write(scene_path, encoding="utf-8", xml_declaration=True)
        require_mujoco()
        mujoco.MjModel.from_xml_path(str(scene_path))
        variants.append(Variant(name=profile["name"], scene_path=scene_path, added_patches=added, status=status))
    write_json(
        variants_dir / "manifest.json",
        {
            "schema_version": 1,
            "source_scene_path": rel_path(source_scene),
            "prototype_metadata": PATCH_METADATA,
            "duck_usage": "concept_reference_only_not_verified_geometry",
            "variants": [
                {
                    "name": variant.name,
                    "scene_path": rel_path(variant.scene_path),
                    "status": variant.status,
                    "added_patches": variant.added_patches,
                    "does_not_modify_source_scene": True,
                }
                for variant in variants
            ],
        },
    )
    return variants


def model_name(model: Any, obj_type: Any, index: int) -> str:
    """Return a MuJoCo object name."""

    return mujoco.mj_id2name(model, obj_type, int(index)) or f"<unnamed:{index}>"


def quat_from_roll_pitch(roll: float, pitch: float) -> np.ndarray:
    """Return a wxyz quaternion for diagnostic base pose perturbations."""

    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    return np.array([cp * cr, cp * sr, sp * cr, -sp * sr], dtype=np.float64)


def side_for_geom(name: str) -> str:
    """Classify side from a geom name."""

    lowered = name.lower()
    if lowered.startswith("r_") or lowered.startswith("right"):
        return "right"
    if lowered.startswith("l_") or lowered.startswith("left"):
        return "left"
    return "unknown"


def region_for_geom(name: str) -> str:
    """Classify prototype contact region from geom name."""

    lowered = name.lower()
    tokens = lowered.split("_")
    if tokens and tokens[-1] in {"center", "toe", "heel", "inner", "outer"}:
        return tokens[-1]
    if lowered in {"r_foot_collision", "l_foot_collision"}:
        return "center"
    if "foot" in lowered and "collision" in lowered:
        return "center"
    return "unknown"


def is_floor_name(name: str) -> bool:
    """Return whether a geom is floor-like."""

    lowered = name.lower()
    return lowered in {"floor", "ground"} or "floor" in lowered or "ground" in lowered


def is_foot_name(name: str) -> bool:
    """Return whether a geom is foot-like."""

    lowered = name.lower()
    return any(token in lowered for token in ("foot", "toe", "heel", "sole", "ankle"))


def contact_force(model: Any, data: Any, index: int) -> float:
    """Return absolute normal force for a contact."""

    wrench = np.zeros(6, dtype=np.float64)
    mujoco.mj_contactForce(model, data, index, wrench)
    return abs(float(wrench[0]))


def collect_contacts(model: Any, data: Any, *, variant: str, profile: str, phase: str, step: int) -> list[dict[str, Any]]:
    """Collect raw contact rows for the current data state."""

    rows: list[dict[str, Any]] = []
    for contact_index in range(int(data.ncon)):
        contact = data.contact[contact_index]
        geom1 = model_name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1))
        geom2 = model_name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2))
        floor_contact = is_floor_name(geom1) or is_floor_name(geom2)
        side, region = classify_contact_pair(geom1, geom2) if floor_contact else ("unknown", "unknown")
        rows.append(
            {
                "foot_variant": variant,
                "actuator_profile": profile,
                "phase": phase,
                "step": step,
                "contact_index": contact_index,
                "geom1": geom1,
                "geom2": geom2,
                "dist": float(contact.dist),
                "pos_x": float(contact.pos[0]),
                "pos_y": float(contact.pos[1]),
                "pos_z": float(contact.pos[2]),
                "normal_force": contact_force(model, data, contact_index),
                "is_floor_contact": bool(floor_contact),
                "involves_foot": is_foot_name(geom1) or is_foot_name(geom2),
                "classified_side": side,
                "classified_region": region,
            }
        )
    return rows


def classify_contact_pair(name1: str, name2: str) -> tuple[str, str]:
    """Return side and region for a floor/foot contact pair."""

    for name in (name1, name2):
        if is_foot_name(name):
            return side_for_geom(name), region_for_geom(name)
    return "unknown", "unknown"


def reset_and_apply_pose(model: Any, data: Any, *, roll: float, pitch: float, settle_steps: int) -> None:
    """Reset, settle with zero ctrl, then apply a diagnostic base pose."""

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    for _ in range(max(0, settle_steps)):
        if data.ctrl.size:
            data.ctrl[:] = 0.0
        mujoco.mj_step(model, data)
    data.qpos[3:7] = quat_from_roll_pitch(roll, pitch)
    data.qvel[:] = 0.0
    if data.ctrl.size:
        data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)
    mujoco.mj_step(model, data)


def scenario_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize force by side and region for one contact scenario."""

    summary: dict[str, Any] = {f"{side}_{region}_force": 0.0 for side in SIDES for region in REGIONS}
    summary.update({"left_contact_count": 0, "right_contact_count": 0})
    for row in rows:
        if not row["is_floor_contact"]:
            continue
        side = row["classified_side"]
        region = row["classified_region"]
        if side in SIDES and region in REGIONS:
            summary[f"{side}_{region}_force"] += float(row["normal_force"])
            summary[f"{side}_contact_count"] += 1
    return summary


def patch_inventory(model: Any, variant: Variant) -> list[dict[str, Any]]:
    """Return foot-related geoms and prototype metadata."""

    manifest_by_name = {item["geom_name"]: item for item in variant.added_patches}
    rows: list[dict[str, Any]] = []
    for geom_id in range(model.ngeom):
        name = model_name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if "foot" not in name.lower():
            continue
        manifest_item = manifest_by_name.get(name, {})
        rows.append(
            {
                "geom_name": name,
                "side": side_for_geom(name),
                "region": region_for_geom(name),
                "contype": int(model.geom_contype[geom_id]),
                "conaffinity": int(model.geom_conaffinity[geom_id]),
                "source": manifest_item.get("source", "v5_22_source_scene"),
                "confidence": manifest_item.get("confidence", "source_scene"),
                "valid_for": manifest_item.get("valid_for", "existing_seedon_contact"),
            }
        )
    return rows


def can_classify_center_toe_heel(inventory: list[dict[str, Any]]) -> bool:
    """Return whether both feet expose center/toe/heel contact geoms."""

    regions_by_side: dict[str, set[str]] = defaultdict(set)
    for row in inventory:
        if int(row["contype"]) == 0 or int(row["conaffinity"]) == 0:
            continue
        side = str(row["side"])
        if side in SIDES:
            regions_by_side[side].add(str(row["region"]))
    return all({"center", "toe", "heel"}.issubset(regions_by_side[side]) for side in SIDES)


def force(summary: dict[str, Any], side: str, region: str) -> float:
    """Return one force value from scenario summary."""

    return float(summary.get(f"{side}_{region}_force", 0.0))


def has_force(summary: dict[str, Any], side: str, region: str) -> bool:
    """Return whether one side/region has non-trivial force."""

    return force(summary, side, region) > FORCE_EPS


def center_first(summary: dict[str, Any], side: str, can_classify: bool) -> bool | str:
    """Return center-first metric or unavailable when classification is impossible."""

    if not can_classify:
        return "unavailable"
    center = force(summary, side, "center")
    toe = force(summary, side, "toe")
    heel = force(summary, side, "heel")
    return bool(center > FORCE_EPS and center >= toe and center >= heel)


def toe_handoff(neutral: dict[str, Any], forward: dict[str, Any], side: str, can_classify: bool) -> bool | str:
    """Return toe handoff candidate metric or unavailable."""

    if not can_classify:
        return "unavailable"
    return bool(
        has_force(neutral, side, "center")
        and has_force(forward, side, "toe")
        and force(forward, side, "toe") >= force(forward, side, "center") * 0.75
    )


def bridge_detected(summaries: dict[str, dict[str, Any]], side: str, can_classify: bool) -> bool | str:
    """Return whether toe and heel bridge in the same contact scenario."""

    if not can_classify:
        return "unavailable"
    return any(has_force(row, side, "toe") and has_force(row, side, "heel") for row in summaries.values())


def base_euler(data: Any) -> tuple[float, float, float]:
    """Return base roll/pitch/yaw from freejoint quaternion."""

    if len(data.qpos) < 7:
        return (0.0, 0.0, 0.0)
    quat = np.array(data.qpos[3:7], dtype=float)
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-12:
        return (0.0, 0.0, 0.0)
    w, x, y, z = quat / norm
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sinp = 2.0 * (w * y - z * x)
    pitch = math.asin(float(np.clip(sinp, -1.0, 1.0)))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return (float(roll), float(pitch), float(yaw))


def angle_delta(value: float, reference: float) -> float:
    """Return wrapped angle delta."""

    return float(math.atan2(math.sin(value - reference), math.cos(value - reference)))


def phase_state(data: Any) -> dict[str, float]:
    """Capture base state."""

    roll, pitch, yaw = base_euler(data)
    return {
        "base_height": float(data.qpos[2]) if len(data.qpos) >= 3 else 0.0,
        "roll": roll,
        "pitch": pitch,
        "yaw": yaw,
    }


def classify_joint_group(joint_name: str) -> str:
    """Map joint to actuator profile group."""

    lowered = joint_name.lower()
    if "hip_pitch" in lowered:
        return "hip_pitch"
    if "ankle_pitch" in lowered:
        return "ankle_pitch"
    return "other_leg_joints"


def profile_limits_for_model(model: Any, profile: dict[str, Any]) -> np.ndarray:
    """Return per-actuator bounded diagnostic command limits."""

    limits = np.zeros(model.nu, dtype=np.float64)
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        joint_name = model_name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) if joint_id >= 0 else ""
        limits[actuator_id] = float(profile[classify_joint_group(joint_name)])
    return limits


def run_pd_hold_with_profile(
    model: Any,
    *,
    profile: dict[str, Any],
    profile_name: str,
    variant_name: str,
    config: dict[str, Any],
    forward_probe: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run bounded PD hold under one actuator profile."""

    sim_config = config["simulation"]
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    if forward_probe:
        data.qpos[3:7] = quat_from_roll_pitch(0.0, math.radians(float(sim_config["forward_probe_pitch_degrees"])))
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
    target_qpos = data.qpos.copy()
    initial = phase_state(data)
    actuator_joint_ids = [int(model.actuator_trnid[index, 0]) for index in range(model.nu)]
    ctrlrange = np.array(model.actuator_ctrlrange, dtype=np.float64)
    profile_limits = profile_limits_for_model(model, profile)
    steps = int(sim_config["forward_probe_steps"] if forward_probe else sim_config["pd_hold_steps"])
    kp = float(sim_config["pd_stiffness"])
    kd = float(sim_config["pd_damping"])
    total_ctrl_values = 0
    saturation_count = 0
    ankle_saturation_count = 0
    ankle_total = 0
    clamp_count = 0
    no_contact_steps = 0
    tilt_max = 0.0
    contacts: list[dict[str, Any]] = []
    exploded = False
    for step in range(steps):
        ctrl = np.zeros(model.nu, dtype=np.float64)
        for actuator_id, joint_id in enumerate(actuator_joint_ids):
            if joint_id < 0:
                continue
            qpos_adr = int(model.jnt_qposadr[joint_id])
            dof_adr = int(model.jnt_dofadr[joint_id])
            raw = kp * (float(target_qpos[qpos_adr]) - float(data.qpos[qpos_adr])) - kd * float(data.qvel[dof_adr])
            mjcf_clipped = float(np.clip(raw, ctrlrange[actuator_id, 0], ctrlrange[actuator_id, 1]))
            bounded = float(np.clip(mjcf_clipped, -profile_limits[actuator_id], profile_limits[actuator_id]))
            if abs(bounded - raw) > 1e-9:
                clamp_count += 1
            ctrl[actuator_id] = bounded
        data.ctrl[:] = ctrl
        saturated = np.abs(ctrl) >= profile_limits - 1e-9
        saturation_count += int(np.count_nonzero(saturated))
        total_ctrl_values += int(model.nu)
        for actuator_id, joint_id in enumerate(actuator_joint_ids):
            joint_name = model_name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) if joint_id >= 0 else ""
            if "ankle_pitch" not in joint_name.lower():
                continue
            ankle_total += 1
            ankle_saturation_count += int(bool(saturated[actuator_id]))
        try:
            mujoco.mj_step(model, data)
        except Exception:
            exploded = True
            break
        if int(data.ncon) == 0:
            no_contact_steps += 1
        roll, pitch, _ = base_euler(data)
        tilt_max = max(tilt_max, abs(roll), abs(pitch))
        contacts.extend(
            collect_contacts(
                model,
                data,
                variant=variant_name,
                profile=profile_name,
                phase="forward_capture_probe" if forward_probe else "nominal_pd_hold",
                step=step + 1,
            )
        )
        if not np.all(np.isfinite(data.qpos)) or float(np.max(np.abs(data.qpos))) > float(sim_config["max_qpos_abs"]):
            exploded = True
            break
    final = phase_state(data)
    large_tilt_rad = math.radians(float(sim_config["large_tilt_degrees"]))
    fall_or_large_tilt = bool(abs(final["roll"]) > large_tilt_rad or abs(final["pitch"]) > large_tilt_rad)
    summary = {
        "steps_run": step + 1 if "step" in locals() else 0,
        "base_height_drift": float(final["base_height"] - initial["base_height"]),
        "roll_drift": angle_delta(final["roll"], initial["roll"]),
        "pitch_drift": angle_delta(final["pitch"], initial["pitch"]),
        "yaw_drift": angle_delta(final["yaw"], initial["yaw"]),
        "tilt_max": float(tilt_max),
        "contact_none_rate": float(no_contact_steps / max(step + 1 if "step" in locals() else 0, 1)),
        "fall_or_large_tilt": fall_or_large_tilt,
        "exploded": bool(exploded),
        "unstable_or_exploding": bool(
            exploded
            or fall_or_large_tilt
            or abs(final["base_height"] - initial["base_height"]) > float(sim_config["max_base_height_drift"])
        ),
        "actuator_saturation_rate": float(saturation_count / max(total_ctrl_values, 1)),
        "ankle_pitch_saturation_rate": float(ankle_saturation_count / max(ankle_total, 1)),
        "joint_target_clamp_rate": float(clamp_count / max(total_ctrl_values, 1)),
        "profile_valid_for": profile["valid_for"],
        "profile_source": profile["source"],
        "profile_confidence": profile["confidence"],
        "torque_side": profile["torque_side"],
    }
    return summary, contacts


def run_contact_scenarios(
    model: Any,
    variant: Variant,
    profile_name: str,
    config: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Run neutral, forward pitch, and backward pitch contact checks."""

    data = mujoco.MjData(model)
    summaries: dict[str, dict[str, Any]] = {}
    raw_contacts: list[dict[str, Any]] = []
    settle_steps = int(config["simulation"]["settle_steps"])
    for scenario_name, (roll, pitch) in SCENARIOS.items():
        reset_and_apply_pose(model, data, roll=roll, pitch=pitch, settle_steps=settle_steps)
        rows = collect_contacts(model, data, variant=variant.name, profile=profile_name, phase=scenario_name, step=0)
        raw_contacts.extend(rows)
        summaries[scenario_name] = scenario_summary(rows)
    return summaries, raw_contacts


def support_force_ratio(summary: dict[str, Any]) -> float | str:
    """Return left/right support force ratio from neutral contact."""

    left_total = sum(force(summary, "left", region) for region in REGIONS)
    right_total = sum(force(summary, "right", region) for region in REGIONS)
    total = left_total + right_total
    if total <= FORCE_EPS:
        return "unavailable"
    return float(min(left_total, right_total) / max(left_total, right_total, FORCE_EPS))


def result_label(metric: dict[str, Any]) -> str:
    """Classify one matrix row."""

    if not metric["can_classify_center_toe_heel"]:
        return "insufficient_contact_observability"
    if metric["fall_or_large_tilt"]:
        return "posture_instability_under_profile"
    if metric["toe_heel_bridge_contact_detected"]:
        return "bridge_contact_blocks_rollover_analysis"
    if metric["forward_pitch_toe_handoff_candidate_left"] is True and metric["forward_pitch_toe_handoff_candidate_right"] is True:
        return "bounded_toe_handoff_candidate"
    return "inconclusive_bounded_diagnostic"


def recommendation(metric: dict[str, Any]) -> str:
    """Return conservative recommendation for one matrix row."""

    if metric["result_label"] == "insufficient_contact_observability":
        return "Tune or replace foot geometry before rollover diagnostics; current foot cannot classify center/toe/heel."
    if metric["result_label"] == "bridge_contact_blocks_rollover_analysis":
        return "Foot patch spacing/height should be tuned before using this variant for rollover analysis."
    if metric["result_label"] == "posture_instability_under_profile":
        return "Controller authority/profile interaction is unstable; keep bounded and request verified motor/controller specs."
    if metric["result_label"] == "bounded_toe_handoff_candidate":
        return "Use only for scripted bounded diagnostics; still not PPO-ready and not walking success."
    return "Evidence is incomplete; keep as bounded diagnostic and request actuator/controller specs."


def build_matrix(config: dict[str, Any], variants: list[Variant]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Run the full foot x actuator sensitivity matrix."""

    require_mujoco()
    profiles = config["actuator_profiles"]
    all_metrics: list[dict[str, Any]] = []
    all_contacts: list[dict[str, Any]] = []
    inventories: dict[str, list[dict[str, Any]]] = {}
    for variant in variants:
        model = mujoco.MjModel.from_xml_path(str(variant.scene_path))
        inventory = patch_inventory(model, variant)
        inventories[variant.name] = inventory
        can_classify = can_classify_center_toe_heel(inventory)
        for profile_name, profile in profiles.items():
            contact_summaries, contact_rows = run_contact_scenarios(model, variant, profile_name, config)
            hold_summary, hold_contacts = run_pd_hold_with_profile(
                model,
                profile=profile,
                profile_name=profile_name,
                variant_name=variant.name,
                config=config,
                forward_probe=False,
            )
            probe_summary, probe_contacts = run_pd_hold_with_profile(
                model,
                profile=profile,
                profile_name=profile_name,
                variant_name=variant.name,
                config=config,
                forward_probe=True,
            )
            all_contacts.extend(contact_rows)
            all_contacts.extend(hold_contacts)
            all_contacts.extend(probe_contacts)
            neutral = contact_summaries["neutral"]
            metric = {
                "foot_variant": variant.name,
                "actuator_profile": profile_name,
                "valid_for": profile["valid_for"],
                "source": profile["source"],
                "confidence": profile["confidence"],
                "torque_side": profile["torque_side"],
                "can_classify_center_toe_heel": can_classify,
                "toe_heel_bridge_contact_detected": any(
                    bridge_detected(contact_summaries, side, can_classify) is True for side in SIDES
                )
                if can_classify
                else "unavailable",
                "neutral_center_first_left": center_first(neutral, "left", can_classify),
                "neutral_center_first_right": center_first(neutral, "right", can_classify),
                "forward_pitch_toe_handoff_candidate_left": toe_handoff(
                    neutral, contact_summaries["forward_pitch"], "left", can_classify
                ),
                "forward_pitch_toe_handoff_candidate_right": toe_handoff(
                    neutral, contact_summaries["forward_pitch"], "right", can_classify
                ),
                "contact_none_rate": probe_summary["contact_none_rate"],
                "fall_or_large_tilt": bool(hold_summary["fall_or_large_tilt"] or probe_summary["fall_or_large_tilt"]),
                "tilt_max": max(float(hold_summary["tilt_max"]), float(probe_summary["tilt_max"])),
                "base_height_drift": probe_summary["base_height_drift"],
                "actuator_saturation_rate": probe_summary["actuator_saturation_rate"],
                "ankle_pitch_saturation_rate": probe_summary["ankle_pitch_saturation_rate"],
                "joint_target_clamp_rate": probe_summary["joint_target_clamp_rate"],
                "min_swing_force": "unavailable",
                "support_force_ratio": support_force_ratio(neutral),
                "walking_success_claimed": False,
            }
            metric["result_label"] = result_label(metric)
            metric["recommendation"] = recommendation(metric)
            all_metrics.append(metric)
    metadata = {
        "schema_version": 1,
        "version": config["version"],
        "valid_for": config["valid_for"],
        "invalid_for": config["invalid_for"],
        "variant_count": len(variants),
        "profile_count": len(profiles),
        "matrix_count": len(all_metrics),
        "foot_variant_inventory": inventories,
        "actuator_profiles": profiles,
    }
    return all_metrics, all_contacts, metadata


def write_json(path: Path, payload: Any) -> None:
    """Write JSON payload."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """Write CSV rows."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def metric_score(row: dict[str, Any]) -> tuple[int, float, float, float]:
    """Return a conservative score tuple for ranking matrix rows."""

    observable = int(bool(row["can_classify_center_toe_heel"]))
    no_bridge = int(row["toe_heel_bridge_contact_detected"] is False)
    stable = int(not row["fall_or_large_tilt"])
    saturation_penalty = float(row["ankle_pitch_saturation_rate"]) + float(row["actuator_saturation_rate"])
    return (observable + no_bridge + stable, -saturation_penalty, -float(row["tilt_max"]), -abs(float(row["base_height_drift"])))


def summarize_results(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """Build high-level summary for report and final artifacts."""

    bridge_by_variant: dict[str, bool | str] = {}
    toe_by_variant_profile: dict[str, dict[str, Any]] = {}
    for row in metrics:
        bridge_by_variant.setdefault(row["foot_variant"], row["toe_heel_bridge_contact_detected"])
        toe_by_variant_profile[f"{row['foot_variant']}::{row['actuator_profile']}"] = {
            "left": row["forward_pitch_toe_handoff_candidate_left"],
            "right": row["forward_pitch_toe_handoff_candidate_right"],
        }
    best = max(metrics, key=metric_score) if metrics else None
    worst = min(metrics, key=metric_score) if metrics else None
    rated_rows = [row for row in metrics if row["actuator_profile"] == "rated_safe"]
    ankle_boost_rows = [row for row in metrics if row["actuator_profile"] == "ankle_boost_hypothesis"]
    return {
        "status": "BOUNDED_SENSITIVITY_COMPLETE" if metrics else "BLOCKED",
        "matrix_rows": len(metrics),
        "best_row": best,
        "worst_row": worst,
        "bridge_by_variant": bridge_by_variant,
        "toe_handoff_by_variant_profile": toe_by_variant_profile,
        "rated_safe_enough": bool(
            rated_rows
            and all(not row["fall_or_large_tilt"] for row in rated_rows)
            and any(row["can_classify_center_toe_heel"] for row in rated_rows)
        ),
        "ankle_boost_changes_toe_handoff_or_bridge": compare_ankle_boost(metrics),
        "ankle_boost_reduces_saturation": compare_ankle_boost_saturation(metrics),
        "next_blocker": "verified actuator/controller specs and foot geometry that avoids toe/heel bridge while preserving toe handoff observability",
    }


def compare_ankle_boost(metrics: list[dict[str, Any]]) -> bool:
    """Return whether ankle boost changes toe handoff or bridge versus mid_burst."""

    by_key = {(row["foot_variant"], row["actuator_profile"]): row for row in metrics}
    for variant in {row["foot_variant"] for row in metrics}:
        mid = by_key.get((variant, "mid_burst"))
        boost = by_key.get((variant, "ankle_boost_hypothesis"))
        if not mid or not boost:
            continue
        fields = (
            "toe_heel_bridge_contact_detected",
            "forward_pitch_toe_handoff_candidate_left",
            "forward_pitch_toe_handoff_candidate_right",
        )
        for field in fields:
            if mid[field] != boost[field]:
                return True
    return False


def compare_ankle_boost_saturation(metrics: list[dict[str, Any]]) -> bool:
    """Return whether ankle boost lowers ankle saturation versus mid_burst."""

    by_key = {(row["foot_variant"], row["actuator_profile"]): row for row in metrics}
    for variant in {row["foot_variant"] for row in metrics}:
        mid = by_key.get((variant, "mid_burst"))
        boost = by_key.get((variant, "ankle_boost_hypothesis"))
        if not mid or not boost:
            continue
        if float(boost["ankle_pitch_saturation_rate"]) < float(mid["ankle_pitch_saturation_rate"]):
            return True
    return False


def write_report(path: Path, payload: dict[str, Any]) -> None:
    """Write the sensitivity Markdown report."""

    path.parent.mkdir(parents=True, exist_ok=True)
    summary = payload["summary"]
    metrics = payload["matrix_results"]
    lines = [
        "# Seedon v5_22 Foot x Actuator Sensitivity Report",
        "",
        "Task class: Class C bounded diagnostic. This report does not run PPO, does not claim walking success, and does not create verified motor specs.",
        "",
        "## Summary",
        "",
        f"- Status: `{summary['status']}`",
        f"- Matrix rows: `{summary['matrix_rows']}`",
        f"- Valid for: `{payload['metadata']['valid_for']}`",
        f"- Invalid for: `{payload['metadata']['invalid_for']}`",
        f"- Rated safe enough for bounded diagnostics: `{summary['rated_safe_enough']}`",
        f"- Ankle boost changes toe handoff or bridge: `{summary['ankle_boost_changes_toe_handoff_or_bridge']}`",
        f"- Ankle boost reduces ankle saturation: `{summary['ankle_boost_reduces_saturation']}`",
        f"- Next blocker: {summary['next_blocker']}",
        "",
        "## Matrix Setup",
        "",
        f"- Foot variants: `{sorted({row['foot_variant'] for row in metrics})}`",
        f"- Actuator profiles: `{sorted({row['actuator_profile'] for row in metrics})}`",
        "- All profiles remain diagnostic only; torque side is `unknown_motor_side_or_joint_output`.",
        "- `ankle_boost_hypothesis` is `source=assumption`, `confidence=low`.",
        "",
        "## Result Table",
        "",
        "| foot variant | profile | classify C/T/H | bridge | toe handoff L/R | fall/tilt | tilt max | ankle sat | result |",
        "|---|---|---:|---:|---|---:|---:|---:|---|",
    ]
    for row in metrics:
        lines.append(
            f"| `{row['foot_variant']}` | `{row['actuator_profile']}` | "
            f"{row['can_classify_center_toe_heel']} | {row['toe_heel_bridge_contact_detected']} | "
            f"{row['forward_pitch_toe_handoff_candidate_left']}/{row['forward_pitch_toe_handoff_candidate_right']} | "
            f"{row['fall_or_large_tilt']} | {float(row['tilt_max']):.6g} | "
            f"{float(row['ankle_pitch_saturation_rate']):.6g} | `{row['result_label']}` |"
        )
    lines.extend(["", "## Bridge Contact Sensitivity", ""])
    for variant, bridge in summary["bridge_by_variant"].items():
        lines.append(f"- `{variant}`: bridge contact = `{bridge}`")
    lines.extend(["", "## Toe Handoff Sensitivity", ""])
    for key, value in summary["toe_handoff_by_variant_profile"].items():
        lines.append(f"- `{key}`: left=`{value['left']}`, right=`{value['right']}`")
    lines.extend(["", "## Ankle Saturation Analysis", ""])
    for row in metrics:
        lines.append(
            f"- `{row['foot_variant']} / {row['actuator_profile']}`: ankle saturation `{float(row['ankle_pitch_saturation_rate']):.6g}`, actuator saturation `{float(row['actuator_saturation_rate']):.6g}`"
        )
    lines.extend(["", "## Rated vs Peak Interpretation", ""])
    lines.append("- `rated_safe` uses provided rated torque only and is valid for bounded diagnostics.")
    lines.append("- `mid_burst` is an intermediate diagnostic profile, not a verified controller mode.")
    lines.append("- `peak_upper_bound` is an upper-bound diagnostic only and is invalid for continuous gait claims.")
    lines.append("- None of these values are verified MuJoCo joint forceranges because torque side is unknown.")
    lines.extend(["", "## What Can Be Concluded", ""])
    lines.append("- v5_22 foot/contact observability and bounded profile saturation can be compared without modifying the source model.")
    lines.append("- Rows with unavailable contact metrics cannot support center-first rollover analysis.")
    lines.append("- Bridge contact remains a geometry issue, not proof of controller success or failure.")
    lines.extend(["", "## What Must Not Be Claimed", ""])
    lines.append("- Do not claim walking success.")
    lines.append("- Do not claim sim2real validity.")
    lines.append("- Do not treat provided torque as verified joint-output forcerange.")
    lines.append("- Do not use `peak_upper_bound` as a continuous gait claim.")
    lines.extend(["", "## Next Recommendation", ""])
    lines.append("- Foot geometry tuning: tune center/toe/heel spacing and height to remove toe/heel bridge.")
    lines.append("- Controller authority tuning: keep tests bounded and compare ankle saturation before changing rewards.")
    lines.append("- Actuator spec request: request torque side, gear ratio, max velocity, current limit, encoder, backlash, and control mode.")
    lines.append("- Do not PPO until foot observability and actuator/controller specs are less ambiguous.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> int:
    """Run the bounded sensitivity workflow."""

    args = parse_args()
    config = load_json_object(args.config, "sensitivity config")
    envelope_path = resolve_repo_path(config["actuator_envelope_path"])
    envelope = load_json_object(envelope_path, "actuator envelope")
    if envelope.get("status") != "PARTIAL_ACTUATOR_ENVELOPE":
        raise ValueError(f"Unexpected actuator envelope status: {envelope.get('status')}")
    variants = create_v5_22_variants(config)
    matrix_results, raw_contacts, metadata = build_matrix(config, variants)
    summary = summarize_results(matrix_results)
    payload = {
        "schema_version": 1,
        "summary": summary,
        "metadata": metadata,
        "matrix_results": matrix_results,
        "source_inputs": {
            "config": rel_path(args.config),
            "model_path": config["model_path"],
            "actuator_envelope_path": config["actuator_envelope_path"],
            "foot_profile_path": config["foot_profile_path"],
        },
    }
    artifacts_dir = resolve_repo_path(config["artifacts_dir"])
    matrix_fields = [
        "foot_variant",
        "actuator_profile",
        "can_classify_center_toe_heel",
        "toe_heel_bridge_contact_detected",
        "neutral_center_first_left",
        "neutral_center_first_right",
        "forward_pitch_toe_handoff_candidate_left",
        "forward_pitch_toe_handoff_candidate_right",
        "contact_none_rate",
        "fall_or_large_tilt",
        "tilt_max",
        "base_height_drift",
        "actuator_saturation_rate",
        "ankle_pitch_saturation_rate",
        "joint_target_clamp_rate",
        "min_swing_force",
        "support_force_ratio",
        "result_label",
        "recommendation",
        "valid_for",
        "source",
        "confidence",
        "torque_side",
    ]
    contact_fields = [
        "foot_variant",
        "actuator_profile",
        "phase",
        "step",
        "contact_index",
        "geom1",
        "geom2",
        "dist",
        "pos_x",
        "pos_y",
        "pos_z",
        "normal_force",
        "is_floor_contact",
        "involves_foot",
        "classified_side",
        "classified_region",
    ]
    write_csv(artifacts_dir / "matrix_results.csv", matrix_results, matrix_fields)
    write_csv(artifacts_dir / "raw_contacts.csv", raw_contacts, contact_fields)
    write_json(artifacts_dir / "metrics.json", payload)
    write_report(resolve_repo_path(config["report_path"]), payload)
    print(f"status={summary['status']}")
    print(f"matrix_rows={summary['matrix_rows']}")
    print(f"output={artifacts_dir / 'metrics.json'}")
    print(f"report={resolve_repo_path(config['report_path'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
