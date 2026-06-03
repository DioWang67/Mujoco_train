"""Run Sedon v5_22 STL-derived foot collision prototype probe.

This Class C diagnostic reads ankle-pitch STL vertex clouds, derives simplified
contact-only MuJoCo collision prototypes, and measures contact persistence plus
local-x contact progression. It does not modify source XML/URDF/STL,
train.py/eval.py/env.py, does not run PPO, and does not claim walking success.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.sedon.diagnostics.v5_22.review_sedon_v5_22_foot_stl_vs_collision import (
    Bounds,
    bounds_from_vertices,
    bounds_payload,
    read_stl_vertices,
)
from tools.sedon.diagnostics.v5_22.run_sedon_v5_22_continuous_foot_bottom_probe import (
    local_contact_x,
)
from tools.sedon.diagnostics.v5_22.run_sedon_v5_22_foot_actuator_sensitivity import (
    FOOT_SPECS,
    FootGeom,
    classify_joint_group,
    contact_force,
    copy_mesh_assets,
    find_required,
    fmt_vec,
    is_floor_name,
    is_foot_name,
    load_json_object,
    mark_original_foot_non_contact,
    model_name,
    rel_path,
    require_mujoco,
    resolve_repo_path,
    resolve_source_feet,
)
from tools.sedon.diagnostics.v5_22.run_sedon_v5_22_toe_handoff_probe import (
    apply_posture_case,
    base_euler,
    quat_from_roll_pitch,
)

try:
    import mujoco
except ModuleNotFoundError as exc:  # pragma: no cover - only when MuJoCo is unavailable.
    mujoco = None
    _MUJOCO_IMPORT_ERROR: ModuleNotFoundError | None = exc
else:
    _MUJOCO_IMPORT_ERROR = None


DEFAULT_CONFIG = REPO_ROOT / "configs" / "sedon" / "sedon_v5_22_stl_derived_foot_collision_probe.yaml"
SIDES = ("left", "right")
REGIONS = ("heel", "center", "toe")
FORCE_EPS = 1e-5
PROTOTYPE_SOURCE = "stl_derived_prototype"
PROTOTYPE_VALID_FOR = "contact_diagnostic_only"


@dataclass(frozen=True)
class StlBottomProfile:
    """Bottom profile extracted from one STL vertex cloud.

    Args:
        side: Semantic side label.
        path: Source STL path.
        vertex_count: Number of parsed STL vertices.
        triangle_count: Number of STL triangles.
        bounds: Full vertex bounds.
        bottom_bounds: Bounds for lowest-z percentile candidate points.
        bottom_percentile: Percentile used for bottom candidate extraction.
        x_profile: Lowest-z profile rows along local x.
        bottom_shape: Heuristic flat / rocker-like / curved candidate label.
        limitations: Known interpretation limitations.
    """

    side: str
    path: Path
    vertex_count: int
    triangle_count: int
    bounds: Bounds
    bottom_bounds: Bounds
    bottom_percentile: float
    x_profile: list[dict[str, Any]]
    bottom_shape: str
    limitations: list[str]


@dataclass(frozen=True)
class FootVariant:
    """Generated artifact-only MuJoCo variant."""

    variant_id: str
    scene_path: Path
    metadata: dict[str, Any]


def object_id(model: Any, obj_type: Any, name: str) -> int:
    """Return MuJoCo object id or -1."""

    return int(mujoco.mj_name2id(model, obj_type, name))


def finite_float(value: Any, default: float) -> float:
    """Return a finite float or a provided default."""

    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def stl_path_for_side(config: dict[str, Any], side: str) -> Path:
    """Resolve the configured STL path for a side."""

    stl_paths = config["stl_paths"]
    if side not in stl_paths:
        raise ValueError(f"Missing STL path for side: {side}")
    path = resolve_repo_path(str(stl_paths[side]))
    if not path.is_file():
        raise FileNotFoundError(f"STL not found for {side}: {path}")
    return path


def x_bin_profile(
    vertices: list[tuple[float, float, float]],
    bounds: Bounds,
    *,
    x_bins: int,
) -> list[dict[str, Any]]:
    """Compute per-bin lowest-z values along local STL x."""

    min_x, max_x = bounds.min_xyz[0], bounds.max_xyz[0]
    span = max(max_x - min_x, 1e-12)
    buckets: list[list[tuple[float, float, float]]] = [[] for _ in range(x_bins)]
    for vertex in vertices:
        index = min(x_bins - 1, max(0, int((vertex[0] - min_x) / span * x_bins)))
        buckets[index].append(vertex)
    rows: list[dict[str, Any]] = []
    for index, bucket in enumerate(buckets):
        x_start = min_x + span * index / x_bins
        x_end = min_x + span * (index + 1) / x_bins
        if bucket:
            min_z = min(vertex[2] for vertex in bucket)
            y_values = [vertex[1] for vertex in bucket if abs(vertex[2] - min_z) <= 1e-5]
        else:
            min_z = None
            y_values = []
        rows.append(
            {
                "bin": index,
                "x_start": x_start,
                "x_end": x_end,
                "x_center": (x_start + x_end) * 0.5,
                "lowest_z": min_z,
                "candidate_y_min": min(y_values) if y_values else None,
                "candidate_y_max": max(y_values) if y_values else None,
                "sample_count": len(bucket),
            }
        )
    return rows


def classify_bottom_shape(x_profile: list[dict[str, Any]], settings: dict[str, Any]) -> str:
    """Classify bottom profile shape from lowest-z x profile."""

    z_values = [float(row["lowest_z"]) for row in x_profile if row["lowest_z"] is not None]
    if not z_values:
        return "unknown"
    z_range = max(z_values) - min(z_values)
    flat_threshold = float(settings["flat_z_range_threshold"])
    curved_threshold = float(settings["curved_z_range_threshold"])
    if z_range <= flat_threshold:
        return "flat"
    if z_range < curved_threshold:
        return "curved_candidate"
    rear = z_values[0]
    center = z_values[len(z_values) // 2]
    front = z_values[-1]
    if center <= rear and center <= front:
        return "rocker_like_candidate"
    return "curved_candidate"


def extract_stl_bottom_profile(config: dict[str, Any], side: str) -> StlBottomProfile:
    """Extract lowest-z percentile bottom profile from one side STL."""

    path = stl_path_for_side(config, side)
    vertices, _ = read_stl_vertices(path)
    if not vertices:
        raise ValueError(f"STL has no vertices: {path}")
    bounds = bounds_from_vertices(vertices)
    settings = config["stl_bottom_extraction"]
    z_values = np.array([vertex[2] for vertex in vertices], dtype=np.float64)
    percentile = float(settings["lowest_z_percentile"])
    threshold_z = float(np.percentile(z_values, percentile))
    bottom_candidates = [vertex for vertex in vertices if vertex[2] <= threshold_z]
    if not bottom_candidates:
        raise ValueError(f"No bottom candidates extracted for {path}")
    bottom_bounds = bounds_from_vertices(bottom_candidates)
    x_profile = x_bin_profile(vertices, bounds, x_bins=int(settings["x_bins"]))
    bottom_shape = classify_bottom_shape(x_profile, settings)
    limitations = [
        "STL orientation is assumed to use local z as vertical; source=assumption, confidence=low.",
        "Ankle-pitch STL is a visual candidate, not a bottom-specific CAD surface.",
        "Lowest-z percentile can include bevels or non-contact visual triangles.",
    ]
    return StlBottomProfile(
        side=side,
        path=path,
        vertex_count=len(vertices),
        triangle_count=len(vertices) // 3,
        bounds=bounds,
        bottom_bounds=bottom_bounds,
        bottom_percentile=percentile,
        x_profile=x_profile,
        bottom_shape=bottom_shape,
        limitations=limitations,
    )


def profile_payload(profile: StlBottomProfile) -> dict[str, Any]:
    """Serialize an STL bottom profile."""

    dims = profile.bounds.dims
    bottom_dims = profile.bottom_bounds.dims
    z_values = [row["lowest_z"] for row in profile.x_profile if row["lowest_z"] is not None]
    return {
        "side": profile.side,
        "stl_path": rel_path(profile.path),
        "vertex_count": profile.vertex_count,
        "triangle_count": profile.triangle_count,
        "bbox": {
            **(bounds_payload(profile.bounds) or {}),
            "length": dims[0],
            "width": dims[1],
            "height": dims[2],
        },
        "bottom_candidate_percentile": profile.bottom_percentile,
        "bottom_candidate_bbox": {
            **(bounds_payload(profile.bottom_bounds) or {}),
            "length": bottom_dims[0],
            "width": bottom_dims[1],
            "height": bottom_dims[2],
        },
        "bottom_lowest_z_profile_along_local_x": profile.x_profile,
        "bottom_lowest_z_range": (max(z_values) - min(z_values)) if z_values else "unavailable",
        "bottom_shape_candidate": profile.bottom_shape,
        "source": PROTOTYPE_SOURCE,
        "valid_for": PROTOTYPE_VALID_FOR,
        "limitations": profile.limitations,
    }


def derive_fit(profile: StlBottomProfile, foot: FootGeom, settings: dict[str, Any]) -> dict[str, Any]:
    """Derive a simple local collision fit from one STL bottom profile."""

    bounds = profile.bottom_bounds
    full = profile.bounds
    min_length = float(settings["min_length"])
    min_width = float(settings["min_width"])
    half_length = max(min_length * 0.5, bounds.dims[0] * 0.5)
    half_width = max(min_width * 0.5, bounds.dims[1] * 0.5)
    half_height = max(float(settings["half_height"]), bounds.dims[2] * 0.5, 0.004)
    center = bounds.center
    # Keep the contact proxy near the observed bottom; old foot box is fallback if STL y center is degenerate.
    y_center = center[1] if abs(center[1]) > 1e-6 else foot.pos[1]
    bottom_z = min(full.min_xyz[2], bounds.min_xyz[2])
    return {
        "center": (center[0], y_center, bottom_z + half_height),
        "size": (half_length, half_width, half_height),
        "bottom_z": bottom_z,
        "x_min": center[0] - half_length,
        "x_max": center[0] + half_length,
        "y_min": y_center - half_width,
        "y_max": y_center + half_width,
    }


def add_metadata_comment(body: ET.Element, variant_id: str) -> None:
    """Add required metadata comment before generated collision geoms."""

    body.append(
        ET.Comment(
            f" source={PROTOTYPE_SOURCE}, valid_for={PROTOTYPE_VALID_FOR}, "
            f"variant_id={variant_id}, not_final_collision=true "
        )
    )


def add_stl_fitted_box(body: ET.Element, side: str, fit: dict[str, Any], variant_id: str, friction: str) -> list[dict[str, Any]]:
    """Add one fitted box collision per foot."""

    geom_name = f"{FOOT_SPECS[side]['prefix']}_foot_contact_{variant_id}"
    ET.SubElement(
        body,
        "geom",
        {
            "name": geom_name,
            "type": "box",
            "pos": fmt_vec(fit["center"]),
            "size": fmt_vec(fit["size"]),
            "rgba": "0.1 0.55 0.9 0.35",
            "friction": friction,
            "contype": "1",
            "conaffinity": "1",
            "group": "3",
        },
    )
    return [{"side": side, "geom_name": geom_name, "type": "box", "pos": fmt_vec(fit["center"]), "size": fmt_vec(fit["size"])}]


def add_lowered_toe_box(
    body: ET.Element,
    side: str,
    fit: dict[str, Any],
    variant_id: str,
    friction: str,
    settings: dict[str, Any],
) -> list[dict[str, Any]]:
    """Add fitted base box plus a lower toe-biased box."""

    added = add_stl_fitted_box(body, side, fit, f"{variant_id}_base", friction)
    size = fit["size"]
    toe_size = (
        max(0.012, size[0] * float(settings["toe_length_scale"])),
        size[1],
        size[2] * float(settings["toe_height_scale"]),
    )
    toe_pos = (
        fit["x_max"] - toe_size[0],
        fit["center"][1],
        fit["bottom_z"] + toe_size[2] + float(settings["toe_z_lowering"]),
    )
    geom_name = f"{FOOT_SPECS[side]['prefix']}_foot_contact_{variant_id}_toe"
    ET.SubElement(
        body,
        "geom",
        {
            "name": geom_name,
            "type": "box",
            "pos": fmt_vec(toe_pos),
            "size": fmt_vec(toe_size),
            "rgba": "0.9 0.45 0.1 0.35",
            "friction": friction,
            "contype": "1",
            "conaffinity": "1",
            "group": "3",
        },
    )
    added.append({"side": side, "geom_name": geom_name, "type": "box", "pos": fmt_vec(toe_pos), "size": fmt_vec(toe_size)})
    return added


def add_rocker_capsules(
    body: ET.Element,
    side: str,
    fit: dict[str, Any],
    variant_id: str,
    friction: str,
    settings: dict[str, Any],
) -> list[dict[str, Any]]:
    """Add three cross-foot capsules approximating a rocker path."""

    added: list[dict[str, Any]] = []
    radius = max(0.004, fit["size"][2] * float(settings["radius_scale"]))
    half_width = max(0.006, fit["size"][1] * float(settings["width_scale"]))
    z_offsets = {
        "heel": float(settings["heel_z_offset"]),
        "center": float(settings["center_z_offset"]),
        "toe": float(settings["toe_z_offset"]),
    }
    x_positions = {
        "heel": fit["center"][0] - fit["size"][0] * float(settings["x_offset_scale"]),
        "center": fit["center"][0],
        "toe": fit["center"][0] + fit["size"][0] * float(settings["x_offset_scale"]),
    }
    for region in ("heel", "center", "toe"):
        x = x_positions[region]
        z = fit["bottom_z"] + radius + z_offsets[region]
        y = fit["center"][1]
        fromto = (x, y - half_width, z, x, y + half_width, z)
        geom_name = f"{FOOT_SPECS[side]['prefix']}_foot_contact_{variant_id}_{region}"
        ET.SubElement(
            body,
            "geom",
            {
                "name": geom_name,
                "type": "capsule",
                "fromto": " ".join(f"{value:.9g}" for value in fromto),
                "size": f"{radius:.9g}",
                "rgba": "0.55 0.2 0.8 0.35",
                "friction": friction,
                "contype": "1",
                "conaffinity": "1",
                "group": "3",
            },
        )
        added.append({"side": side, "region": region, "geom_name": geom_name, "type": "capsule", "fromto": list(fromto), "size": radius})
    return added


def add_continuous_bottom(
    body: ET.Element,
    side: str,
    fit: dict[str, Any],
    variant_id: str,
    friction: str,
    settings: dict[str, Any],
) -> list[dict[str, Any]]:
    """Add a single continuous ellipsoid bottom proxy."""

    size = (
        fit["size"][0] * float(settings["x_scale"]),
        fit["size"][1] * float(settings["y_scale"]),
        max(0.004, fit["size"][2] * float(settings["z_scale"])),
    )
    pos = (fit["center"][0], fit["center"][1], fit["bottom_z"] + size[2] + float(settings["z_offset"]))
    geom_name = f"{FOOT_SPECS[side]['prefix']}_foot_contact_{variant_id}"
    ET.SubElement(
        body,
        "geom",
        {
            "name": geom_name,
            "type": "ellipsoid",
            "pos": fmt_vec(pos),
            "size": fmt_vec(size),
            "rgba": "0.1 0.7 0.35 0.35",
            "friction": friction,
            "contype": "1",
            "conaffinity": "1",
            "group": "3",
        },
    )
    return [{"side": side, "geom_name": geom_name, "type": "ellipsoid", "pos": fmt_vec(pos), "size": fmt_vec(size)}]


def strip_trailing_whitespace(path: Path) -> None:
    """Remove trailing whitespace from a generated text file."""

    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


def create_variants(config: dict[str, Any], profiles: dict[str, StlBottomProfile]) -> list[FootVariant]:
    """Create artifact-only STL-derived MuJoCo variants."""

    require_mujoco()
    source_scene = resolve_repo_path(config["model_path"])
    if not source_scene.is_file():
        raise FileNotFoundError(f"Model XML not found: {source_scene}")
    variants_dir = resolve_repo_path(config["artifacts_dir"]) / "variants"
    variants: list[FootVariant] = []
    for variant_id in config["foot_variants"]:
        tree = ET.parse(source_scene)
        root = tree.getroot()
        feet = resolve_source_feet(root)
        mark_original_foot_non_contact(root)
        added: list[dict[str, Any]] = []
        for side in SIDES:
            foot = feet[side]
            body = find_required(root, f".//body[@name='{foot.body_name}']", foot.body_name)
            profile = profiles[side]
            fit = derive_fit(profile, foot, config["collision_fit"])
            add_metadata_comment(body, str(variant_id))
            if variant_id == "stl_fitted_box":
                added.extend(add_stl_fitted_box(body, side, fit, variant_id, foot.friction))
            elif variant_id == "stl_fitted_lowered_toe_box":
                added.extend(add_lowered_toe_box(body, side, fit, variant_id, foot.friction, config["lowered_toe_box"]))
            elif variant_id == "stl_fitted_rocker_capsules":
                added.extend(add_rocker_capsules(body, side, fit, variant_id, foot.friction, config["rocker_capsules"]))
            elif variant_id == "stl_fitted_continuous_bottom":
                added.extend(add_continuous_bottom(body, side, fit, variant_id, foot.friction, config["continuous_bottom"]))
            else:
                raise ValueError(f"Unknown foot variant: {variant_id}")
        variant_dir = variants_dir / str(variant_id)
        variant_dir.mkdir(parents=True, exist_ok=True)
        copy_mesh_assets(source_scene, variant_dir)
        scene_path = variant_dir / f"training_scene_{variant_id}.xml"
        tree.write(scene_path, encoding="utf-8", xml_declaration=True)
        strip_trailing_whitespace(scene_path)
        mujoco.MjModel.from_xml_path(str(scene_path))
        variants.append(
            FootVariant(
                variant_id=str(variant_id),
                scene_path=scene_path,
                metadata={
                    "source": PROTOTYPE_SOURCE,
                    "valid_for": PROTOTYPE_VALID_FOR,
                    "does_not_modify_source_scene": True,
                    "does_not_use_raw_stl_as_final_collision": True,
                    "added_geoms": added,
                },
            )
        )
    write_json(
        variants_dir / "manifest.json",
        {
            "schema_version": 1,
            "source_scene": rel_path(source_scene),
            "source": PROTOTYPE_SOURCE,
            "valid_for": PROTOTYPE_VALID_FOR,
            "variants": [
                {"variant_id": variant.variant_id, "scene_path": rel_path(variant.scene_path), "metadata": variant.metadata}
                for variant in variants
            ],
        },
    )
    return variants


def profile_limits_for_model(model: Any, profile: dict[str, Any]) -> np.ndarray:
    """Return per-actuator diagnostic command limits."""

    limits = np.zeros(model.nu, dtype=np.float64)
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        joint_name = model_name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) if joint_id >= 0 else ""
        limits[actuator_id] = float(profile[classify_joint_group(joint_name)])
    return limits


def side_for_geom(name: str) -> str:
    """Classify Sedon side from generated foot geom name."""

    lowered = name.lower()
    if lowered.startswith("r_"):
        return "right"
    if lowered.startswith("l_"):
        return "left"
    return "unknown"


def is_variant_foot_geom(name: str) -> bool:
    """Return whether a geom looks like a foot collision/contact geom."""

    lowered = name.lower()
    return "foot" in lowered and ("collision" in lowered or "contact" in lowered)


def region_for_local_x(local_x: float | str, profile: StlBottomProfile) -> str:
    """Classify heel/center/toe from STL bottom x range."""

    if local_x == "unavailable":
        return "unknown"
    lower = profile.bottom_bounds.min_xyz[0]
    upper = profile.bottom_bounds.max_xyz[0]
    width = max(upper - lower, 1e-9)
    normalized = (float(local_x) - lower) / width
    if normalized < 1.0 / 3.0:
        return "heel"
    if normalized < 2.0 / 3.0:
        return "center"
    return "toe"


def collect_contacts(
    model: Any,
    data: Any,
    *,
    variant_id: str,
    posture_case: str,
    actuator_profile: str,
    step: int,
    profiles: dict[str, StlBottomProfile],
) -> list[dict[str, Any]]:
    """Collect raw contacts with STL-derived local-x region classification."""

    rows: list[dict[str, Any]] = []
    for contact_index in range(int(data.ncon)):
        contact = data.contact[contact_index]
        geom1 = model_name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1))
        geom2 = model_name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2))
        floor_contact = is_floor_name(geom1) or is_floor_name(geom2)
        foot_geom = geom1 if is_variant_foot_geom(geom1) else geom2 if is_variant_foot_geom(geom2) else ""
        side = side_for_geom(foot_geom)
        local_x: float | str = "unavailable"
        region = "unknown"
        if floor_contact and side in SIDES:
            local_x = local_contact_x(model, data, side=side, world_pos=np.array(contact.pos, dtype=np.float64))
            region = region_for_local_x(local_x, profiles[side])
        rows.append(
            {
                "variant_id": variant_id,
                "posture_case": posture_case,
                "actuator_profile": actuator_profile,
                "step": step,
                "contact_index": contact_index,
                "geom1": geom1,
                "geom2": geom2,
                "foot_geom": foot_geom,
                "is_floor_contact": bool(floor_contact),
                "involves_foot": is_foot_name(geom1) or is_foot_name(geom2),
                "classified_side": side,
                "classified_region": region,
                "contact_x_local": local_x,
                "contact_pos_x": float(contact.pos[0]),
                "contact_pos_y": float(contact.pos[1]),
                "contact_pos_z": float(contact.pos[2]),
                "normal_force": contact_force(model, data, contact_index),
                "method": "stl_bottom_profile_local_x_projection",
            }
        )
    return rows


def run_case(
    model: Any,
    *,
    variant: FootVariant,
    posture_case: dict[str, Any],
    profile_name: str,
    actuator_profile: dict[str, Any],
    config: dict[str, Any],
    stl_profiles: dict[str, StlBottomProfile],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run one posture/profile contact diagnostic case."""

    sim = config["simulation"]
    thresholds = config["prototype_thresholds"]
    data = mujoco.MjData(model)
    apply_posture_case(model, data, posture_case)
    if "base_roll_degrees" in posture_case and len(data.qpos) >= 7:
        data.qpos[3:7] = quat_from_roll_pitch(
            math.radians(float(posture_case.get("base_roll_degrees", 0.0))),
            math.radians(float(posture_case["base_pitch_degrees"])),
        )
        mujoco.mj_forward(model, data)
    target_qpos = data.qpos.copy()
    initial_height = float(data.qpos[2]) if len(data.qpos) >= 3 else 0.0
    actuator_joint_ids = [int(model.actuator_trnid[index, 0]) for index in range(model.nu)]
    profile_limits = profile_limits_for_model(model, actuator_profile)
    ctrlrange = np.array(model.actuator_ctrlrange, dtype=np.float64)
    hold_steps = int(sim["hold_steps"])
    aggregation_start = max(1, hold_steps - int(sim["aggregation_last_steps"]) + 1)
    no_contact_steps = 0
    tilt_max = 0.0
    exploded = False
    raw_contacts: list[dict[str, Any]] = []
    aggregate_contacts: list[dict[str, Any]] = []
    for step in range(1, hold_steps + 1):
        ctrl = np.zeros(model.nu, dtype=np.float64)
        for actuator_id, joint_id in enumerate(actuator_joint_ids):
            if joint_id < 0:
                continue
            qpos_adr = int(model.jnt_qposadr[joint_id])
            dof_adr = int(model.jnt_dofadr[joint_id])
            raw = float(sim["pd_stiffness"]) * (float(target_qpos[qpos_adr]) - float(data.qpos[qpos_adr]))
            raw -= float(sim["pd_damping"]) * float(data.qvel[dof_adr])
            clipped = float(np.clip(raw, ctrlrange[actuator_id, 0], ctrlrange[actuator_id, 1]))
            ctrl[actuator_id] = float(np.clip(clipped, -profile_limits[actuator_id], profile_limits[actuator_id]))
        data.ctrl[:] = ctrl
        try:
            mujoco.mj_step(model, data)
        except Exception:
            exploded = True
            break
        if int(data.ncon) == 0:
            no_contact_steps += 1
        roll, pitch, _ = base_euler(data)
        tilt_max = max(tilt_max, abs(roll), abs(pitch))
        rows = collect_contacts(
            model,
            data,
            variant_id=variant.variant_id,
            posture_case=str(posture_case["name"]),
            actuator_profile=profile_name,
            step=step,
            profiles=stl_profiles,
        )
        raw_contacts.extend(rows)
        if step >= aggregation_start:
            aggregate_contacts.extend(rows)
        if not np.all(np.isfinite(data.qpos)) or float(np.max(np.abs(data.qpos))) > float(sim["max_qpos_abs"]):
            exploded = True
            break
    final_height = float(data.qpos[2]) if len(data.qpos) >= 3 else 0.0
    _, final_pitch, _ = base_euler(data)
    large_tilt = math.radians(float(thresholds["large_tilt_degrees"]))
    row = {
        "variant_id": variant.variant_id,
        "posture_case": posture_case["name"],
        "actuator_profile": profile_name,
        **summarize_contact_regions(aggregate_contacts),
        "contact_persistence": 1.0 - float(no_contact_steps / max(hold_steps, 1)),
        "contact_none_rate": float(no_contact_steps / max(hold_steps, 1)),
        "fall_or_large_tilt": bool(exploded or tilt_max > large_tilt),
        "base_height": final_height,
        "base_height_drift": float(final_height - initial_height),
        "base_pitch": final_pitch,
        "tilt_max": float(tilt_max),
        "source": PROTOTYPE_SOURCE,
        "valid_for": PROTOTYPE_VALID_FOR,
        "threshold_source": thresholds["source"],
        "threshold_confidence": thresholds["confidence"],
        "method": "stl_derived_collision_contact_probe",
    }
    row["bridge_like_pattern_detected"] = row.pop("toe_heel_bridge_like_pattern_detected")
    row["result_label"] = classify_result(row, thresholds)
    return row, raw_contacts


def summarize_contact_regions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize contact persistence, region ratios, and local-x progression."""

    xs: list[float] = []
    weighted = {region: 0.0 for region in REGIONS}
    side_region_seen = {side: {region: False for region in REGIONS} for side in SIDES}
    for row in rows:
        if not row["is_floor_contact"] or row["classified_side"] not in SIDES:
            continue
        if row["contact_x_local"] != "unavailable":
            xs.append(float(row["contact_x_local"]))
        region = str(row["classified_region"])
        if region in REGIONS:
            force = float(row["normal_force"])
            weighted[region] += force
            if force > FORCE_EPS:
                side_region_seen[str(row["classified_side"])][region] = True
    bridge_like = any(side_region_seen[side]["toe"] and side_region_seen[side]["heel"] for side in SIDES)
    total_force = sum(weighted.values())
    if not xs or total_force <= FORCE_EPS:
        return {
            "contact_x_mean": "unavailable",
            "contact_x_min": "unavailable",
            "contact_x_max": "unavailable",
            "contact_x_progression_score": "unavailable",
            "heel_region_contact_ratio": "unavailable",
            "center_region_contact_ratio": "unavailable",
            "toe_region_contact_ratio": "unavailable",
            "rollover_path_score": "unavailable",
            "toe_heel_bridge_like_pattern_detected": "unavailable",
        }
    heel = float(weighted["heel"] / total_force)
    center = float(weighted["center"] / total_force)
    toe = float(weighted["toe"] / total_force)
    return {
        "contact_x_mean": float(sum(xs) / len(xs)),
        "contact_x_min": float(min(xs)),
        "contact_x_max": float(max(xs)),
        "contact_x_progression_score": float(sum(xs) / len(xs)),
        "heel_region_contact_ratio": heel,
        "center_region_contact_ratio": center,
        "toe_region_contact_ratio": toe,
        "rollover_path_score": float(toe * (1.0 - heel)),
        "toe_heel_bridge_like_pattern_detected": bridge_like,
    }


def classify_result(row: dict[str, Any], thresholds: dict[str, Any]) -> str:
    """Classify one probe row against prototype thresholds."""

    none_rate = float(row["contact_none_rate"])
    if row["fall_or_large_tilt"]:
        return "posture_unstable"
    if row["contact_x_progression_score"] == "unavailable":
        return "projection_unavailable"
    if row["bridge_like_pattern_detected"] is True:
        return "bridge_like_pattern"
    if none_rate < float(thresholds["prototype_success_contact_none_rate_max"]) and float(row["contact_x_progression_score"]) > 0.0:
        return "prototype_success"
    if none_rate < float(thresholds["useful_signal_contact_none_rate_max"]):
        return "useful_signal"
    return "insufficient_contact_persistence"


def rank_row(row: dict[str, Any]) -> tuple[int, float, float, float]:
    """Rank rows by success label, persistence, progression, and rollover score."""

    label_score = {"prototype_success": 3, "useful_signal": 2}.get(str(row["result_label"]), 0)
    progression = -1.0 if row["contact_x_progression_score"] == "unavailable" else float(row["contact_x_progression_score"])
    rollover = -1.0 if row["rollover_path_score"] == "unavailable" else float(row["rollover_path_score"])
    return (label_score, -float(row["contact_none_rate"]), progression, rollover)


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Build high-level probe summary."""

    best = max(results, key=rank_row) if results else None
    success_rows = [row for row in results if row["result_label"] == "prototype_success"]
    useful_rows = [row for row in results if row["result_label"] in {"prototype_success", "useful_signal"}]
    previous_best = min(0.82, 0.85)
    best_none_rate = float(best["contact_none_rate"]) if best else math.nan
    return {
        "status": "STL_DERIVED_FOOT_COLLISION_PROBE_COMPLETE" if results else "BLOCKED",
        "result_rows": len(results),
        "best_row": best,
        "prototype_success_found": bool(success_rows),
        "useful_signal_found": bool(useful_rows),
        "previous_toe_handoff_best_contact_none_rate": 0.85,
        "previous_continuous_bottom_best_contact_none_rate": 0.82,
        "best_contact_none_rate_improvement_vs_previous_best": previous_best - best_none_rate if best else "unavailable",
        "recommend_replace_current_box_prototype": recommend_replacement(best),
        "do_not_claim_walking_success": True,
    }


def recommend_replacement(best: dict[str, Any] | None) -> str:
    """Return a conservative replacement recommendation."""

    if not best:
        return "no; probe produced no usable rows"
    if best["result_label"] == "prototype_success":
        return "yes_for_contact_diagnostic_prototype_only_not_final_collision"
    if best["result_label"] == "useful_signal":
        return "maybe_for_next_contact_diagnostic_iteration_only"
    return "no; contact persistence remains insufficient"


def run_probe(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, StlBottomProfile]]:
    """Run full STL-derived foot collision probe."""

    require_mujoco()
    stl_profiles = {side: extract_stl_bottom_profile(config, side) for side in SIDES}
    variants = create_variants(config, stl_profiles)
    results: list[dict[str, Any]] = []
    raw_contacts: list[dict[str, Any]] = []
    for variant in variants:
        model = mujoco.MjModel.from_xml_path(str(variant.scene_path))
        for profile_name, actuator_profile in config["actuator_profiles"].items():
            for posture_case in config["posture_cases"]:
                row, contacts = run_case(
                    model,
                    variant=variant,
                    posture_case=posture_case,
                    profile_name=profile_name,
                    actuator_profile=actuator_profile,
                    config=config,
                    stl_profiles=stl_profiles,
                )
                results.append(row)
                raw_contacts.extend(contacts)
    metadata = {
        "schema_version": 1,
        "version": config["version"],
        "source": PROTOTYPE_SOURCE,
        "valid_for": PROTOTYPE_VALID_FOR,
        "invalid_for": config["invalid_for"],
        "method_limitations": config["method_limitations"],
        "foot_variants": config["foot_variants"],
        "actuator_profiles": config["actuator_profiles"],
        "posture_cases": config["posture_cases"],
    }
    return results, raw_contacts, metadata, stl_profiles


def write_json(path: Path, payload: Any) -> None:
    """Write JSON with parent directory creation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """Write CSV rows."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    """Format a value for Markdown."""

    if value == "unavailable":
        return "`unavailable`"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_report(path: Path, payload: dict[str, Any]) -> None:
    """Write Markdown probe report."""

    path.parent.mkdir(parents=True, exist_ok=True)
    summary = payload["summary"]
    best = summary["best_row"] or {}
    profile_rows = []
    for side, profile in payload["stl_bottom_profile"]["profiles"].items():
        bbox = profile["bbox"]
        bottom = profile["bottom_candidate_bbox"]
        profile_rows.append(
            f"| `{side}` | `{profile['bottom_shape_candidate']}` | {bbox['length']:.6g} | {bbox['width']:.6g} | "
            f"{bbox['height']:.6g} | {bottom['length']:.6g} | {bottom['width']:.6g} |"
        )
    lines = [
        "# Sedon v5_22 STL-Derived Foot Collision Probe Report",
        "",
        "Task class: Class C contact diagnostic. This workflow reads ankle-pitch STL files, creates artifact-only simplified collision variants, does not modify source XML/URDF/STL/train/eval/env, does not run PPO, and does not claim walking success.",
        "",
        "## Summary",
        "",
        f"- Status: `{summary['status']}`",
        f"- Best variant: `{best.get('variant_id')}`",
        f"- Best posture/profile: `{best.get('posture_case')} / {best.get('actuator_profile')}`",
        f"- Best contact none rate: `{best.get('contact_none_rate')}`",
        f"- Previous toe handoff best contact none rate: `{summary['previous_toe_handoff_best_contact_none_rate']}`",
        f"- Previous continuous bottom best contact none rate: `{summary['previous_continuous_bottom_best_contact_none_rate']}`",
        f"- Improvement vs previous best 0.82: `{summary['best_contact_none_rate_improvement_vs_previous_best']}`",
        f"- Prototype success found: `{summary['prototype_success_found']}`",
        f"- Useful signal found: `{summary['useful_signal_found']}`",
        f"- Replace current box prototype: `{summary['recommend_replace_current_box_prototype']}`",
        "",
        "## STL Bottom Profile",
        "",
        "| side | shape candidate | bbox length | bbox width | bbox height | bottom length | bottom width |",
        "|---|---|---:|---:|---:|---:|---:|",
        *profile_rows,
        "",
        "## Collision Variants",
        "",
        "- `stl_fitted_box`: one fitted box per foot from lowest-z STL bottom bounds.",
        "- `stl_fitted_lowered_toe_box`: fitted base box plus a lower toe-biased box.",
        "- `stl_fitted_rocker_capsules`: three cross-foot capsules approximating heel/center/toe rocker support.",
        "- `stl_fitted_continuous_bottom`: one continuous ellipsoid bottom proxy per foot.",
        "",
        "All variants are tagged `source=stl_derived_prototype, valid_for=contact_diagnostic_only` and are not final collisions.",
        "",
        "## Probe Results",
        "",
        "| variant | profile | posture | none rate | persistence | x progression | heel | center | toe | rollover | bridge | tilt/fall | result |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in payload["results"]:
        lines.append(
            f"| `{row['variant_id']}` | `{row['actuator_profile']}` | `{row['posture_case']}` | "
            f"{fmt(row['contact_none_rate'])} | {fmt(row['contact_persistence'])} | {fmt(row['contact_x_progression_score'])} | "
            f"{fmt(row['heel_region_contact_ratio'])} | {fmt(row['center_region_contact_ratio'])} | {fmt(row['toe_region_contact_ratio'])} | "
            f"{fmt(row['rollover_path_score'])} | `{row['bridge_like_pattern_detected']}` | `{row['fall_or_large_tilt']}` | `{row['result_label']}` |"
        )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- STL orientation is assumed to use local z as vertical; this is `source=assumption, confidence=low`.",
            "- The ankle-pitch STL is not treated as a final collision mesh.",
            "- Contact metrics are diagnostic-only and do not establish walking success.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> int:
    """Run the STL-derived foot collision probe."""

    args = parse_args()
    config = load_json_object(args.config, "STL-derived foot collision probe config")
    results, raw_contacts, metadata, stl_profiles = run_probe(config)
    summary = summarize_results(results)
    stl_payload = {
        "schema_version": 1,
        "source": PROTOTYPE_SOURCE,
        "valid_for": PROTOTYPE_VALID_FOR,
        "profiles": {side: profile_payload(profile) for side, profile in stl_profiles.items()},
    }
    payload = {
        "schema_version": 1,
        "summary": summary,
        "metadata": metadata,
        "stl_bottom_profile": stl_payload,
        "results": results,
        "source_inputs": {
            "config": rel_path(args.config),
            "model_path": config["model_path"],
            "stl_paths": config["stl_paths"],
        },
    }
    artifacts_dir = resolve_repo_path(config["artifacts_dir"])
    result_fields = [
        "variant_id",
        "posture_case",
        "actuator_profile",
        "contact_none_rate",
        "contact_persistence",
        "contact_x_progression_score",
        "heel_region_contact_ratio",
        "center_region_contact_ratio",
        "toe_region_contact_ratio",
        "rollover_path_score",
        "bridge_like_pattern_detected",
        "fall_or_large_tilt",
        "result_label",
        "source",
        "valid_for",
        "method",
    ]
    contact_fields = [
        "variant_id",
        "posture_case",
        "actuator_profile",
        "step",
        "contact_index",
        "geom1",
        "geom2",
        "foot_geom",
        "is_floor_contact",
        "involves_foot",
        "classified_side",
        "classified_region",
        "contact_x_local",
        "contact_pos_x",
        "contact_pos_y",
        "contact_pos_z",
        "normal_force",
        "method",
    ]
    write_json(artifacts_dir / "stl_bottom_profile.yaml", stl_payload)
    write_csv(artifacts_dir / "probe_results.csv", results, result_fields)
    write_csv(artifacts_dir / "raw_contacts.csv", raw_contacts, contact_fields)
    write_json(artifacts_dir / "metrics.json", payload)
    write_report(resolve_repo_path(config["report_path"]), payload)
    print(f"status={summary['status']}")
    print(f"result_rows={summary['result_rows']}")
    print(f"best_variant={summary['best_row']['variant_id'] if summary['best_row'] else None}")
    print(f"best_contact_none_rate={summary['best_row']['contact_none_rate'] if summary['best_row'] else None}")
    print(f"report={resolve_repo_path(config['report_path'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
