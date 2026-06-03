"""Run Seedon v5_22 continuous foot bottom prototype probe.

This Class C diagnostic creates artifact-only continuous / rocker-like foot
bottom variants and measures contact persistence plus local-x contact
progression. It does not modify source XML/URDF, train.py, eval.py, env.py, and
does not run PPO or claim walking success.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.seedon.diagnostics.v5_22.run_seedon_v5_22_foot_actuator_sensitivity import (
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
    model_name,
    parse_vec3,
    quat_from_roll_pitch,
    rel_path,
    resolve_repo_path,
    resolve_source_feet,
)
from tools.seedon.diagnostics.v5_22.run_seedon_v5_22_toe_handoff_probe import (
    apply_posture_case,
    base_euler,
)

try:
    import mujoco
except ModuleNotFoundError as exc:  # pragma: no cover - only when MuJoCo is unavailable.
    mujoco = None
    _MUJOCO_IMPORT_ERROR: ModuleNotFoundError | None = exc
else:
    _MUJOCO_IMPORT_ERROR = None


DEFAULT_CONFIG = REPO_ROOT / "configs" / "seedon" / "seedon_v5_22_continuous_foot_bottom_probe.yaml"
SIDES = ("left", "right")
REGIONS = ("heel", "center", "toe")
FORCE_EPS = 1e-5
POSTURE_ORDER = {
    "neutral_stance": 0,
    "small_forward_base_pitch": 1,
    "medium_forward_base_pitch": 2,
    "ankle_toe_down_bias": 3,
    "mild_knee_hip_flexion_forward_lean": 4,
}


@dataclass(frozen=True)
class FootVariant:
    """Generated foot bottom variant."""

    name: str
    scene_path: Path
    classifier_mode: str
    status: str
    metadata: dict[str, Any]


def require_mujoco() -> None:
    """Raise if MuJoCo is unavailable."""

    if mujoco is None:
        raise RuntimeError(f"MuJoCo is not importable: {_MUJOCO_IMPORT_ERROR}")


def object_id(model: Any, obj_type: Any, name: str) -> int:
    """Return MuJoCo object id or -1."""

    return int(mujoco.mj_name2id(model, obj_type, name))


def mark_original_foot_non_contact(root: ET.Element) -> None:
    """Disable original foot collision contacts in an artifact-only variant."""

    for spec in FOOT_SPECS.values():
        geom = find_required(root, f".//geom[@name='{spec['original_geom']}']", spec["original_geom"])
        geom.set("contype", "0")
        geom.set("conaffinity", "0")
        geom.set("group", "1")
        geom.set("rgba", "0.12 0.12 0.12 0.12")


def add_discrete_patches(root: ET.Element, feet: dict[str, FootGeom], config: dict[str, Any]) -> list[dict[str, Any]]:
    """Add discrete toe/center/heel patches for baseline comparison."""

    added: list[dict[str, Any]] = []
    for side, foot in feet.items():
        body = find_required(root, f".//body[@name='{foot.body_name}']", foot.body_name)
        for region, patch in config["discrete_patch"].items():
            pos = (
                foot.pos[0] + float(patch["normalized_x"]) * foot.size[0],
                foot.pos[1],
                foot.bottom_z + max(0.003, foot.size[2] * float(patch["size_scale"][2])),
            )
            size = (
                max(0.004, foot.size[0] * float(patch["size_scale"][0])),
                max(0.004, foot.size[1] * float(patch["size_scale"][1])),
                max(0.003, foot.size[2] * float(patch["size_scale"][2])),
            )
            geom_name = f"{FOOT_SPECS[side]['prefix']}_foot_contact_discrete_toe_center_heel_{region}"
            ET.SubElement(
                body,
                "geom",
                {
                    "name": geom_name,
                    "type": "box",
                    "pos": fmt_vec(pos),
                    "size": fmt_vec(size),
                    "rgba": "0.1 0.45 0.9 0.35",
                    "friction": foot.friction,
                    "contype": "1",
                    "conaffinity": "1",
                    "group": "3",
                    "user": "1",
                },
            )
            added.append({"side": side, "region": region, "geom_name": geom_name, "pos": fmt_vec(pos), "size": fmt_vec(size)})
    return added


def add_continuous_bottom(root: ET.Element, feet: dict[str, FootGeom], config: dict[str, Any], variant_name: str) -> list[dict[str, Any]]:
    """Add simplified continuous rocker-like bottom primitive."""

    settings = config["continuous_bottom"]
    added: list[dict[str, Any]] = []
    for side, foot in feet.items():
        body = find_required(root, f".//body[@name='{foot.body_name}']", foot.body_name)
        half_z = max(0.004, foot.size[2] * float(settings["z_scale"]))
        pos = (
            foot.pos[0],
            foot.pos[1],
            foot.bottom_z + half_z + float(settings["z_offset"]),
        )
        size = (
            max(0.01, foot.size[0] * float(settings["x_scale"])),
            max(0.01, foot.size[1] * float(settings["y_scale"])),
            half_z,
        )
        geom_name = f"{FOOT_SPECS[side]['prefix']}_foot_contact_{variant_name}_continuous_bottom"
        ET.SubElement(
            body,
            "geom",
            {
                "name": geom_name,
                "type": str(settings["type"]),
                "pos": fmt_vec(pos),
                "size": fmt_vec(size),
                "rgba": "0.15 0.65 0.35 0.36",
                "friction": str(settings["friction"]),
                "contype": "1",
                "conaffinity": "1",
                "group": "3",
                "user": "1",
            },
        )
        added.append({"side": side, "region": "continuous", "geom_name": geom_name, "pos": fmt_vec(pos), "size": fmt_vec(size)})
    return added


def create_variants(config: dict[str, Any]) -> list[FootVariant]:
    """Create artifact-only continuous foot bottom variants."""

    require_mujoco()
    source_scene = resolve_repo_path(config["model_path"])
    variants_dir = resolve_repo_path(config["artifacts_dir"]) / "variants"
    variants: list[FootVariant] = []
    for variant_name in config["foot_variants"]:
        tree = ET.parse(source_scene)
        root = tree.getroot()
        feet = resolve_source_feet(root)
        added: list[dict[str, Any]] = []
        classifier_mode = "contact_point_local_x"
        if variant_name == "current_box_baseline":
            status = "source_box_collision_copied"
        elif variant_name == "discrete_toe_center_heel":
            mark_original_foot_non_contact(root)
            added = add_discrete_patches(root, feet, config)
            status = "prototype_discrete_patches"
        elif variant_name in {"continuous_rocker_bottom", "hybrid_continuous_with_region_classifier"}:
            mark_original_foot_non_contact(root)
            added = add_continuous_bottom(root, feet, config, variant_name)
            status = "prototype_continuous_bottom"
        else:
            raise ValueError(f"Unknown foot variant: {variant_name}")

        variant_dir = variants_dir / variant_name
        variant_dir.mkdir(parents=True, exist_ok=True)
        copy_mesh_assets(source_scene, variant_dir)
        scene_path = variant_dir / f"training_scene_{variant_name}.xml"
        tree.write(scene_path, encoding="utf-8", xml_declaration=True)
        mujoco.MjModel.from_xml_path(str(scene_path))
        variants.append(
            FootVariant(
                name=variant_name,
                scene_path=scene_path,
                classifier_mode=classifier_mode,
                status=status,
                metadata={
                    **config["prototype_metadata"],
                    "added_geoms": added,
                    "does_not_modify_source_scene": True,
                    "source_scene": rel_path(source_scene),
                },
            )
        )
    write_json(
        variants_dir / "manifest.json",
        {
            "schema_version": 1,
            "source_scene": rel_path(source_scene),
            "prototype_metadata": config["prototype_metadata"],
            "variants": [
                {
                    "name": variant.name,
                    "scene_path": rel_path(variant.scene_path),
                    "classifier_mode": variant.classifier_mode,
                    "status": variant.status,
                    "metadata": variant.metadata,
                }
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


def foot_body_for_side(side: str) -> str:
    """Return Seedon foot body for side."""

    return FOOT_SPECS[side]["body"]


def side_for_geom(name: str) -> str:
    """Classify Seedon side from geom name."""

    lowered = name.lower()
    if lowered.startswith("r_"):
        return "right"
    if lowered.startswith("l_"):
        return "left"
    return "unknown"


def is_variant_foot_geom(name: str) -> bool:
    """Return whether name looks like Seedon foot contact geometry."""

    lowered = name.lower()
    return "foot" in lowered and ("collision" in lowered or "contact" in lowered)


def local_contact_x(model: Any, data: Any, *, side: str, world_pos: np.ndarray) -> float | str:
    """Project world contact position into foot body local x."""

    body_name = foot_body_for_side(side)
    body_id = object_id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        return "unavailable"
    xpos = np.array(data.xpos[body_id], dtype=np.float64)
    xmat = np.array(data.xmat[body_id], dtype=np.float64).reshape(3, 3)
    local = xmat.T @ (world_pos - xpos)
    return float(local[0])


def region_for_local_x(local_x: float | str, foot: FootGeom) -> str:
    """Classify heel/center/toe by rear/middle/front thirds."""

    if local_x == "unavailable":
        return "unknown"
    lower = foot.pos[0] - foot.size[0]
    upper = foot.pos[0] + foot.size[0]
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
    foot_variant: str,
    posture_case: str,
    actuator_profile: str,
    step: int,
    feet: dict[str, FootGeom],
) -> list[dict[str, Any]]:
    """Collect raw contacts with local-x region classification."""

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
            region = region_for_local_x(local_x, feet[side])
        rows.append(
            {
                "foot_variant": foot_variant,
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
                "method": "contact_point_projected_to_foot_body_local_x",
            }
        )
    return rows


def run_case(
    model: Any,
    *,
    variant: FootVariant,
    posture_case: dict[str, Any],
    profile_name: str,
    profile: dict[str, Any],
    config: dict[str, Any],
    feet: dict[str, FootGeom],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run one posture/profile case."""

    sim = config["simulation"]
    thresholds = config["prototype_thresholds"]
    data = mujoco.MjData(model)
    apply_posture_case(model, data, posture_case)
    target_qpos = data.qpos.copy()
    initial_height = float(data.qpos[2]) if len(data.qpos) >= 3 else 0.0
    actuator_joint_ids = [int(model.actuator_trnid[index, 0]) for index in range(model.nu)]
    profile_limits = profile_limits_for_model(model, profile)
    ctrlrange = np.array(model.actuator_ctrlrange, dtype=np.float64)
    kp = float(sim["pd_stiffness"])
    kd = float(sim["pd_damping"])
    hold_steps = int(sim["hold_steps"])
    aggregation_start = max(1, hold_steps - int(sim["aggregation_last_steps"]) + 1)
    no_contact_steps = 0
    tilt_max = 0.0
    raw_contacts: list[dict[str, Any]] = []
    aggregate_contacts: list[dict[str, Any]] = []
    exploded = False
    for step in range(1, hold_steps + 1):
        ctrl = np.zeros(model.nu, dtype=np.float64)
        for actuator_id, joint_id in enumerate(actuator_joint_ids):
            if joint_id < 0:
                continue
            qpos_adr = int(model.jnt_qposadr[joint_id])
            dof_adr = int(model.jnt_dofadr[joint_id])
            raw = kp * (float(target_qpos[qpos_adr]) - float(data.qpos[qpos_adr])) - kd * float(data.qvel[dof_adr])
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
            foot_variant=variant.name,
            posture_case=str(posture_case["name"]),
            actuator_profile=profile_name,
            step=step,
            feet=feet,
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
    fall_or_large_tilt = bool(exploded or tilt_max > large_tilt)
    metrics = summarize_contact_regions(aggregate_contacts, feet)
    row = {
        "foot_variant": variant.name,
        "posture_case": posture_case["name"],
        "actuator_profile": profile_name,
        **metrics,
        "contact_persistence": 1.0 - float(no_contact_steps / max(hold_steps, 1)),
        "contact_none_rate": float(no_contact_steps / max(hold_steps, 1)),
        "fall_or_large_tilt": fall_or_large_tilt,
        "base_height": final_height,
        "base_height_drift": float(final_height - initial_height),
        "base_pitch": final_pitch,
        "tilt_max": float(tilt_max),
        "valid_for": config["valid_for"],
        "source": config["prototype_metadata"]["source"],
        "threshold_source": thresholds["source"],
        "method": "contact_point_projected_to_foot_body_local_x",
    }
    row["result_label"] = classify_result(row, thresholds)
    return row, raw_contacts


def summarize_contact_regions(rows: list[dict[str, Any]], feet: dict[str, FootGeom]) -> dict[str, Any]:
    """Summarize local-x contact progression and region ratios."""

    xs: list[float] = []
    weighted = {region: 0.0 for region in REGIONS}
    bridge_like = False
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
            side_region_seen[str(row["classified_side"])][region] = side_region_seen[str(row["classified_side"])][region] or force > FORCE_EPS
    for side in SIDES:
        if side_region_seen[side]["toe"] and side_region_seen[side]["heel"]:
            bridge_like = True
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
    mean_x = float(sum(xs) / len(xs))
    min_x = float(min(xs))
    max_x = float(max(xs))
    heel = float(weighted["heel"] / total_force)
    center = float(weighted["center"] / total_force)
    toe = float(weighted["toe"] / total_force)
    return {
        "contact_x_mean": mean_x,
        "contact_x_min": min_x,
        "contact_x_max": max_x,
        "contact_x_progression_score": mean_x,
        "heel_region_contact_ratio": heel,
        "center_region_contact_ratio": center,
        "toe_region_contact_ratio": toe,
        "rollover_path_score": float(toe * (1.0 - heel)),
        "toe_heel_bridge_like_pattern_detected": bridge_like,
    }


def classify_result(row: dict[str, Any], thresholds: dict[str, Any]) -> str:
    """Classify one matrix row."""

    if row["contact_x_progression_score"] == "unavailable":
        return "projection_unavailable"
    if row["fall_or_large_tilt"]:
        return "posture_unstable"
    if float(row["contact_none_rate"]) >= float(thresholds["contact_none_rate_max"]):
        return "insufficient_contact_persistence"
    if row["toe_heel_bridge_like_pattern_detected"] is True:
        return "bridge_like_pattern"
    if float(row["contact_x_progression_score"]) > float(thresholds["contact_x_progression_min"]):
        return "continuous_rollover_candidate"
    return "no_forward_x_progression"


def progression_by_variant_posture(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute neutral-to-forward progression deltas per variant/profile."""

    by_key = {(row["foot_variant"], row["actuator_profile"], row["posture_case"]): row for row in results}
    output: dict[str, Any] = {}
    for variant in {row["foot_variant"] for row in results}:
        for profile in {row["actuator_profile"] for row in results}:
            neutral = by_key.get((variant, profile, "neutral_stance"))
            forward = by_key.get((variant, profile, "ankle_toe_down_bias")) or by_key.get(
                (variant, profile, "medium_forward_base_pitch")
            )
            if not neutral or not forward:
                continue
            if neutral["contact_x_mean"] == "unavailable" or forward["contact_x_mean"] == "unavailable":
                delta: float | str = "unavailable"
            else:
                delta = float(forward["contact_x_mean"]) - float(neutral["contact_x_mean"])
            output[f"{variant}::{profile}"] = delta
    return output


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Build high-level summary."""

    candidates = [row for row in results if row["result_label"] == "continuous_rollover_candidate"]
    best = max(results, key=rank_row) if results else None
    baseline_rows = [row for row in results if row["foot_variant"] == "discrete_toe_center_heel"]
    continuous_rows = [
        row
        for row in results
        if row["foot_variant"] in {"continuous_rocker_bottom", "hybrid_continuous_with_region_classifier"}
    ]
    baseline_best_persistence = max((1.0 - float(row["contact_none_rate"]) for row in baseline_rows), default=0.0)
    continuous_best_persistence = max((1.0 - float(row["contact_none_rate"]) for row in continuous_rows), default=0.0)
    return {
        "status": "CONTINUOUS_FOOT_BOTTOM_PROBE_COMPLETE" if results else "BLOCKED",
        "result_rows": len(results),
        "best_row": best,
        "continuous_rollover_candidate_found": bool(candidates),
        "candidate_count": len(candidates),
        "progression_by_variant_profile": progression_by_variant_posture(results),
        "continuous_better_than_discrete": bool(continuous_best_persistence > baseline_best_persistence),
        "best_discrete_contact_persistence": baseline_best_persistence,
        "best_continuous_contact_persistence": continuous_best_persistence,
        "reference_contact_none_rate": 0.85,
        "best_contact_none_rate_improvement_vs_reference": (
            0.85 - float(best["contact_none_rate"]) if best and best["contact_none_rate"] != "unavailable" else "unavailable"
        ),
        "next_step": "scripted rollover sequence" if candidates else "mechanical foot bottom redesign / mesh review",
    }


def rank_row(row: dict[str, Any]) -> tuple[int, float, float, float]:
    """Rank rows conservatively."""

    candidate = int(row["result_label"] == "continuous_rollover_candidate")
    none_rate = float(row["contact_none_rate"])
    progression = -1.0 if row["contact_x_progression_score"] == "unavailable" else float(row["contact_x_progression_score"])
    rollover = -1.0 if row["rollover_path_score"] == "unavailable" else float(row["rollover_path_score"])
    return (candidate, -none_rate, progression, rollover)


def run_probe(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Run full continuous foot bottom probe."""

    require_mujoco()
    variants = create_variants(config)
    results: list[dict[str, Any]] = []
    raw_contacts: list[dict[str, Any]] = []
    for variant in variants:
        model = mujoco.MjModel.from_xml_path(str(variant.scene_path))
        source_root = ET.parse(variant.scene_path).getroot()
        feet = resolve_source_feet(source_root)
        for profile_name, profile in config["actuator_profiles"].items():
            for posture_case in config["posture_cases"]:
                row, contacts = run_case(
                    model,
                    variant=variant,
                    posture_case=posture_case,
                    profile_name=profile_name,
                    profile=profile,
                    config=config,
                    feet=feet,
                )
                results.append(row)
                raw_contacts.extend(contacts)
    metadata = {
        "schema_version": 1,
        "version": config["version"],
        "valid_for": config["valid_for"],
        "invalid_for": config["invalid_for"],
        "prototype_metadata": config["prototype_metadata"],
        "method_limitations": config["method_limitations"],
        "variant_count": len(variants),
        "actuator_profiles": config["actuator_profiles"],
        "posture_cases": config["posture_cases"],
    }
    return results, raw_contacts, metadata


def write_json(path: Path, payload: Any) -> None:
    """Write JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """Write CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    """Format report metric."""

    if value == "unavailable":
        return "`unavailable`"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_report(path: Path, payload: dict[str, Any]) -> None:
    """Write Markdown report."""

    path.parent.mkdir(parents=True, exist_ok=True)
    summary = payload["summary"]
    results = payload["results"]
    best = summary["best_row"] or {}
    lines = [
        "# Seedon v5_22 Continuous Foot Bottom Probe Report",
        "",
        "Task class: Class C contact persistence diagnostic. This report does not run PPO, does not claim walking success, and does not define final mechanical foot design.",
        "",
        "## Summary",
        "",
        f"- Status: `{summary['status']}`",
        f"- Result rows: `{summary['result_rows']}`",
        f"- Best variant: `{best.get('foot_variant')}`",
        f"- Best posture/profile: `{best.get('posture_case')} / {best.get('actuator_profile')}`",
        f"- Best contact none rate: `{best.get('contact_none_rate')}`",
        f"- Improvement vs previous 0.85: `{summary['best_contact_none_rate_improvement_vs_reference']}`",
        f"- Continuous better than discrete: `{summary['continuous_better_than_discrete']}`",
        f"- Candidate found: `{summary['continuous_rollover_candidate_found']}`",
        f"- Next step: {summary['next_step']}",
        "",
        "## Why Duck Suggests Continuous Foot Bottom",
        "",
        "- Duck local XML exposes active foot contact candidates as `left_foot_bottom_tpu` and `right_foot_bottom_tpu` mesh collisions.",
        "- Duck does not expose explicit toe/center/heel primitive patches, so this workflow tests continuous bottom contact plus contact-point x-region classification.",
        "",
        "## Variant Design",
        "",
        "- `current_box_baseline`: current v5_22 foot collision box.",
        "- `discrete_toe_center_heel`: artifact-only discrete prototype patches.",
        "- `continuous_rocker_bottom`: artifact-only single ellipsoid bottom per foot.",
        "- `hybrid_continuous_with_region_classifier`: same continuous collision concept, classified by local contact x thirds.",
        "",
        "## Contact Persistence Comparison",
        "",
        f"- Best discrete persistence: `{summary['best_discrete_contact_persistence']}`",
        f"- Best continuous persistence: `{summary['best_continuous_contact_persistence']}`",
        "",
        "## Contact X Progression / Region Classification",
        "",
        "| variant | profile | posture | none rate | x mean | heel | center | toe | progression | result |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in results:
        lines.append(
            f"| `{row['foot_variant']}` | `{row['actuator_profile']}` | `{row['posture_case']}` | "
            f"{fmt(row['contact_none_rate'])} | {fmt(row['contact_x_mean'])} | "
            f"{fmt(row['heel_region_contact_ratio'])} | {fmt(row['center_region_contact_ratio'])} | "
            f"{fmt(row['toe_region_contact_ratio'])} | {fmt(row['contact_x_progression_score'])} | `{row['result_label']}` |"
        )
    lines.extend(["", "## Whether Continuous Bottom Improves Over Discrete Patches", ""])
    lines.append(f"- Continuous better than discrete by best persistence: `{summary['continuous_better_than_discrete']}`")
    lines.extend(["", "## Whether Toe Handoff / Rollover Is Physically Observable", ""])
    lines.append(f"- Continuous rollover candidate found: `{summary['continuous_rollover_candidate_found']}`")
    lines.append("- Classification uses prototype thresholds: rear/middle/front thirds in foot local x.")
    lines.extend(["", "## Recommended Next Step", ""])
    lines.append(f"- {summary['next_step']}")
    lines.extend(["", "## What Must Not Be Claimed", ""])
    lines.append("- Do not claim walking success.")
    lines.append("- Do not claim sim2real validity.")
    lines.append("- Do not treat continuous primitive as final mechanical design.")
    lines.append("- Do not treat `peak_upper_bound` as a continuous gait claim.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> int:
    """Run continuous foot bottom probe."""

    args = parse_args()
    config = load_json_object(args.config, "continuous foot bottom probe config")
    results, raw_contacts, metadata = run_probe(config)
    summary = summarize_results(results)
    payload = {
        "schema_version": 1,
        "summary": summary,
        "metadata": metadata,
        "results": results,
        "source_inputs": {
            "config": rel_path(args.config),
            "model_path": config["model_path"],
        },
    }
    artifacts_dir = resolve_repo_path(config["artifacts_dir"])
    result_fields = [
        "foot_variant",
        "posture_case",
        "actuator_profile",
        "contact_persistence",
        "contact_none_rate",
        "contact_x_mean",
        "contact_x_min",
        "contact_x_max",
        "contact_x_progression_score",
        "heel_region_contact_ratio",
        "center_region_contact_ratio",
        "toe_region_contact_ratio",
        "rollover_path_score",
        "toe_heel_bridge_like_pattern_detected",
        "fall_or_large_tilt",
        "base_height",
        "base_pitch",
        "result_label",
        "valid_for",
        "source",
        "threshold_source",
        "method",
    ]
    contact_fields = [
        "foot_variant",
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
    write_csv(artifacts_dir / "probe_results.csv", results, result_fields)
    write_csv(artifacts_dir / "raw_contacts.csv", raw_contacts, contact_fields)
    write_json(artifacts_dir / "metrics.json", payload)
    write_report(resolve_repo_path(config["report_path"]), payload)
    print(f"status={summary['status']}")
    print(f"result_rows={summary['result_rows']}")
    print(f"best_variant={summary['best_row']['foot_variant'] if summary['best_row'] else None}")
    print(f"best_contact_none_rate={summary['best_row']['contact_none_rate'] if summary['best_row'] else None}")
    print(f"continuous_better_than_discrete={summary['continuous_better_than_discrete']}")
    print(f"report={resolve_repo_path(config['report_path'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
