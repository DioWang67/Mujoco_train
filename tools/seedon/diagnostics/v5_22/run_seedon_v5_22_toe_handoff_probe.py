"""Run Seedon v5_22 toe handoff targeted probe.

This Class C diagnostic sweeps controlled posture cases over artifact-only
v5_22 foot contact variants. It checks whether prototype patches can expose
center-to-toe load transfer. It does not modify source XML/URDF, train.py,
eval.py, env.py runtime behavior, and does not run PPO or claim walking
success.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.seedon.diagnostics.v5_22.run_seedon_v5_22_foot_actuator_sensitivity import (
    REGIONS,
    SIDES,
    can_classify_center_toe_heel,
    classify_joint_group,
    contact_force,
    create_v5_22_variants,
    is_floor_name,
    is_foot_name,
    load_json_object,
    model_name,
    patch_inventory,
    quat_from_roll_pitch,
    region_for_geom,
    rel_path,
    require_mujoco,
    resolve_repo_path,
    side_for_geom,
)

try:
    import mujoco
except ModuleNotFoundError as exc:  # pragma: no cover - only when MuJoCo is unavailable.
    mujoco = None
    _MUJOCO_IMPORT_ERROR: ModuleNotFoundError | None = exc
else:
    _MUJOCO_IMPORT_ERROR = None


DEFAULT_CONFIG = REPO_ROOT / "configs" / "seedon" / "seedon_v5_22_toe_handoff_probe.yaml"
FORCE_EPS = 1e-5
FORCE_REGIONS = ("toe", "center", "heel")


def object_id(model: Any, obj_type: Any, name: str) -> int:
    """Return MuJoCo object id or -1."""

    return int(mujoco.mj_name2id(model, obj_type, name))


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


def profile_limits_for_model(model: Any, profile: dict[str, Any]) -> np.ndarray:
    """Return per-actuator bounded diagnostic command limits."""

    limits = np.zeros(model.nu, dtype=np.float64)
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        joint_name = model_name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) if joint_id >= 0 else ""
        limits[actuator_id] = float(profile[classify_joint_group(joint_name)])
    return limits


def clamp_joint_qpos(model: Any, data: Any, joint_name: str, delta: float) -> bool:
    """Apply one joint qpos delta and return whether it was clamped."""

    joint_id = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        return False
    qpos_adr = int(model.jnt_qposadr[joint_id])
    raw_value = float(data.qpos[qpos_adr] + delta)
    if int(model.jnt_limited[joint_id]):
        lower, upper = [float(value) for value in model.jnt_range[joint_id]]
        value = float(np.clip(raw_value, lower, upper))
    else:
        value = raw_value
    data.qpos[qpos_adr] = value
    return abs(value - raw_value) > 1e-9


def apply_posture_case(model: Any, data: Any, posture_case: dict[str, Any]) -> float:
    """Reset model and apply one controlled posture case.

    Args:
        model: MuJoCo model.
        data: MuJoCo data.
        posture_case: Posture case config.

    Returns:
        Joint target clamp rate for posture biases.
    """

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    if len(data.qpos) >= 7:
        data.qpos[0] += float(posture_case.get("base_x_shift", 0.0))
        data.qpos[3:7] = quat_from_roll_pitch(0.0, math.radians(float(posture_case["base_pitch_degrees"])))
    clamp_count = 0
    bias_count = 0
    for joint_name, delta in posture_case.get("joint_biases", {}).items():
        bias_count += 1
        clamp_count += int(clamp_joint_qpos(model, data, str(joint_name), float(delta)))
    data.qvel[:] = 0.0
    if data.ctrl.size:
        data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)
    return float(clamp_count / max(bias_count, 1)) if bias_count else 0.0


def classify_contact_pair(name1: str, name2: str) -> tuple[str, str]:
    """Return side and patch region for a floor/foot contact pair."""

    for name in (name1, name2):
        if is_foot_name(name):
            return side_for_geom(name), region_for_geom(name)
    return "unknown", "unknown"


def collect_contacts(
    model: Any,
    data: Any,
    *,
    foot_variant: str,
    actuator_profile: str,
    posture_case: str,
    step: int,
) -> list[dict[str, Any]]:
    """Collect current raw contact rows."""

    rows: list[dict[str, Any]] = []
    for contact_index in range(int(data.ncon)):
        contact = data.contact[contact_index]
        geom1 = model_name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1))
        geom2 = model_name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2))
        floor_contact = is_floor_name(geom1) or is_floor_name(geom2)
        side, region = classify_contact_pair(geom1, geom2) if floor_contact else ("unknown", "unknown")
        rows.append(
            {
                "foot_variant": foot_variant,
                "actuator_profile": actuator_profile,
                "posture_case": posture_case,
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
                "method": "raw_mujoco_contact_normal_force",
            }
        )
    return rows


def run_posture_probe(
    model: Any,
    *,
    foot_variant: str,
    profile_name: str,
    profile: dict[str, Any],
    posture_case: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run one foot/profile/posture probe case."""

    sim = config["simulation"]
    thresholds = config["prototype_thresholds"]
    data = mujoco.MjData(model)
    posture_clamp_rate = apply_posture_case(model, data, posture_case)
    target_qpos = data.qpos.copy()
    initial_height = float(data.qpos[2]) if len(data.qpos) >= 3 else 0.0
    actuator_joint_ids = [int(model.actuator_trnid[index, 0]) for index in range(model.nu)]
    ctrlrange = np.array(model.actuator_ctrlrange, dtype=np.float64)
    profile_limits = profile_limits_for_model(model, profile)
    kp = float(sim["pd_stiffness"])
    kd = float(sim["pd_damping"])
    hold_steps = int(sim["hold_steps"])
    aggregation_start = max(1, hold_steps - int(sim["aggregation_last_steps"]) + 1)
    saturation_count = 0
    ankle_saturation_count = 0
    total_ctrl_values = 0
    ankle_total = 0
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
            raw = kp * (float(target_qpos[qpos_adr]) - float(data.qpos[qpos_adr])) - kd * float(data.qvel[dof_adr])
            mjcf_clipped = float(np.clip(raw, ctrlrange[actuator_id, 0], ctrlrange[actuator_id, 1]))
            ctrl[actuator_id] = float(np.clip(mjcf_clipped, -profile_limits[actuator_id], profile_limits[actuator_id]))
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
        rows = collect_contacts(
            model,
            data,
            foot_variant=foot_variant,
            actuator_profile=profile_name,
            posture_case=str(posture_case["name"]),
            step=step,
        )
        raw_contacts.extend(rows)
        if step >= aggregation_start:
            aggregate_contacts.extend(rows)
        if not np.all(np.isfinite(data.qpos)) or float(np.max(np.abs(data.qpos))) > float(sim["max_qpos_abs"]):
            exploded = True
            break
    final_height = float(data.qpos[2]) if len(data.qpos) >= 3 else 0.0
    final_roll, final_pitch, _ = base_euler(data)
    large_tilt_rad = math.radians(float(thresholds["large_tilt_degrees"]))
    fall_or_large_tilt = bool(exploded or abs(final_roll) > large_tilt_rad or abs(final_pitch) > large_tilt_rad)
    force_metrics = build_force_metrics(aggregate_contacts)
    result = {
        "foot_variant": foot_variant,
        "actuator_profile": profile_name,
        "posture_case": posture_case["name"],
        **force_metrics,
        "contact_none_rate": float(no_contact_steps / max(hold_steps, 1)),
        "base_pitch": final_pitch,
        "base_height": final_height,
        "base_height_drift": float(final_height - initial_height),
        "fall_or_large_tilt": fall_or_large_tilt,
        "tilt_max": float(tilt_max),
        "actuator_saturation_rate": float(saturation_count / max(total_ctrl_values, 1)),
        "ankle_pitch_saturation_rate": float(ankle_saturation_count / max(ankle_total, 1)),
        "joint_target_clamp_rate": posture_clamp_rate,
        "valid_for": profile["valid_for"],
        "source": profile["source"],
        "confidence": profile["confidence"],
        "torque_side": profile["torque_side"],
        "threshold_source": thresholds["source"],
        "threshold_confidence": thresholds["confidence"],
        "method": "raw_mujoco_contact_normal_force",
    }
    result["result_label"] = classify_result(result, thresholds)
    return result, raw_contacts


def build_force_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build per-side patch forces and aggregate force ratios."""

    side_region_force = {f"{side}_{region}_normal_force": 0.0 for side in SIDES for region in FORCE_REGIONS}
    for row in rows:
        if not row["is_floor_contact"]:
            continue
        side = str(row["classified_side"])
        region = str(row["classified_region"])
        if side in SIDES and region in FORCE_REGIONS:
            side_region_force[f"{side}_{region}_normal_force"] += float(row["normal_force"])
    for side in SIDES:
        side_region_force[f"{side}_total_foot_normal_force"] = sum(
            side_region_force[f"{side}_{region}_normal_force"] for region in FORCE_REGIONS
        )
    total_toe = sum(side_region_force[f"{side}_toe_normal_force"] for side in SIDES)
    total_center = sum(side_region_force[f"{side}_center_normal_force"] for side in SIDES)
    total_heel = sum(side_region_force[f"{side}_heel_normal_force"] for side in SIDES)
    total = total_toe + total_center + total_heel
    if total <= FORCE_EPS:
        ratios: dict[str, Any] = {
            "toe_force_ratio": "unavailable",
            "center_force_ratio": "unavailable",
            "heel_force_ratio": "unavailable",
            "center_to_toe_transfer_score": "unavailable",
            "rollover_observability_score": "unavailable",
            "toe_heel_bridge_detected": "unavailable",
        }
    else:
        toe_ratio = float(total_toe / total)
        center_ratio = float(total_center / total)
        heel_ratio = float(total_heel / total)
        ratios = {
            "toe_force_ratio": toe_ratio,
            "center_force_ratio": center_ratio,
            "heel_force_ratio": heel_ratio,
            "center_to_toe_transfer_score": float(toe_ratio - center_ratio),
            "rollover_observability_score": float(toe_ratio * (1.0 - heel_ratio)),
            "toe_heel_bridge_detected": any(
                side_region_force[f"{side}_toe_normal_force"] > FORCE_EPS
                and side_region_force[f"{side}_heel_normal_force"] > FORCE_EPS
                for side in SIDES
            ),
        }
    return {**side_region_force, **ratios}


def classify_result(result: dict[str, Any], thresholds: dict[str, Any]) -> str:
    """Classify one probe result using prototype thresholds."""

    required = ("toe_force_ratio", "center_force_ratio", "heel_force_ratio")
    if any(result[field] == "unavailable" for field in required):
        return "force_unavailable"
    if result["fall_or_large_tilt"]:
        return "posture_unstable"
    if float(result["contact_none_rate"]) > float(thresholds["contact_none_rate_max"]):
        return "insufficient_contact_persistence"
    if result["toe_heel_bridge_detected"] is True:
        return "toe_heel_bridge_detected"
    if (
        float(result["toe_force_ratio"]) > float(thresholds["toe_force_ratio_min"])
        and float(result["center_force_ratio"]) < float(thresholds["center_force_ratio_max"])
        and float(result["heel_force_ratio"]) < float(thresholds["heel_force_ratio_max"])
    ):
        return "toe_handoff_candidate"
    return "no_toe_handoff_candidate"


def run_probe(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Run the full toe handoff targeted probe matrix."""

    if mujoco is None:
        raise RuntimeError(f"MuJoCo is not importable: {_MUJOCO_IMPORT_ERROR}")
    require_mujoco()
    variants = create_v5_22_variants(config)
    results: list[dict[str, Any]] = []
    raw_contacts: list[dict[str, Any]] = []
    inventories: dict[str, list[dict[str, Any]]] = {}
    for variant in variants:
        model = mujoco.MjModel.from_xml_path(str(variant.scene_path))
        inventory = patch_inventory(model, variant)
        inventories[variant.name] = inventory
        if not can_classify_center_toe_heel(inventory):
            continue
        for profile_name, profile in config["actuator_profiles"].items():
            for posture_case in config["posture_cases"]:
                row, contacts = run_posture_probe(
                    model,
                    foot_variant=variant.name,
                    profile_name=profile_name,
                    profile=profile,
                    posture_case=posture_case,
                    config=config,
                )
                results.append(row)
                raw_contacts.extend(contacts)
    metadata = {
        "schema_version": 1,
        "version": config["version"],
        "valid_for": config["valid_for"],
        "invalid_for": config["invalid_for"],
        "method_limitations": config["method_limitations"],
        "prototype_thresholds": config["prototype_thresholds"],
        "foot_variant_inventory": inventories,
        "posture_cases": config["posture_cases"],
        "actuator_profiles": config["actuator_profiles"],
    }
    return results, raw_contacts, metadata


def numeric_value(value: Any, default: float = -1.0) -> float:
    """Return float value or default for unavailable fields."""

    if value == "unavailable":
        return default
    return float(value)


def candidate_rank(row: dict[str, Any]) -> tuple[int, float, float, float]:
    """Return a ranking tuple for best candidate selection."""

    is_candidate = int(row["result_label"] == "toe_handoff_candidate")
    return (
        is_candidate,
        numeric_value(row["rollover_observability_score"]),
        numeric_value(row["center_to_toe_transfer_score"]),
        -float(row["ankle_pitch_saturation_rate"]),
    )


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Build high-level probe summary."""

    candidates = [row for row in results if row["result_label"] == "toe_handoff_candidate"]
    best = max(results, key=candidate_rank) if results else None
    by_profile = aggregate_by_key(results, "actuator_profile")
    by_variant = aggregate_by_key(results, "foot_variant")
    return {
        "status": "TOE_HANDOFF_PROBE_COMPLETE" if results else "BLOCKED",
        "probe_rows": len(results),
        "toe_handoff_candidate_found": bool(candidates),
        "candidate_count": len(candidates),
        "best_row": best,
        "ankle_boost_changes_force_transfer": force_transfer_changed(results, "actuator_profile", "ankle_boost_hypothesis", "rated_safe"),
        "foot_variant_changes_force_transfer": force_transfer_changed(results, "foot_variant", "duck_like_multi_patch", "simple_toe_center_heel"),
        "aggregate_by_profile": by_profile,
        "aggregate_by_variant": by_variant,
        "next_recommendation": next_recommendation(candidates),
    }


def aggregate_by_key(results: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    """Aggregate average force ratios by one result key."""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        grouped.setdefault(str(row[key]), []).append(row)
    output: dict[str, dict[str, Any]] = {}
    for name, rows in grouped.items():
        output[name] = {
            "rows": len(rows),
            "candidate_count": sum(1 for row in rows if row["result_label"] == "toe_handoff_candidate"),
            "avg_toe_force_ratio": average_available(rows, "toe_force_ratio"),
            "avg_center_force_ratio": average_available(rows, "center_force_ratio"),
            "avg_heel_force_ratio": average_available(rows, "heel_force_ratio"),
            "avg_rollover_observability_score": average_available(rows, "rollover_observability_score"),
        }
    return output


def average_available(rows: list[dict[str, Any]], field: str) -> float | str:
    """Average one numeric field, returning unavailable when none exists."""

    values = [float(row[field]) for row in rows if row[field] != "unavailable"]
    if not values:
        return "unavailable"
    return float(sum(values) / len(values))


def force_transfer_changed(results: list[dict[str, Any]], key: str, left_value: str, right_value: str) -> bool:
    """Return whether two groups show different force transfer labels or ratios."""

    left_rows = [row for row in results if row[key] == left_value]
    right_rows = [row for row in results if row[key] == right_value]
    if not left_rows or not right_rows:
        return False
    left_avg = average_available(left_rows, "toe_force_ratio")
    right_avg = average_available(right_rows, "toe_force_ratio")
    if left_avg == "unavailable" or right_avg == "unavailable":
        return False
    left_candidates = sum(1 for row in left_rows if row["result_label"] == "toe_handoff_candidate")
    right_candidates = sum(1 for row in right_rows if row["result_label"] == "toe_handoff_candidate")
    return bool(left_candidates != right_candidates or abs(float(left_avg) - float(right_avg)) > 0.05)


def next_recommendation(candidates: list[dict[str, Any]]) -> str:
    """Return conservative next recommendation."""

    if candidates:
        return "controller gait sequencing can be scripted next, still bounded and no PPO."
    return "foot geometry tuning first; targeted posture probes did not expose center-to-toe handoff."


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


def write_report(path: Path, payload: dict[str, Any]) -> None:
    """Write the toe handoff probe report."""

    path.parent.mkdir(parents=True, exist_ok=True)
    summary = payload["summary"]
    results = payload["probe_results"]
    best = summary["best_row"] or {}
    lines = [
        "# Seedon v5_22 Toe Handoff Targeted Probe Report",
        "",
        "Task class: Class C bounded diagnostic. This report does not run PPO, does not claim walking success, and does not create verified motor specs.",
        "",
        "## Summary",
        "",
        f"- Status: `{summary['status']}`",
        f"- Probe rows: `{summary['probe_rows']}`",
        f"- Toe handoff candidate found: `{summary['toe_handoff_candidate_found']}`",
        f"- Candidate count: `{summary['candidate_count']}`",
        f"- Best row: `{best.get('foot_variant')} / {best.get('posture_case')} / {best.get('actuator_profile')}`",
        f"- Next recommendation: {summary['next_recommendation']}",
        "",
        "## Probe Setup",
        "",
        f"- Foot variants: `{sorted({row['foot_variant'] for row in results})}`",
        f"- Actuator profiles: `{sorted({row['actuator_profile'] for row in results})}`",
        f"- Posture cases: `{sorted({row['posture_case'] for row in results})}`",
        f"- Prototype thresholds: `{payload['metadata']['prototype_thresholds']}`",
        "- Method limitation: MuJoCo contact force is read from raw contact normal force via `mj_contactForce`; patch attribution depends on prototype geom names.",
        "",
        "## Best Toe Handoff Candidates",
        "",
    ]
    candidates = [row for row in results if row["result_label"] == "toe_handoff_candidate"]
    if not candidates:
        lines.append("- No toe handoff candidate met the prototype thresholds.")
    else:
        for row in sorted(candidates, key=candidate_rank, reverse=True)[:5]:
            lines.append(
                f"- `{row['foot_variant']} / {row['posture_case']} / {row['actuator_profile']}`: "
                f"toe={row['toe_force_ratio']}, center={row['center_force_ratio']}, heel={row['heel_force_ratio']}"
            )
    lines.extend(
        [
            "",
            "## Force Ratio Table",
            "",
            "| foot variant | profile | posture | toe ratio | center ratio | heel ratio | transfer score | observable score | result |",
            "|---|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in results:
        lines.append(
            f"| `{row['foot_variant']}` | `{row['actuator_profile']}` | `{row['posture_case']}` | "
            f"{format_metric(row['toe_force_ratio'])} | {format_metric(row['center_force_ratio'])} | "
            f"{format_metric(row['heel_force_ratio'])} | {format_metric(row['center_to_toe_transfer_score'])} | "
            f"{format_metric(row['rollover_observability_score'])} | `{row['result_label']}` |"
        )
    lines.extend(["", "## Whether Ankle Boost Changes Force Transfer", ""])
    lines.append(f"- Changes force transfer versus rated_safe: `{summary['ankle_boost_changes_force_transfer']}`")
    lines.extend(["", "## Whether Foot Variant Changes Force Transfer", ""])
    lines.append(f"- Changes force transfer between variants: `{summary['foot_variant_changes_force_transfer']}`")
    lines.extend(["", "## Whether Toe Handoff Is Physically Observable", ""])
    if summary["toe_handoff_candidate_found"]:
        lines.append("- Toe handoff is observable only under prototype thresholds and raw MuJoCo contact-force attribution.")
    else:
        lines.append("- Toe handoff is not observable under the current controlled posture sweep and prototype thresholds.")
    lines.extend(["", "## If Not Observable, Recommended Foot Geometry Tuning", ""])
    lines.append("- Increase toe patch ability to carry load without simultaneously loading center/heel.")
    lines.append("- Sweep toe patch height and x-offset before changing gait rewards.")
    lines.append("- Keep center/toe/heel patch names explicit so diagnostics remain observable.")
    lines.extend(["", "## What Must Not Be Claimed", ""])
    lines.append("- Do not claim walking success.")
    lines.append("- Do not claim sim2real validity.")
    lines.append("- Do not treat provided torque as verified joint-output forcerange.")
    lines.append("- Do not treat `peak_upper_bound` as a continuous gait claim.")
    lines.extend(["", "## Next Recommendation", ""])
    lines.append(f"- {summary['next_recommendation']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_metric(value: Any) -> str:
    """Format a metric for Markdown."""

    if value == "unavailable":
        return "`unavailable`"
    return f"{float(value):.6g}"


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> int:
    """Run the toe handoff targeted probe."""

    args = parse_args()
    config = load_json_object(args.config, "toe handoff probe config")
    results, raw_contacts, metadata = run_probe(config)
    summary = summarize_results(results)
    payload = {
        "schema_version": 1,
        "summary": summary,
        "metadata": metadata,
        "probe_results": results,
        "source_inputs": {
            "config": rel_path(args.config),
            "model_path": config["model_path"],
            "foot_profile_path": config["foot_profile_path"],
        },
    }
    artifacts_dir = resolve_repo_path(config["artifacts_dir"])
    result_fields = [
        "foot_variant",
        "actuator_profile",
        "posture_case",
        "left_toe_normal_force",
        "left_center_normal_force",
        "left_heel_normal_force",
        "left_total_foot_normal_force",
        "right_toe_normal_force",
        "right_center_normal_force",
        "right_heel_normal_force",
        "right_total_foot_normal_force",
        "toe_force_ratio",
        "center_force_ratio",
        "heel_force_ratio",
        "center_to_toe_transfer_score",
        "rollover_observability_score",
        "contact_none_rate",
        "toe_heel_bridge_detected",
        "base_pitch",
        "base_height",
        "base_height_drift",
        "fall_or_large_tilt",
        "actuator_saturation_rate",
        "ankle_pitch_saturation_rate",
        "joint_target_clamp_rate",
        "result_label",
        "valid_for",
        "source",
        "confidence",
        "torque_side",
        "threshold_source",
        "threshold_confidence",
        "method",
    ]
    contact_fields = [
        "foot_variant",
        "actuator_profile",
        "posture_case",
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
        "method",
    ]
    write_csv(artifacts_dir / "probe_results.csv", results, result_fields)
    write_csv(artifacts_dir / "raw_contacts.csv", raw_contacts, contact_fields)
    write_json(artifacts_dir / "metrics.json", payload)
    write_report(resolve_repo_path(config["report_path"]), payload)
    print(f"status={summary['status']}")
    print(f"probe_rows={summary['probe_rows']}")
    print(f"toe_handoff_candidate_found={summary['toe_handoff_candidate_found']}")
    print(f"output={artifacts_dir / 'metrics.json'}")
    print(f"report={resolve_repo_path(config['report_path'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
