"""Run the Seedon v5_22 mechanical sanity gate.

This Class C diagnostic checks whether the current v5_22 MuJoCo model is a
usable mechanical baseline before controller, motor, actuator-envelope, or foot
contact sensitivity work. It does not modify source XML/URDF, does not call
existing extraction/readiness/foot prototype pipelines, and does not run PPO.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import mujoco
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only when MuJoCo is absent.
    mujoco = None
    _MUJOCO_IMPORT_ERROR: ModuleNotFoundError | None = exc
else:
    _MUJOCO_IMPORT_ERROR = None


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "seedon" / "seedon_v5_22_sanity_gate.yaml"
DEFAULT_ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "seedon_debug" / "v5_22_sanity_gate"
DEFAULT_REPORT_PATH = REPO_ROOT / "docs" / "seedon_v5_22_sanity_gate_report.md"
DEFAULT_MODEL_SUMMARY_PATH = DEFAULT_ARTIFACTS_DIR / "model_summary.yaml"
DEFAULT_METRICS_PATH = DEFAULT_ARTIFACTS_DIR / "metrics.json"
DEFAULT_CONTACTS_PATH = DEFAULT_ARTIFACTS_DIR / "raw_contacts.csv"

MOTOR_CONTROLLER_FIELDS = (
    "ctrlrange",
    "forcerange",
    "gear",
    "kp",
    "kd",
    "rated_torque",
    "peak_torque",
    "max_velocity",
    "control_mode",
    "current_limit",
    "encoder_resolution",
    "backlash",
)


@dataclass(frozen=True)
class ResolvedSource:
    """Resolved model source.

    Args:
        path: Model file path.
        source_type: Source type label.
        appears_v5_22: Whether the path appears to be Seedon v5_22.
        warnings: Source-resolution warnings.
    """

    path: Path | None
    source_type: str
    appears_v5_22: bool
    warnings: list[str]


def rel_path(path: Path | None) -> str | None:
    """Return repository-relative path text when possible."""

    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def load_config(path: Path) -> dict[str, Any]:
    """Load JSON-compatible YAML config.

    Args:
        path: Config path.

    Returns:
        Parsed config object.

    Raises:
        FileNotFoundError: If the config is absent.
        ValueError: If the file does not contain a JSON object.
    """

    if not path.is_file():
        raise FileNotFoundError(f"Sanity gate config not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Sanity gate config must decode to an object: {path}")
    return data


def resolve_model_source(config: dict[str, Any]) -> ResolvedSource:
    """Resolve the preferred loadable v5_22 model source without converting."""

    warnings: list[str] = []
    preferred = config.get("preferred_model_path")
    candidates = list(config.get("candidate_model_paths", []))
    ordered: list[str] = []
    if isinstance(preferred, str):
        ordered.append(preferred)
    ordered.extend(str(item) for item in candidates if str(item) not in ordered)

    for raw_path in ordered:
        path = (REPO_ROOT / raw_path).resolve() if not Path(raw_path).is_absolute() else Path(raw_path)
        if not path.is_file():
            continue
        source_type = classify_source_type(path)
        appears_v5_22 = "v5_22" in str(path).lower() or "seedon_urdf_5_22" in str(path).lower()
        if path.name == "training_scene.xml" and not appears_v5_22:
            warnings.append("training_scene.xml candidate does not appear to be v5_22.")
        return ResolvedSource(path=path, source_type=source_type, appears_v5_22=appears_v5_22, warnings=warnings)
    return ResolvedSource(path=None, source_type="unknown", appears_v5_22=False, warnings=["No candidate model file found."])


def classify_source_type(path: Path) -> str:
    """Classify one model path."""

    suffix = path.suffix.lower()
    if suffix == ".urdf":
        return "urdf"
    if suffix == ".xml" and "seedon_v5_22" in str(path).lower():
        return "converted_mjcf"
    if suffix == ".xml":
        return "mjcf"
    return "unknown"


def load_model(path: Path) -> Any:
    """Load a MuJoCo model from XML/URDF path."""

    if mujoco is None:
        raise RuntimeError(f"MuJoCo is not importable: {_MUJOCO_IMPORT_ERROR}")
    return mujoco.MjModel.from_xml_path(str(path))


def model_name(model: Any, obj_type: Any, index: int) -> str:
    """Return a MuJoCo object name."""

    return mujoco.mj_id2name(model, obj_type, int(index)) or f"<unnamed:{index}>"


def object_id(model: Any, obj_type: Any, name: str) -> int:
    """Return a MuJoCo object id or -1."""

    return int(mujoco.mj_name2id(model, obj_type, name))


def model_counts(model: Any) -> dict[str, int]:
    """Return core model counts."""

    return {
        "bodies": int(model.nbody),
        "joints": int(model.njnt),
        "actuators": int(model.nu),
        "geoms": int(model.ngeom),
        "sensors": int(model.nsensor),
        "keyframes": int(model.nkey),
    }


def expected_structure_check(model: Any, config: dict[str, Any]) -> dict[str, Any]:
    """Check joints, actuators, geoms, and right-leg mapping symmetry."""

    expected_joints = tuple(config["expected_joints"])
    missing_joints = [name for name in expected_joints if object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name) < 0]
    actuator_mapping = actuator_joint_mapping(model)
    actuated_joints = [item["joint"] for item in actuator_mapping]
    missing_actuated_joints = [name for name in expected_joints if name not in actuated_joints]
    unexpected_actuated_joints = [name for name in actuated_joints if name not in expected_joints]
    expected_geoms = tuple(config["expected_foot_geoms"]) + ("floor", "base_proxy")
    missing_geoms = [name for name in expected_geoms if object_id(model, mujoco.mjtObj.mjOBJ_GEOM, name) < 0]
    right_leg_warnings = right_leg_mapping_warnings(model)
    return {
        "expected_joints_present": not missing_joints,
        "expected_10_actuated_leg_joints_present": len(missing_actuated_joints) == 0 and model.nu == 10,
        "missing_joints": missing_joints,
        "missing_actuated_joints": missing_actuated_joints,
        "unexpected_actuated_joints": unexpected_actuated_joints,
        "missing_geoms": missing_geoms,
        "actuator_name_to_joint_mapping": actuator_mapping,
        "right_leg_mapping_warnings": right_leg_warnings,
        "severe_mapping_problem": bool(missing_joints or missing_actuated_joints or missing_geoms),
    }


def actuator_joint_mapping(model: Any) -> list[dict[str, str]]:
    """Return actuator name to joint name mapping."""

    rows: list[dict[str, str]] = []
    for actuator_id in range(model.nu):
        actuator_name = model_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        joint_name = model_name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) if joint_id >= 0 else "<none>"
        rows.append({"actuator": actuator_name, "joint": joint_name})
    return rows


def right_leg_mapping_warnings(model: Any) -> list[str]:
    """Return warnings for right-leg naming/order/axis asymmetry."""

    warnings: list[str] = []
    pairs = (
        ("R_joint_hip_yaw", "L_joint_hip_yaw"),
        ("R_joint_hip_roll", "L_joint_hip_roll"),
        ("R_joint_hip_pitch", "L_joint_hip_pitch"),
        ("R_joint_knee_pitch", "L_joint_knee_pitch"),
        ("R_joint_ankle_pitch", "L_joint_ankle_pitch"),
    )
    for right, left in pairs:
        right_id = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, right)
        left_id = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, left)
        if right_id < 0 or left_id < 0:
            warnings.append(f"Missing pair member: {right}/{left}")
            continue
        right_axis = np.array(model.jnt_axis[right_id], dtype=float)
        left_axis = np.array(model.jnt_axis[left_id], dtype=float)
        if not np.allclose(right_axis, left_axis, atol=1e-6):
            warnings.append(f"Axis asymmetry: {right}={right_axis.tolist()} {left}={left_axis.tolist()}")
        right_range = np.array(model.jnt_range[right_id], dtype=float)
        left_range = np.array(model.jnt_range[left_id], dtype=float)
        if not np.allclose(right_range, left_range, atol=1e-6):
            warnings.append(f"Range asymmetry: {right}={right_range.tolist()} {left}={left_range.tolist()}")
    return warnings


def reset_pose_sanity(model: Any, config: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    """Run reset-pose checks."""

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    roll, pitch, yaw = base_euler(data)
    joint_limit_report = joint_qpos_limit_report(model, data)
    contacts = current_contact_rows(model, data, step=0, phase="reset")
    foot_geoms = [name for name in config["expected_foot_geoms"] if object_id(model, mujoco.mjtObj.mjOBJ_GEOM, name) >= 0]
    return (
        {
            "base_height": float(data.qpos[2]) if model.nq >= 3 else None,
            "base_orientation_rpy": {"roll": roll, "pitch": pitch, "yaw": yaw},
            "joint_qpos_within_limits": joint_limit_report["within_limits"],
            "joint_limit_violations": joint_limit_report["violations"],
            "contact_count": int(data.ncon),
            "obvious_contact_explosion": int(data.ncon) > 40,
            "foot_geoms_detected": foot_geoms,
        },
        data,
    )


def joint_qpos_limit_report(model: Any, data: Any) -> dict[str, Any]:
    """Check hinge/slide joint qpos values against model ranges."""

    violations: list[dict[str, Any]] = []
    for joint_id in range(model.njnt):
        if int(model.jnt_limited[joint_id]) == 0:
            continue
        joint_type = int(model.jnt_type[joint_id])
        if joint_type not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
            continue
        qpos_adr = int(model.jnt_qposadr[joint_id])
        value = float(data.qpos[qpos_adr])
        lower, upper = [float(item) for item in model.jnt_range[joint_id]]
        if value < lower - 1e-6 or value > upper + 1e-6:
            violations.append(
                {
                    "joint": model_name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id),
                    "qpos": value,
                    "lower": lower,
                    "upper": upper,
                }
            )
    return {"within_limits": not violations, "violations": violations}


def base_euler(data: Any) -> tuple[float, float, float]:
    """Return base roll/pitch/yaw from freejoint quaternion."""

    if len(data.qpos) < 7:
        return (0.0, 0.0, 0.0)
    quat = np.array(data.qpos[3:7], dtype=float)
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-12:
        return (0.0, 0.0, 0.0)
    quat = quat / norm
    w, x, y, z = quat
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sinp = 2.0 * (w * y - z * x)
    pitch = math.asin(float(np.clip(sinp, -1.0, 1.0)))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return (float(roll), float(pitch), float(yaw))


def run_zero_action_settle(model: Any, config: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run a short zero-action simulation."""

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    data.ctrl[:] = 0.0
    return run_simulation_phase(model, data, phase="zero_action", steps=int(config["zero_action_steps"]), config=config)


def run_nominal_pd_hold(model: Any, config: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run conservative PD hold against reset qpos targets."""

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    target_qpos = data.qpos.copy()
    actuator_joint_ids = [int(model.actuator_trnid[index, 0]) for index in range(model.nu)]
    kp = float(config["pd_hold"]["kp"])
    kd = float(config["pd_hold"]["kd"])
    ctrlrange = np.array(model.actuator_ctrlrange, dtype=float)
    saturation_steps = 0
    clamp_count = 0
    total_ctrl_values = 0
    initial = phase_state(data)
    contacts: list[dict[str, Any]] = []
    exploded = False
    for step in range(int(config["pd_hold_steps"])):
        ctrl = np.zeros(model.nu, dtype=float)
        for actuator_id, joint_id in enumerate(actuator_joint_ids):
            if joint_id < 0:
                continue
            qpos_adr = int(model.jnt_qposadr[joint_id])
            dof_adr = int(model.jnt_dofadr[joint_id])
            raw = kp * (float(target_qpos[qpos_adr]) - float(data.qpos[qpos_adr])) - kd * float(data.qvel[dof_adr])
            clipped = float(np.clip(raw, ctrlrange[actuator_id, 0], ctrlrange[actuator_id, 1]))
            if abs(clipped - raw) > 1e-9:
                clamp_count += 1
            ctrl[actuator_id] = clipped
        data.ctrl[:] = ctrl
        if model.nu:
            lower_hit = ctrl <= ctrlrange[:, 0] + 1e-9
            upper_hit = ctrl >= ctrlrange[:, 1] - 1e-9
            saturation_steps += int(np.count_nonzero(lower_hit | upper_hit))
            total_ctrl_values += int(model.nu)
        try:
            mujoco.mj_step(model, data)
        except Exception:
            exploded = True
            break
        contacts.extend(current_contact_rows(model, data, step=step + 1, phase="nominal_pd_hold"))
        if not np.all(np.isfinite(data.qpos)) or float(np.max(np.abs(data.qpos))) > float(config["max_qpos_abs"]):
            exploded = True
            break
    summary = phase_summary(
        data=data,
        initial=initial,
        config=config,
        phase="nominal_pd_hold",
        steps_run=step + 1 if "step" in locals() else 0,
        contact_rows=contacts,
        exploded=exploded,
    )
    summary["pd_hold"] = {
        "kp": kp,
        "kd": kd,
        "source": "assumption",
        "confidence": "low",
        "valid_for": "sanity_gate_only",
    }
    summary["joint_target_clamp_rate"] = (
        float(clamp_count / max(total_ctrl_values, 1)) if total_ctrl_values else 0.0
    )
    summary["actuator_saturation_rate"] = (
        float(saturation_steps / max(total_ctrl_values, 1)) if total_ctrl_values else 0.0
    )
    return summary, contacts


def run_simulation_phase(
    model: Any,
    data: Any,
    *,
    phase: str,
    steps: int,
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run one simulation phase and summarize state drift."""

    initial = phase_state(data)
    contacts: list[dict[str, Any]] = []
    no_contact_steps = 0
    exploded = False
    for step in range(steps):
        try:
            mujoco.mj_step(model, data)
        except Exception:
            exploded = True
            break
        if int(data.ncon) == 0:
            no_contact_steps += 1
        contacts.extend(current_contact_rows(model, data, step=step + 1, phase=phase))
        if not np.all(np.isfinite(data.qpos)) or float(np.max(np.abs(data.qpos))) > float(config["max_qpos_abs"]):
            exploded = True
            break
    summary = phase_summary(
        data=data,
        initial=initial,
        config=config,
        phase=phase,
        steps_run=step + 1 if "step" in locals() else 0,
        contact_rows=contacts,
        exploded=exploded,
    )
    summary["contact_none_rate"] = float(no_contact_steps / max(summary["steps_run"], 1))
    return summary, contacts


def phase_state(data: Any) -> dict[str, Any]:
    """Capture base state."""

    roll, pitch, yaw = base_euler(data)
    return {
        "base_height": float(data.qpos[2]) if len(data.qpos) >= 3 else 0.0,
        "roll": roll,
        "pitch": pitch,
        "yaw": yaw,
    }


def phase_summary(
    *,
    data: Any,
    initial: dict[str, Any],
    config: dict[str, Any],
    phase: str,
    steps_run: int,
    contact_rows: list[dict[str, Any]],
    exploded: bool,
) -> dict[str, Any]:
    """Build drift and stability summary for one simulation phase."""

    final = phase_state(data)
    base_height_drift = float(final["base_height"] - initial["base_height"])
    roll_drift = angle_delta(final["roll"], initial["roll"])
    pitch_drift = angle_delta(final["pitch"], initial["pitch"])
    yaw_drift = angle_delta(final["yaw"], initial["yaw"])
    large_tilt_rad = math.radians(float(config["large_tilt_degrees"]))
    fall_or_large_tilt = abs(final["roll"]) > large_tilt_rad or abs(final["pitch"]) > large_tilt_rad
    unstable = exploded or abs(base_height_drift) > float(config["max_base_height_drift"]) or fall_or_large_tilt
    return {
        "phase": phase,
        "steps_run": int(steps_run),
        "initial_base_height": initial["base_height"],
        "final_base_height": final["base_height"],
        "base_height_drift": base_height_drift,
        "roll_drift": roll_drift,
        "pitch_drift": pitch_drift,
        "yaw_drift": yaw_drift,
        "final_roll": final["roll"],
        "final_pitch": final["pitch"],
        "final_yaw": final["yaw"],
        "contact_count_total": len(contact_rows),
        "fall_or_large_tilt": bool(fall_or_large_tilt),
        "unstable_or_exploding": bool(unstable),
        "exploded": bool(exploded),
    }


def angle_delta(value: float, reference: float) -> float:
    """Return wrapped angle difference."""

    return float(math.atan2(math.sin(value - reference), math.cos(value - reference)))


def current_contact_rows(model: Any, data: Any, *, step: int, phase: str) -> list[dict[str, Any]]:
    """Return current raw contact rows."""

    rows: list[dict[str, Any]] = []
    for contact_index in range(int(data.ncon)):
        contact = data.contact[contact_index]
        geom1 = model_name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1))
        geom2 = model_name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2))
        rows.append(
            {
                "phase": phase,
                "step": int(step),
                "contact_index": int(contact_index),
                "geom1": geom1,
                "geom2": geom2,
                "dist": float(contact.dist),
                "pos_x": float(contact.pos[0]),
                "pos_y": float(contact.pos[1]),
                "pos_z": float(contact.pos[2]),
                "involves_foot": is_foot_related(geom1) or is_foot_related(geom2),
            }
        )
    return rows


def is_foot_related(name: str) -> bool:
    """Return whether a geom name looks foot/ankle-related."""

    lowered = name.lower()
    return any(token in lowered for token in ("foot", "toe", "heel", "sole", "ankle"))


def foot_contact_observability(contact_rows: list[dict[str, Any]], model: Any) -> dict[str, Any]:
    """Classify foot contact observability from raw contact names."""

    foot_rows = [row for row in contact_rows if row["involves_foot"]]
    geom_names = [model_name(model, mujoco.mjtObj.mjOBJ_GEOM, index) for index in range(model.ngeom)]
    has_center = any("center" in name.lower() for name in geom_names)
    has_toe = any("toe" in name.lower() for name in geom_names)
    has_heel = any("heel" in name.lower() for name in geom_names)
    return {
        "foot_contact_pair_count": len(foot_rows),
        "foot_contact_pairs_observed": sorted({f"{row['geom1']}::{row['geom2']}" for row in foot_rows}),
        "center_toe_heel_classification_possible": bool(has_center and has_toe and has_heel),
        "contact_model_observable": bool(foot_rows),
        "classification_note": (
            "center/toe/heel labels found"
            if has_center and has_toe and has_heel
            else "Geom names are insufficient for center/toe/heel classification."
        ),
    }


def actuator_motor_field_check(model: Any) -> dict[str, Any]:
    """Check actuator/motor fields and mark missing specs manual_required."""

    ctrlrange_found = bool(model.nu > 0 and np.any(np.abs(model.actuator_ctrlrange) > 0))
    gear_found = bool(model.nu > 0 and np.any(np.abs(model.actuator_gear[:, 0]) != 1.0))
    fields = {field: "manual_required" for field in MOTOR_CONTROLLER_FIELDS}
    fields["ctrlrange"] = "FOUND" if ctrlrange_found else "manual_required"
    fields["gear"] = "FOUND" if gear_found else "manual_required"
    return {
        "fields": fields,
        "partial_actuator_envelope": bool(ctrlrange_found),
        "note": "Missing motor/controller fields remain manual_required; no values are inferred.",
    }


def decide_gate(
    *,
    loadable: bool,
    structure: dict[str, Any] | None,
    reset: dict[str, Any] | None,
    zero_action: dict[str, Any] | None,
    pd_hold: dict[str, Any] | None,
    foot: dict[str, Any] | None,
    actuator: dict[str, Any] | None,
) -> str:
    """Apply gate decision rules."""

    if not loadable:
        return "BLOCKED"
    if structure and structure["severe_mapping_problem"]:
        return "FAIL"
    if reset and (not reset["joint_qpos_within_limits"] or reset["obvious_contact_explosion"]):
        return "FAIL"
    if zero_action and zero_action["exploded"]:
        return "FAIL"
    if pd_hold and pd_hold["unstable_or_exploding"]:
        return "FAIL"
    if foot and not foot["contact_model_observable"]:
        return "FAIL"
    if actuator and any(status == "manual_required" for field, status in actuator["fields"].items() if field != "ctrlrange"):
        return "PARTIAL_PASS"
    if zero_action and zero_action["fall_or_large_tilt"]:
        return "PARTIAL_PASS"
    return "PASS"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON payload."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_contacts(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write raw contact CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["phase", "step", "contact_index", "geom1", "geom2", "dist", "pos_x", "pos_y", "pos_z", "involves_foot"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def write_report(path: Path, payload: dict[str, Any]) -> None:
    """Write Markdown sanity gate report."""

    path.parent.mkdir(parents=True, exist_ok=True)
    summary = payload["summary"]
    source = payload["model_source_resolution"]
    counts = payload.get("model_load", {}).get("counts", {})
    lines = [
        "# Seedon v5_22 Mechanical Sanity Gate Report",
        "",
        "Task class: Class C mechanical sanity diagnostic. This report does not claim walking success and does not run PPO.",
        "",
        "## Summary",
        "",
        f"- Gate decision: `{summary['gate_decision']}`",
        f"- Source path: `{source.get('source_path')}`",
        f"- Source type: `{source.get('source_type')}`",
        f"- Appears v5_22: `{source.get('appears_v5_22')}`",
        f"- Bodies / joints / geoms / actuators: `{counts.get('bodies')}` / `{counts.get('joints')}` / `{counts.get('geoms')}` / `{counts.get('actuators')}`",
        "",
        "## Model Source Resolution",
        "",
    ]
    for key, value in source.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Model Load", ""])
    model_load = payload.get("model_load", {})
    for key, value in model_load.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Expected Structure", ""])
    structure = payload.get("expected_structure_check", {})
    lines.append(f"- Expected joints present: `{structure.get('expected_joints_present')}`")
    lines.append(f"- Expected 10 actuated leg joints present: `{structure.get('expected_10_actuated_leg_joints_present')}`")
    lines.append(f"- Missing joints: `{structure.get('missing_joints')}`")
    lines.append(f"- Missing geoms: `{structure.get('missing_geoms')}`")
    lines.append(f"- Right-leg mapping warnings: `{structure.get('right_leg_mapping_warnings')}`")
    lines.extend(["", "## Reset Pose", ""])
    for key, value in payload.get("reset_pose_sanity", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Zero-Action Settle", ""])
    for key, value in payload.get("zero_action_settle", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Nominal PD Hold", ""])
    for key, value in payload.get("nominal_pd_hold", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Foot Contact Observability", ""])
    for key, value in payload.get("foot_contact_pair_dump", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Actuator / Motor Missing Fields", ""])
    for key, value in payload.get("actuator_motor_missing_field_check", {}).get("fields", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Recommendation", ""])
    lines.append(f"- Can proceed to controller / motor gap closure: `{summary['can_proceed_to_controller_motor_gap_closure']}`")
    lines.append("- Keep all missing motor/controller values as `manual_required` until external specs are provided.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_gate(config: dict[str, Any]) -> dict[str, Any]:
    """Run the full sanity gate and return metrics payload."""

    artifacts_dir = (REPO_ROOT / config.get("artifacts_dir", str(DEFAULT_ARTIFACTS_DIR))).resolve()
    source = resolve_model_source(config)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "version": config.get("version", "v5_22"),
        "model_source_resolution": {
            "source_path": rel_path(source.path),
            "source_type": source.source_type,
            "appears_v5_22": source.appears_v5_22,
            "warnings": source.warnings,
        },
    }
    all_contacts: list[dict[str, Any]] = []
    loadable = False
    if source.path is None:
        payload["model_load"] = {"success": False, "error": "No loadable model candidate found."}
        gate = "BLOCKED"
    else:
        try:
            model = load_model(source.path)
            loadable = True
            payload["model_load"] = {"success": True, "counts": model_counts(model)}
            payload["expected_structure_check"] = expected_structure_check(model, config)
            reset, _ = reset_pose_sanity(model, config)
            payload["reset_pose_sanity"] = reset
            zero_action, zero_contacts = run_zero_action_settle(model, config)
            pd_hold, pd_contacts = run_nominal_pd_hold(model, config)
            all_contacts = zero_contacts + pd_contacts
            payload["zero_action_settle"] = zero_action
            payload["nominal_pd_hold"] = pd_hold
            payload["foot_contact_pair_dump"] = foot_contact_observability(all_contacts, model)
            payload["actuator_motor_missing_field_check"] = actuator_motor_field_check(model)
            gate = decide_gate(
                loadable=loadable,
                structure=payload["expected_structure_check"],
                reset=payload["reset_pose_sanity"],
                zero_action=payload["zero_action_settle"],
                pd_hold=payload["nominal_pd_hold"],
                foot=payload["foot_contact_pair_dump"],
                actuator=payload["actuator_motor_missing_field_check"],
            )
        except Exception as exc:  # noqa: BLE001 - diagnostic must report blocking load failures.
            payload["model_load"] = {"success": False, "error": f"{type(exc).__name__}: {exc}"}
            gate = "BLOCKED"
    payload["summary"] = {
        "gate_decision": gate,
        "can_proceed_to_controller_motor_gap_closure": gate in {"PASS", "PARTIAL_PASS"},
        "walking_success_claimed": False,
        "artifacts_dir": rel_path(artifacts_dir),
    }
    write_contacts(artifacts_dir / "raw_contacts.csv", all_contacts)
    write_json(artifacts_dir / "metrics.json", payload)
    write_json(artifacts_dir / "model_summary.yaml", summarize_model_payload(payload))
    report_path = (REPO_ROOT / config.get("report_path", str(DEFAULT_REPORT_PATH))).resolve()
    write_report(report_path, payload)
    return payload


def summarize_model_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return compact model summary payload."""

    return {
        "version": payload.get("version"),
        "source": payload.get("model_source_resolution"),
        "model_load": payload.get("model_load"),
        "expected_structure_check": payload.get("expected_structure_check"),
        "gate_decision": payload.get("summary", {}).get("gate_decision"),
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def main() -> int:
    """Run the v5_22 mechanical sanity gate."""

    args = parse_args()
    config = load_config(args.config)
    payload = run_gate(config)
    print(f"gate_decision={payload['summary']['gate_decision']}")
    print(f"source={payload['model_source_resolution']['source_path']}")
    print(f"metrics={DEFAULT_METRICS_PATH}")
    print(f"report={config.get('report_path', str(DEFAULT_REPORT_PATH))}")
    return 0 if payload["summary"]["gate_decision"] != "BLOCKED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
