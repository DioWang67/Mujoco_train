"""Phase M0 Sedon vs Open Duck morphology and gait-reference audit.

This diagnostic compares MuJoCo model structure, contact geometry style,
actuator metadata, approximate morphology, and high-level Duck gait metrics.
It is intentionally a local analysis tool: it does not train, does not modify
Sedon training code, and does not rewrite source XML files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import mujoco
import numpy as np

from tools.sedon_debug_common import DEBUG_OUT_DIR, geom_name, geom_type_name, require_scene


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEDON_SCENE = DEBUG_OUT_DIR / "blue_like_sole_experiments_v5" / "training_scene_v5_a.xml"
DEFAULT_DUCK_XML = Path(
    "C:/Users/diowang/open_duck_mini_ws/Open_Duck_Playground/"
    "playground/open_duck_mini_v2/xmls/open_duck_mini_v2.xml"
)
DEFAULT_DUCK_REFERENCE_JSON = Path(
    "C:/Users/diowang/open_duck_mini_ws/Open_Duck_Playground/"
    "artifacts/open_duck_mini_debug/open_duck_vs_sedon_contact_reference.json"
)
DEFAULT_OUTPUT_DIR = DEBUG_OUT_DIR / "phase_m0_duck_morphology_audit"

FOOT_TOKENS = ("foot", "toe", "heel", "center", "sole", "bottom", "collision")
LEG_DOF_TOKENS = ("hip_yaw", "hip_roll", "hip_pitch", "knee_pitch", "ankle_pitch", "ankle_roll")
DUCK_FALLBACK = {
    "target_vx": 0.10,
    "gait_period": 0.48,
    "foot_clearance": 0.023,
    "max_abs_roll": 0.08,
    "max_abs_pitch": 0.08,
    "double_support_ratio": 0.50,
    "flight_ratio": 0.0,
    "action_scale": 0.25,
}


@dataclass(frozen=True)
class AuditConfig:
    """Runtime configuration for the Phase M0 audit."""

    sedon_scene_path: Path
    duck_xml_path: Path
    duck_reference_json: Path
    output_dir: Path
    steps_for_joint_probe: int
    epsilon: float


def mj_name(model: mujoco.MjModel, obj_type: mujoco.mjtObj, index: int, fallback: str) -> str:
    """Return a stable MuJoCo object name."""
    name = mujoco.mj_id2name(model, obj_type, int(index))
    return name or f"{fallback}_{index}"


def body_name(model: mujoco.MjModel, body_id: int) -> str:
    """Return body name for an id."""
    return mj_name(model, mujoco.mjtObj.mjOBJ_BODY, body_id, "body")


def joint_name(model: mujoco.MjModel, joint_id: int) -> str:
    """Return joint name for an id."""
    return mj_name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id, "joint")


def actuator_name(model: mujoco.MjModel, actuator_id: int) -> str:
    """Return actuator name for an id."""
    return mj_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id, "actuator")


def site_name(model: mujoco.MjModel, site_id: int) -> str:
    """Return site name for an id."""
    return mj_name(model, mujoco.mjtObj.mjOBJ_SITE, site_id, "site")


def sensor_name(model: mujoco.MjModel, sensor_id: int) -> str:
    """Return sensor name for an id."""
    return mj_name(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_id, "sensor")


def csv_vec(values: Iterable[Any]) -> str:
    """Format vector values for stable CSV output."""
    return " ".join(f"{float(value):.9g}" for value in values)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """Write CSV rows with a stable header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write an indented JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_model(path: Path) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Load a MuJoCo model and forward a neutral state."""
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    return model, data


def infer_side(name: str) -> str:
    """Infer left/right side from common robot naming patterns."""
    lowered = name.lower()
    if lowered.startswith("r_") or lowered.startswith("right") or "_r_" in lowered:
        return "right"
    if lowered.startswith("l_") or lowered.startswith("left") or "_l_" in lowered:
        return "left"
    return "unknown"


def infer_dof(name: str) -> str:
    """Infer a leg degree of freedom from an object name."""
    lowered = name.lower()
    for token in LEG_DOF_TOKENS:
        if token in lowered:
            return token
    if "knee" in lowered:
        return "knee_pitch"
    if "ankle" in lowered and "roll" not in lowered:
        return "ankle_pitch"
    return "unknown"


def infer_foot_region(name: str) -> str:
    """Infer contact region from a foot-related geom name."""
    lowered = name.lower()
    if "toe" in lowered:
        return "toe"
    if "heel" in lowered:
        return "heel"
    if "center" in lowered:
        return "center"
    if lowered in {"r_foot_collision", "l_foot_collision"}:
        return "center"
    if "bottom" in lowered:
        return "foot_bottom"
    if "sole" in lowered:
        return "sole"
    return "unknown"


def model_inventory(model: mujoco.MjModel) -> dict[str, Any]:
    """Return high-level model object counts and filtered names."""
    bodies = [body_name(model, index) for index in range(model.nbody)]
    joints = [joint_name(model, index) for index in range(model.njnt)]
    actuators = [actuator_name(model, index) for index in range(model.nu)]
    geoms = [geom_name(model, index) for index in range(model.ngeom)]
    sensors = [sensor_name(model, index) for index in range(model.nsensor)]
    sites = [site_name(model, index) for index in range(model.nsite)]
    keywords = ("hip", "knee", "ankle", *FOOT_TOKENS)
    return {
        "counts": {
            "bodies": model.nbody,
            "joints": model.njnt,
            "actuators": model.nu,
            "geoms": model.ngeom,
            "sensors": model.nsensor,
            "sites": model.nsite,
        },
        "filtered": {
            "bodies": [name for name in bodies if any(token in name.lower() for token in keywords)],
            "joints": [name for name in joints if any(token in name.lower() for token in keywords)],
            "actuators": [name for name in actuators if any(token in name.lower() for token in keywords)],
            "geoms": [name for name in geoms if any(token in name.lower() for token in keywords)],
            "sensors": [name for name in sensors if any(token in name.lower() for token in keywords)],
            "sites": [name for name in sites if any(token in name.lower() for token in keywords)],
        },
    }


def joint_topology_rows(model: mujoco.MjModel, robot: str) -> list[dict[str, Any]]:
    """Return joint topology rows for a model."""
    rows: list[dict[str, Any]] = []
    for joint_id in range(model.njnt):
        name = joint_name(model, joint_id)
        joint_type = mujoco.mjtJoint(int(model.jnt_type[joint_id])).name.replace("mjJNT_", "").lower()
        rows.append(
            {
                "robot": robot,
                "joint_id": joint_id,
                "joint_name": name,
                "joint_type": joint_type,
                "body_name": body_name(model, int(model.jnt_bodyid[joint_id])),
                "qposadr": int(model.jnt_qposadr[joint_id]),
                "dofadr": int(model.jnt_dofadr[joint_id]),
                "range": csv_vec(model.jnt_range[joint_id]),
                "limited": bool(model.jnt_limited[joint_id]),
                "inferred_side": infer_side(name),
                "inferred_dof": infer_dof(name),
                "is_leg_joint": infer_dof(name) != "unknown",
            }
        )
    return rows


def actuator_inventory_rows(model: mujoco.MjModel, robot: str) -> list[dict[str, Any]]:
    """Return actuator metadata rows for a model."""
    rows: list[dict[str, Any]] = []
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        joint = joint_name(model, joint_id) if 0 <= joint_id < model.njnt else ""
        rows.append(
            {
                "robot": robot,
                "actuator_id": actuator_id,
                "actuator_name": actuator_name(model, actuator_id),
                "joint_name": joint,
                "ctrlrange": csv_vec(model.actuator_ctrlrange[actuator_id]),
                "ctrllimited": bool(model.actuator_ctrllimited[actuator_id]),
                "gear": csv_vec(model.actuator_gear[actuator_id]),
                "forcerange": csv_vec(model.actuator_forcerange[actuator_id]),
                "forcelimited": bool(model.actuator_forcelimited[actuator_id]),
                "joint_range": csv_vec(model.jnt_range[joint_id]) if 0 <= joint_id < model.njnt else "",
                "inferred_side": infer_side(joint or actuator_name(model, actuator_id)),
                "inferred_dof": infer_dof(joint or actuator_name(model, actuator_id)),
            }
        )
    return rows


def foot_geom_rows(model: mujoco.MjModel, robot: str) -> list[dict[str, Any]]:
    """Return foot-related geom inventory rows."""
    rows: list[dict[str, Any]] = []
    for geom_id in range(model.ngeom):
        name = geom_name(model, geom_id)
        lowered = name.lower()
        if not any(token in lowered for token in FOOT_TOKENS):
            continue
        rows.append(
            {
                "robot": robot,
                "geom_id": geom_id,
                "geom_name": name,
                "body_name": body_name(model, int(model.geom_bodyid[geom_id])),
                "type": geom_type_name(model, geom_id),
                "pos": csv_vec(model.geom_pos[geom_id]),
                "size": csv_vec(model.geom_size[geom_id]),
                "friction": csv_vec(model.geom_friction[geom_id]),
                "contype": int(model.geom_contype[geom_id]),
                "conaffinity": int(model.geom_conaffinity[geom_id]),
                "inferred_side": infer_side(name),
                "inferred_region": infer_foot_region(name),
            }
        )
    return rows


def compare_by_side_dof(sedon_rows: list[dict[str, Any]], duck_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare row presence by side and inferred DOF."""
    keys = sorted(
        {
            (str(row["inferred_side"]), str(row["inferred_dof"]))
            for row in sedon_rows + duck_rows
            if row.get("inferred_dof") != "unknown"
        }
    )
    rows: list[dict[str, Any]] = []
    for side, dof in keys:
        sedon = [row for row in sedon_rows if row["inferred_side"] == side and row["inferred_dof"] == dof]
        duck = [row for row in duck_rows if row["inferred_side"] == side and row["inferred_dof"] == dof]
        rows.append(
            {
                "side": side,
                "dof": dof,
                "sedon_count": len(sedon),
                "duck_count": len(duck),
                "sedon_names": ";".join(str(row.get("joint_name") or row.get("actuator_name")) for row in sedon),
                "duck_names": ";".join(str(row.get("joint_name") or row.get("actuator_name")) for row in duck),
                "compatible_presence": bool(sedon and duck),
            }
        )
    return rows


def named_body_position(model: mujoco.MjModel, data: mujoco.MjData, candidates: list[str]) -> np.ndarray | None:
    """Return the first body position matching candidate substrings."""
    for body_id in range(model.nbody):
        name = body_name(model, body_id).lower()
        if all(token in name for token in candidates):
            return data.xpos[body_id].copy()
    return None


def first_body_position(model: mujoco.MjModel, data: mujoco.MjData, candidate_sets: list[list[str]]) -> np.ndarray | None:
    """Return the first body position matching one of several token sets."""
    for candidates in candidate_sets:
        position = named_body_position(model, data, candidates)
        if position is not None:
            return position
    return None


def named_geom_positions(model: mujoco.MjModel, data: mujoco.MjData, side: str) -> list[np.ndarray]:
    """Return world positions of foot-related geoms for one side."""
    rows: list[np.ndarray] = []
    side_prefix = "r_" if side == "right" else "l_"
    side_word = "right" if side == "right" else "left"
    for geom_id in range(model.ngeom):
        name = geom_name(model, geom_id).lower()
        if (name.startswith(side_prefix) or name.startswith(side_word)) and any(token in name for token in FOOT_TOKENS):
            rows.append(data.geom_xpos[geom_id].copy())
    return rows


def foot_extent(model: mujoco.MjModel, data: mujoco.MjData, side: str) -> tuple[float | None, float | None]:
    """Return approximate foot length and width from foot-related geoms."""
    xs: list[float] = []
    ys: list[float] = []
    side_prefix = "r_" if side == "right" else "l_"
    side_word = "right" if side == "right" else "left"
    for geom_id in range(model.ngeom):
        name = geom_name(model, geom_id).lower()
        if not ((name.startswith(side_prefix) or name.startswith(side_word)) and any(token in name for token in FOOT_TOKENS)):
            continue
        center = data.geom_xpos[geom_id]
        size = model.geom_size[geom_id]
        xs.extend([float(center[0] - size[0]), float(center[0] + size[0])])
        y_radius = float(size[1] if size.shape[0] > 1 else size[0])
        ys.extend([float(center[1] - y_radius), float(center[1] + y_radius)])
    return (max(xs) - min(xs) if xs else None, max(ys) - min(ys) if ys else None)


def distance(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    """Return Euclidean distance when both vectors exist."""
    if a is None or b is None:
        return None
    return float(np.linalg.norm(a - b))


def joint_anchor(model: mujoco.MjModel, data: mujoco.MjData, side: str, dof: str) -> np.ndarray | None:
    """Return neutral world joint anchor for a side/DOF when available."""
    for joint_id in range(model.njnt):
        name = joint_name(model, joint_id)
        if infer_side(name) == side and infer_dof(name) == dof:
            return data.xanchor[joint_id].copy()
    return None


def mean_metric(values: list[float | None]) -> float | None:
    """Return mean of present numeric values."""
    present = [value for value in values if value is not None]
    return float(np.mean(present)) if present else None


def morphology_metrics(model: mujoco.MjModel, data: mujoco.MjData, root_hint: str) -> dict[str, Any]:
    """Compute approximate neutral-pose morphology metrics."""
    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_hint)
    if root_id < 0:
        root_id = 1 if model.nbody > 1 else 0
    base_height = float(data.xpos[root_id][2])

    right_hip = first_body_position(model, data, [["r_", "hip"], ["right", "hip"]])
    left_hip = first_body_position(model, data, [["l_", "hip"], ["left", "hip"]])
    right_knee = first_body_position(model, data, [["r_", "knee"], ["right", "knee"]])
    left_knee = first_body_position(model, data, [["l_", "knee"], ["left", "knee"]])
    right_ankle = first_body_position(model, data, [["r_", "ankle"], ["right", "ankle"]])
    left_ankle = first_body_position(model, data, [["l_", "ankle"], ["left", "ankle"]])
    right_foot_positions = named_geom_positions(model, data, "right")
    left_foot_positions = named_geom_positions(model, data, "left")
    right_foot = np.mean(right_foot_positions, axis=0) if right_foot_positions else None
    left_foot = np.mean(left_foot_positions, axis=0) if left_foot_positions else None
    right_hip_anchor = joint_anchor(model, data, "right", "hip_pitch")
    left_hip_anchor = joint_anchor(model, data, "left", "hip_pitch")
    right_knee_anchor = joint_anchor(model, data, "right", "knee_pitch")
    left_knee_anchor = joint_anchor(model, data, "left", "knee_pitch")
    right_ankle_anchor = joint_anchor(model, data, "right", "ankle_pitch")
    left_ankle_anchor = joint_anchor(model, data, "left", "ankle_pitch")
    right_hip = right_hip if right_hip is not None else right_hip_anchor
    left_hip = left_hip if left_hip is not None else left_hip_anchor
    right_knee = right_knee if right_knee is not None else right_knee_anchor
    left_knee = left_knee if left_knee is not None else left_knee_anchor
    right_ankle = right_ankle if right_ankle is not None else right_ankle_anchor
    left_ankle = left_ankle if left_ankle is not None else left_ankle_anchor
    right_length, right_width = foot_extent(model, data, "right")
    left_length, left_width = foot_extent(model, data, "left")
    hip_width = abs(float(right_hip[1] - left_hip[1])) if right_hip is not None and left_hip is not None else None
    stance_width = (
        abs(float(right_foot[1] - left_foot[1]))
        if right_foot is not None and left_foot is not None
        else None
    )
    right_leg = [distance(right_hip, right_knee), distance(right_knee, right_ankle), distance(right_ankle, right_foot)]
    left_leg = [distance(left_hip, left_knee), distance(left_knee, left_ankle), distance(left_ankle, left_foot)]
    return {
        "root_body": body_name(model, int(root_id)),
        "base_height": base_height,
        "hip_width": hip_width,
        "right_hip_to_knee_distance": right_leg[0],
        "left_hip_to_knee_distance": left_leg[0],
        "right_knee_to_ankle_distance": right_leg[1],
        "left_knee_to_ankle_distance": left_leg[1],
        "right_ankle_to_foot_distance": right_leg[2],
        "left_ankle_to_foot_distance": left_leg[2],
        "right_approx_leg_length": sum(value for value in right_leg if value is not None),
        "left_approx_leg_length": sum(value for value in left_leg if value is not None),
        "approx_leg_length": mean_metric(
            [
                sum(value for value in right_leg if value is not None),
                sum(value for value in left_leg if value is not None),
            ]
        ),
        "right_foot_length": right_length,
        "left_foot_length": left_length,
        "foot_length": mean_metric([right_length, left_length]),
        "right_foot_width": right_width,
        "left_foot_width": left_width,
        "foot_width": mean_metric([right_width, left_width]),
        "stance_width": stance_width,
        "right_foot_center": right_foot.tolist() if right_foot is not None else None,
        "left_foot_center": left_foot.tolist() if left_foot is not None else None,
    }


def morphology_comparison(sedon: dict[str, Any], duck: dict[str, Any]) -> dict[str, Any]:
    """Return Sedon/Duck morphology ratio metrics."""
    out: dict[str, Any] = {}
    for key in ("base_height", "hip_width", "approx_leg_length", "foot_length", "foot_width", "stance_width"):
        sedon_value = sedon.get(key)
        duck_value = duck.get(key)
        ratio = None
        if isinstance(sedon_value, (int, float)) and isinstance(duck_value, (int, float)) and abs(float(duck_value)) > 1e-9:
            ratio = float(sedon_value) / float(duck_value)
        out[key] = {"sedon": sedon_value, "duck": duck_value, "sedon_to_duck_ratio": ratio}
    return out


def flatten_morphology_comparison(comparison: dict[str, Any]) -> list[dict[str, Any]]:
    """Return CSV rows for morphology comparison."""
    return [
        {
            "metric": metric,
            "sedon": values.get("sedon"),
            "duck": values.get("duck"),
            "sedon_to_duck_ratio": values.get("sedon_to_duck_ratio"),
        }
        for metric, values in comparison.items()
    ]


def qpos_for_joint(model: mujoco.MjModel, joint_id: int) -> int | None:
    """Return qpos address for scalar hinge/slide joints."""
    if int(model.jnt_type[joint_id]) not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
        return None
    return int(model.jnt_qposadr[joint_id])


def foot_center(model: mujoco.MjModel, data: mujoco.MjData, side: str) -> np.ndarray | None:
    """Return average foot-related geom center for one side."""
    positions = named_geom_positions(model, data, side)
    return np.mean(positions, axis=0) if positions else None


def sedon_joint_effect_probe(
    model: mujoco.MjModel,
    steps: int,
    epsilon: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Probe Sedon joint sign effects by perturbing joint qpos in neutral pose.

    Sedon actuators in the diagnostic scene are torque motors. A tiny direct
    torque command is not a reliable joint-sign audit, so this probe uses a
    kinematic qpos perturbation and records the resulting foot displacement.
    """
    rows: list[dict[str, Any]] = []
    mapping: dict[str, Any] = {}
    base_data = mujoco.MjData(model)
    mujoco.mj_resetData(model, base_data)
    mujoco.mj_forward(model, base_data)
    base_right = foot_center(model, base_data, "right")
    base_left = foot_center(model, base_data, "left")

    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        if joint_id < 0 or joint_id >= model.njnt:
            continue
        joint = joint_name(model, joint_id)
        dof = infer_dof(joint)
        side = infer_side(joint)
        if dof == "unknown" or side not in {"right", "left"}:
            continue
        qadr = qpos_for_joint(model, joint_id)
        if qadr is None:
            continue
        for sign_label, sign in (("positive", 1.0), ("negative", -1.0)):
            data = mujoco.MjData(model)
            mujoco.mj_resetData(model, data)
            data.qpos[qadr] += sign * epsilon
            mujoco.mj_forward(model, data)
            for _ in range(max(0, steps)):
                mujoco.mj_forward(model, data)
            right = foot_center(model, data, "right")
            left = foot_center(model, data, "left")
            base = base_right if side == "right" else base_left
            foot = right if side == "right" else left
            delta = foot - base if foot is not None and base is not None else np.array([math.nan, math.nan, math.nan])
            row = {
                "actuator_id": actuator_id,
                "actuator_name": actuator_name(model, actuator_id),
                "joint_name": joint,
                "inferred_side": side,
                "inferred_dof": dof,
                "probe_type": "qpos_perturbation",
                "sign": sign_label,
                "epsilon": epsilon,
                "steps": steps,
                "foot_dx": float(delta[0]),
                "foot_dy": float(delta[1]),
                "foot_dz": float(delta[2]),
            }
            rows.append(row)
        positive = next(
            row for row in reversed(rows) if row["joint_name"] == joint and row["sign"] == "positive"
        )
        mapping[joint] = {
            "side": side,
            "dof": dof,
            "positive_dx": positive["foot_dx"],
            "positive_dy": positive["foot_dy"],
            "positive_dz": positive["foot_dz"],
            "interpretation": interpret_joint_effect(dof, positive),
        }
    return rows, mapping


def interpret_joint_effect(dof: str, row: dict[str, Any]) -> str:
    """Return a short interpretation for a positive perturbation."""
    dx = float(row["foot_dx"])
    dy = float(row["foot_dy"])
    dz = float(row["foot_dz"])
    if dof == "hip_pitch":
        return "positive moves foot forward" if dx > 0.0 else "positive moves foot backward"
    if dof == "knee_pitch":
        return "positive raises foot" if dz > 0.0 else "positive lowers foot"
    if dof == "ankle_pitch":
        return "positive changes toe/foot pitch proxy; inspect dx/dz"
    if dof == "hip_roll":
        return "positive shifts foot laterally outward/inward; inspect dy"
    return f"positive delta dx={dx:.6f}, dy={dy:.6f}, dz={dz:.6f}"


def duck_reference_values(path: Path) -> dict[str, float]:
    """Load high-level Open Duck reference values with conservative fallbacks."""
    values = dict(DUCK_FALLBACK)
    if not path.is_file():
        return values
    payload = json.loads(path.read_text(encoding="utf-8"))
    metric_sets = payload.get("open_duck_metrics", {})
    if isinstance(metric_sets, dict) and metric_sets:
        metric_values = list(metric_sets.values())
        values["target_vx"] = float(np.mean([item.get("average_forward_velocity", values["target_vx"]) for item in metric_values]))
        values["gait_period"] = float(np.mean([item.get("estimated_gait_period_seconds", values["gait_period"]) for item in metric_values]))
        values["foot_clearance"] = float(
            np.mean(
                [
                    0.5
                    * (
                        item.get("approximate_left_clearance", values["foot_clearance"])
                        + item.get("approximate_right_clearance", values["foot_clearance"])
                    )
                    for item in metric_values
                ]
            )
        )
        values["max_abs_roll"] = float(max(item.get("max_abs_roll", values["max_abs_roll"]) for item in metric_values))
        values["max_abs_pitch"] = float(max(item.get("max_abs_pitch", values["max_abs_pitch"]) for item in metric_values))
        values["double_support_ratio"] = float(np.mean([item.get("double_support_ratio", values["double_support_ratio"]) for item in metric_values]))
        values["flight_ratio"] = float(np.mean([item.get("flight_ratio", values["flight_ratio"]) for item in metric_values]))
    design = payload.get("open_duck_design", {})
    if isinstance(design, dict):
        values["action_scale"] = float(design.get("action_scale", values["action_scale"]))
    return values


def scaled_gait_reference(duck_values: dict[str, float], morph: dict[str, Any]) -> dict[str, Any]:
    """Create a morphology-scaled Sedon gait-level reference."""
    height_ratio = morph.get("base_height", {}).get("sedon_to_duck_ratio") or 1.0
    raw_leg_ratio = morph.get("approx_leg_length", {}).get("sedon_to_duck_ratio")
    raw_foot_ratio = morph.get("foot_length", {}).get("sedon_to_duck_ratio")
    leg_ratio = float(raw_leg_ratio) if isinstance(raw_leg_ratio, (int, float)) and raw_leg_ratio > 0.0 else float(height_ratio)
    leg_ratio = min(max(leg_ratio, 0.5), 3.0)
    if isinstance(raw_foot_ratio, (int, float)) and 0.5 <= raw_foot_ratio <= 4.0:
        foot_ratio = float(raw_foot_ratio)
        foot_ratio_source = "measured_foot_length_ratio"
    else:
        foot_ratio = leg_ratio
        foot_ratio_source = "leg_or_height_ratio_fallback_due_to_unreliable_mesh_foot_extent"
    period = duck_values["gait_period"] * min(max(math.sqrt(float(leg_ratio)), 1.05), 1.20)
    return {
        "source": "Open Duck high-level gait metrics scaled by approximate Sedon/Duck morphology ratios",
        "duck_reference": duck_values,
        "morphology_ratios": {
            "leg_length_ratio": leg_ratio,
            "foot_length_ratio": foot_ratio,
            "raw_foot_length_ratio": raw_foot_ratio,
            "foot_ratio_source": foot_ratio_source,
            "base_height_ratio": height_ratio,
        },
        "sedon_target_vx": duck_values["target_vx"] * math.sqrt(float(leg_ratio)),
        "sedon_gait_period": period,
        "sedon_step_frequency": 1.0 / period if period > 1e-9 else None,
        "sedon_foot_clearance_target": duck_values["foot_clearance"] * float(foot_ratio),
        "sedon_double_support_target_range": [
            max(0.0, duck_values["double_support_ratio"] - 0.08),
            min(1.0, duck_values["double_support_ratio"] + 0.08),
        ],
        "sedon_flight_ratio_target": duck_values["flight_ratio"],
        "sedon_max_abs_roll": duck_values["max_abs_roll"],
        "sedon_max_abs_pitch": duck_values["max_abs_pitch"],
        "sedon_action_scale_initial": duck_values["action_scale"],
        "notes": [
            "Do not transfer raw Duck joint angles/actions/ONNX weights.",
            "Velocity is scaled by sqrt(leg_length_ratio) as a conservative gait-level heuristic.",
            "Foot clearance is scaled by foot_length_ratio when available.",
            "Gait period is slowed by sqrt(leg_length_ratio), clamped to 1.05-1.20x Duck period.",
            "Action scale starts from Duck 0.25 only as an initial normalized residual amplitude, not as a direct control mapping.",
        ],
    }


def contact_style_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize foot contact style from inferred geom regions."""
    regions = {str(row["inferred_region"]) for row in rows}
    has_split = bool({"toe", "heel"} & regions) and ("center" in regions or "foot_bottom" in regions or "sole" in regions)
    has_single_bottom = "foot_bottom" in regions or ("sole" in regions and not {"toe", "heel"} & regions)
    return {
        "regions": sorted(regions),
        "toe_center_heel_split": has_split,
        "single_foot_bottom_style": has_single_bottom and not has_split,
        "foot_geom_count": len(rows),
    }


def write_contact_style_report(path: Path, sedon_rows: list[dict[str, Any]], duck_rows: list[dict[str, Any]]) -> None:
    """Write a short foot contact style comparison report."""
    sedon = contact_style_summary(sedon_rows)
    duck = contact_style_summary(duck_rows)
    lines = [
        "# Foot Contact Style Comparison",
        "",
        f"- Sedon regions: `{', '.join(sedon['regions'])}`",
        f"- Duck regions: `{', '.join(duck['regions'])}`",
        f"- Sedon toe/center/heel split: `{sedon['toe_center_heel_split']}`",
        f"- Duck single foot bottom style: `{duck['single_foot_bottom_style']}`",
        "",
        "Sedon uses a more complex contact patch layout than Open Duck Mini v2. "
        "Open Duck's reference gait succeeds with a single bottom-style foot contact, "
        "so Sedon should keep a simplified foot_bottom_collision comparison variant "
        "available before spending more PPO time on split toe/center/heel geometry.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_final_report(
    path: Path,
    *,
    sedon_inventory: dict[str, Any],
    duck_inventory: dict[str, Any],
    joint_comparison: list[dict[str, Any]],
    morphology: dict[str, Any],
    contact_summary: dict[str, Any],
    scaled_reference: dict[str, Any],
    sign_mapping: dict[str, Any],
) -> None:
    """Write the Phase M0 Markdown report."""
    missing_duck_dofs = [row for row in joint_comparison if row["sedon_count"] > 0 and row["duck_count"] == 0]
    lines = [
        "# Phase M0 Sedon vs Open Duck Morphology Audit",
        "",
        "## A. Executive Summary",
        "",
        "Open Duck gait-level metrics are useful as reference targets, but raw Duck joint angles, actions, ONNX weights, and controls are not directly transferable to Sedon.",
        "",
        "## B. Can Duck Parameters Be Transferred Directly?",
        "",
        "No. Use Duck gait-level metrics only: velocity, gait period, clearance, support ratios, roll/pitch envelope, and action-scale starting point. Direct transfer is blocked by morphology, actuator, joint naming/sign, and contact-geometry differences.",
        "",
        "## C. Sedon vs Duck Joint Topology Comparison",
        "",
        f"- Sedon counts: `{sedon_inventory['counts']}`",
        f"- Duck counts: `{duck_inventory['counts']}`",
        f"- Missing Duck DOF matches for Sedon rows: `{len(missing_duck_dofs)}`",
        "",
        "## D. Sedon vs Duck Morphology Comparison",
        "",
    ]
    for metric, values in morphology.items():
        lines.append(
            f"- {metric}: Sedon=`{values.get('sedon')}`, Duck=`{values.get('duck')}`, ratio=`{values.get('sedon_to_duck_ratio')}`"
        )
    lines.extend(
        [
            "",
            "## E. Sedon vs Duck Foot Contact Style Comparison",
            "",
            f"- Sedon contact summary: `{contact_summary['sedon']}`",
            f"- Duck contact summary: `{contact_summary['duck']}`",
            "",
            "## F. Sedon Joint Sign/Effect Mapping",
            "",
            f"- Probed joints: `{len(sign_mapping)}`",
            "- See `sedon_joint_effect_probe.csv` and `sedon_joint_sign_mapping.json` for per-joint dx/dy/dz.",
            "",
            "## G. Sedon-Scaled DuckRef Gait Parameters",
            "",
            f"- sedon_target_vx: `{scaled_reference['sedon_target_vx']}`",
            f"- sedon_gait_period: `{scaled_reference['sedon_gait_period']}`",
            f"- sedon_foot_clearance_target: `{scaled_reference['sedon_foot_clearance_target']}`",
            f"- sedon_double_support_target_range: `{scaled_reference['sedon_double_support_target_range']}`",
            f"- sedon_flight_ratio_target: `{scaled_reference['sedon_flight_ratio_target']}`",
            "",
            "## H. Risks",
            "",
            "- Approximate morphology is based on neutral MuJoCo body/geom/site positions, not CAD-grade link measurements.",
            "- Sedon torque motors mean qpos perturbation is the clearest sign probe; it is not an actuator authority test.",
            "- Duck's single-bottom contact success does not prove Sedon must remove split geoms, but it justifies a simplified comparison variant.",
            "",
            "## I. Recommendation",
            "",
            "- Proceed to Phase W0-DuckRef only after contact geometry diagnostics remain consistent.",
            "- Build or keep a simplified foot_bottom_collision comparison variant.",
            "- Use Duck gait-level metrics as targets, not raw Duck controls.",
            "- Do not transfer Duck ONNX weights or raw joint trajectories.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_progress_log(path: Path, output_dir: Path, scaled_reference: dict[str, Any]) -> None:
    """Append or update the Phase M0 progress-log section for 2026-06-01."""
    title = "## 2026-06-01 - Phase M0 Sedon vs Open Duck Morphology Audit"
    section = f"""{title}

### Goal

Audit whether Open Duck Mini v2 gait-level references can guide Sedon without directly transferring Duck joint angles, raw actions, ONNX weights, or controls.

### Files

* `tools/sedon_phase_m0_duck_morphology_audit.py`
* `{output_dir / 'sedon_joint_topology.csv'}`
* `{output_dir / 'duck_joint_topology.csv'}`
* `{output_dir / 'joint_topology_comparison.csv'}`
* `{output_dir / 'sedon_actuator_inventory.csv'}`
* `{output_dir / 'duck_actuator_inventory.csv'}`
* `{output_dir / 'actuator_comparison.csv'}`
* `{output_dir / 'sedon_morphology_metrics.json'}`
* `{output_dir / 'duck_morphology_metrics.json'}`
* `{output_dir / 'morphology_comparison.json'}`
* `{output_dir / 'morphology_comparison.csv'}`
* `{output_dir / 'sedon_foot_geom_inventory.csv'}`
* `{output_dir / 'duck_foot_geom_inventory.csv'}`
* `{output_dir / 'foot_contact_style_comparison.md'}`
* `{output_dir / 'sedon_joint_effect_probe.csv'}`
* `{output_dir / 'sedon_joint_sign_mapping.json'}`
* `{output_dir / 'sedon_duck_scaled_gait_reference.json'}`
* `{output_dir / 'phase_m0_duck_morphology_audit_report.md'}`
* `docs/sedon_blue_like_dynamic_gait_progress_log.md`

### Commands

```text
python -m py_compile tools/sedon_phase_m0_duck_morphology_audit.py
.venv\\Scripts\\python.exe -m tools.sedon_phase_m0_duck_morphology_audit
```

### Key Metrics

| Metric | Value |
| --- | ---: |
| sedon_target_vx | {scaled_reference['sedon_target_vx']:.6f} |
| sedon_gait_period | {scaled_reference['sedon_gait_period']:.6f} |
| sedon_foot_clearance_target | {scaled_reference['sedon_foot_clearance_target']:.6f} |
| sedon_flight_ratio_target | {scaled_reference['sedon_flight_ratio_target']:.6f} |
| sedon_action_scale_initial | {scaled_reference['sedon_action_scale_initial']:.6f} |

### Result

INCONCLUSIVE

### Engineering Interpretation

Open Duck Mini v2 provides a useful gait-level reference, especially forward velocity, gait period, clearance, no-flight support timing, and roll/pitch envelope. It should not be used as a raw joint/action transfer source because Sedon differs in morphology, actuator semantics, joint sign mapping, and contact geometry. Sedon's split contact geometry remains a risk relative to Duck's single foot-bottom style.

### Next Decision

1. Keep Phase W0-DuckRef at gait-metric level only.
2. Keep or generate a simplified foot_bottom_collision comparison variant before PPO tuning on split patches.
3. Use Sedon joint sign mapping for controller/reference construction.
4. Do not train until Phase G1/G2 contact semantics are consistent.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8") if path.is_file() else ""
    if title in existing:
        start = existing.index(title)
        next_start = existing.find("\n## ", start + 1)
        if next_start == -1:
            updated = existing[:start].rstrip() + "\n\n" + section.rstrip() + "\n"
        else:
            updated = existing[:start].rstrip() + "\n\n" + section.rstrip() + "\n\n" + existing[next_start + 1 :].lstrip()
    else:
        updated = existing.rstrip() + "\n\n" + section.rstrip() + "\n" if existing.strip() else section
    path.write_text(updated, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--sedon-scene-path", type=Path, default=DEFAULT_SEDON_SCENE)
    parser.add_argument("--duck-xml-path", type=Path, default=DEFAULT_DUCK_XML)
    parser.add_argument("--duck-reference-json", type=Path, default=DEFAULT_DUCK_REFERENCE_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--steps-for-joint-probe", type=int, default=80)
    parser.add_argument("--epsilon", type=float, default=0.03)
    return parser.parse_args()


def main() -> None:
    """Run the Phase M0 audit and write all artifacts."""
    args = parse_args()
    if args.steps_for_joint_probe < 0:
        raise ValueError("--steps-for-joint-probe must be non-negative")
    if args.epsilon <= 0.0:
        raise ValueError("--epsilon must be positive")
    config = AuditConfig(
        sedon_scene_path=require_scene(args.sedon_scene_path),
        duck_xml_path=require_scene(args.duck_xml_path),
        duck_reference_json=args.duck_reference_json,
        output_dir=args.output_dir,
        steps_for_joint_probe=args.steps_for_joint_probe,
        epsilon=args.epsilon,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)

    sedon_model, sedon_data = load_model(config.sedon_scene_path)
    duck_model, duck_data = load_model(config.duck_xml_path)

    sedon_inventory = model_inventory(sedon_model)
    duck_inventory = model_inventory(duck_model)
    sedon_joints = joint_topology_rows(sedon_model, "sedon")
    duck_joints = joint_topology_rows(duck_model, "open_duck")
    joint_comparison = compare_by_side_dof(sedon_joints, duck_joints)
    sedon_actuators = actuator_inventory_rows(sedon_model, "sedon")
    duck_actuators = actuator_inventory_rows(duck_model, "open_duck")
    actuator_comparison = compare_by_side_dof(sedon_actuators, duck_actuators)
    sedon_morph = morphology_metrics(sedon_model, sedon_data, "base_link")
    duck_design = {}
    if config.duck_reference_json.is_file():
        duck_design = json.loads(config.duck_reference_json.read_text(encoding="utf-8")).get("open_duck_design", {})
    duck_root = str(duck_design.get("root_body_name", "trunk_assembly")) if isinstance(duck_design, dict) else "trunk_assembly"
    duck_morph = morphology_metrics(duck_model, duck_data, duck_root)
    morph_comparison = morphology_comparison(sedon_morph, duck_morph)
    sedon_foot = foot_geom_rows(sedon_model, "sedon")
    duck_foot = foot_geom_rows(duck_model, "open_duck")
    probe_rows, sign_mapping = sedon_joint_effect_probe(sedon_model, config.steps_for_joint_probe, config.epsilon)
    duck_values = duck_reference_values(config.duck_reference_json)
    scaled_reference = scaled_gait_reference(duck_values, morph_comparison)
    contact_summary = {
        "sedon": contact_style_summary(sedon_foot),
        "duck": contact_style_summary(duck_foot),
    }

    write_json(config.output_dir / "model_inventory.json", {"sedon": sedon_inventory, "open_duck": duck_inventory})
    write_csv(config.output_dir / "sedon_joint_topology.csv", sedon_joints, list(sedon_joints[0].keys()))
    write_csv(config.output_dir / "duck_joint_topology.csv", duck_joints, list(duck_joints[0].keys()))
    write_csv(config.output_dir / "joint_topology_comparison.csv", joint_comparison, list(joint_comparison[0].keys()))
    write_csv(config.output_dir / "sedon_actuator_inventory.csv", sedon_actuators, list(sedon_actuators[0].keys()))
    write_csv(config.output_dir / "duck_actuator_inventory.csv", duck_actuators, list(duck_actuators[0].keys()))
    write_csv(config.output_dir / "actuator_comparison.csv", actuator_comparison, list(actuator_comparison[0].keys()))
    write_json(config.output_dir / "sedon_morphology_metrics.json", sedon_morph)
    write_json(config.output_dir / "duck_morphology_metrics.json", duck_morph)
    write_json(config.output_dir / "morphology_comparison.json", morph_comparison)
    write_csv(
        config.output_dir / "morphology_comparison.csv",
        flatten_morphology_comparison(morph_comparison),
        ["metric", "sedon", "duck", "sedon_to_duck_ratio"],
    )
    write_csv(config.output_dir / "sedon_foot_geom_inventory.csv", sedon_foot, list(sedon_foot[0].keys()))
    write_csv(config.output_dir / "duck_foot_geom_inventory.csv", duck_foot, list(duck_foot[0].keys()))
    write_contact_style_report(config.output_dir / "foot_contact_style_comparison.md", sedon_foot, duck_foot)
    write_csv(config.output_dir / "sedon_joint_effect_probe.csv", probe_rows, list(probe_rows[0].keys()))
    write_json(config.output_dir / "sedon_joint_sign_mapping.json", sign_mapping)
    write_json(config.output_dir / "sedon_duck_scaled_gait_reference.json", scaled_reference)
    write_final_report(
        config.output_dir / "phase_m0_duck_morphology_audit_report.md",
        sedon_inventory=sedon_inventory,
        duck_inventory=duck_inventory,
        joint_comparison=joint_comparison,
        morphology=morph_comparison,
        contact_summary=contact_summary,
        scaled_reference=scaled_reference,
        sign_mapping=sign_mapping,
    )
    update_progress_log(
        REPO_ROOT / "docs" / "sedon_blue_like_dynamic_gait_progress_log.md",
        config.output_dir,
        scaled_reference,
    )

    print(f"output_dir={config.output_dir}")
    print(f"sedon_target_vx={scaled_reference['sedon_target_vx']:.6f}")
    print(f"sedon_gait_period={scaled_reference['sedon_gait_period']:.6f}")
    print(f"sedon_foot_clearance_target={scaled_reference['sedon_foot_clearance_target']:.6f}")
    print(f"report={config.output_dir / 'phase_m0_duck_morphology_audit_report.md'}")


if __name__ == "__main__":
    main()
