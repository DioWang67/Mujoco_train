"""Phase Pre-W0 free-space joint sign validation for Seedon.

This Class C diagnostic validates Seedon leg joint sign effects without ground
contact. It lifts the base, perturbs each leg joint by +/- epsilon, and records
world-space foot/toe deltas to compare against the Phase M0 joint sign mapping.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import mujoco
import numpy as np

from tools.seedon_debug_common import DEBUG_OUT_DIR, geom_name, require_scene


DEFAULT_SCENE_PATH = DEBUG_OUT_DIR / "blue_like_sole_experiments_v5" / "training_scene_v5_a.xml"
DEFAULT_M0_SIGN_MAPPING = DEBUG_OUT_DIR / "phase_m0_duck_morphology_audit" / "seedon_joint_sign_mapping.json"
DEFAULT_OUTPUT_DIR = DEBUG_OUT_DIR / "phase_pre_w0_free_space_joint_sign_validation"
FOOT_TOKENS = ("foot", "sole", "bottom", "collision", "ankle", "end_effector")
TOE_TOKENS = ("toe",)
LEG_DOF_TOKENS = ("hip_yaw", "hip_roll", "hip_pitch", "knee_pitch", "ankle_pitch", "ankle_roll")


@dataclass(frozen=True)
class ValidationConfig:
    """Runtime configuration for free-space sign validation.

    Parameters:
        scene_path: MuJoCo XML scene to inspect.
        m0_sign_mapping: Phase M0 sign mapping JSON.
        output_dir: Directory where CSV, JSON, and Markdown outputs are written.
        epsilon: Positive and negative joint perturbation magnitude.
        steps: Kept in output for reproducibility; the probe is kinematic.
        base_lift: Added z offset for the root free joint before probing.
    """

    scene_path: Path
    m0_sign_mapping: Path
    output_dir: Path
    epsilon: float
    steps: int
    base_lift: float


def mj_name(model: mujoco.MjModel, obj_type: mujoco.mjtObj, index: int, fallback: str) -> str:
    """Return a stable MuJoCo object name."""

    name = mujoco.mj_id2name(model, obj_type, int(index))
    return name or f"{fallback}_{index}"


def joint_name(model: mujoco.MjModel, joint_id: int) -> str:
    """Return joint name for an id."""

    return mj_name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id, "joint")


def actuator_name(model: mujoco.MjModel, actuator_id: int) -> str:
    """Return actuator name for an id."""

    return mj_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id, "actuator")


def site_name(model: mujoco.MjModel, site_id: int) -> str:
    """Return site name for an id."""

    return mj_name(model, mujoco.mjtObj.mjOBJ_SITE, site_id, "site")


def body_name(model: mujoco.MjModel, body_id: int) -> str:
    """Return body name for an id."""

    return mj_name(model, mujoco.mjtObj.mjOBJ_BODY, body_id, "body")


def csv_vec(values: Iterable[Any]) -> str:
    """Format vector-like values for compact CSV cells."""

    return " ".join(f"{float(value):.9g}" for value in values)


def infer_side(name: str) -> str:
    """Infer left/right side from common Seedon naming patterns."""

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


def leg_actuator_joint_pairs(model: mujoco.MjModel) -> list[tuple[int, int]]:
    """Return actuator/joint pairs that look like Seedon leg joints."""

    pairs: list[tuple[int, int]] = []
    seen_joints: set[int] = set()
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        if not (0 <= joint_id < model.njnt):
            continue
        combined = f"{actuator_name(model, actuator_id)} {joint_name(model, joint_id)}"
        if infer_side(combined) == "unknown" or infer_dof(combined) == "unknown":
            continue
        pairs.append((actuator_id, joint_id))
        seen_joints.add(joint_id)
    for joint_id in range(model.njnt):
        if joint_id in seen_joints:
            continue
        name = joint_name(model, joint_id)
        if infer_side(name) != "unknown" and infer_dof(name) != "unknown":
            pairs.append((-1, joint_id))
    return pairs


def load_m0_mapping(path: Path) -> dict[str, Any]:
    """Load Phase M0 sign mapping if available."""

    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def apply_base_lift(model: mujoco.MjModel, data: mujoco.MjData, base_lift: float) -> None:
    """Lift the free root joint, if the model has one."""

    for joint_id in range(model.njnt):
        if int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE):
            qposadr = int(model.jnt_qposadr[joint_id])
            data.qpos[qposadr + 2] += base_lift
            return
    if data.qpos.size >= 3:
        data.qpos[2] += base_lift


def reset_lifted(model: mujoco.MjModel, data: mujoco.MjData, base_lift: float) -> np.ndarray:
    """Reset, lift the base, and return a neutral qpos copy."""

    mujoco.mj_resetData(model, data)
    apply_base_lift(model, data, base_lift)
    data.qvel[:] = 0.0
    if data.ctrl.size:
        data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)
    return data.qpos.copy()


def named_points(model: mujoco.MjModel, data: mujoco.MjData, side: str, tokens: tuple[str, ...]) -> list[np.ndarray]:
    """Collect candidate world-space points from matching sites, bodies, and geoms."""

    points: list[np.ndarray] = []
    for site_id in range(model.nsite):
        name = site_name(model, site_id).lower()
        if infer_side(name) == side and any(token in name for token in tokens):
            points.append(data.site_xpos[site_id].copy())
    for body_id in range(model.nbody):
        name = body_name(model, body_id).lower()
        if infer_side(name) == side and any(token in name for token in tokens):
            points.append(data.xpos[body_id].copy())
    for geom_id in range(model.ngeom):
        name = geom_name(model, geom_id).lower()
        if infer_side(name) == side and any(token in name for token in tokens):
            points.append(data.geom_xpos[geom_id].copy())
    return points


def representative_point(model: mujoco.MjModel, data: mujoco.MjData, side: str, tokens: tuple[str, ...]) -> np.ndarray:
    """Return a representative world point for a side and token set."""

    points = named_points(model, data, side, tokens)
    if not points:
        return np.array([float("nan"), float("nan"), float("nan")], dtype=np.float64)
    return np.mean(np.vstack(points), axis=0)


def contact_count(model: mujoco.MjModel, data: mujoco.MjData) -> int:
    """Return current contact count after a forward pass."""

    del model
    return int(data.ncon)


def set_joint_qpos(model: mujoco.MjModel, data: mujoco.MjData, joint_id: int, value: float) -> None:
    """Set a scalar hinge/slide joint qpos value."""

    joint_type = int(model.jnt_type[joint_id])
    if joint_type not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
        raise ValueError(f"Joint is not scalar hinge/slide: {joint_name(model, joint_id)}")
    qposadr = int(model.jnt_qposadr[joint_id])
    data.qpos[qposadr] = value


def probe_joint(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    neutral_qpos: np.ndarray,
    joint_id: int,
    perturbation: float,
) -> dict[str, Any]:
    """Probe one scalar joint perturbation and return foot/toe deltas."""

    data.qpos[:] = neutral_qpos
    data.qvel[:] = 0.0
    if data.ctrl.size:
        data.ctrl[:] = 0.0
    qposadr = int(model.jnt_qposadr[joint_id])
    set_joint_qpos(model, data, joint_id, float(neutral_qpos[qposadr] + perturbation))
    mujoco.mj_forward(model, data)

    return {
        "left_foot": representative_point(model, data, "left", FOOT_TOKENS),
        "right_foot": representative_point(model, data, "right", FOOT_TOKENS),
        "left_toe": representative_point(model, data, "left", TOE_TOKENS),
        "right_toe": representative_point(model, data, "right", TOE_TOKENS),
        "contact_count": contact_count(model, data),
    }


def classify_effect(side: str, dof: str, foot_delta: np.ndarray, toe_delta: np.ndarray) -> str:
    """Return a compact engineering interpretation for a positive perturbation."""

    del side
    if dof == "hip_pitch":
        return "positive moves foot forward" if foot_delta[0] > 0.0 else "positive moves foot backward"
    if dof == "knee_pitch":
        return "positive lifts foot" if foot_delta[2] > 0.0 else "positive lowers foot"
    if dof == "ankle_pitch":
        return "positive raises toe" if toe_delta[2] > 0.0 else "positive lowers toe"
    if dof == "hip_roll":
        return "positive shifts foot laterally; inspect dy"
    return f"positive delta dx={foot_delta[0]:.6f}, dy={foot_delta[1]:.6f}, dz={foot_delta[2]:.6f}"


def m0_agreement(m0_entry: dict[str, Any] | None, positive_foot_delta: np.ndarray) -> bool | str:
    """Compare positive perturbation foot sign with Phase M0 mapping."""

    if not m0_entry:
        return "inconclusive"
    checks = []
    for key, index in (("positive_dx", 0), ("positive_dy", 1), ("positive_dz", 2)):
        m0_value = float(m0_entry.get(key, 0.0))
        current = float(positive_foot_delta[index])
        if abs(m0_value) <= 1e-9 or abs(current) <= 1e-9:
            continue
        checks.append((m0_value > 0.0) == (current > 0.0))
    if not checks:
        return "inconclusive"
    return all(checks)


def run_validation(config: ValidationConfig) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run free-space joint sign validation.

    Parameters:
        config: Runtime configuration.

    Returns:
        Tuple of CSV rows and summary mapping payload.
    """

    model = mujoco.MjModel.from_xml_path(str(config.scene_path))
    data = mujoco.MjData(model)
    neutral_qpos = reset_lifted(model, data, config.base_lift)
    neutral_points = {
        "left_foot": representative_point(model, data, "left", FOOT_TOKENS),
        "right_foot": representative_point(model, data, "right", FOOT_TOKENS),
        "left_toe": representative_point(model, data, "left", TOE_TOKENS),
        "right_toe": representative_point(model, data, "right", TOE_TOKENS),
    }
    m0_mapping = load_m0_mapping(config.m0_sign_mapping)

    rows: list[dict[str, Any]] = []
    summary_mapping: dict[str, Any] = {}
    for actuator_id, joint_id in leg_actuator_joint_pairs(model):
        joint = joint_name(model, joint_id)
        actuator = actuator_name(model, actuator_id) if actuator_id >= 0 else ""
        side = infer_side(f"{actuator} {joint}")
        dof = infer_dof(f"{actuator} {joint}")
        positive = probe_joint(model, data, neutral_qpos, joint_id, config.epsilon)
        negative = probe_joint(model, data, neutral_qpos, joint_id, -config.epsilon)
        contact_max = max(int(positive["contact_count"]), int(negative["contact_count"]))
        reliable = contact_max == 0
        positive_foot_delta = positive[f"{side}_foot"] - neutral_points[f"{side}_foot"]
        positive_toe_delta = positive[f"{side}_toe"] - neutral_points[f"{side}_toe"]
        m0_entry = m0_mapping.get(joint)
        agrees = m0_agreement(m0_entry, positive_foot_delta)

        for sign, probe in (("+", positive), ("-", negative)):
            left_foot_delta = probe["left_foot"] - neutral_points["left_foot"]
            right_foot_delta = probe["right_foot"] - neutral_points["right_foot"]
            left_toe_delta = probe["left_toe"] - neutral_points["left_toe"]
            right_toe_delta = probe["right_toe"] - neutral_points["right_toe"]
            rows.append(
                {
                    "actuator_name": actuator,
                    "joint_name": joint,
                    "side": side,
                    "dof_guess": dof,
                    "perturbation_sign": sign,
                    "epsilon": config.epsilon,
                    "left_foot_dx": left_foot_delta[0],
                    "left_foot_dy": left_foot_delta[1],
                    "left_foot_dz": left_foot_delta[2],
                    "right_foot_dx": right_foot_delta[0],
                    "right_foot_dy": right_foot_delta[1],
                    "right_foot_dz": right_foot_delta[2],
                    "left_toe_dx": left_toe_delta[0],
                    "left_toe_dz": left_toe_delta[2],
                    "right_toe_dx": right_toe_delta[0],
                    "right_toe_dz": right_toe_delta[2],
                    "contact_count": int(probe["contact_count"]),
                    "validation_reliable": reliable,
                    "inferred_effect": classify_effect(side, dof, positive_foot_delta, positive_toe_delta),
                    "m0_mapping_effect": str(m0_entry.get("interpretation", "")) if m0_entry else "",
                    "agrees_with_m0_mapping": agrees,
                }
            )

        summary_mapping[joint] = {
            "actuator_name": actuator,
            "side": side,
            "dof": dof,
            "hip_pitch_forward_sign": sign_label(positive_foot_delta[0]) if dof == "hip_pitch" else "",
            "knee_lift_sign": sign_label(positive_foot_delta[2]) if dof == "knee_pitch" else "",
            "ankle_toe_up_sign": sign_label(positive_toe_delta[2]) if dof == "ankle_pitch" else "",
            "hip_roll_lateral_sign": sign_label(positive_foot_delta[1]) if dof == "hip_roll" else "",
            "positive_dx": float(positive_foot_delta[0]),
            "positive_dy": float(positive_foot_delta[1]),
            "positive_dz": float(positive_foot_delta[2]),
            "positive_toe_dx": float(positive_toe_delta[0]),
            "positive_toe_dz": float(positive_toe_delta[2]),
            "confidence": "high" if reliable and agrees is not False else "medium" if reliable else "low",
            "disagreement_with_m0": agrees is False,
        }
    return rows, summary_mapping


def sign_label(value: float) -> str:
    """Return sign label for a scalar effect."""

    if value > 1e-9:
        return "+epsilon"
    if value < -1e-9:
        return "-epsilon"
    return "near_zero"


def build_summary(mapping: dict[str, Any]) -> dict[str, Any]:
    """Build high-level validation summary JSON."""

    disagreements = [joint for joint, item in mapping.items() if item.get("disagreement_with_m0")]
    reliable = [item for item in mapping.values() if item.get("confidence") in {"high", "medium"}]
    return {
        "validated_joint_count": len(mapping),
        "reliable_joint_count": len(reliable),
        "free_space_joint_sign_validated": bool(mapping) and len(disagreements) == 0 and len(reliable) == len(mapping),
        "m0_vs_free_space_sign_disagreements_count": len(disagreements),
        "disagreement_with_m0": disagreements,
        "per_joint": mapping,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """Write CSV rows with a stable header."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, config: ValidationConfig, summary: dict[str, Any]) -> None:
    """Write Markdown validation report."""

    lines = [
        "# Phase Pre-W0 Free-Space Joint Sign Validation",
        "",
        f"- scene_path: `{config.scene_path}`",
        f"- m0_sign_mapping: `{config.m0_sign_mapping}`",
        f"- epsilon: `{config.epsilon}`",
        f"- steps: `{config.steps}`",
        f"- base_lift: `{config.base_lift}`",
        "",
        "## Findings",
        "",
        f"- validated_joint_count: `{summary['validated_joint_count']}`",
        f"- reliable_joint_count: `{summary['reliable_joint_count']}`",
        f"- free_space_joint_sign_validated: `{summary['free_space_joint_sign_validated']}`",
        f"- m0_vs_free_space_sign_disagreements_count: `{summary['m0_vs_free_space_sign_disagreements_count']}`",
        "",
        "## Engineering Interpretation",
        "",
        "This probe is kinematic and intentionally avoids integrating dynamics. That keeps contact constraints from hiding the sign of each joint effect.",
        "If any row reports contact_count > 0, treat that joint effect as low-confidence and inspect the scene/base lift before W0.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-path", type=Path, default=DEFAULT_SCENE_PATH)
    parser.add_argument("--m0-sign-mapping", type=Path, default=DEFAULT_M0_SIGN_MAPPING)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epsilon", type=float, default=0.03)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--base-lift", type=float, default=0.20)
    return parser.parse_args()


def main() -> None:
    """Run free-space joint sign validation."""

    args = parse_args()
    if args.epsilon <= 0.0:
        raise ValueError("--epsilon must be positive")
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.base_lift < 0.0:
        raise ValueError("--base-lift must be non-negative")
    config = ValidationConfig(
        scene_path=require_scene(args.scene_path),
        m0_sign_mapping=args.m0_sign_mapping,
        output_dir=args.output_dir,
        epsilon=float(args.epsilon),
        steps=int(args.steps),
        base_lift=float(args.base_lift),
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)

    rows, mapping = run_validation(config)
    summary = build_summary(mapping)
    write_csv(
        config.output_dir / "seedon_free_space_joint_effect_probe.csv",
        rows,
        [
            "actuator_name",
            "joint_name",
            "side",
            "dof_guess",
            "perturbation_sign",
            "epsilon",
            "left_foot_dx",
            "left_foot_dy",
            "left_foot_dz",
            "right_foot_dx",
            "right_foot_dy",
            "right_foot_dz",
            "left_toe_dx",
            "left_toe_dz",
            "right_toe_dx",
            "right_toe_dz",
            "contact_count",
            "validation_reliable",
            "inferred_effect",
            "m0_mapping_effect",
            "agrees_with_m0_mapping",
        ],
    )
    (config.output_dir / "seedon_free_space_joint_sign_mapping.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(config.output_dir / "seedon_free_space_joint_sign_validation_report.md", config, summary)

    print(f"effect_probe={config.output_dir / 'seedon_free_space_joint_effect_probe.csv'}")
    print(f"mapping={config.output_dir / 'seedon_free_space_joint_sign_mapping.json'}")
    print(f"report={config.output_dir / 'seedon_free_space_joint_sign_validation_report.md'}")
    print(f"free_space_joint_sign_validated={summary['free_space_joint_sign_validated']}")


if __name__ == "__main__":
    main()
