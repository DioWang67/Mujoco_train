"""Phase G1 raw contact-pair diagnostic for Sedon foot geometry.

This Class C diagnostic inspects raw MuJoCo contact pairs to explain why the
Phase 2C right-support profile showed zero right-center force, high right-toe
force, and no toe handoff. It does not train, edit rewards, modify Sedon
training code, or change the source scene XML.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from tools.sedon_debug_common import DEBUG_OUT_DIR, DEFAULT_SCENE_PATH, geom_name, geom_type_name, require_scene


DEFAULT_V5A_SCENE = DEBUG_OUT_DIR / "blue_like_sole_experiments_v5" / "training_scene_v5_a.xml"
DEFAULT_OUT_DIR = DEBUG_OUT_DIR / "phase_g1_raw_contact_pair_diagnostic"
FOOT_NAME_TOKENS = ("foot", "toe", "heel", "center", "sole", "bottom", "collision")
SUMMARY_FORCE_FIELDS = (
    "right_center_force",
    "right_toe_force",
    "right_heel_force",
    "right_foot_bottom_force",
    "left_center_force",
    "left_toe_force",
    "left_heel_force",
    "left_foot_bottom_force",
    "unknown_right_foot_force",
    "unknown_left_foot_force",
)


@dataclass(frozen=True)
class DiagnosticConfig:
    """Runtime configuration for Phase G1 diagnostics."""

    scene_path: Path
    steps: int
    output_dir: Path
    pitch_sweep: bool
    neutral_only: bool


def default_scene_path() -> Path:
    """Return the best available diagnostic scene without modifying XML files."""
    return DEFAULT_V5A_SCENE if DEFAULT_V5A_SCENE.is_file() else DEFAULT_SCENE_PATH


def csv_text(value: Any) -> str:
    """Format scalar/list values for compact CSV cells."""
    if isinstance(value, np.ndarray):
        return " ".join(f"{float(item):.9g}" for item in value.reshape(-1))
    if isinstance(value, (list, tuple)):
        return " ".join(str(item) for item in value)
    return str(value)


def body_name(model: mujoco.MjModel, body_id: int) -> str:
    """Return a stable body name."""
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(body_id))
    return name or f"body_{body_id}"


def is_floor_name(name: str) -> bool:
    """Return whether a geom name should be treated as floor/ground."""
    lowered = name.lower()
    return lowered in {"floor", "ground"} or "floor" in lowered or "ground" in lowered


def side_for_geom(name: str) -> str:
    """Classify a Sedon foot geom side from its name."""
    lowered = name.lower()
    if lowered.startswith("r_") or lowered.startswith("right"):
        return "right"
    if lowered.startswith("l_") or lowered.startswith("left"):
        return "left"
    return "unknown"


def region_for_geom(name: str) -> str:
    """Classify a Sedon foot geom contact region from its name."""
    lowered = name.lower()
    if "toe" in lowered:
        return "toe"
    if "heel" in lowered:
        return "heel"
    if "center" in lowered:
        return "center"
    if lowered in {"r_foot_collision", "l_foot_collision"}:
        return "center"
    if "bottom" in lowered or "sole" in lowered:
        return "foot_bottom"
    if "foot" in lowered and "collision" in lowered:
        return "unknown"
    return "unknown"


def classify_contact(name_a: str, name_b: str) -> tuple[str, str]:
    """Return side and region for a contact pair, if any foot geom is involved."""
    for name in (name_a, name_b):
        side = side_for_geom(name)
        if side in {"right", "left"} and any(token in name.lower() for token in FOOT_NAME_TOKENS):
            return side, region_for_geom(name)
    return "unknown", "unknown"


def contact_force(model: mujoco.MjModel, data: mujoco.MjData, contact_index: int) -> tuple[float, float, float, float]:
    """Return normal force and first three contact-frame force components."""
    wrench = np.zeros(6, dtype=np.float64)
    mujoco.mj_contactForce(model, data, contact_index, wrench)
    return abs(float(wrench[0])), float(wrench[0]), float(wrench[1]), float(wrench[2])


def quat_from_pitch(pitch: float) -> np.ndarray:
    """Return a wxyz quaternion for a pure pitch rotation."""
    half = 0.5 * pitch
    return np.array([math.cos(half), 0.0, math.sin(half), 0.0], dtype=np.float64)


def reset_and_settle(model: mujoco.MjModel, data: mujoco.MjData, steps: int = 20) -> None:
    """Reset and settle the model under zero control."""
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    for _ in range(steps):
        if data.ctrl.size:
            data.ctrl[:] = 0.0
        mujoco.mj_step(model, data)


def apply_pitch_for_scenario(data: mujoco.MjData, scenario: str, step: int, steps: int) -> None:
    """Apply slow base pitch sweep for diagnostic scenarios."""
    if scenario == "neutral":
        return
    if steps <= 1:
        alpha = 1.0
    else:
        alpha = step / float(steps - 1)
    max_pitch = math.radians(10.0)
    pitch = alpha * max_pitch
    if scenario == "pitch_backward":
        pitch = -pitch
    data.qpos[3:7] = quat_from_pitch(pitch)
    data.qvel[3:6] = 0.0


def scenario_names(config: DiagnosticConfig) -> list[str]:
    """Return scenario names requested by CLI."""
    if config.neutral_only:
        return ["neutral"]
    scenarios = ["neutral"]
    if config.pitch_sweep:
        scenarios.extend(["pitch_forward", "pitch_backward"])
    return scenarios


def foot_geom_inventory(model: mujoco.MjModel) -> list[dict[str, Any]]:
    """Return all foot/toe/heel/center/sole/bottom/collision-related geoms."""
    rows: list[dict[str, Any]] = []
    for geom_id in range(model.ngeom):
        name = geom_name(model, geom_id)
        lowered = name.lower()
        if not any(token in lowered for token in FOOT_NAME_TOKENS):
            continue
        rows.append(
            {
                "geom_id": geom_id,
                "geom_name": name,
                "body_name": body_name(model, int(model.geom_bodyid[geom_id])),
                "type": geom_type_name(model, geom_id),
                "pos": csv_text(model.geom_pos[geom_id]),
                "size": csv_text(model.geom_size[geom_id]),
                "contype": int(model.geom_contype[geom_id]),
                "conaffinity": int(model.geom_conaffinity[geom_id]),
                "friction": csv_text(model.geom_friction[geom_id]),
                "rgba": csv_text(model.geom_rgba[geom_id]),
            }
        )
    return rows


def collect_step_contacts(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    scenario: str,
    step: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Collect raw contact-pair rows and one per-step region summary."""
    raw_rows: list[dict[str, Any]] = []
    summary = {field: 0.0 for field in SUMMARY_FORCE_FIELDS}
    raw_right_count = 0
    raw_left_count = 0
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        geom1_id = int(contact.geom1)
        geom2_id = int(contact.geom2)
        name1 = geom_name(model, geom1_id)
        name2 = geom_name(model, geom2_id)
        side, region = classify_contact(name1, name2)
        normal_force, force_x, force_y, force_z = contact_force(model, data, contact_index)
        is_floor_contact = is_floor_name(name1) or is_floor_name(name2)
        if is_floor_contact and side in {"right", "left"}:
            if side == "right":
                raw_right_count += 1
            else:
                raw_left_count += 1
            key = f"{side}_{region}_force"
            if key in summary:
                summary[key] += normal_force
            else:
                summary[f"unknown_{side}_foot_force"] += normal_force
        raw_rows.append(
            {
                "scenario": scenario,
                "step": step,
                "time": float(data.time),
                "geom1_id": geom1_id,
                "geom1_name": name1,
                "geom2_id": geom2_id,
                "geom2_name": name2,
                "body1_name": body_name(model, int(model.geom_bodyid[geom1_id])),
                "body2_name": body_name(model, int(model.geom_bodyid[geom2_id])),
                "contact_pos_x": float(contact.pos[0]),
                "contact_pos_y": float(contact.pos[1]),
                "contact_pos_z": float(contact.pos[2]),
                "contact_dist": float(contact.dist),
                "normal_force": normal_force,
                "force_x": force_x,
                "force_y": force_y,
                "force_z": force_z,
                "classified_side": side,
                "classified_region": region,
                "is_floor_contact": bool(is_floor_contact),
            }
        )
    summary.update(
        {
            "scenario": scenario,
            "step": step,
            "time": float(data.time),
            "raw_right_foot_contact_count": raw_right_count,
            "raw_left_foot_contact_count": raw_left_count,
        }
    )
    return raw_rows, summary


def run_diagnostic(config: DiagnosticConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Run requested Phase G1 diagnostic scenarios."""
    model = mujoco.MjModel.from_xml_path(str(config.scene_path))
    data = mujoco.MjData(model)
    inventory = foot_geom_inventory(model)
    raw_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for scenario in scenario_names(config):
        reset_and_settle(model, data)
        for step in range(config.steps):
            apply_pitch_for_scenario(data, scenario, step, config.steps)
            if data.ctrl.size:
                data.ctrl[:] = 0.0
            mujoco.mj_forward(model, data)
            mujoco.mj_step(model, data)
            step_raw, step_summary = collect_step_contacts(model, data, scenario=scenario, step=step)
            raw_rows.extend(step_raw)
            summary_rows.append(step_summary)
    return raw_rows, summary_rows, inventory


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """Write rows to CSV with a stable header, even when rows are empty."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def max_force(rows: list[dict[str, Any]], field: str) -> float:
    """Return max force value for a summary field."""
    return max((float(row.get(field, 0.0)) for row in rows), default=0.0)


def detected_geom(inventory: list[dict[str, Any]], side: str, region: str) -> bool:
    """Return whether inventory contains a side/region geom."""
    prefix = "R_" if side == "right" else "L_"
    for row in inventory:
        name = str(row["geom_name"])
        if name.startswith(prefix) and region_for_geom(name) == region:
            return True
    return False


def unknown_force_ratio(rows: list[dict[str, Any]]) -> float:
    """Return unknown-foot force ratio among all classified foot floor contacts."""
    unknown = 0.0
    total = 0.0
    for row in rows:
        fields = {key: float(row.get(key, 0.0)) for key in SUMMARY_FORCE_FIELDS}
        unknown += fields["unknown_right_foot_force"] + fields["unknown_left_foot_force"]
        total += sum(fields.values())
    return unknown / total if total > 1e-9 else 0.0


def neutral_dominates(rows: list[dict[str, Any]]) -> bool:
    """Return whether right toe force dominates right center during neutral."""
    neutral_rows = [row for row in rows if row.get("scenario") == "neutral"]
    if not neutral_rows:
        return False
    max_toe = max_force(neutral_rows, "right_toe_force")
    max_center = max_force(neutral_rows, "right_center_force")
    return max_toe > 0.1 and max_toe > max_center


def infer_likely_root_cause(summary: dict[str, Any]) -> str:
    """Infer a concise root-cause hypothesis from Phase G1 metrics."""
    if not summary["center_geom_detected_right"]:
        return "right_center_geom_missing"
    if not summary["any_right_center_raw_contact"]:
        return "right_center_geom_exists_but_never_contacts_floor"
    if summary["right_center_force_zero_all_steps"]:
        return "right_center_raw_contact_absent_or_zero_force"
    if summary["right_toe_force_dominates_neutral"]:
        return "right_toe_contacts_too_early_or_bridges_neutral_stance"
    if summary["contact_classifier_unknown_force_ratio"] > 0.25:
        return "classifier_unknown_force_ratio_high"
    return "inconclusive_contact_geometry_requires_pitch_profile_review"


def build_summary(summary_rows: list[dict[str, Any]], inventory: list[dict[str, Any]]) -> dict[str, Any]:
    """Build Phase G1 summary JSON payload."""
    max_right_center = max_force(summary_rows, "right_center_force")
    payload: dict[str, Any] = {
        "center_geom_detected_right": detected_geom(inventory, "right", "center"),
        "center_geom_detected_left": detected_geom(inventory, "left", "center"),
        "any_right_center_raw_contact": any(float(row["right_center_force"]) > 0.0 for row in summary_rows),
        "any_left_center_raw_contact": any(float(row["left_center_force"]) > 0.0 for row in summary_rows),
        "max_right_center_force": max_right_center,
        "max_right_toe_force": max_force(summary_rows, "right_toe_force"),
        "max_right_heel_force": max_force(summary_rows, "right_heel_force"),
        "max_left_center_force": max_force(summary_rows, "left_center_force"),
        "max_left_toe_force": max_force(summary_rows, "left_toe_force"),
        "max_left_heel_force": max_force(summary_rows, "left_heel_force"),
        "right_toe_force_dominates_neutral": neutral_dominates(summary_rows),
        "right_center_force_zero_all_steps": max_right_center <= 1e-9,
        "contact_classifier_unknown_force_ratio": unknown_force_ratio(summary_rows),
    }
    payload["likely_root_cause"] = infer_likely_root_cause(payload)
    return payload


def write_report(path: Path, config: DiagnosticConfig, summary: dict[str, Any]) -> None:
    """Write Phase G1 engineering report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    center_exists = bool(summary["center_geom_detected_right"])
    center_contact = bool(summary["any_right_center_raw_contact"])
    toe_dominates = bool(summary["right_toe_force_dominates_neutral"])
    classifier_likely = center_exists and center_contact and summary["max_right_center_force"] > 0.0
    lines = [
        "# Phase G1 Raw Contact-Pair Diagnostic",
        "",
        f"- scene_path: `{config.scene_path}`",
        f"- steps: `{config.steps}`",
        f"- pitch_sweep: `{config.pitch_sweep}`",
        f"- neutral_only: `{config.neutral_only}`",
        "",
        "## Findings",
        "",
        f"- right center geom exists: `{center_exists}`",
        f"- right center raw contact pair observed: `{center_contact}`",
        f"- max_right_center_force: `{summary['max_right_center_force']:.6f}`",
        f"- max_right_toe_force: `{summary['max_right_toe_force']:.6f}`",
        f"- max_right_heel_force: `{summary['max_right_heel_force']:.6f}`",
        f"- right toe force dominates neutral: `{toe_dominates}`",
        f"- contact_classifier_unknown_force_ratio: `{summary['contact_classifier_unknown_force_ratio']:.6f}`",
        f"- likely_root_cause: `{summary['likely_root_cause']}`",
        "",
        "## Engineering Answers",
        "",
        f"Right center geom exists: {'yes' if center_exists else 'no'}.",
        "",
        f"Right center geom has raw contact pair: {'yes' if center_contact else 'no'}.",
        "",
        (
            "Right center force = 0 is more likely a classifier/reporting issue because raw right-center force is present."
            if classifier_likely
            else "Right center force = 0 is more likely physical/contact-geometry related in this diagnostic, because raw right-center force was not observed."
        ),
        "",
        (
            "Toe dominates in neutral stance, which supports the hypothesis that toe collision touches too early or bridges the intended center-first support."
            if toe_dominates
            else "Toe did not dominate neutral stance in this run; pitch/profile rows should be inspected before blaming neutral toe bridging."
        ),
        "",
        "A simplified foot_bottom_collision comparison is supported if center geoms are missing, never contact, or toe/heel geoms dominate neutral stance. This diagnostic should be reviewed before PPO training.",
        "",
        "## Next Recommendations",
        "",
        "1. If center geoms are absent or never contact, repair right foot center patch pose/size/collision settings.",
        "2. If toe dominates neutral, adjust toe rocker height/spacing/margin and rerun Phase G1.",
        "3. If raw center contacts exist but old summaries still show zero center force, fix the classifier and validate against raw_contact_pairs.csv.",
        "4. Do not run PPO training until center-first and toe-handoff contact semantics are physically observable.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-path", type=Path, default=default_scene_path())
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--pitch-sweep", action="store_true", default=False)
    parser.add_argument("--neutral-only", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    """Run Phase G1 raw contact-pair diagnostics."""
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    config = DiagnosticConfig(
        scene_path=require_scene(args.scene_path),
        steps=args.steps,
        output_dir=args.output_dir,
        pitch_sweep=bool(args.pitch_sweep),
        neutral_only=bool(args.neutral_only),
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)

    raw_rows, summary_rows, inventory_rows = run_diagnostic(config)
    summary = build_summary(summary_rows, inventory_rows)

    write_csv(
        config.output_dir / "raw_contact_pairs.csv",
        raw_rows,
        [
            "scenario",
            "step",
            "time",
            "geom1_id",
            "geom1_name",
            "geom2_id",
            "geom2_name",
            "body1_name",
            "body2_name",
            "contact_pos_x",
            "contact_pos_y",
            "contact_pos_z",
            "contact_dist",
            "normal_force",
            "force_x",
            "force_y",
            "force_z",
            "classified_side",
            "classified_region",
            "is_floor_contact",
        ],
    )
    write_csv(
        config.output_dir / "contact_region_summary.csv",
        summary_rows,
        [
            "scenario",
            "step",
            "time",
            *SUMMARY_FORCE_FIELDS,
            "raw_right_foot_contact_count",
            "raw_left_foot_contact_count",
        ],
    )
    write_csv(
        config.output_dir / "contact_geom_inventory.csv",
        inventory_rows,
        [
            "geom_id",
            "geom_name",
            "body_name",
            "type",
            "pos",
            "size",
            "contype",
            "conaffinity",
            "friction",
            "rgba",
        ],
    )
    (config.output_dir / "phase_g1_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(config.output_dir / "phase_g1_report.md", config, summary)

    print(f"raw_contact_pairs={config.output_dir / 'raw_contact_pairs.csv'}")
    print(f"contact_region_summary={config.output_dir / 'contact_region_summary.csv'}")
    print(f"contact_geom_inventory={config.output_dir / 'contact_geom_inventory.csv'}")
    print(f"summary={config.output_dir / 'phase_g1_summary.json'}")
    print(f"report={config.output_dir / 'phase_g1_report.md'}")
    print(f"likely_root_cause={summary['likely_root_cause']}")


if __name__ == "__main__":
    main()
