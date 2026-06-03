"""Compare Seedon foot-contact prototype variants.

This Class C diagnostic runs short MuJoCo contact checks only. It does not
modify the source Seedon model, train.py, eval.py, or PPO settings, and it does
not claim walking success.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from tools.seedon_debug_common import DEBUG_OUT_DIR, geom_name, require_scene


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_VARIANT_DIR = DEBUG_OUT_DIR / "foot_contact_variants"
DEFAULT_MANIFEST = DEFAULT_VARIANT_DIR / "manifest.json"
DEFAULT_REPORT = REPO_ROOT / "docs" / "seedon_foot_contact_variant_comparison.md"
FLOOR_NAMES = {"floor", "ground"}
FORCE_EPS = 1e-5
PATCH_METADATA = {
    "source": "assumption/prototype",
    "confidence": "low",
    "valid_for": "simulation_prototype_only",
}
REGIONS = ("center", "toe", "heel", "inner", "outer", "unknown")
FORCE_FIELDS = tuple(f"{side}_{region}_force" for side in ("left", "right") for region in REGIONS)


@dataclass(frozen=True)
class Variant:
    """One generated foot contact variant.

    Args:
        name: Variant name.
        scene_path: Generated scene path.
        added_patches: Manifest patch metadata.
        status: Generation status.
    """

    name: str
    scene_path: Path
    added_patches: list[dict[str, Any]]
    status: str


@dataclass(frozen=True)
class Scenario:
    """One contact-comparison pose scenario."""

    name: str
    pitch_rad: float = 0.0
    roll_rad: float = 0.0


SCENARIOS = (
    Scenario("neutral"),
    Scenario("forward_pitch", pitch_rad=math.radians(8.0)),
    Scenario("backward_pitch", pitch_rad=math.radians(-8.0)),
    Scenario("left_support_roll", roll_rad=math.radians(5.0)),
    Scenario("right_support_roll", roll_rad=math.radians(-5.0)),
)


def load_manifest(path: Path) -> dict[str, Any]:
    """Load the generator manifest.

    Args:
        path: Manifest path.

    Returns:
        Parsed JSON object.

    Raises:
        ValueError: If the manifest is missing or malformed.
    """

    if not path.is_file():
        raise ValueError(f"Manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("variants"), list):
        raise ValueError(f"Manifest must contain a variants list: {path}")
    return payload


def load_variants(path: Path) -> list[Variant]:
    """Load generated variant references from a manifest."""

    manifest = load_manifest(path)
    variants: list[Variant] = []
    for item in manifest["variants"]:
        scene_path = require_scene(Path(item["scene_path"]))
        variants.append(
            Variant(
                name=str(item["name"]),
                scene_path=scene_path,
                added_patches=list(item.get("added_patches", [])),
                status=str(item.get("status", "unknown")),
            )
        )
    return variants


def quat_from_roll_pitch(roll: float, pitch: float) -> np.ndarray:
    """Return a wxyz quaternion for roll then pitch diagnostic poses."""

    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    # yaw is zero; q = q_pitch * q_roll in wxyz convention.
    return np.array(
        [
            cp * cr,
            cp * sr,
            sp * cr,
            -sp * sr,
        ],
        dtype=np.float64,
    )


def body_name(model: mujoco.MjModel, body_id: int) -> str:
    """Return a stable body name."""

    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(body_id))
    return name or f"body_{body_id}"


def side_for_geom(name: str) -> str:
    """Classify Seedon side from a geom name."""

    lowered = name.lower()
    if lowered.startswith("r_") or lowered.startswith("right"):
        return "right"
    if lowered.startswith("l_") or lowered.startswith("left"):
        return "left"
    return "unknown"


def region_for_geom(name: str) -> str:
    """Classify contact region from prototype or baseline geom names."""

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
    """Return whether a geom name is floor-like."""

    lowered = name.lower()
    return lowered in FLOOR_NAMES or "floor" in lowered or "ground" in lowered


def classify_pair(name_a: str, name_b: str) -> tuple[str, str]:
    """Return side and region for a floor/foot contact pair."""

    for name in (name_a, name_b):
        side = side_for_geom(name)
        if side in {"left", "right"} and "foot" in name.lower():
            return side, region_for_geom(name)
    return "unknown", "unknown"


def contact_force(model: mujoco.MjModel, data: mujoco.MjData, index: int) -> tuple[float, float, float, float]:
    """Return normal force and first three contact-frame force components."""

    wrench = np.zeros(6, dtype=np.float64)
    mujoco.mj_contactForce(model, data, index, wrench)
    return abs(float(wrench[0])), float(wrench[0]), float(wrench[1]), float(wrench[2])


def reset_and_apply_pose(model: mujoco.MjModel, data: mujoco.MjData, scenario: Scenario, settle_steps: int) -> None:
    """Reset, settle, apply diagnostic base pose, and step once."""

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    for _ in range(max(0, settle_steps)):
        if data.ctrl.size:
            data.ctrl[:] = 0.0
        mujoco.mj_step(model, data)
    data.qpos[3:7] = quat_from_roll_pitch(scenario.roll_rad, scenario.pitch_rad)
    data.qvel[:] = 0.0
    if data.ctrl.size:
        data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)
    mujoco.mj_step(model, data)


def collect_contacts(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    variant_name: str,
    scenario_name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Collect raw contact rows and one scenario force summary."""

    raw_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "variant": variant_name,
        "scenario": scenario_name,
        **{field: 0.0 for field in FORCE_FIELDS},
        "left_contact_count": 0,
        "right_contact_count": 0,
        "unknown_contact_count": 0,
    }
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        geom1_id = int(contact.geom1)
        geom2_id = int(contact.geom2)
        name1 = geom_name(model, geom1_id)
        name2 = geom_name(model, geom2_id)
        normal_force, force_x, force_y, force_z = contact_force(model, data, contact_index)
        floor_contact = is_floor_name(name1) or is_floor_name(name2)
        side, region = classify_pair(name1, name2) if floor_contact else ("unknown", "unknown")
        if floor_contact and side in {"left", "right"}:
            summary[f"{side}_contact_count"] += 1
            summary[f"{side}_{region}_force"] += normal_force
        elif floor_contact:
            summary["unknown_contact_count"] += 1
        raw_rows.append(
            {
                "variant": variant_name,
                "scenario": scenario_name,
                "contact_index": contact_index,
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
                "is_floor_contact": bool(floor_contact),
            }
        )
    return raw_rows, summary


def patch_inventory(model: mujoco.MjModel, variant: Variant) -> list[dict[str, Any]]:
    """Return foot-related geoms and manifest metadata for one variant."""

    manifest_by_name = {str(item["geom_name"]): item for item in variant.added_patches}
    rows: list[dict[str, Any]] = []
    for geom_id in range(model.ngeom):
        name = geom_name(model, geom_id)
        if "foot" not in name.lower():
            continue
        manifest_item = manifest_by_name.get(name, {})
        rows.append(
            {
                "variant": variant.name,
                "geom_id": geom_id,
                "geom_name": name,
                "body_name": body_name(model, int(model.geom_bodyid[geom_id])),
                "side": side_for_geom(name),
                "region": region_for_geom(name),
                "is_added_patch": bool(manifest_item),
                "source": manifest_item.get("source", "seedon_source_scene"),
                "confidence": manifest_item.get("confidence", "source_scene"),
                "valid_for": manifest_item.get("valid_for", "existing_seedon_contact"),
                "contype": int(model.geom_contype[geom_id]),
                "conaffinity": int(model.geom_conaffinity[geom_id]),
                "size": " ".join(f"{float(value):.9g}" for value in model.geom_size[geom_id]),
            }
        )
    return rows


def run_variant(variant: Variant, settle_steps: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Run all contact comparison scenarios for one variant."""

    model = mujoco.MjModel.from_xml_path(str(variant.scene_path))
    data = mujoco.MjData(model)
    raw_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        reset_and_apply_pose(model, data, scenario, settle_steps)
        step_raw, step_summary = collect_contacts(
            model,
            data,
            variant_name=variant.name,
            scenario_name=scenario.name,
        )
        raw_rows.extend(step_raw)
        summary_rows.append(step_summary)
    return raw_rows, summary_rows, patch_inventory(model, variant)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """Write CSV rows with a stable header."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def scenario_row(rows: list[dict[str, Any]], variant: str, scenario: str) -> dict[str, Any]:
    """Return one summary row for a variant/scenario pair."""

    for row in rows:
        if row["variant"] == variant and row["scenario"] == scenario:
            return row
    return {"variant": variant, "scenario": scenario, **{field: 0.0 for field in FORCE_FIELDS}}


def force(row: dict[str, Any], side: str, region: str) -> float:
    """Return one force field from a summary row."""

    return float(row.get(f"{side}_{region}_force", 0.0))


def has_force(row: dict[str, Any], side: str, region: str) -> bool:
    """Return whether a side/region has non-trivial force."""

    return force(row, side, region) > FORCE_EPS


def can_classify_center_toe_heel(inventory: list[dict[str, Any]], variant: str) -> bool:
    """Return whether both sides have center/toe/heel contact geoms."""

    regions_by_side: dict[str, set[str]] = defaultdict(set)
    for row in inventory:
        if row["variant"] != variant:
            continue
        if int(row.get("contype", 0)) == 0 or int(row.get("conaffinity", 0)) == 0:
            continue
        side = str(row["side"])
        if side in {"left", "right"}:
            regions_by_side[side].add(str(row["region"]))
    return all({"center", "toe", "heel"}.issubset(regions_by_side[side]) for side in ("left", "right"))


def neutral_center_first(row: dict[str, Any], side: str) -> bool:
    """Return whether neutral support is center-first for one side."""

    center = force(row, side, "center")
    toe = force(row, side, "toe")
    heel = force(row, side, "heel")
    return center > FORCE_EPS and center >= toe and center >= heel


def forward_toe_handoff(neutral: dict[str, Any], forward: dict[str, Any], side: str) -> bool:
    """Return whether forward pitch creates a toe-handoff candidate."""

    return (
        has_force(neutral, side, "center")
        and has_force(forward, side, "toe")
        and force(forward, side, "toe") >= force(forward, side, "center") * 0.75
    )


def bridge_detected(rows: list[dict[str, Any]], variant: str, side: str) -> bool:
    """Return whether toe and heel both carry force in the same scenario."""

    for row in rows:
        if row["variant"] != variant:
            continue
        if has_force(row, side, "toe") and has_force(row, side, "heel"):
            return True
    return False


def symmetry_score(rows: list[dict[str, Any]], variant: str) -> float:
    """Return a 0..1 left/right force symmetry score for neutral contact."""

    neutral = scenario_row(rows, variant, "neutral")
    left_total = sum(force(neutral, "left", region) for region in REGIONS)
    right_total = sum(force(neutral, "right", region) for region in REGIONS)
    denom = max(left_total, right_total, FORCE_EPS)
    return 1.0 - min(1.0, abs(left_total - right_total) / denom)


def recommendation(metrics: dict[str, Any]) -> str:
    """Build a conservative recommendation for the mechanical team."""

    if metrics["contact_model_sufficient_for_rollover_analysis"]:
        return "Promote only to scripted rollover diagnostics; still not walking success and not PPO-ready."
    if not metrics["can_classify_center_toe_heel"]:
        return "Not sufficient: add or repair named center/toe/heel patches before rollover analysis."
    if metrics["toe_heel_bridge_contact_detected_left"] or metrics["toe_heel_bridge_contact_detected_right"]:
        return "Not sufficient: toe/heel bridge contact appears; adjust patch spacing or height."
    if not metrics["left_right_symmetry"]:
        return "Not sufficient: left/right neutral contact asymmetry should be corrected first."
    return "Not sufficient: center-first or toe-handoff evidence is incomplete; keep as prototype only."


def build_metrics(summary_rows: list[dict[str, Any]], inventory_rows: list[dict[str, Any]], variants: list[Variant]) -> list[dict[str, Any]]:
    """Build required comparison metrics for every variant."""

    payload: list[dict[str, Any]] = []
    for variant in variants:
        neutral = scenario_row(summary_rows, variant.name, "neutral")
        forward = scenario_row(summary_rows, variant.name, "forward_pitch")
        can_classify = can_classify_center_toe_heel(inventory_rows, variant.name)
        metric = {
            "variant": variant.name,
            "status": variant.status,
            "can_classify_center_toe_heel": can_classify,
            "neutral_center_first_left": neutral_center_first(neutral, "left"),
            "neutral_center_first_right": neutral_center_first(neutral, "right"),
            "forward_pitch_toe_handoff_candidate_left": forward_toe_handoff(neutral, forward, "left"),
            "forward_pitch_toe_handoff_candidate_right": forward_toe_handoff(neutral, forward, "right"),
            "toe_heel_bridge_contact_detected_left": bridge_detected(summary_rows, variant.name, "left"),
            "toe_heel_bridge_contact_detected_right": bridge_detected(summary_rows, variant.name, "right"),
            "left_right_symmetry_score": symmetry_score(summary_rows, variant.name),
        }
        metric["left_right_symmetry"] = bool(metric["left_right_symmetry_score"] >= 0.8)
        metric["contact_model_sufficient_for_rollover_analysis"] = bool(
            can_classify
            and metric["neutral_center_first_left"]
            and metric["neutral_center_first_right"]
            and metric["forward_pitch_toe_handoff_candidate_left"]
            and metric["forward_pitch_toe_handoff_candidate_right"]
            and not metric["toe_heel_bridge_contact_detected_left"]
            and not metric["toe_heel_bridge_contact_detected_right"]
            and metric["left_right_symmetry"]
        )
        metric["recommendation_to_mechanical_team"] = recommendation(metric)
        metric["walking_success_claimed"] = False
        metric.update(PATCH_METADATA)
        payload.append(metric)
    return payload


def write_report(path: Path, metrics: list[dict[str, Any]], manifest_path: Path) -> None:
    """Write the comparison markdown report."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Seedon Foot Contact Variant Comparison",
        "",
        "Task class: Class C contact prototype diagnostic. This report does not modify the source model, enter PPO, or claim walking success.",
        "",
        f"- manifest: `{manifest_path}`",
        "- Duck foot geometry usage: concept/reference only, not Seedon verified geometry.",
        "- Added patch metadata: `source=assumption/prototype`, `confidence=low`, `valid_for=simulation_prototype_only`.",
        "",
        "## Metrics",
        "",
        "| variant | classify C/T/H | neutral center L | neutral center R | forward toe L | forward toe R | bridge L | bridge R | symmetry | rollover-analysis sufficient |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics:
        lines.append(
            f"| `{row['variant']}` | {row['can_classify_center_toe_heel']} | "
            f"{row['neutral_center_first_left']} | {row['neutral_center_first_right']} | "
            f"{row['forward_pitch_toe_handoff_candidate_left']} | "
            f"{row['forward_pitch_toe_handoff_candidate_right']} | "
            f"{row['toe_heel_bridge_contact_detected_left']} | "
            f"{row['toe_heel_bridge_contact_detected_right']} | "
            f"{row['left_right_symmetry']} | "
            f"{row['contact_model_sufficient_for_rollover_analysis']} |"
        )
    lines.extend(["", "## Recommendations", ""])
    for row in metrics:
        lines.append(f"- `{row['variant']}`: {row['recommendation_to_mechanical_team']}")
    lines.extend(
        [
            "",
            "## Safety Notes",
            "",
            "- `contact_model_sufficient_for_rollover_analysis` means only that the contact model may be useful for scripted contact diagnostics.",
            "- It is not walking success, not mechanical validation, and not PPO readiness.",
            "- Variants remain simulation prototypes until a mechanical owner validates geometry and contact behavior.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_VARIANT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--settle-steps", type=int, default=40)
    return parser.parse_args()


def main() -> int:
    """Run foot variant contact comparison."""

    args = parse_args()
    if args.settle_steps < 0:
        raise ValueError("--settle-steps must be non-negative")
    variants = load_variants(args.manifest)
    all_raw: list[dict[str, Any]] = []
    all_summary: list[dict[str, Any]] = []
    all_inventory: list[dict[str, Any]] = []
    for variant in variants:
        raw_rows, summary_rows, inventory_rows = run_variant(variant, args.settle_steps)
        all_raw.extend(raw_rows)
        all_summary.extend(summary_rows)
        all_inventory.extend(inventory_rows)
    metrics = build_metrics(all_summary, all_inventory, variants)

    write_csv(
        args.out_dir / "raw_contact_pairs.csv",
        all_raw,
        [
            "variant",
            "scenario",
            "contact_index",
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
        args.out_dir / "contact_patch_classification.csv",
        all_inventory,
        [
            "variant",
            "geom_id",
            "geom_name",
            "body_name",
            "side",
            "region",
            "is_added_patch",
            "source",
            "confidence",
            "valid_for",
            "contype",
            "conaffinity",
            "size",
        ],
    )
    write_csv(
        args.out_dir / "contact_scenario_summary.csv",
        all_summary,
        [
            "variant",
            "scenario",
            *FORCE_FIELDS,
            "left_contact_count",
            "right_contact_count",
            "unknown_contact_count",
        ],
    )
    write_csv(
        args.out_dir / "variant_metrics.csv",
        metrics,
        [
            "variant",
            "status",
            "can_classify_center_toe_heel",
            "neutral_center_first_left",
            "neutral_center_first_right",
            "forward_pitch_toe_handoff_candidate_left",
            "forward_pitch_toe_handoff_candidate_right",
            "toe_heel_bridge_contact_detected_left",
            "toe_heel_bridge_contact_detected_right",
            "left_right_symmetry_score",
            "left_right_symmetry",
            "contact_model_sufficient_for_rollover_analysis",
            "recommendation_to_mechanical_team",
            "walking_success_claimed",
            "source",
            "confidence",
            "valid_for",
        ],
    )
    (args.out_dir / "variant_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    write_report(args.report, metrics, args.manifest)
    print(f"variants={len(variants)}")
    print(f"metrics={args.out_dir / 'variant_metrics.json'}")
    print(f"report={args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
