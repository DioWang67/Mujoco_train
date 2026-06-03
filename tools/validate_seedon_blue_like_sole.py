"""Minimal validation for Seedon Blue-like rounded sole variants.

This Class C diagnostic runs short scripted checks only. It does not train,
does not change rewards, and does not modify robot morphology.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from seedon_baseline.env import SeedonStandingEnv, load_seedon_config_from_env
from tools.seedon_debug_common import DEBUG_OUT_DIR, geom_name, require_scene


OUT_DIR = DEBUG_OUT_DIR / "blue_like_sole_experiments"
MANIFEST_PATH = OUT_DIR / "manifest.json"
REPORT_PATH = OUT_DIR / "validation_report.md"
FLOOR_GEOM = "floor"


@dataclass(frozen=True)
class VariantInput:
    """One scene variant to validate."""

    name: str
    scene_path: Path


def _load_variants(manifest_path: Path) -> list[VariantInput]:
    """Load blue-like variant scenes from the generated manifest."""
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    variants: list[VariantInput] = []
    for item in payload:
        scene_path = require_scene(Path(item["generated_scene_path"]))
        variant = item.get("variant", {})
        variants.append(
            VariantInput(
                name=str(variant.get("name", scene_path.stem)),
                scene_path=scene_path,
            )
        )
    return variants


def _is_foot_geom(name: str) -> bool:
    """Return whether a geom belongs to a Seedon foot collision assembly."""
    return name.startswith("R_foot_collision") or name.startswith("L_foot_collision")


def _side_for_geom(name: str) -> str:
    """Return left/right label for a Seedon foot geom name."""
    if name.startswith("R_"):
        return "right"
    if name.startswith("L_"):
        return "left"
    return "unknown"


def _region_for_geom(name: str, contact_x_local: float | None = None) -> str:
    """Classify a foot contact into center/toe/heel/shoulder."""
    if "toe_rocker" in name:
        return "toe"
    if "heel_rocker" in name:
        return "heel"
    if "lateral_shoulder" in name:
        return "lateral_shoulder"
    if contact_x_local is not None:
        if contact_x_local > 0.055:
            return "toe"
        if contact_x_local < -0.015:
            return "heel"
    return "center"


def _foot_contacts(env: SeedonStandingEnv) -> list[dict[str, Any]]:
    """Return all floor/foot contact records with world and local x positions."""
    rows: list[dict[str, Any]] = []
    for contact_index in range(env.data.ncon):
        contact = env.data.contact[contact_index]
        name_a = geom_name(env.model, int(contact.geom1))
        name_b = geom_name(env.model, int(contact.geom2))
        if FLOOR_GEOM not in {name_a, name_b}:
            continue
        foot_name = name_b if name_a == FLOOR_GEOM else name_a
        if not _is_foot_geom(foot_name):
            continue
        foot_geom_id = int(contact.geom2 if name_a == FLOOR_GEOM else contact.geom1)
        foot_body_id = int(env.model.geom_bodyid[foot_geom_id])
        world_pos = np.asarray(contact.pos, dtype=np.float64)
        local_pos = _world_to_body_local(env, foot_body_id, world_pos)
        rows.append(
            {
                "geom": foot_name,
                "side": _side_for_geom(foot_name),
                "region": _region_for_geom(foot_name, float(local_pos[0])),
                "is_center_geom": foot_name in {"R_foot_collision", "L_foot_collision"},
                "is_toe_rocker_geom": "toe_rocker" in foot_name,
                "is_heel_rocker_geom": "heel_rocker" in foot_name,
                "world_x": float(world_pos[0]),
                "world_y": float(world_pos[1]),
                "world_z": float(world_pos[2]),
                "local_x": float(local_pos[0]),
            }
        )
    return rows


def _world_to_body_local(
    env: SeedonStandingEnv,
    body_id: int,
    world_pos: np.ndarray,
) -> np.ndarray:
    """Convert a world position to one body's local coordinates."""
    body_pos = env.data.xpos[body_id]
    body_xmat = env.data.xmat[body_id].reshape(3, 3)
    return body_xmat.T @ (world_pos - body_pos)


def _contact_state(contacts: list[dict[str, Any]]) -> str:
    """Return compact left/right foot contact state using all foot geoms."""
    sides = {str(row["side"]) for row in contacts}
    if "left" in sides and "right" in sides:
        return "both"
    if "left" in sides:
        return "left_only"
    if "right" in sides:
        return "right_only"
    return "none"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write flat diagnostic rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _base_pitch(env: SeedonStandingEnv) -> float:
    """Return base pitch angle in radians from the free-joint quaternion."""
    w, x, y, z = [float(value) for value in env.data.qpos[3:7]]
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        return float(math.copysign(math.pi / 2.0, sinp))
    return float(math.asin(sinp))


def run_zero_action(variant: VariantInput, out_dir: Path, steps: int) -> dict[str, Any]:
    """Run zero-action rollout and save per-step stability/contact rows."""
    env = SeedonStandingEnv(
        scene_path=variant.scene_path,
        reset_noise_scale=0.0,
        reward_config=load_seedon_config_from_env(),
    )
    rows: list[dict[str, Any]] = []
    try:
        env.reset(seed=42)
        action = np.zeros(env.action_space.shape, dtype=np.float64)
        for step in range(1, steps + 1):
            _, _, terminated, truncated, _ = env.step(action)
            contacts = _foot_contacts(env)
            regions = Counter(str(row["region"]) for row in contacts)
            center_geom_contact = any(bool(row["is_center_geom"]) for row in contacts)
            toe_rocker_contact = any(bool(row["is_toe_rocker_geom"]) for row in contacts)
            heel_rocker_contact = any(bool(row["is_heel_rocker_geom"]) for row in contacts)
            rows.append(
                {
                    "step": step,
                    "base_z": float(env._base_height()),
                    "upright": float(env._base_upright()),
                    "base_x": float(env.data.qpos[0]),
                    "base_pitch": _base_pitch(env),
                    "contact_state": _contact_state(contacts),
                    "contact_count": len(contacts),
                    "center_contacts": regions["center"],
                    "toe_contacts": regions["toe"],
                    "heel_contacts": regions["heel"],
                    "center_geom_contact": center_geom_contact,
                    "toe_rocker_geom_contact": toe_rocker_contact,
                    "heel_rocker_geom_contact": heel_rocker_contact,
                    "lateral_shoulder_contacts": regions["lateral_shoulder"],
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }
            )
            if terminated or truncated:
                break
    finally:
        env.close()

    _write_csv(out_dir / variant.name / "zero_action_blue_like.csv", rows)
    states = Counter(str(row["contact_state"]) for row in rows)
    return {
        "steps": len(rows),
        "terminated": bool(rows[-1]["terminated"]) if rows else False,
        "truncated": bool(rows[-1]["truncated"]) if rows else False,
        "min_base_z": min((float(row["base_z"]) for row in rows), default=float("nan")),
        "min_upright": min((float(row["upright"]) for row in rows), default=float("nan")),
        "none_steps": states["none"],
        "both_steps": states["both"],
        "toe_steps": sum(int(row["toe_contacts"]) > 0 for row in rows),
        "heel_steps": sum(int(row["heel_contacts"]) > 0 for row in rows),
        "center_steps": sum(int(row["center_contacts"]) > 0 for row in rows),
        "center_geom_steps": sum(bool(row["center_geom_contact"]) for row in rows),
        "toe_rocker_geom_steps": sum(bool(row["toe_rocker_geom_contact"]) for row in rows),
        "heel_rocker_geom_steps": sum(bool(row["heel_rocker_geom_contact"]) for row in rows),
    }


def run_rocker_probe(variant: VariantInput, out_dir: Path) -> dict[str, Any]:
    """Probe whether toe/heel rocker geoms participate under scripted pitch motion."""
    env = SeedonStandingEnv(
        scene_path=variant.scene_path,
        reset_noise_scale=0.0,
        reward_config=load_seedon_config_from_env(),
    )
    rows: list[dict[str, Any]] = []
    try:
        env.reset(seed=7)
        nominal = env._nominal_joint_qpos.copy()
        for step in range(1, 121):
            target = nominal.copy()
            if step <= 40:
                target[2] += 0.04 * step / 40.0
                target[7] += 0.04 * step / 40.0
            elif step <= 80:
                target[2] += 0.04 * (1.0 - (step - 40) / 40.0)
                target[7] += 0.04 * (1.0 - (step - 40) / 40.0)
            else:
                target[2] -= 0.04 * (step - 80) / 40.0
                target[7] -= 0.04 * (step - 80) / 40.0
            env._do_pd_simulation(env._apply_safe_joint_target_clamps(target))
            contacts = _foot_contacts(env)
            regions = Counter(str(row["region"]) for row in contacts)
            center_geom_contact = any(bool(row["is_center_geom"]) for row in contacts)
            toe_rocker_contact = any(bool(row["is_toe_rocker_geom"]) for row in contacts)
            heel_rocker_contact = any(bool(row["is_heel_rocker_geom"]) for row in contacts)
            xs = [float(row["local_x"]) for row in contacts]
            rows.append(
                {
                    "step": step,
                    "base_z": float(env._base_height()),
                    "upright": float(env._base_upright()),
                    "base_pitch": _base_pitch(env),
                    "contact_state": _contact_state(contacts),
                    "contact_count": len(contacts),
                    "min_contact_local_x": min(xs) if xs else "",
                    "max_contact_local_x": max(xs) if xs else "",
                    "center_contacts": regions["center"],
                    "toe_contacts": regions["toe"],
                    "heel_contacts": regions["heel"],
                    "center_geom_contact": center_geom_contact,
                    "toe_rocker_geom_contact": toe_rocker_contact,
                    "heel_rocker_geom_contact": heel_rocker_contact,
                    "lateral_shoulder_contacts": regions["lateral_shoulder"],
                }
            )
    finally:
        env.close()

    _write_csv(out_dir / variant.name / "rocker_contact_probe.csv", rows)
    return {
        "toe_steps": sum(int(row["toe_contacts"]) > 0 for row in rows),
        "heel_steps": sum(int(row["heel_contacts"]) > 0 for row in rows),
        "center_steps": sum(int(row["center_contacts"]) > 0 for row in rows),
        "center_geom_steps": sum(bool(row["center_geom_contact"]) for row in rows),
        "toe_rocker_geom_steps": sum(bool(row["toe_rocker_geom_contact"]) for row in rows),
        "heel_rocker_geom_steps": sum(bool(row["heel_rocker_geom_contact"]) for row in rows),
        "min_base_z": min((float(row["base_z"]) for row in rows), default=float("nan")),
        "min_upright": min((float(row["upright"]) for row in rows), default=float("nan")),
        "none_steps": sum(row["contact_state"] == "none" for row in rows),
    }


def run_forward_fall_catch_preview(variant: VariantInput, out_dir: Path) -> dict[str, Any]:
    """Run a minimal scripted forward-roll and swing-foot-catch preview."""
    env = SeedonStandingEnv(
        scene_path=variant.scene_path,
        reset_noise_scale=0.0,
        reward_config=load_seedon_config_from_env(),
    )
    rows: list[dict[str, Any]] = []
    catch_step: int | None = None
    right_was_airborne = False
    right_had_clearance = False
    try:
        env.reset(seed=11)
        qpos = env.data.qpos.copy()
        qvel = env.data.qvel.copy()
        qvel[0] = 0.08
        env.set_state(qpos, qvel)
        mujoco.mj_forward(env.model, env.data)
        nominal = env._nominal_joint_qpos.copy()
        initial_x = float(env.data.qpos[0])
        initial_right_x = _mean_side_contact_x(_foot_contacts(env), "right")

        for step in range(1, 141):
            progress = min(1.0, step / 70.0)
            lower = max(0.0, (step - 70) / 70.0)
            swing = progress if step <= 70 else 1.0 - lower
            target = nominal.copy()
            target[1] += -0.035 * swing
            target[6] += 0.035 * swing
            target[2] += 0.18 * swing
            target[3] += -0.20 * swing
            target[4] += -0.10 * swing
            target[7] += 0.025 * swing
            env._do_pd_simulation(env._apply_safe_joint_target_clamps(target))
            contacts = _foot_contacts(env)
            right_contact = any(row["side"] == "right" for row in contacts)
            left_contact = any(row["side"] == "left" for row in contacts)
            foot_bottoms = env._foot_bottom_heights()
            right_clearance_over_left = float(foot_bottoms[0] - foot_bottoms[1])
            if right_clearance_over_left > 0.005:
                right_had_clearance = True
            if right_had_clearance and not right_contact:
                right_was_airborne = True
            if right_was_airborne and right_contact and catch_step is None and step > 40:
                catch_step = step
            regions = Counter(str(row["region"]) for row in contacts)
            center_geom_contact = any(bool(row["is_center_geom"]) for row in contacts)
            toe_rocker_contact = any(bool(row["is_toe_rocker_geom"]) for row in contacts)
            heel_rocker_contact = any(bool(row["is_heel_rocker_geom"]) for row in contacts)
            right_contact_x = _mean_side_contact_x(contacts, "right")
            rows.append(
                {
                    "step": step,
                    "base_x": float(env.data.qpos[0]),
                    "base_x_delta": float(env.data.qpos[0] - initial_x),
                    "base_z": float(env._base_height()),
                    "upright": float(env._base_upright()),
                    "base_pitch": _base_pitch(env),
                    "forward_velocity": float(env.data.qvel[0]),
                    "contact_state": _contact_state(contacts),
                    "right_contact": right_contact,
                    "left_contact": left_contact,
                    "right_foot_bottom_z": float(foot_bottoms[0]),
                    "left_foot_bottom_z": float(foot_bottoms[1]),
                    "right_clearance_over_left": right_clearance_over_left,
                    "catch_step": catch_step if catch_step is not None else "",
                    "right_contact_x_delta": (
                        right_contact_x - initial_right_x
                        if right_contact_x is not None and initial_right_x is not None
                        else ""
                    ),
                    "center_contacts": regions["center"],
                    "toe_contacts": regions["toe"],
                    "heel_contacts": regions["heel"],
                    "center_geom_contact": center_geom_contact,
                    "toe_rocker_geom_contact": toe_rocker_contact,
                    "heel_rocker_geom_contact": heel_rocker_contact,
                    "lateral_shoulder_contacts": regions["lateral_shoulder"],
                }
            )
    finally:
        env.close()

    _write_csv(out_dir / variant.name / "forward_fall_catch_preview.csv", rows)
    return {
        "steps": len(rows),
        "base_x_delta": float(rows[-1]["base_x_delta"]) if rows else float("nan"),
        "min_base_z": min((float(row["base_z"]) for row in rows), default=float("nan")),
        "min_upright": min((float(row["upright"]) for row in rows), default=float("nan")),
        "none_steps": sum(row["contact_state"] == "none" for row in rows),
        "toe_steps": sum(int(row["toe_contacts"]) > 0 for row in rows),
        "heel_steps": sum(int(row["heel_contacts"]) > 0 for row in rows),
        "center_geom_steps": sum(bool(row["center_geom_contact"]) for row in rows),
        "toe_rocker_geom_steps": sum(bool(row["toe_rocker_geom_contact"]) for row in rows),
        "heel_rocker_geom_steps": sum(bool(row["heel_rocker_geom_contact"]) for row in rows),
        "catch_step": catch_step,
        "right_was_airborne": right_was_airborne,
        "right_had_clearance": right_had_clearance,
        "max_right_clearance_over_left": max(
            (float(row["right_clearance_over_left"]) for row in rows),
            default=float("nan"),
        ),
        "final_right_contact_x_delta": _last_numeric(rows, "right_contact_x_delta"),
    }


def _mean_side_contact_x(contacts: list[dict[str, Any]], side: str) -> float | None:
    """Return mean world x for all contacts on one side."""
    xs = [float(row["world_x"]) for row in contacts if row["side"] == side]
    if not xs:
        return None
    return float(np.mean(xs))


def _last_numeric(rows: list[dict[str, Any]], key: str) -> float | None:
    """Return last numeric value in a row sequence."""
    for row in reversed(rows):
        value = row.get(key)
        if value == "" or value is None:
            continue
        return float(value)
    return None


def write_report(summaries: dict[str, dict[str, Any]], report_path: Path) -> None:
    """Write the validation report markdown."""
    lines = [
        "# Blue-Like Sole Minimal Validation",
        "",
        "Task class: Class C experiment diagnostic. No PPO, reward change, train.py edit, or morphology generation was performed in this validation step.",
        "",
        "## Zero-Action Stability",
        "",
        "| variant | steps | min base_z | min upright | both steps | none steps | center geom steps | toe rocker steps | heel rocker steps |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    interpretation_lines: list[str] = []
    for name, summary in summaries.items():
        zero = summary["zero_action"]
        lines.append(
            f"| {name} | {zero['steps']} | {zero['min_base_z']:.4f} | {zero['min_upright']:.4f} | "
            f"{zero['both_steps']} | {zero['none_steps']} | {zero['center_geom_steps']} | "
            f"{zero['toe_rocker_geom_steps']} | {zero['heel_rocker_geom_steps']} |"
        )
    lines.extend(
        [
            "",
            "## Rocker Contact Probe",
            "",
            "| variant | toe rocker steps | heel rocker steps | center geom steps | none steps | min base_z | min upright |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, summary in summaries.items():
        rocker = summary["rocker_probe"]
        lines.append(
            f"| {name} | {rocker['toe_rocker_geom_steps']} | {rocker['heel_rocker_geom_steps']} | "
            f"{rocker['center_geom_steps']} | {rocker['none_steps']} | "
            f"{rocker['min_base_z']:.4f} | {rocker['min_upright']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Forward-Fall-Catch Preview",
            "",
            "| variant | base_x delta | min base_z | min upright | max right clearance | none steps | toe rocker steps | heel rocker steps | right airborne | catch step | final right contact x delta |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, summary in summaries.items():
        preview = summary["forward_preview"]
        lines.append(
            f"| {name} | {preview['base_x_delta']:.4f} | {preview['min_base_z']:.4f} | "
            f"{preview['min_upright']:.4f} | {preview['max_right_clearance_over_left']:.4f} | "
            f"{preview['none_steps']} | "
            f"{preview['toe_rocker_geom_steps']} | {preview['heel_rocker_geom_steps']} | "
            f"{preview['right_was_airborne']} | {_fmt_optional(preview['catch_step'])} | "
            f"{_fmt_optional(preview['final_right_contact_x_delta'])} |"
        )
    lines.extend(["", "## Interpretation", ""])
    lines.append(
        "This validation uses dynamic preconditions: contact continuity, rocker participation, forward roll, and catch. It does not use static single-support margin or force-ratio unload as the main criterion."
    )
    lines.extend(
        [
            "",
            "## Contact Ordering Review",
            "",
            "Previous v1/v2 reference: zero-action center contact was 0 steps, toe/heel dominated all 80 steps, and forward preview had roughly 122-124 no-contact steps with min base_z near 0.14.",
            "",
            "| variant | center improved | toe/heel still dominate zero-action | toe participates in lean | preview improved vs v1/v2 | contact-ordering fit |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    best_name: str | None = None
    best_score = -1.0
    for name, summary in summaries.items():
        zero = summary["zero_action"]
        rocker = summary["rocker_probe"]
        preview = summary["forward_preview"]
        stable = zero["steps"] == 80 and zero["none_steps"] == 0 and zero["min_upright"] > 0.9
        rocker_active = (
            preview["toe_rocker_geom_steps"] > 0
            or preview["heel_rocker_geom_steps"] > 0
        )
        catch_ok = (
            bool(preview["right_was_airborne"])
            and preview["catch_step"] is not None
            and preview["none_steps"] <= 10
            and preview["min_upright"] > 0.75
            and preview["min_base_z"] > 0.34
        )
        center_improved = zero["center_geom_steps"] > 0
        toe_heel_dominate = (
            zero["toe_rocker_geom_steps"] == zero["steps"]
            and zero["heel_rocker_geom_steps"] == zero["steps"]
        )
        preview_improved = (
            preview["none_steps"] < 122
            or preview["min_base_z"] > 0.14
            or preview["min_upright"] > 0.005
        )
        ordering_score = 0.0
        ordering_score += 2.0 if stable else 0.0
        ordering_score += 2.0 if center_improved else 0.0
        ordering_score += 1.0 if not toe_heel_dominate else 0.0
        ordering_score += 1.0 if rocker_active else 0.0
        ordering_score += 1.0 if preview_improved else 0.0
        ordering_score -= max(0.0, float(preview["none_steps"]) - 122.0) / 50.0
        if ordering_score > best_score:
            best_score = ordering_score
            best_name = name
        fit = "fail"
        if center_improved and rocker_active and preview_improved:
            fit = "partial"
        if center_improved and not toe_heel_dominate and catch_ok:
            fit = "pass"
        lines.append(
            f"| {name} | {center_improved} | {toe_heel_dominate} | {rocker_active} | "
            f"{preview_improved} | {fit} |"
        )
        interpretation_lines.append(
            f"- `{name}`: zero_stable={stable}, rocker_active={rocker_active}, "
            f"forward_catch_candidate={catch_ok}."
        )
    lines.append("")
    lines.extend(interpretation_lines)
    lines.append("")
    if best_name is None:
        lines.append("No variant produced a useful contact-ordering improvement.")
    else:
        lines.append(
            f"Closest current contact ordering: `{best_name}`. Treat this as a geometry precheck only; it is not a gait success criterion."
        )
    lines.append(
        "If rocker_active is false, the generated helper geoms are not doing useful Blue-like work. If catch is false but zero-action is stable, the next step should be a better scripted catch preview, not PPO."
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt_optional(value: Any) -> str:
    """Format optional scalar for markdown."""
    if value is None:
        return "n/a"
    return str(value)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--report-path", type=Path, default=REPORT_PATH)
    parser.add_argument("--zero-steps", type=int, default=80)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run minimal validation and write markdown report."""
    args = build_parser().parse_args(argv)
    variants = _load_variants(args.manifest)
    summaries: dict[str, dict[str, Any]] = {}
    for variant in variants:
        summaries[variant.name] = {
            "zero_action": run_zero_action(variant, args.out_dir, args.zero_steps),
            "rocker_probe": run_rocker_probe(variant, args.out_dir),
            "forward_preview": run_forward_fall_catch_preview(variant, args.out_dir),
        }
    write_report(summaries, args.report_path)
    print(f"validated: {len(summaries)}")
    print(f"report: {args.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
