"""Phase 1 controlled-falling / rollover diagnostic for Seedon.

This Class C diagnostic does not train PPO and does not modify scenes. It runs
a deterministic MuJoCo rollout and records whether Seedon can preserve forward
momentum, keep contact, roll from foot center to toe, and show left/right load
alternation before visible swing clearance is required.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from seedon_baseline.env import SeedonStandingEnv, load_seedon_config_from_env
from tools.seedon_debug_common import DEBUG_OUT_DIR, DEFAULT_SCENE_PATH, geom_name, require_scene


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V5A_SCENE = DEBUG_OUT_DIR / "blue_like_sole_experiments_v5" / "training_scene_v5_a.xml"
DEFAULT_OUT_DIR = DEBUG_OUT_DIR / "blue_like_phase1_rollover"
FOOT_FORCE_THRESHOLD_N = 0.1


@dataclass(frozen=True)
class RolloverSummary:
    """Aggregate Phase 1 diagnostic metrics.

    Args:
        steps: Number of simulated control steps.
        mean_forward_velocity: Mean base x velocity.
        forward_displacement: Final base x minus initial base x.
        max_base_pitch: Maximum absolute base pitch in radians.
        max_base_roll: Maximum absolute base roll in radians.
        min_upright: Minimum base local-z alignment with world z.
        contact_none_ratio: Fraction of steps with no foot/floor contact.
        jump_count: Count of contiguous no-contact or jump-indicator bursts.
        both_contact_ratio: Fraction of steps with both feet loaded/contacting.
        single_contact_ratio: Fraction of steps with one foot loaded/contacting.
        support_force_ratio_mean: Mean support-side force fraction.
        support_force_ratio_peak: Peak support-side force fraction.
        swing_force_ratio_min: Minimum swing-side force fraction.
        toe_handoff_detected: Whether center-to-toe handoff appeared.
        left_right_phase_switch_count: Count of left/right support-side switches.
        passed: Whether conservative Phase 1 criteria passed.
        fail_reasons: Comma-separated failed criteria.
        csv_path: Per-step timeline CSV path.
        json_path: Summary JSON path.
    """

    steps: int
    mean_forward_velocity: float
    forward_displacement: float
    max_base_pitch: float
    max_base_roll: float
    min_upright: float
    contact_none_ratio: float
    jump_count: int
    both_contact_ratio: float
    single_contact_ratio: float
    support_force_ratio_mean: float
    support_force_ratio_peak: float
    swing_force_ratio_min: float
    toe_handoff_detected: bool
    left_right_phase_switch_count: int
    passed: bool
    fail_reasons: str
    csv_path: str
    json_path: str


def _default_scene() -> Path:
    """Return the preferred v5_a scene when present, otherwise the standard scene."""

    return DEFAULT_V5A_SCENE if DEFAULT_V5A_SCENE.is_file() else DEFAULT_SCENE_PATH


def _is_foot_geom(name: str) -> bool:
    """Return whether a geom is part of the Seedon foot collision assembly."""

    return name.startswith("R_foot_collision") or name.startswith("L_foot_collision")


def _side_for_geom(name: str) -> str:
    """Return a stable side label for one Seedon foot geom name."""

    if name.startswith("R_"):
        return "right"
    if name.startswith("L_"):
        return "left"
    return "unknown"


def _region_for_geom(name: str, local_x: float | None) -> str:
    """Classify a foot/floor contact into center, toe, heel, or shoulder."""

    if "toe_rocker" in name:
        return "toe"
    if "heel_rocker" in name:
        return "heel"
    if "lateral_shoulder" in name:
        return "lateral_shoulder"
    if local_x is not None:
        if local_x > 0.055:
            return "toe"
        if local_x < -0.015:
            return "heel"
    return "center"


def _world_to_body_local(env: SeedonStandingEnv, body_id: int, world_pos: np.ndarray) -> np.ndarray:
    """Convert a world-space point into one body's local frame."""

    body_pos = env.data.xpos[body_id]
    body_xmat = env.data.xmat[body_id].reshape(3, 3)
    return body_xmat.T @ (world_pos - body_pos)


def _foot_contact_forces(env: SeedonStandingEnv) -> dict[str, float | bool | int]:
    """Return side/region contact forces for all Seedon foot geoms.

    Raises:
        ValueError: If MuJoCo contact force extraction fails due to a malformed
            contact index.
    """

    forces: dict[str, float | bool | int] = {
        "left_force": 0.0,
        "right_force": 0.0,
        "left_center_force": 0.0,
        "left_toe_force": 0.0,
        "left_heel_force": 0.0,
        "right_center_force": 0.0,
        "right_toe_force": 0.0,
        "right_heel_force": 0.0,
        "left_center_contact": False,
        "left_toe_contact": False,
        "right_center_contact": False,
        "right_toe_contact": False,
        "contact_count": 0,
    }
    wrench = np.zeros(6, dtype=np.float64)
    for contact_index in range(env.data.ncon):
        contact = env.data.contact[contact_index]
        name_a = geom_name(env.model, int(contact.geom1))
        name_b = geom_name(env.model, int(contact.geom2))
        if "floor" not in {name_a, name_b}:
            continue
        foot_name = name_b if name_a == "floor" else name_a
        if not _is_foot_geom(foot_name):
            continue
        foot_geom_id = int(contact.geom2 if name_a == "floor" else contact.geom1)
        foot_body_id = int(env.model.geom_bodyid[foot_geom_id])
        local_pos = _world_to_body_local(env, foot_body_id, np.asarray(contact.pos, dtype=np.float64))
        side = _side_for_geom(foot_name)
        region = _region_for_geom(foot_name, float(local_pos[0]))
        mujoco.mj_contactForce(env.model, env.data, contact_index, wrench)
        normal_force = abs(float(wrench[0]))
        if side not in {"left", "right"}:
            continue
        forces[f"{side}_force"] = float(forces[f"{side}_force"]) + normal_force
        if region in {"center", "toe", "heel"}:
            forces[f"{side}_{region}_force"] = float(forces[f"{side}_{region}_force"]) + normal_force
        if region == "center":
            forces[f"{side}_center_contact"] = True
        if region == "toe":
            forces[f"{side}_toe_contact"] = True
        forces["contact_count"] = int(forces["contact_count"]) + 1
    return forces


def _contact_state(left_force: float, right_force: float) -> str:
    """Return both/left/right/none from force thresholded contacts."""

    left = left_force > FOOT_FORCE_THRESHOLD_N
    right = right_force > FOOT_FORCE_THRESHOLD_N
    if left and right:
        return "both"
    if left:
        return "left"
    if right:
        return "right"
    return "none"


def _support_swing(left_force: float, right_force: float) -> tuple[str, str, float, float]:
    """Return support side, swing side, and force fractions."""

    total = max(left_force + right_force, 1e-9)
    if right_force >= left_force:
        return "right", "left", right_force / total, left_force / total
    return "left", "right", left_force / total, right_force / total


def _count_bursts(rows: list[dict[str, Any]], key: str) -> int:
    """Count contiguous truthy bursts for a boolean row key."""

    count = 0
    in_burst = False
    for row in rows:
        active = bool(row[key])
        if active and not in_burst:
            count += 1
            in_burst = True
        elif not active:
            in_burst = False
    return count


def _support_switches(rows: list[dict[str, Any]]) -> int:
    """Count side switches while ignoring no-contact rows."""

    switches = 0
    previous = ""
    for row in rows:
        side = str(row["support_side"])
        if side not in {"left", "right"}:
            continue
        if previous and side != previous:
            switches += 1
        previous = side
    return switches


def _toe_handoff_detected(rows: list[dict[str, Any]]) -> bool:
    """Return whether any side shows center contact followed by toe contact."""

    for side in ("left", "right"):
        saw_center = False
        for row in rows:
            if bool(row[f"{side}_center_contact"]) or float(row[f"{side}_center_force"]) > FOOT_FORCE_THRESHOLD_N:
                saw_center = True
            if saw_center and (
                bool(row[f"{side}_toe_contact"])
                or float(row[f"{side}_toe_force"]) > FOOT_FORCE_THRESHOLD_N
            ):
                return True
    return False


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write timeline rows as CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _summarize(rows: list[dict[str, Any]], csv_path: Path, json_path: Path) -> RolloverSummary:
    """Aggregate per-step rows into the conservative Phase 1 summary."""

    if not rows:
        return RolloverSummary(
            steps=0,
            mean_forward_velocity=0.0,
            forward_displacement=0.0,
            max_base_pitch=0.0,
            max_base_roll=0.0,
            min_upright=0.0,
            contact_none_ratio=1.0,
            jump_count=0,
            both_contact_ratio=0.0,
            single_contact_ratio=0.0,
            support_force_ratio_mean=0.0,
            support_force_ratio_peak=0.0,
            swing_force_ratio_min=0.0,
            toe_handoff_detected=False,
            left_right_phase_switch_count=0,
            passed=False,
            fail_reasons="empty_rollout",
            csv_path=str(csv_path),
            json_path=str(json_path),
        )

    states = Counter(str(row["contact_state"]) for row in rows)
    toe_handoff = _toe_handoff_detected(rows)
    phase_switches = _support_switches(rows)
    contact_none_ratio = states["none"] / len(rows)
    jump_count = _count_bursts(rows, "jump_indicator")
    support_ratios = [float(row["support_force_ratio"]) for row in rows]
    swing_ratios = [float(row["swing_force_ratio"]) for row in rows]
    forward_displacement = float(rows[-1]["base_x"]) - float(rows[0]["base_x"])
    fail_reasons: list[str] = []
    min_upright = min(float(row["upright"]) for row in rows)
    mean_forward_velocity = float(np.mean([float(row["base_vx"]) for row in rows]))
    if contact_none_ratio != 0.0:
        fail_reasons.append("contact_none")
    if jump_count != 0:
        fail_reasons.append("jump")
    if min_upright < 0.98:
        fail_reasons.append("upright")
    if forward_displacement <= 0.0:
        fail_reasons.append("forward_displacement")
    if mean_forward_velocity <= 0.0:
        fail_reasons.append("mean_forward_velocity")
    if not toe_handoff:
        fail_reasons.append("toe_handoff")
    if phase_switches < 1:
        fail_reasons.append("left_right_phase_switch")

    return RolloverSummary(
        steps=len(rows),
        mean_forward_velocity=mean_forward_velocity,
        forward_displacement=forward_displacement,
        max_base_pitch=max(abs(float(row["base_pitch"])) for row in rows),
        max_base_roll=max(abs(float(row["base_roll"])) for row in rows),
        min_upright=min_upright,
        contact_none_ratio=contact_none_ratio,
        jump_count=jump_count,
        both_contact_ratio=states["both"] / len(rows),
        single_contact_ratio=(states["left"] + states["right"]) / len(rows),
        support_force_ratio_mean=float(np.mean(support_ratios)),
        support_force_ratio_peak=max(support_ratios),
        swing_force_ratio_min=min(swing_ratios),
        toe_handoff_detected=toe_handoff,
        left_right_phase_switch_count=phase_switches,
        passed=not fail_reasons,
        fail_reasons=",".join(fail_reasons),
        csv_path=str(csv_path),
        json_path=str(json_path),
    )


def run_diagnostic(
    *,
    scene_path: Path,
    csv_path: Path,
    json_path: Path,
    steps: int,
    seed: int,
    initial_forward_velocity: float,
) -> RolloverSummary:
    """Run the deterministic Phase 1 diagnostic and persist CSV/JSON outputs.

    Args:
        scene_path: MuJoCo XML scene to load.
        csv_path: Destination timeline CSV.
        json_path: Destination summary JSON.
        steps: Number of environment control steps.
        seed: Deterministic environment seed.
        initial_forward_velocity: Initial base x velocity used to probe rollover.

    Returns:
        Aggregated Phase 1 summary.

    Raises:
        FileNotFoundError: If the scene path does not exist.
        ValueError: If steps is not positive.
    """

    if steps <= 0:
        raise ValueError("steps must be positive.")
    scene = require_scene(scene_path)
    env = SeedonStandingEnv(
        scene_path=scene,
        reset_noise_scale=0.0,
        reward_config=load_seedon_config_from_env(),
    )
    rows: list[dict[str, Any]] = []
    try:
        env.reset(seed=seed)
        env.data.qvel[0] = float(initial_forward_velocity)
        mujoco.mj_forward(env.model, env.data)
        initial_x = float(env.data.qpos[0])
        initial_y = float(env.data.qpos[1])
        initial_yaw = float(env.data.qpos[6])
        robot_weight = float(np.sum(env.model.body_mass) * 9.81)
        previous_total_force = 0.0
        action = np.zeros(env.action_space.shape, dtype=np.float64)
        for step in range(1, steps + 1):
            _, _, terminated, truncated, info = env.step(action)
            forces = _foot_contact_forces(env)
            left_force = float(forces["left_force"])
            right_force = float(forces["right_force"])
            support_side, swing_side, support_ratio, swing_ratio = _support_swing(left_force, right_force)
            total_force = left_force + right_force
            contact_state = _contact_state(left_force, right_force)
            contact_none = contact_state == "none"
            jump_indicator = bool(contact_none or (float(env.data.qvel[2]) > 0.05 and total_force < 0.10 * robot_weight))
            row = {
                "time": float(env.data.time),
                "step": step,
                "base_x": float(env.data.qpos[0]),
                "base_y": float(env.data.qpos[1]),
                "base_z": float(env._base_height()),
                "base_vx": float(env.data.qvel[0]),
                "base_pitch": float(env._base_pitch()),
                "base_roll": float(env._base_roll()),
                "upright": float(env._base_upright()),
                "left_foot_force": left_force,
                "right_foot_force": right_force,
                "support_side": support_side,
                "swing_side": swing_side,
                "support_force_ratio": support_ratio,
                "swing_force_ratio": swing_ratio,
                "left_center_contact": bool(forces["left_center_contact"]),
                "left_toe_contact": bool(forces["left_toe_contact"]),
                "right_center_contact": bool(forces["right_center_contact"]),
                "right_toe_contact": bool(forces["right_toe_contact"]),
                "left_center_force": float(forces["left_center_force"]),
                "left_toe_force": float(forces["left_toe_force"]),
                "right_center_force": float(forces["right_center_force"]),
                "right_toe_force": float(forces["right_toe_force"]),
                "contact_state": contact_state,
                "contact_none": contact_none,
                "jump_indicator": jump_indicator,
                "forward_displacement": float(env.data.qpos[0]) - initial_x,
                "lateral_drift": float(env.data.qpos[1]) - initial_y,
                "yaw_drift": float(env.data.qpos[6]) - initial_yaw,
                "touchdown_impact_proxy": max(0.0, total_force - previous_total_force) / max(robot_weight, 1e-9),
                "total_force_normalized": total_force / max(robot_weight, 1e-9),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "termination_reason": str(info.get("termination_reason", "none")),
            }
            rows.append(row)
            previous_total_force = total_force
            if terminated or truncated:
                break
    finally:
        env.close()

    _write_csv(csv_path, rows)
    summary = _summarize(rows, csv_path, json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(asdict(summary), indent=2) + "\n", encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-path", type=Path, default=_default_scene())
    parser.add_argument("--csv", type=Path, default=DEFAULT_OUT_DIR / "phase1_rollover_timeline.csv")
    parser.add_argument("--json-summary", type=Path, default=DEFAULT_OUT_DIR / "phase1_rollover_summary.json")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--initial-forward-velocity", type=float, default=0.12)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the Phase 1 diagnostic from the command line."""

    args = build_parser().parse_args(argv)
    summary = run_diagnostic(
        scene_path=args.scene_path,
        csv_path=args.csv,
        json_path=args.json_summary,
        steps=args.steps,
        seed=args.seed,
        initial_forward_velocity=args.initial_forward_velocity,
    )
    print(
        "phase1_pass={passed} forward={forward:.4f} mean_vx={vx:.4f} "
        "toe_handoff={toe} switches={switches} none={none:.4f} jump={jump} reasons={reasons}".format(
            passed=summary.passed,
            forward=summary.forward_displacement,
            vx=summary.mean_forward_velocity,
            toe=summary.toe_handoff_detected,
            switches=summary.left_right_phase_switch_count,
            none=summary.contact_none_ratio,
            jump=summary.jump_count,
            reasons=summary.fail_reasons or "-",
        )
    )
    print(f"csv={summary.csv_path}")
    print(f"json={summary.json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
