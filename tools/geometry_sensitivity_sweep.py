"""Sweep temporary Sedon geometry variants for lateral preload sensitivity."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np

from sedon_baseline.env import JOINT_NAMES, SedonStandingEnv, load_sedon_config_from_env
from tools.sedon_debug_common import (
    DEBUG_OUT_DIR,
    DEFAULT_SCENE_PATH,
    FLOOR_GEOM,
    LEFT_FOOT_GEOM,
    RIGHT_FOOT_GEOM,
    geom_id,
    require_scene,
)


DEFAULT_OUT_CSV = DEBUG_OUT_DIR / "geometry_sensitivity_sweep.csv"
HIP_ROLL_INDEX = {"right": JOINT_NAMES.index("R_joint_hip_roll"), "left": JOINT_NAMES.index("L_joint_hip_roll")}
HIP_YAW_BODY = {"right": "R_link_hip_yaw", "left": "L_link_hip_yaw"}
ASSIST_FORCE_N = {"none": 0.0, "small": 5.0, "medium": 10.0}


@dataclass(frozen=True)
class GeometryVariant:
    """One temporary geometry/control-assist variant."""

    foot_width_scale: float
    stance_width_scale: float
    base_com_height_offset: float
    lateral_assist: str


@dataclass(frozen=True)
class GeometrySensitivityRow:
    """One geometry sensitivity result."""

    foot_width_scale: float
    stance_width_scale: float
    base_com_height_offset: float
    lateral_assist: str
    peak_target_ratio: float
    min_swing_ratio: float
    total_force_normalized: float
    penetration: float
    upright: float
    base_height_drop: float
    diagnosis: str
    score: float


def _smoothstep(phase: float) -> float:
    """Return smoothstep preload alpha."""
    phase = float(np.clip(phase, 0.0, 1.0))
    return phase * phase * (3.0 - 2.0 * phase)


def _apply_foot_width_scale(env: SedonStandingEnv, scale: float) -> None:
    """Scale lateral foot collision half-width in memory."""
    for name in (LEFT_FOOT_GEOM, RIGHT_FOOT_GEOM):
        foot_id = geom_id(env.model, name)
        env.model.geom_size[foot_id][1] *= scale
        env.model.geom_rbound[foot_id] = float(np.linalg.norm(env.model.geom_size[foot_id]))


def _apply_stance_width_scale(env: SedonStandingEnv, scale: float) -> None:
    """Scale hip-yaw body lateral offsets in memory."""
    for side, body_name in HIP_YAW_BODY.items():
        body_id = env._body_id(body_name)
        direction = -1.0 if side == "right" else 1.0
        env.model.body_pos[body_id][1] = direction * abs(float(env.model.body_pos[body_id][1])) * scale


def _apply_base_com_height_offset(env: SedonStandingEnv, offset: float) -> None:
    """Shift base inertial COM z in memory."""
    base_id = env._body_id("base_link")
    env.model.body_ipos[base_id][2] += offset


def _build_target(env: SedonStandingEnv, alpha: float) -> np.ndarray:
    """Use the current dynamic_preload best candidate as a right-support target."""
    target = env._nominal_joint_qpos.copy()
    target[HIP_ROLL_INDEX["right"]] += 0.005 * alpha
    target[HIP_ROLL_INDEX["left"]] -= (0.020 + 0.020) * alpha
    return target


def _foot_forces(env: SedonStandingEnv) -> tuple[float, float, float]:
    """Return left/right world-z forces and max foot-floor penetration."""
    left = 0.0
    right = 0.0
    max_penetration = 0.0
    wrench = np.zeros(6, dtype=np.float64)
    for contact_index in range(env.data.ncon):
        contact = env.data.contact[contact_index]
        geom1 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1))
        geom2 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2))
        pair = {geom1, geom2}
        if pair not in ({FLOOR_GEOM, LEFT_FOOT_GEOM}, {FLOOR_GEOM, RIGHT_FOOT_GEOM}):
            continue
        mujoco.mj_contactForce(env.model, env.data, contact_index, wrench)
        contact_frame = np.asarray(contact.frame, dtype=np.float64).reshape(3, 3)
        world_force = contact_frame.T @ wrench[:3]
        force_z = abs(float(world_force[2]))
        max_penetration = max(max_penetration, max(0.0, -float(contact.dist)))
        if pair == {FLOOR_GEOM, LEFT_FOOT_GEOM}:
            left += force_z
        else:
            right += force_z
    return left, right, max_penetration


def _step_pd_with_lateral_assist(env: SedonStandingEnv, target: np.ndarray, assist_name: str) -> None:
    """Step PD control while optionally applying a virtual lateral pelvis force."""
    assist_force = ASSIST_FORCE_N[assist_name]
    base_id = env._base_body_id
    for _ in range(env.frame_skip):
        env.data.ctrl[:] = env._pd_control(target)
        if assist_force:
            env.data.xfrc_applied[base_id, 1] = -assist_force
        mujoco.mj_step(env.model, env.data)
        env.data.xfrc_applied[base_id, :] = 0.0


def _evaluate_variant(scene_path: Path, variant: GeometryVariant) -> GeometrySensitivityRow:
    """Evaluate one temporary geometry variant."""
    env = SedonStandingEnv(
        scene_path=require_scene(scene_path),
        reset_noise_scale=0.0,
        reward_config=load_sedon_config_from_env(),
    )
    _apply_foot_width_scale(env, variant.foot_width_scale)
    _apply_stance_width_scale(env, variant.stance_width_scale)
    _apply_base_com_height_offset(env, variant.base_com_height_offset)
    env.reset(seed=0)
    initial_base_height = float(env.data.xpos[env._base_body_id][2])
    steps = max(1, int(round(0.2 / float(env.model.opt.timestep))))
    peak_target_ratio = 0.0
    min_swing_ratio = 1.0
    max_total_force_normalized = 0.0
    max_penetration = 0.0
    min_upright = 1.0
    min_base_height = initial_base_height
    total_weight = float(np.sum(env.model.body_mass) * 9.81)
    for step_index in range(steps):
        alpha = _smoothstep(step_index / max(1, steps - 1))
        _step_pd_with_lateral_assist(env, _build_target(env, alpha), variant.lateral_assist)
        left_force, right_force, penetration = _foot_forces(env)
        total_force = left_force + right_force
        right_ratio = float(right_force / (total_force + 1e-6))
        left_ratio = float(left_force / (total_force + 1e-6))
        peak_target_ratio = max(peak_target_ratio, right_ratio)
        min_swing_ratio = min(min_swing_ratio, left_ratio)
        max_total_force_normalized = max(max_total_force_normalized, total_force / max(total_weight, 1e-6))
        max_penetration = max(max_penetration, penetration)
        min_upright = min(min_upright, float(env._base_upright()))
        min_base_height = min(min_base_height, float(env.data.xpos[env._base_body_id][2]))
    base_height_drop = max(0.0, initial_base_height - min_base_height)
    diagnosis = _diagnose(variant, peak_target_ratio)
    score = max(0.0, 0.58 - peak_target_ratio)
    score += max(0.0, max_penetration - 0.0015) * 100.0
    score += max(0.0, 0.98 - min_upright) * 5.0
    return GeometrySensitivityRow(
        foot_width_scale=variant.foot_width_scale,
        stance_width_scale=variant.stance_width_scale,
        base_com_height_offset=variant.base_com_height_offset,
        lateral_assist=variant.lateral_assist,
        peak_target_ratio=float(peak_target_ratio),
        min_swing_ratio=float(min_swing_ratio),
        total_force_normalized=float(max_total_force_normalized),
        penetration=float(max_penetration),
        upright=float(min_upright),
        base_height_drop=float(base_height_drop),
        diagnosis=diagnosis,
        score=float(score),
    )


def _diagnose(variant: GeometryVariant, peak_ratio: float) -> str:
    """Classify what kind of modification enabled the target ratio."""
    if peak_ratio < 0.58:
        return "not_reached"
    if variant.lateral_assist != "none":
        return "controller_or_actuator_authority"
    if (
        variant.foot_width_scale > 1.0
        or variant.stance_width_scale != 1.0
        or variant.base_com_height_offset < 0.0
    ):
        return "geometry_limited"
    return "baseline_reaches_target"


def _variants() -> list[GeometryVariant]:
    """Return requested geometry sensitivity grid."""
    return [
        GeometryVariant(foot_scale, stance_scale, com_offset, assist)
        for foot_scale in (1.0, 1.2, 1.4)
        for stance_scale in (0.9, 1.0, 1.1, 1.2)
        for com_offset in (0.0, -0.01, -0.02, -0.03)
        for assist in ("none", "small", "medium")
    ]


def _write_csv(path: Path, rows: list[GeometrySensitivityRow]) -> None:
    """Write geometry sensitivity CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def run_sweep(scene_path: Path, out_csv: Path, top_k: int) -> list[GeometrySensitivityRow]:
    """Run geometry sensitivity sweep."""
    rows = [_evaluate_variant(scene_path, variant) for variant in _variants()]
    _write_csv(out_csv, rows)
    top_rows = sorted(rows, key=lambda row: row.score)[:top_k]
    print(f"evaluated {len(rows)} variants")
    print(f"wrote rows to {out_csv}")
    for rank, row in enumerate(top_rows, start=1):
        print(
            f"{rank:>2}. ratio={row.peak_target_ratio:.3f} swing={row.min_swing_ratio:.3f} "
            f"force_norm={row.total_force_normalized:.3f} pen={row.penetration * 1000.0:.2f}mm "
            f"upright={row.upright:.3f} drop={row.base_height_drop:.4f} "
            f"foot_w={row.foot_width_scale:.1f} stance={row.stance_width_scale:.1f} "
            f"com_z={row.base_com_height_offset:+.2f} assist={row.lateral_assist} "
            f"diagnosis={row.diagnosis}"
        )
    reached = [row for row in rows if row.peak_target_ratio >= 0.58]
    if not reached:
        print("diagnosis: no variant reached 0.58; reference/PD/contact mode needs rethinking.")
    elif any(row.diagnosis == "geometry_limited" for row in reached):
        print("diagnosis: geometry changes can cross 0.58; current limit is likely geometric.")
    elif any(row.diagnosis == "controller_or_actuator_authority" for row in reached):
        print("diagnosis: only lateral assist crossed 0.58; likely controller/actuator authority.")
    return rows


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE_PATH)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--top-k", type=int, default=20)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run geometry sensitivity sweep."""
    args = build_parser().parse_args(argv)
    run_sweep(args.scene, args.out_csv, args.top_k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
