"""Diagnose whether Sedon can transfer load into usable single support."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np

from sedon_baseline.env import SedonStandingEnv, load_sedon_config_from_env
from tools.sedon_debug_common import DEFAULT_SCENE_PATH, DEBUG_OUT_DIR, contact_pairs, require_scene


DEFAULT_OUT_CSV = DEBUG_OUT_DIR / "single_support_load_transfer.csv"
RIGHT_FOOT_GEOM = "R_foot_collision"
LEFT_FOOT_GEOM = "L_foot_collision"
FLOOR_GEOM = "floor"
SUPPORT_TO_SWING = {"left": "right", "right": "left"}
HIP_ROLL_INDEX = {"right": 1, "left": 6}
HIP_PITCH_INDEX = {"right": 2, "left": 7}
KNEE_PITCH_INDEX = {"right": 3, "left": 8}
ANKLE_PITCH_INDEX = {"right": 4, "left": 9}


@dataclass(frozen=True)
class LoadTransferRow:
    """One time-step sample for the load-transfer diagnostic."""

    step: int
    time_s: float
    stage: str
    support_side: str
    swing_side: str
    base_pos_y: float
    com_y: float
    left_foot_y: float
    right_foot_y: float
    support_foot_y: float
    swing_foot_y: float
    support_margin_y: float
    left_contact_count: int
    right_contact_count: int
    support_contact_count: int
    swing_contact_count: int
    left_normal_force: float
    right_normal_force: float
    support_normal_force: float
    swing_normal_force: float
    support_force_ratio: float
    support_hip_roll_target: float
    support_hip_roll_qpos: float
    support_hip_roll_error: float
    swing_hip_pitch_target: float
    swing_hip_pitch_qpos: float
    swing_knee_pitch_target: float
    swing_knee_pitch_qpos: float
    swing_ankle_pitch_target: float
    swing_ankle_pitch_qpos: float
    support_foot_bottom_z: float
    swing_foot_bottom_z: float
    swing_lift_delta_z: float
    contact_state: str
    can_try_lift: bool
    base_z: float
    upright: float
    terminated: bool


def _overall_com(env: SedonStandingEnv) -> np.ndarray:
    """Return the mass-weighted whole-body COM in world coordinates."""
    masses = env.model.body_mass
    total_mass = float(np.sum(masses))
    if total_mass <= 0.0:
        raise ValueError("Model has no positive body mass.")
    return np.sum(env.data.xipos * masses[:, None], axis=0) / total_mass


def _foot_geom_name(side: str) -> str:
    """Return the MuJoCo foot geom name for one side."""
    return LEFT_FOOT_GEOM if side == "left" else RIGHT_FOOT_GEOM


def _contact_state(env: SedonStandingEnv) -> str:
    """Return a compact foot/base contact state."""
    right = False
    left = False
    base = False
    for name_a, name_b, _ in contact_pairs(env.model, env.data):
        pair = {name_a, name_b}
        if pair == {FLOOR_GEOM, RIGHT_FOOT_GEOM}:
            right = True
        elif pair == {FLOOR_GEOM, LEFT_FOOT_GEOM}:
            left = True
        elif pair == {FLOOR_GEOM, "base_proxy"}:
            base = True
    if right and left:
        state = "both"
    elif right:
        state = "right_only"
    elif left:
        state = "left_only"
    else:
        state = "none"
    if base:
        state = f"{state}+base"
    return state


def _foot_floor_load(env: SedonStandingEnv, side: str) -> tuple[int, float]:
    """Return floor-contact count and summed normal force for one foot."""
    foot_geom_name = _foot_geom_name(side)
    contact_count = 0
    normal_force_sum = 0.0
    wrench = np.zeros(6, dtype=np.float64)
    for contact_index in range(env.data.ncon):
        contact = env.data.contact[contact_index]
        name_a = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1))
        name_b = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2))
        if {name_a, name_b} != {FLOOR_GEOM, foot_geom_name}:
            continue
        mujoco.mj_contactForce(env.model, env.data, contact_index, wrench)
        contact_count += 1
        normal_force_sum += abs(float(wrench[0]))
    return contact_count, normal_force_sum


def _support_roll_offsets(support_side: str, magnitude: float) -> tuple[float, float]:
    """Return right/left hip-roll offsets for shifting onto one support foot."""
    if support_side == "left":
        return magnitude, -magnitude
    if support_side == "right":
        return -magnitude, magnitude
    raise ValueError(f"Unsupported support side: {support_side}")


def _make_target(
    env: SedonStandingEnv,
    *,
    support_side: str,
    support_roll_scale: float,
    swing_hip_pitch_delta: float,
    swing_knee_pitch_delta: float,
    swing_ankle_pitch_delta: float,
) -> np.ndarray:
    """Return one deterministic target for load transfer or micro-lift."""
    target = env._nominal_joint_qpos.copy()
    right_roll, left_roll = _support_roll_offsets(support_side, support_roll_scale)
    target[HIP_ROLL_INDEX["right"]] += right_roll
    target[HIP_ROLL_INDEX["left"]] += left_roll

    swing_side = SUPPORT_TO_SWING[support_side]
    target[HIP_PITCH_INDEX[swing_side]] += swing_hip_pitch_delta
    target[KNEE_PITCH_INDEX[swing_side]] += swing_knee_pitch_delta
    target[ANKLE_PITCH_INDEX[swing_side]] += swing_ankle_pitch_delta
    return env._apply_safe_joint_target_clamps(target)


def _sample_row(
    env: SedonStandingEnv,
    *,
    step: int,
    stage: str,
    support_side: str,
    support_margin_threshold: float,
    force_ratio_threshold: float,
    target_positions: np.ndarray,
    standing_swing_foot_bottom_z: float,
) -> LoadTransferRow:
    """Capture one diagnostic sample from the current simulation state."""
    swing_side = SUPPORT_TO_SWING[support_side]
    foot_bottoms = env._foot_bottom_heights()
    support_foot_geom_id = env._geom_id(_foot_geom_name(support_side))
    swing_foot_geom_id = env._geom_id(_foot_geom_name(swing_side))
    left_contact_count, left_normal_force = _foot_floor_load(env, "left")
    right_contact_count, right_normal_force = _foot_floor_load(env, "right")
    support_contact_count = (
        left_contact_count if support_side == "left" else right_contact_count
    )
    swing_contact_count = (
        right_contact_count if support_side == "left" else left_contact_count
    )
    support_normal_force = (
        left_normal_force if support_side == "left" else right_normal_force
    )
    swing_normal_force = (
        right_normal_force if support_side == "left" else left_normal_force
    )
    support_force_ratio = support_normal_force / max(swing_normal_force, 1e-9)
    com_y = float(_overall_com(env)[1])
    left_foot_geom_id = env._geom_id(LEFT_FOOT_GEOM)
    right_foot_geom_id = env._geom_id(RIGHT_FOOT_GEOM)
    left_foot_y = float(env.data.geom_xpos[left_foot_geom_id][1])
    right_foot_y = float(env.data.geom_xpos[right_foot_geom_id][1])
    support_foot_y = float(env.data.geom_xpos[support_foot_geom_id][1])
    swing_foot_y = float(env.data.geom_xpos[swing_foot_geom_id][1])
    support_margin_y = abs(com_y - support_foot_y)
    can_try_lift = (
        support_margin_y < support_margin_threshold
        and support_force_ratio > force_ratio_threshold
    )
    obs = env._get_obs()
    base_z = env._base_height()
    upright = env._base_upright()
    joint_positions = env._joint_positions()
    terminated = env._is_terminated(base_z, upright, obs)
    return LoadTransferRow(
        step=step,
        time_s=float(step * env.dt),
        stage=stage,
        support_side=support_side,
        swing_side=swing_side,
        base_pos_y=float(env.data.qpos[1]),
        com_y=com_y,
        left_foot_y=left_foot_y,
        right_foot_y=right_foot_y,
        support_foot_y=support_foot_y,
        swing_foot_y=swing_foot_y,
        support_margin_y=support_margin_y,
        left_contact_count=left_contact_count,
        right_contact_count=right_contact_count,
        support_contact_count=support_contact_count,
        swing_contact_count=swing_contact_count,
        left_normal_force=left_normal_force,
        right_normal_force=right_normal_force,
        support_normal_force=support_normal_force,
        swing_normal_force=swing_normal_force,
        support_force_ratio=support_force_ratio,
        support_hip_roll_target=float(target_positions[HIP_ROLL_INDEX[support_side]]),
        support_hip_roll_qpos=float(joint_positions[HIP_ROLL_INDEX[support_side]]),
        support_hip_roll_error=float(
            joint_positions[HIP_ROLL_INDEX[support_side]] - target_positions[HIP_ROLL_INDEX[support_side]]
        ),
        swing_hip_pitch_target=float(target_positions[HIP_PITCH_INDEX[swing_side]]),
        swing_hip_pitch_qpos=float(joint_positions[HIP_PITCH_INDEX[swing_side]]),
        swing_knee_pitch_target=float(target_positions[KNEE_PITCH_INDEX[swing_side]]),
        swing_knee_pitch_qpos=float(joint_positions[KNEE_PITCH_INDEX[swing_side]]),
        swing_ankle_pitch_target=float(target_positions[ANKLE_PITCH_INDEX[swing_side]]),
        swing_ankle_pitch_qpos=float(joint_positions[ANKLE_PITCH_INDEX[swing_side]]),
        support_foot_bottom_z=float(foot_bottoms[0] if support_side == "right" else foot_bottoms[1]),
        swing_foot_bottom_z=float(foot_bottoms[0] if swing_side == "right" else foot_bottoms[1]),
        swing_lift_delta_z=float(
            (foot_bottoms[0] if swing_side == "right" else foot_bottoms[1]) - standing_swing_foot_bottom_z
        ),
        contact_state=_contact_state(env),
        can_try_lift=can_try_lift,
        base_z=float(base_z),
        upright=float(upright),
        terminated=terminated,
    )


def _write_rows(path: Path, rows: list[LoadTransferRow]) -> None:
    """Write diagnostic rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows([asdict(row) for row in rows])


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support-side", choices=("left", "right"), default="left")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scene-path", type=Path, default=DEFAULT_SCENE_PATH)
    parser.add_argument("--load-steps", type=int, default=120)
    parser.add_argument("--lift-steps", type=int, default=80)
    parser.add_argument("--support-roll", type=float, default=0.10)
    parser.add_argument("--support-margin-threshold", type=float, default=0.035)
    parser.add_argument("--force-ratio-threshold", type=float, default=1.20)
    parser.add_argument("--swing-hip-pitch-delta", type=float, default=-0.020)
    parser.add_argument("--swing-knee-pitch-delta", type=float, default=-0.040)
    parser.add_argument("--swing-ankle-pitch-delta", type=float, default=0.020)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the single-support load-transfer diagnostic."""
    args = build_parser().parse_args(argv)
    if args.load_steps <= 0:
        raise ValueError("--load-steps must be positive.")
    if args.lift_steps <= 0:
        raise ValueError("--lift-steps must be positive.")
    if args.print_every <= 0:
        raise ValueError("--print-every must be positive.")

    reward_config = load_sedon_config_from_env()
    env = SedonStandingEnv(
        scene_path=require_scene(args.scene_path),
        reset_noise_scale=0.0,
        reward_config=reward_config,
    )
    rows: list[LoadTransferRow] = []
    lift_stage_started = False
    gate_row: LoadTransferRow | None = None
    try:
        env.reset(seed=args.seed)
        swing_side = SUPPORT_TO_SWING[args.support_side]
        standing_foot_bottoms = env._foot_bottom_heights()
        standing_swing_foot_bottom_z = float(
            standing_foot_bottoms[0] if swing_side == "right" else standing_foot_bottoms[1]
        )
        print(
            "step stage com_y support_y margin support_force swing_force "
            "force_ratio hip_roll_err swing_lift contact base_z upright can_lift"
        )

        for step in range(1, args.load_steps + 1):
            alpha = min(1.0, step / max(args.load_steps, 1))
            target = _make_target(
                env,
                support_side=args.support_side,
                support_roll_scale=args.support_roll * alpha,
                swing_hip_pitch_delta=0.0,
                swing_knee_pitch_delta=0.0,
                swing_ankle_pitch_delta=0.0,
            )
            env._do_pd_simulation(target)
            row = _sample_row(
                env,
                step=step,
                stage="load_transfer",
                support_side=args.support_side,
                support_margin_threshold=args.support_margin_threshold,
                force_ratio_threshold=args.force_ratio_threshold,
                target_positions=target,
                standing_swing_foot_bottom_z=standing_swing_foot_bottom_z,
            )
            rows.append(row)
            gate_row = row
            if step == 1 or step % args.print_every == 0 or row.terminated:
                print(
                    f"{row.step:>4} {row.stage:>14} {row.com_y:>7.4f} {row.support_foot_y:>9.4f} "
                    f"{row.support_margin_y:>7.4f} {row.support_normal_force:>12.4f} "
                    f"{row.swing_normal_force:>11.4f} {row.support_force_ratio:>10.3f} "
                    f"{row.support_hip_roll_error:>12.4f} {row.swing_lift_delta_z:>10.4f} "
                    f"{row.contact_state:>10} {row.base_z:>7.4f} {row.upright:>7.4f} {str(row.can_try_lift):>9}"
                )
            if row.terminated:
                break

        if gate_row is not None and gate_row.can_try_lift and not gate_row.terminated:
            lift_stage_started = True
            for lift_index in range(1, args.lift_steps + 1):
                alpha = min(1.0, lift_index / max(args.lift_steps, 1))
                target = _make_target(
                    env,
                    support_side=args.support_side,
                    support_roll_scale=args.support_roll,
                    swing_hip_pitch_delta=args.swing_hip_pitch_delta * alpha,
                    swing_knee_pitch_delta=args.swing_knee_pitch_delta * alpha,
                    swing_ankle_pitch_delta=args.swing_ankle_pitch_delta * alpha,
                )
                env._do_pd_simulation(target)
                row = _sample_row(
                    env,
                    step=args.load_steps + lift_index,
                    stage="micro_lift",
                    support_side=args.support_side,
                    support_margin_threshold=args.support_margin_threshold,
                    force_ratio_threshold=args.force_ratio_threshold,
                    target_positions=target,
                    standing_swing_foot_bottom_z=standing_swing_foot_bottom_z,
                )
                rows.append(row)
                if lift_index == 1 or lift_index % args.print_every == 0 or row.terminated:
                    print(
                        f"{row.step:>4} {row.stage:>14} {row.com_y:>7.4f} {row.support_foot_y:>9.4f} "
                        f"{row.support_margin_y:>7.4f} {row.support_normal_force:>12.4f} "
                        f"{row.swing_normal_force:>11.4f} {row.support_force_ratio:>10.3f} "
                        f"{row.support_hip_roll_error:>12.4f} {row.swing_lift_delta_z:>10.4f} "
                        f"{row.contact_state:>10} {row.base_z:>7.4f} {row.upright:>7.4f} {str(row.can_try_lift):>9}"
                    )
                if row.terminated:
                    break
    finally:
        env.close()

    _write_rows(args.out_csv, rows)

    print(f"\ncsv: {args.out_csv}")
    print(f"rows: {len(rows)}")
    if gate_row is not None:
        print(
            "gate_at_load_end: "
            f"can_try_lift={gate_row.can_try_lift} "
            f"support_margin_y={gate_row.support_margin_y:.5f} "
            f"support_force_ratio={gate_row.support_force_ratio:.3f}"
        )
    if rows:
        max_lift = max(row.swing_lift_delta_z for row in rows)
        max_support_force_ratio = max(row.support_force_ratio for row in rows)
        min_support_margin = min(row.support_margin_y for row in rows)
        support_only_steps = sum(
            1
            for row in rows
            if row.contact_state == f"{args.support_side}_only"
        )
        print(
            "summary: "
            f"lift_stage_started={lift_stage_started} "
            f"max_swing_lift_delta_z={max_lift:.5f} "
            f"min_support_margin_y={min_support_margin:.5f} "
            f"max_support_force_ratio={max_support_force_ratio:.3f} "
            f"support_only_steps={support_only_steps}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
