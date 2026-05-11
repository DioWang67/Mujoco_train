"""Check whether Sedon can lift a swing foot when support load is forced.

This diagnostic separates two failure modes:

1. The swing leg cannot generate clearance even when the robot is externally
   biased onto the support foot.
2. The swing leg can lift when support load is forced, so the missing piece is
   the normal controller/curriculum's lateral load transfer.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np

from sedon_baseline.env import DEFAULT_SCENE_PATH, SedonStandingConfig, SedonStandingEnv
from tools.sedon_debug_common import (
    BASE_PROXY_GEOM,
    DEBUG_OUT_DIR,
    FLOOR_GEOM,
    LEFT_FOOT_GEOM,
    RIGHT_FOOT_GEOM,
    contact_pairs,
    geom_id,
)


DEFAULT_OUT_CSV = DEBUG_OUT_DIR / "forced_support_lift_check.csv"
SUPPORT_TO_SWING = {"left": "right", "right": "left"}
FOOT_GEOM_BY_SIDE = {"left": LEFT_FOOT_GEOM, "right": RIGHT_FOOT_GEOM}
FOOT_INDEX_BY_SIDE = {"right": 0, "left": 1}
HIP_ROLL_INDEX = {"right": 1, "left": 6}
HIP_PITCH_INDEX = {"right": 2, "left": 7}
KNEE_PITCH_INDEX = {"right": 3, "left": 8}
ANKLE_PITCH_INDEX = {"right": 4, "left": 9}
STANCE_ROOT_BODIES = {"right": "R_link_hip_yaw", "left": "L_link_hip_yaw"}
BASE_BODY_NAME = "base_link"


@dataclass(frozen=True)
class ForcedSupportSample:
    """One sample from a forced-support lift rollout."""

    case_name: str
    step: int
    stage: str
    support_side: str
    lateral_force_n: float
    support_roll: float
    com_y: float
    support_foot_y: float
    swing_foot_y: float
    support_margin_y: float
    support_normal_force: float
    swing_normal_force: float
    support_force_fraction: float
    support_force_ratio: float
    support_contact_count: int
    swing_contact_count: int
    support_foot_bottom_z: float
    swing_foot_bottom_z: float
    swing_lift_delta_z: float
    base_z: float
    upright: float
    contact_state: str
    base_proxy_floor_contact: bool
    terminated: bool


@dataclass(frozen=True)
class ForcedSupportCaseSummary:
    """Aggregate result for one force magnitude candidate."""

    case_name: str
    support_side: str
    lateral_force_n: float
    max_support_force_fraction: float
    max_support_force_ratio: float
    min_support_margin_y: float
    max_swing_lift_delta_z: float
    max_clean_swing_lift_delta_z: float
    final_swing_lift_delta_z: float
    reached_lift_stage: bool
    swing_air_steps: int
    clean_lift_steps: int
    base_proxy_floor_steps: int
    terminated: bool
    terminated_step: int | None
    diagnosis: str


def _parse_float_list(raw_value: str) -> list[float]:
    """Parse comma-separated float values.

    Args:
        raw_value: Comma-separated string such as ``"0,5,10"``.

    Returns:
        Parsed float values.

    Raises:
        argparse.ArgumentTypeError: If no values are provided.
    """
    values = [float(item.strip()) for item in raw_value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one float.")
    return values


def _overall_com(env: SedonStandingEnv) -> np.ndarray:
    """Return the mass-weighted whole-body COM in world coordinates."""
    masses = env.model.body_mass
    total_mass = float(np.sum(masses))
    if total_mass <= 0.0:
        raise ValueError("Sedon model has no positive body mass.")
    return np.sum(env.data.xipos * masses[:, None], axis=0) / total_mass


def _foot_floor_load(env: SedonStandingEnv, side: str) -> tuple[int, float]:
    """Return floor-contact count and summed normal force for one foot."""
    foot_geom_name = FOOT_GEOM_BY_SIDE[side]
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


def _contact_state(env: SedonStandingEnv) -> tuple[str, bool]:
    """Return foot-contact state and whether the base proxy touches the floor."""
    pairs = [set((name_a, name_b)) for name_a, name_b, _ in contact_pairs(env.model, env.data)]
    right = {FLOOR_GEOM, RIGHT_FOOT_GEOM} in pairs
    left = {FLOOR_GEOM, LEFT_FOOT_GEOM} in pairs
    base = {FLOOR_GEOM, BASE_PROXY_GEOM} in pairs
    if right and left:
        return "both", base
    if right:
        return "right_only", base
    if left:
        return "left_only", base
    return "none", base


def _support_roll_offsets(support_side: str, magnitude: float) -> tuple[float, float]:
    """Return right/left hip-roll offsets that bias load toward a support side."""
    if support_side == "left":
        return magnitude, -magnitude
    if support_side == "right":
        return -magnitude, magnitude
    raise ValueError(f"Unsupported support side: {support_side}")


def _target_positions(
    env: SedonStandingEnv,
    *,
    support_side: str,
    support_roll: float,
    lift_scale: float,
    swing_hip_pitch_delta: float,
    swing_knee_pitch_delta: float,
    swing_ankle_pitch_delta: float,
) -> np.ndarray:
    """Build a support-bias plus swing-lift target."""
    target = env._nominal_joint_qpos.copy()
    right_roll, left_roll = _support_roll_offsets(support_side, support_roll)
    target[HIP_ROLL_INDEX["right"]] += right_roll
    target[HIP_ROLL_INDEX["left"]] += left_roll

    swing_side = SUPPORT_TO_SWING[support_side]
    target[HIP_PITCH_INDEX[swing_side]] += lift_scale * swing_hip_pitch_delta
    target[KNEE_PITCH_INDEX[swing_side]] += lift_scale * swing_knee_pitch_delta
    target[ANKLE_PITCH_INDEX[swing_side]] += lift_scale * swing_ankle_pitch_delta
    return env._apply_safe_joint_target_clamps(target)


def _apply_lateral_force(env: SedonStandingEnv, support_side: str, force_n: float) -> None:
    """Apply a world-y force to the base body toward the support side."""
    env.data.xfrc_applied[:] = 0.0
    direction = 1.0 if support_side == "left" else -1.0
    env.data.xfrc_applied[env._base_body_id, 1] = direction * abs(force_n)


def _apply_mechanical_overrides(
    env: SedonStandingEnv,
    *,
    foot_xy_scale: float,
    foot_z_scale: float,
    foot_friction: float,
    base_com_z_offset: float,
    stance_width_scale: float,
) -> None:
    """Apply optional in-memory foot/COM/stance overrides."""
    if min(foot_xy_scale, foot_z_scale, foot_friction, stance_width_scale) <= 0.0:
        raise ValueError("foot, friction, and stance scales must be positive.")
    for name in (RIGHT_FOOT_GEOM, LEFT_FOOT_GEOM):
        foot_id = geom_id(env.model, name)
        original_size = env.model.geom_size[foot_id].copy()
        env.model.geom_size[foot_id] = np.array(
            [
                original_size[0] * foot_xy_scale,
                original_size[1] * foot_xy_scale,
                original_size[2] * foot_z_scale,
            ],
            dtype=np.float64,
        )
        env.model.geom_rbound[foot_id] = float(np.linalg.norm(env.model.geom_size[foot_id]))
        env.model.geom_friction[foot_id][0] = foot_friction
    env.model.body_ipos[env._body_id(BASE_BODY_NAME)][2] += base_com_z_offset
    for side, body_name in STANCE_ROOT_BODIES.items():
        body_id = env._body_id(body_name)
        direction = -1.0 if side == "right" else 1.0
        env.model.body_pos[body_id][1] = (
            direction * abs(float(env.model.body_pos[body_id][1])) * stance_width_scale
        )


def _clear_lateral_force(env: SedonStandingEnv) -> None:
    """Clear any force injected by this diagnostic."""
    env.data.xfrc_applied[:] = 0.0


def _sample(
    env: SedonStandingEnv,
    *,
    case_name: str,
    step: int,
    stage: str,
    support_side: str,
    lateral_force_n: float,
    support_roll: float,
    standing_swing_foot_bottom_z: float,
) -> ForcedSupportSample:
    """Collect one rollout sample."""
    swing_side = SUPPORT_TO_SWING[support_side]
    support_contact_count, support_force = _foot_floor_load(env, support_side)
    swing_contact_count, swing_force = _foot_floor_load(env, swing_side)
    total_foot_force = support_force + swing_force
    support_force_fraction = support_force / max(total_foot_force, 1e-9)
    support_force_ratio = support_force / max(swing_force, 1e-9)

    support_geom_id = env._geom_id(FOOT_GEOM_BY_SIDE[support_side])
    swing_geom_id = env._geom_id(FOOT_GEOM_BY_SIDE[swing_side])
    foot_bottoms = env._foot_bottom_heights()
    support_foot_bottom_z = float(foot_bottoms[FOOT_INDEX_BY_SIDE[support_side]])
    swing_foot_bottom_z = float(foot_bottoms[FOOT_INDEX_BY_SIDE[swing_side]])
    contact_state, base_proxy_contact = _contact_state(env)
    obs = env._get_obs()
    base_z = env._base_height()
    upright = env._base_upright()
    terminated = env._is_terminated(base_z, upright, obs)

    com_y = float(_overall_com(env)[1])
    support_foot_y = float(env.data.geom_xpos[support_geom_id][1])
    swing_foot_y = float(env.data.geom_xpos[swing_geom_id][1])
    return ForcedSupportSample(
        case_name=case_name,
        step=step,
        stage=stage,
        support_side=support_side,
        lateral_force_n=float(lateral_force_n),
        support_roll=float(support_roll),
        com_y=com_y,
        support_foot_y=support_foot_y,
        swing_foot_y=swing_foot_y,
        support_margin_y=abs(com_y - support_foot_y),
        support_normal_force=support_force,
        swing_normal_force=swing_force,
        support_force_fraction=support_force_fraction,
        support_force_ratio=support_force_ratio,
        support_contact_count=support_contact_count,
        swing_contact_count=swing_contact_count,
        support_foot_bottom_z=support_foot_bottom_z,
        swing_foot_bottom_z=swing_foot_bottom_z,
        swing_lift_delta_z=swing_foot_bottom_z - standing_swing_foot_bottom_z,
        base_z=float(base_z),
        upright=float(upright),
        contact_state=contact_state,
        base_proxy_floor_contact=base_proxy_contact,
        terminated=terminated,
    )


def _diagnose_case(
    *,
    reached_lift_stage: bool,
    max_support_force_fraction: float,
    max_clean_swing_lift_delta_z: float,
    terminated: bool,
    force_fraction_gate: float,
    lift_delta_gate: float,
) -> str:
    """Return a compact diagnosis for one candidate rollout."""
    if not reached_lift_stage:
        return "forced_support_unstable_before_lift"
    loaded = max_support_force_fraction >= force_fraction_gate
    lifted = max_clean_swing_lift_delta_z >= lift_delta_gate
    if loaded and lifted and not terminated:
        return "forced_load_can_lift"
    if loaded and lifted and terminated:
        return "forced_load_can_lift_but_unstable"
    if loaded and not lifted:
        return "load_transfer_ok_but_lift_fails"
    return "forced_load_transfer_still_fails"


def _run_case(
    *,
    support_side: str,
    lateral_force_n: float,
    support_roll: float,
    settle_steps: int,
    lift_steps: int,
    swing_hip_pitch_delta: float,
    swing_knee_pitch_delta: float,
    swing_ankle_pitch_delta: float,
    force_fraction_gate: float,
    lift_delta_gate: float,
    seed: int,
    foot_xy_scale: float,
    foot_z_scale: float,
    base_com_z_offset: float,
    stance_width_scale: float,
    foot_friction: float,
    target_base_height: float,
    scene_path: Path | None,
) -> tuple[ForcedSupportCaseSummary, list[ForcedSupportSample]]:
    """Run one forced-support lift candidate."""
    case_name = f"{support_side}_force_{lateral_force_n:g}_roll_{support_roll:g}"
    env = SedonStandingEnv(
        scene_path=scene_path or DEFAULT_SCENE_PATH,
        reset_noise_scale=0.0,
        reward_config=SedonStandingConfig(
            gait_mode="fsm",
            target_base_height=target_base_height,
            min_base_height=min(0.34, target_base_height - 0.10),
            max_base_height=max(0.65, target_base_height + 0.20),
        ),
    )
    samples: list[ForcedSupportSample] = []
    try:
        _apply_mechanical_overrides(
            env,
            foot_xy_scale=foot_xy_scale,
            foot_z_scale=foot_z_scale,
            foot_friction=foot_friction,
            base_com_z_offset=base_com_z_offset,
            stance_width_scale=stance_width_scale,
        )
        mujoco.mj_setConst(env.model, env.data)
        env.reset(seed=seed)
        mujoco.mj_forward(env.model, env.data)
        swing_side = SUPPORT_TO_SWING[support_side]
        standing_swing_foot_bottom_z = float(
            env._foot_bottom_heights()[FOOT_INDEX_BY_SIDE[swing_side]]
        )

        terminated = False
        terminated_step: int | None = None
        step = 0
        for index in range(settle_steps):
            step += 1
            _apply_lateral_force(env, support_side, lateral_force_n)
            target = _target_positions(
                env,
                support_side=support_side,
                support_roll=support_roll,
                lift_scale=0.0,
                swing_hip_pitch_delta=swing_hip_pitch_delta,
                swing_knee_pitch_delta=swing_knee_pitch_delta,
                swing_ankle_pitch_delta=swing_ankle_pitch_delta,
            )
            env._do_pd_simulation(target)
            sample = _sample(
                env,
                case_name=case_name,
                step=step,
                stage="forced_support_settle",
                support_side=support_side,
                lateral_force_n=lateral_force_n,
                support_roll=support_roll,
                standing_swing_foot_bottom_z=standing_swing_foot_bottom_z,
            )
            samples.append(sample)
            if sample.terminated:
                terminated = True
                terminated_step = step
                break

        if not terminated:
            for index in range(lift_steps):
                step += 1
                lift_scale = min(1.0, (index + 1) / max(1, lift_steps // 2))
                _apply_lateral_force(env, support_side, lateral_force_n)
                target = _target_positions(
                    env,
                    support_side=support_side,
                    support_roll=support_roll,
                    lift_scale=lift_scale,
                    swing_hip_pitch_delta=swing_hip_pitch_delta,
                    swing_knee_pitch_delta=swing_knee_pitch_delta,
                    swing_ankle_pitch_delta=swing_ankle_pitch_delta,
                )
                env._do_pd_simulation(target)
                sample = _sample(
                    env,
                    case_name=case_name,
                    step=step,
                    stage="forced_support_lift",
                    support_side=support_side,
                    lateral_force_n=lateral_force_n,
                    support_roll=support_roll,
                    standing_swing_foot_bottom_z=standing_swing_foot_bottom_z,
                )
                samples.append(sample)
                if sample.terminated:
                    terminated = True
                    terminated_step = step
                    break
    finally:
        _clear_lateral_force(env)
        env.close()

    if not samples:
        raise RuntimeError(f"No samples collected for case {case_name}.")

    max_support_force_fraction = max(row.support_force_fraction for row in samples)
    max_support_force_ratio = max(row.support_force_ratio for row in samples)
    min_support_margin_y = min(row.support_margin_y for row in samples)
    max_swing_lift_delta_z = max(row.swing_lift_delta_z for row in samples)
    clean_lift_samples = [
        row
        for row in samples
        if row.stage == "forced_support_lift"
        and row.support_contact_count > 0
        and row.swing_contact_count == 0
        and not row.base_proxy_floor_contact
    ]
    max_clean_swing_lift_delta_z = (
        max(row.swing_lift_delta_z for row in clean_lift_samples)
        if clean_lift_samples
        else 0.0
    )
    final_swing_lift_delta_z = samples[-1].swing_lift_delta_z
    reached_lift_stage = any(row.stage == "forced_support_lift" for row in samples)
    swing_air_steps = sum(
        1
        for row in samples
        if row.stage == "forced_support_lift" and row.swing_contact_count == 0
    )
    clean_lift_steps = len(clean_lift_samples)
    base_proxy_floor_steps = sum(1 for row in samples if row.base_proxy_floor_contact)
    summary = ForcedSupportCaseSummary(
        case_name=case_name,
        support_side=support_side,
        lateral_force_n=float(lateral_force_n),
        max_support_force_fraction=max_support_force_fraction,
        max_support_force_ratio=max_support_force_ratio,
        min_support_margin_y=min_support_margin_y,
        max_swing_lift_delta_z=max_swing_lift_delta_z,
        max_clean_swing_lift_delta_z=max_clean_swing_lift_delta_z,
        final_swing_lift_delta_z=final_swing_lift_delta_z,
        reached_lift_stage=reached_lift_stage,
        swing_air_steps=swing_air_steps,
        clean_lift_steps=clean_lift_steps,
        base_proxy_floor_steps=base_proxy_floor_steps,
        terminated=terminated,
        terminated_step=terminated_step,
        diagnosis=_diagnose_case(
            reached_lift_stage=reached_lift_stage,
            max_support_force_fraction=max_support_force_fraction,
            max_clean_swing_lift_delta_z=max_clean_swing_lift_delta_z,
            terminated=terminated,
            force_fraction_gate=force_fraction_gate,
            lift_delta_gate=lift_delta_gate,
        ),
    )
    return summary, samples


def _write_csv(path: Path, rows: list[ForcedSupportSample]) -> None:
    """Write rollout samples to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows([asdict(row) for row in rows])


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support-side", choices=("left", "right", "both"), default="both")
    parser.add_argument("--lateral-forces", type=_parse_float_list, default=[0.0, 2.0, 5.0, 10.0, 15.0])
    parser.add_argument("--support-roll", type=float, default=0.04)
    parser.add_argument("--settle-steps", type=int, default=80)
    parser.add_argument("--lift-steps", type=int, default=80)
    parser.add_argument("--swing-hip-pitch-delta", type=float, default=-0.30)
    parser.add_argument("--swing-knee-pitch-delta", type=float, default=-0.30)
    parser.add_argument("--swing-ankle-pitch-delta", type=float, default=-0.10)
    parser.add_argument("--force-fraction-gate", type=float, default=0.70)
    parser.add_argument("--lift-delta-gate", type=float, default=0.005)
    parser.add_argument("--foot-xy-scale", type=float, default=1.0)
    parser.add_argument("--foot-z-scale", type=float, default=1.0)
    parser.add_argument("--foot-friction", type=float, default=1.0)
    parser.add_argument("--base-com-z-offset", type=float, default=0.0)
    parser.add_argument("--stance-width-scale", type=float, default=1.0)
    parser.add_argument("--target-base-height", type=float, default=0.446)
    parser.add_argument("--scene-path", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the forced-support lift diagnostic."""
    args = build_parser().parse_args(argv)
    if args.settle_steps <= 0:
        raise ValueError("--settle-steps must be positive.")
    if args.lift_steps <= 0:
        raise ValueError("--lift-steps must be positive.")
    if args.force_fraction_gate <= 0.0 or args.force_fraction_gate >= 1.0:
        raise ValueError("--force-fraction-gate must be between 0 and 1.")
    if args.lift_delta_gate <= 0.0:
        raise ValueError("--lift-delta-gate must be positive.")

    support_sides = ("left", "right") if args.support_side == "both" else (args.support_side,)
    all_samples: list[ForcedSupportSample] = []
    summaries: list[ForcedSupportCaseSummary] = []
    for support_side in support_sides:
        for lateral_force_n in args.lateral_forces:
            summary, samples = _run_case(
                support_side=support_side,
                lateral_force_n=float(lateral_force_n),
                support_roll=args.support_roll,
                settle_steps=args.settle_steps,
                lift_steps=args.lift_steps,
                swing_hip_pitch_delta=args.swing_hip_pitch_delta,
                swing_knee_pitch_delta=args.swing_knee_pitch_delta,
                swing_ankle_pitch_delta=args.swing_ankle_pitch_delta,
                force_fraction_gate=args.force_fraction_gate,
                lift_delta_gate=args.lift_delta_gate,
                seed=args.seed,
                foot_xy_scale=args.foot_xy_scale,
                foot_z_scale=args.foot_z_scale,
                foot_friction=args.foot_friction,
                base_com_z_offset=args.base_com_z_offset,
                stance_width_scale=args.stance_width_scale,
                target_base_height=args.target_base_height,
                scene_path=args.scene_path,
            )
            summaries.append(summary)
            all_samples.extend(samples)

    _write_csv(args.out_csv, all_samples)
    summaries.sort(
        key=lambda row: (
            row.max_support_force_fraction,
            row.max_swing_lift_delta_z,
            -float(row.terminated),
        ),
        reverse=True,
    )

    print(
        "rank side force_N support_frac force_ratio min_margin "
        "max_lift clean_lift final_lift lift_stage air_steps clean_steps "
        "base_steps terminated diagnosis"
    )
    for rank, row in enumerate(summaries, start=1):
        print(
            f"{rank:>4} {row.support_side:>5} {row.lateral_force_n:>7.2f} "
            f"{row.max_support_force_fraction:>12.3f} "
            f"{row.max_support_force_ratio:>11.3f} "
            f"{row.min_support_margin_y:>10.4f} "
            f"{row.max_swing_lift_delta_z:>8.4f} "
            f"{row.max_clean_swing_lift_delta_z:>10.4f} "
            f"{row.final_swing_lift_delta_z:>10.4f} "
            f"{str(row.reached_lift_stage):>10} "
            f"{row.swing_air_steps:>9} "
            f"{row.clean_lift_steps:>11} "
            f"{row.base_proxy_floor_steps:>10} "
            f"{str(row.terminated):>10} "
            f"{row.diagnosis}"
        )

    print(f"\nCSV: {args.out_csv}")
    actionable = [row for row in summaries if row.diagnosis.startswith("forced_load_can_lift")]
    if actionable:
        best = actionable[0]
        print(
            "interpretation: forced support loading can produce swing lift; "
            "focus next on load-transfer controller/reward instead of lift geometry."
        )
        print(
            "best_case: "
            f"side={best.support_side} force_N={best.lateral_force_n:.2f} "
            f"support_frac={best.max_support_force_fraction:.3f} "
            f"max_lift={best.max_swing_lift_delta_z:.4f}"
        )
    else:
        print(
            "interpretation: even forced support loading did not produce a clean lift; "
            "inspect contact geometry, actuator limits, and swing target feasibility next."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
