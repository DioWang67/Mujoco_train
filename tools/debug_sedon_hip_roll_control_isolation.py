"""Isolate Sedon hip-roll controllability across kinematic and dynamics modes."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np

from sedon_baseline.env import SedonStandingEnv, load_sedon_config_from_env
from tools.sedon_debug_common import DEBUG_OUT_DIR, contact_pairs


DEFAULT_OUT_CSV = DEBUG_OUT_DIR / "hip_roll_control_isolation.csv"
RIGHT_HIP_ROLL_INDEX = 1
LEFT_HIP_ROLL_INDEX = 6
RIGHT_HIP_ROLL_NAME = "R_joint_hip_roll"
LEFT_HIP_ROLL_NAME = "L_joint_hip_roll"
RIGHT_HIP_ROLL_ACTUATOR = "R_joint_hip_roll_motor"
LEFT_HIP_ROLL_ACTUATOR = "L_joint_hip_roll_motor"
RIGHT_FOOT_GEOM = "R_foot_collision"
LEFT_FOOT_GEOM = "L_foot_collision"
FLOOR_GEOM = "floor"
MODES = ("kinematic_only", "fixed_base", "free_base_no_floor", "free_base_with_floor")


@dataclass(frozen=True)
class HipRollIsolationResult:
    """One hip-roll controllability result row."""

    test_mode: str
    leg_mode: str
    offset: float
    target_right_qpos: float
    target_left_qpos: float
    actual_right_qpos: float
    actual_left_qpos: float
    right_tracking_error: float
    left_tracking_error: float
    right_ctrl_value: float
    left_ctrl_value: float
    right_ctrl_max_abs: float
    left_ctrl_max_abs: float
    overall_ctrl_max_abs: float
    right_ctrl_saturation_ratio: float
    left_ctrl_saturation_ratio: float
    base_y: float
    com_y: float
    base_dy: float
    com_dy: float
    right_foot_x: float
    right_foot_y: float
    right_foot_z: float
    left_foot_x: float
    left_foot_y: float
    left_foot_z: float
    right_foot_contact_ratio: float
    left_foot_contact_ratio: float
    base_z: float
    upright: float
    terminated: bool
    steps: int


def _parse_offsets(raw_value: str) -> list[float]:
    """Parse comma-separated offsets."""
    offsets = [float(part.strip()) for part in raw_value.split(",") if part.strip()]
    if not offsets:
        raise argparse.ArgumentTypeError("Expected at least one offset.")
    return offsets


def _overall_com(env: SedonStandingEnv) -> np.ndarray:
    """Return whole-body COM in world coordinates."""
    masses = env.model.body_mass
    total_mass = float(np.sum(masses))
    if total_mass <= 0.0:
        raise ValueError("Model has no positive body mass.")
    return np.sum(env.data.xipos * masses[:, None], axis=0) / total_mass


def _hip_roll_actuator_ids(env: SedonStandingEnv) -> tuple[int, int]:
    """Resolve hip-roll actuator ids."""
    right_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, RIGHT_HIP_ROLL_ACTUATOR)
    left_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, LEFT_HIP_ROLL_ACTUATOR)
    if right_id < 0 or left_id < 0:
        raise ValueError("Failed to resolve hip-roll actuators.")
    return int(right_id), int(left_id)


def _foot_contact_flags(env: SedonStandingEnv) -> tuple[bool, bool]:
    """Return whether each foot currently contacts the floor."""
    right = False
    left = False
    for name_a, name_b, _ in contact_pairs(env.model, env.data):
        pair = {name_a, name_b}
        if pair == {FLOOR_GEOM, RIGHT_FOOT_GEOM}:
            right = True
        elif pair == {FLOOR_GEOM, LEFT_FOOT_GEOM}:
            left = True
    return right, left


def _target_pair(mode: str, offset: float) -> tuple[float, float]:
    """Return right/left hip-roll target offsets."""
    if mode == "right_only":
        return offset, 0.0
    if mode == "left_only":
        return 0.0, offset
    raise ValueError(f"Unsupported leg_mode: {mode}")


def _scenario_label(mode: str) -> str:
    """Return a shorter human-facing scenario label."""
    return {
        "kinematic_only": "A_kinematic_only",
        "fixed_base": "B_fixed_base",
        "free_base_no_floor": "C_free_base_no_floor",
        "free_base_with_floor": "D_free_base_with_floor",
    }[mode]


def _restore_floor_contact(env: SedonStandingEnv, floor_geom_id: int, original_contype: int, original_conaffinity: int) -> None:
    """Restore floor geom contact flags."""
    env.model.geom_contype[floor_geom_id] = original_contype
    env.model.geom_conaffinity[floor_geom_id] = original_conaffinity
    mujoco.mj_forward(env.model, env.data)


def _disable_floor_contact(env: SedonStandingEnv, floor_geom_id: int) -> tuple[int, int]:
    """Disable floor contact in-memory and return the original values."""
    original_contype = int(env.model.geom_contype[floor_geom_id])
    original_conaffinity = int(env.model.geom_conaffinity[floor_geom_id])
    env.model.geom_contype[floor_geom_id] = 0
    env.model.geom_conaffinity[floor_geom_id] = 0
    mujoco.mj_forward(env.model, env.data)
    return original_contype, original_conaffinity


def _pin_base(env: SedonStandingEnv, base_qpos: np.ndarray) -> None:
    """Restore the floating-base pose and zero its velocity."""
    env.data.qpos[0:7] = base_qpos
    env.data.qvel[0:6] = 0.0


def _apply_tracking_step(
    env: SedonStandingEnv,
    target_positions: np.ndarray,
    *,
    fixed_base: bool,
    fixed_base_qpos: np.ndarray | None,
) -> None:
    """Apply one RL-step worth of PD simulation."""
    for _ in range(env.frame_skip):
        env.data.ctrl[:] = env._pd_control(target_positions)
        mujoco.mj_step(env.model, env.data)
        if fixed_base:
            if fixed_base_qpos is None:
                raise ValueError("fixed_base_qpos is required when fixed_base=True.")
            _pin_base(env, fixed_base_qpos)
            mujoco.mj_forward(env.model, env.data)


def _result_from_state(
    env: SedonStandingEnv,
    *,
    test_mode: str,
    leg_mode: str,
    offset: float,
    target_right_qpos: float,
    target_left_qpos: float,
    initial_base_y: float,
    initial_com_y: float,
    right_ctrl_value: float,
    left_ctrl_value: float,
    right_ctrl_max_abs: float,
    left_ctrl_max_abs: float,
    overall_ctrl_max_abs: float,
    right_ctrl_saturation_ratio: float,
    left_ctrl_saturation_ratio: float,
    right_contact_ratio: float,
    left_contact_ratio: float,
    terminated: bool,
    steps: int,
) -> HipRollIsolationResult:
    """Build one flat result from the current simulation state."""
    joint_positions = env._joint_positions()
    right_foot_geom_id = env._geom_id(RIGHT_FOOT_GEOM)
    left_foot_geom_id = env._geom_id(LEFT_FOOT_GEOM)
    return HipRollIsolationResult(
        test_mode=_scenario_label(test_mode),
        leg_mode=leg_mode,
        offset=offset,
        target_right_qpos=target_right_qpos,
        target_left_qpos=target_left_qpos,
        actual_right_qpos=float(joint_positions[RIGHT_HIP_ROLL_INDEX]),
        actual_left_qpos=float(joint_positions[LEFT_HIP_ROLL_INDEX]),
        right_tracking_error=float(joint_positions[RIGHT_HIP_ROLL_INDEX] - target_right_qpos),
        left_tracking_error=float(joint_positions[LEFT_HIP_ROLL_INDEX] - target_left_qpos),
        right_ctrl_value=right_ctrl_value,
        left_ctrl_value=left_ctrl_value,
        right_ctrl_max_abs=right_ctrl_max_abs,
        left_ctrl_max_abs=left_ctrl_max_abs,
        overall_ctrl_max_abs=overall_ctrl_max_abs,
        right_ctrl_saturation_ratio=right_ctrl_saturation_ratio,
        left_ctrl_saturation_ratio=left_ctrl_saturation_ratio,
        base_y=float(env.data.qpos[1]),
        com_y=float(_overall_com(env)[1]),
        base_dy=float(env.data.qpos[1] - initial_base_y),
        com_dy=float(_overall_com(env)[1] - initial_com_y),
        right_foot_x=float(env.data.geom_xpos[right_foot_geom_id][0]),
        right_foot_y=float(env.data.geom_xpos[right_foot_geom_id][1]),
        right_foot_z=float(env.data.geom_xpos[right_foot_geom_id][2]),
        left_foot_x=float(env.data.geom_xpos[left_foot_geom_id][0]),
        left_foot_y=float(env.data.geom_xpos[left_foot_geom_id][1]),
        left_foot_z=float(env.data.geom_xpos[left_foot_geom_id][2]),
        right_foot_contact_ratio=right_contact_ratio,
        left_foot_contact_ratio=left_contact_ratio,
        base_z=env._base_height(),
        upright=env._base_upright(),
        terminated=terminated,
        steps=steps,
    )


def _run_kinematic_only(
    env: SedonStandingEnv,
    *,
    leg_mode: str,
    offset: float,
    right_qpos_adr: int,
    left_qpos_adr: int,
) -> HipRollIsolationResult:
    """Set qpos directly and run only mj_forward."""
    env.reset(seed=42)
    baseline_qpos = env.data.qpos.copy()
    baseline_qvel = np.zeros_like(env.data.qvel)
    initial_base_y = float(env.data.qpos[1])
    initial_com_y = float(_overall_com(env)[1])
    target_right_offset, target_left_offset = _target_pair(leg_mode, offset)
    target_right_qpos = float(baseline_qpos[right_qpos_adr] + target_right_offset)
    target_left_qpos = float(baseline_qpos[left_qpos_adr] + target_left_offset)
    baseline_qpos[right_qpos_adr] = target_right_qpos
    baseline_qpos[left_qpos_adr] = target_left_qpos
    env.set_state(baseline_qpos, baseline_qvel)
    mujoco.mj_forward(env.model, env.data)
    return _result_from_state(
        env,
        test_mode="kinematic_only",
        leg_mode=leg_mode,
        offset=offset,
        target_right_qpos=target_right_qpos,
        target_left_qpos=target_left_qpos,
        initial_base_y=initial_base_y,
        initial_com_y=initial_com_y,
        right_ctrl_value=0.0,
        left_ctrl_value=0.0,
        right_ctrl_max_abs=0.0,
        left_ctrl_max_abs=0.0,
        overall_ctrl_max_abs=0.0,
        right_ctrl_saturation_ratio=0.0,
        left_ctrl_saturation_ratio=0.0,
        right_contact_ratio=0.0,
        left_contact_ratio=0.0,
        terminated=False,
        steps=0,
    )


def _run_dynamics(
    env: SedonStandingEnv,
    *,
    scenario: str,
    leg_mode: str,
    offset: float,
    steps: int,
    right_qpos_adr: int,
    left_qpos_adr: int,
    right_actuator_id: int,
    left_actuator_id: int,
) -> HipRollIsolationResult:
    """Run one dynamics scenario for one isolated hip-roll target."""
    env.reset(seed=42)
    initial_base_y = float(env.data.qpos[1])
    initial_com_y = float(_overall_com(env)[1])
    target_positions = env._joint_positions().copy()
    target_right_offset, target_left_offset = _target_pair(leg_mode, offset)
    target_positions[RIGHT_HIP_ROLL_INDEX] += target_right_offset
    target_positions[LEFT_HIP_ROLL_INDEX] += target_left_offset
    target_right_qpos = float(env.data.qpos[right_qpos_adr] + target_right_offset)
    target_left_qpos = float(env.data.qpos[left_qpos_adr] + target_left_offset)

    floor_geom_id = env._geom_id(FLOOR_GEOM)
    original_contype = int(env.model.geom_contype[floor_geom_id])
    original_conaffinity = int(env.model.geom_conaffinity[floor_geom_id])
    fixed_base_qpos = env.data.qpos[0:7].copy() if scenario == "fixed_base" else None
    if scenario == "free_base_no_floor":
        _disable_floor_contact(env, floor_geom_id)

    right_contact_steps = 0
    left_contact_steps = 0
    right_saturated_steps = 0
    left_saturated_steps = 0
    terminated = False
    right_ctrl_value = 0.0
    left_ctrl_value = 0.0
    right_ctrl_max_abs = 0.0
    left_ctrl_max_abs = 0.0
    overall_ctrl_max_abs = 0.0
    ctrl_lower = env.model.actuator_ctrlrange[:, 0]
    ctrl_upper = env.model.actuator_ctrlrange[:, 1]
    try:
        for _ in range(steps):
            _apply_tracking_step(
                env,
                target_positions,
                fixed_base=scenario == "fixed_base",
                fixed_base_qpos=fixed_base_qpos,
            )
            right_ctrl_value = float(env.data.ctrl[right_actuator_id])
            left_ctrl_value = float(env.data.ctrl[left_actuator_id])
            right_ctrl_max_abs = max(right_ctrl_max_abs, abs(right_ctrl_value))
            left_ctrl_max_abs = max(left_ctrl_max_abs, abs(left_ctrl_value))
            overall_ctrl_max_abs = max(overall_ctrl_max_abs, float(np.max(np.abs(env.data.ctrl))))
            right_saturated_steps += int(
                abs(right_ctrl_value - float(ctrl_lower[right_actuator_id])) <= 1e-6
                or abs(right_ctrl_value - float(ctrl_upper[right_actuator_id])) <= 1e-6
            )
            left_saturated_steps += int(
                abs(left_ctrl_value - float(ctrl_lower[left_actuator_id])) <= 1e-6
                or abs(left_ctrl_value - float(ctrl_upper[left_actuator_id])) <= 1e-6
            )
            right_contact, left_contact = _foot_contact_flags(env)
            right_contact_steps += int(right_contact)
            left_contact_steps += int(left_contact)
            obs = env._get_obs()
            terminated = terminated or env._is_terminated(env._base_height(), env._base_upright(), obs)
    finally:
        if scenario == "free_base_no_floor":
            _restore_floor_contact(env, floor_geom_id, original_contype, original_conaffinity)

    return _result_from_state(
        env,
        test_mode=scenario,
        leg_mode=leg_mode,
        offset=offset,
        target_right_qpos=target_right_qpos,
        target_left_qpos=target_left_qpos,
        initial_base_y=initial_base_y,
        initial_com_y=initial_com_y,
        right_ctrl_value=right_ctrl_value,
        left_ctrl_value=left_ctrl_value,
        right_ctrl_max_abs=right_ctrl_max_abs,
        left_ctrl_max_abs=left_ctrl_max_abs,
        overall_ctrl_max_abs=overall_ctrl_max_abs,
        right_ctrl_saturation_ratio=right_saturated_steps / max(steps, 1),
        left_ctrl_saturation_ratio=left_saturated_steps / max(steps, 1),
        right_contact_ratio=right_contact_steps / max(steps, 1),
        left_contact_ratio=left_contact_steps / max(steps, 1),
        terminated=terminated,
        steps=steps,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--offsets",
        type=_parse_offsets,
        default=[-0.2, -0.1, 0.1, 0.2],
        help="Comma-separated hip-roll offsets in radians.",
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the hip-roll control isolation diagnostic."""
    args = build_parser().parse_args(argv)
    if args.steps <= 0:
        raise ValueError("--steps must be positive.")

    reward_config = load_sedon_config_from_env()
    env = SedonStandingEnv(reset_noise_scale=0.0, reward_config=reward_config)
    rows: list[HipRollIsolationResult] = []
    try:
        right_joint_id = env._joint_id(RIGHT_HIP_ROLL_NAME)
        left_joint_id = env._joint_id(LEFT_HIP_ROLL_NAME)
        right_qpos_adr = int(env.model.jnt_qposadr[right_joint_id])
        left_qpos_adr = int(env.model.jnt_qposadr[left_joint_id])
        right_actuator_id, left_actuator_id = _hip_roll_actuator_ids(env)
        for leg_mode in ("right_only", "left_only"):
            for offset in args.offsets:
                rows.append(
                    _run_kinematic_only(
                        env,
                        leg_mode=leg_mode,
                        offset=offset,
                        right_qpos_adr=right_qpos_adr,
                        left_qpos_adr=left_qpos_adr,
                    )
                )
        for scenario in ("fixed_base", "free_base_no_floor", "free_base_with_floor"):
            for leg_mode in ("right_only", "left_only"):
                for offset in args.offsets:
                    rows.append(
                        _run_dynamics(
                            env,
                            scenario=scenario,
                            leg_mode=leg_mode,
                            offset=offset,
                            steps=args.steps,
                            right_qpos_adr=right_qpos_adr,
                            left_qpos_adr=left_qpos_adr,
                            right_actuator_id=right_actuator_id,
                            left_actuator_id=left_actuator_id,
                        )
                    )
    finally:
        env.close()

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows([asdict(row) for row in rows])

    print(f"csv: {args.out_csv}")
    print(
        "test_mode leg_mode offset target_R target_L actual_R actual_L "
        "err_R err_L ctrl_R ctrl_L sat_R sat_L base_dy com_dy "
        "right_contact left_contact base_z upright terminated"
    )
    for row in rows:
        print(
            f"{row.test_mode:>20} {row.leg_mode:>10} {row.offset:>6.2f} "
            f"{row.target_right_qpos:>8.3f} {row.target_left_qpos:>8.3f} "
            f"{row.actual_right_qpos:>8.3f} {row.actual_left_qpos:>8.3f} "
            f"{row.right_tracking_error:>8.3f} {row.left_tracking_error:>8.3f} "
            f"{row.right_ctrl_value:>7.2f} {row.left_ctrl_value:>7.2f} "
            f"{row.right_ctrl_saturation_ratio:>5.2f} {row.left_ctrl_saturation_ratio:>5.2f} "
            f"{row.base_dy:>8.5f} {row.com_dy:>8.5f} "
            f"{row.right_foot_contact_ratio:>6.2f} {row.left_foot_contact_ratio:>6.2f} "
            f"{row.base_z:>7.4f} {row.upright:>7.4f} {str(row.terminated):>10}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
