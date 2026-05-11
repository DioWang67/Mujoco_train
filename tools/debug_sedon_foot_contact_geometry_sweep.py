"""Sweep Sedon foot contact geometry overrides under with-floor load transfer."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import mujoco
import numpy as np

from sedon_baseline.env import SedonStandingEnv, load_sedon_config_from_env
from tools.sedon_debug_common import (
    DEBUG_OUT_DIR,
    LEFT_FOOT_GEOM,
    RIGHT_FOOT_GEOM,
    apply_foot_size_override,
    contact_pairs,
    geom_id,
)


DEFAULT_OUT_CSV = DEBUG_OUT_DIR / "foot_contact_geometry_sweep.csv"
FLOOR_GEOM = "floor"
HIP_ROLL_INDEX = {"right": 1, "left": 6}
BASELINE_FOOT_SIZE = (0.07, 0.04, 0.025)
BASELINE_FRICTION = (1.0, 0.005, 0.0001)


@dataclass(frozen=True)
class SweepScenario:
    """One in-memory diagnostic override scenario."""

    name: str
    foot_size: tuple[float, float, float] | None = None
    friction: tuple[float, float, float] | None = None
    base_height_delta: float = 0.0


@dataclass(frozen=True)
class FootContactSweepResult:
    """One result row for a scenario/support-side pair."""

    scenario: str
    support_side: str
    foot_size_x: float
    foot_size_y: float
    foot_size_z: float
    friction_tangent: float
    friction_torsional: float
    friction_rolling: float
    base_height_delta: float
    initial_base_z: float
    initial_left_foot_bottom_z: float
    initial_right_foot_bottom_z: float
    initial_left_contact_count: int
    initial_right_contact_count: int
    initial_contact_state: str
    steps: int
    terminated: bool
    terminated_step: int
    final_contact_state: str
    support_only_steps: int
    both_contact_steps: int
    none_contact_steps: int
    max_abs_com_y_delta: float
    max_abs_base_roll_delta: float
    final_com_y_delta: float
    final_base_roll_delta: float
    final_left_force_ratio: float
    final_right_force_ratio: float
    max_left_force_ratio: float
    max_right_force_ratio: float
    max_abs_support_hip_roll_error: float
    final_support_hip_roll_error: float
    min_base_z: float
    min_upright: float


def _overall_com(env: SedonStandingEnv) -> np.ndarray:
    """Return mass-weighted whole-body COM in world coordinates."""
    masses = env.model.body_mass
    total_mass = float(np.sum(masses))
    if total_mass <= 0.0:
        raise ValueError("Model has no positive body mass.")
    return np.sum(env.data.xipos * masses[:, None], axis=0) / total_mass


def _quat_to_roll(quat: np.ndarray) -> float:
    """Return base roll in radians from a MuJoCo quaternion."""
    w, x, y, z = [float(value) for value in quat]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    return float(math.atan2(sinr_cosp, cosr_cosp))


def _contact_state(env: SedonStandingEnv) -> str:
    """Return compact left/right foot floor-contact state."""
    right = False
    left = False
    for name_a, name_b, _ in contact_pairs(env.model, env.data):
        pair = {name_a, name_b}
        if pair == {FLOOR_GEOM, RIGHT_FOOT_GEOM}:
            right = True
        elif pair == {FLOOR_GEOM, LEFT_FOOT_GEOM}:
            left = True
    if right and left:
        return "both"
    if right:
        return "right_only"
    if left:
        return "left_only"
    return "none"


def _foot_floor_load(env: SedonStandingEnv, side: str) -> tuple[int, float]:
    """Return floor-contact count and summed normal force for one foot."""
    foot_geom_name = LEFT_FOOT_GEOM if side == "left" else RIGHT_FOOT_GEOM
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


def _apply_friction_override(
    env: SedonStandingEnv,
    friction: tuple[float, float, float],
) -> None:
    """Apply one friction triple to floor and both foot collision geoms."""
    if len(friction) != 3:
        raise ValueError("friction must contain exactly three values.")
    friction_array = np.array(friction, dtype=np.float64)
    for geom_name in (FLOOR_GEOM, RIGHT_FOOT_GEOM, LEFT_FOOT_GEOM):
        env.model.geom_friction[geom_id(env.model, geom_name)] = friction_array
    mujoco.mj_forward(env.model, env.data)


def _apply_base_height_delta(env: SedonStandingEnv, base_height_delta: float) -> None:
    """Shift the reset base height in-memory without changing the training scene."""
    if abs(base_height_delta) <= 0.0:
        return
    qpos = env.data.qpos.copy()
    qvel = np.zeros_like(env.data.qvel)
    qpos[2] += base_height_delta
    env.set_state(qpos, qvel)
    mujoco.mj_forward(env.model, env.data)


def _target_positions(env: SedonStandingEnv, support_side: str, support_roll: float) -> np.ndarray:
    """Return mirrored hip-roll load-transfer target positions."""
    target = env._nominal_joint_qpos.copy()
    if support_side == "left":
        target[HIP_ROLL_INDEX["right"]] += support_roll
        target[HIP_ROLL_INDEX["left"]] -= support_roll
    elif support_side == "right":
        target[HIP_ROLL_INDEX["right"]] -= support_roll
        target[HIP_ROLL_INDEX["left"]] += support_roll
    else:
        raise ValueError(f"Unsupported support side: {support_side}")
    return env._apply_safe_joint_target_clamps(target)


def _build_scenarios() -> list[SweepScenario]:
    """Return the default diagnostic scenario list."""
    return [
        SweepScenario(
            name="baseline",
            foot_size=BASELINE_FOOT_SIZE,
            friction=BASELINE_FRICTION,
            base_height_delta=0.0,
        ),
        SweepScenario(
            name="narrow_y",
            foot_size=(0.07, 0.02, 0.025),
            friction=BASELINE_FRICTION,
            base_height_delta=0.0,
        ),
        SweepScenario(
            name="short_x",
            foot_size=(0.04, 0.04, 0.025),
            friction=BASELINE_FRICTION,
            base_height_delta=0.0,
        ),
        SweepScenario(
            name="low_friction",
            foot_size=BASELINE_FOOT_SIZE,
            friction=(0.4, 0.002, 0.00005),
            base_height_delta=0.0,
        ),
        SweepScenario(
            name="higher_reset_0.01",
            foot_size=BASELINE_FOOT_SIZE,
            friction=BASELINE_FRICTION,
            base_height_delta=0.01,
        ),
        SweepScenario(
            name="higher_reset_0.02",
            foot_size=BASELINE_FOOT_SIZE,
            friction=BASELINE_FRICTION,
            base_height_delta=0.02,
        ),
    ]


def _write_csv(path: Path, rows: list[FootContactSweepResult]) -> None:
    """Write flat result rows to CSV."""
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
    parser.add_argument("--support-sides", default="left,right")
    parser.add_argument("--support-roll", type=float, default=0.10)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    return parser


def _run_case(
    scenario: SweepScenario,
    *,
    support_side: str,
    support_roll: float,
    steps: int,
    seed: int,
) -> FootContactSweepResult:
    """Run one scenario/support-side pair and return contact/geometry metrics."""
    base_config = load_sedon_config_from_env()
    reward_config = replace(
        base_config,
        target_base_height=base_config.target_base_height + scenario.base_height_delta,
    )
    env = SedonStandingEnv(reset_noise_scale=0.0, reward_config=reward_config)
    try:
        env.reset(seed=seed)
        if scenario.foot_size is not None:
            apply_foot_size_override(env.model, env.data, scenario.foot_size)
        if scenario.friction is not None:
            _apply_friction_override(env, scenario.friction)
        _apply_base_height_delta(env, scenario.base_height_delta)

        target_positions = _target_positions(env, support_side, support_roll)
        initial_base_z = float(env._base_height())
        initial_base_roll = _quat_to_roll(env.data.xquat[env._base_body_id])
        initial_com_y = float(_overall_com(env)[1])
        initial_foot_bottoms = env._foot_bottom_heights()
        initial_left_contact_count, _ = _foot_floor_load(env, "left")
        initial_right_contact_count, _ = _foot_floor_load(env, "right")
        initial_contact_state = _contact_state(env)

        support_joint_index = HIP_ROLL_INDEX[support_side]
        support_only_label = f"{support_side}_only"
        max_abs_com_y_delta = 0.0
        max_abs_base_roll_delta = 0.0
        max_abs_support_hip_roll_error = 0.0
        max_left_force_ratio = 0.0
        max_right_force_ratio = 0.0
        min_base_z = float("inf")
        min_upright = float("inf")
        terminated = False
        terminated_step = 0
        final_contact_state = initial_contact_state
        final_left_force_ratio = 0.0
        final_right_force_ratio = 0.0
        contact_counts = {"both": 0, "right_only": 0, "left_only": 0, "none": 0}

        for step_index in range(1, steps + 1):
            env._do_pd_simulation(target_positions)
            joint_positions = env._joint_positions()
            support_error = float(joint_positions[support_joint_index] - target_positions[support_joint_index])
            com_y_delta = float(_overall_com(env)[1] - initial_com_y)
            base_roll_delta = float(_quat_to_roll(env.data.xquat[env._base_body_id]) - initial_base_roll)
            max_abs_com_y_delta = max(max_abs_com_y_delta, abs(com_y_delta))
            max_abs_base_roll_delta = max(max_abs_base_roll_delta, abs(base_roll_delta))
            max_abs_support_hip_roll_error = max(max_abs_support_hip_roll_error, abs(support_error))

            _, left_force = _foot_floor_load(env, "left")
            _, right_force = _foot_floor_load(env, "right")
            total_force = max(left_force + right_force, 1e-9)
            final_left_force_ratio = left_force / total_force
            final_right_force_ratio = right_force / total_force
            max_left_force_ratio = max(max_left_force_ratio, final_left_force_ratio)
            max_right_force_ratio = max(max_right_force_ratio, final_right_force_ratio)

            final_contact_state = _contact_state(env)
            contact_counts[final_contact_state] += 1
            base_z = float(env._base_height())
            upright = float(env._base_upright())
            min_base_z = min(min_base_z, base_z)
            min_upright = min(min_upright, upright)
            obs = env._get_obs()
            terminated = bool(env._is_terminated(base_z, upright, obs))
            if terminated:
                terminated_step = step_index
                break

        if not terminated:
            terminated_step = steps

        final_joint_positions = env._joint_positions()
        return FootContactSweepResult(
            scenario=scenario.name,
            support_side=support_side,
            foot_size_x=float((scenario.foot_size or BASELINE_FOOT_SIZE)[0]),
            foot_size_y=float((scenario.foot_size or BASELINE_FOOT_SIZE)[1]),
            foot_size_z=float((scenario.foot_size or BASELINE_FOOT_SIZE)[2]),
            friction_tangent=float((scenario.friction or BASELINE_FRICTION)[0]),
            friction_torsional=float((scenario.friction or BASELINE_FRICTION)[1]),
            friction_rolling=float((scenario.friction or BASELINE_FRICTION)[2]),
            base_height_delta=float(scenario.base_height_delta),
            initial_base_z=initial_base_z,
            initial_left_foot_bottom_z=float(initial_foot_bottoms[1]),
            initial_right_foot_bottom_z=float(initial_foot_bottoms[0]),
            initial_left_contact_count=initial_left_contact_count,
            initial_right_contact_count=initial_right_contact_count,
            initial_contact_state=initial_contact_state,
            steps=steps,
            terminated=terminated,
            terminated_step=terminated_step,
            final_contact_state=final_contact_state,
            support_only_steps=contact_counts[support_only_label],
            both_contact_steps=contact_counts["both"],
            none_contact_steps=contact_counts["none"],
            max_abs_com_y_delta=max_abs_com_y_delta,
            max_abs_base_roll_delta=max_abs_base_roll_delta,
            final_com_y_delta=float(_overall_com(env)[1] - initial_com_y),
            final_base_roll_delta=float(
                _quat_to_roll(env.data.xquat[env._base_body_id]) - initial_base_roll
            ),
            final_left_force_ratio=final_left_force_ratio,
            final_right_force_ratio=final_right_force_ratio,
            max_left_force_ratio=max_left_force_ratio,
            max_right_force_ratio=max_right_force_ratio,
            max_abs_support_hip_roll_error=max_abs_support_hip_roll_error,
            final_support_hip_roll_error=float(
                final_joint_positions[support_joint_index] - target_positions[support_joint_index]
            ),
            min_base_z=min_base_z,
            min_upright=min_upright,
        )
    finally:
        env.close()


def main(argv: list[str] | None = None) -> int:
    """Run the foot contact/geometry sweep and print compact conclusions."""
    args = build_parser().parse_args(argv)
    if args.steps <= 0:
        raise ValueError("--steps must be positive.")
    support_sides = [part.strip() for part in args.support_sides.split(",") if part.strip()]
    if not support_sides:
        raise ValueError("--support-sides must contain at least one side.")
    for side in support_sides:
        if side not in {"left", "right"}:
            raise ValueError(f"Unsupported support side: {side}")

    scenarios = _build_scenarios()
    rows: list[FootContactSweepResult] = []
    print(
        "scenario side initLz initRz initLc initRc max_com_dy max_roll "
        "forceL forceR support_only both term_step contact"
    )
    for scenario in scenarios:
        for support_side in support_sides:
            row = _run_case(
                scenario,
                support_side=support_side,
                support_roll=args.support_roll,
                steps=args.steps,
                seed=args.seed,
            )
            rows.append(row)
            print(
                f"{row.scenario:>16} {row.support_side:>5} "
                f"{row.initial_left_foot_bottom_z:>7.4f} {row.initial_right_foot_bottom_z:>7.4f} "
                f"{row.initial_left_contact_count:>6} {row.initial_right_contact_count:>6} "
                f"{row.max_abs_com_y_delta:>10.4f} {row.max_abs_base_roll_delta:>8.4f} "
                f"{row.max_left_force_ratio:>6.3f} {row.max_right_force_ratio:>6.3f} "
                f"{row.support_only_steps:>12} {row.both_contact_steps:>4} "
                f"{row.terminated_step:>9} {row.final_contact_state:>10}"
            )

    _write_csv(args.out_csv, rows)
    print(f"\ncsv: {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
