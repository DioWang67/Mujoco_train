"""Sweep Seedon mass distribution and stance width for lateral COM diagnostics.

This diagnostic keeps the same standing seed and the same deterministic
hip-roll shift target across all cases. It does not modify reward terms, PPO,
or the committed training scene on disk.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

import mujoco
import numpy as np

from seedon_baseline.env import SeedonStandingEnv, load_seedon_config_from_env
from tools.seedon_debug_common import (
    BASE_PROXY_GEOM,
    DEBUG_OUT_DIR,
    FLOOR_GEOM,
    LEFT_FOOT_GEOM,
    RIGHT_FOOT_GEOM,
    contact_pairs,
)


DEFAULT_OUT_CSV = DEBUG_OUT_DIR / "com_mass_distribution_sweep.csv"
HIP_ROLL_INDEX = {"right": 1, "left": 6}
TORSO_BODY_NAMES = ("base_link",)
PELVIS_BODY_NAMES = (
    "R_link_hip_yaw",
    "R_link_hip_roll",
    "L_link_hip_yaw",
    "L_link_hip_roll",
)
LEG_BODY_NAMES = (
    "R_link_hip_pitch",
    "R_link_knee_pitch",
    "R_link_ankle_pitch",
    "L_link_hip_pitch",
    "L_link_knee_pitch",
    "L_link_ankle_pitch",
)
STANCE_ROOT_BODIES = {
    "right": "R_link_hip_yaw",
    "left": "L_link_hip_yaw",
}


@dataclass(frozen=True)
class SweepCase:
    """One mass/stance override candidate."""

    torso_mass_scale: float
    pelvis_mass_scale: float
    leg_mass_scale: float
    stance_width_scale: float

    @property
    def case_name(self) -> str:
        """Return a compact stable case label."""
        return (
            f"torso_{self.torso_mass_scale:.2f}"
            f"__pelvis_{self.pelvis_mass_scale:.2f}"
            f"__leg_{self.leg_mass_scale:.2f}"
            f"__stance_{self.stance_width_scale:.2f}"
        )


@dataclass(frozen=True)
class SweepResult:
    """One COM/mass-distribution sweep row."""

    case_name: str
    torso_mass_scale: float
    pelvis_mass_scale: float
    leg_mass_scale: float
    stance_width_scale: float
    initial_com_x: float
    initial_com_y: float
    initial_com_z: float
    max_abs_com_y_delta: float
    mean_com_y_delta_last_50: float
    left_force_ratio: float
    right_force_ratio: float
    support_only_steps: int
    both_contact_ratio: float
    none_contact_ratio: float
    terminated: bool
    terminated_step: int
    score: float


def _parse_float_list(raw_value: str) -> list[float]:
    """Parse a comma-separated float list."""
    values = [float(part.strip()) for part in raw_value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one float.")
    return values


def _overall_com(env: SeedonStandingEnv) -> np.ndarray:
    """Return the whole-body COM in world coordinates."""
    masses = env.model.body_mass
    total_mass = float(np.sum(masses))
    if total_mass <= 0.0:
        raise ValueError("Model has no positive body mass.")
    return np.sum(env.data.xipos * masses[:, None], axis=0) / total_mass


def _quat_to_roll(quat: np.ndarray) -> float:
    """Return base roll angle from a MuJoCo quaternion."""
    w, x, y, z = [float(value) for value in quat]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    return float(math.atan2(sinr_cosp, cosr_cosp))


def _body_ids(env: SeedonStandingEnv, body_names: tuple[str, ...]) -> list[int]:
    """Resolve body names into MuJoCo body ids."""
    return [env._body_id(name) for name in body_names]


def _foot_floor_load(env: SeedonStandingEnv, side: str) -> tuple[int, float]:
    """Return floor-contact count and normal-force sum for one foot."""
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


def _contact_state(env: SeedonStandingEnv) -> tuple[str, bool]:
    """Return compact floor-contact state plus base-proxy contact flag."""
    right = False
    left = False
    base = False
    for name_a, name_b, _ in contact_pairs(env.model, env.data):
        pair = {name_a, name_b}
        if pair == {FLOOR_GEOM, RIGHT_FOOT_GEOM}:
            right = True
        elif pair == {FLOOR_GEOM, LEFT_FOOT_GEOM}:
            left = True
        elif pair == {FLOOR_GEOM, BASE_PROXY_GEOM}:
            base = True
    if right and left:
        return "both", base
    if left:
        return "left_only", base
    if right:
        return "right_only", base
    return "none", base


def _apply_mass_scale(
    env: SeedonStandingEnv,
    body_ids: list[int],
    scale: float,
    original_mass: np.ndarray,
    original_inertia: np.ndarray,
) -> None:
    """Scale mass and diagonal inertia together for one body group."""
    for body_id in body_ids:
        env.model.body_mass[body_id] = original_mass[body_id] * scale
        env.model.body_inertia[body_id] = original_inertia[body_id] * scale


def _apply_stance_width_scale(env: SeedonStandingEnv, scale: float, original_body_pos: np.ndarray) -> None:
    """Scale the lateral offset of the two hip-yaw root bodies."""
    for side, body_name in STANCE_ROOT_BODIES.items():
        body_id = env._body_id(body_name)
        env.model.body_pos[body_id] = original_body_pos[body_id]
        direction = -1.0 if side == "right" else 1.0
        env.model.body_pos[body_id][1] = direction * abs(float(original_body_pos[body_id][1])) * scale


def _left_support_target(env: SeedonStandingEnv, hip_roll_offset: float) -> np.ndarray:
    """Return the shared left-support hip-roll shift target."""
    target = env._nominal_joint_qpos.copy()
    target[HIP_ROLL_INDEX["right"]] += hip_roll_offset
    target[HIP_ROLL_INDEX["left"]] -= hip_roll_offset
    return env._apply_safe_joint_target_clamps(target)


def _score_result(
    *,
    max_abs_com_y_delta: float,
    mean_com_y_delta_last_50: float,
    left_force_ratio: float,
    both_contact_ratio: float,
    none_contact_ratio: float,
    terminated: bool,
) -> float:
    """Return a ranking score that favors stable large lateral COM shifts."""
    return (
        max_abs_com_y_delta * 1000.0
        + mean_com_y_delta_last_50 * 1200.0
        + left_force_ratio * 40.0
        + both_contact_ratio * 20.0
        - none_contact_ratio * 30.0
        - (80.0 if terminated else 0.0)
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torso-mass-scales", default="0.5,0.75,1.0,1.25")
    parser.add_argument("--pelvis-mass-scales", default="0.5,0.75,1.0,1.25")
    parser.add_argument("--leg-mass-scales", default="0.5,0.75,1.0")
    parser.add_argument("--stance-width-scales", default="0.7,0.85,1.0")
    parser.add_argument("--settle-steps", type=int, default=20)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hip-roll-offset", type=float, default=0.10)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    return parser


def _iter_cases(args: argparse.Namespace) -> list[SweepCase]:
    """Expand the CLI grids into concrete sweep cases."""
    torso_mass_scales = _parse_float_list(args.torso_mass_scales)
    pelvis_mass_scales = _parse_float_list(args.pelvis_mass_scales)
    leg_mass_scales = _parse_float_list(args.leg_mass_scales)
    stance_width_scales = _parse_float_list(args.stance_width_scales)
    return [
        SweepCase(
            torso_mass_scale=torso_mass_scale,
            pelvis_mass_scale=pelvis_mass_scale,
            leg_mass_scale=leg_mass_scale,
            stance_width_scale=stance_width_scale,
        )
        for torso_mass_scale, pelvis_mass_scale, leg_mass_scale, stance_width_scale in product(
            torso_mass_scales,
            pelvis_mass_scales,
            leg_mass_scales,
            stance_width_scales,
        )
    ]


def _run_case(
    case: SweepCase,
    *,
    settle_steps: int,
    steps: int,
    seed: int,
    hip_roll_offset: float,
) -> SweepResult:
    """Run one COM/mass-distribution diagnostic case."""
    reward_config = load_seedon_config_from_env()
    env = SeedonStandingEnv(reset_noise_scale=0.0, reward_config=reward_config)
    try:
        torso_ids = _body_ids(env, TORSO_BODY_NAMES)
        pelvis_ids = _body_ids(env, PELVIS_BODY_NAMES)
        leg_ids = _body_ids(env, LEG_BODY_NAMES)
        original_mass = env.model.body_mass.copy()
        original_inertia = env.model.body_inertia.copy()
        original_body_pos = env.model.body_pos.copy()

        _apply_mass_scale(env, torso_ids, case.torso_mass_scale, original_mass, original_inertia)
        _apply_mass_scale(env, pelvis_ids, case.pelvis_mass_scale, original_mass, original_inertia)
        _apply_mass_scale(env, leg_ids, case.leg_mass_scale, original_mass, original_inertia)
        _apply_stance_width_scale(env, case.stance_width_scale, original_body_pos)

        env.reset(seed=seed)
        mujoco.mj_forward(env.model, env.data)

        nominal_target = env._apply_safe_joint_target_clamps(env._nominal_joint_qpos.copy())
        for _ in range(settle_steps):
            env._do_pd_simulation(nominal_target)

        initial_com = _overall_com(env)
        target = _left_support_target(env, hip_roll_offset)
        com_y_deltas: list[float] = []
        left_force_ratios: list[float] = []
        right_force_ratios: list[float] = []
        support_only_steps = 0
        both_steps = 0
        none_steps = 0
        terminated = False
        terminated_step = 0
        min_base_z = float("inf")
        min_upright = float("inf")
        max_abs_base_roll = 0.0

        for step_index in range(1, steps + 1):
            env._do_pd_simulation(target)
            com_y_delta = float(_overall_com(env)[1] - initial_com[1])
            com_y_deltas.append(com_y_delta)

            _, left_force = _foot_floor_load(env, "left")
            _, right_force = _foot_floor_load(env, "right")
            total_force = left_force + right_force
            if total_force > 1e-9:
                left_force_ratio = left_force / total_force
                right_force_ratio = right_force / total_force
            else:
                left_force_ratio = 0.0
                right_force_ratio = 0.0
            left_force_ratios.append(left_force_ratio)
            right_force_ratios.append(right_force_ratio)

            contact_state, base_proxy_contact = _contact_state(env)
            if contact_state == "left_only":
                support_only_steps += 1
            elif contact_state == "both":
                both_steps += 1
            elif contact_state == "none":
                none_steps += 1
            if base_proxy_contact:
                none_steps += 0

            base_z = float(env._base_height())
            upright = float(env._base_upright())
            min_base_z = min(min_base_z, base_z)
            min_upright = min(min_upright, upright)
            max_abs_base_roll = max(
                max_abs_base_roll,
                abs(_quat_to_roll(env.data.xquat[env._base_body_id])),
            )

            observation = env._get_obs()
            terminated = env._is_terminated(base_z, upright, observation)
            if terminated:
                terminated_step = step_index
                break

        if not terminated:
            terminated_step = steps

        executed_steps = max(len(com_y_deltas), 1)
        tail = com_y_deltas[-50:] if len(com_y_deltas) >= 50 else com_y_deltas
        left_force_tail = left_force_ratios[-50:] if len(left_force_ratios) >= 50 else left_force_ratios
        right_force_tail = right_force_ratios[-50:] if len(right_force_ratios) >= 50 else right_force_ratios
        max_abs_com_y_delta = max(abs(value) for value in com_y_deltas) if com_y_deltas else 0.0
        mean_com_y_delta_last_50 = float(np.mean(tail)) if tail else 0.0
        left_force_ratio = float(np.mean(left_force_tail)) if left_force_tail else 0.0
        right_force_ratio = float(np.mean(right_force_tail)) if right_force_tail else 0.0
        both_contact_ratio = both_steps / executed_steps
        none_contact_ratio = none_steps / executed_steps
        score = _score_result(
            max_abs_com_y_delta=max_abs_com_y_delta,
            mean_com_y_delta_last_50=mean_com_y_delta_last_50,
            left_force_ratio=left_force_ratio,
            both_contact_ratio=both_contact_ratio,
            none_contact_ratio=none_contact_ratio,
            terminated=terminated,
        )

        return SweepResult(
            case_name=case.case_name,
            torso_mass_scale=case.torso_mass_scale,
            pelvis_mass_scale=case.pelvis_mass_scale,
            leg_mass_scale=case.leg_mass_scale,
            stance_width_scale=case.stance_width_scale,
            initial_com_x=float(initial_com[0]),
            initial_com_y=float(initial_com[1]),
            initial_com_z=float(initial_com[2]),
            max_abs_com_y_delta=float(max_abs_com_y_delta),
            mean_com_y_delta_last_50=float(mean_com_y_delta_last_50),
            left_force_ratio=float(left_force_ratio),
            right_force_ratio=float(right_force_ratio),
            support_only_steps=int(support_only_steps),
            both_contact_ratio=float(both_contact_ratio),
            none_contact_ratio=float(none_contact_ratio),
            terminated=bool(terminated),
            terminated_step=int(terminated_step),
            score=float(score),
        )
    finally:
        env.close()


def _write_csv(path: Path, rows: list[SweepResult]) -> None:
    """Write sweep rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows([asdict(row) for row in rows])


def main(argv: list[str] | None = None) -> int:
    """Run the Seedon COM/mass-distribution sweep."""
    args = build_parser().parse_args(argv)
    if args.settle_steps < 0:
        raise ValueError("--settle-steps must be non-negative.")
    if args.steps <= 0:
        raise ValueError("--steps must be positive.")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive.")

    cases = _iter_cases(args)
    results = [
        _run_case(
            case,
            settle_steps=args.settle_steps,
            steps=args.steps,
            seed=args.seed,
            hip_roll_offset=args.hip_roll_offset,
        )
        for case in cases
    ]
    _write_csv(args.out_csv, results)

    ranked = sorted(
        results,
        key=lambda row: (
            row.score,
            not row.terminated,
            row.max_abs_com_y_delta,
            row.mean_com_y_delta_last_50,
        ),
        reverse=True,
    )

    print(f"cases: {len(results)}")
    print(
        "rank case max_com_dy mean_last50 left_ratio both_ratio "
        "none_ratio terminated term_step score"
    )
    for index, row in enumerate(ranked[: args.top_k], start=1):
        print(
            f"{index:>4} {row.case_name:>44} {row.max_abs_com_y_delta:>10.4f} "
            f"{row.mean_com_y_delta_last_50:>11.4f} {row.left_force_ratio:>10.3f} "
            f"{row.both_contact_ratio:>10.3f} {row.none_contact_ratio:>10.3f} "
            f"{str(row.terminated):>10} {row.terminated_step:>9} {row.score:>8.2f}"
        )

    print(f"\ncsv: {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
