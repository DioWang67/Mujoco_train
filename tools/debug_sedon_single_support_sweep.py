"""Sweep single-support hold candidates for the Sedon MuJoCo model.

This diagnostic does not train PPO and does not change reward/gait logic.

Example:
    python -m tools.debug_sedon_single_support_sweep --relaxed-foot
"""

from __future__ import annotations

import argparse
import csv
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import mujoco
import numpy as np

from sedon_baseline.env import SedonStandingConfig, SedonStandingEnv
from tools.sedon_debug_common import DEBUG_OUT_DIR, RELAXED_FOOT_SIZE


DEFAULT_OUT_CSV = DEBUG_OUT_DIR / "single_support_hold_sweep.csv"

FOOT_INDEX_BY_LEG = {"right": 0, "left": 1}
SUPPORT_LEG_BY_LIFTED_LEG = {"right": "left", "left": "right"}

LEG_JOINT_INDEX = {
    "right": {
        "hip_roll": 1,
        "hip_pitch": 2,
        "knee_pitch": 3,
        "ankle_pitch": 4,
    },
    "left": {
        "hip_roll": 6,
        "hip_pitch": 7,
        "knee_pitch": 8,
        "ankle_pitch": 9,
    },
}

# From kinematic foot-height map.
BASE_SWING_DELTAS = {
    "hip_pitch": 0.30,
    "knee_pitch": -0.30,
    "ankle_pitch": -0.20,
}


@dataclass(frozen=True)
class SweepCandidate:
    lifted_leg: str
    support_roll: float
    lift_scale: float
    lift_steps: int
    hold_steps: int
    invert_support_roll: bool


@dataclass(frozen=True)
class SweepResult:
    lifted_leg: str
    support_leg: str
    support_roll: float
    lift_scale: float
    lift_steps: int
    hold_steps: int
    invert_support_roll: bool
    total_steps: int
    terminated: bool
    pass_candidate: bool
    max_lifted_foot_bottom_z: float
    min_lifted_foot_bottom_z: float
    support_foot_min_bottom_z: float
    support_foot_max_bottom_z: float
    single_support_steps: int
    both_contact_steps: int
    no_contact_steps: int
    min_base_z: float
    min_upright: float
    max_forward_velocity: float
    base_proxy_floor_steps: int
    score: float


def _parse_float_list(raw_value: str) -> list[float]:
    values = [float(item.strip()) for item in raw_value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one float value.")
    return values


def _parse_int_list(raw_value: str) -> list[int]:
    values = [int(item.strip()) for item in raw_value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one integer value.")
    return values


def _smoothstep(value: float) -> float:
    t = float(np.clip(value, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def _apply_relaxed_foot_if_requested(env: SedonStandingEnv, relaxed_foot: bool) -> None:
    """Apply relaxed foot size override directly on MuJoCo model."""
    if not relaxed_foot:
        return

    for geom_name in ("R_foot_collision", "L_foot_collision"):
        geom_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if geom_id < 0:
            raise ValueError(f"Geom not found: {geom_name}")

        env.model.geom_size[geom_id][:] = np.asarray(
            RELAXED_FOOT_SIZE,
            dtype=np.float64,
        )

    mujoco.mj_forward(env.model, env.data)


def _target_from_candidate(
    env: SedonStandingEnv,
    candidate: SweepCandidate,
    *,
    lift_progress: float,
    hold: bool,
) -> np.ndarray:
    targets = env._nominal_joint_qpos.copy()
    lifted_leg = candidate.lifted_leg
    support_leg = SUPPORT_LEG_BY_LIFTED_LEG[lifted_leg]
    ramp = 1.0 if hold else _smoothstep(lift_progress)

    support_sign = -1.0 if support_leg == "right" else 1.0
    if candidate.invert_support_roll:
        support_sign *= -1.0

    support_roll_index = LEG_JOINT_INDEX[support_leg]["hip_roll"]
    targets[support_roll_index] += support_sign * candidate.support_roll * ramp

    for joint_name, base_delta in BASE_SWING_DELTAS.items():
        joint_index = LEG_JOINT_INDEX[lifted_leg][joint_name]
        targets[joint_index] += base_delta * candidate.lift_scale * ramp

    return targets


def _step_with_target(env: SedonStandingEnv, target_positions: np.ndarray) -> None:
    for _ in range(env.frame_skip):
        env.data.ctrl[:] = env._pd_control(target_positions)
        mujoco.mj_step(env.model, env.data)


def _contact_flags(env: SedonStandingEnv, lifted_leg: str) -> tuple[bool, bool, int]:
    foot_bottoms = env._foot_bottom_heights()
    lifted_index = FOOT_INDEX_BY_LEG[lifted_leg]
    support_index = 1 - lifted_index

    lifted_contact = bool(abs(float(foot_bottoms[lifted_index])) <= 0.003)
    support_contact = bool(abs(float(foot_bottoms[support_index])) <= 0.003)

    base_proxy_floor_contact = 0
    for contact_index in range(env.data.ncon):
        contact = env.data.contact[contact_index]
        geom1_name = mujoco.mj_id2name(
            env.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            contact.geom1,
        )
        geom2_name = mujoco.mj_id2name(
            env.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            contact.geom2,
        )
        pair = {geom1_name, geom2_name}
        if "base_proxy" in pair and "floor" in pair:
            base_proxy_floor_contact = 1
            break

    return lifted_contact, support_contact, base_proxy_floor_contact


def _run_candidate(
    candidate: SweepCandidate,
    *,
    relaxed_foot: bool,
    clearance_threshold: float,
    min_single_support_steps: int,
    min_base_z_threshold: float,
    min_upright_threshold: float,
    max_forward_velocity_threshold: float,
) -> SweepResult:
    env = SedonStandingEnv(
        reset_noise_scale=0.0,
        reward_config=SedonStandingConfig(gait_mode="fsm"),
    )

    try:
        env.reset(seed=42)
        _apply_relaxed_foot_if_requested(env, relaxed_foot)

        total_plan_steps = candidate.lift_steps + candidate.hold_steps

        max_lifted_foot_bottom_z = -1e9
        min_lifted_foot_bottom_z = 1e9
        support_foot_min_bottom_z = 1e9
        support_foot_max_bottom_z = -1e9
        min_base_z = 1e9
        min_upright = 1e9
        max_forward_velocity = 0.0

        single_support_steps = 0
        both_contact_steps = 0
        no_contact_steps = 0
        base_proxy_floor_steps = 0

        terminated = False
        total_steps = 0

        lifted_index = FOOT_INDEX_BY_LEG[candidate.lifted_leg]
        support_index = 1 - lifted_index

        for step_index in range(total_plan_steps):
            hold = step_index >= candidate.lift_steps
            lift_progress = min(1.0, (step_index + 1) / max(1, candidate.lift_steps))

            target_positions = _target_from_candidate(
                env,
                candidate,
                lift_progress=lift_progress,
                hold=hold,
            )

            _step_with_target(env, target_positions)
            total_steps = step_index + 1

            foot_bottoms = env._foot_bottom_heights()
            lifted_z = float(foot_bottoms[lifted_index])
            support_z = float(foot_bottoms[support_index])

            max_lifted_foot_bottom_z = max(max_lifted_foot_bottom_z, lifted_z)
            min_lifted_foot_bottom_z = min(min_lifted_foot_bottom_z, lifted_z)
            support_foot_min_bottom_z = min(support_foot_min_bottom_z, support_z)
            support_foot_max_bottom_z = max(support_foot_max_bottom_z, support_z)

            base_z = env._base_height()
            upright = env._base_upright()

            min_base_z = min(min_base_z, base_z)
            min_upright = min(min_upright, upright)
            max_forward_velocity = max(max_forward_velocity, abs(float(env.data.qvel[0])))

            lifted_contact, support_contact, base_proxy_contact = _contact_flags(
                env,
                candidate.lifted_leg,
            )
            base_proxy_floor_steps += base_proxy_contact

            if support_contact and not lifted_contact:
                single_support_steps += 1
            elif support_contact and lifted_contact:
                both_contact_steps += 1
            elif not support_contact and not lifted_contact:
                no_contact_steps += 1

            obs = env._get_obs()
            if env._is_terminated(base_z, upright, obs):
                terminated = True
                break

        pass_candidate = (
            max_lifted_foot_bottom_z > clearance_threshold
            and single_support_steps >= min_single_support_steps
            and min_base_z > min_base_z_threshold
            and min_upright > min_upright_threshold
            and max_forward_velocity < max_forward_velocity_threshold
            and base_proxy_floor_steps == 0
            and not terminated
        )

        score = (
            max_lifted_foot_bottom_z * 100.0
            + single_support_steps * 0.2
            + min_upright * 10.0
            + min_base_z * 5.0
            - base_proxy_floor_steps * 10.0
            - (50.0 if terminated else 0.0)
            - max(0.0, max_forward_velocity - max_forward_velocity_threshold) * 20.0
        )

        return SweepResult(
            lifted_leg=candidate.lifted_leg,
            support_leg=SUPPORT_LEG_BY_LIFTED_LEG[candidate.lifted_leg],
            support_roll=candidate.support_roll,
            lift_scale=candidate.lift_scale,
            lift_steps=candidate.lift_steps,
            hold_steps=candidate.hold_steps,
            invert_support_roll=candidate.invert_support_roll,
            total_steps=total_steps,
            terminated=terminated,
            pass_candidate=pass_candidate,
            max_lifted_foot_bottom_z=max_lifted_foot_bottom_z,
            min_lifted_foot_bottom_z=min_lifted_foot_bottom_z,
            support_foot_min_bottom_z=support_foot_min_bottom_z,
            support_foot_max_bottom_z=support_foot_max_bottom_z,
            single_support_steps=single_support_steps,
            both_contact_steps=both_contact_steps,
            no_contact_steps=no_contact_steps,
            min_base_z=min_base_z,
            min_upright=min_upright,
            max_forward_velocity=max_forward_velocity,
            base_proxy_floor_steps=base_proxy_floor_steps,
            score=score,
        )

    finally:
        env.close()


def _build_candidates(
    *,
    support_roll_values: Iterable[float],
    lift_scale_values: Iterable[float],
    lift_step_values: Iterable[int],
    hold_step_values: Iterable[int],
    include_inverted_support_roll: bool,
) -> list[SweepCandidate]:
    candidates: list[SweepCandidate] = []

    invert_values = [False, True] if include_inverted_support_roll else [False]

    for lifted_leg, support_roll, lift_scale, lift_steps, hold_steps, invert_roll in itertools.product(
        ("right", "left"),
        support_roll_values,
        lift_scale_values,
        lift_step_values,
        hold_step_values,
        invert_values,
    ):
        candidates.append(
            SweepCandidate(
                lifted_leg=lifted_leg,
                support_roll=float(support_roll),
                lift_scale=float(lift_scale),
                lift_steps=int(lift_steps),
                hold_steps=int(hold_steps),
                invert_support_roll=bool(invert_roll),
            )
        )

    return candidates


def _write_csv(path: Path, rows: list[SweepResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(SweepResult.__dataclass_fields__.keys())

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({field: getattr(row, field) for field in fieldnames})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--relaxed-foot", action="store_true")
    parser.add_argument("--include-inverted-support-roll", action="store_true")

    parser.add_argument(
        "--support-roll-values",
        type=_parse_float_list,
        default=[0.04, 0.06, 0.08, 0.10, 0.12],
    )
    parser.add_argument(
        "--lift-scale-values",
        type=_parse_float_list,
        default=[0.3, 0.4, 0.5, 0.6, 0.8],
    )
    parser.add_argument(
        "--lift-step-values",
        type=_parse_int_list,
        default=[120, 160, 200],
    )
    parser.add_argument(
        "--hold-step-values",
        type=_parse_int_list,
        default=[20, 40, 60],
    )

    parser.add_argument("--clearance-threshold", type=float, default=0.005)
    parser.add_argument("--min-single-support-steps", type=int, default=20)
    parser.add_argument("--min-base-z", type=float, default=0.37)
    parser.add_argument("--min-upright", type=float, default=0.85)
    parser.add_argument("--max-forward-velocity", type=float, default=0.30)

    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.top <= 0:
        raise ValueError("--top must be positive.")

    candidates = _build_candidates(
        support_roll_values=args.support_roll_values,
        lift_scale_values=args.lift_scale_values,
        lift_step_values=args.lift_step_values,
        hold_step_values=args.hold_step_values,
        include_inverted_support_roll=args.include_inverted_support_roll,
    )

    print(f"candidates: {len(candidates)}")

    results: list[SweepResult] = []

    for index, candidate in enumerate(candidates, start=1):
        result = _run_candidate(
            candidate,
            relaxed_foot=args.relaxed_foot,
            clearance_threshold=args.clearance_threshold,
            min_single_support_steps=args.min_single_support_steps,
            min_base_z_threshold=args.min_base_z,
            min_upright_threshold=args.min_upright,
            max_forward_velocity_threshold=args.max_forward_velocity,
        )
        results.append(result)

        print(
            f"[{index:>4}/{len(candidates)}] "
            f"leg={result.lifted_leg:<5} "
            f"roll={result.support_roll:.3f} "
            f"scale={result.lift_scale:.2f} "
            f"lift={result.lift_steps:<3} "
            f"hold={result.hold_steps:<3} "
            f"inv_roll={result.invert_support_roll} "
            f"pass={result.pass_candidate} "
            f"z={result.max_lifted_foot_bottom_z:.4f} "
            f"single={result.single_support_steps:<3} "
            f"base_z={result.min_base_z:.3f} "
            f"up={result.min_upright:.3f} "
            f"term={result.terminated}"
        )

    results.sort(key=lambda item: (item.pass_candidate, item.score), reverse=True)
    _write_csv(args.out_csv, results)

    print(f"\nCSV: {args.out_csv}")
    print(
        "rank pass leg support roll scale lift hold inv_roll "
        "z single base_z upright max_fwd term score"
    )

    for rank, result in enumerate(results[: args.top], start=1):
        print(
            f"{rank:>4} "
            f"{str(result.pass_candidate):>5} "
            f"{result.lifted_leg:>5} "
            f"{result.support_leg:>7} "
            f"{result.support_roll:>5.3f} "
            f"{result.lift_scale:>5.2f} "
            f"{result.lift_steps:>4} "
            f"{result.hold_steps:>4} "
            f"{str(result.invert_support_roll):>8} "
            f"{result.max_lifted_foot_bottom_z:>7.4f} "
            f"{result.single_support_steps:>6} "
            f"{result.min_base_z:>6.3f} "
            f"{result.min_upright:>7.3f} "
            f"{result.max_forward_velocity:>7.3f} "
            f"{str(result.terminated):>5} "
            f"{result.score:>7.2f}"
        )

    pass_count = sum(1 for result in results if result.pass_candidate)
    print(f"\npass_count: {pass_count}/{len(results)}")

    if pass_count == 0:
        print(
            "No passing single-support candidate found. "
            "Try --include-inverted-support-roll or relax thresholds for diagnosis."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())