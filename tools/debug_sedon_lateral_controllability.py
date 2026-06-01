"""Measure how strongly Sedon support-roll and unload targets move COM laterally."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sedon_baseline.env import SedonStandingEnv, load_sedon_config_from_env
from tools.sedon_debug_common import (
    DEFAULT_SCENE_PATH,
    DEBUG_OUT_DIR,
    RELAXED_FOOT_SIZE,
    apply_foot_size_override,
    contact_pairs,
    require_scene,
)


DEFAULT_OUT_CSV = DEBUG_OUT_DIR / "lateral_controllability.csv"


@dataclass(frozen=True)
class LateralControlResult:
    """Measured response for one support-side and unload-scale candidate."""

    support_side: str
    unload_scale: float
    steps: int
    terminated: bool
    min_base_z: float
    min_upright: float
    final_base_y_delta: float
    max_abs_base_y_delta: float
    final_com_y_delta: float
    max_abs_com_y_delta: float
    max_swing_height_delta: float
    support_contact_ratio: float
    swing_contact_ratio: float
    base_proxy_floor_steps: int
    contact_state_counts: dict[str, int]


def _overall_com(env: SedonStandingEnv) -> np.ndarray:
    """Return mass-weighted whole-model COM in world coordinates."""
    masses = env.model.body_mass
    total_mass = float(np.sum(masses))
    if total_mass <= 0.0:
        raise ValueError("Model has no positive body mass.")
    return np.sum(env.data.xipos * masses[:, None], axis=0) / total_mass


def _contact_flags(env: SedonStandingEnv) -> dict[str, bool]:
    """Return current foot/base floor contact flags."""
    flags = {"right": False, "left": False, "base_proxy": False}
    for name_a, name_b, _ in contact_pairs(env.model, env.data):
        pair = {name_a, name_b}
        if pair == {"floor", "R_foot_collision"}:
            flags["right"] = True
        elif pair == {"floor", "L_foot_collision"}:
            flags["left"] = True
        elif pair == {"floor", "base_proxy"}:
            flags["base_proxy"] = True
    return flags


def _contact_state(flags: dict[str, bool]) -> str:
    """Return compact foot-contact state label."""
    if flags["right"] and flags["left"]:
        return "both"
    if flags["right"]:
        return "right_only"
    if flags["left"]:
        return "left_only"
    return "none"


def _support_and_swing(flags: dict[str, bool], support_side: str) -> tuple[bool, bool]:
    """Return support/swing contact booleans for one side."""
    if support_side == "right":
        return flags["right"], flags["left"]
    if support_side == "left":
        return flags["left"], flags["right"]
    raise ValueError(f"Unsupported support side: {support_side}")


def _support_and_swing_heights(
    env: SedonStandingEnv,
    support_side: str,
) -> tuple[float, float]:
    """Return support and swing foot bottom heights."""
    foot_bottoms = env._foot_bottom_heights()
    if support_side == "right":
        return float(foot_bottoms[0]), float(foot_bottoms[1])
    if support_side == "left":
        return float(foot_bottoms[1]), float(foot_bottoms[0])
    raise ValueError(f"Unsupported support side: {support_side}")


def _parse_float_list(raw_value: str) -> list[float]:
    """Parse comma-separated floats."""
    values = [float(part.strip()) for part in raw_value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one float value.")
    return values


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ramp-steps", type=int, default=60)
    parser.add_argument("--hold-steps", type=int, default=160)
    parser.add_argument("--support-sides", default="right,left")
    parser.add_argument(
        "--unload-scales",
        type=_parse_float_list,
        default=[0.0, 0.5, 1.0],
        help="Comma-separated unload scales, for example '0,0.5,1.0'.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--scene-path", type=Path, default=DEFAULT_SCENE_PATH)
    parser.add_argument("--relaxed-foot", action="store_true")
    return parser


def _run_candidate(
    env: SedonStandingEnv,
    *,
    support_side: str,
    unload_scale: float,
    ramp_steps: int,
    hold_steps: int,
    seed: int,
) -> LateralControlResult:
    """Run one deterministic support-roll candidate and return response metrics."""
    env.reset(seed=seed)
    nominal = env._nominal_joint_qpos.copy()
    initial_base_y = float(env.data.qpos[1])
    initial_com_y = float(_overall_com(env)[1])
    support_pose = env._com_shift_support_pose(support_side, unload_scale=unload_scale)
    target = nominal + support_pose

    min_base_z = float("inf")
    min_upright = float("inf")
    max_abs_base_y_delta = 0.0
    max_abs_com_y_delta = 0.0
    max_swing_height_delta = 0.0
    support_contact_steps = 0
    swing_contact_steps = 0
    base_proxy_floor_steps = 0
    contact_state_counts: Counter[str] = Counter()
    terminated = False
    steps = 0
    final_base_y_delta = 0.0
    final_com_y_delta = 0.0

    for index in range(ramp_steps + hold_steps):
        alpha = min(1.0, (index + 1) / max(ramp_steps, 1))
        env._do_pd_simulation(nominal + (target - nominal) * alpha)
        obs = env._get_obs()
        base_z = env._base_height()
        upright = env._base_upright()
        terminated = env._is_terminated(base_z, upright, obs)
        flags = _contact_flags(env)
        support_contact, swing_contact = _support_and_swing(flags, support_side)
        support_height, swing_height = _support_and_swing_heights(env, support_side)
        base_y_delta = float(env.data.qpos[1]) - initial_base_y
        com_y_delta = float(_overall_com(env)[1]) - initial_com_y
        final_base_y_delta = base_y_delta
        final_com_y_delta = com_y_delta
        max_abs_base_y_delta = max(max_abs_base_y_delta, abs(base_y_delta))
        max_abs_com_y_delta = max(max_abs_com_y_delta, abs(com_y_delta))
        max_swing_height_delta = max(max_swing_height_delta, swing_height - support_height)
        min_base_z = min(min_base_z, base_z)
        min_upright = min(min_upright, upright)
        support_contact_steps += int(support_contact)
        swing_contact_steps += int(swing_contact)
        base_proxy_floor_steps += int(flags["base_proxy"])
        contact_state_counts.update([_contact_state(flags)])
        steps = index + 1
        if terminated:
            break

    step_count = max(steps, 1)
    return LateralControlResult(
        support_side=support_side,
        unload_scale=unload_scale,
        steps=steps,
        terminated=terminated,
        min_base_z=min_base_z,
        min_upright=min_upright,
        final_base_y_delta=final_base_y_delta,
        max_abs_base_y_delta=max_abs_base_y_delta,
        final_com_y_delta=final_com_y_delta,
        max_abs_com_y_delta=max_abs_com_y_delta,
        max_swing_height_delta=max_swing_height_delta,
        support_contact_ratio=support_contact_steps / step_count,
        swing_contact_ratio=swing_contact_steps / step_count,
        base_proxy_floor_steps=base_proxy_floor_steps,
        contact_state_counts={
            "both": contact_state_counts["both"],
            "right_only": contact_state_counts["right_only"],
            "left_only": contact_state_counts["left_only"],
            "none": contact_state_counts["none"],
        },
    )


def _write_csv(path: Path, results: list[LateralControlResult]) -> None:
    """Write sweep results to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for result in results:
        row = {
            "support_side": result.support_side,
            "unload_scale": result.unload_scale,
            "steps": result.steps,
            "terminated": result.terminated,
            "min_base_z": result.min_base_z,
            "min_upright": result.min_upright,
            "final_base_y_delta": result.final_base_y_delta,
            "max_abs_base_y_delta": result.max_abs_base_y_delta,
            "final_com_y_delta": result.final_com_y_delta,
            "max_abs_com_y_delta": result.max_abs_com_y_delta,
            "max_swing_height_delta": result.max_swing_height_delta,
            "support_contact_ratio": result.support_contact_ratio,
            "swing_contact_ratio": result.swing_contact_ratio,
            "base_proxy_floor_steps": result.base_proxy_floor_steps,
            "both_steps": result.contact_state_counts["both"],
            "right_only_steps": result.contact_state_counts["right_only"],
            "left_only_steps": result.contact_state_counts["left_only"],
            "none_steps": result.contact_state_counts["none"],
        }
        rows.append(row)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    """Run the lateral controllability sweep and print compact conclusions."""
    args = build_parser().parse_args(argv)
    if args.ramp_steps <= 0:
        raise ValueError("--ramp-steps must be positive.")
    if args.hold_steps <= 0:
        raise ValueError("--hold-steps must be positive.")
    support_sides = [part.strip() for part in args.support_sides.split(",") if part.strip()]
    if not support_sides:
        raise ValueError("--support-sides must contain at least one side.")
    for side in support_sides:
        if side not in {"right", "left"}:
            raise ValueError(f"Unsupported support side: {side}")

    reward_config = load_sedon_config_from_env()
    env = SedonStandingEnv(
        scene_path=require_scene(args.scene_path),
        reset_noise_scale=0.0,
        reward_config=reward_config,
    )
    results: list[LateralControlResult] = []
    try:
        if args.relaxed_foot:
            env.reset(seed=args.seed)
            apply_foot_size_override(env.model, env.data, RELAXED_FOOT_SIZE)
        for support_side in support_sides:
            for unload_scale in args.unload_scales:
                result = _run_candidate(
                    env,
                    support_side=support_side,
                    unload_scale=unload_scale,
                    ramp_steps=args.ramp_steps,
                    hold_steps=args.hold_steps,
                    seed=args.seed,
                )
                results.append(result)
    finally:
        env.close()

    _write_csv(args.out_csv, results)
    print(f"csv: {args.out_csv}")
    for result in sorted(
        results,
        key=lambda item: (item.max_abs_com_y_delta, item.max_swing_height_delta),
        reverse=True,
    ):
        print(
            f"{result.support_side:>5} unload={result.unload_scale:>4.2f} "
            f"steps={result.steps:>3} term={str(result.terminated):>5} "
            f"base_dy={result.max_abs_base_y_delta:>7.4f} "
            f"com_dy={result.max_abs_com_y_delta:>7.4f} "
            f"swing_dz={result.max_swing_height_delta:>7.4f} "
            f"support_contact={result.support_contact_ratio:>5.2f} "
            f"swing_contact={result.swing_contact_ratio:>5.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
