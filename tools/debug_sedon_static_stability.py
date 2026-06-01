"""Check Sedon COM placement and static single-foot support stability."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from sedon_baseline.env import JOINT_NAMES, SedonStandingConfig, SedonStandingEnv
from tools.sedon_debug_common import (
    BASE_PROXY_GEOM,
    DEFAULT_SCENE_PATH,
    LEFT_FOOT_GEOM,
    RELAXED_FOOT_SIZE,
    RIGHT_FOOT_GEOM,
    apply_foot_size_override,
    contact_pairs,
    geom_id,
    require_scene,
    snapshot_geom,
)


@dataclass(frozen=True)
class SupportBox:
    """Projected support rectangle for one foot collision box."""

    name: str
    center_xy: np.ndarray
    half_extents: np.ndarray
    x_axis_xy: np.ndarray
    y_axis_xy: np.ndarray


@dataclass(frozen=True)
class PoseResult:
    """Static pose rollout metrics."""

    name: str
    terminated: bool
    steps: int
    min_base_z: float
    min_upright: float
    final_com: np.ndarray
    support_foot: str
    com_relative_xy: np.ndarray
    com_inside_support: bool
    contact_state_counts: dict[str, int]
    base_proxy_floor_steps: int
    max_left_foot_bottom_z: float
    max_right_foot_bottom_z: float


@dataclass(frozen=True)
class ValidationCase:
    """One static stability validation configuration."""

    scenario: str
    foot_size: tuple[float, float, float]
    support_roll: float


def _format_vec(values: np.ndarray) -> str:
    """Format numeric vectors for compact console output."""
    return " ".join(f"{float(value): .5f}" for value in values)


def _overall_com(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Return mass-weighted whole-model COM in world coordinates."""
    masses = model.body_mass
    total_mass = float(np.sum(masses))
    if total_mass <= 0.0:
        raise ValueError("Model has no positive body mass.")
    return np.sum(data.xipos * masses[:, None], axis=0) / total_mass


def _support_box(env: SedonStandingEnv, geom_name: str) -> SupportBox:
    """Return a foot support rectangle projected onto world XY."""
    resolved_id = geom_id(env.model, geom_name)
    xmat = env.data.geom_xmat[resolved_id].reshape(3, 3)
    x_axis_xy = xmat[:, 0][:2]
    y_axis_xy = xmat[:, 1][:2]
    x_norm = float(np.linalg.norm(x_axis_xy))
    y_norm = float(np.linalg.norm(y_axis_xy))
    if x_norm > 1e-9:
        x_axis_xy = x_axis_xy / x_norm
    if y_norm > 1e-9:
        y_axis_xy = y_axis_xy / y_norm
    return SupportBox(
        name=geom_name,
        center_xy=env.data.geom_xpos[resolved_id][:2].copy(),
        half_extents=env.model.geom_size[resolved_id][:2].copy(),
        x_axis_xy=x_axis_xy,
        y_axis_xy=y_axis_xy,
    )


def _point_in_support(point_xy: np.ndarray, box: SupportBox, margin: float = 0.0) -> bool:
    """Return whether a world XY point lies inside a projected foot box."""
    rel = point_xy - box.center_xy
    local_x = float(np.dot(rel, box.x_axis_xy))
    local_y = float(np.dot(rel, box.y_axis_xy))
    return (
        abs(local_x) <= float(box.half_extents[0]) + margin
        and abs(local_y) <= float(box.half_extents[1]) + margin
    )


def _point_in_combined_support(point_xy: np.ndarray, boxes: list[SupportBox]) -> bool:
    """Return whether point lies in the axis-aligned hull of both support boxes."""
    corners: list[np.ndarray] = []
    for box in boxes:
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                corners.append(
                    box.center_xy
                    + sx * box.half_extents[0] * box.x_axis_xy
                    + sy * box.half_extents[1] * box.y_axis_xy
                )
    all_corners = np.vstack(corners)
    mins = np.min(all_corners, axis=0)
    maxs = np.max(all_corners, axis=0)
    return bool(np.all(point_xy >= mins) and np.all(point_xy <= maxs))


def _contact_state(env: SedonStandingEnv) -> tuple[str, bool]:
    """Return current foot contact state and base-proxy-floor flag."""
    pairs = [set((name_a, name_b)) for name_a, name_b, _ in contact_pairs(env.model, env.data)]
    right = {"floor", RIGHT_FOOT_GEOM} in pairs
    left = {"floor", LEFT_FOOT_GEOM} in pairs
    base = {"floor", BASE_PROXY_GEOM} in pairs
    if right and left:
        return "both", base
    if right:
        return "right_only", base
    if left:
        return "left_only", base
    return "none", base


def _set_foot_sizes(env: SedonStandingEnv, foot_size: tuple[float, float, float]) -> None:
    """Apply a temporary foot collision size to both foot boxes."""
    apply_foot_size_override(env.model, env.data, foot_size)


def _joint_target_for_pose(
    env: SedonStandingEnv,
    *,
    pose_name: str,
    support_roll: float,
    hip_lift: float,
    knee_lift: float,
    ankle_lift: float,
) -> tuple[np.ndarray, str]:
    """Return joint target and support foot name for a static single-leg pose."""
    target = env._nominal_joint_qpos.copy()
    if pose_name == "left_support_right_lift":
        support_foot = LEFT_FOOT_GEOM
        target[1] = support_roll
        target[6] = -support_roll
        target[2] += hip_lift
        target[3] += knee_lift
        target[4] += ankle_lift
    elif pose_name == "right_support_left_lift":
        support_foot = RIGHT_FOOT_GEOM
        target[1] = -support_roll
        target[6] = support_roll
        target[7] += hip_lift
        target[8] += knee_lift
        target[9] += ankle_lift
    else:
        raise ValueError(f"Unsupported pose: {pose_name}")
    return target, support_foot


def _run_static_pose(
    *,
    foot_size: tuple[float, float, float],
    pose_name: str,
    ramp_steps: int,
    hold_steps: int,
    support_roll: float,
    hip_lift: float,
    knee_lift: float,
    ankle_lift: float,
    scene_path: Path,
) -> PoseResult:
    """Run one static single-leg pose test."""
    env = SedonStandingEnv(
        scene_path=scene_path,
        reset_noise_scale=0.0,
        reward_config=SedonStandingConfig(gait_mode="fsm"),
    )
    try:
        env.reset(seed=42)
        _set_foot_sizes(env, foot_size)
        target, support_foot = _joint_target_for_pose(
            env,
            pose_name=pose_name,
            support_roll=support_roll,
            hip_lift=hip_lift,
            knee_lift=knee_lift,
            ankle_lift=ankle_lift,
        )
        nominal = env._nominal_joint_qpos.copy()
        support_box = _support_box(env, support_foot)
        contact_counts = {"both": 0, "right_only": 0, "left_only": 0, "none": 0}
        base_proxy_floor_steps = 0
        min_base_z = float("inf")
        min_upright = float("inf")
        max_left_foot_bottom_z = -float("inf")
        max_right_foot_bottom_z = -float("inf")
        terminated = False
        steps = 0
        final_com = _overall_com(env.model, env.data)

        for index in range(ramp_steps + hold_steps):
            alpha = min(1.0, (index + 1) / ramp_steps)
            env._do_pd_simulation(nominal + (target - nominal) * alpha)
            obs = env._get_obs()
            base_z = env._base_height()
            upright = env._base_upright()
            terminated = env._is_terminated(base_z, upright, obs)
            state, base_contact = _contact_state(env)
            contact_counts[state] += 1
            if base_contact:
                base_proxy_floor_steps += 1
            foot_bottoms = env._foot_bottom_heights()
            max_right_foot_bottom_z = max(max_right_foot_bottom_z, float(foot_bottoms[0]))
            max_left_foot_bottom_z = max(max_left_foot_bottom_z, float(foot_bottoms[1]))
            min_base_z = min(min_base_z, base_z)
            min_upright = min(min_upright, upright)
            final_com = _overall_com(env.model, env.data)
            steps = index + 1
            if terminated:
                break

        support_box = _support_box(env, support_foot)
        com_relative = final_com[:2] - support_box.center_xy
        return PoseResult(
            name=pose_name,
            terminated=terminated,
            steps=steps,
            min_base_z=min_base_z,
            min_upright=min_upright,
            final_com=final_com.copy(),
            support_foot=support_foot,
            com_relative_xy=com_relative,
            com_inside_support=_point_in_support(final_com[:2], support_box),
            contact_state_counts=contact_counts,
            base_proxy_floor_steps=base_proxy_floor_steps,
            max_left_foot_bottom_z=max_left_foot_bottom_z,
            max_right_foot_bottom_z=max_right_foot_bottom_z,
        )
    finally:
        env.close()


def _print_reset_report(foot_size: tuple[float, float, float], scene_path: Path) -> None:
    """Print COM and support polygon diagnostics at reset."""
    env = SedonStandingEnv(
        scene_path=scene_path,
        reset_noise_scale=0.0,
        reward_config=SedonStandingConfig(gait_mode="fsm"),
    )
    try:
        env.reset(seed=42)
        _set_foot_sizes(env, foot_size)
        com = _overall_com(env.model, env.data)
        left_box = _support_box(env, LEFT_FOOT_GEOM)
        right_box = _support_box(env, RIGHT_FOOT_GEOM)
        print(f"\nfoot_size: {foot_size}")
        print(f"overall_com: {_format_vec(com)}")
        for geom_name, box in ((LEFT_FOOT_GEOM, left_box), (RIGHT_FOOT_GEOM, right_box)):
            snap = snapshot_geom(env.model, env.data, name=geom_name)
            rel = com[:2] - box.center_xy
            print(f"- {geom_name}")
            print(f"  world_pos       : {_format_vec(snap.position)}")
            print(f"  size            : {_format_vec(snap.size)}")
            print(f"  xy_range_center : {_format_vec(box.center_xy)}")
            print(f"  com_relative_xy : {_format_vec(rel)}")
            print(f"  com_inside      : {_point_in_support(com[:2], box)}")
        print(f"combined_support_contains_com: {_point_in_combined_support(com[:2], [left_box, right_box])}")
    finally:
        env.close()


def _parse_foot_sizes(raw_value: str) -> list[tuple[float, float, float]]:
    """Parse semicolon-separated MuJoCo box half-size triples."""
    sizes: list[tuple[float, float, float]] = []
    for chunk in raw_value.split(";"):
        values = [float(item.strip()) for item in chunk.split(",") if item.strip()]
        if len(values) != 3:
            raise argparse.ArgumentTypeError(
                "--foot-sizes must be semicolon-separated triples, e.g. 0.07,0.04,0.025;0.10,0.06,0.025"
            )
        if any(value <= 0 for value in values):
            raise argparse.ArgumentTypeError("Foot sizes must be positive.")
        sizes.append((values[0], values[1], values[2]))
    return sizes


def _parse_float_list(raw_value: str) -> list[float]:
    """Parse comma-separated floats."""
    values = [float(item.strip()) for item in raw_value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one float value.")
    return values


def _validation_cases(
    *,
    plan: str,
    foot_sizes: list[tuple[float, float, float]],
    support_roll: float,
    support_rolls: list[float],
) -> list[ValidationCase]:
    """Build the A/B validation cases without creating a full cross product by default."""
    base_foot_size = foot_sizes[0]
    cases: list[ValidationCase] = []

    if plan in ("contact", "both"):
        for foot_size in foot_sizes:
            cases.append(
                ValidationCase(
                    scenario="A_contact_relaxed",
                    foot_size=foot_size,
                    support_roll=support_roll,
                )
            )

    if plan in ("load", "both"):
        for roll in support_rolls:
            cases.append(
                ValidationCase(
                    scenario="B_load_phase",
                    foot_size=base_foot_size,
                    support_roll=roll,
                )
            )

    if plan == "matrix":
        for foot_size in foot_sizes:
            for roll in support_rolls:
                cases.append(
                    ValidationCase(
                        scenario="matrix",
                        foot_size=foot_size,
                        support_roll=roll,
                    )
                )

    deduped: list[ValidationCase] = []
    seen: set[tuple[str, tuple[float, float, float], float]] = set()
    for case in cases:
        key = (case.scenario, case.foot_size, case.support_roll)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(case)
    return deduped


def _single_support_steps(result: PoseResult) -> int:
    """Return steps where only the intended support foot contacts the floor."""
    if result.support_foot == LEFT_FOOT_GEOM:
        return result.contact_state_counts["left_only"]
    if result.support_foot == RIGHT_FOOT_GEOM:
        return result.contact_state_counts["right_only"]
    return 0


def _wrong_support_steps(result: PoseResult) -> int:
    """Return steps where only the non-support foot contacts the floor."""
    if result.support_foot == LEFT_FOOT_GEOM:
        return result.contact_state_counts["right_only"]
    if result.support_foot == RIGHT_FOOT_GEOM:
        return result.contact_state_counts["left_only"]
    return 0


def _result_rows(
    *,
    case: ValidationCase,
    result: PoseResult,
    total_expected_steps: int,
) -> dict[str, str | int | float | bool]:
    """Return a flat CSV row for one pose result."""
    counts = result.contact_state_counts
    return {
        "scenario": case.scenario,
        "foot_size_x": case.foot_size[0],
        "foot_size_y": case.foot_size[1],
        "foot_size_z": case.foot_size[2],
        "support_roll": case.support_roll,
        "pose": result.name,
        "support_foot": result.support_foot,
        "steps": result.steps,
        "held_full_steps": result.steps == total_expected_steps and not result.terminated,
        "terminated": result.terminated,
        "min_base_z": result.min_base_z,
        "min_upright": result.min_upright,
        "com_x": float(result.final_com[0]),
        "com_y": float(result.final_com[1]),
        "com_z": float(result.final_com[2]),
        "com_relative_x": float(result.com_relative_xy[0]),
        "com_relative_y": float(result.com_relative_xy[1]),
        "com_inside_support": result.com_inside_support,
        "both_steps": counts["both"],
        "right_only_steps": counts["right_only"],
        "left_only_steps": counts["left_only"],
        "none_steps": counts["none"],
        "single_support_steps": _single_support_steps(result),
        "wrong_support_steps": _wrong_support_steps(result),
        "base_proxy_floor_steps": result.base_proxy_floor_steps,
        "max_left_foot_bottom_z": result.max_left_foot_bottom_z,
        "max_right_foot_bottom_z": result.max_right_foot_bottom_z,
    }


def _write_csv(rows: list[dict[str, str | int | float | bool]], csv_path: Path) -> None:
    """Write static stability summary rows to CSV."""
    if not rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        choices=("contact", "load", "both", "matrix"),
        default="both",
        help=(
            "Validation plan: A contact relaxation, B load-phase roll sweep, "
            "both, or full foot-size/support-roll matrix."
        ),
    )
    parser.add_argument(
        "--relaxed-foot",
        action="store_true",
        help=(
            "Use the relaxed debug contact proxy foot size "
            f"{RELAXED_FOOT_SIZE} instead of the realistic/proposed foot-size list."
        ),
    )
    parser.add_argument("--ramp-steps", type=int, default=40)
    parser.add_argument("--hold-steps", type=int, default=100)
    parser.add_argument("--support-roll", type=float, default=0.06)
    parser.add_argument("--scene-path", type=Path, default=DEFAULT_SCENE_PATH)
    parser.add_argument(
        "--support-rolls",
        type=_parse_float_list,
        default=_parse_float_list("0.00,0.03,0.06,0.09,0.12,0.15"),
        help="Comma-separated roll values for the B load-phase sweep.",
    )
    parser.add_argument("--hip-lift", type=float, default=-0.15)
    parser.add_argument("--knee-lift", type=float, default=-0.12)
    parser.add_argument("--ankle-lift", type=float, default=-0.04)
    parser.add_argument(
        "--foot-sizes",
        type=_parse_foot_sizes,
        default=_parse_foot_sizes(
            "0.07,0.04,0.025;0.09,0.05,0.025;0.10,0.06,0.025;"
            "0.12,0.07,0.025;0.16,0.15,0.025"
        ),
        help="Semicolon-separated MuJoCo foot box half-size triples.",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path("artifacts") / "sedon_debug" / "static_stability_summary.csv",
        help="Output CSV summary path.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run static support polygon and single-leg stability checks."""
    args = build_parser().parse_args(argv)
    if args.ramp_steps <= 0:
        raise ValueError("--ramp-steps must be positive.")
    if args.hold_steps <= 0:
        raise ValueError("--hold-steps must be positive.")

    foot_sizes = [RELAXED_FOOT_SIZE] if args.relaxed_foot else args.foot_sizes
    scene_path = require_scene(args.scene_path)
    cases = _validation_cases(
        plan=args.plan,
        foot_sizes=foot_sizes,
        support_roll=args.support_roll,
        support_rolls=args.support_rolls,
    )
    rows: list[dict[str, str | int | float | bool]] = []
    reported_reset_sizes: set[tuple[float, float, float]] = set()
    total_expected_steps = args.ramp_steps + args.hold_steps

    for case in cases:
        if case.foot_size not in reported_reset_sizes:
            _print_reset_report(case.foot_size, scene_path)
            reported_reset_sizes.add(case.foot_size)
        print(f"\nscenario: {case.scenario}")
        print(f"support_roll: {case.support_roll:.5f}")
        for pose_name in ("left_support_right_lift", "right_support_left_lift"):
            result = _run_static_pose(
                foot_size=case.foot_size,
                pose_name=pose_name,
                ramp_steps=args.ramp_steps,
                hold_steps=args.hold_steps,
                support_roll=case.support_roll,
                hip_lift=args.hip_lift,
                knee_lift=args.knee_lift,
                ankle_lift=args.ankle_lift,
                scene_path=scene_path,
            )
            rows.append(
                _result_rows(
                    case=case,
                    result=result,
                    total_expected_steps=total_expected_steps,
                )
            )
            counts = result.contact_state_counts
            print(f"pose: {result.name}")
            print(f"  support_foot          : {result.support_foot}")
            print(f"  steps                 : {result.steps}")
            print(f"  terminated            : {result.terminated}")
            print(f"  min_base_z            : {result.min_base_z:.5f}")
            print(f"  min_upright           : {result.min_upright:.5f}")
            print(f"  final_com             : {_format_vec(result.final_com)}")
            print(f"  com_relative_support  : {_format_vec(result.com_relative_xy)}")
            print(f"  com_inside_support    : {result.com_inside_support}")
            print(
                "  contact_state_counts  : "
                f"both={counts['both']} right_only={counts['right_only']} "
                f"left_only={counts['left_only']} none={counts['none']}"
            )
            print(f"  single_support_steps  : {_single_support_steps(result)}")
            print(f"  wrong_support_steps   : {_wrong_support_steps(result)}")
            print(f"  base_proxy_floor_steps: {result.base_proxy_floor_steps}")
            print(f"  max_left_foot_bottom_z : {result.max_left_foot_bottom_z:.5f}")
            print(f"  max_right_foot_bottom_z: {result.max_right_foot_bottom_z:.5f}")
    _write_csv(rows, args.csv_path)
    print(f"\nCSV: {args.csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
