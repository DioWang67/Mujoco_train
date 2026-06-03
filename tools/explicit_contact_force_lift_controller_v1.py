"""Gate IK micro-lift on explicit swing-foot normal-force reduction."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from seedon_baseline.env import SeedonStandingEnv
from tools.audit_seedon_shuffle_v0 import _count_contact_none_bursts, _load_config, audit_shuffle
from tools.blue_unload_mechanism_search import (
    DEFAULT_BASE_CONFIG,
    JOINT_NAMES,
    L_ANKLE,
    L_HIP_PITCH,
    L_HIP_ROLL,
    L_KNEE,
    R_ANKLE,
    R_HIP_PITCH,
    R_HIP_ROLL,
    R_KNEE,
    REPO_ROOT,
    UnloadCandidate,
    _left_unload_target,
    _right_unload_target,
    _zero,
)


DEFAULT_SOURCE_TOP = (
    REPO_ROOT
    / "artifacts"
    / "seedon_debug"
    / "blue_unload_refine_v2"
    / "blue_unload_refine_v2_top20.csv"
)
DEFAULT_OUT_DIR = REPO_ROOT / "artifacts" / "seedon_debug" / "explicit_contact_force_lift_controller_v1"
KINEMATIC_GAIN1_CLEARANCE_M = 0.00023308280933499043


@dataclass(frozen=True)
class ForceLiftCandidate:
    """One explicit force-gated lift controller candidate."""

    candidate_id: str
    unload: UnloadCandidate
    threshold_n: float
    target_clearance: float
    lift_duration: int
    landing_duration: int
    max_unload_steps: int


@dataclass(frozen=True)
class ForceLiftAudit:
    """Audit row for one force-gated IK lift candidate."""

    candidate_id: str
    source_candidate_id: str
    threshold_n: float
    target_clearance: float
    lift_duration: int
    landing_duration: int
    max_unload_steps: int
    threshold_reached: bool
    threshold_reach_step: int
    right_threshold_reached: bool
    left_threshold_reached: bool
    right_threshold_reach_step: int
    left_threshold_reach_step: int
    swing_force_at_lift_start: float
    support_force_at_lift_start: float
    max_clearance: float
    actual_foot_z_delta: float
    contact_none_ratio: float
    jump_count: int
    impact_post: float
    base_drop_post: float
    min_upright: float
    ctrl_saturation_mean: float
    ctrl_saturation_max: float
    force_saturation_mean: float
    force_saturation_max: float
    a_passed: bool
    b_passed: bool
    c_passed: bool
    score: float
    timeline_path: str


def _parse_float_list(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one float")
    return values


def _parse_int_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("expected positive integers")
    return values


def _source_row(path: Path, candidate_id: str | None) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing source top candidates CSV: {path}")
    with path.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"No rows in {path}")
    if candidate_id is None:
        return rows[0]
    for row in rows:
        if row["candidate_id"] == candidate_id:
            return row
    raise ValueError(f"Missing candidate_id={candidate_id} in {path}")


def _source_to_unload(row: dict[str, str]) -> UnloadCandidate:
    return UnloadCandidate(
        candidate_id=row["candidate_id"],
        preload_duration=int(row["preload_duration"]),
        support_hip_roll=float(row["support_hip_roll"]),
        swing_hip_roll=float(row["swing_hip_roll"]),
        stance_knee=float(row["stance_knee"]),
        stance_ankle=float(row["stance_ankle"]),
        pelvis_lean_proxy=float(row["pelvis_lean_proxy"]),
        lateral_offset_proxy=float(row["lateral_offset_proxy"]),
        tiny_lift_amplitude=0.0,
    )


def _minimum_jerk(t: float) -> float:
    return 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5


def _ik_delta(gain: float, *, swing_side: str) -> np.ndarray:
    eps = 0.005 * float(gain)
    delta = np.zeros(len(JOINT_NAMES), dtype=np.float64)
    if swing_side == "right":
        delta[R_HIP_PITCH] = 0.5 * eps
        delta[R_KNEE] = -eps
        delta[R_ANKLE] = -0.25 * eps
    elif swing_side == "left":
        delta[L_HIP_PITCH] = 0.5 * eps
        delta[L_KNEE] = -eps
        delta[L_ANKLE] = -0.25 * eps
    else:
        raise ValueError(f"Unsupported swing side: {swing_side}")
    return delta


def _unload_target(unload: UnloadCandidate, swing_side: str) -> np.ndarray:
    if swing_side == "right":
        return _right_unload_target(unload)
    if swing_side == "left":
        return _left_unload_target(unload)
    raise ValueError(f"Unsupported swing side: {swing_side}")


def _feedback_unload_target(base: np.ndarray, swing_side: str, swing_force: float, threshold: float) -> np.ndarray:
    """Return a conservative hip-roll/lean adjustment while waiting for force gate."""

    error_ratio = max(0.0, (float(swing_force) - float(threshold)) / max(float(threshold), 1.0))
    lean_delta = min(0.035, 0.006 * error_ratio)
    support_delta = min(0.030, 0.005 * error_ratio)
    swing_delta = min(0.020, 0.003 * error_ratio)
    target = base.copy()
    if swing_side == "right":
        target[L_HIP_ROLL] += support_delta + lean_delta
        target[R_HIP_ROLL] += swing_delta + lean_delta
    elif swing_side == "left":
        target[R_HIP_ROLL] -= support_delta + lean_delta
        target[L_HIP_ROLL] -= swing_delta + lean_delta
    else:
        raise ValueError(f"Unsupported swing side: {swing_side}")
    return target


def _force_state(env: SeedonStandingEnv, swing_side: str) -> tuple[float, float]:
    forces = env._foot_force_state()
    if swing_side == "right":
        return float(forces["right_force"]), float(forces["left_force"])
    if swing_side == "left":
        return float(forces["left_force"]), float(forces["right_force"])
    raise ValueError(f"Unsupported swing side: {swing_side}")


def _contact_state(env: SeedonStandingEnv) -> str:
    flags = env._floor_contact_flags()
    return env._contact_state(flags)


def _foot_bottom(env: SeedonStandingEnv, side: str) -> float:
    heights = env._foot_bottom_heights()
    if side == "right":
        return float(heights[0])
    if side == "left":
        return float(heights[1])
    raise ValueError(f"Unsupported side: {side}")


def _clearance(env: SeedonStandingEnv, swing_side: str) -> float:
    right_z, left_z = env._foot_bottom_heights()
    if swing_side == "right":
        return max(0.0, float(right_z - left_z))
    if swing_side == "left":
        return max(0.0, float(left_z - right_z))
    return 0.0


def _ctrl_saturation(env: SeedonStandingEnv) -> float:
    span = np.maximum(env._ctrl_range[:, 1] - env._ctrl_range[:, 0], 1e-9)
    lower = np.abs(env.data.ctrl - env._ctrl_range[:, 0])
    upper = np.abs(env.data.ctrl - env._ctrl_range[:, 1])
    return float(np.mean(np.minimum(lower, upper) <= 0.02 * span))


def _force_saturation(env: SeedonStandingEnv) -> float:
    if not hasattr(env.model, "actuator_forcerange") or not hasattr(env.data, "actuator_force"):
        return 0.0
    force_range = np.asarray(env.model.actuator_forcerange, dtype=np.float64)
    actuator_force = np.asarray(env.data.actuator_force, dtype=np.float64)
    if force_range.shape[0] != actuator_force.shape[0] or actuator_force.size == 0:
        return 0.0
    span = np.maximum(force_range[:, 1] - force_range[:, 0], 1e-9)
    finite = np.isfinite(force_range).all(axis=1) & (span > 1e-8)
    lower = np.abs(actuator_force - force_range[:, 0])
    upper = np.abs(actuator_force - force_range[:, 1])
    saturated = finite & (np.minimum(lower, upper) <= 0.02 * span)
    return float(np.mean(saturated))


def _step_target(env: SeedonStandingEnv, target: np.ndarray) -> None:
    env._do_pd_simulation(env._apply_safe_joint_target_clamps(np.asarray(target, dtype=np.float64)))
    env._gait_step += 1


def _state_row(env: SeedonStandingEnv, step: int, phase: str, swing_side: str) -> dict[str, Any]:
    swing_force, support_force = _force_state(env, swing_side)
    return {
        "step": step,
        "phase": phase,
        "swing_side": swing_side,
        "swing_force_n": swing_force,
        "support_force_n": support_force,
        "clearance": _clearance(env, swing_side),
        "swing_foot_z": _foot_bottom(env, swing_side),
        "contact_state": _contact_state(env),
        "base_height": env._base_height(),
        "base_roll": env._base_roll(),
        "base_pitch": env._base_pitch(),
        "upright": env._base_upright(),
        "total_contact_force": float(env._foot_force_state()["right_force"] + env._foot_force_state()["left_force"]),
        "ctrl_saturation": _ctrl_saturation(env),
        "force_saturation": _force_saturation(env),
    }


def _run_side(
    env: SeedonStandingEnv,
    *,
    unload: UnloadCandidate,
    swing_side: str,
    threshold: float,
    target_clearance: float,
    lift_duration: int,
    landing_duration: int,
    max_unload_steps: int,
    step_counter: int,
) -> tuple[list[dict[str, Any]], bool, int, float, float, float]:
    rows: list[dict[str, Any]] = []
    base_target = _unload_target(unload, swing_side)
    reached = False
    reach_step = -1
    swing_at_start = float("inf")
    support_at_start = 0.0
    lift_start_z = _foot_bottom(env, swing_side)
    last_unload_target = base_target

    for local_step in range(1, max_unload_steps + 1):
        swing_force, _ = _force_state(env, swing_side)
        last_unload_target = _feedback_unload_target(base_target, swing_side, swing_force, threshold)
        _step_target(env, last_unload_target)
        step_counter += 1
        rows.append(_state_row(env, step_counter, f"{swing_side}_force_unload", swing_side))
        swing_force, support_force = _force_state(env, swing_side)
        if swing_force <= threshold:
            reached = True
            reach_step = local_step
            swing_at_start = swing_force
            support_at_start = support_force
            lift_start_z = _foot_bottom(env, swing_side)
            break

    if not reached:
        return rows, False, reach_step, swing_at_start, support_at_start, 0.0

    gain = target_clearance / max(KINEMATIC_GAIN1_CLEARANCE_M, 1e-9)
    lift_target = last_unload_target + _ik_delta(gain, swing_side=swing_side)
    for local_step in range(1, lift_duration + 1):
        alpha = _minimum_jerk(local_step / lift_duration)
        _step_target(env, last_unload_target + (lift_target - last_unload_target) * alpha)
        step_counter += 1
        rows.append(_state_row(env, step_counter, f"{swing_side}_force_gated_lift", swing_side))

    for local_step in range(1, landing_duration + 1):
        alpha = _minimum_jerk(local_step / landing_duration)
        _step_target(env, lift_target + (last_unload_target - lift_target) * alpha)
        step_counter += 1
        rows.append(_state_row(env, step_counter, f"{swing_side}_minimum_jerk_land", swing_side))

    actual_delta = max(float(row["swing_foot_z"]) - lift_start_z for row in rows if row["swing_side"] == swing_side)
    return rows, True, reach_step, swing_at_start, support_at_start, max(0.0, actual_delta)


def _write_timeline(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def audit_candidate(
    candidate: ForceLiftCandidate,
    *,
    config_path: Path,
    baseline_impact: float,
    out_dir: Path,
    steps: int,
    seed: int,
    warmup_steps: int,
) -> ForceLiftAudit:
    env = SeedonStandingEnv(reset_noise_scale=0.0, reward_config=_load_config(config_path))
    total_weight = float(np.sum(env.model.body_mass) * 9.81)
    rows: list[dict[str, Any]] = []
    try:
        env.reset(seed=seed)
        step_counter = 0
        for _ in range(30):
            _step_target(env, _zero())
            step_counter += 1
            rows.append(_state_row(env, step_counter, "neutral", "right"))
        right_rows, right_reached, right_step, right_swing, right_support, right_delta = _run_side(
            env,
            unload=candidate.unload,
            swing_side="right",
            threshold=candidate.threshold_n,
            target_clearance=candidate.target_clearance,
            lift_duration=candidate.lift_duration,
            landing_duration=candidate.landing_duration,
            max_unload_steps=candidate.max_unload_steps,
            step_counter=step_counter,
        )
        rows.extend(right_rows)
        step_counter = int(rows[-1]["step"]) if rows else step_counter
        for _ in range(30):
            _step_target(env, _zero())
            step_counter += 1
            rows.append(_state_row(env, step_counter, "neutral_after_right", "left"))
        left_rows, left_reached, left_step, left_swing, left_support, left_delta = _run_side(
            env,
            unload=candidate.unload,
            swing_side="left",
            threshold=candidate.threshold_n,
            target_clearance=candidate.target_clearance,
            lift_duration=candidate.lift_duration,
            landing_duration=candidate.landing_duration,
            max_unload_steps=candidate.max_unload_steps,
            step_counter=step_counter,
        )
        rows.extend(left_rows)
    finally:
        env.close()

    if steps > 0:
        rows = rows[:steps]
    timeline_path = out_dir / "timelines" / f"{candidate.candidate_id}.csv"
    _write_timeline(timeline_path, rows)

    contact_none = sum(1 for row in rows if row["contact_state"] == "none")
    jump_count = _count_contact_none_bursts([{"right_contact": row["contact_state"] in {"right", "both"}, "left_contact": row["contact_state"] in {"left", "both"}} for row in rows])
    post_rows = rows[warmup_steps:] or rows
    impact = max((float(row["total_contact_force"]) for row in post_rows), default=0.0) / max(total_weight, 1e-6)
    base0 = float(post_rows[0]["base_height"]) if post_rows else 0.0
    base_drop = max(0.0, base0 - min((float(row["base_height"]) for row in post_rows), default=base0))
    min_upright = min((float(row["upright"]) for row in rows), default=0.0)
    ctrl_sats = [float(row["ctrl_saturation"]) for row in rows]
    force_sats = [float(row["force_saturation"]) for row in rows]
    threshold_reached = right_reached and left_reached
    swing_start = max(right_swing, left_swing)
    support_start = min(right_support, left_support)
    max_clearance = max((float(row["clearance"]) for row in rows), default=0.0)
    actual_delta = max(right_delta, left_delta)
    a_passed = threshold_reached and swing_start <= candidate.threshold_n and contact_none == 0 and jump_count == 0 and min_upright >= 0.99
    b_passed = a_passed and max_clearance >= 0.001
    c_passed = b_passed and impact <= baseline_impact * 1.2
    stability_score = (
        min_upright * 50.0
        - max(0.0, impact - baseline_impact * 1.2) * 40.0
        - max(0.0, base_drop - 0.015) * 400.0
        - contact_none * 10.0
        - jump_count * 20.0
    )
    score = (
        (300.0 if c_passed else 0.0)
        + (150.0 if b_passed else 0.0)
        + (75.0 if a_passed else 0.0)
        + (20.0 if threshold_reached else 0.0)
        + stability_score
        + min(max_clearance, 0.002) * 1000.0
    )
    return ForceLiftAudit(
        candidate_id=candidate.candidate_id,
        source_candidate_id=candidate.unload.candidate_id,
        threshold_n=candidate.threshold_n,
        target_clearance=candidate.target_clearance,
        lift_duration=candidate.lift_duration,
        landing_duration=candidate.landing_duration,
        max_unload_steps=candidate.max_unload_steps,
        threshold_reached=threshold_reached,
        threshold_reach_step=max(right_step, left_step) if threshold_reached else -1,
        right_threshold_reached=right_reached,
        left_threshold_reached=left_reached,
        right_threshold_reach_step=right_step,
        left_threshold_reach_step=left_step,
        swing_force_at_lift_start=swing_start,
        support_force_at_lift_start=support_start,
        max_clearance=max_clearance,
        actual_foot_z_delta=actual_delta,
        contact_none_ratio=contact_none / max(1, len(rows)),
        jump_count=jump_count,
        impact_post=impact,
        base_drop_post=base_drop,
        min_upright=min_upright,
        ctrl_saturation_mean=float(np.mean(ctrl_sats)) if ctrl_sats else 0.0,
        ctrl_saturation_max=max(ctrl_sats, default=0.0),
        force_saturation_mean=float(np.mean(force_sats)) if force_sats else 0.0,
        force_saturation_max=max(force_sats, default=0.0),
        a_passed=a_passed,
        b_passed=b_passed,
        c_passed=c_passed,
        score=score,
        timeline_path=str(timeline_path),
    )


def build_candidates(args: argparse.Namespace) -> list[ForceLiftCandidate]:
    row = _source_row(args.source_top, args.source_candidate_id)
    unload = _source_to_unload(row)
    candidates: list[ForceLiftCandidate] = []
    for threshold in args.thresholds:
        for target_clearance in args.target_clearances:
            for lift_duration in args.lift_durations:
                for max_unload_steps in args.max_unload_steps:
                    candidate_id = (
                        f"{unload.candidate_id}_thr{threshold:g}n"
                        f"_clr{target_clearance * 1000.0:.1f}mm"
                        f"_ld{lift_duration}_unload{max_unload_steps}"
                    ).replace(".", "p")
                    candidates.append(
                        ForceLiftCandidate(
                            candidate_id=candidate_id,
                            unload=unload,
                            threshold_n=threshold,
                            target_clearance=target_clearance,
                            lift_duration=lift_duration,
                            landing_duration=args.landing_duration,
                            max_unload_steps=max_unload_steps,
                        )
                    )
    return candidates


def write_results(path: Path, rows: list[ForceLiftAudit]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_summary(path: Path, rows: list[ForceLiftAudit], baseline_impact: float) -> None:
    a_count = sum(row.a_passed for row in rows)
    b_count = sum(row.b_passed for row in rows)
    c_count = sum(row.c_passed for row in rows)
    best = max(rows, key=lambda row: row.score) if rows else None
    lines = [
        "# explicit_contact_force_lift_controller_v1",
        "",
        f"candidates: {len(rows)}",
        f"A_pass: {a_count}",
        f"B_pass: {b_count}",
        f"C_pass: {c_count}",
        f"teacher_baseline_impact_post: {baseline_impact:.6f}",
    ]
    if best is not None:
        lines.extend(
            [
                "",
                f"best_candidate: {best.candidate_id}",
                f"threshold_reached: {best.threshold_reached}",
                f"swing_force_at_lift_start: {best.swing_force_at_lift_start:.4f}",
                f"max_clearance: {best.max_clearance:.6f}",
                f"actual_foot_z_delta: {best.actual_foot_z_delta:.6f}",
                f"impact_post: {best.impact_post:.6f}",
                f"min_upright: {best.min_upright:.6f}",
                f"timeline: {best.timeline_path}",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-top", type=Path, default=DEFAULT_SOURCE_TOP)
    parser.add_argument("--source-candidate-id", default=None)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--audit-warmup-steps", type=int, default=20)
    parser.add_argument("--thresholds", type=_parse_float_list, default="20,10,5")
    parser.add_argument("--target-clearances", type=_parse_float_list, default="0.001,0.0015")
    parser.add_argument("--lift-durations", type=_parse_int_list, default="60,90,120")
    parser.add_argument("--landing-duration", type=int, default=120)
    parser.add_argument("--max-unload-steps", type=_parse_int_list, default="120,180,240")
    parser.add_argument("--top-k", type=int, default=10)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    baseline = audit_shuffle(args.base_config, None, None, args.steps, args.seed, audit_warmup_steps=args.audit_warmup_steps)
    candidates = build_candidates(args)
    rows: list[ForceLiftAudit] = []
    print(f"candidates={len(candidates)}")
    for index, candidate in enumerate(candidates, start=1):
        row = audit_candidate(
            candidate,
            config_path=args.base_config,
            baseline_impact=baseline.landing_impact_post_warmup,
            out_dir=args.out_dir,
            steps=args.steps,
            seed=args.seed,
            warmup_steps=args.audit_warmup_steps,
        )
        rows.append(row)
        if index == 1 or index == len(candidates) or index % 10 == 0:
            print(
                f"[{index}/{len(candidates)}] {row.candidate_id}: "
                f"A={row.a_passed} B={row.b_passed} C={row.c_passed} "
                f"gate={row.threshold_reached} force={row.swing_force_at_lift_start:.2f} "
                f"clear={row.max_clearance:.4f} impact={row.impact_post:.3f} upright={row.min_upright:.3f}"
            )
    ranked = sorted(rows, key=lambda row: row.score, reverse=True)
    write_results(args.out_dir / "explicit_contact_force_lift_controller_v1.csv", rows)
    write_results(args.out_dir / "explicit_contact_force_lift_controller_v1_top10.csv", ranked[: args.top_k])
    write_summary(args.out_dir / "summary.md", rows, baseline.landing_impact_post_warmup)
    for row in ranked[: args.top_k]:
        print(
            f"TOP {row.candidate_id}: A={row.a_passed} B={row.b_passed} C={row.c_passed} "
            f"gate={row.threshold_reached} force={row.swing_force_at_lift_start:.2f} "
            f"clear={row.max_clearance:.4f} dz={row.actual_foot_z_delta:.4f} "
            f"impact={row.impact_post:.3f} drop={row.base_drop_post:.4f} upright={row.min_upright:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
