"""Closed-loop unload-only controller audit for Seedon controller v2a."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from seedon_baseline.env import SeedonStandingEnv
from tools.audit_seedon_shuffle_v0 import _count_contact_none_bursts, _load_config, audit_shuffle
from tools.blue_forward_shuffle_v1 import DEFAULT_CONFIG, DEFAULT_MODEL, DEFAULT_VECNORM
from tools.seedon_explicit_locomotion_controller_v2 import (
    L_HIP_ROLL,
    R_HIP_ROLL,
    _contact_state,
    _ctrl_saturation,
    _force_saturation,
    _support_force,
    _support_side,
    _swing_force,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "artifacts" / "seedon_debug" / "controller_v2a_unload"


class UnloadPhase(str, Enum):
    """Unload-only FSM phases."""

    DOUBLE_SUPPORT = "DOUBLE_SUPPORT"
    PRELOAD_RIGHT = "PRELOAD_RIGHT"
    UNLOAD_RIGHT = "UNLOAD_RIGHT"
    HOLD_RIGHT = "HOLD_RIGHT"
    PRELOAD_LEFT = "PRELOAD_LEFT"
    UNLOAD_LEFT = "UNLOAD_LEFT"
    HOLD_LEFT = "HOLD_LEFT"


@dataclass
class UnloadRuntime:
    """Mutable unload controller state."""

    phase: UnloadPhase = UnloadPhase.DOUBLE_SUPPORT
    phase_step: int = 0
    swing_side: str = "right"
    support_correction: float = 0.0
    swing_correction: float = 0.0
    lean_correction: float = 0.0
    hold_support_correction: float = 0.0
    hold_swing_correction: float = 0.0
    hold_lean_correction: float = 0.0
    gate_reach_step: int = -1
    gate_reached_count: int = 0
    timeout_count: int = 0


@dataclass(frozen=True)
class UnloadConfig:
    """Controller constants for one unload target."""

    target_force_n: float
    double_support_steps: int
    preload_steps: int
    unload_steps: int
    hold_steps: int
    feedback_gain: float
    correction_step_limit: float
    support_correction_limit: float
    swing_correction_limit: float
    lean_correction_limit: float
    support_base: float
    swing_base: float
    lean_base: float


@dataclass(frozen=True)
class UnloadAudit:
    """Aggregate unload v2a audit metrics."""

    target_force_n: float
    steps: int
    min_swing_force: float
    mean_swing_force_unload: float
    initial_unload_swing_force: float
    final_unload_swing_force: float
    force_error_slope: float
    gate_reached: bool
    gate_reach_step: int
    gate_reached_count: int
    timeout_count: int
    mean_support_ratio_unload: float
    contact_none_ratio: float
    jump_count: int
    min_upright: float
    base_drop_post: float
    impact_post: float
    ctrl_saturation_max: float
    force_saturation_max: float
    a_passed: bool
    b_passed: bool
    c_passed: bool
    stable: bool
    fail_reasons: str
    timeline_path: str


def _parse_float_list(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one float")
    return values


def _phase_for(swing_side: str, prefix: str) -> UnloadPhase:
    return UnloadPhase[f"{prefix}_{swing_side.upper()}"]


def _next_side(side: str) -> str:
    return "left" if side == "right" else "right"


def _enter(runtime: UnloadRuntime, phase: UnloadPhase) -> None:
    if "HOLD" in phase.value:
        runtime.hold_support_correction = runtime.support_correction
        runtime.hold_swing_correction = runtime.swing_correction
        runtime.hold_lean_correction = runtime.lean_correction
    runtime.phase = phase
    runtime.phase_step = 0
    if "PRELOAD" in phase.value:
        runtime.support_correction = 0.0
        runtime.swing_correction = 0.0
        runtime.lean_correction = 0.0
        runtime.hold_support_correction = 0.0
        runtime.hold_swing_correction = 0.0
        runtime.hold_lean_correction = 0.0


def _signed_terms(swing_side: str, support_value: float, swing_value: float, lean_value: float) -> tuple[float, float]:
    """Return right/left hip-roll corrections for one swing side.

    Positive support/lean shifts are defined as loading the support side.
    """

    if swing_side == "right":
        right = swing_value + lean_value
        left = support_value + lean_value
        return right, left
    right = -support_value - lean_value
    left = -swing_value - lean_value
    return right, left


def _update_corrections(runtime: UnloadRuntime, swing_force: float, config: UnloadConfig) -> float:
    error = max(0.0, swing_force - config.target_force_n)
    raw_delta = config.feedback_gain * error
    delta = float(np.clip(raw_delta, 0.0, config.correction_step_limit))
    runtime.support_correction = min(config.support_correction_limit, runtime.support_correction + delta)
    runtime.swing_correction = min(config.swing_correction_limit, runtime.swing_correction + 0.45 * delta)
    runtime.lean_correction = min(config.lean_correction_limit, runtime.lean_correction + 0.65 * delta)
    return error


def _minimum_jerk(alpha: float) -> float:
    value = float(np.clip(alpha, 0.0, 1.0))
    return value * value * value * (10.0 - 15.0 * value + 6.0 * value * value)


def _target(env: SeedonStandingEnv, runtime: UnloadRuntime, config: UnloadConfig) -> tuple[np.ndarray, float]:
    target = env._nominal_joint_qpos.copy()
    swing_force = _swing_force(env, runtime.swing_side)
    force_error = 0.0
    if runtime.phase in {UnloadPhase.PRELOAD_RIGHT, UnloadPhase.PRELOAD_LEFT}:
        preload_alpha = min(1.0, runtime.phase_step / max(config.preload_steps, 1))
        support = config.support_base * preload_alpha
        swing = config.swing_base * preload_alpha
        lean = config.lean_base * preload_alpha
    elif runtime.phase in {UnloadPhase.UNLOAD_RIGHT, UnloadPhase.UNLOAD_LEFT}:
        force_error = _update_corrections(runtime, swing_force, config)
        support = config.support_base + runtime.support_correction
        swing = config.swing_base + runtime.swing_correction
        lean = config.lean_base + runtime.lean_correction
    elif runtime.phase in {UnloadPhase.HOLD_RIGHT, UnloadPhase.HOLD_LEFT}:
        hold_alpha = 1.0 - _minimum_jerk(runtime.phase_step / max(config.hold_steps, 1))
        support = (config.support_base + runtime.hold_support_correction) * hold_alpha
        swing = (config.swing_base + runtime.hold_swing_correction) * hold_alpha
        lean = (config.lean_base + runtime.hold_lean_correction) * hold_alpha
        runtime.support_correction = runtime.hold_support_correction * hold_alpha
        runtime.swing_correction = runtime.hold_swing_correction * hold_alpha
        runtime.lean_correction = runtime.hold_lean_correction * hold_alpha
    else:
        support = 0.0
        swing = 0.0
        lean = 0.0
    right_roll, left_roll = _signed_terms(runtime.swing_side, support, swing, lean)
    target[R_HIP_ROLL] += right_roll
    target[L_HIP_ROLL] += left_roll
    return env._apply_safe_joint_target_clamps(target), force_error


def _advance(runtime: UnloadRuntime, env: SeedonStandingEnv, config: UnloadConfig) -> str:
    event = ""
    if runtime.phase == UnloadPhase.DOUBLE_SUPPORT and runtime.phase_step >= config.double_support_steps:
        _enter(runtime, _phase_for(runtime.swing_side, "PRELOAD"))
        return "enter_preload"
    if runtime.phase in {UnloadPhase.PRELOAD_RIGHT, UnloadPhase.PRELOAD_LEFT} and runtime.phase_step >= config.preload_steps:
        _enter(runtime, _phase_for(runtime.swing_side, "UNLOAD"))
        return "enter_unload"
    if runtime.phase in {UnloadPhase.UNLOAD_RIGHT, UnloadPhase.UNLOAD_LEFT}:
        if _swing_force(env, runtime.swing_side) <= config.target_force_n:
            runtime.gate_reached_count += 1
            if runtime.gate_reach_step < 0:
                runtime.gate_reach_step = runtime.phase_step
            _enter(runtime, _phase_for(runtime.swing_side, "HOLD"))
            return "gate_reached"
        if runtime.phase_step >= config.unload_steps:
            runtime.timeout_count += 1
            _enter(runtime, _phase_for(runtime.swing_side, "HOLD"))
            return "gate_timeout"
    if runtime.phase in {UnloadPhase.HOLD_RIGHT, UnloadPhase.HOLD_LEFT} and runtime.phase_step >= config.hold_steps:
        runtime.swing_side = _next_side(runtime.swing_side)
        _enter(runtime, _phase_for(runtime.swing_side, "PRELOAD"))
        return "switch_side"
    runtime.phase_step += 1
    return event


def _row(
    step: int,
    env: SeedonStandingEnv,
    runtime: UnloadRuntime,
    event: str,
    force_error: float,
    robot_weight: float,
) -> dict[str, Any]:
    swing = _swing_force(env, runtime.swing_side)
    support = _support_force(env, runtime.swing_side)
    return {
        "step": step,
        "phase": runtime.phase.value,
        "phase_step": runtime.phase_step,
        "swing_side": runtime.swing_side,
        "event": event,
        "swing_force": swing,
        "support_force": support,
        "force_error": force_error,
        "support_ratio": support / max(support + swing, 1e-9),
        "support_correction": runtime.support_correction,
        "swing_correction": runtime.swing_correction,
        "lean_correction": runtime.lean_correction,
        "contact_state": _contact_state(env),
        "right_contact": bool(env._floor_contact_flags()["right"]),
        "left_contact": bool(env._floor_contact_flags()["left"]),
        "impact": (support + swing) / max(robot_weight, 1e-9),
        "base_height": env._base_height(),
        "upright": env._base_upright(),
        "base_roll": env._base_roll(),
        "base_pitch": env._base_pitch(),
        "ctrl_saturation": _ctrl_saturation(env),
        "force_saturation": _force_saturation(env),
    }


def _write_timeline(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _trend(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    slope, _ = np.polyfit(x, y, deg=1)
    return float(slope)


def run_unload(
    config_path: Path,
    unload_config: UnloadConfig,
    *,
    out_dir: Path,
    steps: int,
    seed: int,
    warmup_steps: int,
    teacher_impact: float,
) -> UnloadAudit:
    env = SeedonStandingEnv(reset_noise_scale=0.0, reward_config=_load_config(config_path))
    runtime = UnloadRuntime()
    rows: list[dict[str, Any]] = []
    try:
        env.reset(seed=seed)
        robot_weight = float(np.sum(env.model.body_mass) * 9.81)
        for step in range(1, steps + 1):
            target, force_error = _target(env, runtime, unload_config)
            env._do_pd_simulation(target)
            env._gait_step += 1
            event = _advance(runtime, env, unload_config)
            rows.append(_row(step, env, runtime, event, force_error, robot_weight))
    finally:
        env.close()

    timeline_path = out_dir / "timelines" / f"unload_v2a_target_{unload_config.target_force_n:.0f}n.csv"
    _write_timeline(timeline_path, rows)
    post = rows[warmup_steps:] or rows
    unload_rows = [row for row in rows if str(row["phase"]).startswith("UNLOAD")]
    contact_none_steps = sum(1 for row in rows if row["contact_state"] == "none")
    jump_count = _count_contact_none_bursts(rows)
    min_upright = min((float(row["upright"]) for row in rows), default=0.0)
    base0 = float(post[0]["base_height"]) if post else 0.0
    base_drop = max(0.0, base0 - min((float(row["base_height"]) for row in post), default=base0))
    impact = max((float(row["impact"]) for row in post), default=0.0)
    min_swing = min((float(row["swing_force"]) for row in unload_rows), default=float("inf"))
    mean_swing = float(np.mean([float(row["swing_force"]) for row in unload_rows])) if unload_rows else float("inf")
    initial_swing = float(unload_rows[0]["swing_force"]) if unload_rows else float("inf")
    final_swing = float(unload_rows[-1]["swing_force"]) if unload_rows else float("inf")
    force_errors = [float(row["force_error"]) for row in unload_rows]
    support_ratios = [float(row["support_ratio"]) for row in unload_rows]
    ctrl_sat = max((float(row["ctrl_saturation"]) for row in rows), default=0.0)
    force_sat = max((float(row["force_saturation"]) for row in rows), default=0.0)
    stable_fail: list[str] = []
    if contact_none_steps:
        stable_fail.append("contact_none")
    if jump_count:
        stable_fail.append("jump")
    if min_upright < 0.99:
        stable_fail.append("upright")
    if base_drop > 0.015:
        stable_fail.append("base_drop")
    if impact > teacher_impact * 1.2:
        stable_fail.append("impact")
    if ctrl_sat != 0.0 or force_sat != 0.0:
        stable_fail.append("saturation")
    stable = not stable_fail
    return UnloadAudit(
        target_force_n=unload_config.target_force_n,
        steps=len(rows),
        min_swing_force=min_swing,
        mean_swing_force_unload=mean_swing,
        initial_unload_swing_force=initial_swing,
        final_unload_swing_force=final_swing,
        force_error_slope=_trend(force_errors),
        gate_reached=runtime.gate_reached_count > 0,
        gate_reach_step=runtime.gate_reach_step,
        gate_reached_count=runtime.gate_reached_count,
        timeout_count=runtime.timeout_count,
        mean_support_ratio_unload=float(np.mean(support_ratios)) if support_ratios else 0.0,
        contact_none_ratio=contact_none_steps / max(1, len(rows)),
        jump_count=jump_count,
        min_upright=min_upright,
        base_drop_post=base_drop,
        impact_post=impact,
        ctrl_saturation_max=ctrl_sat,
        force_saturation_max=force_sat,
        a_passed=stable and min_swing <= 35.0,
        b_passed=stable and min_swing <= 30.0,
        c_passed=stable and min_swing <= 20.0,
        stable=stable,
        fail_reasons=",".join(stable_fail),
        timeline_path=str(timeline_path),
    )


def write_results(path: Path, rows: list[UnloadAudit]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_summary(path: Path, rows: list[UnloadAudit], teacher_impact: float) -> None:
    lines = [
        "# Seedon Unload Controller V2A",
        "",
        f"teacher_landing_impact_post_warmup: {teacher_impact:.6f}",
        "",
        "| target | stable | A<=35 | B<=30 | C<=20 | min_force | mean_unload | trend | support | impact | drop | upright | gates | timeouts | reasons |",
        "|---:|:---:|:---:|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.target_force_n:.0f} | {row.stable} | {row.a_passed} | {row.b_passed} | {row.c_passed} | "
            f"{row.min_swing_force:.2f} | {row.mean_swing_force_unload:.2f} | {row.force_error_slope:.4f} | "
            f"{row.mean_support_ratio_unload:.3f} | {row.impact_post:.3f} | {row.base_drop_post:.5f} | "
            f"{row.min_upright:.3f} | {row.gate_reached_count} | {row.timeout_count} | {row.fail_reasons} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--vecnorm-path", type=Path, default=DEFAULT_VECNORM)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--audit-warmup-steps", type=int, default=20)
    parser.add_argument("--target-forces", type=_parse_float_list, default="35,30,20")
    parser.add_argument("--double-support-steps", type=int, default=40)
    parser.add_argument("--preload-steps", type=int, default=80)
    parser.add_argument("--unload-steps", type=int, default=160)
    parser.add_argument("--hold-steps", type=int, default=50)
    parser.add_argument("--feedback-gain", type=float, default=0.0012)
    parser.add_argument("--correction-step-limit", type=float, default=0.0025)
    parser.add_argument("--support-correction-limit", type=float, default=0.07)
    parser.add_argument("--swing-correction-limit", type=float, default=0.035)
    parser.add_argument("--lean-correction-limit", type=float, default=0.035)
    parser.add_argument("--support-base", type=float, default=0.08)
    parser.add_argument("--swing-base", type=float, default=0.02)
    parser.add_argument("--lean-base", type=float, default=0.025)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    teacher = audit_shuffle(
        args.config,
        args.model_path,
        args.vecnorm_path,
        args.steps,
        args.seed,
        audit_warmup_steps=args.audit_warmup_steps,
    )
    rows: list[UnloadAudit] = []
    for target_force in args.target_forces:
        config = UnloadConfig(
            target_force_n=target_force,
            double_support_steps=args.double_support_steps,
            preload_steps=args.preload_steps,
            unload_steps=args.unload_steps,
            hold_steps=args.hold_steps,
            feedback_gain=args.feedback_gain,
            correction_step_limit=args.correction_step_limit,
            support_correction_limit=args.support_correction_limit,
            swing_correction_limit=args.swing_correction_limit,
            lean_correction_limit=args.lean_correction_limit,
            support_base=args.support_base,
            swing_base=args.swing_base,
            lean_base=args.lean_base,
        )
        row = run_unload(
            args.config,
            config,
            out_dir=args.out_dir,
            steps=args.steps,
            seed=args.seed,
            warmup_steps=args.audit_warmup_steps,
            teacher_impact=teacher.landing_impact_post_warmup,
        )
        rows.append(row)
        print(
            f"target={target_force:.0f} stable={row.stable} A={row.a_passed} B={row.b_passed} C={row.c_passed} "
            f"min_force={row.min_swing_force:.2f} mean_unload={row.mean_swing_force_unload:.2f} "
            f"trend={row.force_error_slope:.4f} impact={row.impact_post:.3f} upright={row.min_upright:.3f} "
            f"gates={row.gate_reached_count} timeouts={row.timeout_count} reasons={row.fail_reasons or '-'}"
        )
    write_results(args.out_dir / "seedon_unload_controller_v2a.csv", rows)
    write_summary(args.out_dir / "summary.md", rows, teacher.landing_impact_post_warmup)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
