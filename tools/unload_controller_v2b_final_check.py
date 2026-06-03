"""Final minimal check for Seedon unload controller v2b authority.

This tool tests only the three channels supported by attribution:
swing hip-roll, support hip-roll, and lean proxy. It does not train,
does not add knee/ankle channels, and does not connect to lift.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
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
    _swing_force,
)
from tools.seedon_unload_controller_v2a import (
    UnloadConfig,
    UnloadPhase,
    UnloadRuntime,
    _advance,
    _minimum_jerk,
    _target as _v2a_target,
    run_unload,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "artifacts" / "seedon_debug" / "unload_controller_v2b_final_check"


@dataclass(frozen=True)
class GainProfile:
    """Closed-loop gain and clamp profile for one v2b final-check run."""

    name: str
    feedback_gain: float
    step_limit: float
    swing_limit: float
    support_limit: float
    lean_limit: float


@dataclass(frozen=True)
class ChannelSet:
    """Enabled v2b control channels."""

    name: str
    swing_hip_roll: bool
    support_hip_roll: bool
    lean_proxy: bool


@dataclass
class V2BRuntime:
    """Mutable v2b correction state."""

    fsm: UnloadRuntime
    swing_correction: float = 0.0
    support_correction: float = 0.0
    lean_correction: float = 0.0
    hold_swing_correction: float = 0.0
    hold_support_correction: float = 0.0
    hold_lean_correction: float = 0.0


@dataclass(frozen=True)
class V2BResult:
    """Aggregate result for one v2b channel-set/gain final check."""

    channel_set: str
    gain: str
    stable: bool
    a_passed: bool
    b_passed: bool
    c_passed: bool
    fail_reasons: str
    min_swing_force: float
    mean_swing_force: float
    improvement_vs_v2a: float
    gate_reached_count: int
    gate_reach_step: int
    timeout_count: int
    contact_none_ratio: float
    jump_count: int
    min_upright: float
    impact_post: float
    base_drop_post: float
    ctrl_saturation_max: float
    force_saturation_max: float
    max_swing_correction: float
    max_support_correction: float
    max_lean_correction: float
    timeline_path: str


CHANNEL_SETS: tuple[ChannelSet, ...] = (
    ChannelSet("swing_hip_roll", swing_hip_roll=True, support_hip_roll=False, lean_proxy=False),
    ChannelSet("swing_plus_support_hip_roll", swing_hip_roll=True, support_hip_roll=True, lean_proxy=False),
    ChannelSet("swing_support_plus_lean", swing_hip_roll=True, support_hip_roll=True, lean_proxy=True),
)

GAIN_PROFILES: tuple[GainProfile, ...] = (
    GainProfile(
        name="conservative",
        feedback_gain=0.0006,
        step_limit=0.0010,
        swing_limit=0.018,
        support_limit=0.014,
        lean_limit=0.006,
    ),
    GainProfile(
        name="medium",
        feedback_gain=0.0010,
        step_limit=0.0015,
        swing_limit=0.030,
        support_limit=0.024,
        lean_limit=0.012,
    ),
)


def _write_timeline(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _update_v2b_corrections(
    runtime: V2BRuntime,
    channel_set: ChannelSet,
    gain: GainProfile,
    swing_force: float,
    target_force: float,
) -> None:
    error = max(0.0, swing_force - target_force)
    delta = float(np.clip(error * gain.feedback_gain, 0.0, gain.step_limit))
    if channel_set.swing_hip_roll:
        runtime.swing_correction = min(gain.swing_limit, runtime.swing_correction + delta)
    if channel_set.support_hip_roll:
        runtime.support_correction = min(gain.support_limit, runtime.support_correction + 0.75 * delta)
    if channel_set.lean_proxy:
        runtime.lean_correction = min(gain.lean_limit, runtime.lean_correction + 0.45 * delta)


def _apply_terms(
    target: np.ndarray,
    swing_side: str,
    *,
    swing: float,
    support: float,
    lean: float,
) -> None:
    """Apply v2b roll terms.

    Directions are fixed from attribution:
    - swing hip-roll positive
    - support hip-roll negative
    - lean proxy positive
    Mirroring preserves the same physical meaning for left swing.
    """

    sign = 1.0 if swing_side == "right" else -1.0
    swing_joint = R_HIP_ROLL if swing_side == "right" else L_HIP_ROLL
    support_joint = L_HIP_ROLL if swing_side == "right" else R_HIP_ROLL
    target[swing_joint] += sign * (swing + lean)
    target[support_joint] += sign * (-support + lean)


def _v2b_target(
    env: SeedonStandingEnv,
    runtime: V2BRuntime,
    config: UnloadConfig,
    channel_set: ChannelSet,
    gain: GainProfile,
) -> np.ndarray:
    target, _ = _v2a_target(env, runtime.fsm, config)
    fsm = runtime.fsm
    if fsm.phase in {UnloadPhase.UNLOAD_RIGHT, UnloadPhase.UNLOAD_LEFT}:
        _update_v2b_corrections(runtime, channel_set, gain, _swing_force(env, fsm.swing_side), config.target_force_n)
        swing = runtime.swing_correction
        support = runtime.support_correction
        lean = runtime.lean_correction
    elif fsm.phase in {UnloadPhase.HOLD_RIGHT, UnloadPhase.HOLD_LEFT}:
        alpha = 1.0 - _minimum_jerk(fsm.phase_step / max(config.hold_steps, 1))
        swing = runtime.hold_swing_correction * alpha
        support = runtime.hold_support_correction * alpha
        lean = runtime.hold_lean_correction * alpha
        runtime.swing_correction = runtime.hold_swing_correction * alpha
        runtime.support_correction = runtime.hold_support_correction * alpha
        runtime.lean_correction = runtime.hold_lean_correction * alpha
    else:
        swing = 0.0
        support = 0.0
        lean = 0.0
    _apply_terms(target, fsm.swing_side, swing=swing, support=support, lean=lean)
    return env._apply_safe_joint_target_clamps(target)


def _sync_hold_state(previous_phase: UnloadPhase, runtime: V2BRuntime) -> None:
    if previous_phase in {UnloadPhase.UNLOAD_RIGHT, UnloadPhase.UNLOAD_LEFT} and runtime.fsm.phase in {
        UnloadPhase.HOLD_RIGHT,
        UnloadPhase.HOLD_LEFT,
    }:
        runtime.hold_swing_correction = runtime.swing_correction
        runtime.hold_support_correction = runtime.support_correction
        runtime.hold_lean_correction = runtime.lean_correction
    if previous_phase in {UnloadPhase.HOLD_RIGHT, UnloadPhase.HOLD_LEFT} and runtime.fsm.phase in {
        UnloadPhase.PRELOAD_RIGHT,
        UnloadPhase.PRELOAD_LEFT,
    }:
        runtime.swing_correction = 0.0
        runtime.support_correction = 0.0
        runtime.lean_correction = 0.0
        runtime.hold_swing_correction = 0.0
        runtime.hold_support_correction = 0.0
        runtime.hold_lean_correction = 0.0


def _row(step: int, env: SeedonStandingEnv, runtime: V2BRuntime, event: str, robot_weight: float) -> dict[str, Any]:
    fsm = runtime.fsm
    swing = _swing_force(env, fsm.swing_side)
    support = _support_force(env, fsm.swing_side)
    return {
        "step": step,
        "phase": fsm.phase.value,
        "phase_step": fsm.phase_step,
        "swing_side": fsm.swing_side,
        "event": event,
        "swing_force": swing,
        "support_force": support,
        "support_ratio": support / max(support + swing, 1e-9),
        "swing_correction": runtime.swing_correction,
        "support_correction": runtime.support_correction,
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


def run_v2b(
    config_path: Path,
    unload_config: UnloadConfig,
    *,
    out_dir: Path,
    steps: int,
    seed: int,
    warmup_steps: int,
    teacher_impact: float,
    v2a_min_swing_force: float,
    channel_set: ChannelSet,
    gain: GainProfile,
) -> V2BResult:
    env = SeedonStandingEnv(reset_noise_scale=0.0, reward_config=_load_config(config_path))
    runtime = V2BRuntime(fsm=UnloadRuntime())
    rows: list[dict[str, Any]] = []
    try:
        env.reset(seed=seed)
        robot_weight = float(np.sum(env.model.body_mass) * 9.81)
        for step in range(1, steps + 1):
            target = _v2b_target(env, runtime, unload_config, channel_set, gain)
            env._do_pd_simulation(target)
            env._gait_step += 1
            previous_phase = runtime.fsm.phase
            event = _advance(runtime.fsm, env, unload_config)
            _sync_hold_state(previous_phase, runtime)
            rows.append(_row(step, env, runtime, event, robot_weight))
    finally:
        env.close()

    timeline_path = out_dir / "timelines" / f"{channel_set.name}_{gain.name}.csv"
    _write_timeline(timeline_path, rows)
    post = rows[warmup_steps:] or rows
    unload_rows = [row for row in rows if str(row["phase"]).startswith("UNLOAD")]
    swing_values = [float(row["swing_force"]) for row in unload_rows]
    contact_none_steps = sum(1 for row in rows if row["contact_state"] == "none")
    jump_count = _count_contact_none_bursts(rows)
    min_upright = min((float(row["upright"]) for row in rows), default=0.0)
    base0 = float(post[0]["base_height"]) if post else 0.0
    base_drop = max(0.0, base0 - min((float(row["base_height"]) for row in post), default=base0))
    impact = max((float(row["impact"]) for row in post), default=0.0)
    ctrl_sat = max((float(row["ctrl_saturation"]) for row in rows), default=0.0)
    force_sat = max((float(row["force_saturation"]) for row in rows), default=0.0)
    min_swing = min(swing_values, default=float("inf"))
    mean_swing = float(np.mean(swing_values)) if swing_values else float("inf")
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
    return V2BResult(
        channel_set=channel_set.name,
        gain=gain.name,
        stable=stable,
        a_passed=stable and min_swing < 38.0,
        b_passed=stable and min_swing <= 35.0,
        c_passed=stable and min_swing <= 30.0,
        fail_reasons=",".join(stable_fail),
        min_swing_force=min_swing,
        mean_swing_force=mean_swing,
        improvement_vs_v2a=v2a_min_swing_force - min_swing,
        gate_reached_count=runtime.fsm.gate_reached_count,
        gate_reach_step=runtime.fsm.gate_reach_step,
        timeout_count=runtime.fsm.timeout_count,
        contact_none_ratio=contact_none_steps / max(1, len(rows)),
        jump_count=jump_count,
        min_upright=min_upright,
        impact_post=impact,
        base_drop_post=base_drop,
        ctrl_saturation_max=ctrl_sat,
        force_saturation_max=force_sat,
        max_swing_correction=max((float(row["swing_correction"]) for row in rows), default=0.0),
        max_support_correction=max((float(row["support_correction"]) for row in rows), default=0.0),
        max_lean_correction=max((float(row["lean_correction"]) for row in rows), default=0.0),
        timeline_path=str(timeline_path),
    )


def write_results(path: Path, rows: list[V2BResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_summary(path: Path, rows: list[V2BResult], v2a_min: float, v2a_mean: float) -> None:
    best = min(rows, key=lambda row: row.min_swing_force)
    lines = [
        "# Seedon Unload Controller V2B Final Check",
        "",
        f"v2a_min_swing_force: {v2a_min:.3f} N",
        f"v2a_mean_swing_force: {v2a_mean:.3f} N",
        "",
        "| channel_set | gain | stable | A<38 | B<=35 | C<=30 | min_force | mean_force | improve_vs_v2a | gates | timeouts | upright | impact | drop | reasons |",
        "|---|---|:---:|:---:|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.channel_set} | {row.gain} | {row.stable} | {row.a_passed} | {row.b_passed} | {row.c_passed} | "
            f"{row.min_swing_force:.2f} | {row.mean_swing_force:.2f} | {row.improvement_vs_v2a:.2f} | "
            f"{row.gate_reached_count} | {row.timeout_count} | {row.min_upright:.3f} | {row.impact_post:.3f} | "
            f"{row.base_drop_post:.5f} | {row.fail_reasons} |"
        )
    lines.extend(["", "## Decision", ""])
    if all(row.min_swing_force >= 38.0 or not row.stable for row in rows):
        lines.append(
            "All six v2b checks failed to move stable min_swing_force clearly below 38N. "
            "Current control-channel authority is insufficient; keep Seedon on grounded/forward shuffle."
        )
    elif any(row.stable and row.min_swing_force <= 35.0 for row in rows):
        lines.append(
            f"At least one stable check reached <=35N. Best candidate: {best.channel_set}/{best.gain}."
        )
    else:
        lines.append(
            f"Some stable improvement below 38N exists, but no <=35N gate. Best candidate: {best.channel_set}/{best.gain}."
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--vecnorm-path", type=Path, default=DEFAULT_VECNORM)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--audit-warmup-steps", type=int, default=20)
    parser.add_argument("--target-force", type=float, default=35.0)
    parser.add_argument("--double-support-steps", type=int, default=40)
    parser.add_argument("--preload-steps", type=int, default=80)
    parser.add_argument("--unload-steps", type=int, default=160)
    parser.add_argument("--hold-steps", type=int, default=50)
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
    v2a_config = UnloadConfig(
        target_force_n=args.target_force,
        double_support_steps=args.double_support_steps,
        preload_steps=args.preload_steps,
        unload_steps=args.unload_steps,
        hold_steps=args.hold_steps,
        feedback_gain=0.0012,
        correction_step_limit=0.0025,
        support_correction_limit=0.07,
        swing_correction_limit=0.035,
        lean_correction_limit=0.035,
        support_base=args.support_base,
        swing_base=args.swing_base,
        lean_base=args.lean_base,
    )
    v2a = run_unload(
        args.config,
        v2a_config,
        out_dir=args.out_dir / "v2a_baseline",
        steps=args.steps,
        seed=args.seed,
        warmup_steps=args.audit_warmup_steps,
        teacher_impact=teacher.landing_impact_post_warmup,
    )
    rows: list[V2BResult] = []
    for channel_set in CHANNEL_SETS:
        for gain in GAIN_PROFILES:
            row = run_v2b(
                args.config,
                v2a_config,
                out_dir=args.out_dir,
                steps=args.steps,
                seed=args.seed,
                warmup_steps=args.audit_warmup_steps,
                teacher_impact=teacher.landing_impact_post_warmup,
                v2a_min_swing_force=v2a.min_swing_force,
                channel_set=channel_set,
                gain=gain,
            )
            rows.append(row)
            print(
                f"{channel_set.name}/{gain.name} stable={row.stable} A={row.a_passed} B={row.b_passed} C={row.c_passed} "
                f"min={row.min_swing_force:.2f} mean={row.mean_swing_force:.2f} "
                f"improve={row.improvement_vs_v2a:.2f} gates={row.gate_reached_count} "
                f"impact={row.impact_post:.3f} upright={row.min_upright:.3f} reasons={row.fail_reasons or '-'}"
            )
    write_results(args.out_dir / "unload_controller_v2b_final_check.csv", rows)
    write_summary(args.out_dir / "summary.md", rows, v2a.min_swing_force, v2a.mean_swing_force_unload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
