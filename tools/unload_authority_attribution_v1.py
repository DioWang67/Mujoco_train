"""Attribute Seedon unload authority to individual control channels.

This tool runs the stable v2a unload controller and injects one small
single-channel correction during UNLOAD. It is intentionally diagnostic:
it does not train, does not change lift targets, and does not compose
multi-channel controllers.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from seedon_baseline.env import SeedonStandingEnv
from tools.audit_seedon_shuffle_v0 import _count_contact_none_bursts, _load_config, audit_shuffle
from tools.blue_forward_shuffle_v1 import DEFAULT_CONFIG, DEFAULT_MODEL, DEFAULT_VECNORM
from tools.seedon_explicit_locomotion_controller_v2 import (
    L_ANKLE,
    L_HIP_PITCH,
    L_HIP_ROLL,
    L_KNEE,
    R_ANKLE,
    R_HIP_PITCH,
    R_HIP_ROLL,
    R_KNEE,
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
    _target,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "artifacts" / "seedon_debug" / "unload_authority_attribution_v1"


@dataclass(frozen=True)
class ChannelSpec:
    """Single control channel and how it maps to the active swing side."""

    name: str
    apply_delta: Callable[[np.ndarray, str, float], None]


@dataclass(frozen=True)
class RolloutMetrics:
    """Aggregate metrics for one baseline or single-channel perturbation rollout."""

    channel: str
    delta: float
    stable: bool
    fail_reasons: str
    mean_swing_force_unload: float
    min_swing_force_unload: float
    mean_support_force_unload: float
    base_height_drop_post: float
    min_upright: float
    impact_post: float
    contact_none_ratio: float
    jump_count: int
    ctrl_saturation_max: float
    force_saturation_max: float
    swing_force_delta: float
    support_force_delta: float
    base_height_delta: float
    upright_delta: float
    impact_delta: float
    force_reduction: float
    posture_cost: float
    efficiency: float
    timeline_path: str


def _parse_float_list(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one float")
    return values


def _support_roll(target: np.ndarray, swing_side: str, delta: float) -> None:
    target[L_HIP_ROLL if swing_side == "right" else R_HIP_ROLL] += delta if swing_side == "right" else -delta


def _swing_roll(target: np.ndarray, swing_side: str, delta: float) -> None:
    target[R_HIP_ROLL if swing_side == "right" else L_HIP_ROLL] += delta if swing_side == "right" else -delta


def _lean_proxy(target: np.ndarray, swing_side: str, delta: float) -> None:
    signed = delta if swing_side == "right" else -delta
    target[R_HIP_ROLL] += signed
    target[L_HIP_ROLL] += signed


def _stance_knee(target: np.ndarray, swing_side: str, delta: float) -> None:
    target[L_KNEE if swing_side == "right" else R_KNEE] += delta


def _stance_ankle(target: np.ndarray, swing_side: str, delta: float) -> None:
    target[L_ANKLE if swing_side == "right" else R_ANKLE] += delta


def _support_pitch(target: np.ndarray, swing_side: str, delta: float) -> None:
    target[L_HIP_PITCH if swing_side == "right" else R_HIP_PITCH] += delta


def _swing_knee(target: np.ndarray, swing_side: str, delta: float) -> None:
    target[R_KNEE if swing_side == "right" else L_KNEE] += delta


def _swing_ankle(target: np.ndarray, swing_side: str, delta: float) -> None:
    target[R_ANKLE if swing_side == "right" else L_ANKLE] += delta


CHANNELS: tuple[ChannelSpec, ...] = (
    ChannelSpec("support_hip_roll", _support_roll),
    ChannelSpec("swing_hip_roll", _swing_roll),
    ChannelSpec("lean_proxy", _lean_proxy),
    ChannelSpec("stance_knee_pitch", _stance_knee),
    ChannelSpec("stance_ankle_pitch", _stance_ankle),
    ChannelSpec("support_hip_pitch", _support_pitch),
    ChannelSpec("swing_knee_pitch", _swing_knee),
    ChannelSpec("swing_ankle_pitch", _swing_ankle),
)


def _write_timeline(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _safe_name(channel: str, delta: float) -> str:
    sign = "p" if delta >= 0.0 else "m"
    return f"{channel}_{sign}{abs(delta):.4f}".replace(".", "p")


def _row(step: int, env: SeedonStandingEnv, runtime: UnloadRuntime, event: str, robot_weight: float) -> dict[str, Any]:
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
        "support_ratio": support / max(support + swing, 1e-9),
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


def _summarize_rows(
    rows: list[dict[str, Any]],
    *,
    channel: str,
    delta: float,
    warmup_steps: int,
    teacher_impact: float,
    baseline: RolloutMetrics | None,
    timeline_path: Path,
) -> RolloutMetrics:
    post = rows[warmup_steps:] or rows
    unload_rows = [row for row in rows if str(row["phase"]).startswith("UNLOAD")]
    swing_values = [float(row["swing_force"]) for row in unload_rows]
    support_values = [float(row["support_force"]) for row in unload_rows]
    contact_none_steps = sum(1 for row in rows if row["contact_state"] == "none")
    jump_count = _count_contact_none_bursts(rows)
    base0 = float(post[0]["base_height"]) if post else 0.0
    base_drop = max(0.0, base0 - min((float(row["base_height"]) for row in post), default=base0))
    impact = max((float(row["impact"]) for row in post), default=0.0)
    min_upright = min((float(row["upright"]) for row in rows), default=0.0)
    ctrl_sat = max((float(row["ctrl_saturation"]) for row in rows), default=0.0)
    force_sat = max((float(row["force_saturation"]) for row in rows), default=0.0)
    mean_swing = float(np.mean(swing_values)) if swing_values else float("inf")
    min_swing = min(swing_values, default=float("inf"))
    mean_support = float(np.mean(support_values)) if support_values else float("inf")
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
    if baseline is None:
        swing_delta = 0.0
        support_delta = 0.0
        base_delta = 0.0
        upright_delta = 0.0
        impact_delta = 0.0
        force_reduction = 0.0
        posture_cost = 1.0
        efficiency = 0.0
    else:
        swing_delta = mean_swing - baseline.mean_swing_force_unload
        support_delta = mean_support - baseline.mean_support_force_unload
        base_delta = base_drop - baseline.base_height_drop_post
        upright_delta = min_upright - baseline.min_upright
        impact_delta = impact - baseline.impact_post
        force_reduction = baseline.mean_swing_force_unload - mean_swing
        posture_cost = (
            max(0.0, -upright_delta) * 100.0
            + max(0.0, base_delta) * 100.0
            + max(0.0, impact_delta) * 4.0
            + contact_none_steps * 25.0
            + jump_count * 50.0
            + (100.0 if ctrl_sat != 0.0 or force_sat != 0.0 else 0.0)
            + 0.1
        )
        efficiency = force_reduction / posture_cost
    return RolloutMetrics(
        channel=channel,
        delta=delta,
        stable=not stable_fail,
        fail_reasons=",".join(stable_fail),
        mean_swing_force_unload=mean_swing,
        min_swing_force_unload=min_swing,
        mean_support_force_unload=mean_support,
        base_height_drop_post=base_drop,
        min_upright=min_upright,
        impact_post=impact,
        contact_none_ratio=contact_none_steps / max(1, len(rows)),
        jump_count=jump_count,
        ctrl_saturation_max=ctrl_sat,
        force_saturation_max=force_sat,
        swing_force_delta=swing_delta,
        support_force_delta=support_delta,
        base_height_delta=base_delta,
        upright_delta=upright_delta,
        impact_delta=impact_delta,
        force_reduction=force_reduction,
        posture_cost=posture_cost,
        efficiency=efficiency,
        timeline_path=str(timeline_path),
    )


def run_case(
    config_path: Path,
    unload_config: UnloadConfig,
    *,
    out_dir: Path,
    steps: int,
    seed: int,
    warmup_steps: int,
    teacher_impact: float,
    channel: ChannelSpec | None,
    delta: float,
    baseline: RolloutMetrics | None,
) -> RolloutMetrics:
    """Run one attribution case and return aggregate metrics."""

    env = SeedonStandingEnv(reset_noise_scale=0.0, reward_config=_load_config(config_path))
    runtime = UnloadRuntime()
    rows: list[dict[str, Any]] = []
    case_name = "baseline" if channel is None else _safe_name(channel.name, delta)
    timeline_path = out_dir / "timelines" / f"{case_name}.csv"
    try:
        env.reset(seed=seed)
        robot_weight = float(np.sum(env.model.body_mass) * 9.81)
        for step in range(1, steps + 1):
            target, _ = _target(env, runtime, unload_config)
            if channel is not None and runtime.phase in {UnloadPhase.UNLOAD_RIGHT, UnloadPhase.UNLOAD_LEFT}:
                target = target.copy()
                channel.apply_delta(target, runtime.swing_side, delta)
                target = env._apply_safe_joint_target_clamps(target)
            env._do_pd_simulation(target)
            env._gait_step += 1
            event = _advance(runtime, env, unload_config)
            rows.append(_row(step, env, runtime, event, robot_weight))
    finally:
        env.close()
    _write_timeline(timeline_path, rows)
    return _summarize_rows(
        rows,
        channel="baseline" if channel is None else channel.name,
        delta=delta,
        warmup_steps=warmup_steps,
        teacher_impact=teacher_impact,
        baseline=baseline,
        timeline_path=timeline_path,
    )


def write_results(path: Path, rows: list[RolloutMetrics]) -> None:
    """Write attribution CSV rows."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_summary(path: Path, baseline: RolloutMetrics, rows: list[RolloutMetrics]) -> None:
    """Write human-readable attribution summary."""

    ranked = sorted(
        [row for row in rows if row.channel != "baseline" and row.stable and row.force_reduction > 0.0],
        key=lambda row: (row.efficiency, row.force_reduction),
        reverse=True,
    )
    lines = [
        "# Seedon Unload Authority Attribution V1",
        "",
        f"baseline_mean_swing_force_unload: {baseline.mean_swing_force_unload:.3f} N",
        f"baseline_min_swing_force_unload: {baseline.min_swing_force_unload:.3f} N",
        f"baseline_upright: {baseline.min_upright:.3f}",
        f"baseline_impact_post: {baseline.impact_post:.3f}",
        "",
        "## Top Stable Channels",
        "",
        "| rank | channel | delta | force_reduction | mean_swing | min_swing | efficiency | upright_delta | impact_delta | reasons |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for index, row in enumerate(ranked[:10], start=1):
        lines.append(
            f"| {index} | {row.channel} | {row.delta:.4f} | {row.force_reduction:.3f} | "
            f"{row.mean_swing_force_unload:.3f} | {row.min_swing_force_unload:.3f} | "
            f"{row.efficiency:.3f} | {row.upright_delta:.4f} | {row.impact_delta:.4f} | {row.fail_reasons} |"
        )
    if not ranked:
        lines.append("| - | none | 0 | 0 | 0 | 0 | 0 | 0 | 0 | no stable force-reducing channel |")
    lines.extend(
        [
            "",
            "Allowed for v2b consideration: top 2-3 stable channels only.",
            "",
        ]
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
    parser.add_argument("--deltas", type=_parse_float_list, default="0.005,0.015")
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
    unload_config = UnloadConfig(
        target_force_n=args.target_force,
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
    baseline = run_case(
        args.config,
        unload_config,
        out_dir=args.out_dir,
        steps=args.steps,
        seed=args.seed,
        warmup_steps=args.audit_warmup_steps,
        teacher_impact=teacher.landing_impact_post_warmup,
        channel=None,
        delta=0.0,
        baseline=None,
    )
    rows = [baseline]
    signed_deltas = [delta for value in args.deltas for delta in (value, -value)]
    for channel in CHANNELS:
        for delta in signed_deltas:
            row = run_case(
                args.config,
                unload_config,
                out_dir=args.out_dir,
                steps=args.steps,
                seed=args.seed,
                warmup_steps=args.audit_warmup_steps,
                teacher_impact=teacher.landing_impact_post_warmup,
                channel=channel,
                delta=delta,
                baseline=baseline,
            )
            rows.append(row)
            print(
                f"{channel.name} delta={delta:+.4f} stable={row.stable} "
                f"force_reduction={row.force_reduction:.3f} mean={row.mean_swing_force_unload:.2f} "
                f"min={row.min_swing_force_unload:.2f} eff={row.efficiency:.3f} reasons={row.fail_reasons or '-'}"
            )
    ranked = sorted(
        [row for row in rows if row.channel != "baseline" and row.stable and row.force_reduction > 0.0],
        key=lambda row: (row.efficiency, row.force_reduction),
        reverse=True,
    )
    write_results(args.out_dir / "unload_authority_attribution_v1.csv", rows)
    write_results(args.out_dir / "unload_authority_attribution_v1_top_channels.csv", [baseline, *ranked[:10]])
    write_summary(args.out_dir / "summary.md", baseline, rows)
    print(
        f"baseline mean={baseline.mean_swing_force_unload:.2f} min={baseline.min_swing_force_unload:.2f} "
        f"stable={baseline.stable}"
    )
    if ranked:
        allowed = ", ".join(f"{row.channel}({row.delta:+.4f})" for row in ranked[:3])
        print(f"top_v2b_candidates={allowed}")
    else:
        print("top_v2b_candidates=none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
