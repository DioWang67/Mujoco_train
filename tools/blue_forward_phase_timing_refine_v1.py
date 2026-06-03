"""Refine timing and phase force windows for grounded forward shuffle."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from seedon_baseline.env import SeedonStandingConfig, SeedonStandingEnv
from tools.audit_seedon_shuffle_v0 import _count_contact_none_bursts, _load_config, audit_shuffle
from tools.blue_forward_shuffle_authority_sweep_v1 import (
    DEFAULT_TARGET_VELOCITY,
    PolicyProvider,
    _config_for_cadence,
    _contact_state,
    _fmt,
)
from tools.blue_forward_shuffle_v1 import DEFAULT_CONFIG, DEFAULT_MODEL, DEFAULT_VECNORM
from tools.render_seedon_policy_comparison import _make_side_camera, _save_mp4


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "artifacts" / "seedon_debug" / "blue_forward_phase_timing_refine_v1"


@dataclass(frozen=True)
class TimingCandidate:
    """One timing-local forward-shaping candidate."""

    candidate_id: str
    cadence_scale: float
    timing_offset_steps: int
    duty_cycle: float
    right_force_n: float
    left_force_n: float
    apply_location: str
    phase_gate: str
    config_path: Path


@dataclass(frozen=True)
class TimingAudit:
    """Audit metrics for one timing-local forward-shaping candidate."""

    candidate_id: str
    cadence_scale: float
    timing_offset_steps: int
    duty_cycle: float
    right_force_n: float
    left_force_n: float
    apply_location: str
    phase_gate: str
    mean_forward_velocity: float
    forward_displacement: float
    lateral_drift: float
    yaw_drift: float
    foot_sliding_distance: float
    sliding_per_forward_meter: float
    contact_none_ratio: float
    jump_count: int
    min_upright: float
    base_height_drop_post_warmup: float
    landing_impact_post_warmup: float
    peak_support_ratio: float
    passed: bool
    fail_reasons: str
    score: float
    config_path: str
    timeline_path: str
    render_path: str


def _parse_float_list(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one float")
    return values


def _parse_int_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one int")
    return values


def _parse_str_list(raw: str) -> list[str]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one value")
    return values


def _yaw_from_qpos(qpos: np.ndarray) -> float:
    """Return floating-base yaw from root quaternion."""

    qw, qx, qy, qz = [float(value) for value in qpos[3:7]]
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def _support_local_step(env: SeedonStandingEnv, phase: dict[str, object]) -> tuple[int, int]:
    """Return approximate local support step and duration from reference phase alpha."""

    alpha = float(phase.get("phase_alpha", 0.0))
    seed = env._require_reference_gait_seed()
    phase_name = str(phase["phase_name"])
    duration = 1
    for keyframe in seed.keyframes:
        if keyframe.name == phase_name:
            duration = max(1, int(keyframe.duration_steps))
            break
    local_step = int(np.clip(round(alpha * duration) - 1, 0, duration - 1))
    return local_step, duration


def _timing_gate_active(candidate: TimingCandidate, env: SeedonStandingEnv, phase: dict[str, object]) -> bool:
    support = str(phase["support_side"])
    if support not in {"right", "left"}:
        return False
    if candidate.phase_gate == "right_support" and support != "right":
        return False
    if candidate.phase_gate == "alternating_support":
        pass
    elif candidate.phase_gate != "right_support":
        raise ValueError(f"Unsupported phase gate: {candidate.phase_gate}")
    local_step, duration = _support_local_step(env, phase)
    window = max(1, int(round(duration * candidate.duty_cycle)))
    center = duration // 2 + candidate.timing_offset_steps
    start = int(np.clip(center - window // 2, 0, duration - 1))
    end = int(np.clip(start + window, 1, duration))
    return start <= local_step < end


def _force_for_support(candidate: TimingCandidate, support_side: str) -> float:
    if support_side == "right":
        return candidate.right_force_n
    if support_side == "left":
        return candidate.left_force_n
    return 0.0


def _body_id(env: SeedonStandingEnv, candidate: TimingCandidate) -> int:
    if candidate.apply_location in {"base_link", "base_com"}:
        return int(env._base_body_id)
    raise ValueError(f"Unsupported apply location: {candidate.apply_location}")


def _apply_force(env: SeedonStandingEnv, candidate: TimingCandidate, phase: dict[str, object]) -> float:
    env.data.xfrc_applied[:] = 0.0
    if not _timing_gate_active(candidate, env, phase):
        return 0.0
    force = _force_for_support(candidate, str(phase["support_side"]))
    env.data.xfrc_applied[_body_id(env, candidate), 0] = force
    return force


def _row(step: int, env: SeedonStandingEnv, info: dict[str, Any], applied_force: float, right_slide: float, left_slide: float) -> dict[str, Any]:
    return {
        "step": step,
        "phase_name": str(info["phase_name"]),
        "support_side": str(info["support_side"]),
        "base_x": float(info["base_x_position"]),
        "base_y": float(info["base_y_position"]),
        "base_yaw": _yaw_from_qpos(env.data.qpos),
        "forward_velocity": float(info["forward_velocity"]),
        "support_ratio": float(info["force_ratio"]),
        "contact_state": _contact_state(info),
        "right_contact": bool(info["right_contact"]),
        "left_contact": bool(info["left_contact"]),
        "base_height": float(info["base_height"]),
        "upright": float(info["upright"]),
        "total_contact_force": float(info["right_normal_force"]) + float(info["left_normal_force"]),
        "right_slide_delta": right_slide,
        "left_slide_delta": left_slide,
        "applied_force_n": applied_force,
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


def _render_candidate(path: Path, candidate: TimingCandidate, config: SeedonStandingConfig, policy_provider: PolicyProvider, *, steps: int, seed: int, fps: int, width: int, height: int) -> None:
    """Render one timing candidate from a fixed side camera."""

    env = SeedonStandingEnv(reset_noise_scale=0.0, reward_config=config)
    renderer = mujoco.Renderer(env.model, height=height, width=width)
    camera = _make_side_camera()
    frames: list[np.ndarray] = []
    try:
        obs, _ = env.reset(seed=seed)
        action_provider = policy_provider.bind(env)
        for _ in range(steps):
            phase = env._task_phase_metadata()
            _apply_force(env, candidate, phase)
            action = action_provider(obs)
            obs, _, terminated, truncated, _ = env.step(action)
            env.data.xfrc_applied[:] = 0.0
            renderer.update_scene(env.data, camera=camera)
            frames.append(np.asarray(renderer.render(), dtype=np.uint8))
            if terminated or truncated:
                break
    finally:
        renderer.close()
        env.close()
    _save_mp4(path, frames, fps)


def audit_candidate(
    candidate: TimingCandidate,
    *,
    config: SeedonStandingConfig,
    policy_provider: PolicyProvider,
    out_dir: Path,
    steps: int,
    seed: int,
    warmup_steps: int,
    teacher_impact: float,
    max_sliding_per_forward_meter: float,
    max_lateral_drift: float,
    max_yaw_drift: float,
) -> TimingAudit:
    """Run one local timing-refine candidate."""

    env = SeedonStandingEnv(reset_noise_scale=0.0, reward_config=config)
    dt = float(env.dt)
    robot_weight = float(np.sum(env.model.body_mass) * 9.81)
    rows: list[dict[str, Any]] = []
    right_slide_total = 0.0
    left_slide_total = 0.0
    initial_x = 0.0
    initial_y = 0.0
    initial_yaw = 0.0
    try:
        obs, _ = env.reset(seed=seed)
        action_provider = policy_provider.bind(env)
        initial_x = float(env.data.qpos[0])
        initial_y = float(env.data.qpos[1])
        initial_yaw = _yaw_from_qpos(env.data.qpos)
        previous_right_xy = env.data.geom_xpos[env._foot_geom_ids[0]][:2].copy()
        previous_left_xy = env.data.geom_xpos[env._foot_geom_ids[1]][:2].copy()
        for step in range(1, steps + 1):
            phase = env._task_phase_metadata()
            applied_force = _apply_force(env, candidate, phase)
            action = action_provider(obs)
            obs, _, terminated, truncated, info = env.step(action)
            env.data.xfrc_applied[:] = 0.0
            right_xy = env.data.geom_xpos[env._foot_geom_ids[0]][:2].copy()
            left_xy = env.data.geom_xpos[env._foot_geom_ids[1]][:2].copy()
            right_slide = float(np.linalg.norm(right_xy - previous_right_xy)) if bool(info["right_contact"]) else 0.0
            left_slide = float(np.linalg.norm(left_xy - previous_left_xy)) if bool(info["left_contact"]) else 0.0
            right_slide_total += right_slide
            left_slide_total += left_slide
            previous_right_xy = right_xy
            previous_left_xy = left_xy
            rows.append(_row(step, env, dict(info), applied_force, right_slide, left_slide))
            if terminated or truncated:
                break
    finally:
        env.close()

    timeline_path = out_dir / "timelines" / f"{candidate.candidate_id}.csv"
    _write_timeline(timeline_path, rows)
    post = rows[warmup_steps:] or rows
    final_x = float(rows[-1]["base_x"]) if rows else initial_x
    final_y = float(rows[-1]["base_y"]) if rows else initial_y
    final_yaw = float(rows[-1]["base_yaw"]) if rows else initial_yaw
    displacement = final_x - initial_x
    lateral_drift = final_y - initial_y
    yaw_drift = float(np.arctan2(np.sin(final_yaw - initial_yaw), np.cos(final_yaw - initial_yaw)))
    mean_velocity = displacement / max(len(rows), 1) / max(dt, 1e-9)
    sliding = right_slide_total + left_slide_total
    sliding_ratio = sliding / max(abs(displacement), 1e-6)
    contact_none_steps = sum(1 for row in rows if row["contact_state"] == "none")
    jump_count = _count_contact_none_bursts(
        [{"right_contact": bool(row["right_contact"]), "left_contact": bool(row["left_contact"])} for row in rows]
    )
    min_upright = min((float(row["upright"]) for row in rows), default=0.0)
    base0 = float(post[0]["base_height"]) if post else 0.0
    base_drop = max(0.0, base0 - min((float(row["base_height"]) for row in post), default=base0))
    max_force = max((float(row["total_contact_force"]) for row in post), default=0.0)
    impact = max_force / max(robot_weight, 1e-6)
    support = [float(row["support_ratio"]) for row in rows]
    fail: list[str] = []
    if mean_velocity < 0.005:
        fail.append("forward_velocity")
    if contact_none_steps:
        fail.append("contact_none")
    if jump_count:
        fail.append("jump")
    if min_upright < 0.99:
        fail.append("upright")
    if base_drop > 0.015:
        fail.append("base_drop")
    if impact > teacher_impact * 1.2:
        fail.append("landing_impact")
    if abs(lateral_drift) > max_lateral_drift:
        fail.append("lateral_drift")
    if abs(yaw_drift) > max_yaw_drift:
        fail.append("yaw_drift")
    if sliding_ratio > max_sliding_per_forward_meter:
        fail.append("foot_sliding")
    passed = not fail
    score = (
        (10_000.0 if passed else 0.0)
        + mean_velocity * 1000.0
        - max(0.0, sliding_ratio - 1.5) * 20.0
        - abs(lateral_drift) * 100.0
        - abs(yaw_drift) * 100.0
        - max(0.0, 0.99 - min_upright) * 300.0
    )
    return TimingAudit(
        candidate_id=candidate.candidate_id,
        cadence_scale=candidate.cadence_scale,
        timing_offset_steps=candidate.timing_offset_steps,
        duty_cycle=candidate.duty_cycle,
        right_force_n=candidate.right_force_n,
        left_force_n=candidate.left_force_n,
        apply_location=candidate.apply_location,
        phase_gate=candidate.phase_gate,
        mean_forward_velocity=mean_velocity,
        forward_displacement=displacement,
        lateral_drift=lateral_drift,
        yaw_drift=yaw_drift,
        foot_sliding_distance=sliding,
        sliding_per_forward_meter=sliding_ratio,
        contact_none_ratio=contact_none_steps / max(1, len(rows)),
        jump_count=jump_count,
        min_upright=min_upright,
        base_height_drop_post_warmup=base_drop,
        landing_impact_post_warmup=impact,
        peak_support_ratio=max(support, default=0.0),
        passed=passed,
        fail_reasons=",".join(fail),
        score=score,
        config_path=str(candidate.config_path),
        timeline_path=str(timeline_path),
        render_path="",
    )


def build_candidates(args: argparse.Namespace, base_config: SeedonStandingConfig) -> tuple[list[TimingCandidate], dict[float, SeedonStandingConfig]]:
    configs: dict[float, SeedonStandingConfig] = {}
    config_paths: dict[float, Path] = {}
    for cadence in args.cadence_scales:
        config, path = _config_for_cadence(base_config, cadence, args.out_dir)
        configs[cadence] = config
        config_paths[cadence] = path
    candidates: list[TimingCandidate] = []
    for cadence, offset, duty, right_force, left_force, location, phase_gate in product(
        args.cadence_scales,
        args.timing_offsets,
        args.duty_cycles,
        args.right_forces,
        args.left_forces,
        args.apply_locations,
        args.phase_gates,
    ):
        if abs(right_force - left_force) > args.max_force_asymmetry:
            continue
        candidate_id = (
            f"cad{_fmt(cadence)}_off{offset:+d}_duty{_fmt(duty)}"
            f"_rf{_fmt(right_force)}_lf{_fmt(left_force)}_{location}_{phase_gate}"
        ).replace("+", "p").replace("-", "m")
        candidates.append(
            TimingCandidate(
                candidate_id=candidate_id,
                cadence_scale=cadence,
                timing_offset_steps=offset,
                duty_cycle=duty,
                right_force_n=right_force,
                left_force_n=left_force,
                apply_location=location,
                phase_gate=phase_gate,
                config_path=config_paths[cadence],
            )
        )
    if len(candidates) > args.max_candidates:
        raise ValueError(f"Candidate count {len(candidates)} exceeds --max-candidates {args.max_candidates}")
    return candidates, configs


def write_results(path: Path, rows: list[TimingAudit]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_summary(path: Path, rows: list[TimingAudit], teacher_impact: float) -> None:
    ranked = sorted(rows, key=lambda row: row.score, reverse=True)
    lines = [
        "# blue_forward_phase_timing_refine_v1",
        "",
        f"candidates: {len(rows)}",
        f"passed: {sum(row.passed for row in rows)}",
        f"teacher_landing_impact_post_warmup: {teacher_impact:.6f}",
        f"max_safe_ranked_velocity: {ranked[0].mean_forward_velocity if ranked else 0.0:.6f}",
        "",
        "| candidate | pass | v | dx | lat | yaw | slide/m | impact | drop | upright | reasons |",
        "|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in ranked[:10]:
        lines.append(
            f"| {row.candidate_id} | {row.passed} | {row.mean_forward_velocity:.5f} | "
            f"{row.forward_displacement:.4f} | {row.lateral_drift:.4f} | {row.yaw_drift:.4f} | "
            f"{row.sliding_per_forward_meter:.2f} | {row.landing_impact_post_warmup:.3f} | "
            f"{row.base_height_drop_post_warmup:.4f} | {row.min_upright:.3f} | {row.fail_reasons} |"
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
    parser.add_argument("--cadence-scales", type=_parse_float_list, default="1.4,1.5,1.6,1.7")
    parser.add_argument("--timing-offsets", type=_parse_int_list, default="-20,-10,0,10,20")
    parser.add_argument("--duty-cycles", type=_parse_float_list, default="0.3,0.5,0.7")
    parser.add_argument("--right-forces", type=_parse_float_list, default="2,3")
    parser.add_argument("--left-forces", type=_parse_float_list, default="1,2,3")
    parser.add_argument("--apply-locations", type=_parse_str_list, default="base_link,base_com")
    parser.add_argument("--phase-gates", type=_parse_str_list, default="right_support,alternating_support")
    parser.add_argument("--max-force-asymmetry", type=float, default=1.0)
    parser.add_argument("--max-candidates", type=int, default=1200)
    parser.add_argument("--max-sliding-per-forward-meter", type=float, default=1.5)
    parser.add_argument("--max-lateral-drift", type=float, default=0.04)
    parser.add_argument("--max-yaw-drift", type=float, default=0.12)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--render-top-k", type=int, default=3)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=368)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    base_config = _load_config(args.config)
    teacher = audit_shuffle(
        args.config,
        args.model_path,
        args.vecnorm_path,
        args.steps,
        args.seed,
        audit_warmup_steps=args.audit_warmup_steps,
    )
    candidates, configs = build_candidates(args, base_config)
    policy_provider = PolicyProvider(args.model_path, args.vecnorm_path)
    rows: list[TimingAudit] = []
    print(f"candidates={len(candidates)}")
    for index, candidate in enumerate(candidates, start=1):
        row = audit_candidate(
            candidate,
            config=configs[candidate.cadence_scale],
            policy_provider=policy_provider,
            out_dir=args.out_dir,
            steps=args.steps,
            seed=args.seed,
            warmup_steps=args.audit_warmup_steps,
            teacher_impact=teacher.landing_impact_post_warmup,
            max_sliding_per_forward_meter=args.max_sliding_per_forward_meter,
            max_lateral_drift=args.max_lateral_drift,
            max_yaw_drift=args.max_yaw_drift,
        )
        rows.append(row)
        if index == 1 or index == len(candidates) or index % args.progress_every == 0:
            print(
                f"[{index}/{len(candidates)}] {row.candidate_id}: pass={row.passed} "
                f"v={row.mean_forward_velocity:.5f} slide/m={row.sliding_per_forward_meter:.2f} "
                f"lat={row.lateral_drift:.4f} yaw={row.yaw_drift:.4f} reasons={row.fail_reasons or '-'}"
            )
    ranked = sorted(rows, key=lambda row: row.score, reverse=True)
    render_dir = args.out_dir / "render"
    for row in ranked[: args.render_top_k]:
        candidate = next(item for item in candidates if item.candidate_id == row.candidate_id)
        render_path = render_dir / f"{row.candidate_id}.mp4"
        _render_candidate(
            render_path,
            candidate,
            configs[candidate.cadence_scale],
            policy_provider,
            steps=args.steps,
            seed=args.seed,
            fps=args.fps,
            width=args.width,
            height=args.height,
        )
        updated = asdict(row)
        updated["render_path"] = str(render_path)
        rows[rows.index(row)] = TimingAudit(**updated)
        print(f"rendered: {render_path}")
    ranked = sorted(rows, key=lambda row: row.score, reverse=True)
    write_results(args.out_dir / "blue_forward_phase_timing_refine_v1.csv", rows)
    write_results(args.out_dir / "blue_forward_phase_timing_refine_v1_top10.csv", ranked[: args.top_k])
    write_summary(args.out_dir / "summary.md", rows, teacher.landing_impact_post_warmup)
    for row in ranked[: args.top_k]:
        print(
            f"TOP {row.candidate_id}: pass={row.passed} v={row.mean_forward_velocity:.5f} "
            f"dx={row.forward_displacement:.4f} lat={row.lateral_drift:.4f} yaw={row.yaw_drift:.4f} "
            f"slide/m={row.sliding_per_forward_meter:.2f} impact={row.landing_impact_post_warmup:.3f} "
            f"upright={row.min_upright:.3f} reasons={row.fail_reasons or '-'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
