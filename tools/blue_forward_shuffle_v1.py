"""Audit low-speed forward drift on the grounded teacher shuffle."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from seedon_baseline.env import SeedonStandingConfig, SeedonStandingEnv
from tools.audit_seedon_shuffle_v0 import _count_contact_none_bursts, _load_config, audit_shuffle
from tools.render_seedon_policy_comparison import _make_side_camera, _save_mp4


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "seedon" / "reference_teacher_pose_1_4_imitation.json"
DEFAULT_MODEL = REPO_ROOT / "models" / "seedon" / "teacher_safe_baseline" / "model.zip"
DEFAULT_VECNORM = REPO_ROOT / "models" / "seedon" / "teacher_safe_baseline" / "vecnorm.pkl"
DEFAULT_OUT_DIR = REPO_ROOT / "artifacts" / "seedon_debug" / "blue_forward_shuffle_v1"


@dataclass(frozen=True)
class ForwardShuffleAudit:
    """Audit metrics for one low-speed forward-shuffle target."""

    target_forward_velocity: float
    steps: int
    terminated: bool
    termination_reason: str
    forward_displacement: float
    mean_forward_velocity: float
    final_forward_velocity: float
    velocity_error: float
    velocity_tracking_ratio: float
    target_velocity_reached: bool
    foot_sliding_distance: float
    right_foot_sliding_distance: float
    left_foot_sliding_distance: float
    sliding_per_forward_meter: float
    mean_support_ratio: float
    peak_support_ratio: float
    contact_none_ratio: float
    jump_count: int
    min_upright: float
    mean_base_height: float
    base_height_drop_post_warmup: float
    landing_impact_post_warmup: float
    max_contact_force_post_warmup: float
    mean_assist_force_n: float
    max_assist_force_n: float
    passed: bool
    fail_reasons: str
    config_path: str
    timeline_path: str
    render_path: str


def _parse_float_list(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one float")
    return values


def _load_policy_provider(model_path: Path, vecnorm_path: Path, env: SeedonStandingEnv) -> Callable[[np.ndarray], np.ndarray]:
    """Load a deterministic policy callable with frozen VecNormalize stats."""

    if not model_path.is_file():
        raise FileNotFoundError(f"Policy checkpoint not found: {model_path}")
    if not vecnorm_path.is_file():
        raise FileNotFoundError(f"VecNormalize file not found: {vecnorm_path}")
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    model = PPO.load(str(model_path))
    vec_env = DummyVecEnv([lambda: env])
    vecnorm = VecNormalize.load(str(vecnorm_path), vec_env)
    vecnorm.training = False
    vecnorm.norm_reward = False

    def predict(obs: np.ndarray) -> np.ndarray:
        norm_obs = vecnorm.normalize_obs(obs[None, :])
        action, _ = model.predict(norm_obs, deterministic=True)
        return np.asarray(action[0], dtype=np.float64)

    return predict


def _config_for_velocity(base: SeedonStandingConfig, velocity: float) -> SeedonStandingConfig:
    """Return config variant that records the requested forward curriculum target."""

    payload = dict(base.__dict__)
    payload["target_forward_velocity"] = float(velocity)
    payload["march_forward_velocity"] = float(velocity)
    payload["march_forward_progress_weight"] = 0.0
    payload["march_forward_velocity_weight"] = 0.0
    payload["march_swing_forward_weight"] = 0.0
    return SeedonStandingConfig(**payload)


def _write_config(path: Path, config: SeedonStandingConfig) -> None:
    """Write a compact JSON config variant."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config.__dict__, indent=2) + "\n", encoding="utf-8")


def _contact_state(info: dict[str, object]) -> str:
    right = bool(info["right_contact"])
    left = bool(info["left_contact"])
    if right and left:
        return "both"
    if right:
        return "right"
    if left:
        return "left"
    return "none"


def _assist_force(target_velocity: float, measured_velocity: float, *, kp: float, max_force: float) -> float:
    """Return clipped world-x assist force for deterministic velocity shaping."""

    return float(np.clip(kp * (target_velocity - measured_velocity), -max_force, max_force))


def _timeline_row(
    *,
    step: int,
    info: dict[str, object],
    env: SeedonStandingEnv,
    assist_force: float,
    right_slide_delta: float,
    left_slide_delta: float,
) -> dict[str, object]:
    """Build one per-step audit row."""

    return {
        "step": step,
        "base_x": float(info["base_x_position"]),
        "forward_velocity": float(info["forward_velocity"]),
        "target_side_force_ratio": float(info["target_side_force_ratio"]),
        "support_ratio": float(info["force_ratio"]),
        "swing_force_ratio": float(info["swing_force_ratio"]),
        "contact_state": _contact_state(info),
        "right_contact": bool(info["right_contact"]),
        "left_contact": bool(info["left_contact"]),
        "right_force": float(info["right_normal_force"]),
        "left_force": float(info["left_normal_force"]),
        "base_height": float(info["base_height"]),
        "base_roll": float(info["base_roll"]),
        "base_pitch": float(info["base_pitch"]),
        "upright": float(info["upright"]),
        "landing_impact_force": float(info["left_normal_force"]) + float(info["right_normal_force"]),
        "right_foot_x": float(env.data.geom_xpos[env._foot_geom_ids[0]][0]),
        "left_foot_x": float(env.data.geom_xpos[env._foot_geom_ids[1]][0]),
        "right_foot_y": float(env.data.geom_xpos[env._foot_geom_ids[0]][1]),
        "left_foot_y": float(env.data.geom_xpos[env._foot_geom_ids[1]][1]),
        "right_slide_delta": right_slide_delta,
        "left_slide_delta": left_slide_delta,
        "assist_force_n": assist_force,
    }


def _render_frame(renderer: object, env: SeedonStandingEnv, camera: object) -> np.ndarray:
    renderer.update_scene(env.data, camera=camera)
    return np.asarray(renderer.render(), dtype=np.uint8)


def rollout_velocity(
    *,
    config: SeedonStandingConfig,
    target_velocity: float,
    model_path: Path,
    vecnorm_path: Path,
    out_dir: Path,
    steps: int,
    seed: int,
    warmup_steps: int,
    fps: int,
    width: int,
    height: int,
    render: bool,
    force_kp: float,
    max_assist_force: float,
    teacher_landing_impact: float,
    max_sliding_per_forward_meter: float,
) -> ForwardShuffleAudit:
    """Run one deterministic low-speed forward-shuffle rollout."""

    import mujoco

    env = SeedonStandingEnv(reset_noise_scale=0.0, reward_config=config)
    action_provider = _load_policy_provider(model_path, vecnorm_path, env)
    dt = float(env.dt)
    robot_weight = float(np.sum(env.model.body_mass) * 9.81)
    renderer = mujoco.Renderer(env.model, height=height, width=width) if render else None
    camera = _make_side_camera() if render else None
    frames: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    right_slide = 0.0
    left_slide = 0.0
    assist_forces: list[float] = []
    terminated = False
    termination_reason = "none"
    initial_base_x = 0.0
    try:
        obs, _ = env.reset(seed=seed)
        initial_base_x = float(env.data.qpos[0])
        previous_right_xy = env.data.geom_xpos[env._foot_geom_ids[0]][:2].copy()
        previous_left_xy = env.data.geom_xpos[env._foot_geom_ids[1]][:2].copy()
        if render and renderer is not None and camera is not None:
            frames.append(_render_frame(renderer, env, camera))
        for step in range(1, steps + 1):
            measured_velocity = float(env.data.qvel[0])
            assist = _assist_force(
                target_velocity,
                measured_velocity,
                kp=force_kp,
                max_force=max_assist_force,
            )
            env.data.xfrc_applied[:] = 0.0
            env.data.xfrc_applied[env._base_body_id, 0] = assist
            action = action_provider(obs)
            obs, _, terminated, truncated, info = env.step(action)
            env.data.xfrc_applied[:] = 0.0
            right_xy = env.data.geom_xpos[env._foot_geom_ids[0]][:2].copy()
            left_xy = env.data.geom_xpos[env._foot_geom_ids[1]][:2].copy()
            right_delta = float(np.linalg.norm(right_xy - previous_right_xy)) if bool(info["right_contact"]) else 0.0
            left_delta = float(np.linalg.norm(left_xy - previous_left_xy)) if bool(info["left_contact"]) else 0.0
            right_slide += right_delta
            left_slide += left_delta
            previous_right_xy = right_xy
            previous_left_xy = left_xy
            assist_forces.append(abs(assist))
            rows.append(
                _timeline_row(
                    step=step,
                    info=dict(info),
                    env=env,
                    assist_force=assist,
                    right_slide_delta=right_delta,
                    left_slide_delta=left_delta,
                )
            )
            if render and renderer is not None and camera is not None:
                frames.append(_render_frame(renderer, env, camera))
            termination_reason = str(info.get("termination_reason", "none"))
            if terminated or truncated:
                terminated = True
                break
    finally:
        if renderer is not None:
            renderer.close()
        env.close()

    timeline_path = out_dir / "timelines" / f"target_{target_velocity:.3f}.csv"
    timeline_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with timeline_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    render_path = out_dir / "render" / f"target_{target_velocity:.3f}.mp4"
    if render:
        _save_mp4(render_path, frames[:steps], fps)
    else:
        render_path = Path("")

    post_rows = rows[warmup_steps:] or rows
    contact_none_steps = sum(1 for row in rows if row["contact_state"] == "none")
    jump_count = _count_contact_none_bursts(
        [
            {
                "right_contact": bool(row["right_contact"]),
                "left_contact": bool(row["left_contact"]),
            }
            for row in rows
        ]
    )
    final_base_x = float(rows[-1]["base_x"]) if rows else initial_base_x
    displacement = final_base_x - initial_base_x
    mean_velocity = displacement / max(len(rows), 1) / max(dt, 1e-9)
    sliding_total = right_slide + left_slide
    sliding_ratio = sliding_total / max(abs(displacement), 1e-6)
    base0 = float(post_rows[0]["base_height"]) if post_rows else 0.0
    base_drop = max(0.0, base0 - min((float(row["base_height"]) for row in post_rows), default=base0))
    max_force = max((float(row["landing_impact_force"]) for row in post_rows), default=0.0)
    landing_impact = max_force / max(robot_weight, 1e-6)
    support_ratios = [float(row["support_ratio"]) for row in rows]
    fail_reasons: list[str] = []
    if contact_none_steps != 0:
        fail_reasons.append("contact_none")
    if jump_count != 0:
        fail_reasons.append("jump")
    if min((float(row["upright"]) for row in rows), default=0.0) < 0.99:
        fail_reasons.append("upright")
    if base_drop > 0.015:
        fail_reasons.append("base_height_drop_post_warmup")
    if landing_impact > teacher_landing_impact * 1.2:
        fail_reasons.append("landing_impact_post_warmup")
    if sliding_ratio > max_sliding_per_forward_meter and abs(displacement) > 0.002:
        fail_reasons.append("foot_sliding")
    if displacement < 0.0:
        fail_reasons.append("backward_drift")
    return ForwardShuffleAudit(
        target_forward_velocity=target_velocity,
        steps=len(rows),
        terminated=terminated,
        termination_reason=termination_reason,
        forward_displacement=displacement,
        mean_forward_velocity=mean_velocity,
        final_forward_velocity=float(rows[-1]["forward_velocity"]) if rows else 0.0,
        velocity_error=float(target_velocity - mean_velocity),
        velocity_tracking_ratio=float(mean_velocity / max(target_velocity, 1e-9)),
        target_velocity_reached=bool(mean_velocity >= 0.8 * target_velocity),
        foot_sliding_distance=sliding_total,
        right_foot_sliding_distance=right_slide,
        left_foot_sliding_distance=left_slide,
        sliding_per_forward_meter=sliding_ratio,
        mean_support_ratio=float(np.mean(support_ratios)) if support_ratios else 0.0,
        peak_support_ratio=max(support_ratios, default=0.0),
        contact_none_ratio=contact_none_steps / max(1, len(rows)),
        jump_count=jump_count,
        min_upright=min((float(row["upright"]) for row in rows), default=0.0),
        mean_base_height=float(np.mean([float(row["base_height"]) for row in rows])) if rows else 0.0,
        base_height_drop_post_warmup=base_drop,
        landing_impact_post_warmup=landing_impact,
        max_contact_force_post_warmup=max_force,
        mean_assist_force_n=float(np.mean(assist_forces)) if assist_forces else 0.0,
        max_assist_force_n=max(assist_forces, default=0.0),
        passed=not fail_reasons,
        fail_reasons=",".join(fail_reasons),
        config_path=str(out_dir / "configs" / f"target_{target_velocity:.3f}.json"),
        timeline_path=str(timeline_path),
        render_path=str(render_path),
    )


def write_results(path: Path, rows: list[ForwardShuffleAudit]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_summary(path: Path, rows: list[ForwardShuffleAudit], teacher_impact: float) -> None:
    lines = [
        "# blue_forward_shuffle_v1",
        "",
        f"teacher_landing_impact_post_warmup: {teacher_impact:.6f}",
        f"candidates: {len(rows)}",
        f"passed: {sum(row.passed for row in rows)}",
        "",
        "| target_v | pass | displacement | mean_v | slide | slide/forward | impact | drop | upright | reasons |",
        "|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.target_forward_velocity:.3f} | {row.passed} | "
            f"{row.forward_displacement:.5f} | {row.mean_forward_velocity:.5f} "
            f"({row.velocity_tracking_ratio:.2f}x) | "
            f"{row.foot_sliding_distance:.5f} | {row.sliding_per_forward_meter:.2f} | "
            f"{row.landing_impact_post_warmup:.3f} | {row.base_height_drop_post_warmup:.5f} | "
            f"{row.min_upright:.3f} | {row.fail_reasons} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--vecnorm-path", type=Path, default=DEFAULT_VECNORM)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--target-velocities", type=_parse_float_list, default="0.005,0.01,0.015")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--audit-warmup-steps", type=int, default=20)
    parser.add_argument("--force-kp", type=float, default=20.0)
    parser.add_argument("--max-assist-force", type=float, default=1.0)
    parser.add_argument("--max-sliding-per-forward-meter", type=float, default=4.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=368)
    parser.add_argument("--no-render", action="store_true")
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
    rows: list[ForwardShuffleAudit] = []
    for velocity in args.target_velocities:
        config = _config_for_velocity(base_config, velocity)
        config_path = args.out_dir / "configs" / f"target_{velocity:.3f}.json"
        _write_config(config_path, config)
        row = rollout_velocity(
            config=config,
            target_velocity=velocity,
            model_path=args.model_path,
            vecnorm_path=args.vecnorm_path,
            out_dir=args.out_dir,
            steps=args.steps,
            seed=args.seed,
            warmup_steps=args.audit_warmup_steps,
            fps=args.fps,
            width=args.width,
            height=args.height,
            render=not args.no_render,
            force_kp=args.force_kp,
            max_assist_force=args.max_assist_force,
            teacher_landing_impact=teacher.landing_impact_post_warmup,
            max_sliding_per_forward_meter=args.max_sliding_per_forward_meter,
        )
        rows.append(row)
        print(
            f"target={velocity:.3f} pass={row.passed} dx={row.forward_displacement:.5f} "
            f"v={row.mean_forward_velocity:.5f} slide={row.foot_sliding_distance:.5f} "
            f"impact={row.landing_impact_post_warmup:.3f} upright={row.min_upright:.3f} "
            f"reasons={row.fail_reasons or '-'}"
        )
    write_results(args.out_dir / "blue_forward_shuffle_v1.csv", rows)
    write_summary(args.out_dir / "summary.md", rows, teacher.landing_impact_post_warmup)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
