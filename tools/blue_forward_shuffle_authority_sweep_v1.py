"""Sweep forward-shaping authority for grounded Blue-like shuffle."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from sedon_baseline.env import SedonStandingConfig, SedonStandingEnv
from tools.audit_sedon_shuffle_v0 import _count_contact_none_bursts, _load_config, audit_shuffle
from tools.blue_forward_shuffle_v1 import DEFAULT_CONFIG, DEFAULT_MODEL, DEFAULT_VECNORM


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "artifacts" / "sedon_debug" / "blue_forward_shuffle_authority_sweep_v1"
DEFAULT_FORCES = "0.5,1,2,3"
DEFAULT_LOCATIONS = "base_link,base_com,stance_foot_phase_only"
DEFAULT_PHASE_GATES = "always,right_support,left_support,alternating_support"
DEFAULT_CADENCE_SCALES = "1.0,1.2,1.5"
DEFAULT_TARGET_VELOCITY = 0.01
SLIDING_BASELINE_PER_FORWARD_METER = 2.1


@dataclass(frozen=True)
class AuthorityCandidate:
    """One forward-shaping authority candidate."""

    candidate_id: str
    force_n: float
    apply_location: str
    phase_gate: str
    cadence_scale: float
    config_path: Path


@dataclass(frozen=True)
class AuthorityAudit:
    """Audit metrics for one forward-shaping authority candidate."""

    candidate_id: str
    force_n: float
    apply_location: str
    phase_gate: str
    cadence_scale: float
    mean_forward_velocity: float
    forward_displacement: float
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
    terminated: bool
    termination_reason: str
    passed: bool
    fail_reasons: str
    score: float
    config_path: str
    timeline_path: str


def _parse_float_list(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one float")
    return values


def _parse_str_list(raw: str) -> list[str]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one value")
    return values


def _fmt(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _scaled_seed_path(base_config: SedonStandingConfig, cadence_scale: float, out_dir: Path) -> Path | None:
    """Create a timing-only reference seed copy for one cadence scale."""

    if base_config.reference_gait_seed_path is None:
        return None
    source = Path(base_config.reference_gait_seed_path)
    if not source.is_absolute():
        source = REPO_ROOT / source
    seed = _load_json(source)
    for keyframe in seed.get("keyframes", []):
        duration = int(keyframe.get("duration_steps", 1))
        keyframe["duration_steps"] = max(1, int(round(duration / cadence_scale)))
    seed["description"] = (
        f"Timing-only cadence copy for blue_forward_shuffle_authority_sweep_v1; "
        f"source={source.name}; cadence_scale={cadence_scale:g}"
    )
    path = out_dir / "seeds" / f"{source.stem}_cadence_{_fmt(cadence_scale)}.json"
    _write_json(path, seed)
    return path


def _config_for_cadence(base_config: SedonStandingConfig, cadence_scale: float, out_dir: Path) -> tuple[SedonStandingConfig, Path]:
    """Return a config variant with unchanged poses and scaled seed timing."""

    payload = dict(base_config.__dict__)
    seed_path = _scaled_seed_path(base_config, cadence_scale, out_dir)
    if seed_path is not None:
        payload["reference_gait_seed_path"] = str(seed_path.relative_to(REPO_ROOT)).replace("\\", "/")
    payload["target_forward_velocity"] = DEFAULT_TARGET_VELOCITY
    payload["march_forward_velocity"] = DEFAULT_TARGET_VELOCITY
    payload["march_forward_progress_weight"] = 0.0
    payload["march_forward_velocity_weight"] = 0.0
    payload["march_swing_forward_weight"] = 0.0
    config = SedonStandingConfig(**payload)
    config_path = out_dir / "configs" / f"cadence_{_fmt(cadence_scale)}.json"
    _write_json(config_path, payload)
    return config, config_path


class PolicyProvider:
    """Reusable deterministic PPO provider with per-env VecNormalize stats."""

    def __init__(self, model_path: Path, vecnorm_path: Path) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(f"Policy checkpoint not found: {model_path}")
        if not vecnorm_path.is_file():
            raise FileNotFoundError(f"VecNormalize file not found: {vecnorm_path}")
        from stable_baselines3 import PPO

        self._model = PPO.load(str(model_path))
        self._vecnorm_path = vecnorm_path

    def bind(self, env: SedonStandingEnv):
        """Return a deterministic action function bound to one raw env."""

        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

        vec_env = DummyVecEnv([lambda: env])
        vecnorm = VecNormalize.load(str(self._vecnorm_path), vec_env)
        vecnorm.training = False
        vecnorm.norm_reward = False

        def predict(obs: np.ndarray) -> np.ndarray:
            norm_obs = vecnorm.normalize_obs(obs[None, :])
            action, _ = self._model.predict(norm_obs, deterministic=True)
            return np.asarray(action[0], dtype=np.float64)

        return predict


def _contact_state(info: dict[str, Any]) -> str:
    right = bool(info["right_contact"])
    left = bool(info["left_contact"])
    if right and left:
        return "both"
    if right:
        return "right"
    if left:
        return "left"
    return "none"


def _phase_gate_active(phase_gate: str, support_side: str) -> bool:
    if phase_gate == "always":
        return True
    if phase_gate == "right_support":
        return support_side == "right"
    if phase_gate == "left_support":
        return support_side == "left"
    if phase_gate == "alternating_support":
        return support_side in {"right", "left"}
    raise ValueError(f"Unsupported phase gate: {phase_gate}")


def _force_body_id(env: SedonStandingEnv, apply_location: str, support_side: str) -> int | None:
    if apply_location in {"base_link", "base_com"}:
        return int(env._base_body_id)
    if apply_location == "stance_foot_phase_only":
        if support_side == "right":
            return int(env.model.geom_bodyid[env._foot_geom_ids[0]])
        if support_side == "left":
            return int(env.model.geom_bodyid[env._foot_geom_ids[1]])
        return None
    raise ValueError(f"Unsupported apply location: {apply_location}")


def _apply_forward_force(env: SedonStandingEnv, candidate: AuthorityCandidate, support_side: str) -> float:
    env.data.xfrc_applied[:] = 0.0
    if not _phase_gate_active(candidate.phase_gate, support_side):
        return 0.0
    body_id = _force_body_id(env, candidate.apply_location, support_side)
    if body_id is None:
        return 0.0
    env.data.xfrc_applied[body_id, 0] = candidate.force_n
    return candidate.force_n


def _row(step: int, env: SedonStandingEnv, info: dict[str, Any], force: float, right_slide_delta: float, left_slide_delta: float) -> dict[str, Any]:
    return {
        "step": step,
        "phase_name": str(info["phase_name"]),
        "support_side": str(info["support_side"]),
        "base_x": float(info["base_x_position"]),
        "forward_velocity": float(info["forward_velocity"]),
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
        "total_contact_force": float(info["right_normal_force"]) + float(info["left_normal_force"]),
        "right_foot_x": float(env.data.geom_xpos[env._foot_geom_ids[0]][0]),
        "left_foot_x": float(env.data.geom_xpos[env._foot_geom_ids[1]][0]),
        "right_foot_y": float(env.data.geom_xpos[env._foot_geom_ids[0]][1]),
        "left_foot_y": float(env.data.geom_xpos[env._foot_geom_ids[1]][1]),
        "right_slide_delta": right_slide_delta,
        "left_slide_delta": left_slide_delta,
        "applied_force_n": force,
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


def audit_candidate(
    candidate: AuthorityCandidate,
    *,
    config: SedonStandingConfig,
    policy_provider: PolicyProvider,
    out_dir: Path,
    steps: int,
    seed: int,
    warmup_steps: int,
    teacher_impact: float,
    max_sliding_per_forward_meter: float,
) -> AuthorityAudit:
    """Run one deterministic authority sweep candidate."""

    env = SedonStandingEnv(reset_noise_scale=0.0, reward_config=config)
    dt = float(env.dt)
    robot_weight = float(np.sum(env.model.body_mass) * 9.81)
    rows: list[dict[str, Any]] = []
    right_slide = 0.0
    left_slide = 0.0
    terminated = False
    termination_reason = "none"
    initial_base_x = 0.0
    try:
        obs, _ = env.reset(seed=seed)
        action_provider = policy_provider.bind(env)
        initial_base_x = float(env.data.qpos[0])
        previous_right_xy = env.data.geom_xpos[env._foot_geom_ids[0]][:2].copy()
        previous_left_xy = env.data.geom_xpos[env._foot_geom_ids[1]][:2].copy()
        for step in range(1, steps + 1):
            phase = env._task_phase_metadata()
            applied_force = _apply_forward_force(env, candidate, str(phase["support_side"]))
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
            rows.append(_row(step, env, dict(info), applied_force, right_delta, left_delta))
            termination_reason = str(info.get("termination_reason", "none"))
            if terminated or truncated:
                terminated = True
                break
    finally:
        env.close()

    timeline_path = out_dir / "timelines" / f"{candidate.candidate_id}.csv"
    _write_timeline(timeline_path, rows)
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
    sliding = right_slide + left_slide
    sliding_ratio = sliding / max(abs(displacement), 1e-6)
    base0 = float(post_rows[0]["base_height"]) if post_rows else 0.0
    base_drop = max(0.0, base0 - min((float(row["base_height"]) for row in post_rows), default=base0))
    max_force = max((float(row["total_contact_force"]) for row in post_rows), default=0.0)
    impact = max_force / max(robot_weight, 1e-6)
    support_ratios = [float(row["support_ratio"]) for row in rows]
    min_upright = min((float(row["upright"]) for row in rows), default=0.0)
    fail_reasons: list[str] = []
    if contact_none_steps != 0:
        fail_reasons.append("contact_none")
    if jump_count != 0:
        fail_reasons.append("jump")
    if min_upright < 0.99:
        fail_reasons.append("upright")
    if base_drop > 0.015:
        fail_reasons.append("base_drop")
    if impact > teacher_impact * 1.2:
        fail_reasons.append("landing_impact")
    if mean_velocity < 0.005:
        fail_reasons.append("forward_velocity")
    if sliding_ratio > max_sliding_per_forward_meter and abs(displacement) > 0.002:
        fail_reasons.append("foot_sliding")
    if displacement < 0.0:
        fail_reasons.append("backward_drift")
    passed = not fail_reasons
    safety_only = not any(
        reason in fail_reasons
        for reason in (
            "contact_none",
            "jump",
            "upright",
            "base_drop",
            "landing_impact",
            "foot_sliding",
            "backward_drift",
        )
    )
    if passed:
        score = 10_000.0 + mean_velocity * 1000.0 - max(0.0, sliding_ratio - SLIDING_BASELINE_PER_FORWARD_METER) * 5.0
    elif safety_only:
        score = 1_000.0 + mean_velocity * 1000.0 - max(0.0, sliding_ratio - SLIDING_BASELINE_PER_FORWARD_METER) * 5.0
    else:
        score = (
            -10_000.0
            + min_upright * 100.0
            - contact_none_steps * 20.0
            - jump_count * 50.0
            - max(0.0, base_drop - 0.015) * 500.0
            - max(0.0, impact - teacher_impact * 1.2) * 50.0
        )
    return AuthorityAudit(
        candidate_id=candidate.candidate_id,
        force_n=candidate.force_n,
        apply_location=candidate.apply_location,
        phase_gate=candidate.phase_gate,
        cadence_scale=candidate.cadence_scale,
        mean_forward_velocity=mean_velocity,
        forward_displacement=displacement,
        foot_sliding_distance=sliding,
        right_foot_sliding_distance=right_slide,
        left_foot_sliding_distance=left_slide,
        sliding_per_forward_meter=sliding_ratio,
        mean_support_ratio=float(np.mean(support_ratios)) if support_ratios else 0.0,
        peak_support_ratio=max(support_ratios, default=0.0),
        contact_none_ratio=contact_none_steps / max(1, len(rows)),
        jump_count=jump_count,
        min_upright=min_upright,
        mean_base_height=float(np.mean([float(row["base_height"]) for row in rows])) if rows else 0.0,
        base_height_drop_post_warmup=base_drop,
        landing_impact_post_warmup=impact,
        max_contact_force_post_warmup=max_force,
        terminated=terminated,
        termination_reason=termination_reason,
        passed=passed,
        fail_reasons=",".join(fail_reasons),
        score=score,
        config_path=str(candidate.config_path),
        timeline_path=str(timeline_path),
    )


def build_candidates(args: argparse.Namespace, base_config: SedonStandingConfig) -> tuple[list[AuthorityCandidate], dict[float, SedonStandingConfig]]:
    configs: dict[float, SedonStandingConfig] = {}
    config_paths: dict[float, Path] = {}
    for cadence in args.cadence_scales:
        config, config_path = _config_for_cadence(base_config, cadence, args.out_dir)
        configs[cadence] = config
        config_paths[cadence] = config_path
    candidates: list[AuthorityCandidate] = []
    for force, location, phase_gate, cadence in product(
        args.forward_forces,
        args.apply_locations,
        args.phase_gates,
        args.cadence_scales,
    ):
        candidate_id = (
            f"f{_fmt(force)}n_{location}_{phase_gate}_cad{_fmt(cadence)}"
        )
        candidates.append(
            AuthorityCandidate(
                candidate_id=candidate_id,
                force_n=force,
                apply_location=location,
                phase_gate=phase_gate,
                cadence_scale=cadence,
                config_path=config_paths[cadence],
            )
        )
    return candidates, configs


def write_results(path: Path, rows: list[AuthorityAudit]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_summary(path: Path, rows: list[AuthorityAudit], teacher_impact: float) -> None:
    passed = [row for row in rows if row.passed]
    best = max(rows, key=lambda row: row.score) if rows else None
    max_velocity = max((row.mean_forward_velocity for row in rows), default=0.0)
    lines = [
        "# blue_forward_shuffle_authority_sweep_v1",
        "",
        f"candidates: {len(rows)}",
        f"passed: {len(passed)}",
        f"teacher_landing_impact_post_warmup: {teacher_impact:.6f}",
        f"max_mean_forward_velocity: {max_velocity:.6f}",
    ]
    if best is not None:
        lines.extend(
            [
                "",
                f"best_candidate: {best.candidate_id}",
                f"best_passed: {best.passed}",
                f"best_mean_forward_velocity: {best.mean_forward_velocity:.6f}",
                f"best_sliding_per_forward_meter: {best.sliding_per_forward_meter:.3f}",
                f"best_fail_reasons: {best.fail_reasons}",
            ]
        )
    if max_velocity < 0.005:
        lines.extend(
            [
                "",
                "conclusion: max velocity stayed below 0.005 m/s; grounded shuffle forward authority appears limited under this sweep.",
            ]
        )
    lines.extend(
        [
            "",
            "| candidate | pass | v | dx | slide/m | impact | drop | upright | reasons |",
            "|---|:---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in sorted(rows, key=lambda item: item.score, reverse=True)[:10]:
        lines.append(
            f"| {row.candidate_id} | {row.passed} | {row.mean_forward_velocity:.5f} | "
            f"{row.forward_displacement:.4f} | {row.sliding_per_forward_meter:.2f} | "
            f"{row.landing_impact_post_warmup:.3f} | {row.base_height_drop_post_warmup:.4f} | "
            f"{row.min_upright:.3f} | {row.fail_reasons} |"
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
    parser.add_argument("--forward-forces", type=_parse_float_list, default=DEFAULT_FORCES)
    parser.add_argument("--apply-locations", type=_parse_str_list, default=DEFAULT_LOCATIONS)
    parser.add_argument("--phase-gates", type=_parse_str_list, default=DEFAULT_PHASE_GATES)
    parser.add_argument("--cadence-scales", type=_parse_float_list, default=DEFAULT_CADENCE_SCALES)
    parser.add_argument("--max-sliding-per-forward-meter", type=float, default=2.6)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=10)
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
    rows: list[AuthorityAudit] = []
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
        )
        rows.append(row)
        if index == 1 or index == len(candidates) or index % args.progress_every == 0:
            print(
                f"[{index}/{len(candidates)}] {row.candidate_id}: pass={row.passed} "
                f"v={row.mean_forward_velocity:.5f} dx={row.forward_displacement:.4f} "
                f"slide/m={row.sliding_per_forward_meter:.2f} impact={row.landing_impact_post_warmup:.3f} "
                f"reasons={row.fail_reasons or '-'}"
            )
    ranked = sorted(rows, key=lambda item: item.score, reverse=True)
    write_results(args.out_dir / "blue_forward_shuffle_authority_sweep_v1.csv", rows)
    write_results(args.out_dir / "blue_forward_shuffle_authority_sweep_v1_top10.csv", ranked[: args.top_k])
    write_summary(args.out_dir / "summary.md", rows, teacher.landing_impact_post_warmup)
    for row in ranked[: args.top_k]:
        print(
            f"TOP {row.candidate_id}: pass={row.passed} v={row.mean_forward_velocity:.5f} "
            f"dx={row.forward_displacement:.4f} slide/m={row.sliding_per_forward_meter:.2f} "
            f"impact={row.landing_impact_post_warmup:.3f} upright={row.min_upright:.3f} "
            f"reasons={row.fail_reasons or '-'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
