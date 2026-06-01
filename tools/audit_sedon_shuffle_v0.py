"""Audit Sedon shuffle v0 curriculum against low-threshold in-place criteria."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sedon_baseline.env import SedonStandingConfig, SedonStandingEnv


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "sedon" / "shuffle_v0.json"


@dataclass(frozen=True)
class ShuffleAudit:
    """Aggregated shuffle-v0 audit metrics."""

    steps: int
    audit_warmup_steps: int
    terminated: bool
    termination_reason: str
    both_contact_ratio: float
    single_contact_ratio: float
    contact_none_ratio: float
    peak_support_ratio: float
    min_swing_ratio: float
    max_clearance: float
    base_height_drop_raw: float
    base_height_drop_post_warmup: float
    base_height_drop: float
    jump_count: int
    landing_impact_raw: float
    landing_impact_post_warmup: float
    landing_impact: float
    max_contact_force_raw: float
    max_contact_force_post_warmup: float
    foot_velocity_near_contact_raw: float
    foot_velocity_near_contact_post_warmup: float
    mean_tracking_error: float
    max_tracking_error: float
    tracking_error_variance: float
    contact_transition_ratio: float
    reward_gate_active_ratio: float
    max_abs_forward_drift: float
    passed: bool


@dataclass(frozen=True)
class TeacherRelativeGate:
    """Shared strict teacher-relative audit gate result."""

    passed: bool
    reasons: tuple[str, ...]
    max_landing_impact_post_warmup: float
    max_base_height_drop_post_warmup: float
    max_contact_force_post_warmup: float
    max_foot_velocity_near_contact_post_warmup: float
    max_mean_tracking_error: float


def teacher_relative_gate(
    teacher: ShuffleAudit,
    probe: ShuffleAudit,
    *,
    landing_impact_multiplier: float = 1.15,
    max_contact_force_multiplier: float = 1.15,
    foot_velocity_multiplier: float = 1.5,
    tracking_error_multiplier: float = 1.5,
    base_height_drop_tolerance: float = 1e-9,
) -> TeacherRelativeGate:
    """Evaluate the shared warmup teacher-relative gate.

    Args:
        teacher: Baseline teacher audit summary.
        probe: Policy/checkpoint audit summary to gate.
        landing_impact_multiplier: Allowed post-warmup landing impact multiple.
        max_contact_force_multiplier: Allowed post-warmup force-spike multiple.
        foot_velocity_multiplier: Allowed post-warmup near-contact foot velocity multiple.
        tracking_error_multiplier: Allowed mean tracking error multiple.
        base_height_drop_tolerance: Absolute tolerance added to teacher post-warmup drop.

    Returns:
        Gate result with explicit limits and failed reason labels.
    """
    max_landing_impact = (
        teacher.landing_impact_post_warmup * landing_impact_multiplier
    )
    max_base_height_drop = (
        teacher.base_height_drop_post_warmup + base_height_drop_tolerance
    )
    max_contact_force = (
        teacher.max_contact_force_post_warmup * max_contact_force_multiplier
    )
    max_foot_velocity = max(
        teacher.foot_velocity_near_contact_post_warmup * foot_velocity_multiplier,
        teacher.foot_velocity_near_contact_post_warmup + 1e-6,
    )
    max_mean_tracking_error = teacher.mean_tracking_error * tracking_error_multiplier
    failed: list[str] = []
    if probe.contact_none_ratio != 0.0:
        failed.append("contact_none")
    if probe.jump_count != 0:
        failed.append("jump")
    if probe.peak_support_ratio < 0.60:
        failed.append("support")
    if probe.max_clearance < 0.001:
        failed.append("clearance")
    if probe.landing_impact_post_warmup > max_landing_impact:
        failed.append("landing_impact_post_warmup")
    if probe.base_height_drop_post_warmup > max_base_height_drop:
        failed.append("base_height_drop_post_warmup")
    if probe.max_contact_force_post_warmup > max_contact_force:
        failed.append("max_contact_force_post_warmup")
    if probe.foot_velocity_near_contact_post_warmup > max_foot_velocity:
        failed.append("foot_velocity_near_contact_post_warmup")
    if probe.mean_tracking_error > max_mean_tracking_error:
        failed.append("mean_tracking_error")
    return TeacherRelativeGate(
        passed=not failed,
        reasons=tuple(failed),
        max_landing_impact_post_warmup=max_landing_impact,
        max_base_height_drop_post_warmup=max_base_height_drop,
        max_contact_force_post_warmup=max_contact_force,
        max_foot_velocity_near_contact_post_warmup=max_foot_velocity,
        max_mean_tracking_error=max_mean_tracking_error,
    )


def _load_config(path: Path) -> SedonStandingConfig:
    """Load a Sedon config JSON override."""
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a JSON object: {path}")
    return SedonStandingConfig(**{**SedonStandingConfig().__dict__, **payload})


def _load_model(model_path: Path | None) -> Any | None:
    """Load an optional SB3 model."""
    if model_path is None:
        return None
    from stable_baselines3 import PPO

    return PPO.load(str(model_path))


def _count_contact_none_bursts(infos: list[dict[str, Any]]) -> int:
    """Count contiguous both-air/contact-none bursts."""
    count = 0
    in_burst = False
    for info in infos:
        contact_none = not bool(info["right_contact"]) and not bool(info["left_contact"])
        if contact_none and not in_burst:
            count += 1
            in_burst = True
        elif not contact_none:
            in_burst = False
    return count


def _collect_raw_infos(
    env: SedonStandingEnv,
    model: Any | None,
    steps: int,
    seed: int,
    random_residual: bool,
) -> tuple[list[dict[str, Any]], bool, str]:
    """Collect rollout infos from an unnormalized Sedon environment."""
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    infos: list[dict[str, Any]] = []
    previous_bottoms = env._foot_bottom_heights()
    terminated = False
    termination_reason = "none"
    for _ in range(steps):
        if random_residual:
            action = rng.uniform(-1.0, 1.0, size=env.action_space.shape).astype(np.float64)
        elif model is None:
            action = np.zeros(env.action_space.shape, dtype=np.float64)
        else:
            action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, _, info = env.step(action)
        bottoms = env._foot_bottom_heights()
        foot_velocity = bottoms - previous_bottoms
        previous_bottoms = bottoms.copy()
        near_contact_velocity = 0.0
        if bool(info["left_contact"]):
            near_contact_velocity = max(near_contact_velocity, abs(float(foot_velocity[0])))
        if bool(info["right_contact"]):
            near_contact_velocity = max(near_contact_velocity, abs(float(foot_velocity[1])))
        enriched_info = dict(info)
        enriched_info["foot_velocity_near_contact"] = near_contact_velocity
        infos.append(enriched_info)
        termination_reason = str(info.get("termination_reason", "none"))
        if terminated:
            break
    return infos, bool(terminated), termination_reason


def _collect_vecnorm_infos(
    config: SedonStandingConfig,
    model: Any | None,
    vecnorm_path: Path,
    steps: int,
    seed: int,
    random_residual: bool,
) -> tuple[list[dict[str, Any]], bool, str]:
    """Collect rollout infos through the VecNormalize stats used in training."""
    if not vecnorm_path.is_file():
        raise FileNotFoundError(f"VecNormalize file not found: {vecnorm_path}")
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    def make_env() -> SedonStandingEnv:
        env = SedonStandingEnv(reset_noise_scale=0.0, reward_config=config)
        env.reset(seed=seed)
        return env

    dummy_env = DummyVecEnv([make_env])
    raw_env: SedonStandingEnv = dummy_env.envs[0]
    eval_env = VecNormalize.load(str(vecnorm_path), dummy_env)
    eval_env.training = False
    eval_env.norm_reward = False
    obs = eval_env.reset()
    rng = np.random.default_rng(seed)
    infos: list[dict[str, Any]] = []
    previous_bottoms = raw_env._foot_bottom_heights()
    terminated = False
    termination_reason = "none"
    try:
        for _ in range(steps):
            if random_residual:
                action = rng.uniform(
                    -1.0,
                    1.0,
                    size=(1, eval_env.action_space.shape[0]),
                ).astype(np.float64)
            elif model is None:
                action = np.zeros((1, eval_env.action_space.shape[0]), dtype=np.float64)
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, step_infos = eval_env.step(action)
            info = dict(step_infos[0])
            bottoms = raw_env._foot_bottom_heights()
            foot_velocity = bottoms - previous_bottoms
            previous_bottoms = bottoms.copy()
            near_contact_velocity = 0.0
            if bool(info["left_contact"]):
                near_contact_velocity = max(near_contact_velocity, abs(float(foot_velocity[0])))
            if bool(info["right_contact"]):
                near_contact_velocity = max(near_contact_velocity, abs(float(foot_velocity[1])))
            info["foot_velocity_near_contact"] = near_contact_velocity
            infos.append(info)
            termination_reason = str(info.get("termination_reason", "none"))
            if bool(dones[0]):
                terminated = True
                break
    finally:
        eval_env.close()
    return infos, terminated, termination_reason


def audit_shuffle(
    config_path: Path,
    model_path: Path | None,
    vecnorm_path: Path | None,
    steps: int,
    seed: int,
    random_residual: bool = False,
    audit_warmup_steps: int = 20,
) -> ShuffleAudit:
    """Run one shuffle-v0 rollout and aggregate audit metrics."""
    if random_residual and model_path is not None:
        raise ValueError("random_residual cannot be combined with model_path.")
    if audit_warmup_steps < 0:
        raise ValueError("audit_warmup_steps must be non-negative.")
    config = _load_config(config_path)
    env = SedonStandingEnv(reset_noise_scale=0.0, reward_config=config)
    model = _load_model(model_path)
    if vecnorm_path is None:
        infos, terminated, termination_reason = _collect_raw_infos(
            env,
            model,
            steps,
            seed,
            random_residual,
        )
    else:
        infos, terminated, termination_reason = _collect_vecnorm_infos(
            config,
            model,
            vecnorm_path,
            steps,
            seed,
            random_residual,
        )

    if not infos:
        return ShuffleAudit(
            steps=0,
            audit_warmup_steps=audit_warmup_steps,
            terminated=True,
            termination_reason="no_steps",
            both_contact_ratio=0.0,
            single_contact_ratio=0.0,
            contact_none_ratio=1.0,
            peak_support_ratio=0.0,
            min_swing_ratio=1.0,
            max_clearance=0.0,
            base_height_drop_raw=float("inf"),
            base_height_drop_post_warmup=float("inf"),
            base_height_drop=float("inf"),
            jump_count=1,
            landing_impact_raw=float("inf"),
            landing_impact_post_warmup=float("inf"),
            landing_impact=float("inf"),
            max_contact_force_raw=float("inf"),
            max_contact_force_post_warmup=float("inf"),
            foot_velocity_near_contact_raw=float("inf"),
            foot_velocity_near_contact_post_warmup=float("inf"),
            mean_tracking_error=float("inf"),
            max_tracking_error=float("inf"),
            tracking_error_variance=float("inf"),
            contact_transition_ratio=1.0,
            reward_gate_active_ratio=0.0,
            max_abs_forward_drift=float("inf"),
            passed=False,
        )

    both_steps = 0
    single_steps = 0
    none_steps = 0
    total_weight = float(np.sum(env.model.body_mass) * 9.81)
    initial_base_height = float(infos[0]["base_height"])
    for info in infos:
        left = bool(info["left_contact"])
        right = bool(info["right_contact"])
        if left and right:
            both_steps += 1
        elif left or right:
            single_steps += 1
        else:
            none_steps += 1
    peak_support_ratio = max(float(info["force_ratio"]) for info in infos)
    min_swing_ratio = min(float(info["swing_force_ratio"]) for info in infos)
    max_clearance = max(float(info["foot_clearance"]) for info in infos)
    post_infos = infos[audit_warmup_steps:] or infos
    post_initial_base_height = float(post_infos[0]["base_height"])
    base_height_drop_raw = max(
        0.0,
        initial_base_height - min(float(info["base_height"]) for info in infos),
    )
    base_height_drop_post_warmup = max(
        0.0,
        post_initial_base_height - min(float(info["base_height"]) for info in post_infos),
    )
    contact_forces_raw = [
        float(info["left_normal_force"]) + float(info["right_normal_force"])
        for info in infos
    ]
    contact_forces_post_warmup = [
        float(info["left_normal_force"]) + float(info["right_normal_force"])
        for info in post_infos
    ]
    max_contact_force_raw = max(contact_forces_raw, default=0.0)
    max_contact_force_post_warmup = max(contact_forces_post_warmup, default=0.0)
    landing_impact_raw = max_contact_force_raw / max(total_weight, 1e-6)
    landing_impact_post_warmup = max_contact_force_post_warmup / max(total_weight, 1e-6)
    foot_velocity_near_contact_raw = max(
        float(info.get("foot_velocity_near_contact", 0.0)) for info in infos
    )
    foot_velocity_near_contact_post_warmup = max(
        float(info.get("foot_velocity_near_contact", 0.0)) for info in post_infos
    )
    tracking_errors = [
        float(
            np.sqrt(
                float(info.get("joint_position_error_l2", 0.0))
                / max(int(env.action_space.shape[0]), 1)
            )
        )
        for info in infos
    ]
    contact_transition_ratio = float(
        np.mean([float(info.get("contact_transition", 0.0)) for info in infos])
    )
    reward_gate_active_ratio = float(
        np.mean([float(bool(info.get("reward_gate_active", False))) for info in infos])
    )
    max_abs_forward_drift = max(abs(float(info["base_x_position"])) for info in infos)
    jump_count = _count_contact_none_bursts(infos)
    passed = (
        none_steps == 0
        and jump_count == 0
        and peak_support_ratio >= 0.54
        and min_swing_ratio <= 0.46
        and max_clearance >= 0.0005
        and base_height_drop_post_warmup <= 0.015
        and max_abs_forward_drift <= 0.02
    )
    total_steps = max(1, len(infos))
    return ShuffleAudit(
        steps=len(infos),
        audit_warmup_steps=audit_warmup_steps,
        terminated=bool(terminated),
        termination_reason=termination_reason,
        both_contact_ratio=both_steps / total_steps,
        single_contact_ratio=single_steps / total_steps,
        contact_none_ratio=none_steps / total_steps,
        peak_support_ratio=peak_support_ratio,
        min_swing_ratio=min_swing_ratio,
        max_clearance=max_clearance,
        base_height_drop_raw=base_height_drop_raw,
        base_height_drop_post_warmup=base_height_drop_post_warmup,
        base_height_drop=base_height_drop_post_warmup,
        jump_count=jump_count,
        landing_impact_raw=landing_impact_raw,
        landing_impact_post_warmup=landing_impact_post_warmup,
        landing_impact=landing_impact_post_warmup,
        max_contact_force_raw=max_contact_force_raw,
        max_contact_force_post_warmup=max_contact_force_post_warmup,
        foot_velocity_near_contact_raw=foot_velocity_near_contact_raw,
        foot_velocity_near_contact_post_warmup=foot_velocity_near_contact_post_warmup,
        mean_tracking_error=float(np.mean(tracking_errors)) if tracking_errors else 0.0,
        max_tracking_error=float(max(tracking_errors, default=0.0)),
        tracking_error_variance=float(np.var(tracking_errors)) if tracking_errors else 0.0,
        contact_transition_ratio=contact_transition_ratio,
        reward_gate_active_ratio=reward_gate_active_ratio,
        max_abs_forward_drift=max_abs_forward_drift,
        passed=passed,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--vecnorm-path", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--audit-warmup-steps",
        type=int,
        default=20,
        help="Initial rollout steps excluded from impact/drop audit metrics.",
    )
    parser.add_argument(
        "--random-residual",
        action="store_true",
        help="Use seeded random normalized residual actions instead of zero residuals.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run shuffle-v0 audit."""
    args = build_parser().parse_args(argv)
    summary = audit_shuffle(
        args.config,
        args.model_path,
        args.vecnorm_path,
        args.steps,
        args.seed,
        random_residual=args.random_residual,
        audit_warmup_steps=args.audit_warmup_steps,
    )
    for key, value in summary.__dict__.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
