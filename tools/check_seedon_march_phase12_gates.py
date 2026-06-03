"""Check Seedon march-in-place Phase 1/2 gates before enabling micro-lift."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from seedon_baseline.env import SeedonStandingConfig, SeedonStandingEnv


REPO_ROOT = Path(__file__).resolve().parents[1]
PHASE1_CONFIG = REPO_ROOT / "configs" / "seedon" / "march_in_place_phase1_load_transfer.json"
PHASE2_CONFIG = REPO_ROOT / "configs" / "seedon" / "march_in_place_phase2_swing_unload.json"


@dataclass(frozen=True)
class GateThresholds:
    """Thresholds used to decide whether Phase 1/2 is ready for Phase 3.

    Args:
        phase1_support_ratio: Required max support-foot force fraction.
        stable_seconds: Required uninterrupted rollout duration.
        max_abs_base_roll: Maximum allowed absolute base roll in radians.
        max_abs_base_pitch: Maximum allowed absolute base pitch in radians.
        max_torque_saturation: Maximum allowed actuator saturation fraction.
        phase2_swing_ratio: Required min swing-foot force fraction.
        min_both_contact_fraction: Required fraction of steps with both feet in contact.
        max_foot_penetration: Maximum allowed foot-bottom floor penetration in meters.
        max_base_drift: Maximum allowed horizontal base drift in meters.
    """

    phase1_support_ratio: float = 0.62
    stable_seconds: float = 2.0
    max_abs_base_roll: float = 0.08
    max_abs_base_pitch: float = 0.14
    max_torque_saturation: float = 0.10
    phase2_swing_ratio: float = 0.38
    min_both_contact_fraction: float = 0.90
    max_foot_penetration: float = 0.003
    max_base_drift: float = 0.05


@dataclass(frozen=True)
class GateSummary:
    """Aggregated rollout metrics for one curriculum phase."""

    phase: int
    steps: int
    duration_s: float
    terminated: bool
    termination_reason: str
    max_support_ratio: float
    min_swing_ratio: float
    both_contact_fraction: float
    max_abs_base_roll: float
    max_abs_base_pitch: float
    max_torque_saturation: float
    max_foot_penetration: float
    max_base_drift: float
    raw_action_abs_max: float
    raw_action_abs_mean: float
    raw_action_std_mean: float
    scaled_residual_abs_max: float
    scaled_residual_abs_mean: float
    scaled_residual_std_mean: float
    max_support_ratio_progress: float
    mean_support_ratio_progress: float
    max_target_side_force_ratio: float
    mean_target_side_force_ratio: float
    passed: bool
    failed_checks: tuple[str, ...]


def _load_config(path: Path) -> SeedonStandingConfig:
    """Load a SeedonStandingConfig from a JSON override file.

    Args:
        path: JSON object containing SeedonStandingConfig overrides.

    Returns:
        Complete SeedonStandingConfig.

    Raises:
        FileNotFoundError: If the config file is missing.
        ValueError: If the JSON payload is not an object.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Seedon march config not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Seedon march config must be a JSON object: {path}")
    return SeedonStandingConfig(**{**SeedonStandingConfig().__dict__, **payload})


def _evaluate_phase(
    *,
    phase: int,
    config: SeedonStandingConfig,
    model: Any | None,
    vecnorm_path: Path | None,
    seed: int,
    steps: int,
    thresholds: GateThresholds,
) -> GateSummary:
    """Run one rollout and aggregate Phase 1/2 gate metrics."""
    infos: list[dict[str, Any]] = []
    terminated = False
    truncated = False
    raw_env: SeedonStandingEnv | None = None

    def make_env() -> SeedonStandingEnv:
        nonlocal raw_env
        raw_env = SeedonStandingEnv(reset_noise_scale=0.0, reward_config=config)
        return raw_env

    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    vec_env = DummyVecEnv([make_env])
    try:
        vec_env.seed(seed)
        eval_env = vec_env
        if vecnorm_path is not None:
            if not vecnorm_path.is_file():
                raise FileNotFoundError(f"VecNormalize file not found: {vecnorm_path}")
            eval_env = VecNormalize.load(str(vecnorm_path), vec_env)
            eval_env.training = False
            eval_env.norm_reward = False
        obs = eval_env.reset()
        for _ in range(steps):
            if model is None:
                action = np.zeros((1, raw_env.action_space.shape[0]), dtype=np.float64)
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, step_infos = eval_env.step(action)
            info = step_infos[0]
            terminated = bool(dones[0]) and str(info.get("termination_reason", "none")) != "none"
            truncated = bool(dones[0]) and not terminated
            infos.append(info)
            if terminated or truncated:
                break
        if raw_env is None:
            raise RuntimeError("Seedon gate checker failed to create an environment.")
        dt = float(raw_env.dt)
    finally:
        vec_env.close()

    if not infos:
        return GateSummary(
            phase=phase,
            steps=0,
            duration_s=0.0,
            terminated=True,
            termination_reason="no_steps",
            max_support_ratio=0.0,
            min_swing_ratio=1.0,
            both_contact_fraction=0.0,
            max_abs_base_roll=0.0,
            max_abs_base_pitch=0.0,
            max_torque_saturation=1.0,
            max_foot_penetration=float("inf"),
            max_base_drift=float("inf"),
            raw_action_abs_max=0.0,
            raw_action_abs_mean=0.0,
            raw_action_std_mean=0.0,
            scaled_residual_abs_max=0.0,
            scaled_residual_abs_mean=0.0,
            scaled_residual_std_mean=0.0,
            max_support_ratio_progress=0.0,
            mean_support_ratio_progress=0.0,
            max_target_side_force_ratio=0.0,
            mean_target_side_force_ratio=0.0,
            passed=False,
            failed_checks=("no_steps",),
        )

    both_contact_steps = sum(
        1 for info in infos if bool(info["right_contact"]) and bool(info["left_contact"])
    )
    foot_bottoms = [
        float(info["support_foot_bottom_z"])
        for info in infos
        if np.isfinite(float(info["support_foot_bottom_z"]))
    ] + [
        float(info["swing_foot_bottom_z"])
        for info in infos
        if np.isfinite(float(info["swing_foot_bottom_z"]))
    ]
    max_foot_penetration = max(0.0, -min(foot_bottoms)) if foot_bottoms else 0.0
    max_base_drift = max(
        float(np.hypot(info["base_x_position"], info["base_y_position"]))
        for info in infos
    )
    summary = GateSummary(
        phase=phase,
        steps=len(infos),
        duration_s=len(infos) * dt,
        terminated=bool(terminated),
        termination_reason=str(infos[-1].get("termination_reason", "none")),
        max_support_ratio=max(float(info["force_ratio"]) for info in infos),
        min_swing_ratio=min(float(info["swing_force_ratio"]) for info in infos),
        both_contact_fraction=both_contact_steps / max(1, len(infos)),
        max_abs_base_roll=max(abs(float(info["base_roll"])) for info in infos),
        max_abs_base_pitch=max(abs(float(info["base_pitch"])) for info in infos),
        max_torque_saturation=max(float(info["torque_saturation"]) for info in infos),
        max_foot_penetration=max_foot_penetration,
        max_base_drift=max_base_drift,
        raw_action_abs_max=max(float(info.get("raw_action_abs_max", 0.0)) for info in infos),
        raw_action_abs_mean=float(
            np.mean([float(info.get("raw_action_abs_max", 0.0)) for info in infos])
        ),
        raw_action_std_mean=float(
            np.mean([float(info.get("raw_action_std", 0.0)) for info in infos])
        ),
        scaled_residual_abs_max=max(
            float(info.get("scaled_residual_abs_max", 0.0)) for info in infos
        ),
        scaled_residual_abs_mean=float(
            np.mean([float(info.get("scaled_residual_abs_max", 0.0)) for info in infos])
        ),
        scaled_residual_std_mean=float(
            np.mean([float(info.get("scaled_residual_std", 0.0)) for info in infos])
        ),
        max_support_ratio_progress=max(
            float(info.get("support_ratio_progress", 0.0)) for info in infos
        ),
        mean_support_ratio_progress=float(
            np.mean([float(info.get("support_ratio_progress", 0.0)) for info in infos])
        ),
        max_target_side_force_ratio=max(
            float(info.get("target_side_force_ratio", 0.0)) for info in infos
        ),
        mean_target_side_force_ratio=float(
            np.mean([float(info.get("target_side_force_ratio", 0.0)) for info in infos])
        ),
        passed=False,
        failed_checks=(),
    )
    failures = _phase_failures(summary, thresholds)
    return GateSummary(
        **{
            **summary.__dict__,
            "passed": not failures,
            "failed_checks": tuple(failures),
        }
    )


def _phase_failures(summary: GateSummary, thresholds: GateThresholds) -> list[str]:
    """Return failed gate names for one phase summary."""
    failures: list[str] = []
    if summary.terminated:
        failures.append(f"terminated:{summary.termination_reason}")
    if summary.duration_s < thresholds.stable_seconds:
        failures.append("stable_duration")
    if summary.max_abs_base_roll > thresholds.max_abs_base_roll:
        failures.append("base_roll")
    if summary.max_abs_base_pitch > thresholds.max_abs_base_pitch:
        failures.append("base_pitch")
    if summary.max_torque_saturation > thresholds.max_torque_saturation:
        failures.append("torque_saturation")
    if summary.phase == 1:
        if summary.max_support_ratio < thresholds.phase1_support_ratio:
            failures.append("support_ratio")
    elif summary.phase == 2:
        if summary.min_swing_ratio > thresholds.phase2_swing_ratio:
            failures.append("swing_ratio")
        if summary.both_contact_fraction < thresholds.min_both_contact_fraction:
            failures.append("both_contact_fraction")
        if summary.max_foot_penetration > thresholds.max_foot_penetration:
            failures.append("foot_penetration")
        if summary.max_base_drift > thresholds.max_base_drift:
            failures.append("base_drift")
    return failures


def _load_model(path: Path | None) -> Any | None:
    """Load a PPO model if supplied; otherwise return None for zero residual."""
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")
    from stable_baselines3 import PPO

    return PPO.load(path)


def _print_summary(summary: GateSummary) -> None:
    """Print a compact one-phase gate summary."""
    status = "PASS" if summary.passed else "FAIL"
    print(f"Phase {summary.phase}: {status}")
    print(
        "  "
        f"steps={summary.steps} duration={summary.duration_s:.2f}s "
        f"terminated={summary.terminated} reason={summary.termination_reason}"
    )
    print(
        "  "
        f"max_support_ratio={summary.max_support_ratio:.3f} "
        f"min_swing_ratio={summary.min_swing_ratio:.3f} "
        f"both_contact={summary.both_contact_fraction:.1%}"
    )
    print(
        "  "
        f"roll={summary.max_abs_base_roll:.4f} "
        f"pitch={summary.max_abs_base_pitch:.4f} "
        f"torque_sat={summary.max_torque_saturation:.3f} "
        f"penetration={summary.max_foot_penetration:.4f}m "
        f"drift={summary.max_base_drift:.4f}m"
    )
    print(
        "  "
        f"raw_action abs_max={summary.raw_action_abs_max:.4f} "
        f"abs_mean={summary.raw_action_abs_mean:.4f} "
        f"std_mean={summary.raw_action_std_mean:.4f}"
    )
    print(
        "  "
        f"scaled_residual abs_max={summary.scaled_residual_abs_max:.5f} "
        f"abs_mean={summary.scaled_residual_abs_mean:.5f} "
        f"std_mean={summary.scaled_residual_std_mean:.5f}"
    )
    print(
        "  "
        f"support_progress max={summary.max_support_ratio_progress:.3f} "
        f"mean={summary.mean_support_ratio_progress:.3f} "
        f"target_side_ratio max={summary.max_target_side_force_ratio:.3f} "
        f"mean={summary.mean_target_side_force_ratio:.3f}"
    )
    if summary.failed_checks:
        print(f"  failed_checks={', '.join(summary.failed_checks)}")


def build_parser() -> argparse.ArgumentParser:
    """Build CLI arguments for the Phase 1/2 gate checker."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--vecnorm-path", type=Path, default=None)
    parser.add_argument("--phase1-config", type=Path, default=PHASE1_CONFIG)
    parser.add_argument("--phase2-config", type=Path, default=PHASE2_CONFIG)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--phase1-support-ratio", type=float, default=0.62)
    parser.add_argument("--phase2-swing-ratio", type=float, default=0.38)
    parser.add_argument("--stable-seconds", type=float, default=2.0)
    parser.add_argument("--max-abs-base-roll", type=float, default=0.08)
    parser.add_argument("--max-abs-base-pitch", type=float, default=0.14)
    parser.add_argument("--max-torque-saturation", type=float, default=0.10)
    parser.add_argument("--min-both-contact-fraction", type=float, default=0.90)
    parser.add_argument("--max-foot-penetration", type=float, default=0.003)
    parser.add_argument("--max-base-drift", type=float, default=0.05)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Check Phase 1 and Phase 2 gates and return nonzero on failure."""
    args = build_parser().parse_args(argv)
    if args.steps <= 0:
        raise ValueError("--steps must be positive.")
    thresholds = GateThresholds(
        phase1_support_ratio=args.phase1_support_ratio,
        stable_seconds=args.stable_seconds,
        max_abs_base_roll=args.max_abs_base_roll,
        max_abs_base_pitch=args.max_abs_base_pitch,
        max_torque_saturation=args.max_torque_saturation,
        phase2_swing_ratio=args.phase2_swing_ratio,
        min_both_contact_fraction=args.min_both_contact_fraction,
        max_foot_penetration=args.max_foot_penetration,
        max_base_drift=args.max_base_drift,
    )
    model = _load_model(args.model_path)
    phase1 = _evaluate_phase(
        phase=1,
        config=_load_config(args.phase1_config),
        model=model,
        vecnorm_path=args.vecnorm_path,
        seed=args.seed,
        steps=args.steps,
        thresholds=thresholds,
    )
    phase2 = _evaluate_phase(
        phase=2,
        config=_load_config(args.phase2_config),
        model=model,
        vecnorm_path=args.vecnorm_path,
        seed=args.seed + 1,
        steps=args.steps,
        thresholds=thresholds,
    )
    _print_summary(phase1)
    _print_summary(phase2)
    return 0 if phase1.passed and phase2.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
