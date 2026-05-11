"""Verify that a Sedon standing seed remains safe under zero action."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sedon_baseline.env import SedonStandingConfig, SedonStandingEnv


DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "sedon" / "zero_action_safe_stand.json"
)


@dataclass(frozen=True)
class StaticSeedSummary:
    """Aggregated safety metrics for one zero-action rollout."""

    steps_survived: int
    terminated: bool
    max_abs_forward_velocity: float
    min_base_height: float
    min_upright: float
    knee_violation_steps: int
    base_proxy_floor_steps: int

    @property
    def completed_steps(self) -> bool:
        """Return whether the rollout survived the full requested horizon."""
        return not self.terminated


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to a Sedon config JSON file.",
    )
    parser.add_argument("--steps", type=int, default=400, help="Number of zero-action steps.")
    parser.add_argument("--seed", type=int, default=42, help="Environment reset seed.")
    parser.add_argument(
        "--max-abs-fwd-limit",
        type=float,
        default=0.25,
        help="Maximum allowed absolute forward velocity.",
    )
    return parser


def load_reward_config(path: Path) -> SedonStandingConfig:
    """Load a ``SedonStandingConfig`` from a JSON file."""
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    raw_payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_payload, dict):
        raise ValueError(f"Config file must contain a JSON object: {path}")
    return SedonStandingConfig(**raw_payload)


def run_zero_action_rollout(
    reward_config: SedonStandingConfig,
    *,
    steps: int,
    seed: int,
) -> StaticSeedSummary:
    """Run a zero-action Sedon rollout and aggregate safety metrics."""
    if steps <= 0:
        raise ValueError("steps must be positive.")

    env = SedonStandingEnv(reset_noise_scale=0.0, reward_config=reward_config)
    try:
        env.reset(seed=seed)
        action = np.zeros(env.action_space.shape, dtype=np.float64)
        max_abs_forward_velocity = 0.0
        min_base_height = float("inf")
        min_upright = float("inf")
        knee_violation_steps = 0
        base_proxy_floor_steps = 0
        terminated = False
        steps_survived = 0

        for step_index in range(steps):
            _, _, terminated, truncated, info = env.step(action)
            if truncated:
                raise RuntimeError("Sedon static-seed verifier received unexpected truncation.")
            steps_survived = step_index + 1
            max_abs_forward_velocity = max(
                max_abs_forward_velocity,
                abs(float(info.get("forward_velocity", 0.0))),
            )
            min_base_height = min(min_base_height, float(info.get("base_height", np.nan)))
            min_upright = min(min_upright, float(info.get("upright", np.nan)))
            knee_violation_steps += int(float(info.get("knee_safe_violation_sum", 0.0)) > 0.0)
            base_proxy_floor_steps += int(bool(info.get("base_proxy_floor_contact", False)))
            if terminated:
                break

        return StaticSeedSummary(
            steps_survived=steps_survived,
            terminated=terminated,
            max_abs_forward_velocity=max_abs_forward_velocity,
            min_base_height=min_base_height,
            min_upright=min_upright,
            knee_violation_steps=knee_violation_steps,
            base_proxy_floor_steps=base_proxy_floor_steps,
        )
    finally:
        env.close()


def main(argv: list[str] | None = None) -> int:
    """Run the zero-action static-seed verifier and print pass/fail metrics."""
    args = build_parser().parse_args(argv)
    reward_config = load_reward_config(args.config)
    summary = run_zero_action_rollout(reward_config, steps=args.steps, seed=args.seed)

    passed = (
        summary.steps_survived == args.steps
        and not summary.terminated
        and summary.knee_violation_steps == 0
        and summary.base_proxy_floor_steps == 0
        and summary.max_abs_forward_velocity < args.max_abs_fwd_limit
    )

    print(f"config={args.config}")
    print(f"steps_survived={summary.steps_survived}")
    print(f"terminated={summary.terminated}")
    print(f"max_abs_forward_velocity={summary.max_abs_forward_velocity:.6f}")
    print(f"min_base_height={summary.min_base_height:.6f}")
    print(f"min_upright={summary.min_upright:.6f}")
    print(f"knee_violation_steps={summary.knee_violation_steps}")
    print(f"base_proxy_floor_steps={summary.base_proxy_floor_steps}")
    print(f"pass={passed}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
