"""Smoke test the Sedon standing environment without starting a training run."""

from __future__ import annotations

import argparse

import numpy as np

from sedon_baseline.env import SedonStandingEnv


def build_parser() -> argparse.ArgumentParser:
    """Build the command line parser."""
    parser = argparse.ArgumentParser(description="Run a short Sedon env smoke test.")
    parser.add_argument("--steps", type=int, default=100, help="Number of env steps.")
    parser.add_argument(
        "--random-action",
        action="store_true",
        help="Use random normalized actions instead of zero action.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run a short rollout and print basic simulator health metrics."""
    args = build_parser().parse_args(argv)
    if args.steps <= 0:
        raise ValueError("steps must be positive.")

    env = SedonStandingEnv(reset_noise_scale=0.0)
    try:
        obs, _ = env.reset(seed=42)
        total_reward = 0.0
        terminated = False
        last_info = {}
        for _ in range(args.steps):
            action = (
                env.action_space.sample()
                if args.random_action
                else np.zeros(env.action_space.shape, dtype=np.float64)
            )
            obs, reward, terminated, truncated, last_info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        print(f"obs_shape={obs.shape}")
        print(f"finite_obs={bool(np.isfinite(obs).all())}")
        print(f"terminated={terminated}")
        print(f"total_reward={total_reward:.4f}")
        print(f"base_height={last_info.get('base_height')}")
        print(f"upright={last_info.get('upright')}")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
