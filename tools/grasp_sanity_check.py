"""Sanity checks for the fixed-base grasp environment.

Use this before long RL runs to confirm the reset pose and controller are
keeping the gripper in a learnable region near the cube.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RolloutSummary:
    """Small summary of one deterministic scripted rollout."""

    name: str
    steps: int
    min_distance: float
    max_distance: float
    last_distance: float
    max_height: float
    fell: bool


def _run_scripted_rollout(
    *,
    env,
    name: str,
    action: np.ndarray,
    steps: int,
    seed: int,
) -> RolloutSummary:
    """Roll out a constant action to inspect basic controller behavior."""
    _, _ = env.reset(seed=seed)
    distances: list[float] = []
    heights: list[float] = []
    fell = False

    for _ in range(steps):
        _, _, terminated, truncated, info = env.step(action)
        distances.append(float(info["distance_to_cube"]))
        heights.append(float(info["cube_height_above_table"]))
        fell = fell or bool(info["cube_fell"])
        if terminated or truncated:
            break

    return RolloutSummary(
        name=name,
        steps=len(distances),
        min_distance=min(distances),
        max_distance=max(distances),
        last_distance=distances[-1],
        max_height=max(heights),
        fell=fell,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI options for the grasp sanity checker."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fixed-cube",
        action="store_true",
        help="Disable cube pose randomization while checking the controller.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run environment-level sanity checks for the grasp setup."""
    from grasp_baseline.env import FixedBaseGraspEnv

    args = parse_args(argv)
    env = FixedBaseGraspEnv(
        task_phase="full",
        randomize_cube_pose=not args.fixed_cube,
    )

    try:
        _, _ = env.reset(seed=args.seed)
        start_distance = float(np.linalg.norm(env._cube_pos() - env._gripper_pos()))
        print(f"Reset distance: {start_distance:.4f} m")
        print(f"Default arm targets: {np.round(env._default_arm_targets, 4).tolist()}")
        print()

        rollouts = [
            _run_scripted_rollout(
                env=env,
                name="zero_action_open_gripper",
                action=np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float64),
                steps=args.steps,
                seed=args.seed,
            ),
            _run_scripted_rollout(
                env=env,
                name="zero_action_close_gripper",
                action=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
                steps=args.steps,
                seed=args.seed,
            ),
        ]
        for rollout in rollouts:
            print(
                f"{rollout.name}: "
                f"steps={rollout.steps} "
                f"min_dist={rollout.min_distance:.4f} "
                f"max_dist={rollout.max_distance:.4f} "
                f"last_dist={rollout.last_distance:.4f} "
                f"max_h={rollout.max_height:.4f} "
                f"fell={int(rollout.fell)}"
            )

        if start_distance > 0.10:
            print("\n[fail] Reset pose starts too far from the cube.")
            return 1
        if rollouts[0].last_distance > 0.14:
            print("\n[fail] Zero-action drift moves the gripper out of the learnable reach zone.")
            return 1
        print("\n[ok] Grasp reset/controller sanity checks passed.")
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
