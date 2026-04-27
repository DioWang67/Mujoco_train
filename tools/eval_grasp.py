"""Evaluate a trained fixed-base grasping policy.

Usage:
    python -m tools.eval_grasp
    python -m tools.eval_grasp --episodes 10 --no-render
    python -m tools.eval_grasp --model-path models/grasp/grasp_ppo_900000_steps.zip
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_ROOT = REPO_ROOT / "models" / "grasp"
DEFAULT_VECNORM_PATH = DEFAULT_MODELS_ROOT / "vecnorm.pkl"


@dataclass(frozen=True)
class EpisodeSummary:
    """Compact per-episode metrics for grasp evaluation."""

    episode_index: int
    steps: int
    reward: float
    success: bool
    cube_fell: bool
    max_cube_height: float
    min_distance_to_cube: float
    min_cube_to_gripper_distance: float
    max_lift_hold_steps: int


def _step_checkpoint_paths(models_root: Path) -> list[Path]:
    """Return step checkpoints sorted by training progress descending."""
    checkpoints: list[tuple[int, Path]] = []
    for path in models_root.glob("grasp_ppo_*_steps.zip"):
        stem = path.stem
        try:
            steps = int(stem.removeprefix("grasp_ppo_").removesuffix("_steps"))
        except ValueError:
            continue
        checkpoints.append((steps, path))
    return [path for _, path in sorted(checkpoints, reverse=True)]


def choose_model_path(models_root: Path, model_path: str | None = None) -> Path:
    """Choose the checkpoint to evaluate.

    Priority:
    1. Explicit ``model_path`` when provided.
    2. ``best/best_model.zip``
    3. ``latest_model.zip``
    4. Highest-step ``grasp_ppo_*_steps.zip``
    """
    if model_path is not None:
        selected_path = Path(model_path).expanduser()
        if not selected_path.is_absolute():
            selected_path = (REPO_ROOT / selected_path).resolve()
        if not selected_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {selected_path}")
        return selected_path

    candidates = [
        models_root / "best" / "best_model.zip",
        models_root / "latest_model.zip",
        *_step_checkpoint_paths(models_root),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No grasp checkpoint found. Expected one of: "
        f"{models_root / 'best' / 'best_model.zip'}, "
        f"{models_root / 'latest_model.zip'}, or "
        f"{models_root / 'grasp_ppo_*_steps.zip'}."
    )


def _build_eval_env(
    phase: str,
    randomize_cube_pose: bool,
    render_mode: str | None,
) -> Any:
    """Construct the grasp environment lazily to keep imports lightweight."""
    from gymnasium.wrappers import TimeLimit

    from grasp_baseline.env import FixedBaseGraspEnv
    from grasp_baseline.train import MAX_EPISODE_STEPS

    return TimeLimit(
        FixedBaseGraspEnv(
            task_phase=phase,
            randomize_cube_pose=randomize_cube_pose,
            render_mode=render_mode,
        ),
        max_episode_steps=MAX_EPISODE_STEPS,
    )


def _load_model_and_vecnorm(
    model_path: Path,
    vecnorm_path: Path | None,
) -> tuple[Any, Any]:
    """Load the PPO checkpoint and optional VecNormalize statistics."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    model = PPO.load(str(model_path))
    vec_norm = None
    if vecnorm_path is not None and vecnorm_path.exists():
        dummy = DummyVecEnv([lambda: _build_eval_env("full", True, None)])
        vec_norm = VecNormalize.load(str(vecnorm_path), dummy)
        vec_norm.training = False
        vec_norm.norm_reward = False
    return model, vec_norm


def run_eval(
    *,
    model_path: Path,
    vecnorm_path: Path | None,
    episodes: int,
    phase: str,
    fixed_cube: bool,
    render: bool,
    seed: int,
    print_every: int,
) -> list[EpisodeSummary]:
    """Run deterministic evaluation episodes and return compact metrics."""
    if episodes <= 0:
        raise ValueError("--episodes must be positive.")
    if print_every < 0:
        raise ValueError("--print-every must be non-negative.")

    render_mode = "human" if render else None
    model, vec_norm = _load_model_and_vecnorm(model_path, vecnorm_path)
    env = _build_eval_env(
        phase=phase,
        randomize_cube_pose=not fixed_cube,
        render_mode=render_mode,
    )
    summaries: list[EpisodeSummary] = []

    try:
        for episode_index in range(1, episodes + 1):
            obs, _ = env.reset(seed=seed + episode_index - 1)
            if render:
                env.render()
            done = False
            episode_reward = 0.0
            steps = 0
            success = False
            cube_fell = False
            max_cube_height = 0.0
            min_distance_to_cube = float("inf")
            min_cube_to_gripper_distance = float("inf")
            max_lift_hold_steps = 0

            while not done:
                obs_in = (
                    vec_norm.normalize_obs(obs.reshape(1, -1))[0]
                    if vec_norm is not None
                    else obs
                )
                action, _ = model.predict(obs_in, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += float(reward)
                steps += 1
                done = bool(terminated or truncated)

                success = success or bool(info.get("is_success", False))
                cube_fell = cube_fell or bool(info.get("cube_fell", False))
                max_cube_height = max(
                    max_cube_height,
                    float(info.get("cube_height_above_table", 0.0)),
                )
                min_distance_to_cube = min(
                    min_distance_to_cube,
                    float(info.get("distance_to_cube", float("inf"))),
                )
                min_cube_to_gripper_distance = min(
                    min_cube_to_gripper_distance,
                    float(info.get("cube_to_gripper_distance", float("inf"))),
                )
                max_lift_hold_steps = max(
                    max_lift_hold_steps,
                    int(info.get("lift_hold_steps", 0)),
                )

                if render:
                    env.render()
                    time.sleep(1 / 50)
                if print_every and steps % print_every == 0:
                    print(
                        f"[ep {episode_index}] "
                        f"step={steps:4d} "
                        f"reward={episode_reward:8.2f} "
                        f"success={int(success)} "
                        f"fell={int(cube_fell)} "
                        f"max_h={max_cube_height:.4f} "
                        f"min_dist={min_distance_to_cube:.4f}",
                    )

            summaries.append(
                EpisodeSummary(
                    episode_index=episode_index,
                    steps=steps,
                    reward=episode_reward,
                    success=success,
                    cube_fell=cube_fell,
                    max_cube_height=max_cube_height,
                    min_distance_to_cube=min_distance_to_cube,
                    min_cube_to_gripper_distance=min_cube_to_gripper_distance,
                    max_lift_hold_steps=max_lift_hold_steps,
                )
            )
    finally:
        env.close()
        if vec_norm is not None:
            vec_norm.venv.close()

    return summaries


def _print_summary(summaries: list[EpisodeSummary]) -> None:
    """Print per-episode and aggregate metrics."""
    for summary in summaries:
        print(
            "Episode "
            f"{summary.episode_index:2d} | "
            f"steps={summary.steps:4d} | "
            f"reward={summary.reward:8.2f} | "
            f"success={int(summary.success)} | "
            f"fell={int(summary.cube_fell)} | "
            f"max_h={summary.max_cube_height:.4f} | "
            f"min_dist={summary.min_distance_to_cube:.4f} | "
            f"min_align={summary.min_cube_to_gripper_distance:.4f} | "
            f"hold={summary.max_lift_hold_steps}"
        )

    rewards = np.array([summary.reward for summary in summaries], dtype=np.float64)
    steps = np.array([summary.steps for summary in summaries], dtype=np.float64)
    successes = np.array([summary.success for summary in summaries], dtype=np.float64)
    falls = np.array([summary.cube_fell for summary in summaries], dtype=np.float64)
    max_heights = np.array(
        [summary.max_cube_height for summary in summaries],
        dtype=np.float64,
    )
    print()
    print(
        "Aggregate | "
        f"episodes={len(summaries)} | "
        f"mean_reward={rewards.mean():.2f} | "
        f"mean_steps={steps.mean():.1f} | "
        f"success_rate={100.0 * successes.mean():.1f}% | "
        f"fall_rate={100.0 * falls.mean():.1f}% | "
        f"mean_max_h={max_heights.mean():.4f}"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI options for grasp evaluation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Explicit checkpoint path. Defaults to best/latest/last-step selection.",
    )
    parser.add_argument(
        "--models-root",
        type=Path,
        default=DEFAULT_MODELS_ROOT,
        help="Directory containing grasp checkpoints and vecnorm.pkl.",
    )
    parser.add_argument(
        "--vecnorm-path",
        type=Path,
        default=DEFAULT_VECNORM_PATH,
        help="Path to VecNormalize stats. Use --no-vecnorm to disable.",
    )
    parser.add_argument(
        "--no-vecnorm",
        action="store_true",
        help="Disable VecNormalize loading even if vecnorm.pkl exists.",
    )
    parser.add_argument(
        "--phase",
        choices=["reach", "grasp", "lift", "full"],
        default="full",
        help="Reward phase used to instantiate the eval environment.",
    )
    parser.add_argument(
        "--fixed-cube",
        action="store_true",
        help="Disable cube pose randomization during evaluation.",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Headless evaluation without the MuJoCo viewer.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=50,
        help="Print live progress every N steps. Use 0 to disable.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for grasp evaluation."""
    args = parse_args(argv)
    models_root = Path(args.models_root).resolve()
    vecnorm_path = None if args.no_vecnorm else Path(args.vecnorm_path).resolve()
    model_path = choose_model_path(models_root, args.model_path)

    print(f"Models root : {models_root}")
    print(f"Model       : {model_path}")
    print(f"VecNormalize: {vecnorm_path if vecnorm_path is not None else 'disabled'}")
    print(f"Phase       : {args.phase}")
    print(f"Fixed cube  : {args.fixed_cube}")
    print()

    summaries = run_eval(
        model_path=model_path,
        vecnorm_path=vecnorm_path,
        episodes=args.episodes,
        phase=args.phase,
        fixed_cube=args.fixed_cube,
        render=not args.no_render,
        seed=args.seed,
        print_every=args.print_every,
    )
    _print_summary(summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
