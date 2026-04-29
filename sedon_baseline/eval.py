"""Evaluate a trained Sedon standing policy.

Usage:
    python -m sedon_baseline.eval --episodes 5
    python -m sedon_baseline.eval --model-path models/sedon/latest_model.zip
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from sedon_baseline.env import SedonStandingEnv
from sedon_baseline.train import MAX_EPISODE_STEPS

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_ROOT = REPO_ROOT / "models" / "sedon"
DEFAULT_REPORT_PATH = REPO_ROOT / "reports" / "sedon_eval.csv"


@dataclass(frozen=True)
class SedonEvalEpisode:
    """Metrics collected from one Sedon evaluation episode."""

    episode: int
    reward: float
    length: int
    fell: bool
    final_base_height: float
    final_upright: float


def resolve_model_path(models_root: Path, explicit_model_path: Path | None) -> Path:
    """Resolve the Sedon policy checkpoint to evaluate.

    Args:
        models_root: Directory containing Sedon checkpoints.
        explicit_model_path: Optional user-selected model path.

    Returns:
        Existing model path.

    Raises:
        FileNotFoundError: If no candidate checkpoint exists.
    """
    if explicit_model_path is not None:
        if not explicit_model_path.is_file():
            raise FileNotFoundError(f"Model checkpoint not found: {explicit_model_path}")
        return explicit_model_path

    candidates = [
        models_root / "best" / "best_model.zip",
        models_root / "latest_model.zip",
    ]
    candidates.extend(
        sorted(
            models_root.glob("sedon_ppo_*_steps.zip"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "No Sedon checkpoint found. Expected best/best_model.zip, "
        "latest_model.zip, or sedon_ppo_*_steps.zip under "
        f"{models_root}."
    )


def resolve_vecnorm_path(models_root: Path, explicit_vecnorm_path: Path | None) -> Path:
    """Resolve the VecNormalize stats used with the Sedon policy."""
    vecnorm_path = explicit_vecnorm_path or (models_root / "vecnorm.pkl")
    if not vecnorm_path.is_file():
        raise FileNotFoundError(f"VecNormalize file not found: {vecnorm_path}")
    return vecnorm_path


def _make_eval_env(seed: int):
    """Build one deterministic monitored eval environment."""

    def _thunk():
        env = SedonStandingEnv(reset_noise_scale=0.0)
        env = TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)
        env.reset(seed=seed)
        return env

    return _thunk


def build_eval_vec_env(vecnorm_path: Path, seed: int) -> VecNormalize:
    """Create a normalized Sedon eval environment."""
    vec_env = DummyVecEnv([_make_eval_env(seed)])
    eval_env = VecNormalize.load(str(vecnorm_path), vec_env)
    eval_env.training = False
    eval_env.norm_reward = False
    return eval_env


def evaluate_policy(
    model_path: Path,
    vecnorm_path: Path,
    *,
    episodes: int,
    seed: int,
) -> list[SedonEvalEpisode]:
    """Run deterministic Sedon policy evaluation episodes."""
    if episodes <= 0:
        raise ValueError("episodes must be positive.")

    eval_env = build_eval_vec_env(vecnorm_path, seed)
    model = PPO.load(str(model_path), env=eval_env)
    results: list[SedonEvalEpisode] = []
    try:
        for episode_index in range(1, episodes + 1):
            obs = eval_env.reset()
            episode_reward = 0.0
            episode_length = 0
            final_info: dict = {}

            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, rewards, dones, infos = eval_env.step(action)
                episode_reward += float(rewards[0])
                episode_length += 1
                final_info = infos[0]
                if bool(dones[0]):
                    break

            time_limit_reached = bool(final_info.get("TimeLimit.truncated", False))
            results.append(
                SedonEvalEpisode(
                    episode=episode_index,
                    reward=episode_reward,
                    length=episode_length,
                    fell=not time_limit_reached,
                    final_base_height=float(final_info.get("base_height", np.nan)),
                    final_upright=float(final_info.get("upright", np.nan)),
                )
            )
    finally:
        eval_env.close()
    return results


def write_csv(path: Path, episodes: list[SedonEvalEpisode]) -> None:
    """Write per-episode metrics to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "episode",
                "reward",
                "length",
                "fell",
                "final_base_height",
                "final_upright",
            ],
        )
        writer.writeheader()
        for episode in episodes:
            writer.writerow(episode.__dict__)


def print_summary(episodes: list[SedonEvalEpisode]) -> None:
    """Print compact evaluation summary metrics."""
    rewards = np.array([episode.reward for episode in episodes], dtype=np.float64)
    lengths = np.array([episode.length for episode in episodes], dtype=np.float64)
    falls = np.array([episode.fell for episode in episodes], dtype=np.float64)
    base_heights = np.array([episode.final_base_height for episode in episodes], dtype=np.float64)
    uprights = np.array([episode.final_upright for episode in episodes], dtype=np.float64)

    print("Sedon eval summary")
    print(f"episodes          : {len(episodes)}")
    print(f"mean_reward       : {float(np.mean(rewards)):.3f}")
    print(f"mean_length       : {float(np.mean(lengths)):.1f}")
    print(f"fall_rate         : {100.0 * float(np.mean(falls)):.1f}%")
    print(f"mean_final_base_z : {float(np.nanmean(base_heights)):.3f}")
    print(f"mean_final_upright: {float(np.nanmean(uprights)):.3f}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse Sedon evaluation CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--models-root",
        type=Path,
        default=DEFAULT_MODELS_ROOT,
        help="Directory containing Sedon model artifacts.",
    )
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--vecnorm-path", type=Path, default=None)
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Where to write per-episode CSV metrics.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Evaluate a Sedon standing policy checkpoint."""
    args = parse_args(argv)
    model_path = resolve_model_path(args.models_root, args.model_path)
    vecnorm_path = resolve_vecnorm_path(args.models_root, args.vecnorm_path)
    print(f"Model     : {model_path}")
    print(f"VecNormalize: {vecnorm_path}")

    episodes = evaluate_policy(
        model_path=model_path,
        vecnorm_path=vecnorm_path,
        episodes=args.episodes,
        seed=args.seed,
    )
    write_csv(args.out_csv, episodes)
    print_summary(episodes)
    print(f"CSV       : {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
