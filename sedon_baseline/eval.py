"""Evaluate a trained Sedon standing policy.

Usage:
    python -m sedon_baseline.eval --episodes 5
    python -m sedon_baseline.eval --model-path models/sedon/latest_model.zip
    python -m sedon_baseline.eval --episodes 1 --render
    python -m sedon_baseline.eval --episodes 1 --record
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from sedon_baseline.env import (
    CONFIG_OVERRIDES_ENV,
    DEFAULT_SCENE_PATH,
    SedonStandingConfig,
    SedonStandingEnv,
    load_sedon_config_from_env,
)
from sedon_baseline.train import MAX_EPISODE_STEPS

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_ROOT = REPO_ROOT / "models" / "sedon"
DEFAULT_TRAIN_CONFIG_PATH = REPO_ROOT / "logs" / "sedon" / "train_config.json"
DEFAULT_REPORT_PATH = REPO_ROOT / "reports" / "sedon_eval.csv"
DEFAULT_VIDEO_PATH = REPO_ROOT / "reports" / "sedon_eval.gif"
STEP_CHECKPOINT_RE = re.compile(r"sedon_ppo_(\d+)_steps\.zip$")


def _is_valid_sb3_checkpoint(path: Path) -> bool:
    """Return whether a Stable-Baselines3 checkpoint zip can be read.

    A plain ``zipfile.is_zipfile`` check is not enough here because a partially
    written checkpoint can still look like a valid zip container while failing
    CRC validation when SB3 reads the embedded ``data`` file.
    """
    if not path.is_file() or not zipfile.is_zipfile(path):
        return False
    try:
        with zipfile.ZipFile(path) as archive:
            archive.read("data")
        return True
    except (KeyError, OSError, ValueError, zipfile.BadZipFile):
        return False


@dataclass(frozen=True)
class SedonEvalEpisode:
    """Metrics collected from one Sedon evaluation episode."""

    episode: int
    reward: float
    length: int
    fell: bool
    final_base_height: float
    final_upright: float
    final_base_x: float
    mean_forward_velocity: float
    right_knee_violation_steps: int
    left_knee_violation_steps: int
    total_knee_violation_steps: int
    max_right_knee_violation: float
    max_left_knee_violation: float
    both_contact_ratio: float
    single_contact_ratio: float
    no_contact_ratio: float
    mean_support_force_ratio: float
    mean_swing_force_ratio: float
    support_load_hit_rate: float
    left_load_correct_rate: float
    right_load_correct_rate: float
    max_foot_clearance: float
    max_force_imbalance: float


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
        if not _is_valid_sb3_checkpoint(explicit_model_path):
            raise ValueError(f"Model checkpoint is not a readable SB3 zip: {explicit_model_path}")
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
        if _is_valid_sb3_checkpoint(candidate):
            return candidate
    raise FileNotFoundError(
        "No readable Sedon checkpoint found. Expected a valid best/best_model.zip, "
        "latest_model.zip, or sedon_ppo_*_steps.zip under "
        f"{models_root}."
    )


def resolve_vecnorm_path(
    models_root: Path,
    model_path: Path,
    explicit_vecnorm_path: Path | None,
) -> Path:
    """Resolve the VecNormalize stats used with the Sedon policy."""
    if explicit_vecnorm_path is not None:
        if not explicit_vecnorm_path.is_file():
            raise FileNotFoundError(f"VecNormalize file not found: {explicit_vecnorm_path}")
        return explicit_vecnorm_path

    candidates: list[Path] = []
    if model_path.name == "best_model.zip" and model_path.parent.name == "best":
        candidates.append(model_path.parent / "vecnorm.pkl")

    match = STEP_CHECKPOINT_RE.fullmatch(model_path.name)
    if match:
        candidates.append(models_root / f"sedon_vecnorm_{match.group(1)}_steps.pkl")

    candidates.append(models_root / "vecnorm.pkl")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "VecNormalize file not found. Expected one of: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def _load_train_config(path: Path, *, required: bool) -> dict | None:
    """Load a Sedon training config JSON object if available.

    Args:
        path: Candidate ``train_config.json`` path.
        required: Whether absence should raise instead of falling back.

    Returns:
        Parsed JSON object, or ``None`` when the optional file is absent.

    Raises:
        FileNotFoundError: If ``required`` is true and the path is missing.
        ValueError: If the file does not decode to a JSON object.
    """
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"Sedon train config not found: {path}")
        return None
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Sedon train config must be a JSON object: {path}")
    return data


def _load_reward_config_from_train_config(
    train_config_path: Path,
    *,
    required: bool,
) -> SedonStandingConfig | None:
    """Return ``SedonStandingConfig`` from a saved training config."""
    data = _load_train_config(train_config_path, required=required)
    if data is None:
        return None
    reward_config = data.get("reward_config")
    if reward_config is None:
        if required:
            raise ValueError(f"Sedon train config has no reward_config: {train_config_path}")
        return None
    if not isinstance(reward_config, dict):
        raise ValueError(f"Sedon reward_config must be a JSON object: {train_config_path}")
    try:
        return SedonStandingConfig(**reward_config)
    except TypeError as exc:
        raise ValueError(
            f"Sedon reward_config is incompatible with current env fields: {train_config_path}"
        ) from exc


def _resolve_scene_path(
    *,
    explicit_scene_path: Path | None,
    train_config_path: Path,
    ignore_train_config: bool,
) -> Path:
    """Resolve the MuJoCo scene path for evaluation."""
    if explicit_scene_path is not None:
        return explicit_scene_path.expanduser()

    env_scene_path = os.environ.get("SEDON_SCENE_PATH")
    if env_scene_path:
        return Path(env_scene_path).expanduser()

    if not ignore_train_config:
        data = _load_train_config(train_config_path, required=False)
        if data is not None and isinstance(data.get("scene_path"), str):
            return Path(data["scene_path"]).expanduser()

    return DEFAULT_SCENE_PATH


def _parse_optional_range(raw_value: str | None, *, option_name: str) -> tuple[float, float] | None:
    """Parse a `lower,upper` string into an optional float tuple."""
    if raw_value is None:
        return None
    parts = [part.strip() for part in raw_value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"{option_name} must be 'lower,upper'.")
    lower, upper = float(parts[0]), float(parts[1])
    if lower > upper:
        raise argparse.ArgumentTypeError(f"{option_name} lower must be <= upper.")
    return (lower, upper)


def _build_reward_config_with_safe_ranges(
    right_knee_safe_range: tuple[float, float] | None,
    left_knee_safe_range: tuple[float, float] | None,
    *,
    train_config_path: Path,
    ignore_train_config: bool,
) -> SedonStandingConfig:
    """Return the active Sedon config with optional eval-only knee-safe overrides."""
    if os.environ.get(CONFIG_OVERRIDES_ENV):
        config = load_sedon_config_from_env()
    elif ignore_train_config:
        config = load_sedon_config_from_env()
    else:
        config = (
            _load_reward_config_from_train_config(train_config_path, required=False)
            or load_sedon_config_from_env()
        )
    overrides: dict[str, float] = {}
    if right_knee_safe_range is not None:
        overrides["right_knee_safe_lower"] = right_knee_safe_range[0]
        overrides["right_knee_safe_upper"] = right_knee_safe_range[1]
    if left_knee_safe_range is not None:
        overrides["left_knee_safe_lower"] = left_knee_safe_range[0]
        overrides["left_knee_safe_upper"] = left_knee_safe_range[1]
    if not overrides:
        return config
    return SedonStandingConfig(**{**config.__dict__, **overrides})


def _make_eval_env(
    seed: int,
    render_mode: str | None,
    reward_config: SedonStandingConfig,
    scene_path: Path,
):
    """Build one deterministic monitored eval environment."""

    def _thunk():
        env = SedonStandingEnv(
            scene_path=scene_path,
            reset_noise_scale=0.0,
            render_mode=render_mode,
            reward_config=reward_config,
        )
        env = TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)
        env.reset(seed=seed)
        return env

    return _thunk


def build_eval_vec_env(
    vecnorm_path: Path,
    seed: int,
    render_mode: str | None,
    reward_config: SedonStandingConfig,
    scene_path: Path,
) -> VecNormalize:
    """Create a normalized Sedon eval environment."""
    vec_env = DummyVecEnv([_make_eval_env(seed, render_mode, reward_config, scene_path)])
    eval_env = VecNormalize.load(str(vecnorm_path), vec_env)
    eval_env.training = False
    eval_env.norm_reward = False
    return eval_env


def _capture_rgb_frame(eval_env: VecNormalize) -> np.ndarray | None:
    """Return one RGB frame from a vectorized MuJoCo environment if available."""
    frame = eval_env.render()
    if isinstance(frame, list):
        frame = frame[0] if frame else None
    if frame is None:
        # Some VecEnv wrappers do not return frames directly; the wrapped env
        # still has the MuJoCo renderer configured with render_mode="rgb_array".
        frame = eval_env.venv.envs[0].render()
    if isinstance(frame, list):
        frame = frame[0] if frame else None
    if frame is None:
        return None
    return np.asarray(frame)


def _save_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    """Save captured RGB frames as an animation file.

    Args:
        path: Destination animation path.
        frames: RGB frames captured during evaluation.
        fps: Output video frame rate.

    Raises:
        RuntimeError: If ``imageio`` is unavailable.
    """
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise RuntimeError("Recording requires imageio. Install project requirements first.") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        imageio.mimsave(path, frames, fps=fps)
    except ValueError as exc:
        if path.suffix.lower() == ".mp4":
            raise RuntimeError(
                "MP4 recording requires an ffmpeg backend. Install it with "
                "`pip install imageio[ffmpeg]`, or use a .gif --video-path."
            ) from exc
        raise


def evaluate_policy(
    model_path: Path,
    vecnorm_path: Path,
    *,
    episodes: int,
    seed: int,
    render: bool = False,
    record_path: Path | None = None,
    fps: int = SedonStandingEnv.metadata["render_fps"],
    right_knee_safe_range: tuple[float, float] | None = None,
    left_knee_safe_range: tuple[float, float] | None = None,
    train_config_path: Path = DEFAULT_TRAIN_CONFIG_PATH,
    scene_path: Path | None = None,
    ignore_train_config: bool = False,
) -> list[SedonEvalEpisode]:
    """Run deterministic Sedon policy evaluation episodes."""
    if episodes <= 0:
        raise ValueError("episodes must be positive.")
    if fps <= 0:
        raise ValueError("fps must be positive.")
    if render and record_path is not None:
        raise ValueError("--render and --record cannot be used together.")

    render_mode = "rgb_array" if record_path is not None else ("human" if render else None)
    reward_config = _build_reward_config_with_safe_ranges(
        right_knee_safe_range,
        left_knee_safe_range,
        train_config_path=train_config_path,
        ignore_train_config=ignore_train_config,
    )
    resolved_scene_path = _resolve_scene_path(
        explicit_scene_path=scene_path,
        train_config_path=train_config_path,
        ignore_train_config=ignore_train_config,
    )
    eval_env = build_eval_vec_env(
        vecnorm_path,
        seed,
        render_mode,
        reward_config,
        resolved_scene_path,
    )
    model = PPO.load(str(model_path), env=eval_env)
    results: list[SedonEvalEpisode] = []
    frames: list[np.ndarray] = []
    try:
        for episode_index in range(1, episodes + 1):
            obs = eval_env.reset()
            episode_reward = 0.0
            episode_length = 0
            forward_velocity_sum = 0.0
            right_knee_violation_steps = 0
            left_knee_violation_steps = 0
            max_right_knee_violation = 0.0
            max_left_knee_violation = 0.0
            both_contact_steps = 0
            single_contact_steps = 0
            no_contact_steps = 0
            support_phase_steps = 0
            support_force_ratio_sum = 0.0
            swing_force_ratio_sum = 0.0
            support_load_hit_steps = 0
            left_support_phase_steps = 0
            right_support_phase_steps = 0
            left_load_correct_steps = 0
            right_load_correct_steps = 0
            max_foot_clearance = 0.0
            max_force_imbalance = 0.0
            final_info: dict = {}
            if record_path is not None:
                frame = _capture_rgb_frame(eval_env)
                if frame is not None:
                    frames.append(frame)

            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, rewards, dones, infos = eval_env.step(action)
                episode_reward += float(rewards[0])
                episode_length += 1
                final_info = infos[0]
                forward_velocity_sum += float(final_info.get("forward_velocity", 0.0))
                right_violation = float(final_info.get("right_knee_safe_violation", 0.0))
                left_violation = float(final_info.get("left_knee_safe_violation", 0.0))
                right_knee_violation_steps += int(right_violation > 0.0)
                left_knee_violation_steps += int(left_violation > 0.0)
                max_right_knee_violation = max(max_right_knee_violation, right_violation)
                max_left_knee_violation = max(max_left_knee_violation, left_violation)

                left_contact = bool(final_info.get("left_contact", False))
                right_contact = bool(final_info.get("right_contact", False))
                if left_contact and right_contact:
                    both_contact_steps += 1
                elif left_contact or right_contact:
                    single_contact_steps += 1
                else:
                    no_contact_steps += 1

                left_force_ratio = float(final_info.get("left_force_ratio", 0.0))
                right_force_ratio = float(final_info.get("right_force_ratio", 0.0))
                max_force_imbalance = max(
                    max_force_imbalance,
                    abs(left_force_ratio - right_force_ratio),
                )

                support_side = str(final_info.get("support_side", ""))
                if support_side in {"left", "right"}:
                    support_phase_steps += 1
                    support_force_ratio = float(final_info.get("force_ratio", 0.0))
                    swing_force_ratio = float(final_info.get("swing_force_ratio", 0.0))
                    support_force_ratio_sum += support_force_ratio
                    swing_force_ratio_sum += swing_force_ratio
                    support_load_hit_steps += int(support_force_ratio >= 0.55)
                    max_foot_clearance = max(
                        max_foot_clearance,
                        float(final_info.get("foot_clearance", 0.0)),
                    )
                    if support_side == "left":
                        left_support_phase_steps += 1
                        left_load_correct_steps += int(left_force_ratio > right_force_ratio)
                    else:
                        right_support_phase_steps += 1
                        right_load_correct_steps += int(right_force_ratio > left_force_ratio)
                if record_path is not None:
                    frame = _capture_rgb_frame(eval_env)
                    if frame is not None:
                        frames.append(frame)
                elif render:
                    eval_env.render()
                    time.sleep(1.0 / fps)
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
                    final_base_x=float(final_info.get("base_x_position", np.nan)),
                    mean_forward_velocity=forward_velocity_sum / max(1, episode_length),
                    right_knee_violation_steps=right_knee_violation_steps,
                    left_knee_violation_steps=left_knee_violation_steps,
                    total_knee_violation_steps=(
                        right_knee_violation_steps + left_knee_violation_steps
                    ),
                    max_right_knee_violation=max_right_knee_violation,
                    max_left_knee_violation=max_left_knee_violation,
                    both_contact_ratio=both_contact_steps / max(1, episode_length),
                    single_contact_ratio=single_contact_steps / max(1, episode_length),
                    no_contact_ratio=no_contact_steps / max(1, episode_length),
                    mean_support_force_ratio=(
                        support_force_ratio_sum / max(1, support_phase_steps)
                    ),
                    mean_swing_force_ratio=swing_force_ratio_sum / max(1, support_phase_steps),
                    support_load_hit_rate=support_load_hit_steps / max(1, support_phase_steps),
                    left_load_correct_rate=(
                        left_load_correct_steps / max(1, left_support_phase_steps)
                    ),
                    right_load_correct_rate=(
                        right_load_correct_steps / max(1, right_support_phase_steps)
                    ),
                    max_foot_clearance=max_foot_clearance,
                    max_force_imbalance=max_force_imbalance,
                )
            )
        if record_path is not None:
            if not frames:
                raise RuntimeError("No frames were captured; cannot save evaluation video.")
            _save_video(record_path, frames, fps)
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
                "final_base_x",
                "mean_forward_velocity",
                "right_knee_violation_steps",
                "left_knee_violation_steps",
                "total_knee_violation_steps",
                "max_right_knee_violation",
                "max_left_knee_violation",
                "both_contact_ratio",
                "single_contact_ratio",
                "no_contact_ratio",
                "mean_support_force_ratio",
                "mean_swing_force_ratio",
                "support_load_hit_rate",
                "left_load_correct_rate",
                "right_load_correct_rate",
                "max_foot_clearance",
                "max_force_imbalance",
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
    base_x = np.array([episode.final_base_x for episode in episodes], dtype=np.float64)
    forward_velocities = np.array(
        [episode.mean_forward_velocity for episode in episodes],
        dtype=np.float64,
    )
    knee_violation_steps = np.array(
        [episode.total_knee_violation_steps for episode in episodes],
        dtype=np.float64,
    )
    right_knee_violation = np.array(
        [episode.max_right_knee_violation for episode in episodes],
        dtype=np.float64,
    )
    left_knee_violation = np.array(
        [episode.max_left_knee_violation for episode in episodes],
        dtype=np.float64,
    )
    both_contact_ratios = np.array(
        [episode.both_contact_ratio for episode in episodes],
        dtype=np.float64,
    )
    single_contact_ratios = np.array(
        [episode.single_contact_ratio for episode in episodes],
        dtype=np.float64,
    )
    no_contact_ratios = np.array(
        [episode.no_contact_ratio for episode in episodes],
        dtype=np.float64,
    )
    support_force_ratios = np.array(
        [episode.mean_support_force_ratio for episode in episodes],
        dtype=np.float64,
    )
    swing_force_ratios = np.array(
        [episode.mean_swing_force_ratio for episode in episodes],
        dtype=np.float64,
    )
    support_load_hit_rates = np.array(
        [episode.support_load_hit_rate for episode in episodes],
        dtype=np.float64,
    )
    left_load_correct_rates = np.array(
        [episode.left_load_correct_rate for episode in episodes],
        dtype=np.float64,
    )
    right_load_correct_rates = np.array(
        [episode.right_load_correct_rate for episode in episodes],
        dtype=np.float64,
    )
    max_foot_clearances = np.array(
        [episode.max_foot_clearance for episode in episodes],
        dtype=np.float64,
    )
    max_force_imbalances = np.array(
        [episode.max_force_imbalance for episode in episodes],
        dtype=np.float64,
    )

    print("Sedon eval summary")
    print(f"episodes          : {len(episodes)}")
    print(f"mean_reward       : {float(np.mean(rewards)):.3f}")
    print(f"mean_length       : {float(np.mean(lengths)):.1f}")
    print(f"fall_rate         : {100.0 * float(np.mean(falls)):.1f}%")
    print(f"mean_final_base_z : {float(np.nanmean(base_heights)):.3f}")
    print(f"mean_final_upright: {float(np.nanmean(uprights)):.3f}")
    print(f"mean_final_base_x : {float(np.nanmean(base_x)):.3f}")
    print(f"mean_forward_vel  : {float(np.nanmean(forward_velocities)):.3f}")
    print(f"mean_knee_violation_steps: {float(np.nanmean(knee_violation_steps)):.1f}")
    print(f"max_right_knee_violation : {float(np.nanmax(right_knee_violation)):.4f}")
    print(f"max_left_knee_violation  : {float(np.nanmax(left_knee_violation)):.4f}")
    print(f"both_contact_ratio       : {float(np.nanmean(both_contact_ratios)):.1%}")
    print(f"single_contact_ratio     : {float(np.nanmean(single_contact_ratios)):.1%}")
    print(f"no_contact_ratio         : {float(np.nanmean(no_contact_ratios)):.1%}")
    print(f"mean_support_force_ratio : {float(np.nanmean(support_force_ratios)):.3f}")
    print(f"mean_swing_force_ratio   : {float(np.nanmean(swing_force_ratios)):.3f}")
    print(f"support_load_hit_rate    : {float(np.nanmean(support_load_hit_rates)):.1%}")
    print(f"left_load_correct_rate   : {float(np.nanmean(left_load_correct_rates)):.1%}")
    print(f"right_load_correct_rate  : {float(np.nanmean(right_load_correct_rates)):.1%}")
    print(f"max_foot_clearance       : {float(np.nanmax(max_foot_clearances)):.4f}")
    print(f"max_force_imbalance      : {float(np.nanmax(max_force_imbalances)):.3f}")


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
        "--train-config-path",
        type=Path,
        default=DEFAULT_TRAIN_CONFIG_PATH,
        help=(
            "Training config used to restore Sedon reward/gait config and scene path. "
            "Defaults to logs/sedon/train_config.json."
        ),
    )
    parser.add_argument(
        "--ignore-train-config",
        action="store_true",
        help="Use env/default Sedon config instead of logs/sedon/train_config.json.",
    )
    parser.add_argument(
        "--scene-path",
        type=Path,
        default=None,
        help="Explicit MuJoCo scene path for evaluation.",
    )
    parser.add_argument(
        "--right-knee-safe-range",
        type=lambda value: _parse_optional_range(
            value,
            option_name="--right-knee-safe-range",
        ),
        default=None,
        help="Optional right-knee soft-safe qpos range as 'lower,upper'.",
    )
    parser.add_argument(
        "--left-knee-safe-range",
        type=lambda value: _parse_optional_range(
            value,
            option_name="--left-knee-safe-range",
        ),
        default=None,
        help="Optional left-knee soft-safe qpos range as 'lower,upper'.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Where to write per-episode CSV metrics.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Open the MuJoCo viewer and watch the deterministic policy live.",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record the deterministic policy to an animation instead of opening a viewer.",
    )
    parser.add_argument(
        "--video-path",
        type=Path,
        default=DEFAULT_VIDEO_PATH,
        help="Where to write the animation when --record is enabled.",
    )
    parser.add_argument("--fps", type=int, default=SedonStandingEnv.metadata["render_fps"])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Evaluate a Sedon standing policy checkpoint."""
    args = parse_args(argv)
    model_path = resolve_model_path(args.models_root, args.model_path)
    vecnorm_path = resolve_vecnorm_path(args.models_root, model_path, args.vecnorm_path)
    scene_path = _resolve_scene_path(
        explicit_scene_path=args.scene_path,
        train_config_path=args.train_config_path,
        ignore_train_config=args.ignore_train_config,
    )
    print(f"Model     : {model_path}")
    print(f"VecNormalize: {vecnorm_path}")
    print(f"Train config: {'(ignored)' if args.ignore_train_config else args.train_config_path}")
    print(f"Scene     : {scene_path}")

    episodes = evaluate_policy(
        model_path=model_path,
        vecnorm_path=vecnorm_path,
        episodes=args.episodes,
        seed=args.seed,
        render=args.render,
        record_path=args.video_path if args.record else None,
        fps=args.fps,
        right_knee_safe_range=args.right_knee_safe_range,
        left_knee_safe_range=args.left_knee_safe_range,
        train_config_path=args.train_config_path,
        scene_path=scene_path,
        ignore_train_config=args.ignore_train_config,
    )
    write_csv(args.out_csv, episodes)
    print_summary(episodes)
    print(f"CSV       : {args.out_csv}")
    if args.record:
        print(f"Video     : {args.video_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
