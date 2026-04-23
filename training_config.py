"""Load per-project training configuration from JSON files."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class H1TrainConfig:
    """Static training configuration for the H1 walking policy."""

    n_envs_default: int
    total_timesteps: int
    smoke_timesteps: int
    quick_timesteps: int
    quick_n_envs: int
    n_steps: int
    n_epochs: int
    gamma: float
    gae_lambda: float
    ent_coef: float
    vf_coef: float
    max_grad_norm: float
    net_arch: list[int]
    save_freq: int
    learning_rate_initial: float
    learning_rate_floor: float
    clip_range_initial: float
    clip_range_floor: float
    finetune_ent_coef: float
    finetune_learning_rate_initial: float
    finetune_learning_rate_floor: float
    finetune_clip_range_initial: float
    finetune_clip_range_floor: float
    curriculum_stages: list[tuple[float, float]]
    max_episode_steps: int


@dataclass(frozen=True)
class GraspTrainConfig:
    """Static training configuration for the fixed-base grasp baseline."""

    n_envs_default: int
    total_timesteps: int
    smoke_timesteps: int
    n_steps: int
    n_epochs: int
    gamma: float
    gae_lambda: float
    learning_rate: float
    clip_range: float
    ent_coef: float
    vf_coef: float
    max_grad_norm: float
    net_arch: list[int]
    max_episode_steps: int
    checkpoint_freq_steps: int
    eval_freq_steps: int
    eval_episodes: int


def _load_json(path: Path) -> dict[str, Any]:
    """Load one JSON config file with a useful error if it is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Training config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Training config must be a JSON object: {path}")
    return data


def _require_keys(path: Path, data: dict[str, Any], keys: set[str]) -> None:
    """Validate that all required top-level keys are present."""
    missing = sorted(keys.difference(data))
    if missing:
        raise ValueError(f"Training config missing keys in {path}: {', '.join(missing)}")


def load_h1_train_config(repo_root: Path) -> H1TrainConfig:
    """Load the H1 walking config from ``configs/h1/train.json``."""
    path = repo_root / "configs" / "h1" / "train.json"
    data = _load_json(path)
    _require_keys(
        path,
        data,
        {
            "n_envs_default",
            "total_timesteps",
            "smoke_timesteps",
            "quick_timesteps",
            "quick_n_envs",
            "n_steps",
            "n_epochs",
            "gamma",
            "gae_lambda",
            "ent_coef",
            "vf_coef",
            "max_grad_norm",
            "net_arch",
            "save_freq",
            "learning_rate_initial",
            "learning_rate_floor",
            "clip_range_initial",
            "clip_range_floor",
            "finetune_ent_coef",
            "finetune_learning_rate_initial",
            "finetune_learning_rate_floor",
            "finetune_clip_range_initial",
            "finetune_clip_range_floor",
            "curriculum_stages",
            "max_episode_steps",
        },
    )
    return H1TrainConfig(
        n_envs_default=int(data["n_envs_default"]),
        total_timesteps=int(data["total_timesteps"]),
        smoke_timesteps=int(data["smoke_timesteps"]),
        quick_timesteps=int(data["quick_timesteps"]),
        quick_n_envs=int(data["quick_n_envs"]),
        n_steps=int(data["n_steps"]),
        n_epochs=int(data["n_epochs"]),
        gamma=float(data["gamma"]),
        gae_lambda=float(data["gae_lambda"]),
        ent_coef=float(data["ent_coef"]),
        vf_coef=float(data["vf_coef"]),
        max_grad_norm=float(data["max_grad_norm"]),
        net_arch=[int(value) for value in data["net_arch"]],
        save_freq=int(data["save_freq"]),
        learning_rate_initial=float(data["learning_rate_initial"]),
        learning_rate_floor=float(data["learning_rate_floor"]),
        clip_range_initial=float(data["clip_range_initial"]),
        clip_range_floor=float(data["clip_range_floor"]),
        finetune_ent_coef=float(data["finetune_ent_coef"]),
        finetune_learning_rate_initial=float(data["finetune_learning_rate_initial"]),
        finetune_learning_rate_floor=float(data["finetune_learning_rate_floor"]),
        finetune_clip_range_initial=float(data["finetune_clip_range_initial"]),
        finetune_clip_range_floor=float(data["finetune_clip_range_floor"]),
        curriculum_stages=[
            (float(progress), float(target_velocity))
            for progress, target_velocity in data["curriculum_stages"]
        ],
        max_episode_steps=int(data["max_episode_steps"]),
    )


def load_grasp_train_config(repo_root: Path) -> GraspTrainConfig:
    """Load the grasp config from ``configs/grasp/train.json``."""
    path = repo_root / "configs" / "grasp" / "train.json"
    data = _load_json(path)
    _require_keys(
        path,
        data,
        {
            "n_envs_default",
            "total_timesteps",
            "smoke_timesteps",
            "n_steps",
            "n_epochs",
            "gamma",
            "gae_lambda",
            "learning_rate",
            "clip_range",
            "ent_coef",
            "vf_coef",
            "max_grad_norm",
            "net_arch",
            "max_episode_steps",
            "checkpoint_freq_steps",
            "eval_freq_steps",
            "eval_episodes",
        },
    )
    return GraspTrainConfig(
        n_envs_default=int(data["n_envs_default"]),
        total_timesteps=int(data["total_timesteps"]),
        smoke_timesteps=int(data["smoke_timesteps"]),
        n_steps=int(data["n_steps"]),
        n_epochs=int(data["n_epochs"]),
        gamma=float(data["gamma"]),
        gae_lambda=float(data["gae_lambda"]),
        learning_rate=float(data["learning_rate"]),
        clip_range=float(data["clip_range"]),
        ent_coef=float(data["ent_coef"]),
        vf_coef=float(data["vf_coef"]),
        max_grad_norm=float(data["max_grad_norm"]),
        net_arch=[int(value) for value in data["net_arch"]],
        max_episode_steps=int(data["max_episode_steps"]),
        checkpoint_freq_steps=int(data["checkpoint_freq_steps"]),
        eval_freq_steps=int(data["eval_freq_steps"]),
        eval_episodes=int(data["eval_episodes"]),
    )
