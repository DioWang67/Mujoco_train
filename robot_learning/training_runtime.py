"""Shared helpers for robot training entrypoints.

This module intentionally stays small. Robot-specific env creation, reward
metrics, and curriculum logic belong in each robot package; only mechanical
runtime concerns that repeat across projects live here.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


def git_commit_short(repo_root: Path) -> str:
    """Return the short git commit for ``repo_root``.

    Args:
        repo_root: Repository root used as the git command working directory.

    Returns:
        Short commit hash, or ``"unknown"`` when git metadata is unavailable.
    """
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_root),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def compute_ppo_batch_size(n_envs: int, n_steps: int, *, minimum: int = 128) -> int:
    """Return a PPO batch size compatible with the rollout size.

    Args:
        n_envs: Number of vectorized environments.
        n_steps: PPO rollout steps per environment.
        minimum: Lower bound used for small smoke runs.

    Returns:
        Batch size no larger than ``n_envs * n_steps``.

    Raises:
        ValueError: If ``n_envs`` or ``n_steps`` is not positive.
    """
    if n_envs <= 0:
        raise ValueError("n_envs must be positive.")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")
    rollout_size = n_envs * n_steps
    return max(minimum, min(rollout_size // 4, rollout_size))


def ensure_dirs(*paths: str | Path) -> None:
    """Create all provided directories."""
    for path in paths:
        os.makedirs(path, exist_ok=True)


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    """Write a JSON object with stable formatting."""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def write_run_manifest(
    path: str | Path,
    *,
    repo_root: Path,
    command: list[str],
    models_root: str | Path,
    logs_root: str | Path,
    tb_root: str | Path,
    managed_layout: bool,
) -> None:
    """Persist a compact manifest for a training run.

    Args:
        path: Destination manifest path.
        repo_root: Git repository root.
        command: Command-line argv used for the run.
        models_root: Model artifact directory.
        logs_root: Log directory.
        tb_root: TensorBoard directory.
        managed_layout: Whether artifacts are in a managed remote-style layout.
    """
    write_json(
        path,
        {
            "created_at": datetime.now().isoformat(),
            "git_commit": git_commit_short(repo_root),
            "command": " ".join(command),
            "models_root": str(models_root),
            "logs_root": str(logs_root),
            "tb_root": str(tb_root),
            "managed_layout": managed_layout,
        },
    )
