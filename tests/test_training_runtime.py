from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot_learning.training_runtime import (
    compute_ppo_batch_size,
    ensure_dirs,
    write_run_manifest,
)


def test_compute_ppo_batch_size_stays_within_rollout_size() -> None:
    assert compute_ppo_batch_size(n_envs=4, n_steps=256, minimum=128) == 256
    assert compute_ppo_batch_size(n_envs=1, n_steps=64, minimum=128) == 64


def test_compute_ppo_batch_size_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        compute_ppo_batch_size(n_envs=0, n_steps=256)
    with pytest.raises(ValueError):
        compute_ppo_batch_size(n_envs=1, n_steps=0)


def test_write_run_manifest_persists_expected_fields(tmp_path: Path) -> None:
    output_dir = tmp_path / "logs"
    ensure_dirs(output_dir)
    manifest_path = output_dir / "run_manifest.json"

    write_run_manifest(
        manifest_path,
        repo_root=Path.cwd(),
        command=["train.py", "--project", "sedon"],
        models_root="models/sedon",
        logs_root="logs/sedon",
        tb_root="logs/tb/sedon",
        managed_layout=False,
    )

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert data["command"] == "train.py --project sedon"
    assert data["models_root"] == "models/sedon"
    assert data["managed_layout"] is False
    assert "created_at" in data
    assert "git_commit" in data
