from pathlib import Path

import pytest

from tools.eval_grasp import choose_model_path


def test_choose_model_path_prefers_best_model(tmp_path) -> None:
    models_root = tmp_path / "models" / "grasp"
    best_model = models_root / "best" / "best_model.zip"
    latest_model = models_root / "latest_model.zip"

    best_model.parent.mkdir(parents=True)
    latest_model.parent.mkdir(parents=True, exist_ok=True)
    best_model.write_text("best", encoding="utf-8")
    latest_model.write_text("latest", encoding="utf-8")

    assert choose_model_path(models_root) == best_model


def test_choose_model_path_falls_back_to_latest_step_checkpoint(tmp_path: Path) -> None:
    models_root = tmp_path / "models" / "grasp"
    models_root.mkdir(parents=True)
    (models_root / "grasp_ppo_100000_steps.zip").write_text("1", encoding="utf-8")
    newest = models_root / "grasp_ppo_900000_steps.zip"
    newest.write_text("2", encoding="utf-8")

    assert choose_model_path(models_root) == newest


def test_choose_model_path_raises_when_no_checkpoint_exists(tmp_path: Path) -> None:
    models_root = tmp_path / "models" / "grasp"
    models_root.mkdir(parents=True)

    with pytest.raises(FileNotFoundError):
        choose_model_path(models_root)
