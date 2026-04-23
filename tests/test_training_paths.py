import os
from pathlib import Path

from training_paths import resolve_training_paths


def test_resolve_training_paths_uses_local_legacy_dirs(tmp_path: Path) -> None:
    paths = resolve_training_paths(
        tmp_path,
        "grasp",
        legacy_model_dir="models/grasp",
        legacy_log_dir="logs/grasp",
        legacy_tb_dir="logs/tb/grasp",
    )

    assert not paths.managed_layout
    assert paths.managed_root == tmp_path
    assert paths.models_root == tmp_path / "models" / "grasp"
    assert paths.logs_root == tmp_path / "logs" / "grasp"
    assert paths.tb_root == tmp_path / "logs" / "tb" / "grasp"


def test_resolve_training_paths_uses_project_runs_when_available(tmp_path: Path) -> None:
    project_root = tmp_path / "projects" / "h1"
    code_root = project_root / "current"
    (project_root / "runs").mkdir(parents=True)
    code_root.mkdir(parents=True)

    paths = resolve_training_paths(
        code_root,
        "h1",
        legacy_model_dir="models",
        legacy_log_dir="logs",
        legacy_tb_dir="logs/tb/h1",
    )

    assert paths.managed_layout
    assert paths.managed_root == project_root
    assert paths.models_root == project_root / "runs" / "models" / "h1"
    assert paths.logs_root == project_root / "runs" / "logs" / "h1"
    assert paths.tb_root == project_root / "runs" / "logs" / "tb" / "h1"


def test_resolve_training_paths_handles_release_checkout_layout(tmp_path: Path) -> None:
    project_root = tmp_path / "projects" / "h1"
    release_root = project_root / "releases" / "e3d86bf"
    (project_root / "runs").mkdir(parents=True)
    release_root.mkdir(parents=True)

    paths = resolve_training_paths(
        release_root,
        "grasp",
        legacy_model_dir="models/grasp",
        legacy_log_dir="logs/grasp",
        legacy_tb_dir="logs/tb/grasp",
    )

    assert paths.managed_layout
    assert paths.managed_root == project_root
    assert paths.models_root == project_root / "runs" / "models" / "grasp"
    assert paths.logs_root == project_root / "runs" / "logs" / "grasp"


def test_resolve_training_paths_handles_shared_code_release_layout(tmp_path: Path) -> None:
    managed_root = tmp_path / "mujoco-train-system"
    release_root = managed_root / "code" / "releases" / "2d01c5c"
    (managed_root / "runs").mkdir(parents=True)
    release_root.mkdir(parents=True)
    os.environ["MUJOCO_TRAIN_PROJECT_SLUG"] = "grasp"

    try:
        paths = resolve_training_paths(
            release_root,
            "grasp",
            legacy_model_dir="models/grasp",
            legacy_log_dir="logs/grasp",
            legacy_tb_dir="logs/tb/grasp",
        )
    finally:
        os.environ.pop("MUJOCO_TRAIN_PROJECT_SLUG", None)

    assert paths.managed_layout
    assert paths.managed_root == managed_root
    assert paths.project_slug == "grasp"
    assert paths.models_root == managed_root / "runs" / "grasp" / "models" / "grasp"
    assert paths.logs_root == managed_root / "runs" / "grasp" / "logs" / "grasp"
