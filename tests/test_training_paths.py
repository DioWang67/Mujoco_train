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
    assert paths.project_root == project_root
    assert paths.models_root == project_root / "runs" / "models" / "h1"
    assert paths.logs_root == project_root / "runs" / "logs" / "h1"
    assert paths.tb_root == project_root / "runs" / "logs" / "tb" / "h1"
