import subprocess
from pathlib import Path

import pytest

from tools.agent_workspace import (
    build_sanitized_workspace,
    list_git_candidate_files,
    should_include_path,
    validate_relative_path,
)


def test_should_include_path_blocks_private_assets_and_full_xml() -> None:
    assert should_include_path(Path("private_assets/seedon/robot.stl")) is False
    assert should_include_path(Path("seedon_baseline/assets/scene.xml")) is False
    assert should_include_path(Path("configs/seedon/train.json")) is True
    assert should_include_path(Path("seedon_baseline/env.py")) is True


def test_validate_relative_path_rejects_escape() -> None:
    with pytest.raises(ValueError):
        validate_relative_path(Path("../private_assets/seedon/robot.xml"))


def test_build_sanitized_workspace_copies_only_safe_files(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    output_root = repo_root / "artifacts" / "agent_workspace"
    (repo_root / "seedon_baseline").mkdir(parents=True)
    (repo_root / "seedon_baseline" / "env.py").write_text("print('safe')\n", encoding="utf-8")
    (repo_root / "seedon_baseline" / "scene.xml").write_text("<mujoco />\n", encoding="utf-8")
    (repo_root / "private_assets" / "seedon").mkdir(parents=True)
    (repo_root / "private_assets" / "seedon" / "robot.stl").write_text("mesh", encoding="utf-8")
    (repo_root / "README.md").write_text("# Repo\n", encoding="utf-8")

    result = build_sanitized_workspace(
        repo_root=repo_root,
        output_root=output_root,
        name="probe",
    )

    assert (result.workspace_dir / "seedon_baseline" / "env.py").exists()
    assert (result.workspace_dir / "README.md").exists()
    assert not (result.workspace_dir / "seedon_baseline" / "scene.xml").exists()
    assert not (result.workspace_dir / "private_assets").exists()
    assert (result.workspace_dir / "SANITIZED_MANIFEST.json").exists()


def test_build_sanitized_workspace_requires_force_for_existing_output(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    output_root = repo_root / "artifacts" / "agent_workspace"
    (repo_root / "tools").mkdir(parents=True)
    (repo_root / "tools" / "sample.py").write_text("print('safe')\n", encoding="utf-8")

    build_sanitized_workspace(repo_root=repo_root, output_root=output_root, name="probe")

    with pytest.raises(FileExistsError):
        build_sanitized_workspace(repo_root=repo_root, output_root=output_root, name="probe")

    result = build_sanitized_workspace(
        repo_root=repo_root,
        output_root=output_root,
        name="probe",
        force=True,
    )

    assert (result.workspace_dir / "tools" / "sample.py").exists()


def test_list_git_candidate_files_skips_deleted_tracked_files(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    subprocess_commands = [
        ["git", "init"],
        ["git", "config", "user.email", "test@example.com"],
        ["git", "config", "user.name", "Test User"],
    ]
    for command in subprocess_commands:
        subprocess.run(command, cwd=repo_root, check=True, capture_output=True, text=True)

    tracked_file = repo_root / "README.md"
    tracked_file.write_text("# Repo\n", encoding="utf-8")

    subprocess.run(["git", "add", "README.md"], cwd=repo_root, check=True, capture_output=True, text=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo_root, check=True, capture_output=True, text=True)
    tracked_file.unlink()

    assert list_git_candidate_files(repo_root) == []
