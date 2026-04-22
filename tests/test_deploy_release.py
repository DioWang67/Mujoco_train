import pytest

from tools.deploy_release import build_remote_deploy_script, build_remote_layout, validate_project_slug


def test_validate_project_slug_accepts_simple_slug() -> None:
    assert validate_project_slug("quadruped_grasper") == "quadruped_grasper"


def test_validate_project_slug_rejects_path_escape() -> None:
    with pytest.raises(ValueError):
        validate_project_slug("../bad")


def test_build_remote_deploy_script_contains_release_and_current_paths() -> None:
    layout = build_remote_layout(
        remote_root="/root/mujoco-train-system",
        project_slug="h1",
        commit="abc1234",
        archive_name="h1_source_abc1234.tar.gz",
    )

    script = build_remote_deploy_script(layout, activate=True)

    assert "/root/mujoco-train-system/projects/h1/releases/abc1234" in script
    assert "ln -sfn" in script
    assert "/root/mujoco-train-system/projects/h1/current" in script
