import json
from pathlib import Path

import pytest

from robot_learning.projects import (
    get_robot_project,
    list_project_slugs,
    load_robot_projects,
    validate_project_slug,
)


def _write_project_config(
    configs_root: Path,
    slug: str,
    train_module: str = "robots.example.train",
    extra: dict | None = None,
) -> None:
    project_dir = configs_root / slug
    project_dir.mkdir(parents=True)
    payload = {
        "slug": slug,
        "display_name": f"{slug} robot",
        "train_module": train_module,
        "eval_module": f"robots.{slug}.eval",
        "job_name": slug,
    }
    if extra:
        payload.update(extra)
    (project_dir / "project.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def test_load_robot_projects_discovers_project_json_files(tmp_path: Path) -> None:
    configs_root = tmp_path / "configs"
    _write_project_config(configs_root, "h1", train_module="h1_baseline.train")
    _write_project_config(configs_root, "quadruped")

    projects = load_robot_projects(configs_root)

    assert sorted(projects) == ["h1", "quadruped"]
    assert projects["h1"].train_module == "h1_baseline.train"
    assert projects["quadruped"].eval_module == "robots.quadruped.eval"
    assert projects["quadruped"].smoke_args == ("--smoke",)


def test_get_robot_project_rejects_unknown_slug(tmp_path: Path) -> None:
    configs_root = tmp_path / "configs"
    _write_project_config(configs_root, "h1", train_module="h1_baseline.train")

    with pytest.raises(ValueError, match="Unknown project 'missing'"):
        get_robot_project("missing", configs_root)


def test_validate_project_slug_rejects_path_like_values() -> None:
    with pytest.raises(ValueError):
        validate_project_slug("../h1")


def test_list_project_slugs_returns_stable_order(tmp_path: Path) -> None:
    configs_root = tmp_path / "configs"
    _write_project_config(configs_root, "quadruped")
    _write_project_config(configs_root, "h1", train_module="h1_baseline.train")

    assert list_project_slugs(configs_root) == ["h1", "quadruped"]


def test_load_robot_projects_reads_smoke_args_and_private_asset_dir(
    tmp_path: Path,
) -> None:
    configs_root = tmp_path / "configs"
    _write_project_config(
        configs_root,
        "sedon",
        train_module="sedon_baseline.train",
        extra={
            "smoke_args": ["--smoke", "--n-envs", "1"],
            "private_asset_dir": "private_assets/sedon",
        },
    )

    project = load_robot_projects(configs_root)["sedon"]

    assert project.smoke_args == ("--smoke", "--n-envs", "1")
    assert project.private_asset_dir.name == "sedon"


def test_load_robot_projects_rejects_unsafe_private_asset_dir(tmp_path: Path) -> None:
    configs_root = tmp_path / "configs"
    _write_project_config(
        configs_root,
        "badbot",
        extra={"private_asset_dir": "../secret"},
    )

    with pytest.raises(ValueError, match="repo-relative"):
        load_robot_projects(configs_root)
