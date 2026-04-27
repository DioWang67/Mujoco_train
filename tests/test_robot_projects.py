import json
from pathlib import Path

import pytest

from robot_projects import (
    get_robot_project,
    list_project_slugs,
    load_robot_projects,
    validate_project_slug,
)


def _write_project_config(
    configs_root: Path,
    slug: str,
    train_module: str = "robots.example.train",
) -> None:
    project_dir = configs_root / slug
    project_dir.mkdir(parents=True)
    (project_dir / "project.json").write_text(
        json.dumps(
            {
                "slug": slug,
                "display_name": f"{slug} robot",
                "train_module": train_module,
                "eval_module": f"robots.{slug}.eval",
                "job_name": slug,
            }
        ),
        encoding="utf-8",
    )


def test_load_robot_projects_discovers_project_json_files(tmp_path: Path) -> None:
    configs_root = tmp_path / "configs"
    _write_project_config(configs_root, "h1", train_module="h1_train")
    _write_project_config(configs_root, "quadruped")

    projects = load_robot_projects(configs_root)

    assert sorted(projects) == ["h1", "quadruped"]
    assert projects["h1"].train_module == "h1_train"
    assert projects["quadruped"].eval_module == "robots.quadruped.eval"


def test_get_robot_project_rejects_unknown_slug(tmp_path: Path) -> None:
    configs_root = tmp_path / "configs"
    _write_project_config(configs_root, "h1", train_module="h1_train")

    with pytest.raises(ValueError, match="Unknown project 'missing'"):
        get_robot_project("missing", configs_root)


def test_validate_project_slug_rejects_path_like_values() -> None:
    with pytest.raises(ValueError):
        validate_project_slug("../h1")


def test_list_project_slugs_returns_stable_order(tmp_path: Path) -> None:
    configs_root = tmp_path / "configs"
    _write_project_config(configs_root, "quadruped")
    _write_project_config(configs_root, "h1", train_module="h1_train")

    assert list_project_slugs(configs_root) == ["h1", "quadruped"]
