"""Discover robot training projects from ``configs/<slug>/project.json`` files."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIGS_ROOT = REPO_ROOT / "configs"
DEFAULT_PROJECT_SLUG = "h1"
_PROJECT_SLUG_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class RobotProject:
    """Metadata needed to route common tooling for one robot project.

    Args:
        slug: Stable CLI/project identifier, such as ``h1`` or ``grasp``.
        display_name: Human-readable project name.
        train_module: Python module exposing ``main(argv)``.
        eval_module: Optional Python module for project-specific evaluation.
        job_name: Artifact job name used under models/logs/tensorboard paths.
        config_dir: Directory containing this project's config files.
    """

    slug: str
    display_name: str
    train_module: str
    eval_module: str | None
    job_name: str
    config_dir: Path


def validate_project_slug(slug: str) -> str:
    """Normalize and validate a project slug.

    Args:
        slug: User or config-provided project identifier.

    Returns:
        Lowercase normalized slug.

    Raises:
        ValueError: If the slug is empty or unsafe for config/path use.
    """
    normalized = slug.strip().lower()
    if not _PROJECT_SLUG_PATTERN.fullmatch(normalized):
        raise ValueError(
            "Project slug must start with a lowercase letter and contain only "
            "lowercase letters, numbers, and underscores."
        )
    return normalized


def _read_project_json(path: Path) -> dict[str, Any]:
    """Read and validate the raw project metadata JSON object."""
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Project config must be a JSON object: {path}")
    return data


def _load_project(path: Path) -> RobotProject:
    """Load one project metadata file."""
    data = _read_project_json(path)
    slug = validate_project_slug(str(data.get("slug", path.parent.name)))
    if slug != path.parent.name:
        raise ValueError(
            f"Project slug '{slug}' must match config directory '{path.parent.name}'."
        )

    train_module = str(data.get("train_module", "")).strip()
    if not train_module:
        raise ValueError(f"Project config missing train_module: {path}")

    eval_module_value = data.get("eval_module")
    eval_module = str(eval_module_value).strip() if eval_module_value else None
    display_name = str(data.get("display_name", slug)).strip() or slug
    job_name = validate_project_slug(str(data.get("job_name", slug)))

    return RobotProject(
        slug=slug,
        display_name=display_name,
        train_module=train_module,
        eval_module=eval_module,
        job_name=job_name,
        config_dir=path.parent,
    )


def load_robot_projects(configs_root: Path = DEFAULT_CONFIGS_ROOT) -> dict[str, RobotProject]:
    """Discover robot projects under ``configs_root``.

    Args:
        configs_root: Directory containing ``<slug>/project.json`` files.

    Returns:
        Mapping from project slug to metadata.

    Raises:
        ValueError: If no projects are found or duplicate slugs are configured.
    """
    projects: dict[str, RobotProject] = {}
    for path in sorted(configs_root.glob("*/project.json")):
        project = _load_project(path)
        if project.slug in projects:
            raise ValueError(f"Duplicate project slug: {project.slug}")
        projects[project.slug] = project

    if not projects:
        raise ValueError(f"No robot projects found under: {configs_root}")
    return projects


def list_project_slugs(configs_root: Path = DEFAULT_CONFIGS_ROOT) -> list[str]:
    """Return available project slugs in stable order."""
    return sorted(load_robot_projects(configs_root))


def get_robot_project(
    slug: str,
    configs_root: Path = DEFAULT_CONFIGS_ROOT,
) -> RobotProject:
    """Return metadata for one project slug.

    Raises:
        ValueError: If the slug is unknown.
    """
    normalized = validate_project_slug(slug)
    projects = load_robot_projects(configs_root)
    try:
        return projects[normalized]
    except KeyError as exc:
        available = ", ".join(sorted(projects))
        raise ValueError(
            f"Unknown project '{normalized}'. Available projects: {available}."
        ) from exc
