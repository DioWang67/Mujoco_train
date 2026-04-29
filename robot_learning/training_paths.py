"""Resolve training artifact directories for local and deployed layouts."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingPaths:
    """Resolved output directories for one training target."""

    code_root: Path
    managed_root: Path
    project_slug: str
    managed_layout: bool
    models_root: Path
    logs_root: Path
    tb_root: Path
    reports_root: Path


def _detect_managed_layout(code_root: Path) -> tuple[Path, bool] | None:
    """Infer deployed layout root and whether code is shared across projects."""
    if (code_root.parent / "runs").is_dir():
        return code_root.parent.resolve(), False

    # Shared code layout:
    #   <managed_root>/code/current
    if code_root.name == "current" and code_root.parent.name == "code":
        candidate = code_root.parent.parent
        if (candidate / "runs").is_dir():
            return candidate.resolve(), True

    # Shared code layout:
    #   <managed_root>/code/releases/<commit>
    if code_root.parent.name == "releases" and code_root.parent.parent.name == "code":
        candidate = code_root.parent.parent.parent
        if (candidate / "runs").is_dir():
            return candidate.resolve(), True

    # Deployed release layout:
    #   <project_root>/releases/<commit>
    if code_root.parent.name == "releases" and (code_root.parent.parent / "runs").is_dir():
        return code_root.parent.parent.resolve(), False

    return None


def resolve_training_paths(
    code_root: str | Path,
    job_name: str,
    *,
    legacy_model_dir: str,
    legacy_log_dir: str,
    legacy_tb_dir: str,
    legacy_reports_dir: str = "reports",
) -> TrainingPaths:
    """Resolve artifact paths for local repo runs or deployed release runs.

    If ``MUJOCO_TRAIN_LAYOUT_ROOT`` is set, or if the code root matches a known
    deployed layout, artifacts are written into a managed ``runs`` tree.
    Otherwise the local repo's historical directories are preserved.
    """
    code_root_path = Path(code_root).resolve()
    layout_root_override = os.environ.get("MUJOCO_TRAIN_LAYOUT_ROOT")
    project_slug_override = os.environ.get("MUJOCO_TRAIN_PROJECT_SLUG")

    if layout_root_override:
        managed_root = Path(layout_root_override).resolve()
        managed_layout = True
        shared_code_layout = True
    else:
        detected_layout = _detect_managed_layout(code_root_path)
        if detected_layout is not None:
            managed_root, shared_code_layout = detected_layout
            managed_layout = True
        else:
            managed_root = code_root_path
            shared_code_layout = False
            managed_layout = False

    if managed_layout:
        project_slug = project_slug_override or job_name
        runs_root = (
            managed_root / "runs" / project_slug
            if shared_code_layout
            else managed_root / "runs"
        )
        return TrainingPaths(
            code_root=code_root_path,
            managed_root=managed_root,
            project_slug=project_slug,
            managed_layout=True,
            models_root=runs_root / "models" / job_name,
            logs_root=runs_root / "logs" / job_name,
            tb_root=runs_root / "logs" / "tb" / job_name,
            reports_root=runs_root / "reports" / job_name,
        )

    return TrainingPaths(
        code_root=code_root_path,
        managed_root=managed_root,
        project_slug=project_slug_override or job_name,
        managed_layout=False,
        models_root=code_root_path / legacy_model_dir,
        logs_root=code_root_path / legacy_log_dir,
        tb_root=code_root_path / legacy_tb_dir,
        reports_root=code_root_path / legacy_reports_dir,
    )
