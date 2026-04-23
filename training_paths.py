"""Resolve training artifact directories for local and deployed layouts."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingPaths:
    """Resolved output directories for one training target."""

    code_root: Path
    project_root: Path
    managed_layout: bool
    models_root: Path
    logs_root: Path
    tb_root: Path
    reports_root: Path


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

    If ``MUJOCO_TRAIN_PROJECT_ROOT`` is set, or if the code root has a sibling
    ``runs/`` directory, artifacts are written into ``project_root/runs``.
    Otherwise the local repo's historical directories are preserved.
    """
    code_root_path = Path(code_root).resolve()
    project_root_override = os.environ.get("MUJOCO_TRAIN_PROJECT_ROOT")

    if project_root_override:
        project_root = Path(project_root_override).resolve()
        managed_layout = True
    elif (code_root_path.parent / "runs").is_dir():
        project_root = code_root_path.parent.resolve()
        managed_layout = True
    else:
        project_root = code_root_path
        managed_layout = False

    if managed_layout:
        runs_root = project_root / "runs"
        return TrainingPaths(
            code_root=code_root_path,
            project_root=project_root,
            managed_layout=True,
            models_root=runs_root / "models" / job_name,
            logs_root=runs_root / "logs" / job_name,
            tb_root=runs_root / "logs" / "tb" / job_name,
            reports_root=runs_root / "reports" / job_name,
        )

    return TrainingPaths(
        code_root=code_root_path,
        project_root=project_root,
        managed_layout=False,
        models_root=code_root_path / legacy_model_dir,
        logs_root=code_root_path / legacy_log_dir,
        tb_root=code_root_path / legacy_tb_dir,
        reports_root=code_root_path / legacy_reports_dir,
    )
