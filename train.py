"""Unified robot training dispatcher.

Project-specific training code is discovered from ``configs/<slug>/project.json``.
Each project train module must expose ``main(argv: list[str] | None = None)``.
"""

from __future__ import annotations

import importlib
import os
import sys

from robot_learning.projects import get_robot_project, load_robot_projects
from robot_learning.train_entrypoint import split_mode_args


def configure_numeric_runtime() -> None:
    """Set conservative BLAS/OpenMP defaults before importing training modules."""
    os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def _run_project_train(module_name: str, argv: list[str]) -> int:
    """Import and run one project training module.

    Args:
        module_name: Python module configured in ``project.json``.
        argv: Arguments forwarded after the top-level project selection.

    Returns:
        Process-style exit code.

    Raises:
        ImportError: If the configured module cannot be imported.
        ValueError: If the module does not expose a callable ``main``.
        TypeError: If the module ``main`` has an incompatible signature.
    """
    module = importlib.import_module(module_name)
    module_main = getattr(module, "main", None)
    if not callable(module_main):
        raise ValueError(f"Training module '{module_name}' must expose main(argv).")
    result = module_main(argv)
    return int(result) if result is not None else 0


def main(argv: list[str] | None = None) -> int:
    """Dispatch to the selected robot training project."""
    configure_numeric_runtime()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if raw_argv in (["--help"], ["-h"]) or "--list-projects" in raw_argv:
        print("Usage: python train.py [--project <slug>] [project args...]")
        print()
        print("Available projects:")
        for slug, project in load_robot_projects().items():
            print(f"  {slug:<12} {project.display_name} ({project.train_module})")
        print()
        print("Examples:")
        print("  python train.py --project h1 --smoke")
        print("  python train.py --project grasp --smoke")
        print("  python train.py --list-projects")
        return 0

    try:
        project_slug, forwarded = split_mode_args(raw_argv)
        project = get_robot_project(project_slug)
        return _run_project_train(project.train_module, forwarded)
    except (ImportError, TypeError, ValueError) as exc:
        print(f"[error] {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
