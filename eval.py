"""Unified robot evaluation dispatcher.

Project-specific evaluation code is discovered from ``configs/<slug>/project.json``.
Each project eval module must expose ``main(argv: list[str] | None = None)``.
"""

from __future__ import annotations

import importlib
import sys

from robot_learning.projects import get_robot_project, load_robot_projects
from robot_learning.train_entrypoint import split_mode_args
from train import configure_numeric_runtime


def _run_project_eval(module_name: str, argv: list[str]) -> int:
    """Import and run one project evaluation module.

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
        raise ValueError(f"Evaluation module '{module_name}' must expose main(argv).")
    result = module_main(argv)
    return int(result) if result is not None else 0


def main(argv: list[str] | None = None) -> int:
    """Dispatch to the selected robot evaluation project."""
    configure_numeric_runtime()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if raw_argv in (["--help"], ["-h"]) or "--list-projects" in raw_argv:
        print("Usage: python eval.py [--project <slug>] [project args...]")
        print()
        print("Available projects:")
        for slug, project in load_robot_projects().items():
            eval_module = project.eval_module or "(none)"
            print(f"  {slug:<12} {project.display_name} ({eval_module})")
        print()
        print("Examples:")
        print("  python eval.py --project sedon --episodes 1 --render")
        print("  python eval.py --project sedon --episodes 1 --record")
        print("  python eval.py --project h1 --record")
        print("  python eval.py --list-projects")
        return 0

    try:
        project_slug, forwarded = split_mode_args(raw_argv)
        project = get_robot_project(project_slug)
        if not project.eval_module:
            raise ValueError(f"Project '{project_slug}' does not define eval_module.")
        return _run_project_eval(project.eval_module, forwarded)
    except (ImportError, TypeError, ValueError) as exc:
        print(f"[error] {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
