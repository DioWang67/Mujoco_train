"""Inspect repository organization and disposable local files.

This command is intentionally conservative. It reports the project layout,
tool/script counts, and ignored runtime directories without moving or deleting
anything unless an explicit future cleanup command is added.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_SECTIONS: tuple[tuple[str, str], ...] = (
    ("configs", "Committed training, project, sweep, and gate configuration."),
    ("docs", "Runbooks, architecture notes, and project status documents."),
    ("robot_learning", "Shared project discovery, config, path, and runtime helpers."),
    ("h1_baseline", "H1 walking environment, training, and evaluation code."),
    ("grasp_baseline", "Fixed-base grasp environment, assets, and training code."),
    ("sedon_baseline", "Sedon environment, training, evaluation, and tests."),
    ("tools", "Reusable Python CLIs for eval, diagnostics, release, and maintenance."),
    ("scripts", "Thin operator wrappers for Windows/Linux local and remote workflows."),
    ("tests", "Cross-project unit tests that should stay lightweight."),
)

DISPOSABLE_DIR_NAMES: frozenset[str] = frozenset(
    {
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "models",
        "logs",
        "reports",
    }
)

DISPOSABLE_DIR_PREFIXES: tuple[str, ...] = ("_verify_",)

GENERATED_ARTIFACT_DIRS: tuple[str, ...] = ("artifacts",)


@dataclass(frozen=True)
class DirectorySummary:
    """Summary of one top-level repository directory.

    Args:
        name: Directory name relative to the repository root.
        purpose: Human-readable ownership or cleanup rule.
        exists: Whether the directory exists locally.
        file_count: Number of files under the directory.
    """

    name: str
    purpose: str
    exists: bool
    file_count: int


@dataclass(frozen=True)
class CleanupCandidate:
    """Disposable top-level path that should usually stay out of Git.

    Args:
        path: Repository-relative path.
        reason: Why the path is considered disposable.
        file_count: Number of files under the path.
    """

    path: Path
    reason: str
    file_count: int


def count_files(path: Path) -> int:
    """Return the number of files below ``path``.

    Args:
        path: Directory or file path to count.

    Returns:
        Number of regular files. Missing paths count as zero.
    """

    if not path.exists():
        return 0
    if path.is_file():
        return 1
    return sum(1 for child in path.rglob("*") if child.is_file())


def summarize_directories(repo_root: Path) -> list[DirectorySummary]:
    """Build summaries for the canonical top-level directories.

    Args:
        repo_root: Repository root path.

    Returns:
        Directory summaries in canonical display order.
    """

    summaries: list[DirectorySummary] = []
    for name, purpose in REPO_SECTIONS:
        path = repo_root / name
        summaries.append(
            DirectorySummary(
                name=name,
                purpose=purpose,
                exists=path.is_dir(),
                file_count=count_files(path),
            )
        )
    return summaries


def find_cleanup_candidates(repo_root: Path) -> list[CleanupCandidate]:
    """Find top-level local runtime/cache directories.

    Args:
        repo_root: Repository root path.

    Returns:
        Cleanup candidates sorted by path name.
    """

    candidates: list[CleanupCandidate] = []
    for child in repo_root.iterdir():
        if not child.is_dir():
            continue

        reason = cleanup_reason(child.name)
        if reason is None:
            continue

        candidates.append(
            CleanupCandidate(
                path=child.relative_to(repo_root),
                reason=reason,
                file_count=count_files(child),
            )
        )

    return sorted(candidates, key=lambda candidate: candidate.path.as_posix())


def cleanup_reason(name: str) -> str | None:
    """Return why a top-level directory is disposable, if applicable.

    Args:
        name: Top-level directory name.

    Returns:
        Cleanup reason, or ``None`` when the path should not be treated as
        disposable by this tool.
    """

    if name in DISPOSABLE_DIR_NAMES:
        return "cache or runtime output"
    if name in GENERATED_ARTIFACT_DIRS:
        return "generated artifacts"
    if any(name.startswith(prefix) for prefix in DISPOSABLE_DIR_PREFIXES):
        return "temporary verification output"
    return None


def list_scripts(directory: Path, suffixes: Iterable[str]) -> list[Path]:
    """List script-like files in one directory.

    Args:
        directory: Directory to scan.
        suffixes: Accepted file suffixes.

    Returns:
        Sorted direct child paths. Missing directories return an empty list.
    """

    accepted = set(suffixes)
    if not directory.is_dir():
        return []
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix in accepted)


def format_inventory(repo_root: Path) -> str:
    """Render a concise project organization report.

    Args:
        repo_root: Repository root path.

    Returns:
        Human-readable inventory text.
    """

    directories = summarize_directories(repo_root)
    cleanup_candidates = find_cleanup_candidates(repo_root)
    python_tools = list_scripts(repo_root / "tools", {".py"})
    wrapper_scripts = list_scripts(repo_root / "scripts", {".bat", ".ps1", ".sh"})

    lines = [
        "Project inventory",
        "=================",
        "",
        "Canonical directories:",
    ]
    for directory in directories:
        status = "ok" if directory.exists else "missing"
        lines.append(
            f"- {directory.name:<15} {status:<7} {directory.file_count:>4} files  {directory.purpose}"
        )

    lines.extend(
        [
            "",
            "Tooling summary:",
            f"- Python tools: {len(python_tools)} files under tools/",
            f"- Wrapper scripts: {len(wrapper_scripts)} files under scripts/",
            "",
            "Cleanup candidates:",
        ]
    )

    if cleanup_candidates:
        for candidate in cleanup_candidates:
            lines.append(f"- {candidate.path.as_posix():<35} {candidate.file_count:>4} files  {candidate.reason}")
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "Rules:",
            "- Keep repeatable parameters in configs/.",
            "- Keep reusable Python CLIs in tools/.",
            "- Keep wrapper-only operator commands in scripts/.",
            "- Keep generated outputs under ignored runtime directories.",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        argv: Optional argument list for tests.

    Returns:
        Parsed arguments.
    """

    parser = argparse.ArgumentParser(description="Print repository organization and cleanup inventory.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root to inspect. Defaults to the current working directory.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the project inventory command.

    Args:
        argv: Optional argument list for tests.

    Returns:
        Process exit code.
    """

    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    if not repo_root.is_dir():
        raise SystemExit(f"repo root does not exist or is not a directory: {repo_root}")

    print(format_inventory(repo_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
