"""Build a sanitized debug workspace for AI-assisted inspection.

The workspace is intentionally source-only. It excludes private robot assets,
full XML/MJCF files, models, logs, archives, local env files, and generated
runtime output so an AI agent can inspect code and lightweight configs without
receiving proprietary geometry or complete scene descriptions.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts" / "agent_workspace"

ALLOWED_TOP_LEVEL_DIRS: frozenset[str] = frozenset(
    {
        "configs",
        "docs",
        "grasp_baseline",
        "h1_baseline",
        "robot_learning",
        "scripts",
        "seedon_baseline",
        "tests",
        "tools",
    }
)

ALLOWED_ROOT_FILES: frozenset[str] = frozenset(
    {
        ".gitattributes",
        ".gitignore",
        "README.md",
        "eval.py",
        "pytest.ini",
        "requirements.txt",
        "sweep_train.py",
        "train.py",
    }
)

DENIED_TOP_LEVEL_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "artifacts",
        "deploy_content",
        "logs",
        "models",
        "mujoco_menagerie",
        "private_assets",
        "reports",
    }
)

DENIED_SUFFIXES: frozenset[str] = frozenset(
    {
        ".dae",
        ".mjcf",
        ".mp4",
        ".obj",
        ".png",
        ".stl",
        ".tar",
        ".tgz",
        ".urdf",
        ".xml",
        ".zip",
    }
)

DENIED_FILE_NAMES: frozenset[str] = frozenset(
    {
        ".env",
        ".env.remote",
        "MUJOCO_LOG.TXT",
    }
)


@dataclass(frozen=True)
class WorkspaceBuildResult:
    """Summary of one sanitized workspace build.

    Args:
        workspace_dir: Created workspace directory.
        copied_files: Repository-relative copied file paths.
        skipped_files: Repository-relative skipped file paths.
        manifest_path: Path to the generated manifest file.
    """

    workspace_dir: Path
    copied_files: list[Path]
    skipped_files: list[Path]
    manifest_path: Path


def is_relative_to_path(path: Path, parent: Path) -> bool:
    """Return whether ``path`` is inside ``parent``.

    Args:
        path: Candidate path.
        parent: Expected parent path.

    Returns:
        ``True`` when ``path`` is equal to or below ``parent``.
    """

    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def validate_relative_path(path: Path) -> Path:
    """Validate a repository-relative path from git or filesystem walking.

    Args:
        path: Candidate relative path.

    Returns:
        Normalized relative path.

    Raises:
        ValueError: If the path is absolute or escapes through ``..``.
    """

    if path.is_absolute():
        raise ValueError(f"Expected a relative path, got absolute path: {path}")
    if any(part == ".." for part in path.parts):
        raise ValueError(f"Refusing path traversal outside the repository: {path}")
    return path


def should_include_path(relative_path: Path) -> bool:
    """Return whether a repo-relative path belongs in the AI package.

    Args:
        relative_path: File path relative to the repository root.

    Returns:
        ``True`` for source/config/doc/test files that are safe for the
        sanitized workspace.
    """

    relative_path = validate_relative_path(relative_path)
    if not relative_path.parts:
        return False

    top_level = relative_path.parts[0]
    name = relative_path.name
    suffix = relative_path.suffix.lower()

    if name in DENIED_FILE_NAMES:
        return False
    if top_level in DENIED_TOP_LEVEL_DIRS or top_level.startswith("_verify_"):
        return False
    if suffix in DENIED_SUFFIXES:
        return False
    if any(part == "__pycache__" for part in relative_path.parts):
        return False
    if name.startswith(".env."):
        return False
    if top_level in ALLOWED_TOP_LEVEL_DIRS:
        return True
    return len(relative_path.parts) == 1 and name in ALLOWED_ROOT_FILES


def list_git_candidate_files(repo_root: Path) -> list[Path] | None:
    """Return git tracked/untracked non-ignored files, or ``None`` outside git.

    Args:
        repo_root: Repository root to inspect.

    Returns:
        Relative paths from ``git ls-files``. ``None`` means git metadata is not
        available and the caller should use filesystem walking.
    """

    if not (repo_root / ".git").exists():
        return None
    result = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return [
        relative_path
        for relative_path in (Path(raw_path) for raw_path in result.stdout.split("\0") if raw_path)
        if (repo_root / relative_path).is_file()
    ]


def list_filesystem_candidate_files(repo_root: Path) -> list[Path]:
    """Return candidate files by walking the filesystem.

    Args:
        repo_root: Root directory to inspect.

    Returns:
        Repository-relative file paths.
    """

    paths: list[Path] = []
    for path in repo_root.rglob("*"):
        relative_path = path.relative_to(repo_root)
        if not path.is_file():
            continue
        if any(part in DENIED_TOP_LEVEL_DIRS for part in relative_path.parts):
            continue
        paths.append(relative_path)
    return paths


def iter_candidate_files(repo_root: Path) -> list[Path]:
    """Return candidate source files for workspace packaging.

    Args:
        repo_root: Repository root to inspect.

    Returns:
        Sorted repository-relative file paths.
    """

    files = list_git_candidate_files(repo_root)
    if files is None:
        files = list_filesystem_candidate_files(repo_root)
    return sorted(validate_relative_path(path) for path in files)


def resolve_workspace_dir(output_root: Path, name: str, repo_root: Path) -> Path:
    """Resolve and validate the output workspace directory.

    Args:
        output_root: Base directory for generated workspaces.
        name: Workspace folder name.
        repo_root: Repository root used for safety checks.

    Returns:
        Absolute workspace directory.

    Raises:
        ValueError: If the output escapes the repository or overlaps source.
    """

    if "/" in name or "\\" in name or name in {"", ".", ".."}:
        raise ValueError("Workspace name must be a simple directory name.")

    resolved_output_root = output_root.resolve()
    resolved_repo_root = repo_root.resolve()
    if not is_relative_to_path(resolved_output_root, resolved_repo_root):
        raise ValueError(f"Output root must stay inside the repository: {resolved_output_root}")

    workspace_dir = (resolved_output_root / name).resolve()
    if not is_relative_to_path(workspace_dir, resolved_output_root):
        raise ValueError(f"Workspace path escapes output root: {workspace_dir}")
    if workspace_dir == resolved_repo_root:
        raise ValueError("Workspace directory cannot be the repository root.")
    return workspace_dir


def build_sanitized_workspace(
    *,
    repo_root: Path = REPO_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    name: str | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> WorkspaceBuildResult:
    """Create a sanitized debug workspace.

    Args:
        repo_root: Source repository root.
        output_root: Parent directory for generated workspaces.
        name: Optional workspace name. Defaults to a timestamped name.
        force: Replace an existing workspace with the same name.
        dry_run: Compute the manifest without copying files.

    Returns:
        Build summary with copied/skipped file lists.

    Raises:
        ValueError: If paths are invalid or no safe files are found.
        FileExistsError: If the destination exists and ``force`` is false.
    """

    repo_root = repo_root.resolve()
    if not repo_root.is_dir():
        raise ValueError(f"Repository root does not exist: {repo_root}")

    workspace_name = name or f"debug_{time.strftime('%Y%m%d_%H%M%S')}"
    workspace_dir = resolve_workspace_dir(output_root, workspace_name, repo_root)
    manifest_path = workspace_dir / "SANITIZED_MANIFEST.json"

    candidate_files = iter_candidate_files(repo_root)
    copied_files = [path for path in candidate_files if should_include_path(path)]
    skipped_files = [path for path in candidate_files if not should_include_path(path)]

    if not copied_files:
        raise ValueError("No safe files matched the sanitized workspace rules.")

    if workspace_dir.exists():
        if not force:
            raise FileExistsError(f"Workspace already exists: {workspace_dir}")
        if not is_relative_to_path(workspace_dir, output_root.resolve()):
            raise ValueError(f"Refusing to remove workspace outside output root: {workspace_dir}")
        if not dry_run:
            shutil.rmtree(workspace_dir)

    if not dry_run:
        workspace_dir.mkdir(parents=True, exist_ok=True)
        for relative_path in copied_files:
            source_path = repo_root / relative_path
            destination_path = workspace_dir / relative_path
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination_path)

        manifest = {
            "workspace_dir": str(workspace_dir),
            "repo_root": str(repo_root),
            "rules": {
                "allowed_top_level_dirs": sorted(ALLOWED_TOP_LEVEL_DIRS),
                "denied_top_level_dirs": sorted(DENIED_TOP_LEVEL_DIRS),
                "denied_suffixes": sorted(DENIED_SUFFIXES),
            },
            "copied_files": [path.as_posix() for path in copied_files],
            "skipped_files": [path.as_posix() for path in skipped_files],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    return WorkspaceBuildResult(
        workspace_dir=workspace_dir,
        copied_files=copied_files,
        skipped_files=skipped_files,
        manifest_path=manifest_path,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        argv: Optional argument vector for tests.

    Returns:
        Parsed arguments.
    """

    parser = argparse.ArgumentParser(description="Create a sanitized AI debug workspace.")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--name", default=None, help="Workspace directory name. Defaults to a timestamp.")
    parser.add_argument("--force", action="store_true", help="Replace an existing workspace with the same name.")
    parser.add_argument("--dry-run", action="store_true", help="Print the manifest summary without copying files.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Optional argument vector for tests.

    Returns:
        Process exit code.
    """

    args = parse_args(argv)
    result = build_sanitized_workspace(
        repo_root=args.repo_root,
        output_root=args.output_root,
        name=args.name,
        force=args.force,
        dry_run=args.dry_run,
    )
    print(f"Workspace : {result.workspace_dir}")
    print(f"Manifest  : {result.manifest_path}")
    print(f"Copied    : {len(result.copied_files)} files")
    print(f"Skipped   : {len(result.skipped_files)} files")
    if args.dry_run:
        print("Dry run only; no files were copied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
