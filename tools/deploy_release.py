"""Create and optionally deploy a committed source release to the remote host.

This tool keeps the local repository layout unchanged. It builds a clean
archive from a git ref, then targets the generic remote layout:

    /root/anaconda3/mujoco-train-system/code/releases/<commit>

The default mode is safe and local-only: create the archive and print the
commands needed for upload and activation. Pass ``--upload`` to run ``scp`` and
``ssh`` directly.
"""

from __future__ import annotations

import argparse
import io
import re
import shlex
import shutil
import subprocess
import tarfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from shutil import copy2, copytree

from robot_learning.projects import get_robot_project


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "artifacts" / "sync"
DEFAULT_TEMP_DIR = REPO_ROOT / "artifacts" / "tmp"
DEFAULT_DEPLOY_CONTENT_DIR = REPO_ROOT / "deploy_content"
DEFAULT_REMOTE_ROOT = "/root/anaconda3/mujoco-train-system"
DEFAULT_PROJECT_SLUG = "h1"
VALID_PROJECT_SLUG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
EXTRA_RELEASE_PATHS = ("mujoco_menagerie",)


@dataclass(frozen=True)
class RemoteLayout:
    """Resolved remote paths for one release deployment."""

    remote_root: str
    project_slug: str
    commit: str
    archive_name: str

    @property
    def incoming_dir(self) -> str:
        """Return the remote upload staging directory."""
        return f"{self.remote_root}/shared/incoming"

    @property
    def incoming_archive(self) -> str:
        """Return the full remote path of the uploaded archive."""
        return f"{self.incoming_dir}/{self.archive_name}"

    @property
    def project_root(self) -> str:
        """Return the per-project root directory."""
        return f"{self.remote_root}/projects/{self.project_slug}"

    @property
    def code_root(self) -> str:
        """Return the shared code root directory."""
        return f"{self.remote_root}/code"

    @property
    def release_dir(self) -> str:
        """Return the immutable release directory."""
        return f"{self.code_root}/releases/{self.commit}"

    @property
    def current_link(self) -> str:
        """Return the current symlink path."""
        return f"{self.code_root}/current"

    @property
    def runs_dir(self) -> str:
        """Return the project-specific run output root."""
        return f"{self.remote_root}/runs/{self.project_slug}"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build and optionally upload a release archive for remote training.",
    )
    parser.add_argument(
        "--project-slug",
        default=DEFAULT_PROJECT_SLUG,
        help="Remote project slug under projects/<slug>.",
    )
    parser.add_argument(
        "--remote-root",
        default=DEFAULT_REMOTE_ROOT,
        help="Remote system root directory.",
    )
    parser.add_argument(
        "--ref",
        default="HEAD",
        help="Git ref to archive. Defaults to HEAD.",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=None,
        help="Optional explicit local output archive path.",
    )
    parser.add_argument(
        "--remote-host",
        default=None,
        help="SSH host for upload, for example root@10.6.243.55.",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload and activate the release using scp/ssh.",
    )
    parser.add_argument(
        "--skip-activate",
        action="store_true",
        help="Upload/extract only. Do not update the current symlink.",
    )
    parser.add_argument(
        "--include-private-assets",
        action="store_true",
        help=(
            "Include the project private_asset_dir from project.json. "
            "Use only when proprietary assets are approved for the target host."
        ),
    )
    return parser.parse_args()


def validate_project_slug(project_slug: str) -> str:
    """Validate a project slug for safe local/remote path composition."""
    if not VALID_PROJECT_SLUG.fullmatch(project_slug):
        raise ValueError(
            "Invalid project slug. Use only letters, numbers, hyphen, or underscore.",
        )
    return project_slug


def run_command(command: list[str], *, capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    """Run a subprocess command and return its completed result."""
    return subprocess.run(
        command,
        check=True,
        text=True,
        capture_output=capture_output,
        cwd=REPO_ROOT,
    )


def resolve_commit(ref: str) -> str:
    """Resolve a git ref to its short commit hash."""
    result = run_command(["git", "rev-parse", "--short", ref], capture_output=True)
    return result.stdout.strip()


def has_worktree_changes() -> bool:
    """Return whether the repository has tracked or untracked changes."""
    result = run_command(
        ["git", "status", "--porcelain"],
        capture_output=True,
    )
    return bool(result.stdout.strip())


def build_worktree_release_id(ref: str = "HEAD") -> str:
    """Return a deterministic-looking release id for the current worktree.

    The prefix remains tied to the current commit, while the timestamp makes
    dirty working-tree releases repeatable in layout without overwriting a
    previous deployment.
    """
    commit = resolve_commit(ref)
    suffix = time.strftime("%Y%m%d%H%M%S")
    if has_worktree_changes():
        return f"{commit}-worktree-{suffix}"
    return commit


def iter_existing_extra_release_paths() -> list[Path]:
    """Return non-git release assets that should be bundled when present."""
    return [REPO_ROOT / relative for relative in EXTRA_RELEASE_PATHS if (REPO_ROOT / relative).exists()]


def iter_private_release_paths(project_slug: str, *, include_private_assets: bool) -> list[Path]:
    """Return explicit opt-in private assets for one project.

    Args:
        project_slug: Configured project slug.
        include_private_assets: Whether private assets should be included.

    Returns:
        Existing private asset paths to copy into the release staging tree.

    Raises:
        ValueError: If private assets were requested but the project has no
            ``private_asset_dir`` or that directory does not exist.
    """
    if not include_private_assets:
        return []

    project = get_robot_project(project_slug)
    if project.private_asset_dir is None:
        raise ValueError(
            f"Project '{project_slug}' does not define private_asset_dir in project.json."
        )
    if not project.private_asset_dir.exists():
        raise ValueError(f"Private asset directory not found: {project.private_asset_dir}")
    return [project.private_asset_dir]


def _copy_path(source: Path, destination: Path) -> None:
    """Copy one file or directory into the staging tree."""
    if source.is_dir():
        copytree(source, destination, dirs_exist_ok=True)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        copy2(source, destination)


def iter_deploy_overlay_files(deploy_content_dir: Path = DEFAULT_DEPLOY_CONTENT_DIR) -> list[Path]:
    """Return files from the local deployment overlay directory."""
    if not deploy_content_dir.exists():
        return []
    if not deploy_content_dir.is_dir():
        raise ValueError(f"Deploy content path is not a directory: {deploy_content_dir}")
    return [
        path
        for path in deploy_content_dir.rglob("*")
        if path.is_file() and path.name not in {".gitkeep", "README.md"}
    ]


def copy_deploy_overlay(staging_root: Path, deploy_content_dir: Path = DEFAULT_DEPLOY_CONTENT_DIR) -> int:
    """Copy deployment overlay files into the staging tree.

    Files are copied relative to ``deploy_content_dir`` so the directory acts as
    an overlay on top of the release root.
    """
    copied = 0
    for source_path in iter_deploy_overlay_files(deploy_content_dir):
        destination = staging_root / source_path.relative_to(deploy_content_dir)
        _copy_path(source_path, destination)
        copied += 1
    return copied


def make_staging_root(prefix: str) -> Path:
    """Create a writable staging root under the ignored artifact directory."""
    DEFAULT_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    staging_root = DEFAULT_TEMP_DIR / f"{prefix}_{uuid.uuid4().hex}" / "staging"
    staging_root.mkdir(parents=True, exist_ok=False)
    return staging_root


def remove_staging_root(staging_root: Path) -> None:
    """Remove a staging tree created by ``make_staging_root``."""
    temp_parent = DEFAULT_TEMP_DIR.resolve()
    cleanup_root = staging_root.parent.resolve()
    if temp_parent not in cleanup_root.parents and cleanup_root != temp_parent:
        raise ValueError(f"Refusing to clean staging path outside artifact temp: {cleanup_root}")
    shutil.rmtree(cleanup_root, ignore_errors=True)


def build_archive(
    ref: str,
    archive_path: Path,
    *,
    project_slug: str,
    include_private_assets: bool = False,
    include_extra_release_paths: bool = True,
) -> Path:
    """Build a clean release archive and include required local assets."""
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    staging_root = make_staging_root("deploy_release")
    try:
        archive_bytes = subprocess.check_output(
            ["git", "archive", "--format=tar", ref],
            cwd=REPO_ROOT,
        )
        with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:") as tar:
            tar.extractall(staging_root)

        if include_extra_release_paths:
            for extra_path in iter_existing_extra_release_paths():
                _copy_path(extra_path, staging_root / extra_path.name)
        for private_path in iter_private_release_paths(
            project_slug,
            include_private_assets=include_private_assets,
        ):
            destination = staging_root / private_path.relative_to(REPO_ROOT)
            _copy_path(private_path, destination)
        copy_deploy_overlay(staging_root)

        with tarfile.open(archive_path, mode="w:gz") as tar:
            tar.add(staging_root, arcname=".")
    finally:
        remove_staging_root(staging_root)
    return archive_path


def iter_worktree_release_files() -> list[Path]:
    """Return tracked and untracked non-ignored files for a worktree release."""
    result = run_command(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        capture_output=True,
    )
    paths: list[Path] = []
    for raw_path in result.stdout.split("\0"):
        if not raw_path:
            continue
        path = REPO_ROOT / raw_path
        if path.is_file():
            paths.append(path)
    return paths


def build_worktree_archive(
    archive_path: Path,
    *,
    project_slug: str,
    include_private_assets: bool = False,
    include_extra_release_paths: bool = True,
) -> Path:
    """Build a release archive from the current tracked/untracked worktree.

    Ignored outputs such as models, logs, artifacts, caches, and local env files
    stay out of the archive through ``git ls-files --exclude-standard``.
    """
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    staging_root = make_staging_root("deploy_release_worktree")
    try:
        for source_path in iter_worktree_release_files():
            destination = staging_root / source_path.relative_to(REPO_ROOT)
            _copy_path(source_path, destination)

        if include_extra_release_paths:
            for extra_path in iter_existing_extra_release_paths():
                _copy_path(extra_path, staging_root / extra_path.name)
        for private_path in iter_private_release_paths(
            project_slug,
            include_private_assets=include_private_assets,
        ):
            destination = staging_root / private_path.relative_to(REPO_ROOT)
            _copy_path(private_path, destination)
        copy_deploy_overlay(staging_root)

        with tarfile.open(archive_path, mode="w:gz") as tar:
            tar.add(staging_root, arcname=".")
    finally:
        remove_staging_root(staging_root)
    return archive_path


def build_remote_layout(
    *,
    remote_root: str,
    project_slug: str,
    commit: str,
    archive_name: str,
) -> RemoteLayout:
    """Create the remote layout description for one release."""
    return RemoteLayout(
        remote_root=remote_root.rstrip("/"),
        project_slug=validate_project_slug(project_slug),
        commit=commit,
        archive_name=archive_name,
    )


def build_remote_deploy_script(layout: RemoteLayout, *, activate: bool) -> str:
    """Return the remote shell script used for extraction and activation."""
    release_dir = shlex.quote(layout.release_dir)
    incoming_archive = shlex.quote(layout.incoming_archive)
    current_link = shlex.quote(layout.current_link)
    runs_dir = shlex.quote(layout.runs_dir)
    incoming_dir = shlex.quote(layout.incoming_dir)

    lines = [
        "set -e",
        f"mkdir -p {incoming_dir}",
        f"mkdir -p {layout.code_root}/releases",
        f"mkdir -p {runs_dir}/models {runs_dir}/logs {runs_dir}/reports",
        (
            f'if [ -d {release_dir} ] && [ "$(ls -A {release_dir} 2>/dev/null)" ]; then '
            f'echo "Release already exists: {layout.release_dir}" >&2; exit 1; fi'
        ),
        f"mkdir -p {release_dir}",
        f"tar xzf {incoming_archive} -C {release_dir}",
    ]
    if activate:
        lines.append(f"ln -sfn {release_dir} {current_link}")
    return " && ".join(lines)


def build_remote_prepare_script(layout: RemoteLayout) -> str:
    """Return the remote shell script that prepares upload staging paths."""
    incoming_dir = shlex.quote(layout.incoming_dir)
    return " && ".join(
        [
            "set -e",
            f"mkdir -p {incoming_dir}",
        ],
    )


def build_scp_command(local_archive: Path, layout: RemoteLayout, remote_host: str) -> str:
    """Return a human-readable scp command for upload."""
    target = f"{remote_host}:{layout.incoming_archive}"
    return f"scp {shlex.quote(str(local_archive))} {shlex.quote(target)}"


def upload_release(
    *,
    local_archive: Path,
    layout: RemoteLayout,
    remote_host: str,
    activate: bool,
) -> None:
    """Upload the release archive and run the remote extraction script."""
    run_command(["ssh", remote_host, build_remote_prepare_script(layout)])
    run_command(
        [
            "scp",
            str(local_archive),
            f"{remote_host}:{layout.incoming_archive}",
        ],
    )
    remote_script = build_remote_deploy_script(layout, activate=activate)
    run_command(["ssh", remote_host, remote_script])


def main() -> int:
    """Entry point for the deploy helper."""
    args = parse_args()
    project_slug = validate_project_slug(args.project_slug)
    commit = resolve_commit(args.ref)

    archive_path = args.archive
    if archive_path is None:
        archive_path = DEFAULT_ARTIFACT_DIR / f"{project_slug}_source_{commit}.tar.gz"
    archive_path = archive_path.resolve()

    build_archive(
        args.ref,
        archive_path,
        project_slug=project_slug,
        include_private_assets=args.include_private_assets,
    )
    layout = build_remote_layout(
        remote_root=args.remote_root,
        project_slug=project_slug,
        commit=commit,
        archive_name=archive_path.name,
    )

    print(f"Built archive: {archive_path}")
    print(f"Commit: {commit}")
    print(f"Remote release: {layout.release_dir}")
    print()
    print("Remote commands:")
    if args.remote_host:
        print(
            "ssh "
            + args.remote_host
            + " "
            + shlex.quote(build_remote_prepare_script(layout)),
        )
        print(build_scp_command(archive_path, layout, args.remote_host))
    else:
        print("ssh <user@host> " + shlex.quote(build_remote_prepare_script(layout)))
        print("scp <archive> <user@host>:" + layout.incoming_archive)
    print(
        "ssh "
        + (args.remote_host or "<user@host>")
        + " "
        + shlex.quote(build_remote_deploy_script(layout, activate=not args.skip_activate)),
    )

    if args.upload:
        if not args.remote_host:
            raise ValueError("--upload requires --remote-host.")
        upload_release(
            local_archive=archive_path,
            layout=layout,
            remote_host=args.remote_host,
            activate=not args.skip_activate,
        )
        print("Upload complete.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
