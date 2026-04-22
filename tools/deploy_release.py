"""Create and optionally deploy a committed source release to the remote host.

This tool keeps the local repository layout unchanged. It builds a clean
archive from a git ref, then targets the generic remote layout:

    /root/mujoco-train-system/projects/<slug>/releases/<commit>

The default mode is safe and local-only: create the archive and print the
commands needed for upload and activation. Pass ``--upload`` to run ``scp`` and
``ssh`` directly.
"""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "artifacts" / "sync"
DEFAULT_REMOTE_ROOT = "/root/mujoco-train-system"
DEFAULT_PROJECT_SLUG = "h1"
VALID_PROJECT_SLUG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


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
    def release_dir(self) -> str:
        """Return the immutable release directory."""
        return f"{self.project_root}/releases/{self.commit}"

    @property
    def current_link(self) -> str:
        """Return the current symlink path."""
        return f"{self.project_root}/current"

    @property
    def runs_dir(self) -> str:
        """Return the project-specific run output root."""
        return f"{self.project_root}/runs"


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


def build_archive(ref: str, archive_path: Path) -> Path:
    """Build a clean git archive for the given ref."""
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            "git",
            "archive",
            "--format=tar.gz",
            f"--output={archive_path}",
            ref,
        ],
    )
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

    build_archive(args.ref, archive_path)
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
