"""Deploy and verify a remote release using environment-based credentials.

This tool keeps deployment content separate from transport/authentication:
``tools.deploy_release`` builds the archive, while this module reads local
environment settings, uploads the archive, activates ``code/current``, and runs
optional remote checks.
"""

from __future__ import annotations

import argparse
import getpass
import os
import shutil
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

from tools.deploy_release import (
    DEFAULT_DEPLOY_CONTENT_DIR,
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_REMOTE_ROOT,
    REPO_ROOT,
    RemoteLayout,
    build_archive,
    build_remote_layout,
    build_worktree_archive,
    build_worktree_release_id,
    iter_deploy_overlay_files,
    resolve_commit,
    validate_project_slug,
)


DEFAULT_ENV_FILE = REPO_ROOT / ".env.remote"
DEFAULT_REMOTE_HOST = "root@10.6.243.55"
DEFAULT_REMOTE_PYTHON = "/root/anaconda3/bin/python"
VALID_BACKENDS = ("auto", "openssh", "native", "askpass", "sshpass", "putty", "paramiko")
WINDOWS_OPENSSH = Path("C:/Windows/System32/OpenSSH/ssh.exe")
WINDOWS_OPENSCP = Path("C:/Windows/System32/OpenSSH/scp.exe")
ASKPASS_HELPER = REPO_ROOT / "artifacts" / "tmp" / "ssh_askpass.cmd"


@dataclass(frozen=True)
class DeployConfig:
    """Resolved deployment settings.

    Args:
        project_slug: Project to deploy and verify.
        verify_project: Project to smoke-test. Use ``none`` to skip smoke.
        ref: Git ref used for git-ref source mode.
        source_mode: ``working-tree`` or ``git-ref``.
        remote_host: SSH target, for example ``root@10.6.243.55``.
        remote_root: Remote layout root.
        remote_python: Python executable on the remote host.
        remote_password: Optional SSH password read from environment only.
        backend: Requested SSH backend. ``native`` uses interactive OpenSSH.
        include_private_assets: Whether to package project private assets.
        include_extra_assets: Whether to package large shared extra assets.
        clean_release: Whether to remove an existing release path first.
        smoke_args: Optional explicit smoke args.
        connect_timeout_seconds: SSH connection timeout.
        strict_host_key_checking: OpenSSH host key mode.
    """

    project_slug: str
    verify_project: str
    ref: str
    source_mode: str
    remote_host: str
    remote_root: str
    remote_python: str
    remote_password: str | None
    backend: str
    include_private_assets: bool
    include_extra_assets: bool
    clean_release: bool
    smoke_args: str | None
    connect_timeout_seconds: int
    strict_host_key_checking: str


class NativeRunner:
    """Run native SSH/SCP commands while masking secrets in logs and errors."""

    def __init__(self, *, dry_run: bool, secrets: list[str]) -> None:
        self.dry_run = dry_run
        self.secrets = [secret for secret in secrets if secret]

    def _mask(self, command: list[str]) -> str:
        display_parts: list[str] = []
        for part in command:
            masked = part
            for secret in self.secrets:
                masked = masked.replace(secret, "********")
            display_parts.append(masked)
        return " ".join(shlex.quote(part) for part in display_parts)

    def run(self, command: list[str], *, cwd: Path = REPO_ROOT, env: dict[str, str] | None = None) -> None:
        """Run one command and raise a masked error if it fails."""
        print(f"$ {self._mask(command)}")
        if self.dry_run:
            return
        try:
            subprocess.run(command, cwd=cwd, check=True, env=env)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Command failed with exit code {exc.returncode}: {self._mask(command)}"
            ) from exc


def parse_env_file(path: Path) -> dict[str, str]:
    """Parse a simple KEY=VALUE env file.

    Args:
        path: File path to read.

    Returns:
        Parsed key/value pairs. Missing files return an empty mapping.

    Raises:
        ValueError: If a non-empty, non-comment line is not ``KEY=VALUE``.
    """
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Invalid env line {path}:{line_number}: expected KEY=VALUE")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            raise ValueError(f"Invalid env line {path}:{line_number}: empty key")
        values[key] = value
    return values


def env_value(values: dict[str, str], key: str, default: str) -> str:
    """Return an OS environment value, then env-file value, then default."""
    return os.environ.get(key) or values.get(key) or default


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build, upload, activate, and verify a remote release using env credentials.",
    )
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--project-slug", default=None)
    parser.add_argument("--verify-project", default=None)
    parser.add_argument("--ref", default=None)
    parser.add_argument(
        "--source-mode",
        choices=("working-tree", "git-ref"),
        default=None,
        help="Use working-tree for fast uncommitted iteration, or git-ref for committed releases.",
    )
    parser.add_argument("--remote-host", default=None)
    parser.add_argument("--remote-root", default=None)
    parser.add_argument("--remote-python", default=None)
    parser.add_argument("--backend", choices=VALID_BACKENDS, default=None)
    parser.add_argument("--include-private-assets", action="store_true")
    parser.add_argument("--no-include-private-assets", action="store_true")
    parser.add_argument("--include-extra-assets", action="store_true")
    parser.add_argument("--clean-release", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--prompt-password", action="store_true")
    parser.add_argument("--smoke-args", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace, env_values: dict[str, str]) -> DeployConfig:
    """Resolve CLI and environment values into one deployment config."""
    project_slug = validate_project_slug(
        args.project_slug or env_value(env_values, "REMOTE_PROJECT_SLUG", "seedon")
    )
    verify_project = args.verify_project or env_value(env_values, "REMOTE_VERIFY_PROJECT", project_slug)
    if args.skip_smoke:
        verify_project = "none"

    backend = args.backend or env_value(env_values, "REMOTE_SSH_BACKEND", "auto")
    if backend not in VALID_BACKENDS:
        raise ValueError(f"Invalid REMOTE_SSH_BACKEND: {backend}")

    include_private_assets = (
        args.include_private_assets
        or env_value(env_values, "REMOTE_INCLUDE_PRIVATE_ASSETS", "0") in {"1", "true", "True", "yes"}
    )
    if args.no_include_private_assets:
        include_private_assets = False

    remote_password = os.environ.get("REMOTE_PASSWORD") or env_values.get("REMOTE_PASSWORD")
    if args.prompt_password:
        remote_password = getpass.getpass("Remote SSH password: ")

    config = DeployConfig(
        project_slug=project_slug,
        verify_project=verify_project,
        ref=args.ref or env_value(env_values, "REMOTE_DEPLOY_REF", "HEAD"),
        source_mode=args.source_mode or env_value(env_values, "REMOTE_SOURCE_MODE", "working-tree"),
        remote_host=args.remote_host or env_value(env_values, "REMOTE_HOST", DEFAULT_REMOTE_HOST),
        remote_root=(args.remote_root or env_value(env_values, "REMOTE_ROOT", DEFAULT_REMOTE_ROOT)).rstrip("/"),
        remote_python=args.remote_python
        or env_value(env_values, "REMOTE_PYTHON", DEFAULT_REMOTE_PYTHON),
        remote_password=remote_password,
        backend=backend,
        include_private_assets=include_private_assets,
        include_extra_assets=args.include_extra_assets
        or env_value(env_values, "REMOTE_INCLUDE_EXTRA_ASSETS", "0") in {"1", "true", "True", "yes"},
        clean_release=args.clean_release
        or env_value(env_values, "REMOTE_CLEAN_RELEASE", "0") in {"1", "true", "True", "yes"},
        smoke_args=args.smoke_args or env_values.get("REMOTE_SMOKE_ARGS"),
        connect_timeout_seconds=int(env_value(env_values, "REMOTE_CONNECT_TIMEOUT", "10")),
        strict_host_key_checking=env_value(env_values, "REMOTE_STRICT_HOST_KEY_CHECKING", "accept-new"),
    )
    if config.backend == "openssh" and config.remote_password:
        raise ValueError(
            "REMOTE_PASSWORD is set, but --backend openssh cannot use passwords non-interactively. "
            "Use --backend native for interactive password prompts, --backend paramiko, "
            "install PuTTY/sshpass, or clear REMOTE_PASSWORD and use SSH keys."
        )
    return config


def select_backend(config: DeployConfig) -> str:
    """Select the SSH backend for the configured credential mode."""
    if config.backend != "auto":
        return config.backend
    if not config.remote_password:
        return "openssh"
    if WINDOWS_OPENSSH.exists() and WINDOWS_OPENSCP.exists():
        return "askpass"
    if _paramiko_available():
        return "paramiko"
    if shutil.which("sshpass"):
        return "sshpass"
    if shutil.which("plink") and shutil.which("pscp"):
        return "putty"
    raise RuntimeError(
        "REMOTE_PASSWORD is set, but no password-capable backend was found. "
        "Use REMOTE_SSH_BACKEND=native for interactive OpenSSH prompts, "
        "install Paramiko/PuTTY on Windows or sshpass on Linux/macOS, or use SSH keys."
    )


def _paramiko_available() -> bool:
    """Return whether Paramiko can be imported in the current Python env."""
    try:
        import paramiko  # noqa: F401
    except ImportError:
        return False
    return True


def parse_remote_host(remote_host: str) -> tuple[str | None, str, int]:
    """Parse ``[user@]host[:port]`` for Paramiko connections."""
    user: str | None = None
    host_port = remote_host
    if "@" in remote_host:
        user, host_port = remote_host.split("@", 1)
    host = host_port
    port = 22
    if ":" in host_port and host_port.count(":") == 1:
        host, raw_port = host_port.rsplit(":", 1)
        if raw_port:
            port = int(raw_port)
    if not host:
        raise ValueError(f"Invalid remote host: {remote_host}")
    return user, host, port


class ParamikoSession:
    """Minimal SSH/SFTP session for password-based deployment."""

    def __init__(self, config: DeployConfig, *, dry_run: bool) -> None:
        self.config = config
        self.dry_run = dry_run
        self._client = None

    def __enter__(self) -> "ParamikoSession":
        if self.dry_run:
            return self
        if not self.config.remote_password:
            raise ValueError("Paramiko backend requires REMOTE_PASSWORD.")
        import paramiko

        username, host, port = parse_remote_host(self.config.remote_host)
        client = self._new_client(paramiko)
        try:
            client.connect(
                hostname=host,
                port=port,
                username=username,
                password=self.config.remote_password,
                timeout=self.config.connect_timeout_seconds,
                banner_timeout=self.config.connect_timeout_seconds,
                auth_timeout=self.config.connect_timeout_seconds,
                look_for_keys=False,
                allow_agent=False,
            )
        except paramiko.AuthenticationException as exc:
            client.close()
            try:
                client = self._connect_keyboard_interactive(paramiko, username, host, port)
            except paramiko.AuthenticationException as keyboard_exc:
                raise RuntimeError(
                    f"SSH authentication failed for {self.config.remote_host}. "
                    "Check REMOTE_PASSWORD, REMOTE_HOST user, or remote password-login policy."
                ) from keyboard_exc
            except Exception as keyboard_exc:
                raise RuntimeError(
                    f"SSH keyboard-interactive login failed for {self.config.remote_host}: {keyboard_exc}"
                ) from keyboard_exc
        self._client = client
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._client is not None:
            self._client.close()

    def _new_client(self, paramiko_module: object):
        """Create a Paramiko SSH client with the configured host-key policy."""
        client = paramiko_module.SSHClient()
        client.load_system_host_keys()
        if self.config.strict_host_key_checking in {"no", "accept-new"}:
            client.set_missing_host_key_policy(paramiko_module.AutoAddPolicy())
        return client

    def _connect_keyboard_interactive(
        self,
        paramiko_module: object,
        username: str | None,
        host: str,
        port: int,
    ):
        """Connect using keyboard-interactive auth for PAM-style SSH servers."""
        if username is None:
            raise ValueError("Keyboard-interactive auth requires a username in REMOTE_HOST.")

        transport = paramiko_module.Transport((host, port))
        transport.banner_timeout = self.config.connect_timeout_seconds
        transport.auth_timeout = self.config.connect_timeout_seconds
        transport.start_client(timeout=self.config.connect_timeout_seconds)

        def handler(title: str, instructions: str, prompts: list[tuple[str, bool]]) -> list[str]:
            return [self.config.remote_password or "" for _prompt, _echo in prompts]

        transport.auth_interactive(username, handler)
        if not transport.is_authenticated():
            transport.close()
            raise paramiko_module.AuthenticationException("keyboard-interactive authentication failed")

        client = self._new_client(paramiko_module)
        client._transport = transport
        return client

    def run(self, remote_command: str) -> None:
        """Run one remote command."""
        print(f"$ ssh-paramiko {self.config.remote_host} {shlex.quote(remote_command)}")
        if self.dry_run:
            return
        if self._client is None:
            raise RuntimeError("Paramiko session is not connected.")
        stdin, stdout, stderr = self._client.exec_command(remote_command)
        stdin.close()
        stdout_text = stdout.read().decode("utf-8", errors="replace")
        stderr_text = stderr.read().decode("utf-8", errors="replace")
        exit_code = stdout.channel.recv_exit_status()
        if stdout_text:
            print(stdout_text, end="" if stdout_text.endswith("\n") else "\n")
        if stderr_text:
            print(stderr_text, end="" if stderr_text.endswith("\n") else "\n")
        if exit_code != 0:
            raise RuntimeError(f"Remote command failed with exit code {exit_code}: {remote_command}")

    def put(self, local_path: Path, remote_path: str) -> None:
        """Upload one file using SFTP."""
        print(f"$ sftp-paramiko put {shlex.quote(str(local_path))} {self.config.remote_host}:{remote_path}")
        if self.dry_run:
            return
        if self._client is None:
            raise RuntimeError("Paramiko session is not connected.")
        with self._client.open_sftp() as sftp:
            sftp.put(str(local_path), remote_path)


def openssh_options(config: DeployConfig) -> list[str]:
    """Return common OpenSSH options."""
    return [
        "-o",
        f"ConnectTimeout={config.connect_timeout_seconds}",
        "-o",
        f"StrictHostKeyChecking={config.strict_host_key_checking}",
    ]


def password_openssh_options(config: DeployConfig) -> list[str]:
    """Return OpenSSH options for non-interactive password auth."""
    return [
        *openssh_options(config),
        "-o",
        "PreferredAuthentications=password",
        "-o",
        "PubkeyAuthentication=no",
    ]


def ensure_askpass_helper() -> Path:
    """Create the local askpass helper used by Windows OpenSSH."""
    ASKPASS_HELPER.parent.mkdir(parents=True, exist_ok=True)
    ASKPASS_HELPER.write_text(
        "@echo off\r\n"
        "if \"%REMOTE_PASSWORD%\"==\"\" exit /b 1\r\n"
        "echo(%REMOTE_PASSWORD%\r\n",
        encoding="ascii",
    )
    return ASKPASS_HELPER


def build_askpass_env(config: DeployConfig) -> dict[str, str]:
    """Return child environment that lets OpenSSH read REMOTE_PASSWORD."""
    if not config.remote_password:
        raise ValueError("askpass backend requires REMOTE_PASSWORD.")
    helper = ensure_askpass_helper()
    env = os.environ.copy()
    env["REMOTE_PASSWORD"] = config.remote_password
    env["SSH_ASKPASS"] = str(helper)
    env["SSH_ASKPASS_REQUIRE"] = "force"
    env.setdefault("DISPLAY", "localhost:0")
    return env


def build_ssh_command(config: DeployConfig, backend: str, remote_command: str) -> list[str]:
    """Build a backend-specific SSH command."""
    if backend == "openssh":
        return ["ssh", *openssh_options(config), config.remote_host, remote_command]
    if backend == "native":
        ssh_exe = str(WINDOWS_OPENSSH) if WINDOWS_OPENSSH.exists() else "ssh"
        return [ssh_exe, "-tt", *openssh_options(config), config.remote_host, remote_command]
    if backend == "askpass":
        ssh_exe = str(WINDOWS_OPENSSH) if WINDOWS_OPENSSH.exists() else "ssh"
        return [ssh_exe, *password_openssh_options(config), config.remote_host, remote_command]
    if backend == "sshpass":
        if not config.remote_password:
            raise ValueError("sshpass backend requires REMOTE_PASSWORD.")
        return [
            "sshpass",
            "-p",
            config.remote_password,
            "ssh",
            *openssh_options(config),
            config.remote_host,
            remote_command,
        ]
    if backend == "putty":
        if not config.remote_password:
            raise ValueError("putty backend requires REMOTE_PASSWORD.")
        return ["plink", "-batch", "-ssh", "-pw", config.remote_password, config.remote_host, remote_command]
    raise ValueError(f"Unsupported backend: {backend}")


def build_scp_command(
    config: DeployConfig,
    backend: str,
    local_archive: Path,
    remote_archive: str,
) -> list[str]:
    """Build a backend-specific upload command."""
    target = f"{config.remote_host}:{remote_archive}"
    if backend == "openssh":
        return ["scp", *openssh_options(config), str(local_archive), target]
    if backend == "native":
        scp_exe = str(WINDOWS_OPENSCP) if WINDOWS_OPENSCP.exists() else "scp"
        return [scp_exe, *openssh_options(config), str(local_archive), target]
    if backend == "askpass":
        scp_exe = str(WINDOWS_OPENSCP) if WINDOWS_OPENSCP.exists() else "scp"
        return [scp_exe, *password_openssh_options(config), str(local_archive), target]
    if backend == "sshpass":
        if not config.remote_password:
            raise ValueError("sshpass backend requires REMOTE_PASSWORD.")
        return [
            "sshpass",
            "-p",
            config.remote_password,
            "scp",
            *openssh_options(config),
            str(local_archive),
            target,
        ]
    if backend == "putty":
        if not config.remote_password:
            raise ValueError("putty backend requires REMOTE_PASSWORD.")
        return ["pscp", "-batch", "-scp", "-pw", config.remote_password, str(local_archive), target]
    raise ValueError(f"Unsupported backend: {backend}")


def shell_join(commands: list[str]) -> str:
    """Join remote shell commands with fail-fast semantics."""
    return " && ".join(commands)


def quote_remote(value: str) -> str:
    """Quote one value for the remote POSIX shell."""
    return shlex.quote(value)


def build_release_archive(config: DeployConfig) -> tuple[Path, RemoteLayout]:
    """Build the requested release archive and return its remote layout."""
    if config.source_mode == "git-ref":
        release_id = resolve_commit(config.ref)
        archive_path = DEFAULT_ARTIFACT_DIR / f"{config.project_slug}_source_{release_id}.tar.gz"
        build_archive(
            config.ref,
            archive_path,
            project_slug=config.project_slug,
            include_private_assets=config.include_private_assets,
            include_extra_release_paths=config.include_extra_assets,
        )
    elif config.source_mode == "working-tree":
        release_id = build_worktree_release_id(config.ref)
        archive_path = DEFAULT_ARTIFACT_DIR / f"{config.project_slug}_source_{release_id}.tar.gz"
        build_worktree_archive(
            archive_path,
            project_slug=config.project_slug,
            include_private_assets=config.include_private_assets,
            include_extra_release_paths=config.include_extra_assets,
        )
    else:
        raise ValueError(f"Unsupported source mode: {config.source_mode}")

    layout = build_remote_layout(
        remote_root=config.remote_root,
        project_slug=config.project_slug,
        commit=release_id,
        archive_name=archive_path.name,
    )
    return archive_path.resolve(), layout


def build_remote_asset_link_steps(config: DeployConfig, layout: RemoteLayout) -> list[str]:
    """Return remote commands that reuse already-deployed large asset folders."""
    steps: list[str] = []
    releases_root = quote_remote(layout.code_root + "/releases")
    release_dir = quote_remote(layout.release_dir)
    if not config.include_private_assets:
        steps.append(
            "if [ ! -e {release}/private_assets ]; then "
            "asset_src=$(find {releases} -maxdepth 2 -type d -name private_assets "
            "! -path {release}/private_assets 2>/dev/null | head -n 1); "
            'if [ -n "$asset_src" ]; then ln -sfn "$asset_src" {release}/private_assets; fi; '
            "fi".format(release=release_dir, releases=releases_root)
        )
    if not config.include_extra_assets:
        steps.append(
            "if [ ! -e {release}/mujoco_menagerie ]; then "
            "asset_src=$(find {releases} -maxdepth 2 -type d -name mujoco_menagerie "
            "! -path {release}/mujoco_menagerie 2>/dev/null | head -n 1); "
            'if [ -n "$asset_src" ]; then ln -sfn "$asset_src" {release}/mujoco_menagerie; fi; '
            "fi".format(release=release_dir, releases=releases_root)
        )
    return steps


def get_smoke_args(config: DeployConfig) -> str:
    """Return smoke arguments for the verify step."""
    return config.smoke_args or "--smoke"


def build_remote_smoke_script(config: DeployConfig, code_root: str) -> str | None:
    """Return the remote smoke-check script, or ``None`` when disabled."""
    if config.verify_project == "none":
        return None
    smoke_args = get_smoke_args(config)
    return shell_join(
        [
            f"cd {quote_remote(code_root)}",
            "export "
            f"MUJOCO_TRAIN_LAYOUT_ROOT={quote_remote(config.remote_root)} "
            f"MUJOCO_TRAIN_PROJECT_SLUG={quote_remote(config.verify_project)} "
            "MKL_THREADING_LAYER=GNU "
            "OMP_NUM_THREADS=1 "
            "MKL_NUM_THREADS=1 "
            "OPENBLAS_NUM_THREADS=1 "
            "NUMEXPR_NUM_THREADS=1",
            f"{quote_remote(config.remote_python)} train.py --project "
            f"{quote_remote(config.verify_project)} {smoke_args}",
        ]
    )


def run_remote_health_check(config: DeployConfig, *, dry_run: bool) -> None:
    """Check remote connectivity, active release, Python, and optional smoke."""
    backend = select_backend(config)
    current_link = f"{config.remote_root}/code/current"
    runner = NativeRunner(dry_run=dry_run, secrets=[config.remote_password or ""])
    health_script = shell_join(
        [
            f"test -d {quote_remote(config.remote_root)}",
            f"test -L {quote_remote(current_link)}",
            f"readlink -f {quote_remote(current_link)}",
            f"test -f {quote_remote(current_link + '/train.py')}",
            f"test -f {quote_remote(current_link + '/seedon_baseline/env.py')}",
            f"{quote_remote(config.remote_python)} --version",
        ]
    )
    smoke_script = build_remote_smoke_script(config, current_link)

    print(f"Project      : {config.project_slug}")
    print(f"Remote host  : {config.remote_host}")
    print(f"Current link : {current_link}")
    print(f"Backend      : {backend}")
    print()

    if backend == "paramiko":
        with ParamikoSession(config, dry_run=dry_run) as session:
            session.run(health_script)
            if smoke_script is not None:
                session.run(smoke_script)
        return

    command_env = build_askpass_env(config) if backend == "askpass" else None
    runner.run(build_ssh_command(config, backend, health_script), env=command_env)
    if smoke_script is not None:
        runner.run(build_ssh_command(config, backend, smoke_script), env=command_env)


def deploy_and_verify(config: DeployConfig, *, dry_run: bool) -> None:
    """Run the full build, upload, activation, and verification flow."""
    backend = select_backend(config)
    archive_path, layout = build_release_archive(config)

    print(f"Project       : {config.project_slug}")
    print(f"Source mode   : {config.source_mode}")
    print(f"Archive       : {archive_path}")
    print(f"Remote host   : {config.remote_host}")
    print(f"Remote release: {layout.release_dir}")
    print(f"Backend       : {backend}")
    print(f"Private assets: {config.include_private_assets}")
    print(f"Extra assets  : {config.include_extra_assets}")
    print(f"Deploy overlay: {DEFAULT_DEPLOY_CONTENT_DIR} ({len(iter_deploy_overlay_files())} files)")
    print()

    prepare_command = shell_join(
        [
            f"mkdir -p {quote_remote(layout.incoming_dir)}",
            f"mkdir -p {quote_remote(layout.code_root + '/releases')}",
            f"mkdir -p {quote_remote(layout.runs_dir + '/models')} "
            f"{quote_remote(layout.runs_dir + '/logs')} "
            f"{quote_remote(layout.runs_dir + '/reports')}",
        ]
    )
    extract_steps = []
    if config.clean_release:
        extract_steps.append(f"rm -rf {quote_remote(layout.release_dir)}")
    extract_steps.extend(
        [
            f"mkdir -p {quote_remote(layout.release_dir)}",
            f"tar xzf {quote_remote(layout.incoming_archive)} -C {quote_remote(layout.release_dir)}",
            *build_remote_asset_link_steps(config, layout),
            f"test -f {quote_remote(layout.release_dir + '/train.py')}",
            f"ln -sfn {quote_remote(layout.release_dir)} {quote_remote(layout.current_link)}",
            f"test \"$(readlink -f {quote_remote(layout.current_link)})\" = {quote_remote(layout.release_dir)}",
        ]
    )

    verify_script = build_remote_smoke_script(config, layout.current_link)

    if backend == "paramiko":
        with ParamikoSession(config, dry_run=dry_run) as session:
            session.run(prepare_command)
            session.put(archive_path, layout.incoming_archive)
            session.run(f"test -s {quote_remote(layout.incoming_archive)}")
            session.run(shell_join(extract_steps))
            if verify_script is not None:
                session.run(verify_script)
        return

    runner = NativeRunner(dry_run=dry_run, secrets=[config.remote_password or ""])
    command_env = build_askpass_env(config) if backend == "askpass" else None
    runner.run(build_ssh_command(config, backend, prepare_command), env=command_env)
    runner.run(build_scp_command(config, backend, archive_path, layout.incoming_archive), env=command_env)
    runner.run(
        build_ssh_command(config, backend, f"test -s {quote_remote(layout.incoming_archive)}"),
        env=command_env,
    )
    runner.run(build_ssh_command(config, backend, shell_join(extract_steps)), env=command_env)
    if verify_script is not None:
        runner.run(build_ssh_command(config, backend, verify_script), env=command_env)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    env_values = parse_env_file(args.env_file)
    config = build_config(args, env_values)
    if args.check_only:
        run_remote_health_check(config, dry_run=args.dry_run)
    else:
        deploy_and_verify(config, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1) from None
