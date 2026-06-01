from pathlib import Path

import pytest

from tools.remote_auto_deploy import (
    NativeRunner,
    build_config,
    build_scp_command,
    build_ssh_command,
    parse_remote_host,
    parse_args,
    parse_env_file,
    select_backend,
)


def test_parse_env_file_ignores_comments_and_strips_quotes(tmp_path: Path) -> None:
    env_file = tmp_path / ".env.remote"
    env_file.write_text(
        "\n".join(
            [
                "# comment",
                "REMOTE_HOST='root@example.test'",
                'REMOTE_ROOT="/opt/mujoco"',
                "REMOTE_PASSWORD=secret",
            ]
        ),
        encoding="utf-8",
    )

    values = parse_env_file(env_file)

    assert values["REMOTE_HOST"] == "root@example.test"
    assert values["REMOTE_ROOT"] == "/opt/mujoco"
    assert values["REMOTE_PASSWORD"] == "secret"


def test_parse_env_file_rejects_invalid_line(tmp_path: Path) -> None:
    env_file = tmp_path / ".env.remote"
    env_file.write_text("REMOTE_HOST root@example.test\n", encoding="utf-8")

    with pytest.raises(ValueError, match="expected KEY=VALUE"):
        parse_env_file(env_file)


def test_build_config_defaults_to_worktree_sedon() -> None:
    config = build_config(parse_args([]), {})

    assert config.project_slug == "sedon"
    assert config.verify_project == "sedon"
    assert config.source_mode == "working-tree"


def test_no_include_private_assets_overrides_env() -> None:
    config = build_config(
        parse_args(["--no-include-private-assets"]),
        {"REMOTE_INCLUDE_PRIVATE_ASSETS": "1"},
    )

    assert config.include_private_assets is False


def test_extra_assets_default_to_false_for_fast_deploy() -> None:
    config = build_config(parse_args([]), {})

    assert config.include_extra_assets is False


def test_openssh_rejects_password_mode() -> None:
    with pytest.raises(ValueError, match="cannot use passwords"):
        build_config(parse_args(["--backend", "openssh"]), {"REMOTE_PASSWORD": "pw"})


def test_select_backend_uses_openssh_without_password() -> None:
    config = build_config(parse_args([]), {"REMOTE_PASSWORD": ""})

    assert select_backend(config) == "openssh"


def test_parse_remote_host_accepts_user_host_port() -> None:
    user, host, port = parse_remote_host("root@example.test:2222")

    assert user == "root"
    assert host == "example.test"
    assert port == 2222


def test_select_backend_can_request_paramiko() -> None:
    config = build_config(parse_args(["--backend", "paramiko"]), {"REMOTE_PASSWORD": "pw"})

    assert select_backend(config) == "paramiko"


def test_native_backend_allows_interactive_password_mode() -> None:
    config = build_config(parse_args(["--backend", "native"]), {"REMOTE_PASSWORD": "pw"})

    assert select_backend(config) == "native"


def test_askpass_backend_allows_env_password_mode() -> None:
    config = build_config(parse_args(["--backend", "askpass"]), {"REMOTE_PASSWORD": "pw"})

    assert select_backend(config) == "askpass"


def test_password_commands_use_requested_putty_backend() -> None:
    config = build_config(
        parse_args(["--backend", "putty"]),
        {
            "REMOTE_PASSWORD": "pw",
            "REMOTE_HOST": "root@example.test",
        },
    )

    ssh_command = build_ssh_command(config, "putty", "echo ok")
    scp_command = build_scp_command(config, "putty", Path("release.tar.gz"), "/tmp/release.tar.gz")

    assert ssh_command[:5] == ["plink", "-batch", "-ssh", "-pw", "pw"]
    assert "root@example.test" in ssh_command
    assert scp_command[:5] == ["pscp", "-batch", "-scp", "-pw", "pw"]


def test_runner_masks_password_in_dry_run_output(capsys: pytest.CaptureFixture[str]) -> None:
    runner = NativeRunner(dry_run=True, secrets=["super-secret"])

    runner.run(["plink", "-pw", "super-secret", "root@example.test", "echo ok"])

    output = capsys.readouterr().out
    assert "super-secret" not in output
    assert "********" in output
