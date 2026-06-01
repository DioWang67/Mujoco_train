from pathlib import Path

from tools.deploy_release import copy_deploy_overlay, iter_deploy_overlay_files


def test_iter_deploy_overlay_files_skips_docs_placeholders(tmp_path: Path) -> None:
    deploy_dir = tmp_path / "deploy_content"
    deploy_dir.mkdir()
    (deploy_dir / "README.md").write_text("docs", encoding="utf-8")
    (deploy_dir / ".gitkeep").write_text("", encoding="utf-8")
    nested = deploy_dir / "configs" / "sedon"
    nested.mkdir(parents=True)
    payload = nested / "custom.json"
    payload.write_text("{}", encoding="utf-8")

    files = iter_deploy_overlay_files(deploy_dir)

    assert files == [payload]


def test_copy_deploy_overlay_preserves_repo_relative_paths(tmp_path: Path) -> None:
    deploy_dir = tmp_path / "deploy_content"
    staging_root = tmp_path / "staging"
    source = deploy_dir / "configs" / "sedon" / "custom.json"
    source.parent.mkdir(parents=True)
    source.write_text('{"ok": true}', encoding="utf-8")

    copied = copy_deploy_overlay(staging_root, deploy_dir)

    assert copied == 1
    assert (staging_root / "configs" / "sedon" / "custom.json").read_text(
        encoding="utf-8"
    ) == '{"ok": true}'
