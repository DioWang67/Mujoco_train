from pathlib import Path

from tools.project_inventory import (
    cleanup_reason,
    find_cleanup_candidates,
    format_inventory,
    summarize_directories,
)


def test_cleanup_reason_marks_known_runtime_outputs() -> None:
    assert cleanup_reason("models") == "cache or runtime output"
    assert cleanup_reason("_verify_reverse_knee") == "temporary verification output"
    assert cleanup_reason("artifacts") == "generated artifacts"
    assert cleanup_reason("sedon_baseline") is None


def test_summarize_directories_reports_missing_and_existing_paths(tmp_path: Path) -> None:
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "train.json").write_text("{}", encoding="utf-8")

    summaries = {summary.name: summary for summary in summarize_directories(tmp_path)}

    assert summaries["configs"].exists is True
    assert summaries["configs"].file_count == 1
    assert summaries["tools"].exists is False
    assert summaries["tools"].file_count == 0


def test_find_cleanup_candidates_only_returns_top_level_disposable_dirs(tmp_path: Path) -> None:
    (tmp_path / "_verify_probe").mkdir()
    (tmp_path / "_verify_probe" / "result.txt").write_text("ok", encoding="utf-8")
    (tmp_path / "logs").mkdir()
    (tmp_path / "sedon_baseline").mkdir()

    candidates = find_cleanup_candidates(tmp_path)
    candidate_paths = {candidate.path.as_posix() for candidate in candidates}

    assert candidate_paths == {"_verify_probe", "logs"}


def test_format_inventory_includes_rules_and_counts(tmp_path: Path) -> None:
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "sample.py").write_text("print('ok')", encoding="utf-8")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "sample.bat").write_text("@echo off", encoding="utf-8")

    report = format_inventory(tmp_path)

    assert "Project inventory" in report
    assert "Python tools: 1 files under tools/" in report
    assert "Wrapper scripts: 1 files under scripts/" in report
    assert "Keep repeatable parameters in configs/." in report
