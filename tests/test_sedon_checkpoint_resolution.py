"""Tests for Sedon checkpoint path resolution."""

from __future__ import annotations

import zipfile
import re
from pathlib import Path


def _extract_function_source(path: Path, function_name: str) -> str:
    """Return the exact source code for one top-level function."""
    lines = path.read_text(encoding="utf-8").splitlines()
    start = None
    indent = 0
    for index, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith(f"def {function_name}("):
            start = index
            indent = len(line) - len(stripped)
            break
    if start is None:
        raise AssertionError(f"Function {function_name} not found in {path}.")

    end = len(lines)
    for index in range(start + 1, len(lines)):
        line = lines[index]
        stripped = line.lstrip()
        current_indent = len(line) - len(stripped)
        if (stripped.startswith("def ") or stripped.startswith("class ") or stripped.startswith("@")) and current_indent == indent:
            end = index
            break
    return "\n".join(lines[start:end]) + "\n"


MODULE_PATH = Path(__file__).resolve().parents[1] / "sedon_baseline" / "eval.py"
MODULE_GLOBALS = {
    "Path": Path,
    "zipfile": zipfile,
    "STEP_CHECKPOINT_RE": re.compile(r"sedon_ppo_(\d+)_steps\.zip$"),
}
exec(_extract_function_source(MODULE_PATH, "_is_valid_sb3_checkpoint"), MODULE_GLOBALS)
exec(_extract_function_source(MODULE_PATH, "resolve_model_path"), MODULE_GLOBALS)
exec(_extract_function_source(MODULE_PATH, "resolve_vecnorm_path"), MODULE_GLOBALS)
resolve_model_path = MODULE_GLOBALS["resolve_model_path"]
resolve_vecnorm_path = MODULE_GLOBALS["resolve_vecnorm_path"]


def _write_good_checkpoint(path: Path) -> None:
    """Create a minimal SB3-like zip with a readable ``data`` payload."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("data", "{}")


def _write_bad_checkpoint(path: Path) -> None:
    """Create a file that looks like a zip by name only."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not-a-zip")


def test_resolve_model_path_skips_corrupt_best_checkpoint(tmp_path: Path) -> None:
    models_root = tmp_path / "models" / "sedon"
    _write_bad_checkpoint(models_root / "best" / "best_model.zip")
    _write_good_checkpoint(models_root / "latest_model.zip")

    resolved = resolve_model_path(models_root, explicit_model_path=None)

    assert resolved == models_root / "latest_model.zip"


def test_resolve_model_path_rejects_explicit_corrupt_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "broken.zip"
    _write_bad_checkpoint(checkpoint)

    try:
        resolve_model_path(tmp_path, explicit_model_path=checkpoint)
    except ValueError as exc:
        assert "not a readable SB3 zip" in str(exc)
    else:
        raise AssertionError("Expected explicit corrupt checkpoint to raise ValueError.")


def test_resolve_vecnorm_path_prefers_matching_step_file(tmp_path: Path) -> None:
    models_root = tmp_path / "models" / "sedon"
    model_path = models_root / "sedon_ppo_49920_steps.zip"
    vecnorm_path = models_root / "sedon_vecnorm_49920_steps.pkl"
    fallback_vecnorm_path = models_root / "vecnorm.pkl"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"model")
    vecnorm_path.write_bytes(b"matching")
    fallback_vecnorm_path.write_bytes(b"fallback")

    resolved = resolve_vecnorm_path(models_root, model_path, explicit_vecnorm_path=None)

    assert resolved == vecnorm_path


def test_resolve_vecnorm_path_prefers_best_model_vecnorm(tmp_path: Path) -> None:
    models_root = tmp_path / "models" / "sedon"
    model_path = models_root / "best" / "best_model.zip"
    best_vecnorm_path = models_root / "best" / "vecnorm.pkl"
    fallback_vecnorm_path = models_root / "vecnorm.pkl"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"model")
    best_vecnorm_path.write_bytes(b"best")
    fallback_vecnorm_path.write_bytes(b"fallback")

    resolved = resolve_vecnorm_path(models_root, model_path, explicit_vecnorm_path=None)

    assert resolved == best_vecnorm_path
