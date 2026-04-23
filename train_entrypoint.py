"""Lightweight helpers for the unified training entrypoint."""

from __future__ import annotations


def split_mode_args(argv: list[str]) -> tuple[str, list[str]]:
    """Extract top-level mode flags and return remaining arguments."""
    if "--h1" in argv and "--grasp" in argv:
        raise ValueError("Cannot use --h1 and --grasp together.")

    mode = "grasp" if "--grasp" in argv else "h1"
    forwarded = [arg for arg in argv if arg not in {"--h1", "--grasp"}]
    return mode, forwarded
