"""Lightweight helpers for the unified training entrypoint."""

from __future__ import annotations


def split_mode_args(argv: list[str]) -> tuple[str, list[str]]:
    """Extract top-level mode flags and return remaining arguments."""
    if "--h1" in argv and "--grasp" in argv:
        raise ValueError("Cannot use --h1 and --grasp together.")

    mode: str | None = None
    forwarded: list[str] = []
    index = 0

    while index < len(argv):
        arg = argv[index]
        if arg in {"--h1", "--grasp"}:
            selected_mode = "grasp" if arg == "--grasp" else "h1"
            if mode is not None and mode != selected_mode:
                raise ValueError("Cannot set multiple training targets.")
            mode = selected_mode
            index += 1
            continue
        if arg == "--project":
            if index + 1 >= len(argv):
                raise ValueError("--project requires a value.")
            selected_mode = argv[index + 1].strip().lower()
            if selected_mode not in {"h1", "grasp"}:
                raise ValueError("--project must be one of: h1, grasp.")
            if mode is not None and mode != selected_mode:
                raise ValueError("Cannot mix --project with a conflicting mode flag.")
            mode = selected_mode
            index += 2
            continue
        forwarded.append(arg)
        index += 1

    if mode is None:
        mode = "h1"
    return mode, forwarded
