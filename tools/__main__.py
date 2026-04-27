"""Print the repository tool index.

Usage:
    python -m tools
"""

from __future__ import annotations

from tools.registry import tools_by_category


def main() -> int:
    """Print available tools grouped by purpose."""
    print("Available tools:")
    for category, entries in tools_by_category().items():
        print(f"\n[{category}]")
        for entry in entries:
            print(f"  {entry.command:<34} {entry.summary}")
            print(f"    e.g. {entry.example}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
