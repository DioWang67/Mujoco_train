"""Compatibility wrapper for the Sedon contact reset diagnostic.

The implementation moved to ``tools.sedon.diagnostics.contact.debug_sedon_contacts``.
Keep this module so existing ``python -m tools.debug_sedon_contacts`` commands and
imports continue to work.
"""

from __future__ import annotations

from tools.sedon.diagnostics.contact.debug_sedon_contacts import *  # noqa: F401,F403
from tools.sedon.diagnostics.contact.debug_sedon_contacts import main


if __name__ == "__main__":
    raise SystemExit(main())
