"""Compatibility wrapper for the Seedon contact reset diagnostic.

The implementation moved to ``tools.seedon.diagnostics.contact.debug_seedon_contacts``.
Keep this module so existing ``python -m tools.debug_seedon_contacts`` commands and
imports continue to work.
"""

from __future__ import annotations

from tools.seedon.diagnostics.contact.debug_seedon_contacts import *  # noqa: F401,F403
from tools.seedon.diagnostics.contact.debug_seedon_contacts import main


if __name__ == "__main__":
    raise SystemExit(main())
