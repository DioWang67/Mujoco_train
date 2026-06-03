"""Compatibility wrapper for the Seedon Phase 2C contact mapping diagnostic.

The implementation moved to
``tools.seedon.diagnostics.contact.phase2c_contact_constrained_foot_mapping``.
Keep this module so existing ``python -m tools.seedon_phase2c_contact_constrained_foot_mapping``
commands and imports continue to work.
"""

from __future__ import annotations

from tools.seedon.diagnostics.contact.phase2c_contact_constrained_foot_mapping import *  # noqa: F401,F403
from tools.seedon.diagnostics.contact.phase2c_contact_constrained_foot_mapping import main


if __name__ == "__main__":
    raise SystemExit(main())
