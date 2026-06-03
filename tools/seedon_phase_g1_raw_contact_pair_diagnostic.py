"""Compatibility wrapper for the Seedon Phase G1 raw contact-pair diagnostic.

The implementation moved to
``tools.seedon.diagnostics.contact.phase_g1_raw_contact_pair_diagnostic``.
Keep this module so existing ``python -m tools.seedon_phase_g1_raw_contact_pair_diagnostic``
commands and imports continue to work.
"""

from __future__ import annotations

from tools.seedon.diagnostics.contact.phase_g1_raw_contact_pair_diagnostic import *  # noqa: F401,F403
from tools.seedon.diagnostics.contact.phase_g1_raw_contact_pair_diagnostic import main


if __name__ == "__main__":
    raise SystemExit(main())
