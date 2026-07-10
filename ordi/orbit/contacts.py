"""Contact window computation (Skyfield + SGP4 backend)."""
from __future__ import annotations

from ordi.orbit._contact_types import ContactEvent, DEFAULT_GROUND_STATIONS  # noqa: F401
from ordi.orbit._skyfield_backend import (  # noqa: F401
    build_synthetic_walker,
    compute_contact_windows,
    compute_sat_groundtracks,
)

__all__ = [
    "ContactEvent",
    "DEFAULT_GROUND_STATIONS",
    "build_synthetic_walker",
    "compute_contact_windows",
    "compute_sat_groundtracks",
]
