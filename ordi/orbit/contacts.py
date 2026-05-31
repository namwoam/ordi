"""
Contact window computation — backend selectable via ORBIT_BACKEND env var.

Supported values:
  skyfield  (default) — Skyfield + SGP4
  brahe               — brahe (Rust-based, https://github.com/duncaneddy/brahe)

Usage:
  ORBIT_BACKEND=brahe task e1
"""
from __future__ import annotations
import os

from ordi.orbit._contact_types import ContactEvent, DEFAULT_GROUND_STATIONS  # noqa: F401

_BACKEND = os.environ.get("ORBIT_BACKEND", "skyfield").lower()

if _BACKEND == "brahe":
    from ordi.orbit._brahe_backend import (  # noqa: F401
        build_synthetic_walker,
        compute_contact_windows,
        compute_sat_groundtracks,
    )
elif _BACKEND == "skyfield":
    from ordi.orbit._skyfield_backend import (  # noqa: F401
        build_synthetic_walker,
        compute_contact_windows,
        compute_sat_groundtracks,
    )
else:
    raise ValueError(
        f"Unknown ORBIT_BACKEND={_BACKEND!r}. Valid values: 'skyfield', 'brahe'"
    )

__all__ = [
    "ContactEvent",
    "DEFAULT_GROUND_STATIONS",
    "build_synthetic_walker",
    "compute_contact_windows",
    "compute_sat_groundtracks",
]
