"""Shared types and constants for orbital contact computation backends."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

GS_MIN_ELEVATION_DEG = 25.0
# Starlink mini optical terminals specify links up to 4,000 km.  At the
# simulated 550 km shell this also leaves about 255 km Earth-limb clearance.
ISL_MAX_RANGE_KM = 4000.0
DOWNLINK_RATE_BPS = 100e6
ISL_RATE_BPS = 200e6
UPLINK_RATE_BPS = 10e6

DEFAULT_GROUND_STATIONS: List[Tuple[str, float, float]] = [
    ("fairbanks",    64.8,  -147.7),
    ("svalbard",     78.2,    15.4),
    ("punta_arenas", -53.2,  -70.9),
    ("singapore",     1.35, 103.8),
    ("nairobi",      -1.3,   36.8),
    ("hawaii",       19.7, -155.1),
    ("norway",       69.7,   18.9),
    ("diego_garcia",  -7.3,  72.4),
    ("mcmurdo",     -77.9,  166.7),
    ("greenwich",    51.5,   -0.1),
]


@dataclass
class ContactEvent:
    t_start: float      # unix timestamp
    t_end: float        # unix timestamp
    node_a: str         # satellite id or ground station name
    node_b: str
    rate_bps: float     # available link rate
    link_type: str      # "downlink" | "uplink" | "isl"

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start

    @property
    def capacity_bits(self) -> float:
        return self.rate_bps * self.duration
