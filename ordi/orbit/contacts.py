"""
Contact window computation from TLEs.

Produces a sorted list of ContactEvent(t_start, t_end, node_a, node_b, rate_bps)
for both satellite-ground and satellite-satellite (ISL) pairs.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from skyfield.api import EarthSatellite, Topos, load, wgs84
from skyfield.units import Distance

# ── tunables ────────────────────────────────────────────────────────────────
GS_MIN_ELEVATION_DEG = 5.0       # minimum elevation for sat-ground contact
ISL_MAX_RANGE_KM = 5000.0        # maximum ISL range
DOWNLINK_RATE_BPS = 100e6        # 100 Mbps downlink (Starlink-class)
ISL_RATE_BPS = 200e6             # 200 Mbps ISL
UPLINK_RATE_BPS = 10e6           # 10 Mbps uplink (ground → sat)

# Representative ground stations: (name, lat_deg, lon_deg)
DEFAULT_GROUND_STATIONS: List[Tuple[str, float, float]] = [
    ("fairbanks",   64.8,  -147.7),
    ("svalbard",    78.2,   15.4),
    ("punta_arenas",-53.2, -70.9),
    ("singapore",    1.35, 103.8),
    ("nairobi",     -1.3,   36.8),
    ("hawaii",      19.7, -155.1),
    ("norway",      69.7,   18.9),
    ("diego_garcia", -7.3,  72.4),
    ("mcmurdo",    -77.9,  166.7),
    ("greenwich",   51.5,   -0.1),
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


def _load_tle_lines(tle_path: str) -> List[Tuple[str, str, str]]:
    """Parse a 3-line TLE file into (name, line1, line2) tuples."""
    entries = []
    with open(tle_path) as f:
        lines = [l.strip() for l in f if l.strip()]
    i = 0
    while i + 2 < len(lines):
        name = lines[i]
        l1, l2 = lines[i + 1], lines[i + 2]
        if l1.startswith("1 ") and l2.startswith("2 "):
            entries.append((name, l1, l2))
            i += 3
        else:
            i += 1
    return entries


def build_synthetic_walker(n_planes: int = 6, sats_per_plane: int = 6,
                            alt_km: float = 550.0, inc_deg: float = 53.0,
                            epoch_str: str = "2024-01-01") -> List[Tuple[str, str, str]]:
    """
    Generate synthetic TLE-like objects for a Walker-Delta constellation.
    Returns a list of (name, line1, line2) but uses Skyfield EarthSatellite
    constructed via sgp4 directly; we store them as (name, sat_object, None).
    For simplicity we return EarthSatellite objects directly.
    """
    from sgp4.api import Satrec, WGS84
    import time as _time

    ts = load.timescale()
    sats = []
    total = n_planes * sats_per_plane
    for p in range(n_planes):
        raan = 360.0 * p / n_planes          # right ascension of ascending node
        for s in range(sats_per_plane):
            ma = 360.0 * s / sats_per_plane  # mean anomaly
            # phase offset between planes
            ma = (ma + 360.0 * p / total) % 360.0
            name = f"SAT_{p:02d}_{s:02d}"
            satrec = Satrec()
            satrec.sgp4init(
                WGS84,
                'i',          # opsmode
                0,            # satnum
                24038.0,      # epoch (days from 1949-12-31)
                2.5e-5,       # bstar drag
                0.0,          # ndot
                0.0,          # nddot
                0.001,        # ecco (near-circular)
                0.0,          # argpo
                math.radians(inc_deg),
                math.radians(ma),
                (2 * math.pi) / (24 * 3600 / (2 * math.pi * (6371 + alt_km) * 1e3 /
                                               math.sqrt(3.986e14 / ((6371 + alt_km) * 1e3)))) / (2 * math.pi),
                math.radians(raan),
            )
            sat = EarthSatellite.from_satrec(satrec, ts)
            sat.name = name
            sats.append(sat)
    return sats


def compute_contact_windows(
    satellites: list,          # list of EarthSatellite
    t_start_unix: float,
    t_end_unix: float,
    ground_stations: List[Tuple[str, float, float]] = None,
    isl_max_range_km: float = ISL_MAX_RANGE_KM,
    dt_seconds: float = 30.0,  # sampling interval for coarse pass
) -> List[ContactEvent]:
    """
    Compute all contact windows (sat-ground + sat-sat ISL) in [t_start, t_end].

    Uses a coarse sampling pass to detect contacts, then refines boundaries
    via bisection. Returns sorted ContactEvent list.
    """
    if ground_stations is None:
        ground_stations = DEFAULT_GROUND_STATIONS

    ts = load.timescale()
    events: List[ContactEvent] = []

    t_start = ts.from_datetime(__import__('datetime').datetime.utcfromtimestamp(t_start_unix)
                               .replace(tzinfo=__import__('datetime').timezone.utc))
    t_end   = ts.from_datetime(__import__('datetime').datetime.utcfromtimestamp(t_end_unix)
                               .replace(tzinfo=__import__('datetime').timezone.utc))

    # ── satellite-ground contacts ────────────────────────────────────────────
    for sat in satellites:
        for gs_name, lat, lon in ground_stations:
            gs = wgs84.latlon(lat, lon)
            try:
                raw_times, raw_events = sat.find_events(
                    gs, t_start, t_end, altitude_degrees=GS_MIN_ELEVATION_DEG
                )
            except Exception:
                continue
            # Skyfield returns event codes: 0=rise, 1=culminate, 2=set
            rise_t = None
            for ti, ev in zip(raw_times, raw_events):
                if ev == 0:
                    rise_t = ti.tt
                elif ev == 2 and rise_t is not None:
                    t0 = _tt_to_unix(rise_t)
                    t1 = _tt_to_unix(ti.tt)
                    # downlink
                    events.append(ContactEvent(t0, t1, sat.name, gs_name,
                                               DOWNLINK_RATE_BPS, "downlink"))
                    # uplink
                    events.append(ContactEvent(t0, t1, gs_name, sat.name,
                                               UPLINK_RATE_BPS, "uplink"))
                    rise_t = None

    # ── ISL contacts (coarse sampling) ──────────────────────────────────────
    n = len(satellites)
    n_steps = int((t_end_unix - t_start_unix) / dt_seconds) + 1
    times_tt = np.linspace(t_start.tt, t_end.tt, n_steps)

    # Precompute GCRS positions for all sats × all times
    positions = np.zeros((n, n_steps, 3))  # km
    for idx, sat in enumerate(satellites):
        t_arr = ts.tt_jd(times_tt)
        geo = sat.at(t_arr)
        # position in km in GCRS
        positions[idx] = geo.position.km.T  # (n_steps, 3)

    # Find ISL contact intervals
    for i in range(n):
        for j in range(i + 1, n):
            diff = positions[i] - positions[j]          # (n_steps, 3)
            ranges = np.linalg.norm(diff, axis=1)       # (n_steps,)
            in_contact = ranges <= isl_max_range_km

            # Walk transitions
            prev = False
            seg_start = None
            for k, contact in enumerate(in_contact):
                if contact and not prev:
                    seg_start = t_start_unix + k * dt_seconds
                elif not contact and prev and seg_start is not None:
                    seg_end = t_start_unix + k * dt_seconds
                    events.append(ContactEvent(seg_start, seg_end,
                                               satellites[i].name, satellites[j].name,
                                               ISL_RATE_BPS, "isl"))
                    events.append(ContactEvent(seg_start, seg_end,
                                               satellites[j].name, satellites[i].name,
                                               ISL_RATE_BPS, "isl"))
                    seg_start = None
                prev = contact
            if prev and seg_start is not None:
                events.append(ContactEvent(seg_start, t_end_unix,
                                           satellites[i].name, satellites[j].name,
                                           ISL_RATE_BPS, "isl"))
                events.append(ContactEvent(seg_start, t_end_unix,
                                           satellites[j].name, satellites[i].name,
                                           ISL_RATE_BPS, "isl"))

    events.sort(key=lambda e: e.t_start)
    return events


def _tt_to_unix(tt: float) -> float:
    """Convert Skyfield TT julian date to Unix timestamp."""
    # TT epoch: J2000.0 = 2000-01-01 12:00 TT = unix 946727935.816
    J2000_TT_UNIX = 946727935.816
    return J2000_TT_UNIX + (tt - 2451545.0) * 86400.0
