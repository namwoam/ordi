"""Brahe-based orbital contact computation backend."""

from __future__ import annotations
from typing import Dict, List, Tuple

import numpy as np
import brahe as bh

from ordi.orbit._contact_types import (
    ContactEvent, DEFAULT_GROUND_STATIONS,
    GS_MIN_ELEVATION_DEG, ISL_MAX_RANGE_KM,
    DOWNLINK_RATE_BPS, ISL_RATE_BPS, UPLINK_RATE_BPS,
)

# EOP data is required for frame transforms inside location_accesses.
# Cached to ~/.cache/brahe/ after the first download (~2 MB).
bh.initialize_eop()


class _NamedPropagator:
    """Wraps a brahe KeplerianPropagator and exposes a .name attribute.

    Necessary because callers use `sat.name` (skyfield convention) but brahe
    propagators only provide `.get_name()` / `.set_name()`.
    """

    def __init__(self, propagator: bh.KeplerianPropagator) -> None:
        self._prop = propagator
        self.name: str = propagator.get_name()

    # Delegate propagation calls so this object can be passed directly to
    # brahe functions that expect a KeplerianPropagator.
    def get_name(self) -> str:
        return self.name

    def states_ecef(self, epochs):
        return self._prop.states_ecef(epochs)

    def states_eci(self, epochs):
        return self._prop.states_eci(epochs)


def build_synthetic_walker(
    n_planes: int = 6,
    sats_per_plane: int = 6,
    alt_km: float = 550.0,
    inc_deg: float = 53.0,
    epoch_str: str = "2024-01-01",
) -> List[_NamedPropagator]:
    """
    Generate a Walker-Delta constellation using brahe's KeplerianPropagator.

    The propagator epoch is set to t_start_unix=0.0 (1970-01-01) so that
    contact searches starting from that reference need zero backward propagation.
    epoch_str is accepted for API compatibility but not used.
    """
    epc0 = bh.Epoch.from_unix_timestamp(0.0)
    a_m = (6371.0 + alt_km) * 1e3
    total = n_planes * sats_per_plane

    gen = bh.WalkerConstellationGenerator(
        t=total,
        p=n_planes,
        f=1,
        semi_major_axis=a_m,
        eccentricity=0.001,
        inclination=inc_deg,
        argument_of_perigee=0.0,
        reference_raan=0.0,
        reference_mean_anomaly=0.0,
        epoch=epc0,
        angle_format=bh.AngleFormat.DEGREES,
        pattern=bh.WalkerPattern.DELTA,
    ).with_base_name("SAT")

    return [_NamedPropagator(p) for p in gen.as_keplerian_propagators(30.0)]


def compute_contact_windows(
    satellites: List[_NamedPropagator],
    t_start_unix: float,
    t_end_unix: float,
    ground_stations: List[Tuple[str, float, float]] = None,
    isl_max_range_km: float = ISL_MAX_RANGE_KM,
    dt_seconds: float = 30.0,
    min_elevation_deg: float = GS_MIN_ELEVATION_DEG,
) -> List[ContactEvent]:
    """
    Compute all contact windows (sat-ground + sat-sat ISL) in [t_start, t_end].

    Ground contacts use brahe's location_accesses() with an ElevationConstraint.
    ISL contacts use coarse position sampling identical to the skyfield backend.
    """
    if ground_stations is None:
        ground_stations = DEFAULT_GROUND_STATIONS

    epc_start = bh.Epoch.from_unix_timestamp(t_start_unix)
    epc_end   = bh.Epoch.from_unix_timestamp(t_end_unix)
    events: List[ContactEvent] = []

    # ── satellite-ground contacts ────────────────────────────────────────────
    constraint = bh.ElevationConstraint(min_elevation_deg=min_elevation_deg)
    bh_locs = [
        bh.PointLocation(lon=lon, lat=lat, alt=0.0).with_name(gs_name)
        for gs_name, lat, lon in ground_stations
    ]

    raw_props = [s._prop for s in satellites]
    windows = bh.location_accesses(bh_locs, raw_props, epc_start, epc_end, constraint)
    for w in windows:
        t0 = w.t_start.unix_timestamp()
        t1 = w.t_end.unix_timestamp()
        sat_name = w.satellite_name
        gs_name  = w.location_name
        events.append(ContactEvent(t0, t1, sat_name, gs_name, DOWNLINK_RATE_BPS, "downlink"))
        events.append(ContactEvent(t0, t1, gs_name, sat_name, UPLINK_RATE_BPS,   "uplink"))

    # ── ISL contacts (coarse position sampling) ──────────────────────────────
    n = len(satellites)
    n_steps = int((t_end_unix - t_start_unix) / dt_seconds) + 1
    epochs = [bh.Epoch.from_unix_timestamp(t_start_unix + k * dt_seconds)
              for k in range(n_steps)]

    # positions[i]: (n_steps, 3) array of ECEF positions in km
    positions = np.zeros((n, n_steps, 3))
    for idx, sat in enumerate(satellites):
        states = sat.states_ecef(epochs)          # list of (6,) arrays, meters
        positions[idx] = np.array(states)[:, :3] / 1000.0

    for i in range(n):
        for j in range(i + 1, n):
            diff   = positions[i] - positions[j]
            ranges = np.linalg.norm(diff, axis=1)
            in_contact = ranges <= isl_max_range_km

            prev, seg_start = False, None
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


def compute_sat_groundtracks(
    satellites: List[_NamedPropagator],
    t_start_unix: float,
    t_end_unix: float,
    dt_seconds: float = 60.0,
) -> Dict[str, List[Tuple[float, float, float]]]:
    """
    Return {sat_id: [(t_unix, lat_deg, lon_deg), ...]} sampled every dt_seconds.
    """
    n_steps = int((t_end_unix - t_start_unix) / dt_seconds) + 1
    t_unix_grid = [t_start_unix + i * dt_seconds for i in range(n_steps)]
    epochs = [bh.Epoch.from_unix_timestamp(t) for t in t_unix_grid]

    result: dict = {}
    for sat in satellites:
        states = sat.states_ecef(epochs)  # list of (6,) arrays, meters
        track = []
        for t_unix, state in zip(t_unix_grid, states):
            pos_ecef_m = np.array(state[:3])
            # position_ecef_to_geodetic returns [lon_deg, lat_deg, alt_m]
            geodetic = bh.position_ecef_to_geodetic(pos_ecef_m, bh.AngleFormat.DEGREES)
            track.append((t_unix, float(geodetic[1]), float(geodetic[0])))
        result[sat.name] = track
    return result
