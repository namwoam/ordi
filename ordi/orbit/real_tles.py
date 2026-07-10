"""Real LEO constellation loader (CelesTrak TLEs).

Fetches a real Earth-observation constellation (default: Planet Flock/SkySat) as
Two-Line Element sets from CelesTrak and builds Skyfield ``EarthSatellite``
objects that are drop-in replacements for ``build_synthetic_walker``'s output.

Real satellites are renamed to the synthetic ``SAT_<plane>_<idx>`` convention so
that the plane-aware machinery keeps working: ``scheduler.ordi._plane_of`` parses
the plane id for plane-disjoint backups, and ``eval.experiments`` targets faults
by ``SAT_{plane:02d}_`` prefix.  Real constellations like Planet are not clean
Walker patterns, so "planes" are synthesized by clustering on RAAN (right
ascension of the ascending node); this is an explicit approximation.

Network access is only needed to (re)fill the cache.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ordi.orbit._skyfield_backend import _load_tle_lines

CELESTRAK_GROUP_URL = (
    "https://celestrak.org/NORAD/elements/gp.php?GROUP={group}&FORMAT=tle"
)


def _data_dir() -> Path:
    root = Path(os.environ.get("ORDI_DATA_DIR", Path(__file__).resolve().parents[2] / "data"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def fetch_constellation_tles(
    group: str = "planet",
    cache_path: Optional[str] = None,
    refresh: bool = False,
    timeout_s: float = 30.0,
) -> str:
    """Download a CelesTrak GP group to a local ``.tle`` cache and return its path.

    Skips the network fetch if the cache already exists (unless ``refresh``).
    """
    cache = Path(cache_path) if cache_path else _data_dir() / f"celestrak_{group}.tle"
    if cache.exists() and not refresh:
        return str(cache)
    url = CELESTRAK_GROUP_URL.format(group=group)
    with urllib.request.urlopen(url, timeout=timeout_s) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    if "1 " not in body:
        raise RuntimeError(f"CelesTrak returned no TLEs for group={group!r}: {body[:200]!r}")
    cache.write_text(body)
    return str(cache)


def _raan_deg(line2: str) -> float:
    """Right ascension of the ascending node (deg) from TLE line 2, cols 18–25."""
    try:
        return float(line2[17:25])
    except ValueError:
        return 0.0


def _assign_planes(
    entries: List[Tuple[str, str, str]], n_planes: int
) -> List[int]:
    """Cluster satellites into n_planes synthetic planes by RAAN.

    Sort by RAAN and bin into equal-count buckets so every plane is populated
    even when the real RAAN distribution is lumpy.  Returns a plane id per entry
    in the original ordering.
    """
    order = sorted(range(len(entries)), key=lambda i: _raan_deg(entries[i][2]))
    plane_of = [0] * len(entries)
    for rank, idx in enumerate(order):
        plane_of[idx] = min(rank * n_planes // max(len(entries), 1), n_planes - 1)
    return plane_of


def load_real_constellation(
    tle_path: str,
    n_planes: int = 6,
    max_sats: Optional[int] = None,
) -> Tuple[list, Dict[str, str]]:
    """Build EarthSatellite objects from a real TLE file.

    Returns ``(satellites, name_map)`` where each satellite's ``.name`` follows
    the synthetic ``SAT_<plane>_<idx>`` convention and ``name_map`` maps that
    synthetic name back to the real catalog name for reporting.

    max_sats caps the constellation size (evenly subsampled across the file so
    the RAAN spread is preserved) to keep contact-window computation tractable.
    """
    from sgp4.api import Satrec
    from skyfield.api import EarthSatellite, load

    entries = _load_tle_lines(tle_path)
    if not entries:
        raise ValueError(f"No TLEs parsed from {tle_path}")

    if max_sats is not None and len(entries) > max_sats:
        step = len(entries) / max_sats
        entries = [entries[int(i * step)] for i in range(max_sats)]

    plane_of = _assign_planes(entries, n_planes)
    per_plane_idx: Dict[int, int] = {}

    ts = load.timescale()
    sats: list = []
    name_map: Dict[str, str] = {}
    for (real_name, l1, l2), plane in zip(entries, plane_of):
        try:
            satrec = Satrec.twoline2rv(l1, l2)
            sat = EarthSatellite.from_satrec(satrec, ts)
        except Exception:
            continue
        idx = per_plane_idx.get(plane, 0)
        per_plane_idx[plane] = idx + 1
        syn_name = f"SAT_{plane:02d}_{idx:02d}"
        sat.name = syn_name
        name_map[syn_name] = real_name.strip()
        sats.append(sat)
    return sats, name_map


def load_planet_constellation(
    n_planes: int = 6,
    max_sats: Optional[int] = 24,
    refresh: bool = False,
) -> Tuple[list, Dict[str, str]]:
    """Convenience: fetch (cached) Planet TLEs and build the constellation."""
    tle_path = fetch_constellation_tles(group="planet", refresh=refresh)
    return load_real_constellation(tle_path, n_planes=n_planes, max_sats=max_sats)
