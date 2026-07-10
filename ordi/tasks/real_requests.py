"""Real EO tasking *requests* from NASA FIRMS active-fire detections.

Unlike Sentinel-2 acquisitions (systematic survey captures used as a stand-in
for demand — see ``real_acquisitions.py``), a FIRMS fire detection is a genuine
"something happened here, image it now" event.  Each detection therefore
supplies the full request semantics that the acquisition loader had to sample:

  - when      : real detection timestamp (acq_date + acq_time UTC)
  - where     : real fire latitude/longitude
  - type      : wildfire (not sampled) -> the MobileNetV2 wildfire profile
  - urgency   : fire radiative power (FRP, MW) -> utility scale + deadline
  - priority  : detection confidence gates low-quality events

FIRMS NRT feeds (VIIRS/MODIS) are public with a free MAP_KEY:
    https://firms.modaps.eosdis.nasa.gov/api/area/
Set FIRMS_MAP_KEY in the environment (or pass map_key=).  Results are cached
under ORDI_DATA_DIR so the network fetch runs once.
"""

from __future__ import annotations

import csv
import io
import os
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ordi.tasks.generator import (
    EOTask, build_tiles, draw_deadline, groundtrack_lookup, sat_over_target,
)
from ordi.tasks.profiles import PROFILES

FIRMS_AREA_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/{src}/{bbox}/{days}/{start}"

# FRP (MW) at/above which a fire is treated as maximally urgent. VIIRS pixels
# span < 1 MW (smouldering) to hundreds of MW (large active fronts); ~100 MW is
# a strong front, so utility saturates there.
_FRP_FULL_URGENCY_MW = 100.0
# Deadline tightening: the most intense fires get the shortest fraction of the
# nominal wildfire deadline slack (disaster response), calmest get the full slack.
_MIN_DEADLINE_FRAC = 0.4


@dataclass
class FireRequest:
    t_unix: float
    lat: float
    lon: float
    frp_mw: float          # fire radiative power (intensity)
    confidence: str        # 'l' | 'n' | 'h' (VIIRS) or numeric (MODIS)


def _data_dir() -> Path:
    root = Path(os.environ.get("ORDI_DATA_DIR", Path(__file__).resolve().parents[2] / "data"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def _parse_firms_time(acq_date: str, acq_time: str) -> float:
    """acq_date='YYYY-MM-DD', acq_time='HHMM' (or fewer digits) UTC -> unix."""
    hhmm = acq_time.strip().zfill(4)
    dt = datetime.strptime(f"{acq_date} {hhmm}", "%Y-%m-%d %H%M").replace(tzinfo=timezone.utc)
    return dt.timestamp()


def fetch_fire_requests(
    start_date: str,
    days: int = 1,
    bbox: str = "-180,-90,180,90",
    source: str = "VIIRS_SNPP_NRT",
    map_key: Optional[str] = None,
    cache_path: Optional[str] = None,
    refresh: bool = False,
    timeout_s: float = 60.0,
) -> List[FireRequest]:
    """Fetch (and cache) FIRMS active-fire detections as tasking requests.

    start_date : 'YYYY-MM-DD' (UTC); days : 1..10 window from start_date.
    Returns detections sorted by time.
    """
    key = map_key or os.environ.get("FIRMS_MAP_KEY", "")
    if not key:
        raise RuntimeError("FIRMS map key required: set FIRMS_MAP_KEY or pass map_key=.")

    cache = (Path(cache_path) if cache_path
             else _data_dir() / f"firms_{source}_{start_date}_{days}d.csv")
    if cache.exists() and not refresh:
        text = cache.read_text()
    else:
        url = FIRMS_AREA_URL.format(key=key, src=source, bbox=bbox, days=days, start=start_date)
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            text = resp.read().decode("utf-8", errors="replace")
        if not text.lstrip().lower().startswith("latitude"):
            raise RuntimeError(f"FIRMS returned no fire CSV: {text[:200]!r}")
        cache.write_text(text)

    reqs: List[FireRequest] = []
    for row in csv.DictReader(io.StringIO(text)):
        try:
            t = _parse_firms_time(row["acq_date"], row["acq_time"])
            frp = float(row.get("frp") or 0.0)
        except (ValueError, KeyError):
            continue
        reqs.append(FireRequest(
            t_unix=t, lat=float(row["latitude"]), lon=float(row["longitude"]),
            frp_mw=frp, confidence=str(row.get("confidence", "")).strip(),
        ))
    reqs.sort(key=lambda r: r.t_unix)
    return reqs


def _earliest_pass(
    sat_ids, lat, lon, pos_at, fov_range_km, start_rel, sim_duration_s, rng,
    step_s: float = 60.0,
):
    """First (satellite, relative_time) whose groundtrack covers (lat, lon) at or
    after start_rel, scanning forward in step_s increments to the horizon. Ties
    among satellites covering in the same step are broken randomly. Returns
    (None, None) if no pass occurs within the horizon."""
    t = start_rel
    while t < sim_duration_s:
        covering = sat_over_target(sat_ids, lat, lon, lambda sid: pos_at(sid, t), fov_range_km)
        if covering:
            return rng.choice(covering), t
        t += step_s
    return None, None


def _confidence_ok(conf: str) -> bool:
    """Drop low-confidence detections. VIIRS uses l/n/h; MODIS uses 0-100."""
    c = conf.lower()
    if c in ("l", "n", "h"):
        return c != "l"
    try:
        return float(c) >= 30.0
    except ValueError:
        return True


def fire_requests_to_tasks(
    reqs: List[FireRequest],
    sat_ids: List[str],
    sat_groundtrack: Dict[str, List[Tuple[float, float, float]]],
    window_start_unix: float,
    sim_duration_s: float,
    deadline_slack_s: float = 600.0,
    deadline_lognorm_sigma: float = 0.6,
    n_tiles_side: int = 4,
    n_replicas_max: int = 2,
    fov_range_km: float = 600.0,
    seed: int = 0,
) -> List[EOTask]:
    """Convert FIRMS fire detections into wildfire EOTasks.

    Everything the acquisition loader sampled is now derived from the real
    detection: task type is fixed to 'wildfire'; per-task utility scales with
    fire radiative power (FRP); and the deadline tightens with FRP so intense
    fires demand faster delivery. Only the source-satellite choice among those
    in FOV, per-tile jitter, and the log-normal deadline spread remain random.

    A fire is a *persistent target*: it is detected by a different satellite
    (VIIRS), so the request's release_time is when the FIRST constellation
    satellite passes within FOV at or after the detection, and the source is
    that satellite. Fires never overflown within the horizon are dropped.
    """
    import random
    rng = random.Random(seed)
    profile = PROFILES["wildfire"]
    pos_at_abs = groundtrack_lookup(sat_groundtrack)

    tasks: List[EOTask] = []
    task_id = 0
    for r in reqs:
        if not _confidence_ok(r.confidence):
            continue
        detect_rel = r.t_unix - window_start_unix
        if detect_rel >= sim_duration_s:
            continue
        # Earliest covering pass at/after detection: fire is a standing target.
        src, release_rel = _earliest_pass(
            sat_ids, r.lat, r.lon, pos_at_abs, fov_range_km,
            max(0.0, detect_rel), sim_duration_s, rng,
        )
        if src is None:
            continue

        # Real urgency from FRP in [0, 1]: hotter fire -> higher utility, tighter deadline.
        urgency = min(1.0, r.frp_mw / _FRP_FULL_URGENCY_MW)
        util_scale = 1.0 + urgency          # 1x (cool) .. 2x (intense)
        frp_slack = deadline_slack_s * (1.0 - (1.0 - _MIN_DEADLINE_FRAC) * urgency)

        task = EOTask(
            task_id=task_id,
            source_sat=src,
            release_time=release_rel,
            deadline=draw_deadline(profile, release_rel, frp_slack,
                                   deadline_lognorm_sigma, rng),
            task_type="wildfire",
            n_tiles_side=n_tiles_side,
        )
        task.tiles = build_tiles(task_id, profile, n_tiles_side, n_replicas_max,
                                 rng, utility_scale=util_scale)
        tasks.append(task)
        task_id += 1
    return tasks
