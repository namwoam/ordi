"""Real EO task loader from Sentinel-2 acquisition metadata (STAC).

Replaces the synthetic Poisson+FOV arrival process with the real acquisition
timeline of Sentinel-2: each STAC item is a real image capture with a real
timestamp and geolocation.  We query the AWS Earth-search STAC API for
``sentinel-2-l2a`` scenes in a time+area window, then convert each acquisition
into an ``EOTask`` that matches ``generate_tasks``' output contract exactly.

An acquisition becomes a task only if one of the constellation's satellites is
within the camera FOV of the acquisition's location at capture time (same FOV
gate as the synthetic generator).  The Sentinel-2 metadata carries no
wildfire/ship/etc. label, and the four ORDI compute profiles are benchmark
derived, so the task *type* (and thus compute profile) is sampled from PROFILES
— an explicit, documented approximation; the acquisition supplies the real
*when* and *where*.
"""

from __future__ import annotations

import json
import os
import random
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ordi.tasks.generator import (
    EOTask, build_tiles, draw_deadline, groundtrack_lookup, sat_over_target,
)
from ordi.tasks.profiles import PROFILES, TASK_TYPES

STAC_SEARCH_URL = "https://earth-search.aws.element84.com/v1/search"


@dataclass
class Acquisition:
    t_unix: float
    lat: float
    lon: float
    scene_id: str


def _data_dir() -> Path:
    root = Path(os.environ.get("ORDI_DATA_DIR", Path(__file__).resolve().parents[2] / "data"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def _parse_iso(dt: str) -> float:
    dt = dt.replace("Z", "+00:00")
    return datetime.fromisoformat(dt).replace(tzinfo=timezone.utc).timestamp()


def fetch_sentinel2_acquisitions(
    datetime_range: str,
    limit: int = 500,
    page_size: int = 100,
    cache_path: Optional[str] = None,
    refresh: bool = False,
    timeout_s: float = 30.0,
) -> List[Acquisition]:
    """Fetch (and cache) Sentinel-2 L2A acquisitions from the Earth-search STAC API.

    datetime_range : STAC datetime interval, e.g. "2026-06-01T00:00:00Z/2026-06-02T00:00:00Z".
    Returns acquisitions sorted by capture time.
    """
    cache = (Path(cache_path) if cache_path
             else _data_dir() / f"sentinel2_{datetime_range.replace('/', '__').replace(':', '')}.json")
    if cache.exists() and not refresh:
        raw = json.loads(cache.read_text())
    else:
        raw = _stac_fetch(datetime_range, limit, page_size, timeout_s)
        cache.write_text(json.dumps(raw))

    acqs: List[Acquisition] = []
    for feat in raw:
        props = feat.get("properties", {})
        dt = props.get("datetime")
        bbox = feat.get("bbox")
        if not dt or not bbox or len(bbox) < 4:
            continue
        lat = (bbox[1] + bbox[3]) / 2.0
        lon = (bbox[0] + bbox[2]) / 2.0
        acqs.append(Acquisition(_parse_iso(dt), lat, lon, feat.get("id", "")))
    acqs.sort(key=lambda a: a.t_unix)
    return acqs


def _stac_fetch(datetime_range: str, limit: int, page_size: int, timeout_s: float) -> list:
    """Paginate the STAC search endpoint, returning raw feature dicts."""
    features: list = []
    body = {
        "collections": ["sentinel-2-l2a"],
        "datetime": datetime_range,
        "limit": min(page_size, limit),
    }
    url = STAC_SEARCH_URL
    while len(features) < limit:
        req = urllib.request.Request(
            url, data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            page = json.loads(resp.read().decode("utf-8"))
        feats = page.get("features", [])
        if not feats:
            break
        features.extend(feats)
        next_link = next((l for l in page.get("links", []) if l.get("rel") == "next"), None)
        if not next_link:
            break
        url = next_link.get("href", STAC_SEARCH_URL)
        body = next_link.get("body", body)
    return features[:limit]


def acquisitions_to_tasks(
    acqs: List[Acquisition],
    sat_ids: List[str],
    sat_groundtrack: Dict[str, List[Tuple[float, float, float]]],
    window_start_unix: float,
    sim_duration_s: float,
    deadline_slack_s: float = 600.0,
    deadline_lognorm_sigma: float = 0.6,
    n_tiles_side: int = 4,
    task_type_weights: Optional[Dict[str, float]] = None,
    n_replicas_max: int = 2,
    fov_range_km: float = 600.0,
    seed: int = 0,
) -> List[EOTask]:
    """Convert real acquisitions into EOTasks matching generate_tasks' contract.

    window_start_unix : the unix time mapped to sim t=0; acquisition times become
        relative release times ``t_unix - window_start_unix`` and must fall within
        [0, sim_duration_s).  The satellite groundtracks must be sampled over the
        same absolute window so the FOV gate resolves positions correctly.
    """
    rng = random.Random(seed)
    if task_type_weights is None:
        task_type_weights = {t: 1.0 for t in TASK_TYPES}
    types = list(task_type_weights.keys())
    weights = [task_type_weights[t] for t in types]

    pos_at_abs = groundtrack_lookup(sat_groundtrack)

    tasks: List[EOTask] = []
    task_id = 0
    for acq in acqs:
        t_rel = acq.t_unix - window_start_unix
        if t_rel < 0 or t_rel >= sim_duration_s:
            continue
        # FOV gate: which satellite is over this acquisition at capture time?
        # Groundtracks are sampled starting at window_start_unix, so the lookup
        # (which indexes from track sample 0) takes the relative time t_rel.
        visible = sat_over_target(
            sat_ids, acq.lat, acq.lon,
            lambda sid: pos_at_abs(sid, t_rel), fov_range_km,
        )
        if not visible:
            continue
        src = rng.choice(visible)

        ttype = rng.choices(types, weights=weights, k=1)[0]
        profile = PROFILES[ttype]
        task = EOTask(
            task_id=task_id,
            source_sat=src,
            release_time=t_rel,
            deadline=draw_deadline(profile, t_rel, deadline_slack_s,
                                   deadline_lognorm_sigma, rng),
            task_type=ttype,
            n_tiles_side=n_tiles_side,
        )
        task.tiles = build_tiles(task_id, profile, n_tiles_side, n_replicas_max, rng)
        tasks.append(task)
        task_id += 1
    return tasks
