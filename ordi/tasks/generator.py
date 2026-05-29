"""
EO task and tile generator.

Task k:
  source satellite s_k — the satellite whose camera covers target_k at r_k
  release time r_k     (Poisson arrivals over ground targets)
  deadline D_k         = r_k + deadline_slack_s
  tile set V_k         = grid of n_tiles_side × n_tiles_side tiles

When sat_groundtrack is supplied (FOV-aware mode), each task event samples a
random ground target and picks a satellite currently within fov_range_km of
it as the source.  This replaces the original uniform-random source assignment
with physically motivated imaging geometry.
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ordi.tasks.profiles import TileProfile, PROFILES, TASK_TYPES


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two (lat, lon) points in degrees."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1))
         * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return 2.0 * R * math.asin(math.sqrt(max(0.0, min(1.0, a))))


@dataclass
class Tile:
    task_id: int
    tile_id: int            # local index within task
    profile: TileProfile
    d_in_bits: float        # may differ from profile if spatial variation applied
    d_out_bits: float
    compute_ops: float
    utility: float          # u_kv
    row: int                # grid position
    col: int
    n_replicas_max: int = 2


@dataclass
class EOTask:
    task_id: int
    source_sat: str
    release_time: float     # seconds from sim start
    deadline: float         # absolute deadline (seconds from sim start)
    task_type: str
    tiles: List[Tile] = field(default_factory=list)
    n_tiles_side: int = 4   # n_tiles_side × n_tiles_side grid

    @property
    def n_tiles(self) -> int:
        return len(self.tiles)

    @property
    def deadline_slack(self) -> float:
        return self.deadline - self.release_time


def generate_tasks(
    sat_ids: List[str],
    sim_duration_s: float,
    arrival_rate_per_orbit: float = 3.0,
    orbit_period_s: float = 5760.0,    # ~96 min LEO orbit at 550 km
    deadline_slack_s: float = 300.0,
    n_tiles_side: int = 4,
    task_type_weights: Dict[str, float] = None,
    n_replicas_max: int = 2,
    seed: int = 0,
    # ── FOV-aware mode (optional) ──────────────────────────────────────────
    sat_groundtrack: Optional[Dict[str, List[Tuple[float, float, float]]]] = None,
    # {sat_id: [(t_s, lat_deg, lon_deg), ...]} sampled at regular intervals
    ground_targets: Optional[List[Tuple[float, float]]] = None,
    # [(lat_deg, lon_deg), ...] imaging targets
    fov_range_km: float = 500.0,
    # max ground distance for the camera footprint at 550 km altitude
    # ~500 km corresponds to ±arctan(500/550) ≈ 42° off-nadir (wide-field)
) -> List[EOTask]:
    """
    Generate EO tasks via a Poisson arrival process.

    arrival_rate_per_orbit : expected task events per orbit

    FOV-aware mode (when sat_groundtrack and ground_targets are given):
      Each task event first picks a random ground target, then finds all
      satellites with their sub-satellite point within fov_range_km of the
      target at time t.  If none are in range the event is skipped; otherwise
      one of the visible satellites is chosen as the source.  This ensures
      that every task corresponds to a satellite that is physically overhead
      the imaging target at the moment of data capture — the physically
      correct model for EO scheduling.
    """
    rng = random.Random(seed)

    if task_type_weights is None:
        task_type_weights = {t: 1.0 for t in TASK_TYPES}

    types = list(task_type_weights.keys())
    weights = [task_type_weights[t] for t in types]

    lam = arrival_rate_per_orbit / orbit_period_s  # tasks per second

    # Build a time→index lookup for the groundtrack (O(1) per event)
    gt_dt: float = 0.0
    gt_n: int = 0
    if sat_groundtrack:
        sample_ts = next(iter(sat_groundtrack.values()))
        if len(sample_ts) >= 2:
            gt_dt = sample_ts[1][0] - sample_ts[0][0]
            gt_n = len(sample_ts)

    def _sat_pos_at(sat_id: str, t: float) -> Tuple[float, float]:
        """Nearest-sample lat/lon for sat at time t."""
        track = sat_groundtrack[sat_id]
        idx = min(max(0, int(round(t / gt_dt))), gt_n - 1)
        return track[idx][1], track[idx][2]

    tasks: List[EOTask] = []
    t = 0.0
    task_id = 0

    while t < sim_duration_s:
        # exponential inter-arrival
        inter = rng.expovariate(lam)
        t += inter
        if t >= sim_duration_s:
            break

        # ── source satellite selection ──────────────────────────────────────
        if sat_groundtrack and ground_targets:
            # FOV-aware: find every satellite that is over at least one target.
            # Scanning all targets for each satellite is fast (100 targets × 24
            # sats = 2400 checks) and avoids the "pick random target first" bias
            # that skips ~94% of events when per-target coverage is ~5%.
            visible = []
            for sid in sat_ids:
                s_lat, s_lon = _sat_pos_at(sid, t)
                if any(_haversine_km(g_lat, g_lon, s_lat, s_lon) <= fov_range_km
                       for g_lat, g_lon in ground_targets):
                    visible.append(sid)
            if not visible:
                # No satellite over any target; skip event (very rare with
                # dense targets, ~0% of the time per the coverage analysis).
                continue
            src = rng.choice(visible)
        else:
            src = rng.choice(sat_ids)

        ttype = rng.choices(types, weights=weights, k=1)[0]
        profile = PROFILES[ttype]

        # Vary deadline slightly per-task
        slack = deadline_slack_s * rng.uniform(0.8, 1.2)
        task = EOTask(
            task_id=task_id,
            source_sat=src,
            release_time=t,
            deadline=t + slack,
            task_type=ttype,
            n_tiles_side=n_tiles_side,
        )

        # Generate tiles with spatial utility variation
        # Center tiles (closer to image center) get slightly higher utility
        tiles = []
        center = (n_tiles_side - 1) / 2.0
        for row in range(n_tiles_side):
            for col in range(n_tiles_side):
                dist_from_center = math.sqrt((row - center) ** 2 + (col - center) ** 2)
                max_dist = math.sqrt(2) * center
                spatial_weight = 1.0 + 0.3 * (1.0 - dist_from_center / max(max_dist, 1e-6))
                u = profile.base_utility * spatial_weight * rng.uniform(0.9, 1.1)

                tile = Tile(
                    task_id=task_id,
                    tile_id=len(tiles),
                    profile=profile,
                    d_in_bits=profile.d_in_bits * rng.uniform(0.9, 1.1),
                    d_out_bits=profile.d_out_bits * rng.uniform(0.9, 1.1),
                    compute_ops=profile.compute_ops * rng.uniform(0.85, 1.15),
                    utility=u,
                    row=row,
                    col=col,
                    n_replicas_max=n_replicas_max,
                )
                tiles.append(tile)

        task.tiles = tiles
        tasks.append(task)
        task_id += 1

    return tasks
