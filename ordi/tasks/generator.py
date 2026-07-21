"""
EO task and tile generator.

Task k:
  source satellite s_k — the satellite whose camera covers target_k at r_k
  release time r_k     (Poisson arrivals over ground targets)
  deadline D_k         = r_k + deadline_slack_s
  tile set V_k         = grid of n_tiles_side × n_tiles_side tiles

When sat_groundtrack is supplied, fixed-target mode gates an event against the
configured access radius. Groundtrack mode instead constructs a feasible
near-nadir acquisition at a sampled spacecraft subpoint, avoiding confusion
between pointing access and the much smaller instantaneous camera footprint.
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
    burst_id: int = -1      # parent arrival; clustered requests share this ID

    @property
    def n_tiles(self) -> int:
        return len(self.tiles)

    @property
    def deadline_slack(self) -> float:
        return self.deadline - self.release_time


_DEADLINE_SCALE_BASE = 600.0  # reference scale for deadline_slack_s


def draw_deadline(
    profile: TileProfile,
    release_time: float,
    deadline_slack_s: float,
    deadline_lognorm_sigma: float,
    rng: random.Random,
) -> float:
    """Absolute deadline for a task: release + a log-normal draw around the
    type-specific median scaled by deadline_slack_s.  Shared by the synthetic
    Poisson generator and the real-acquisition loader so both use one model."""
    type_median = profile.deadline_median_s * (deadline_slack_s / _DEADLINE_SCALE_BASE)
    if deadline_lognorm_sigma > 0.0:
        slack = max(type_median * math.exp(rng.gauss(0.0, deadline_lognorm_sigma)), 60.0)
    else:
        slack = type_median
    return release_time + slack


def build_tiles(
    task_id: int,
    profile: TileProfile,
    n_tiles_side: int,
    n_replicas_max: int,
    rng: random.Random,
    utility_scale: float = 1.0,
    input_band_count: Optional[int] = None,
) -> List[Tile]:
    """Build the n_tiles_side × n_tiles_side tile grid for one task.

    Center tiles get slightly higher utility; per-tile data sizes and compute
    demand are perturbed ±10–15% to model spatial variation.  Shared by the
    synthetic generator and the real-acquisition loader.

    utility_scale multiplies every tile's base utility, letting a real-request
    loader fold measured event importance (e.g. fire radiative power) into the
    priority of the task.
    """
    tiles: List[Tile] = []
    center = (n_tiles_side - 1) / 2.0
    for row in range(n_tiles_side):
        for col in range(n_tiles_side):
            dist_from_center = math.sqrt((row - center) ** 2 + (col - center) ** 2)
            max_dist = math.sqrt(2) * center
            spatial_weight = 1.0 + 0.3 * (1.0 - dist_from_center / max(max_dist, 1e-6))
            u = profile.base_utility * spatial_weight * utility_scale * rng.uniform(0.9, 1.1)

            band_scale = (
                input_band_count / profile.input_bands
                if input_band_count is not None else 1.0
            )
            tiles.append(Tile(
                task_id=task_id,
                tile_id=len(tiles),
                profile=profile,
                d_in_bits=(profile.d_in_bits * band_scale
                           * rng.uniform(0.9, 1.1)),
                d_out_bits=profile.d_out_bits * rng.uniform(0.9, 1.1),
                compute_ops=profile.compute_ops * rng.uniform(0.85, 1.15),
                utility=u,
                row=row,
                col=col,
                n_replicas_max=n_replicas_max,
            ))
    return tiles


def sat_over_target(
    sat_ids: List[str],
    lat: float,
    lon: float,
    pos_at,
    fov_range_km: float,
) -> List[str]:
    """Satellites whose sub-satellite point is within fov_range_km of (lat, lon).
    pos_at(sat_id) -> (lat_deg, lon_deg) resolves the satellite's position at the
    caller's time of interest.  Shared FOV gate for both task sources."""
    visible = []
    for sid in sat_ids:
        s_lat, s_lon = pos_at(sid)
        if _haversine_km(lat, lon, s_lat, s_lon) <= fov_range_km:
            visible.append(sid)
    return visible


def groundtrack_lookup(sat_groundtrack: Dict[str, List[Tuple[float, float, float]]]):
    """Return a nearest-sample position lookup pos_at(sat_id, t) -> (lat, lon)
    over a {sat_id: [(t, lat, lon), ...]} groundtrack sampled at regular steps."""
    sample_ts = next(iter(sat_groundtrack.values()))
    gt_dt = sample_ts[1][0] - sample_ts[0][0] if len(sample_ts) >= 2 else 1.0
    gt_n = len(sample_ts)

    def pos_at(sat_id: str, t: float) -> Tuple[float, float]:
        track = sat_groundtrack[sat_id]
        idx = min(max(0, int(round(t / gt_dt))), gt_n - 1)
        return track[idx][1], track[idx][2]

    return pos_at


def generate_tasks(
    sat_ids: List[str],
    sim_duration_s: float,
    arrival_rate_per_orbit: float = 3.0,
    orbit_period_s: float = 5760.0,    # ~96 min LEO orbit at 550 km
    deadline_slack_s: float = 600.0,
    deadline_lognorm_sigma: float = 0.6,
    n_tiles_side: int = 4,
    task_type_weights: Dict[str, float] = None,
    n_replicas_max: int = 2,
    seed: int = 0,
    burst_probability: float = 0.0,
    burst_size_range: Tuple[int, int] = (2, 4),
    burst_window_s: float = 60.0,
    # ── FOV-aware mode (optional) ──────────────────────────────────────────
    sat_groundtrack: Optional[Dict[str, List[Tuple[float, float, float]]]] = None,
    # {sat_id: [(t_s, lat_deg, lon_deg), ...]} sampled at regular intervals
    ground_targets: Optional[List[Tuple[float, float]]] = None,
    # [(lat_deg, lon_deg), ...] imaging targets
    fov_range_km: float = 500.0,
    # max ground distance for the camera footprint at 550 km altitude
    # ~500 km corresponds to ±arctan(500/550) ≈ 42° off-nadir (wide-field)
    acquisition_mode: str = "targets",
    # "targets" gates against fixed targets. "groundtrack" constructs a
    # feasible near-nadir ROI at a sampled spacecraft subpoint.
    input_band_counts: Optional[Dict[str, int]] = None,
) -> List[EOTask]:
    """
    Generate EO tasks via a Poisson arrival process.

    arrival_rate_per_orbit : expected task events per orbit

    deadline_slack_s : global deadline scale (reference = 600 s).  Each task
        type has its own median deadline (from TileProfile.deadline_median_s)
        which is multiplied by deadline_slack_s / _DEADLINE_SCALE_BASE.  Set to
        600 to use profile medians as-is.

    deadline_lognorm_sigma : log-space std-dev for per-task deadline sampling.
        σ=0.6 gives a realistic spread (5th/95th percentile ≈ 0.30× / 3.32×
        the median), consistent with cluster-job deadline distributions
        (Lublin & Feitelson 2003; Google Borg trace analysis).
        Set to 0.0 for deterministic (fixed) deadlines.

    burst_probability : probability that one parent arrival creates a cluster
        of requests.  Every request in a cluster uses the same source and task
        type and is released within ``burst_window_s``.  The parent-event rate
        is reduced so ``arrival_rate_per_orbit`` remains the expected REQUEST
        rate rather than becoming the cluster rate.

    burst_size_range : inclusive minimum/maximum requests in a burst.

    burst_window_s : maximum release-time spread within one hot-source burst.

    FOV-aware mode (when sat_groundtrack and ground_targets are given):
      Each task event first picks a random ground target, then finds all
      satellites with their sub-satellite point within fov_range_km of the
      target at time t.  If none are in range the event is skipped; otherwise
      one of the visible satellites is chosen as the source.  This ensures
      that every task corresponds to a satellite that is physically overhead
      the imaging target at the moment of data capture — the physically
      correct model for EO scheduling.

    Groundtrack acquisition mode (when sat_groundtrack is given):
      Each event selects a spacecraft and defines its ROI at that spacecraft's
      sampled subpoint. ``input_band_counts`` optionally scales only the input
      tensor volume relative to each profile's baseline channel count.
    """
    rng = random.Random(seed)
    if acquisition_mode not in {"targets", "groundtrack"}:
        raise ValueError("acquisition_mode must be 'targets' or 'groundtrack'")
    if input_band_counts is not None:
        invalid = {
            name: count for name, count in input_band_counts.items()
            if name not in PROFILES or count <= 0
        }
        if invalid:
            raise ValueError(f"invalid input band counts: {invalid}")

    if task_type_weights is None:
        task_type_weights = {t: 1.0 for t in TASK_TYPES}

    types = list(task_type_weights.keys())
    weights = [task_type_weights[t] for t in types]

    burst_min, burst_max = burst_size_range
    if not 0.0 <= burst_probability <= 1.0:
        raise ValueError("burst_probability must be between 0 and 1")
    if burst_min < 2 or burst_max < burst_min:
        raise ValueError("burst_size_range must satisfy 2 <= min <= max")
    if burst_window_s < 0.0:
        raise ValueError("burst_window_s must be non-negative")
    mean_burst_size = (burst_min + burst_max) / 2.0
    mean_cluster_size = 1.0 + burst_probability * (mean_burst_size - 1.0)
    # Parent clusters per second.  Dividing by mean cluster size preserves the
    # requested mean number of individual tasks per orbit.
    lam = arrival_rate_per_orbit / orbit_period_s / mean_cluster_size

    # Build a time→index lookup for the groundtrack (O(1) per event)
    pos_at = groundtrack_lookup(sat_groundtrack) if sat_groundtrack else None

    tasks: List[EOTask] = []
    t = 0.0
    task_id = 0
    burst_id = 0

    while t < sim_duration_s:
        # exponential inter-arrival
        inter = rng.expovariate(lam)
        t += inter
        if t >= sim_duration_s:
            break

        # ── source satellite selection ──────────────────────────────────────
        if sat_groundtrack and acquisition_mode == "groundtrack":
            # Define the acquired ROI at this spacecraft's sampled subpoint.
            # It is therefore feasible by construction, without treating a
            # wide off-nadir access envelope as the sensor footprint.
            src = rng.choice(sat_ids)
        elif sat_groundtrack and ground_targets:
            # FOV-aware: find every satellite that is over at least one target.
            # Scanning all targets for each satellite is fast (100 targets × 24
            # sats = 2400 checks) and avoids the "pick random target first" bias
            # that skips ~94% of events when per-target coverage is ~5%.
            visible = []
            for sid in sat_ids:
                s_lat, s_lon = pos_at(sid, t)
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

        cluster_size = (rng.randint(burst_min, burst_max)
                        if rng.random() < burst_probability else 1)
        release_times = [t]
        release_times.extend(
            t + rng.uniform(0.0, burst_window_s)
            for _ in range(cluster_size - 1)
        )
        for release_time in sorted(release_times):
            if release_time >= sim_duration_s:
                continue
            task = EOTask(
                task_id=task_id,
                source_sat=src,
                release_time=release_time,
                deadline=draw_deadline(
                    profile, release_time, deadline_slack_s,
                    deadline_lognorm_sigma, rng,
                ),
                task_type=ttype,
                n_tiles_side=n_tiles_side,
                burst_id=burst_id,
            )
            task.tiles = build_tiles(
                task_id, profile, n_tiles_side, n_replicas_max, rng,
                input_band_count=(input_band_counts or {}).get(ttype),
            )
            tasks.append(task)
            task_id += 1
        burst_id += 1

    return tasks
