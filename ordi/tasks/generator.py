"""
EO task and tile generator.

Task k:
  source satellite s_k (uniformly sampled from constellation)
  release time r_k     (Poisson arrivals)
  deadline D_k         = r_k + deadline_slack_s
  tile set V_k         = grid of n_tiles_side × n_tiles_side tiles

Each tile v ∈ V_k inherits the parent task's TileProfile with slight
spatial variation in utility (center tiles often higher priority).
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import List, Dict

from ordi.tasks.profiles import TileProfile, PROFILES, TASK_TYPES


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
    orbit_period_s: float = 5400.0,    # ~90 min LEO orbit
    deadline_slack_s: float = 300.0,   # 5-minute deadline
    n_tiles_side: int = 4,
    task_type_weights: Dict[str, float] = None,
    n_replicas_max: int = 2,
    seed: int = 0,
) -> List[EOTask]:
    """
    Generate EO tasks via a Poisson arrival process.

    arrival_rate_per_orbit : expected tasks arriving per orbit
    """
    rng = random.Random(seed)

    if task_type_weights is None:
        task_type_weights = {t: 1.0 for t in TASK_TYPES}

    types = list(task_type_weights.keys())
    weights = [task_type_weights[t] for t in types]

    lam = arrival_rate_per_orbit / orbit_period_s  # tasks per second

    tasks: List[EOTask] = []
    t = 0.0
    task_id = 0

    while t < sim_duration_s:
        # exponential inter-arrival
        inter = rng.expovariate(lam)
        t += inter
        if t >= sim_duration_s:
            break

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
