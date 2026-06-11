"""
Metric computation from SchedulerResult lists.

Metrics (per the proposal's evaluation plan):
  - deadline_miss_ratio      : fraction of tiles not delivered before D_k
  - delivered_utility        : Σ u_kv * z_kv (expected)
  - partial_coverage         : per-task fraction of tiles with z_kv > 0
  - recovery_latency         : mean L_hat of replanned tiles
  - isl_traffic_bits         : total ISL data transferred
  - downlink_volume_bits     : total downlink data
  - energy_joules            : total helper energy consumed
  - helper_utilization       : fraction of helper compute capacity used
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List

from ordi.scheduler.ordi import SchedulerResult, TileAssignment
from ordi.tasks.generator import EOTask


@dataclass
class EpochMetrics:
    epoch: int
    deadline_miss_ratio: float = 0.0
    delivered_utility: float = 0.0
    partial_coverage: float = 0.0
    recovery_latency: float = 0.0
    isl_traffic_bits: float = 0.0
    downlink_volume_bits: float = 0.0
    energy_joules: float = 0.0
    helper_utilization: float = 0.0
    objective: float = 0.0
    n_tiles_total: int = 0
    n_tiles_feasible: int = 0


def compute_metrics(
    result: SchedulerResult,
    tasks: List[EOTask],
    epoch_start: float,
    sat_compute_capacity: Dict[str, float],   # sat_id → C_i * epoch_length (cycles)
    alpha: float = 0.002,
) -> EpochMetrics:
    m = EpochMetrics(epoch=result.epoch)

    tile_lookup: Dict[tuple, EOTask] = {}
    for task in tasks:
        for tile in task.tiles:
            tile_lookup[(task.task_id, tile.tile_id)] = (task, tile)

    n_tiles = sum(len(t.tiles) for t in tasks)
    n_miss = 0
    utility_sum = 0.0
    partial_coverages = []
    recovery_lats = []

    task_delivered: Dict[int, int] = {}
    task_total: Dict[int, int] = {}

    for assignment in result.assignments:
        key = (assignment.task_id, assignment.tile_id)
        if key not in tile_lookup:
            continue
        task_obj, tile = tile_lookup[key]
        task_total[task_obj.task_id] = task_total.get(task_obj.task_id, 0) + 1

        if assignment.z_kv == 0.0 or math.isinf(assignment.L_hat):
            n_miss += 1
        else:
            utility_sum += tile.utility * assignment.z_kv * math.exp(
                -alpha * assignment.L_hat
            )
            task_delivered[task_obj.task_id] = task_delivered.get(task_obj.task_id, 0) + 1
            recovery_lats.append(assignment.L_hat)

        # ISL traffic from replicas
        for replica in assignment.replicas:
            m.isl_traffic_bits += replica.d_in_bits + replica.d_out_bits
            m.energy_joules += replica.e_compute + replica.e_rx + replica.e_tx

    # Downlink volume: sum of output bits from all assignments with z_kv > 0
    m.downlink_volume_bits = sum(
        a.replicas[0].d_out_bits
        for a in result.assignments
        if a.replicas and a.z_kv > 0
    )

    # Partial coverage per task
    for tid, total in task_total.items():
        delivered = task_delivered.get(tid, 0)
        partial_coverages.append(delivered / total if total > 0 else 0.0)

    m.n_tiles_total = n_tiles
    m.n_tiles_feasible = n_tiles - n_miss
    m.deadline_miss_ratio = n_miss / n_tiles if n_tiles > 0 else 0.0
    m.delivered_utility = utility_sum
    m.partial_coverage = sum(partial_coverages) / len(partial_coverages) if partial_coverages else 0.0
    m.recovery_latency = sum(recovery_lats) / len(recovery_lats) if recovery_lats else 0.0
    m.objective = result.objective

    # Helper utilization: fraction of total constellation compute used
    total_capacity = sum(sat_compute_capacity.values())
    compute_used = sum(
        sum(r.e_compute for r in a.replicas)
        for a in result.assignments
    )
    # Convert energy back to compute cycles: e = P * t; t = cycles / C
    # approximate: utilization = compute_used_energy / (total_capacity_cycles * P_per_cycle)
    m.helper_utilization = min(1.0, compute_used / max(total_capacity * 1e-9, 1e-9))

    return m


def aggregate_metrics(epoch_metrics: List[EpochMetrics]) -> Dict[str, float]:
    """Aggregate per-run metrics into mean plus sample std (<key>_std).

    With one lifetime record per simulation run, the std columns are the
    across-run (seed) dispersion; 0.0 when only a single run is supplied.
    """
    if not epoch_metrics:
        return {}
    n = len(epoch_metrics)
    keys = [
        "deadline_miss_ratio", "delivered_utility", "partial_coverage",
        "recovery_latency", "isl_traffic_bits", "downlink_volume_bits",
        "energy_joules", "helper_utilization", "objective",
    ]
    out: Dict[str, float] = {}
    for k in keys:
        vals = [getattr(m, k) for m in epoch_metrics]
        mean = sum(vals) / n
        var = sum((v - mean) ** 2 for v in vals) / (n - 1) if n > 1 else 0.0
        out[k] = mean
        out[f"{k}_std"] = math.sqrt(var)
    return out
