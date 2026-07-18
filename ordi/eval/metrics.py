"""
Metric computation from SchedulerResult lists.

Metrics (per the proposal's evaluation plan):
  - deadline_miss_ratio      : fraction of tiles not delivered before D_k
  - delivered_utility        : Σ u_kv * z_kv (expected)
  - partial_coverage         : per-task fraction of tiles with z_kv > 0
  - recovery_latency         : mean L_hat of replanned tiles
  - isl_traffic_bits         : total ISL data transferred
  - downlink_volume_bits     : total downlink data
  - energy_joules            : onboard compute, ISL, and ground-downlink energy
  - helper_utilization       : fraction of helper compute capacity used
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

from ordi.scheduler.ordi import SchedulerResult, TileAssignment
from ordi.sim.reliability import ReliabilityModel
from ordi.tasks.generator import EOTask
from ordi.sim.satellite import DEFAULT_COMMS_POWER_W, DEFAULT_DOWNLINK_RATE_BPS


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
    n_replicas_avg: float = 0.0
    n_tiles_total: int = 0
    n_tiles_feasible: int = 0
    # Monte Carlo realized outcomes (sampled from the reliability model rather
    # than scored on the modeled expectation z_kv). Populated by
    # compute_realized_metrics; left at 0 when realized scoring is not run.
    realized_miss_ratio: float = 0.0
    realized_utility: float = 0.0
    realized_coverage: float = 0.0


def compute_metrics(
    result: SchedulerResult,
    tasks: List[EOTask],
    epoch_start: float,
    sat_compute_capacity: Dict[str, float],   # sat_id → FLOP budget (C_i·epoch length, summed over the horizon for lifetime records)
    alpha: float = 0.002,
    downlink_power_w: float = DEFAULT_COMMS_POWER_W,
    downlink_rate_bps: float = DEFAULT_DOWNLINK_RATE_BPS,
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

        # Compute and ISL energy from replicas.
        for replica in assignment.replicas:
            m.isl_traffic_bits += replica.d_in_bits + replica.d_out_bits
            m.energy_joules += replica.e_compute + replica.e_rx + replica.e_tx

        # Every delivered tile incurs spacecraft-to-ground transmit energy.
        # Normal inference schedulers downlink the compact result; direct and
        # compression baselines record their raw/compressed size explicitly.
        if assignment.z_kv > 0.0 and not math.isinf(assignment.L_hat):
            downlink_bits = (tile.d_out_bits if assignment.downlink_bits is None
                             else assignment.downlink_bits)
            m.downlink_volume_bits += downlink_bits
            m.energy_joules += downlink_power_w * downlink_bits / downlink_rate_bps

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
    m.n_replicas_avg = (sum(len(a.replicas) for a in result.assignments) / n_tiles
                        if n_tiles > 0 else 0.0)

    # Helper utilization: fraction of the constellation's compute budget the
    # scheduled work consumes. Numerator is actual FLOPs (each replica
    # runs its tile's compute_ops on the helper); denominator is the summed
    # per-satellite capacity C_i·epoch_length passed by the caller (horizon-wide
    # when the assignment set is a lifetime record). Both sides are in FLOPs,
    # so the ratio is dimensionless.
    total_capacity = sum(sat_compute_capacity.values())
    compute_used = sum(
        tile_lookup[(a.task_id, a.tile_id)][1].compute_ops * len(a.replicas)
        for a in result.assignments
        if (a.task_id, a.tile_id) in tile_lookup
    )
    m.helper_utilization = (min(1.0, compute_used / total_capacity)
                            if total_capacity > 0 else 0.0)

    return m


def _replica_components(replica, source: str) -> Tuple[Tuple[str, ...], Tuple[Tuple[str, str], ...], str]:
    """Decompose a replica into the (nodes, isl_links, downlink_node) whose
    joint survival determines whether the replica delivers, mirroring the terms
    of ReliabilityModel.replica_success_prob.

    A replica succeeds iff the helper node, the source→helper ISL, the
    helper→aggregator ISL, and the aggregator's downlink all survive. (The
    source node is shared across all replicas of a tile and sampled once at the
    tile level, so it is excluded here.) Self-processing replicas where the
    helper is the source carry no source→helper hop; helper-as-aggregator
    replicas carry no helper→aggregator hop.

    The component set mirrors replica_success_prob exactly (helper node only,
    not the aggregator node), so under independent draws the Monte Carlo miss
    ratio converges to the modeled z_kv; the only divergence comes from draws
    shared across replicas, which is the correlation effect this layer exists
    to expose.
    """
    helper = replica.helper
    agg = replica.aggregator
    nodes = {helper}
    isl_links: Set[Tuple[str, str]] = set()
    if helper != source:
        isl_links.add((min(source, helper), max(source, helper)))
    if helper != agg:
        isl_links.add((min(helper, agg), max(helper, agg)))
    return tuple(sorted(nodes)), tuple(sorted(isl_links)), agg


def compute_realized_metrics(
    result: SchedulerResult,
    tasks: List[EOTask],
    reliability: ReliabilityModel,
    alpha: float = 0.002,
    n_trials: int = 200,
    seed: int = 0,
) -> EpochMetrics:
    """Monte Carlo realized-outcome scoring.

    Unlike compute_metrics (which scores the scheduler's *modeled* expectation
    z_kv directly), this samples per-trial Bernoulli outcomes for every node,
    ISL link, and downlink from the reliability model's π values, using the
    SAME random draw for a component shared by multiple replicas of a tile.
    A replica delivers iff all its components survive; a tile is delivered iff
    any of its replicas delivers. The realized miss ratio, delivered utility,
    and coverage are averaged over trials and written to the realized_* fields,
    leaving the modeled deadline_miss_ratio/delivered_utility intact so the two
    can be compared directly. Deterministic cost metrics (ISL, downlink,
    energy, replicas) come from compute_metrics.

    Shared draws across replicas make correlated structural failures (a plane
    or a shared aggregator taking down several replicas at once) show up in the
    realized numbers instead of being hidden by the independence assumption.
    """
    # Deterministic cost metrics + structure come from the modeled pass.
    base = compute_metrics(result, tasks, 0.0, {}, alpha)

    tile_lookup: Dict[tuple, tuple] = {}
    for task in tasks:
        for tile in task.tiles:
            tile_lookup[(task.task_id, tile.tile_id)] = (task, tile)

    n_tiles = base.n_tiles_total
    if n_tiles == 0:
        return base

    scored = []  # (task_id, tile, utility_weight, [replica_component_tuples], source)
    for a in result.assignments:
        key = (a.task_id, a.tile_id)
        if key not in tile_lookup:
            continue
        task_obj, tile = tile_lookup[key]
        if not a.replicas or math.isinf(a.L_hat):
            # B1-style no-replica assignments: fall back to a single synthetic
            # downlink replica off the primary aggregator so they are still
            # sampled rather than silently treated as always-delivered.
            if a.z_kv > 0 and not math.isinf(a.L_hat) and a.primary_aggregator:
                scored.append((task_obj.task_id, tile, a.L_hat,
                               [(("__none__",), (), a.primary_aggregator)],
                               task_obj.source_sat))
            else:
                scored.append((task_obj.task_id, tile, a.L_hat, [], task_obj.source_sat))
            continue
        comps = [_replica_components(r, task_obj.source_sat) for r in a.replicas]
        scored.append((task_obj.task_id, tile, a.L_hat, comps, task_obj.source_sat))

    rng = random.Random(seed)
    miss_trials = []
    util_trials = []
    cvg_trials = []

    for _ in range(n_trials):
        node_alive: Dict[str, bool] = {}
        link_alive: Dict[Tuple[str, str], bool] = {}
        down_alive: Dict[str, bool] = {}

        def node_ok(nid: str) -> bool:
            if nid == "__none__":
                return True
            if nid not in node_alive:
                node_alive[nid] = rng.random() < reliability.node_pi(nid)
            return node_alive[nid]

        def link_ok(pair: Tuple[str, str]) -> bool:
            if pair not in link_alive:
                link_alive[pair] = rng.random() < reliability.link_pi(pair[0], pair[1], "isl")
            return link_alive[pair]

        def down_ok(nid: str) -> bool:
            if nid not in down_alive:
                down_alive[nid] = rng.random() < reliability.downlink_pi(nid)
            return down_alive[nid]

        n_miss = 0
        util = 0.0
        task_delivered: Dict[int, int] = {}
        task_total: Dict[int, int] = {}

        for tid, tile, L_hat, comps, src in scored:
            task_total[tid] = task_total.get(tid, 0) + 1
            delivered = False
            if comps and node_ok(src):
                for nodes, isl_links, agg in comps:
                    if (all(node_ok(n) for n in nodes)
                            and all(link_ok(l) for l in isl_links)
                            and down_ok(agg)):
                        delivered = True
                        break
            if delivered:
                util += tile.utility * math.exp(-alpha * L_hat)
                task_delivered[tid] = task_delivered.get(tid, 0) + 1
            else:
                n_miss += 1

        miss_trials.append(n_miss / n_tiles)
        util_trials.append(util)
        cov = [task_delivered.get(t, 0) / tot for t, tot in task_total.items() if tot]
        cvg_trials.append(sum(cov) / len(cov) if cov else 0.0)

    base.realized_miss_ratio = sum(miss_trials) / n_trials
    base.realized_utility = sum(util_trials) / n_trials
    base.realized_coverage = sum(cvg_trials) / n_trials
    return base


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
        "energy_joules", "helper_utilization", "objective", "n_replicas_avg",
        "realized_miss_ratio", "realized_utility", "realized_coverage",
    ]
    out: Dict[str, float] = {}
    for k in keys:
        vals = [getattr(m, k) for m in epoch_metrics]
        mean = sum(vals) / n
        var = sum((v - mean) ** 2 for v in vals) / (n - 1) if n > 1 else 0.0
        out[k] = mean
        out[f"{k}_std"] = math.sqrt(var)
    return out
