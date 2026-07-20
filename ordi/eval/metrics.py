"""
Metric computation from algorithm ``Decision`` objects.

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

from ordi.algorithms import Decision
from ordi.sim.reliability import ReliabilityModel
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
    protocol_messages: float = 0.0
    control_traffic_bits: float = 0.0
    delivery_latency_p50_s: float = 0.0
    delivery_latency_p95_s: float = 0.0
    isl_traffic_bits_per_delivered_tile: float = 0.0
    control_traffic_bits_per_delivered_tile: float = 0.0
    protocol_messages_per_delivered_tile: float = 0.0
    energy_j_per_delivered_tile: float = 0.0
    downlink_bits_per_delivered_tile: float = 0.0
    control_traffic_ratio: float = 0.0
    active_helper_fraction: float = 0.0
    compute_load_balance: float = 0.0
    helper_request_count: float = 0.0
    helper_acceptance_ratio: float = 1.0
    state_age_mean_s: float = 0.0
    state_age_p95_s: float = 0.0
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


def _percentile(values, quantile):
    """Linearly interpolated percentile without an external dependency."""
    if not values:
        return 0.0
    ordered = sorted(values)
    position = max(0.0, min(1.0, quantile)) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def compute_metrics(
    result: Decision,
    tasks: List[EOTask],
    epoch_start: float,
    sat_compute_capacity: Dict[str, float],   # sat_id → FLOP budget (C_i·epoch length, summed over the horizon for lifetime records)
    alpha: float = 0.002,
    physical_energy_j: float = 0.0,
) -> EpochMetrics:
    m = EpochMetrics(epoch=result.epoch)
    # Space-segment energy is measured from workload power nodes integrated by
    # Basilisk. Ground inference is outside Basilisk, so its H100 profile adds
    # the corresponding active energy below.
    m.energy_joules = max(0.0, float(physical_energy_j))

    tile_lookup: Dict[tuple, EOTask] = {}
    for task in tasks:
        for tile in task.tiles:
            tile_lookup[(task.task_id, tile.tile_id)] = (task, tile)

    n_tiles = sum(len(t.tiles) for t in tasks)
    n_miss = 0
    utility_sum = 0.0
    partial_coverages = []
    recovery_lats = []
    state_ages = []
    helper_requests = 0
    helper_accepts = 0

    task_delivered: Dict[int, int] = {}
    task_total: Dict[int, int] = {}

    for assignment in result.assignments:
        key = (assignment.task_id, assignment.tile_id)
        if key not in tile_lookup:
            continue
        task_obj, tile = tile_lookup[key]
        m.energy_joules += max(0.0, float(
            assignment.metadata.get("ground_compute_energy_j", 0.0)
        ))
        task_total[task_obj.task_id] = task_total.get(task_obj.task_id, 0) + 1

        reliability = float(assignment.metadata.get(
            "reliability", assignment.metadata.get("reconstruction_probability", 0.0)
        ))
        latency = float(assignment.metadata.get("latency", math.inf))
        if "max_state_age_s" in assignment.metadata:
            state_ages.append(float(assignment.metadata["max_state_age_s"]))
        if reliability <= 0.0 or math.isinf(latency):
            n_miss += 1
        else:
            utility_sum += tile.utility * reliability * math.exp(
                -alpha * latency
            )
            task_delivered[task_obj.task_id] = task_delivered.get(task_obj.task_id, 0) + 1
            recovery_lats.append(latency)

        # The policy schema records placement, while Basilisk owns physical
        # energy evolution. Traffic is derived from the selected paths and the
        # optional energy estimate is useful only as an objective diagnostic.
        for index, (helper, aggregator) in enumerate(
                zip(assignment.helpers, assignment.aggregators)):
            input_fraction = (assignment.input_fractions[index]
                              if index < len(assignment.input_fractions) else 1.0)
            output_fraction = (assignment.output_fractions[index]
                               if index < len(assignment.output_fractions) else 1.0)
            protocol_header = float(
                assignment.metadata.get("protocol_header_bits", 0.0)
            )
            if index < len(assignment.routes):
                route_in, route_out, route_down = assignment.routes[index]
                m.isl_traffic_bits += (tile.d_in_bits * input_fraction
                                       * max(0, len(route_in) - 1))
                m.isl_traffic_bits += (tile.d_out_bits * output_fraction
                                       * max(0, len(route_out) - 1))
                m.isl_traffic_bits += (tile.d_out_bits * output_fraction
                                       * max(0, len(route_down) - 2))
                control_hops = (
                    max(0, len(route_in) - 1)
                    + max(0, len(route_out) - 1)
                    + max(0, len(route_down) - 1)
                )
                m.control_traffic_bits += protocol_header * control_hops
                m.isl_traffic_bits += protocol_header * (
                    max(0, len(route_in) - 1)
                    + max(0, len(route_out) - 1)
                    + max(0, len(route_down) - 2)
                )
            else:
                if helper != assignment.source:
                    m.isl_traffic_bits += tile.d_in_bits * input_fraction
                if helper != aggregator:
                    m.isl_traffic_bits += tile.d_out_bits * output_fraction
        advertisement_bits = float(
            assignment.metadata.get("advertisement_control_bits", 0.0)
        )
        handshake_bits = float(
            assignment.metadata.get("handshake_control_bits", 0.0)
        )
        m.control_traffic_bits += advertisement_bits
        m.isl_traffic_bits += advertisement_bits
        m.control_traffic_bits += handshake_bits
        m.isl_traffic_bits += handshake_bits
        m.protocol_messages += float(
            assignment.metadata.get("protocol_message_count", 0.0)
        )
        for event in assignment.message_events:
            if (event.event == "sent"
                    and event.kind in {"split_request", "replica_request"}):
                helper_requests += 1
            elif (event.event == "delivered"
                    and event.kind in {"split_accept", "replica_accept"}):
                helper_accepts += 1

        # Normal inference schedulers downlink the compact result; direct and
        # compression baselines record their raw/compressed size explicitly.
        # Basilisk accounts for the corresponding transmitter power.
        if reliability > 0.0 and not math.isinf(latency):
            downlink_bits = float(assignment.metadata.get(
                "protocol_ground_bits",
                assignment.metadata.get(
                    "downlink_bits",
                    tile.d_in_bits if assignment.downlink_only else tile.d_out_bits,
                ),
            ))
            m.downlink_volume_bits += downlink_bits

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
    m.delivery_latency_p50_s = _percentile(recovery_lats, 0.50)
    m.delivery_latency_p95_s = _percentile(recovery_lats, 0.95)
    m.objective = sum(float(a.metadata.get("objective", 0.0))
                      for a in result.assignments)
    m.n_replicas_avg = (sum(float(a.metadata.get(
        "effective_replicas", len(a.helpers))) for a in result.assignments)
                        / n_tiles if n_tiles > 0 else 0.0)

    # Helper utilization: fraction of the constellation's compute budget the
    # scheduled work consumes. Numerator is actual FLOPs (each replica
    # runs its tile's compute_ops on the helper); denominator is the summed
    # per-satellite capacity C_i·epoch_length passed by the caller (horizon-wide
    # when the assignment set is a lifetime record). Both sides are in FLOPs,
    # so the ratio is dimensionless.
    total_capacity = sum(sat_compute_capacity.values())
    compute_by_helper = {sat_id: 0.0 for sat_id in sat_compute_capacity}
    for assignment in result.assignments:
        key = (assignment.task_id, assignment.tile_id)
        if key not in tile_lookup:
            continue
        tile = tile_lookup[key][1]
        for index, helper in enumerate(assignment.helpers):
            fraction = (
                assignment.work_fractions[index]
                if index < len(assignment.work_fractions) else 1.0
            )
            compute_by_helper[helper] = (
                compute_by_helper.get(helper, 0.0)
                + tile.compute_ops * fraction
            )
    compute_used = sum(compute_by_helper.values())
    m.helper_utilization = (min(1.0, compute_used / total_capacity)
                            if total_capacity > 0 else 0.0)
    constellation_size = len(sat_compute_capacity)
    active_helpers = sum(
        amount > 0.0 for amount in compute_by_helper.values()
    )
    m.active_helper_fraction = (
        active_helpers / constellation_size if constellation_size else 0.0
    )
    squared_load = sum(amount * amount for amount in compute_by_helper.values())
    m.compute_load_balance = (
        compute_used * compute_used / (constellation_size * squared_load)
        if constellation_size and squared_load > 0.0 else 0.0
    )

    decision_metadata = getattr(result, "metadata", {})
    decision_advertisement_bits = float(
        decision_metadata.get("advertisement_control_bits", 0.0)
    )
    m.protocol_messages += float(
        decision_metadata.get("protocol_message_count", 0.0)
    )
    m.control_traffic_bits += decision_advertisement_bits
    m.isl_traffic_bits += decision_advertisement_bits

    delivered_tiles = len(recovery_lats)
    if delivered_tiles:
        m.isl_traffic_bits_per_delivered_tile = (
            m.isl_traffic_bits / delivered_tiles
        )
        m.control_traffic_bits_per_delivered_tile = (
            m.control_traffic_bits / delivered_tiles
        )
        m.protocol_messages_per_delivered_tile = (
            m.protocol_messages / delivered_tiles
        )
        m.energy_j_per_delivered_tile = m.energy_joules / delivered_tiles
        m.downlink_bits_per_delivered_tile = (
            m.downlink_volume_bits / delivered_tiles
        )
    total_transmitted_bits = m.isl_traffic_bits + m.downlink_volume_bits
    m.control_traffic_ratio = (
        m.control_traffic_bits / total_transmitted_bits
        if total_transmitted_bits > 0.0 else 0.0
    )
    m.helper_request_count = float(helper_requests)
    m.helper_acceptance_ratio = (
        helper_accepts / helper_requests if helper_requests else 1.0
    )
    m.state_age_mean_s = (
        sum(state_ages) / len(state_ages) if state_ages else 0.0
    )
    m.state_age_p95_s = _percentile(state_ages, 0.95)

    return m


def _replica_components(helper: str, agg: str, source: str, routes=None
                        ) -> Tuple[Tuple[str, ...], Tuple[Tuple[str, str], ...], str]:
    """Decompose a replica into the (nodes, isl_links, downlink_node) whose
    joint survival determines whether the replica delivers, mirroring the terms
    of ReliabilityModel.replica_success_prob.

    A replica succeeds iff the helper and aggregator nodes, source→helper ISL,
    helper→aggregator ISL, and the aggregator's downlink all survive. (The
    source node is shared across all replicas of a tile and sampled once at the
    tile level, so it is excluded here.) Self-processing replicas where the
    helper is the source carry no source→helper hop; helper-as-aggregator
    replicas carry no helper→aggregator hop.

    The component set mirrors the policy placement probability, so under
    independent draws the Monte Carlo miss ratio converges to the modeled
    expectation; the only divergence comes from draws
    shared across replicas, which is the correlation effect this layer exists
    to expose.
    """
    nodes = {helper, agg}
    nodes.discard(source)
    isl_links: Set[Tuple[str, str]] = set()
    downlink_node = agg
    if routes:
        route_in, route_out, route_down = routes
        for path in (route_in, route_out):
            isl_links.update((min(a, b), max(a, b))
                             for a, b in zip(path, path[1:]))
        if len(route_down) >= 2:
            isl_links.update((min(a, b), max(a, b))
                             for a, b in zip(route_down[:-2], route_down[1:-1]))
            downlink_node = route_down[-2]
    else:
        if helper != source:
            isl_links.add((min(source, helper), max(source, helper)))
        if helper != agg:
            isl_links.add((min(helper, agg), max(helper, agg)))
    return tuple(sorted(nodes)), tuple(sorted(isl_links)), downlink_node


def compute_realized_metrics(
    result: Decision,
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

    # (task_id, tile, latency, [(components, required), ...], source)
    scored = []
    for a in result.assignments:
        key = (a.task_id, a.tile_id)
        if key not in tile_lookup:
            continue
        task_obj, tile = tile_lookup[key]
        latency = float(a.metadata.get("latency", math.inf))
        reliability_estimate = float(a.metadata.get(
            "reliability", a.metadata.get("reconstruction_probability", 0.0)
        ))
        if not a.helpers or math.isinf(latency):
            # B1-style no-replica assignments: fall back to a single synthetic
            # downlink replica off the primary aggregator so they are still
            # sampled rather than silently treated as always-delivered.
            if reliability_estimate > 0 and not math.isinf(latency):
                aggregator = a.aggregators[0] if a.aggregators else a.source
                scored.append((task_obj.task_id, tile, latency,
                               [([(("__none__",), (), aggregator)], 1)],
                               task_obj.source_sat))
            else:
                scored.append((task_obj.task_id, tile, latency, [],
                               task_obj.source_sat))
            continue
        comps = [_replica_components(
            h, g, task_obj.source_sat,
            a.routes[index] if index < len(a.routes) else None,
        ) for index, (h, g) in enumerate(zip(a.helpers, a.aggregators))]
        required = int(a.metadata.get("data_shards", 1))
        shard_labels = a.metadata.get("shard_groups")
        if shard_labels is not None and len(shard_labels) == len(comps):
            by_group = {}
            for label, component in zip(shard_labels, comps):
                by_group.setdefault(label, []).append(component)
            groups = [
                (group, required) for group in by_group.values()
                if len(group) >= required
            ]
        elif required > 1:
            groups = [(comps, required)] if len(comps) >= required else []
        else:
            groups = [([component], 1) for component in comps]
        scored.append((task_obj.task_id, tile, latency, groups,
                       task_obj.source_sat))

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

        for tid, tile, L_hat, groups, src in scored:
            task_total[tid] = task_total.get(tid, 0) + 1
            delivered = False
            if groups and node_ok(src):
                delivered = any(
                    sum(
                        all(node_ok(n) for n in nodes)
                        and all(link_ok(link) for link in isl_links)
                        and down_ok(agg)
                        for nodes, isl_links, agg in group
                    ) >= required
                    for group, required in groups
                )
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
        "protocol_messages", "control_traffic_bits",
        "delivery_latency_p50_s", "delivery_latency_p95_s",
        "isl_traffic_bits_per_delivered_tile",
        "control_traffic_bits_per_delivered_tile",
        "protocol_messages_per_delivered_tile",
        "energy_j_per_delivered_tile",
        "downlink_bits_per_delivered_tile", "control_traffic_ratio",
        "active_helper_fraction", "compute_load_balance",
        "helper_request_count", "helper_acceptance_ratio",
        "state_age_mean_s", "state_age_p95_s",
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
