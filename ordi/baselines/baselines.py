"""
Eight comparison baselines for ORDI evaluation (E1).

B1  Direct downlink        - no onboard inference; downlink raw tiles to ground
B2  Onboard-only           - source satellite processes all tiles locally, no helpers
B3  Compression-only       - compress tiles before downlink, no distributed compute
B4  Serval-like            - priority-queue bifurcation; single satellite per task
B5  SECO-like              - multi-sat placement, no redundancy, greedy min-latency
B6  Full replication       - replicate every tile to r_max helpers
B7  Random replication     - replicate to random feasible helpers
B8  CoCoI-like             - MDS redundancy (adapted to contact-window setting)

All baselines implement the same interface:
    schedule(epoch, t_sim_start, pending_tasks) → SchedulerResult
"""

from __future__ import annotations
import math
import random
from typing import Dict, List, Optional, Set, Tuple

from ordi.orbit.graph import EpochContactGraph, earliest_arrival, earliest_downlink
from ordi.sim.satellite import SatelliteState
from ordi.sim.reliability import ReliabilityModel
from ordi.tasks.generator import EOTask, Tile
from ordi.scheduler.feasibility import compute_candidates, ReplicaCandidate
from ordi.scheduler.ordi import ORDIConfig, TileAssignment, SchedulerResult
from ordi.scheduler.routing_cache import EpochRoutingCacheMixin

# Compression ratio for B3 (JPEG-style lossy compression on raw EO tiles)
COMPRESSION_RATIO = 0.15   # compressed to 15% of original size


def _build_node_index(
    states: Dict[str, any],
    ground_stations: set,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Build sat_index and node_index enabling the numpy Dijkstra fast path."""
    sat_ids = list(states.keys())
    n = len(sat_ids)
    sat_index: Dict[str, int] = {sid: i for i, sid in enumerate(sat_ids)}
    node_index: Dict[str, int] = {
        **sat_index,
        **{gs: n + i for i, gs in enumerate(sorted(ground_stations))},
    }
    return sat_index, node_index


# ── shared helpers ────────────────────────────────────────────────────────────

def _empty_result(epoch: int) -> SchedulerResult:
    return SchedulerResult(
        epoch=epoch, assignments=[], total_utility=0.0,
        energy_penalty=0.0, comm_penalty=0.0, rep_penalty=0.0,
        objective=0.0, link_utilization={},
    )


def _make_assignment_from_replica(
    task: EOTask, tile: Tile, replica: ReplicaCandidate,
    reliability: ReliabilityModel,
) -> TileAssignment:
    a = TileAssignment(task_id=task.task_id, tile_id=tile.tile_id)
    a.replicas = [replica]
    a.primary_aggregator = replica.aggregator
    a.z_kv = replica.p_success
    a.L_hat = replica.latency
    return a


# ── B1: Direct downlink ───────────────────────────────────────────────────────

class DirectDownlink:
    """
    Raw tiles downlinked directly from the source satellite to a ground station
    — no ISL relay, no onboard inference.  The source satellite must have an
    active satellite-to-ground contact window within the deadline; if not, the
    tile is a miss.  This models the naive baseline where the operator simply
    waits for the satellite to orbit into view of a ground dish.

    At realistic minimum elevation angles (≥ 20°) each pass lasts only 4–6 min,
    so with tight deadlines most source satellites are not in view and B1 misses
    heavily — highlighting ORDI's advantage of routing results via ISL to
    whichever satellite is currently visible.
    """
    name = "B1_direct_downlink"

    def __init__(self, graphs, states, ground_stations, reliability, cfg):
        self.graphs = graphs
        self.states = states
        self.ground_stations = ground_stations
        self.reliability = reliability
        self.cfg = cfg

    def schedule(self, epoch, t_sim_start, pending_tasks) -> SchedulerResult:
        cfg = self.cfg
        epoch_start = t_sim_start + epoch * cfg.epoch_length
        assignments = []
        total_utility = 0.0

        for task in pending_tasks:
            tau_k = task.deadline - epoch_start
            if tau_k <= 0:
                continue
            src = task.source_sat
            src_state = self.states.get(src)
            if src_state is None or not src_state.A_i:
                continue

            for tile in task.tiles:
                # Direct-only: scan forward for the earliest epoch where src has
                # a live satellite-to-ground downlink edge.  No ISL relay used.
                _max_ep = int(math.ceil(tau_k / cfg.epoch_length)) + 1
                ell_down = math.inf
                for ep_offset in range(_max_ep):
                    ep_idx = min(epoch + ep_offset, len(self.graphs) - 1)
                    g = self.graphs[ep_idx]
                    t_wait = ep_offset * cfg.epoch_length
                    for (na, nb, rate, cap, lt) in g.edges:
                        if na == src and nb in self.ground_stations and lt == 'downlink':
                            t_xfer = tile.d_in_bits / max(rate, 1.0)
                            ell_down = min(ell_down, t_wait + t_xfer)
                    if not math.isinf(ell_down):
                        break  # earliest direct contact found

                feasible = ell_down <= tau_k
                p_down = self.reliability.default_downlink_pi if feasible else 0.0
                z_kv = p_down

                a = TileAssignment(task_id=task.task_id, tile_id=tile.tile_id)
                a.primary_aggregator = src
                a.z_kv = z_kv
                a.L_hat = ell_down if feasible else math.inf
                assignments.append(a)
                if feasible:
                    total_utility += tile.utility * z_kv * math.exp(-cfg.alpha * ell_down)

        return SchedulerResult(
            epoch=epoch, assignments=assignments, total_utility=total_utility,
            energy_penalty=0.0, comm_penalty=0.0, rep_penalty=0.0,
            objective=total_utility, link_utilization={},
        )


# ── B2: Onboard-only ─────────────────────────────────────────────────────────

class OnboardOnly:
    """Source satellite processes all tiles locally and downlinks results."""
    name = "B2_onboard_only"

    def __init__(self, graphs, states, ground_stations, reliability, cfg):
        self.graphs = graphs
        self.states = states
        self.ground_stations = ground_stations
        self.reliability = reliability
        self.cfg = cfg
        self.sat_index, self.node_index = _build_node_index(states, ground_stations)

    def schedule(self, epoch, t_sim_start, pending_tasks) -> SchedulerResult:
        cfg = self.cfg
        epoch_start = t_sim_start + epoch * cfg.epoch_length
        assignments = []
        total_utility = 0.0

        for task in pending_tasks:
            tau_k = task.deadline - epoch_start
            if tau_k <= 0:
                continue
            src = task.source_sat
            src_state = self.states.get(src)
            if src_state is None or not src_state.A_i:
                continue

            for tile in task.tiles:
                t_compute = tile.compute_ops / max(src_state.C_i, 1.0)
                _max_ep = int(math.ceil(tau_k / cfg.epoch_length)) + 1
                ell_down = earliest_downlink(
                    src, epoch, self.graphs, tile.d_out_bits, self.ground_stations,
                    max_search_epochs=_max_ep, node_index=self.node_index,
                )
                L = t_compute + ell_down
                feasible = L <= tau_k
                p = self.reliability.node_pi(src) * self.reliability.default_downlink_pi
                z_kv = p if feasible else 0.0

                a = TileAssignment(task_id=task.task_id, tile_id=tile.tile_id)
                a.primary_aggregator = src
                a.z_kv = z_kv
                a.L_hat = L if feasible else math.inf
                if feasible:
                    a.replicas = [ReplicaCandidate(
                        task_id=task.task_id, tile_id=tile.tile_id,
                        helper=src, aggregator=src, epoch=epoch,
                        latency=L, p_success=p,
                        e_compute=src_state.energy_for_compute(tile.compute_ops),
                        e_rx=0.0, e_tx=0.0,
                        feasible=True,
                        d_in_bits=0.0, d_out_bits=0.0,
                    )]
                    total_utility += tile.utility * z_kv * math.exp(-cfg.alpha * L)
                assignments.append(a)

        return SchedulerResult(
            epoch=epoch, assignments=assignments, total_utility=total_utility,
            energy_penalty=0.0, comm_penalty=0.0, rep_penalty=0.0,
            objective=total_utility, link_utilization={},
        )


# ── B3: Compression-only ─────────────────────────────────────────────────────

class CompressionOnly:
    """Compress raw tiles before downlink; no distributed compute."""
    name = "B3_compression_only"

    def __init__(self, graphs, states, ground_stations, reliability, cfg):
        self.graphs = graphs
        self.states = states
        self.ground_stations = ground_stations
        self.reliability = reliability
        self.cfg = cfg
        self.sat_index, self.node_index = _build_node_index(states, ground_stations)

    def schedule(self, epoch, t_sim_start, pending_tasks) -> SchedulerResult:
        cfg = self.cfg
        epoch_start = t_sim_start + epoch * cfg.epoch_length
        assignments = []
        total_utility = 0.0

        for task in pending_tasks:
            tau_k = task.deadline - epoch_start
            if tau_k <= 0:
                continue
            src = task.source_sat
            src_state = self.states.get(src)
            if src_state is None or not src_state.A_i:
                continue

            for tile in task.tiles:
                compressed_bits = tile.d_in_bits * COMPRESSION_RATIO
                _max_ep = int(math.ceil(tau_k / cfg.epoch_length)) + 1
                ell_down = earliest_downlink(
                    src, epoch, self.graphs, compressed_bits, self.ground_stations,
                    max_search_epochs=_max_ep, node_index=self.node_index,
                )
                feasible = ell_down <= tau_k
                p = self.reliability.default_downlink_pi if feasible else 0.0
                z_kv = p

                a = TileAssignment(task_id=task.task_id, tile_id=tile.tile_id)
                a.primary_aggregator = src
                a.z_kv = z_kv
                a.L_hat = ell_down if feasible else math.inf
                if feasible:
                    # Compression compute is ~10% of full inference cost
                    a.replicas = [ReplicaCandidate(
                        task_id=task.task_id, tile_id=tile.tile_id,
                        helper=src, aggregator=src, epoch=epoch,
                        latency=ell_down, p_success=p,
                        e_compute=src_state.energy_for_compute(tile.compute_ops * 0.1),
                        e_rx=0.0, e_tx=0.0,
                        feasible=True,
                        d_in_bits=0.0, d_out_bits=0.0,
                    )]
                    total_utility += tile.utility * z_kv * math.exp(-cfg.alpha * ell_down)
                assignments.append(a)

        return SchedulerResult(
            epoch=epoch, assignments=assignments, total_utility=total_utility,
            energy_penalty=0.0, comm_penalty=0.0, rep_penalty=0.0,
            objective=total_utility, link_utilization={},
        )


# ── B4: Serval-like ───────────────────────────────────────────────────────────

class ServalLike:
    """
    Priority-queue bifurcation inspired by Serval (NSDI '24).
    Tiles are classified as high/low priority based on utility.
    High-priority tiles are processed on source sat and downlinked first.
    No inter-satellite helpers used.
    """
    name = "B4_serval_like"
    HIGH_PRIORITY_THRESHOLD = 0.8

    def __init__(self, graphs, states, ground_stations, reliability, cfg):
        self.graphs = graphs
        self.states = states
        self.ground_stations = ground_stations
        self.reliability = reliability
        self.cfg = cfg
        self.sat_index, self.node_index = _build_node_index(states, ground_stations)

    def schedule(self, epoch, t_sim_start, pending_tasks) -> SchedulerResult:
        cfg = self.cfg
        epoch_start = t_sim_start + epoch * cfg.epoch_length
        assignments = []
        total_utility = 0.0

        for task in pending_tasks:
            tau_k = task.deadline - epoch_start
            if tau_k <= 0:
                continue
            src = task.source_sat
            src_state = self.states.get(src)
            if src_state is None or not src_state.A_i:
                continue

            # Classify tiles by priority
            max_u = max(t.utility for t in task.tiles)
            high_priority = [t for t in task.tiles
                             if t.utility >= self.HIGH_PRIORITY_THRESHOLD * max_u]
            low_priority  = [t for t in task.tiles
                             if t.utility < self.HIGH_PRIORITY_THRESHOLD * max_u]

            time_offset = 0.0
            for tile in high_priority + low_priority:
                t_compute = tile.compute_ops / max(src_state.C_i, 1.0)
                _max_ep = int(math.ceil(tau_k / cfg.epoch_length)) + 1
                ell_down = earliest_downlink(
                    src, epoch, self.graphs, tile.d_out_bits, self.ground_stations,
                    max_search_epochs=_max_ep, node_index=self.node_index,
                )
                L = time_offset + t_compute + ell_down
                feasible = L <= tau_k
                p = self.reliability.node_pi(src) * self.reliability.default_downlink_pi
                z_kv = p if feasible else 0.0

                a = TileAssignment(task_id=task.task_id, tile_id=tile.tile_id)
                a.primary_aggregator = src
                a.z_kv = z_kv
                a.L_hat = L if feasible else math.inf
                if feasible:
                    a.replicas = [ReplicaCandidate(
                        task_id=task.task_id, tile_id=tile.tile_id,
                        helper=src, aggregator=src, epoch=epoch,
                        latency=L, p_success=p,
                        e_compute=src_state.energy_for_compute(tile.compute_ops),
                        e_rx=0.0, e_tx=0.0,
                        feasible=True,
                        d_in_bits=0.0, d_out_bits=0.0,
                    )]
                    total_utility += tile.utility * z_kv * math.exp(-cfg.alpha * L)
                time_offset += t_compute  # sequential on source sat
                assignments.append(a)

        return SchedulerResult(
            epoch=epoch, assignments=assignments, total_utility=total_utility,
            energy_penalty=0.0, comm_penalty=0.0, rep_penalty=0.0,
            objective=total_utility, link_utilization={},
        )


# ── B5: SECO-like ─────────────────────────────────────────────────────────────

class SECOLike(EpochRoutingCacheMixin):
    """
    Multi-satellite placement without redundancy, greedy min-latency.
    Inspired by SECO (INFOCOM '24): assign each tile to the helper
    with minimum end-to-end time cost, no backup.
    """
    name = "B5_seco_like"

    def __init__(self, graphs, states, ground_stations, reliability, cfg):
        self.graphs = graphs
        self.states = states
        self.ground_stations = ground_stations
        self.reliability = reliability
        self.cfg = cfg
        self.sat_index, self.node_index = _build_node_index(states, ground_stations)

    def schedule(self, epoch, t_sim_start, pending_tasks) -> SchedulerResult:
        cfg = self.cfg
        epoch_start = t_sim_start + epoch * cfg.epoch_length
        assignments = []
        total_utility = 0.0
        sat_index, ell_down_c, ell_ia_c, ell_ski_c = \
            self._build_epoch_caches(epoch, epoch_start, pending_tasks)

        for task in pending_tasks:
            tau_k = task.deadline - epoch_start
            if tau_k <= 0:
                continue
            for tile in task.tiles:
                candidates = compute_candidates(
                    task, tile, epoch, epoch_start,
                    self.graphs, self.states, self.reliability,
                    self.ground_stations, tau_k,
                    sat_index=sat_index, node_index=self.node_index,
                    ell_down_cache=ell_down_c[tile.d_out_bits],
                    ell_ia_cache=ell_ia_c[tile.d_out_bits],
                    ell_ski_cache=ell_ski_c[tile.d_in_bits][task.source_sat],
                )
                if not candidates:
                    assignments.append(TileAssignment(task_id=task.task_id, tile_id=tile.tile_id))
                    continue

                # Pick minimum latency (SECO greedy)
                best = min(candidates, key=lambda c: c.latency)
                a = _make_assignment_from_replica(task, tile, best, self.reliability)
                assignments.append(a)
                total_utility += tile.utility * a.z_kv * math.exp(-cfg.alpha * a.L_hat)

        return SchedulerResult(
            epoch=epoch, assignments=assignments, total_utility=total_utility,
            energy_penalty=0.0, comm_penalty=0.0, rep_penalty=0.0,
            objective=total_utility, link_utilization={},
        )


# ── B6: Full replication ──────────────────────────────────────────────────────

class FullReplication(EpochRoutingCacheMixin):
    """Replicate every tile to all r_max feasible helpers."""
    name = "B6_full_replication"

    def __init__(self, graphs, states, ground_stations, reliability, cfg):
        self.graphs = graphs
        self.states = states
        self.ground_stations = ground_stations
        self.reliability = reliability
        self.cfg = cfg
        self.sat_index, self.node_index = _build_node_index(states, ground_stations)

    def schedule(self, epoch, t_sim_start, pending_tasks) -> SchedulerResult:
        cfg = self.cfg
        epoch_start = t_sim_start + epoch * cfg.epoch_length
        assignments = []
        total_utility = 0.0
        sat_index, ell_down_c, ell_ia_c, ell_ski_c = \
            self._build_epoch_caches(epoch, epoch_start, pending_tasks)

        for task in pending_tasks:
            tau_k = task.deadline - epoch_start
            if tau_k <= 0:
                continue
            for tile in task.tiles:
                candidates = compute_candidates(
                    task, tile, epoch, epoch_start,
                    self.graphs, self.states, self.reliability,
                    self.ground_stations, tau_k,
                    sat_index=sat_index, node_index=self.node_index,
                    ell_down_cache=ell_down_c[tile.d_out_bits],
                    ell_ia_cache=ell_ia_c[tile.d_out_bits],
                    ell_ski_cache=ell_ski_c[tile.d_in_bits][task.source_sat],
                )
                # Use up to r_max replicas with distinct aggregators
                selected = []
                seen_agg = set()
                for c in sorted(candidates, key=lambda c: -c.p_success):
                    if c.aggregator not in seen_agg and len(selected) < tile.n_replicas_max:
                        selected.append(c)
                        seen_agg.add(c.aggregator)

                a = TileAssignment(task_id=task.task_id, tile_id=tile.tile_id)
                a.replicas = selected
                if selected:
                    a.primary_aggregator = selected[0].aggregator
                    if len(selected) > 1:
                        a.backup_aggregator = selected[1].aggregator
                    probs = [c.p_success for c in selected]
                    a.z_kv = self.reliability.tile_delivery_prob(
                        probs, self.reliability.node_pi(task.source_sat))
                    a.L_hat = min(c.latency for c in selected)
                    total_utility += tile.utility * a.z_kv * math.exp(-cfg.alpha * a.L_hat)

                assignments.append(a)

        return SchedulerResult(
            epoch=epoch, assignments=assignments, total_utility=total_utility,
            energy_penalty=0.0, comm_penalty=0.0, rep_penalty=0.0,
            objective=total_utility, link_utilization={},
        )


# ── B7: Random replication ────────────────────────────────────────────────────

class RandomReplication(EpochRoutingCacheMixin):
    """Replicate to random feasible helpers (up to r_max)."""
    name = "B7_random_replication"

    def __init__(self, graphs, states, ground_stations, reliability, cfg, seed=42):
        self.graphs = graphs
        self.states = states
        self.ground_stations = ground_stations
        self.reliability = reliability
        self.cfg = cfg
        self.rng = random.Random(seed)
        self.sat_index, self.node_index = _build_node_index(states, ground_stations)

    def schedule(self, epoch, t_sim_start, pending_tasks) -> SchedulerResult:
        cfg = self.cfg
        epoch_start = t_sim_start + epoch * cfg.epoch_length
        assignments = []
        total_utility = 0.0
        sat_index, ell_down_c, ell_ia_c, ell_ski_c = \
            self._build_epoch_caches(epoch, epoch_start, pending_tasks)

        for task in pending_tasks:
            tau_k = task.deadline - epoch_start
            if tau_k <= 0:
                continue
            for tile in task.tiles:
                candidates = compute_candidates(
                    task, tile, epoch, epoch_start,
                    self.graphs, self.states, self.reliability,
                    self.ground_stations, tau_k,
                    sat_index=sat_index, node_index=self.node_index,
                    ell_down_cache=ell_down_c[tile.d_out_bits],
                    ell_ia_cache=ell_ia_c[tile.d_out_bits],
                    ell_ski_cache=ell_ski_c[tile.d_in_bits][task.source_sat],
                )
                if not candidates:
                    assignments.append(TileAssignment(task_id=task.task_id, tile_id=tile.tile_id))
                    continue

                n = min(tile.n_replicas_max, len(candidates))
                selected = self.rng.sample(candidates, n)

                a = TileAssignment(task_id=task.task_id, tile_id=tile.tile_id)
                a.replicas = selected
                if selected:
                    a.primary_aggregator = selected[0].aggregator
                    probs = [c.p_success for c in selected]
                    a.z_kv = self.reliability.tile_delivery_prob(
                        probs, self.reliability.node_pi(task.source_sat))
                    a.L_hat = min(c.latency for c in selected)
                    total_utility += tile.utility * a.z_kv * math.exp(-cfg.alpha * a.L_hat)

                assignments.append(a)

        return SchedulerResult(
            epoch=epoch, assignments=assignments, total_utility=total_utility,
            energy_penalty=0.0, comm_penalty=0.0, rep_penalty=0.0,
            objective=total_utility, link_utilization={},
        )


# ── B8: CoCoI-like ────────────────────────────────────────────────────────────

class CoCoILike(EpochRoutingCacheMixin):
    """
    MDS coded redundancy adapted to contact-window setting.
    Inspired by CoCoI (arXiv 2025): k-of-n MDS coding where any k
    of n assigned helpers suffice to reconstruct the result.
    Uses n = r_max workers and k = ceil(n/2) threshold.
    Success probability = P(at least k of n replicas succeed).
    """
    name = "B8_cocoi_like"

    def __init__(self, graphs, states, ground_stations, reliability, cfg):
        self.graphs = graphs
        self.states = states
        self.ground_stations = ground_stations
        self.reliability = reliability
        self.cfg = cfg
        self.sat_index, self.node_index = _build_node_index(states, ground_stations)

    def _k_of_n_prob(self, probs: List[float], k: int, source_pi: float = 1.0) -> float:
        """P(at least k of n succeed); source survival factored once at tile level."""
        if source_pi <= 0.0:
            return 0.0
        probs = [min(1.0, p / source_pi) for p in probs]
        n = len(probs)
        if k > n:
            return 0.0
        # DP: dp[j] = P(exactly j successes)
        dp = [0.0] * (n + 1)
        dp[0] = 1.0
        for p in probs:
            new_dp = [0.0] * (n + 1)
            for j in range(n + 1):
                if dp[j] == 0.0:
                    continue
                if j + 1 <= n:
                    new_dp[j + 1] += dp[j] * p
                new_dp[j] += dp[j] * (1 - p)
            dp = new_dp
        return source_pi * sum(dp[j] for j in range(k, n + 1))

    def schedule(self, epoch, t_sim_start, pending_tasks) -> SchedulerResult:
        cfg = self.cfg
        epoch_start = t_sim_start + epoch * cfg.epoch_length
        assignments = []
        total_utility = 0.0
        sat_index, ell_down_c, ell_ia_c, ell_ski_c = \
            self._build_epoch_caches(epoch, epoch_start, pending_tasks)

        for task in pending_tasks:
            tau_k = task.deadline - epoch_start
            if tau_k <= 0:
                continue
            for tile in task.tiles:
                candidates = compute_candidates(
                    task, tile, epoch, epoch_start,
                    self.graphs, self.states, self.reliability,
                    self.ground_stations, tau_k,
                    sat_index=sat_index, node_index=self.node_index,
                    ell_down_cache=ell_down_c[tile.d_out_bits],
                    ell_ia_cache=ell_ia_c[tile.d_out_bits],
                    ell_ski_cache=ell_ski_c[tile.d_in_bits][task.source_sat],
                )
                if not candidates:
                    assignments.append(TileAssignment(task_id=task.task_id, tile_id=tile.tile_id))
                    continue

                n = min(tile.n_replicas_max, len(candidates))
                # Pick best n by success probability
                selected = sorted(candidates, key=lambda c: -c.p_success)[:n]
                k_thresh = math.ceil(n / 2)  # need at least half to succeed
                probs = [c.p_success for c in selected]
                z_kv = self._k_of_n_prob(probs, k_thresh,
                                         self.reliability.node_pi(task.source_sat))

                a = TileAssignment(task_id=task.task_id, tile_id=tile.tile_id)
                a.replicas = selected
                if selected:
                    a.primary_aggregator = selected[0].aggregator
                    a.z_kv = z_kv
                    a.L_hat = min(c.latency for c in selected)
                    total_utility += tile.utility * z_kv * math.exp(-cfg.alpha * a.L_hat)

                assignments.append(a)

        return SchedulerResult(
            epoch=epoch, assignments=assignments, total_utility=total_utility,
            energy_penalty=0.0, comm_penalty=0.0, rep_penalty=0.0,
            objective=total_utility, link_utilization={},
        )


# ── registry ──────────────────────────────────────────────────────────────────

ALL_BASELINES = [
    DirectDownlink, OnboardOnly, CompressionOnly, ServalLike,
    SECOLike, FullReplication, RandomReplication, CoCoILike,
]


def build_all_baselines(graphs, states, ground_stations, reliability, cfg) -> dict:
    return {
        cls.name: cls(graphs, states, ground_stations, reliability, cfg)
        for cls in ALL_BASELINES
    }
