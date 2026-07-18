"""
ORDI rolling-horizon greedy scheduler.

Per epoch:
  1. Update satellite states
  2. Rebuild epoch contact graph
  3. For each pending (task, tile):
       a. Enumerate feasible (helper, aggregator) candidates
       b. Score by marginal objective: ΔU - λ_E·ΔE - λ_C·ΔC - λ_R·ΔR
       c. Assign primary replica (best score)
       d. Add backups up to the configured cap while marginal gain > 0
  4. Commit assignments; update Q_i, B_i, link utilization
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from ordi.orbit.graph import EpochContactGraph, earliest_downlink
from ordi.scheduler.routing_cache import EpochRoutingCacheMixin
from ordi.sim.satellite import SatelliteState
from ordi.sim.reliability import ReliabilityModel
from ordi.tasks.generator import EOTask, Tile
from ordi.scheduler.feasibility import ReplicaCandidate, compute_candidates


# ── scheduler parameters (λ weights from proposal) ───────────────────────────
@dataclass
class ORDIConfig:
    lambda_E: float = 1e-5    # utility per Joule (energy penalty weight)
    lambda_C: float = 1e-12   # utility per weighted-bit (comm penalty weight)
    lambda_R: float = 0.05    # utility per extra replica (replication penalty)
    alpha: float = 0.002      # freshness decay rate (1/s)
    epoch_length: float = 60.0  # seconds
    isl_rate_bps: float = 200e6
    # Maximum number of selectively admitted backups per tile. The paper's
    # default policy uses one; E9 sweeps larger caps as an ablation. The task's
    # n_replicas_max remains the hard upper bound on total replicas.
    max_backups: int = 1
    # Under a correlated-failure threat model (E7), require the backup's helper
    # to occupy a different orbital plane than the primary's helper. Off by
    # default so nominal experiments keep the emergent-placement framing.
    plane_disjoint_backup: bool = False


# ── assignment record ─────────────────────────────────────────────────────────
@dataclass
class TileAssignment:
    task_id: int
    tile_id: int
    replicas: List[ReplicaCandidate] = field(default_factory=list)
    primary_aggregator: Optional[str] = None
    backup_aggregator: Optional[str] = None
    z_kv: float = 0.0          # modeled delivery probability
    L_hat: float = math.inf    # min latency across replicas
    # None means the normal inference-result size from the task's Tile record.
    # Bent-pipe and compression baselines override this with raw/compressed bits.
    downlink_bits: Optional[float] = None


@dataclass
class SchedulerResult:
    epoch: int
    assignments: List[TileAssignment]
    total_utility: float
    energy_penalty: float
    comm_penalty: float
    rep_penalty: float
    objective: float
    link_utilization: Dict[Tuple[str, str], float]  # (a,b) → bits used


class ORDIScheduler(EpochRoutingCacheMixin):
    def __init__(
        self,
        config: ORDIConfig,
        sat_ids: List[str],
        ground_stations: set,
        graphs: List[EpochContactGraph],
        states: Dict[str, SatelliteState],
        reliability: ReliabilityModel,
    ):
        self.cfg = config
        self.sat_ids = sat_ids
        self.ground_stations = ground_stations
        self.graphs = graphs
        self.states = states
        self.reliability = reliability

        # Link utilization tracking: (a,b,epoch) → bits consumed
        self._link_used: Dict[Tuple[str, str, int], float] = {}

    # ── main scheduling entry point ───────────────────────────────────────────

    def schedule_epoch(
        self,
        epoch: int,
        t_sim_start: float,
        pending_tasks: List[EOTask],
    ) -> SchedulerResult:
        """
        Schedule all pending tasks/tiles for one epoch.
        Returns a SchedulerResult with all assignments.
        """
        cfg = self.cfg
        g = self.graphs[epoch]
        epoch_start = t_sim_start + epoch * cfg.epoch_length

        assignments: List[TileAssignment] = []
        total_utility = 0.0
        energy_used: Dict[str, float] = {s: 0.0 for s in self.sat_ids}
        link_used: Dict[Tuple[str, str], float] = {}

        # Precompute routing caches once per epoch to avoid O(N²) Dijkstra
        # redundancy across the many (task, tile) pairs scheduled this epoch.
        sat_index, ell_down_caches, ell_ia_caches, ell_ski_caches = \
            self._build_epoch_caches(epoch, epoch_start, pending_tasks)

        # Sort (task, tile) pairs by descending utility/urgency so high-value
        # tiles claim routing resources before lower-priority ones.
        pending_tile_list = []
        for task in pending_tasks:
            tau_k = task.deadline - epoch_start
            if tau_k <= 0:
                continue  # already past deadline
            for tile in task.tiles:
                pending_tile_list.append((tile.utility / max(tau_k, 1.0), task, tile, tau_k))
        pending_tile_list.sort(key=lambda x: -x[0])

        for _, task, tile, tau_k in pending_tile_list:
            assignment = self._schedule_tile(
                task, tile, epoch, epoch_start, tau_k,
                energy_used, link_used,
                sat_index,
                ell_down_caches.get(tile.d_out_bits),
                ell_ia_caches.get(tile.d_out_bits),
                ell_ski_caches.get(tile.d_in_bits, {}).get(task.source_sat),
            )
            assignments.append(assignment)
            total_utility += tile.utility * assignment.z_kv * math.exp(
                -cfg.alpha * (assignment.L_hat if not math.isinf(assignment.L_hat) else 0)
            )

        # Compute penalties
        e_total = sum(
            sum(r.e_compute + r.e_rx + r.e_tx for r in a.replicas)
            for a in assignments
        )
        c_total = sum(
            (1.0 / max(g.capacity_between(a, b), 1.0)) * bits
            for (a, b), bits in link_used.items()
        )
        r_total = sum(max(0, len(a.replicas) - 1) for a in assignments)

        obj = (total_utility
               - cfg.lambda_E * e_total
               - cfg.lambda_C * c_total
               - cfg.lambda_R * r_total)

        return SchedulerResult(
            epoch=epoch,
            assignments=assignments,
            total_utility=total_utility,
            energy_penalty=cfg.lambda_E * e_total,
            comm_penalty=cfg.lambda_C * c_total,
            rep_penalty=cfg.lambda_R * r_total,
            objective=obj,
            link_utilization=link_used,
        )

    def _schedule_tile(
        self,
        task: EOTask,
        tile: Tile,
        epoch: int,
        epoch_start: float,
        tau_k: float,
        energy_used: Dict[str, float],
        link_used: Dict[Tuple[str, str], float],
        sat_index: Optional[Dict[str, int]] = None,
        ell_down_cache: Optional[np.ndarray] = None,   # shape (n_active,)
        ell_ia_cache: Optional[np.ndarray] = None,     # shape (n_active, n_active)
        ell_ski_cache: Optional[np.ndarray] = None,    # shape (n_active,)
    ) -> TileAssignment:
        cfg = self.cfg
        assignment = TileAssignment(task_id=task.task_id, tile_id=tile.tile_id)
        source_pi = self.reliability.node_pi(task.source_sat)

        candidates = compute_candidates(
            task, tile, epoch, epoch_start,
            self.graphs, self.states, self.reliability,
            self.ground_stations, tau_k,
            sat_index=sat_index,
            ell_down_cache=ell_down_cache,
            ell_ia_cache=ell_ia_cache,
            ell_ski_cache=ell_ski_cache,
        )

        # Add source-satellite self-processing as a candidate so ORDI never
        # has a higher miss rate than onboard-only baselines. Zero ISL bits/energy.
        src_state = self.states.get(task.source_sat)
        if src_state and src_state.A_i:
            t_compute = (src_state.Q_i + tile.compute_ops) / max(src_state.C_i, 1.0)
            # Reuse the epoch-level downlink cache for the source satellite.
            src_idx = sat_index.get(task.source_sat) if sat_index is not None else None
            if ell_down_cache is not None and src_idx is not None:
                ell_down = float(ell_down_cache[src_idx])
            else:
                _max_ep = int(math.ceil(tau_k / cfg.epoch_length)) + 1
                ell_down = earliest_downlink(
                    task.source_sat, epoch, self.graphs, tile.d_out_bits,
                    self.ground_stations, max_search_epochs=_max_ep,
                )
            L_self = t_compute + ell_down
            if L_self <= tau_k:
                p_self = (self.reliability.node_pi(task.source_sat)
                          * self.reliability.downlink_pi(task.source_sat))
                candidates.append(ReplicaCandidate(
                    task_id=task.task_id,
                    tile_id=tile.tile_id,
                    helper=task.source_sat,
                    aggregator=task.source_sat,
                    epoch=epoch,
                    latency=L_self,
                    p_success=p_self,
                    e_compute=src_state.energy_for_compute(tile.compute_ops),
                    e_rx=0.0,
                    e_tx=0.0,
                    feasible=True,
                    d_in_bits=0.0,
                    d_out_bits=0.0,
                ))

        if not candidates:
            return assignment  # no feasible replica

        # Score candidates by marginal objective gain
        scored = []
        for c in candidates:
            # Utility gain from this replica (as if it's the only one)
            z_single = c.p_success
            u_gain = tile.utility * z_single * math.exp(-cfg.alpha * c.latency)
            e_cost = cfg.lambda_E * (c.e_compute + c.e_rx + c.e_tx)

            # Comm cost approximation: bits × congestion price on key links
            g = self.graphs[epoch]
            c_cost = cfg.lambda_C * (
                c.d_in_bits / max(g.capacity_between(task.source_sat, c.helper), 1.0) +
                c.d_out_bits / max(g.capacity_between(c.helper, c.aggregator), 1.0)
            )

            # Check energy + thermal feasibility
            h_state = self.states[c.helper]
            energy_budget = h_state.B_i - h_state.params.battery_min_j - energy_used.get(c.helper, 0.0)
            if energy_budget < (c.e_compute + c.e_rx + c.e_tx):
                continue  # helper cannot afford this replica

            score = u_gain - e_cost - c_cost
            scored.append((score, c))

        if not scored:
            return assignment

        scored.sort(key=lambda x: -x[0])

        # ── primary replica ───────────────────────────────────────────────────
        _, primary = scored[0]
        assignment.replicas.append(primary)
        assignment.primary_aggregator = primary.aggregator
        assignment.L_hat = primary.latency
        _charge_resources(primary, energy_used, link_used, task.source_sat)

        # Update delivery probability
        replica_probs = [primary.p_success]
        assignment.z_kv = self.reliability.tile_delivery_prob(replica_probs, source_pi)

        # ── backup replicas ───────────────────────────────────────────────────
        # Existing experiments retain the paper's one-backup policy through
        # cfg.max_backups=1. The cap ablation raises this value and the task's
        # hard replica limit, admitting each additional backup only while its
        # reliability gain remains positive.
        replica_cap = min(tile.n_replicas_max, 1 + max(0, cfg.max_backups))
        if replica_cap >= 2:
            selected_helpers = {primary.helper}
            selected_aggregators = {primary.aggregator}
            selected_planes = {_plane_of(primary.helper)}
            for _, backup in scored[1:]:
                if len(assignment.replicas) >= replica_cap:
                    break
                # Replicas must use mutually distinct helpers and aggregators.
                if backup.aggregator in selected_aggregators:
                    continue
                if backup.helper in selected_helpers:
                    continue
                # Under a correlated-failure threat model, reject backups whose
                # helper shares a plane with any already-selected replica.
                if cfg.plane_disjoint_backup:
                    bp = _plane_of(backup.helper)
                    if bp is not None and bp in selected_planes:
                        continue

                h_state = self.states[backup.helper]
                energy_budget = (h_state.B_i - h_state.params.battery_min_j
                                 - energy_used.get(backup.helper, 0.0))
                if energy_budget < (backup.e_compute + backup.e_rx + backup.e_tx):
                    continue

                # Check marginal gain: reliability improvement vs. replica penalty
                new_z = self.reliability.tile_delivery_prob(
                    replica_probs + [backup.p_success], source_pi
                )
                delta_z = new_z - assignment.z_kv
                delta_utility = tile.utility * delta_z * math.exp(-cfg.alpha * backup.latency)
                delta_energy = cfg.lambda_E * (backup.e_compute + backup.e_rx + backup.e_tx)
                delta_rep = cfg.lambda_R  # one extra replica

                if delta_utility - delta_energy - delta_rep > 0:
                    assignment.replicas.append(backup)
                    if assignment.backup_aggregator is None:
                        assignment.backup_aggregator = backup.aggregator
                    replica_probs.append(backup.p_success)
                    assignment.z_kv = self.reliability.tile_delivery_prob(replica_probs, source_pi)
                    assignment.L_hat = min(assignment.L_hat, backup.latency)
                    _charge_resources(backup, energy_used, link_used, task.source_sat)
                    selected_helpers.add(backup.helper)
                    selected_aggregators.add(backup.aggregator)
                    selected_planes.add(_plane_of(backup.helper))

        return assignment

    # ── replanning ────────────────────────────────────────────────────────────

    def replan(
        self,
        epoch: int,
        t_sim_start: float,
        failed_helpers: Set[str],
        missed_tiles: List[Tuple[int, int]],   # (task_id, tile_id) pairs
        pending_tasks: List[EOTask],
    ) -> SchedulerResult:
        """
        Triggered on: helper failure, missed contact, straggler, or high-priority arrival.
        Marks failed helpers unavailable and reschedules affected tiles.
        """
        for h in failed_helpers:
            if h in self.states:
                self.states[h].inject_failure()

        # Filter to only affected tasks
        affected_task_ids = {t for t, _ in missed_tiles}
        affected = [task for task in pending_tasks if task.task_id in affected_task_ids]

        return self.schedule_epoch(epoch, t_sim_start, affected)


def _plane_of(sat_id: str) -> Optional[str]:
    """Parse the orbital-plane id from a ``SAT_<plane>_<idx>`` name.
    Returns None if the name doesn't match, so unknown helpers never compare
    equal to each other (an unparseable pair is treated as plane-disjoint)."""
    parts = sat_id.split("_")
    return parts[1] if len(parts) >= 3 and parts[0] == "SAT" else None


def _charge_resources(
    c: ReplicaCandidate,
    energy_used: Dict[str, float],
    link_used: Dict[Tuple[str, str], float],
    source: str,
):
    energy_used[c.helper] = energy_used.get(c.helper, 0.0) + c.e_compute + c.e_rx + c.e_tx
    key_in  = (source, c.helper)
    key_out = (c.helper, c.aggregator)
    link_used[key_in]  = link_used.get(key_in,  0.0) + c.d_in_bits
    link_used[key_out] = link_used.get(key_out, 0.0) + c.d_out_bits
