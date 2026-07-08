"""
Fault injection framework for ORDI evaluation (Phase 6 / E2, E3, E7).

Supports eight fault types from the proposal:
  1. ISL disruption          - remove specific ISL edge for N epochs
  2. Orbital-plane outage    - disable all satellites in a plane for N epochs
  3. Helper failure          - flip A_i=0 for a specific satellite
  4. Straggler               - scale C_i by factor for a helper during execution
  5. Ground-contact miss     - remove downlink window for N epochs
  6. Battery shortage        - drain B_i below B_min
  7. Thermal throttling      - set Θ_i above Θ_max to force throttle
  8. Adverse downlink        - reduce aggregator downlink π (weather)
"""

from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from ordi.sim.satellite import SatelliteState
from ordi.sim.reliability import ReliabilityModel, DEFAULT_DOWNLINK_ADV_PI
from ordi.orbit.contacts import ContactEvent
from ordi.orbit.graph import EpochContactGraph


@dataclass
class FaultEvent:
    fault_type: str
    start_epoch: int
    duration_epochs: int        # how many epochs the fault persists
    targets: List[str]          # satellite IDs or link tuples
    params: dict = field(default_factory=dict)  # type-specific parameters

    @property
    def end_epoch(self) -> int:
        return self.start_epoch + self.duration_epochs


class FaultInjector:
    """
    Applies and withdraws faults to satellite states and the reliability model.
    Call apply(epoch) at the start of each epoch and withdraw(epoch) at the end.
    """

    def __init__(
        self,
        states: Dict[str, SatelliteState],
        reliability: ReliabilityModel,
        contacts: List[ContactEvent],
        rng_seed: int = 0,
        graphs: Optional[List[EpochContactGraph]] = None,
        gs_names: Optional[Set[str]] = None,
    ):
        self.states = states
        self.reliability = reliability
        self.contacts = contacts
        self.rng = random.Random(rng_seed)
        self.graphs = graphs or []
        self.gs_names = gs_names or set()
        self._active: List[FaultEvent] = []
        self._scheduled: List[FaultEvent] = []
        # Maps fault id → {ep_idx: [(a, b, rate, cap, ltype), ...]} for ground_contact_miss
        self._removed_edges: Dict[int, Dict[int, list]] = {}
        # Maps fault id → {sat_id: C_i} snapshot at apply time, for straggler restore
        self._compute_snapshots: Dict[int, Dict[str, float]] = {}

    def schedule(self, fault: FaultEvent):
        """Register a fault to be applied at its start_epoch."""
        self._scheduled.append(fault)

    def apply_epoch(self, epoch: int):
        """Apply all faults that start at this epoch."""
        for fault in self._scheduled:
            if fault.start_epoch == epoch:
                self._active.append(fault)
                self._apply(fault)

    def withdraw_epoch(self, epoch: int):
        """Withdraw faults whose duration has expired."""
        expired = [f for f in self._active if f.end_epoch <= epoch]
        for fault in expired:
            self._withdraw(fault)
        self._active = [f for f in self._active if f.end_epoch > epoch]

    # ── apply/withdraw per fault type ─────────────────────────────────────────

    def _apply(self, fault: FaultEvent):
        ft = fault.fault_type

        if ft == "isl_disruption":
            # targets: list of "sat_a:sat_b" strings.
            # Zero the reliability (used by z_kv) AND drop the ISL edge from the
            # epoch graphs so earliest_arrival — which ignores reliability — can
            # no longer route the tile over the disrupted link at full latency.
            disrupted = set()
            for link_str in fault.targets:
                a, b = link_str.split(":")
                self.reliability.disable_link(a, b)
                self.reliability.disable_link(b, a)
                disrupted.add((a, b))
                disrupted.add((b, a))
            self._remove_edges(fault, lambda na, nb: (na, nb) in disrupted)

        elif ft == "plane_outage":
            # targets: list of satellite IDs in the affected plane
            for sat_id in fault.targets:
                if sat_id in self.states:
                    self.states[sat_id].inject_failure()
                self.reliability.set_node_pi(sat_id, 0.0)

        elif ft == "helper_failure":
            for sat_id in fault.targets:
                if sat_id in self.states:
                    self.states[sat_id].inject_failure()
                self.reliability.set_node_pi(sat_id, 0.0)

        elif ft == "straggler":
            factor = fault.params.get("factor", 0.1)
            snapshot = self._compute_snapshots.setdefault(id(fault), {})
            for sat_id in fault.targets:
                if sat_id in self.states:
                    snapshot[sat_id] = self.states[sat_id].C_i
                    self.states[sat_id].C_i *= factor

        elif ft == "ground_contact_miss":
            # Remove sat→ground edges from epoch graphs for the fault duration
            # so both feasibility routing and the realized-MC layer see no downlink.
            gs = self.gs_names
            tgt = set(fault.targets)
            self._remove_edges(
                fault,
                lambda na, nb: (na in tgt and nb in gs) or (nb in tgt and na in gs),
            )

        elif ft == "downlink_adverse":
            # Adverse weather / degraded ground contact: the aggregator can still
            # route (edge stays up) but its downlink succeeds with reduced π.
            adv_pi = fault.params.get("pi", DEFAULT_DOWNLINK_ADV_PI)
            for sat_id in fault.targets:
                self.reliability.set_downlink_pi(sat_id, adv_pi)

        elif ft == "battery_shortage":
            for sat_id in fault.targets:
                if sat_id in self.states:
                    s = self.states[sat_id]
                    s.B_i = s.params.battery_min_j * 0.5   # below minimum
                    s._update_availability()

        elif ft == "thermal_throttle":
            for sat_id in fault.targets:
                if sat_id in self.states:
                    s = self.states[sat_id]
                    s.Theta_i = s.params.thermal_max_c + 5.0
                    s.C_i = s._throttled_compute_rate()
                    s._update_availability()

    def _withdraw(self, fault: FaultEvent):
        ft = fault.fault_type

        if ft == "isl_disruption":
            for link_str in fault.targets:
                a, b = link_str.split(":")
                # Restore to defaults
                if (a, b) in self.reliability._link_overrides:
                    del self.reliability._link_overrides[(a, b)]
                if (b, a) in self.reliability._link_overrides:
                    del self.reliability._link_overrides[(b, a)]
            self._restore_edges(fault)

        elif ft in ("plane_outage", "helper_failure"):
            for sat_id in fault.targets:
                if sat_id in self.states:
                    self.states[sat_id].recover()
                if sat_id in self.reliability._node_overrides:
                    del self.reliability._node_overrides[sat_id]

        elif ft == "straggler":
            # Restore the exact pre-fault C_i captured at apply time, rather than
            # inverting the multiply (which loses state if throttling changed C_i
            # during the fault window or if straggler windows overlapped).
            snapshot = self._compute_snapshots.pop(id(fault), {})
            for sat_id in fault.targets:
                if sat_id in self.states and sat_id in snapshot:
                    self.states[sat_id].C_i = snapshot[sat_id]

        elif ft == "ground_contact_miss":
            self._restore_edges(fault)

        elif ft == "downlink_adverse":
            for sat_id in fault.targets:
                self.reliability._downlink_overrides.pop(sat_id, None)

        elif ft == "battery_shortage":
            for sat_id in fault.targets:
                if sat_id in self.states:
                    s = self.states[sat_id]
                    # Restore battery above minimum so _update_availability sets A_i=1.
                    s.B_i = s.params.battery_min_j * 1.5
                    s.recover()

        elif ft == "thermal_throttle":
            for sat_id in fault.targets:
                if sat_id in self.states:
                    s = self.states[sat_id]
                    # Restore temperature below throttle threshold.
                    s.Theta_i = s.params.thermal_ambient_c
                    s.C_i = s._throttled_compute_rate()
                    s.recover()

    # ── epoch-graph edge removal/restore (shared by graph-mutating faults) ─────

    def _remove_edges(self, fault: FaultEvent, drop):
        """Remove every edge (a→b) with drop(a, b) True from the epoch graphs in
        [start_epoch, end_epoch), recording them so _restore_edges can replay.
        `drop` is a predicate over ordered endpoints; pass a symmetric predicate
        to remove both directions of a link."""
        fault_id = id(fault)
        self._removed_edges[fault_id] = {}
        for ep_idx in range(fault.start_epoch, fault.end_epoch):
            if ep_idx >= len(self.graphs):
                continue
            g = self.graphs[ep_idx]
            kept, removed = [], []
            for edge in g.edges:
                if drop(edge[0], edge[1]):
                    removed.append(edge)
                else:
                    kept.append(edge)
            if removed:
                self._removed_edges[fault_id][ep_idx] = removed
                g.edges = kept
                adj: Dict[str, list] = {}
                for (na, nb, rate, cap, _t) in kept:
                    adj.setdefault(na, []).append((nb, rate, cap))
                g.adj = adj

    def _restore_edges(self, fault: FaultEvent):
        """Replay the edges removed by _remove_edges for this fault and rebuild
        each affected graph's adjacency list."""
        removed_by_ep = self._removed_edges.pop(id(fault), {})
        for ep_idx, edges in removed_by_ep.items():
            if ep_idx < len(self.graphs):
                g = self.graphs[ep_idx]
                g.edges.extend(edges)
                adj: Dict[str, list] = {}
                for (na, nb, rate, cap, _t) in g.edges:
                    adj.setdefault(na, []).append((nb, rate, cap))
                g.adj = adj

    # ── convenience factory methods ───────────────────────────────────────────

    @staticmethod
    def isl_disruption(sat_a: str, sat_b: str, start_epoch: int,
                       duration: int = 3) -> FaultEvent:
        return FaultEvent("isl_disruption", start_epoch, duration,
                          [f"{sat_a}:{sat_b}"])

    @staticmethod
    def plane_outage(sat_ids: List[str], start_epoch: int,
                     duration: int = 5) -> FaultEvent:
        return FaultEvent("plane_outage", start_epoch, duration, sat_ids)

    @staticmethod
    def helper_failure(sat_id: str, start_epoch: int,
                       duration: int = 2) -> FaultEvent:
        return FaultEvent("helper_failure", start_epoch, duration, [sat_id])

    @staticmethod
    def straggler(sat_id: str, start_epoch: int, duration: int = 1,
                  factor: float = 0.1) -> FaultEvent:
        return FaultEvent("straggler", start_epoch, duration, [sat_id],
                          {"factor": factor})

    @staticmethod
    def ground_contact_miss(agg_sat: str, start_epoch: int,
                            duration: int = 2) -> FaultEvent:
        return FaultEvent("ground_contact_miss", start_epoch, duration, [agg_sat])

    @staticmethod
    def battery_shortage(sat_id: str, start_epoch: int,
                         duration: int = 3) -> FaultEvent:
        return FaultEvent("battery_shortage", start_epoch, duration, [sat_id])

    @staticmethod
    def thermal_throttle(sat_id: str, start_epoch: int,
                         duration: int = 2) -> FaultEvent:
        return FaultEvent("thermal_throttle", start_epoch, duration, [sat_id])

    @staticmethod
    def downlink_adverse(agg_sat: str, start_epoch: int, duration: int = 5,
                         pi: float = DEFAULT_DOWNLINK_ADV_PI) -> FaultEvent:
        return FaultEvent("downlink_adverse", start_epoch, duration, [agg_sat],
                          {"pi": pi})


def random_fault_schedule(
    sat_ids: List[str],
    n_epochs: int,
    fault_rate: float = 0.05,   # probability of a fault occurring per epoch
    seed: int = 42,
) -> List[FaultEvent]:
    """
    Generate a randomized fault schedule at a given fault_rate.
    Used for E3 (fault intensity sweep).
    """
    rng = random.Random(seed)
    fault_types = [
        "helper_failure", "straggler", "battery_shortage",
        "thermal_throttle", "isl_disruption",
    ]
    faults = []
    for epoch in range(n_epochs):
        if rng.random() < fault_rate:
            ft = rng.choice(fault_types)
            target = rng.choice(sat_ids)
            if ft == "isl_disruption":
                other = rng.choice([s for s in sat_ids if s != target])
                faults.append(FaultEvent(ft, epoch, 2, [f"{target}:{other}"]))
            else:
                faults.append(FaultEvent(ft, epoch, rng.randint(1, 3), [target]))
    return faults
