"""
Fault injection framework for the E2 and E3 robustness evaluations.

Supports eight fault types from the proposal:
  1. ISL disruption          - remove specific ISL edge for N epochs
  2. Orbital-plane outage    - disable all satellites in a plane for N epochs
  3. Helper failure          - flip A_i=0 for a specific satellite
  4. Straggler               - scale C_i by factor for a helper during execution
  5. Ground-contact miss     - remove downlink window for N epochs
  6. Battery shortage        - add a Basilisk electrical fault load
  7. Thermal throttling      - add Basilisk heat and throttle compute control
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


RANDOM_FAULT_TYPES = (
    "helper_failure", "straggler", "battery_shortage",
    "thermal_throttle", "isl_disruption", "ground_contact_miss",
    "downlink_adverse",
)


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
        # Active effects are rebuilt from these baselines after every transition.
        # This makes overlapping faults compositional: withdrawing one event can
        # never restore a component still covered by another active event.
        self._base_node_overrides = reliability._node_overrides.copy()
        self._base_link_overrides = reliability._link_overrides.copy()
        self._base_downlink_overrides = reliability._downlink_overrides.copy()
        self._base_failures = {
            sat_id: bool(getattr(state, "_injected_failure", False))
            for sat_id, state in states.items()
        }
        self._base_availability = {
            sat_id: int(bool(getattr(state, "A_i", 1)))
            for sat_id, state in states.items()
        }
        self._base_compute_multipliers = {
            sat_id: float(getattr(state, "_compute_rate_multiplier", 1.0))
            for sat_id, state in states.items()
        }
        self._base_thermal_multipliers = {
            sat_id: float(getattr(state, "_thermal_rate_multiplier", 1.0))
            for sat_id, state in states.items()
        }
        self._base_graphs = [
            (list(graph.edges), list(graph.edge_windows))
            for graph in self.graphs
        ]

    def schedule(self, fault: FaultEvent):
        """Register a fault to be applied at its start_epoch."""
        self._scheduled.append(fault)

    def disrupted_links(self) -> Set[Tuple[str, str]]:
        """Directed ISL edges disabled by currently active hard faults."""
        links = set()
        for fault in self._active:
            if fault.fault_type != "isl_disruption":
                continue
            for link_str in fault.targets:
                a, b = link_str.split(":")
                links.update(((a, b), (b, a)))
        return links

    def missed_downlink_satellites(self) -> Set[str]:
        """Satellites whose ground contact is currently unavailable."""
        return {
            sat_id
            for fault in self._active
            if fault.fault_type == "ground_contact_miss"
            for sat_id in fault.targets
        }

    def apply_epoch(self, epoch: int):
        """Apply all faults that start at this epoch."""
        for fault in self._scheduled:
            if fault.start_epoch == epoch:
                self._active.append(fault)
                self._apply(fault)
        self._refresh_active_effects(rebuild_graphs=True)

    def withdraw_epoch(self, epoch: int):
        """Withdraw faults whose duration has expired."""
        self._active = [f for f in self._active if f.end_epoch > epoch]
        self._refresh_active_effects(rebuild_graphs=True)

    def refresh_active_state(self):
        """Reassert state faults that a physical/telemetry update may overwrite.

        Faults are applied once at their start epoch, while battery, temperature,
        and compute rate evolve every epoch.  Reasserting only the affected state
        keeps multi-epoch faults active without replaying graph mutations or
        taking duplicate straggler snapshots.
        """
        self._refresh_active_effects(rebuild_graphs=False)

    def _refresh_active_effects(self, rebuild_graphs: bool) -> None:
        """Rebuild all reversible effects from the complete active-fault set."""
        self.reliability._node_overrides = self._base_node_overrides.copy()
        self.reliability._link_overrides = self._base_link_overrides.copy()
        self.reliability._downlink_overrides = (
            self._base_downlink_overrides.copy()
        )

        failed = set()
        compute_factors = {sat_id: 1.0 for sat_id in self.states}
        thermal_factors = {sat_id: 1.0 for sat_id in self.states}
        for fault in self._active:
            if fault.fault_type in {"plane_outage", "helper_failure"}:
                failed.update(fault.targets)
            elif fault.fault_type == "straggler":
                factor = max(0.0, float(fault.params.get("factor", 0.1)))
                for sat_id in fault.targets:
                    if sat_id in compute_factors:
                        compute_factors[sat_id] *= factor
            elif fault.fault_type == "thermal_throttle":
                factor = max(
                    0.0, min(1.0, float(fault.params.get("factor", 0.5)))
                )
                for sat_id in fault.targets:
                    if sat_id in thermal_factors:
                        thermal_factors[sat_id] *= factor
            elif fault.fault_type == "isl_disruption":
                for link_str in fault.targets:
                    a, b = link_str.split(":")
                    self.reliability._link_overrides[(a, b)] = 0.0
                    self.reliability._link_overrides[(b, a)] = 0.0
            elif fault.fault_type == "downlink_adverse":
                probability = float(
                    fault.params.get("pi", DEFAULT_DOWNLINK_ADV_PI)
                )
                for sat_id in fault.targets:
                    current = self.reliability._downlink_overrides.get(
                        sat_id, self.reliability.default_downlink_pi
                    )
                    self.reliability._downlink_overrides[sat_id] = min(
                        current, probability
                    )

        for sat_id in failed:
            self.reliability._node_overrides[sat_id] = 0.0
        for sat_id, state in self.states.items():
            state._injected_failure = (
                self._base_failures.get(sat_id, False) or sat_id in failed
            )
            state._compute_rate_multiplier = (
                self._base_compute_multipliers.get(sat_id, 1.0)
                * compute_factors[sat_id]
            )
            state._thermal_rate_multiplier = (
                self._base_thermal_multipliers.get(sat_id, 1.0)
                * thermal_factors[sat_id]
            )
            if hasattr(state, "_effective_compute_rate"):
                state.C_i = state._effective_compute_rate()
            if hasattr(state, "_update_availability"):
                state._update_availability()
            else:
                state.A_i = int(
                    self._base_availability.get(sat_id, 1)
                    and not state._injected_failure
                )

        if rebuild_graphs:
            self._rebuild_graphs()

    def _rebuild_graphs(self) -> None:
        for graph, (base_edges, base_windows) in zip(
                self.graphs, self._base_graphs):
            disrupted = set()
            missed = set()
            for fault in self._active:
                if not fault.start_epoch <= graph.epoch < fault.end_epoch:
                    continue
                if fault.fault_type == "isl_disruption":
                    for link_str in fault.targets:
                        a, b = link_str.split(":")
                        disrupted.update(((a, b), (b, a)))
                elif fault.fault_type == "ground_contact_miss":
                    missed.update(fault.targets)
            kept = []
            windows = []
            for edge, window in zip(base_edges, base_windows):
                source, target = edge[:2]
                if (source, target) in disrupted:
                    continue
                if ((source in missed and target in self.gs_names)
                        or (target in missed and source in self.gs_names)):
                    continue
                kept.append(edge)
                windows.append(window)
            graph.edges = kept
            graph.edge_windows = windows
            adjacency: Dict[str, list] = {}
            for source, target, rate, capacity, _kind in kept:
                adjacency.setdefault(source, []).append(
                    (target, rate, capacity)
                )
            graph.adj = adjacency

    def physical_workloads(self, epoch_length_s: float) -> Dict[str, dict]:
        """Return active fault loads for Basilisk's power/thermal nodes."""
        effects = {
            sat_id: {"power_w": 0.0, "heat_w": 0.0}
            for sat_id in self.states
        }
        duration = max(float(epoch_length_s), 1e-9)
        for fault in self._active:
            if fault.fault_type == "battery_shortage":
                for sat_id in fault.targets:
                    state = self.states.get(sat_id)
                    if state is None:
                        continue
                    target = state.params.battery_min_j * float(
                        fault.params.get("target_fraction_of_min", 0.5)
                    )
                    drain = max(0.0, state.B_i - target) / duration
                    # Counter nominal panel generation so Basilisk, rather than
                    # a state overwrite, determines the realized battery path.
                    drain += max(0.0, state.params.solar_power_w)
                    effects[sat_id]["power_w"] += float(
                        fault.params.get("power_w", drain)
                    )
            elif fault.fault_type == "thermal_throttle":
                for sat_id in fault.targets:
                    state = self.states.get(sat_id)
                    if state is None:
                        continue
                    heat = float(fault.params.get(
                        "heat_w", state.params.compute_power_w
                    ))
                    effects[sat_id]["power_w"] += max(0.0, heat)
                    effects[sat_id]["heat_w"] += max(0.0, heat)
        return effects

    # ── apply/withdraw per fault type ─────────────────────────────────────────

    def _apply(self, fault: FaultEvent):
        ft = fault.fault_type

        if ft == "isl_disruption":
            # targets: list of "sat_a:sat_b" strings.
            # Zero the reliability (used by z_kv) AND drop the ISL edge from the
            # epoch graphs so earliest_arrival — which ignores reliability — can
            # no longer route the tile over the disrupted link at full latency.
            for link_str in fault.targets:
                a, b = link_str.split(":")
                for epoch in range(fault.start_epoch, fault.end_epoch):
                    self.reliability.record_link_pi(a, b, epoch, 0.0)
                    self.reliability.record_link_pi(b, a, epoch, 0.0)

        elif ft == "plane_outage":
            # targets: list of satellite IDs in the affected plane
            for sat_id in fault.targets:
                for epoch in range(fault.start_epoch, fault.end_epoch):
                    self.reliability.record_node_pi(sat_id, epoch, 0.0)

        elif ft == "helper_failure":
            for sat_id in fault.targets:
                for epoch in range(fault.start_epoch, fault.end_epoch):
                    self.reliability.record_node_pi(sat_id, epoch, 0.0)

        elif ft == "straggler":
            pass

        elif ft == "ground_contact_miss":
            # Remove sat→ground edges from epoch graphs for the fault duration
            # so both feasibility routing and the realized-MC layer see no downlink.
            for sat_id in fault.targets:
                for epoch in range(fault.start_epoch, fault.end_epoch):
                    self.reliability.record_downlink_pi(
                        sat_id, epoch, 0.0
                    )

        elif ft == "downlink_adverse":
            # Adverse weather / degraded ground contact: the aggregator can still
            # route (edge stays up) but its downlink succeeds with reduced π.
            adv_pi = float(fault.params.get("pi", DEFAULT_DOWNLINK_ADV_PI))
            for sat_id in fault.targets:
                for epoch in range(fault.start_epoch, fault.end_epoch):
                    existing = self.reliability._downlink_history.get(
                        (sat_id, epoch), self.reliability.default_downlink_pi
                    )
                    self.reliability.record_downlink_pi(
                        sat_id, epoch, min(existing, adv_pi)
                    )

        elif ft == "battery_shortage":
            # The active fault is translated into a Basilisk power-sink load
            # by physical_workloads(); battery state is never assigned here.
            pass

        elif ft == "thermal_throttle":
            pass

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
    graphs: Optional[List[EpochContactGraph]] = None,
) -> List[FaultEvent]:
    """
    Generate a randomized fault schedule at a given fault_rate.
    Used for E2 (fault intensity sweep).
    """
    if not 0.0 <= fault_rate <= 1.0:
        raise ValueError("fault_rate must be between 0 and 1")
    rng = random.Random(seed)
    faults = []
    for epoch in range(n_epochs):
        # Draw the complete candidate before applying the intensity threshold.
        # With a shared seed, every lower-rate schedule is therefore a strict
        # subset of every higher-rate schedule instead of consuming a different
        # RNG stream after its first additional event.
        trigger = rng.random()
        ft = rng.choice(RANDOM_FAULT_TYPES)
        target = rng.choice(sat_ids)
        duration = rng.randint(1, 3)
        if ft == "isl_disruption":
            active_edges = []
            if graphs and epoch < len(graphs):
                active_edges = [
                    (a, b) for a, b, _rate, _cap, kind in graphs[epoch].edges
                    if kind == "isl" and a in sat_ids and b in sat_ids
                ]
            if active_edges:
                target, other = rng.choice(active_edges)
            else:
                other = rng.choice([s for s in sat_ids if s != target])
            candidate = FaultEvent(
                ft, epoch, 2, [f"{target}:{other}"]
            )
        elif ft == "ground_contact_miss":
            active_sources = []
            if graphs and epoch < len(graphs):
                active_sources = [
                    a for a, _b, _rate, _cap, kind in graphs[epoch].edges
                    if kind == "downlink" and a in sat_ids
                ]
            if active_sources:
                target = rng.choice(active_sources)
            candidate = FaultEvent(ft, epoch, duration, [target])
        else:
            candidate = FaultEvent(ft, epoch, duration, [target])
        if trigger < fault_rate:
            faults.append(candidate)
    return faults
