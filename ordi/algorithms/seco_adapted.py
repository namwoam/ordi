"""SECO-aligned processing baseline with orbital resource constraints.

This implements the processing portion of SECO in the experiment's existing
post-capture setting.  Each captured tile may be spatially split, and the
baseline jointly chooses routes, compute helpers, result aggregators, and a
ground route to minimize completion time.  Unlike ORDI it uses no utility or
reliability term and creates no replicas.

The extension over SECO's nominal processing model is explicit feasibility
against satellite availability, battery reserve, time-varying contacts, and
residual link/compute capacity.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import math

from .schema import Assignment, Decision
from ordi.eval.validation import DecisionFeasibilityModel, InvalidDecisionError


@dataclass(frozen=True)
class ReservedRoute:
    arrival: float
    reliability: float
    path: tuple[str, ...]
    contact_indices: tuple[int, ...]
    contact_finishes: tuple[float, ...]
    bits: float


@dataclass
class ResourceLedger:
    """Resources committed by decisions made in the current scheduling pass."""

    compute_ready_at: dict[str, float]
    contact_ready_at: dict[int, float]
    contact_residual_bits: dict[int, float]
    energy_committed_j: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_request(cls, request):
        ready = {
            sid: request.sim_time
            + state.queued_flops / max(state.compute_rate, 1.0)
            for sid, state in request.satellites.items()
        }
        contact_ready = {
            index: max(request.sim_time, contact.opens)
            for index, contact in enumerate(request.contacts)
        }
        residual = {
            index: max(0.0, contact.closes - contact.opens)
            * max(contact.rate_bps, 0.0)
            for index, contact in enumerate(request.contacts)
        }
        return cls(ready, contact_ready, residual)

    def clone(self):
        return ResourceLedger(
            self.compute_ready_at.copy(),
            self.contact_ready_at.copy(),
            self.contact_residual_bits.copy(),
            self.energy_committed_j.copy(),
        )


@dataclass(frozen=True)
class PartPlacement:
    helper: str
    aggregator: str
    route_in: ReservedRoute
    route_out: ReservedRoute
    route_down: ReservedRoute
    completion: float
    reliability: float
    energy_j: float


@dataclass(frozen=True)
class SplitPlan:
    split_count: int
    work_fraction: float
    input_fraction: float
    output_fraction: float
    parts: tuple[PartPlacement, ...]
    completion: float
    ledger_after: ResourceLedger


def _route(request, ledger, source, targets, bits, start):
    """Earliest route whose contacts still have time and bit capacity."""
    targets = set(targets)
    if source in targets:
        return ReservedRoute(start, 1.0, (source,), (), (), bits)

    by_source: dict[str, list[tuple[int, object]]] = {}
    for index, contact in enumerate(request.contacts):
        by_source.setdefault(contact.source, []).append((index, contact))
    for windows in by_source.values():
        windows.sort(key=lambda item: item[1].opens)

    best = {source: start}
    reliability = {source: 1.0}
    paths = {source: (source,)}
    used = {source: ()}
    finishes = {source: ()}
    queue = [(start, source)]
    while queue:
        now, node = heapq.heappop(queue)
        if now != best[node]:
            continue
        if node in targets:
            return ReservedRoute(
                now, reliability[node], paths[node], used[node],
                finishes[node], bits
            )
        for index, contact in by_source.get(node, ()):
            if ledger.contact_residual_bits[index] + 1e-9 < bits:
                continue
            depart = max(
                now, contact.opens, ledger.contact_ready_at[index]
            )
            finish = depart + bits / max(contact.rate_bps, 1.0)
            if finish > contact.closes:
                continue
            if finish < best.get(contact.target, math.inf):
                best[contact.target] = finish
                reliability[contact.target] = (
                    reliability[node] * contact.reliability
                )
                paths[contact.target] = paths[node] + (contact.target,)
                used[contact.target] = used[node] + (index,)
                finishes[contact.target] = finishes[node] + (finish,)
                heapq.heappush(queue, (finish, contact.target))
    return None


def _reserve_route(request, ledger, route):
    for index, finish in zip(route.contact_indices, route.contact_finishes):
        ledger.contact_residual_bits[index] -= route.bits
        ledger.contact_ready_at[index] = finish


def _route_energy_by_node(request, route):
    energy: dict[str, float] = {}
    for index in route.contact_indices:
        contact = request.contacts[index]
        airtime = route.bits / max(contact.rate_bps, 1.0)
        # Charge both spacecraft radios on an ISL; a ground receiver is outside
        # the modeled satellite energy budget.
        for node in (contact.source, contact.target):
            state = request.satellites.get(node)
            if state is not None:
                energy[node] = energy.get(node, 0.0) + (
                    state.comms_power_w * airtime
                )
    return energy


def _merge_energy(*items):
    merged: dict[str, float] = {}
    for item in items:
        for node, amount in item.items():
            merged[node] = merged.get(node, 0.0) + amount
    return merged


class SECOAdapted:
    """Queue-aware, capacity-aware non-redundant shard placement."""

    name = "seco_adapted"

    def __init__(self, split_options=(1, 2, 4), halo_fraction=0.05,
                 battery_reserve_frac=0.15):
        self.split_options = tuple(sorted(set(split_options)))
        self.halo_fraction = halo_fraction
        self.battery_reserve_frac = battery_reserve_frac
        # The SECO planning ledger below handles contention within one call.
        # This persistent model ledger also protects future contacts and compute
        # slots already committed by decisions from earlier epochs.
        self.resources = DecisionFeasibilityModel()

    def _part_candidate(self, request, task, tile, ledger, work_fraction,
                        input_fraction, output_fraction):
        best = None
        source_state = request.satellites.get(task.source_sat)
        if source_state is None or not source_state.available:
            return None

        input_bits = tile.d_in_bits * input_fraction
        output_bits = tile.d_out_bits * output_fraction
        work = tile.compute_ops * work_fraction

        for helper, hstate in request.satellites.items():
            if not hstate.available:
                continue
            trial = ledger.clone()
            route_in = _route(
                request, trial, task.source_sat, {helper}, input_bits,
                request.sim_time,
            )
            if route_in is None:
                continue
            _reserve_route(request, trial, route_in)

            compute_start = max(
                route_in.arrival, trial.compute_ready_at[helper]
            )
            compute_done = compute_start + work / max(
                hstate.compute_rate, 1.0
            )
            trial.compute_ready_at[helper] = compute_done

            # The time-dependent path implicitly chooses all result relays and
            # the final downlink satellite.  Represent that last satellite as
            # the aggregator instead of redundantly enumerating every node.
            route_down = _route(
                request, trial, helper, request.ground_stations,
                output_bits, compute_done,
            )
            if route_down is None or route_down.arrival > task.deadline:
                continue
            _reserve_route(request, trial, route_down)
            aggregator = (route_down.path[-2]
                          if len(route_down.path) >= 2 else helper)
            route_out = ReservedRoute(
                compute_done, 1.0, (helper,), (), (), output_bits
            )

            compute_energy = {
                helper: hstate.compute_power_w * work
                / max(hstate.compute_rate, 1.0)
            }
            energy_by_node = _merge_energy(
                compute_energy,
                _route_energy_by_node(request, route_in),
                _route_energy_by_node(request, route_down),
            )
            physically_feasible = True
            for node, energy in energy_by_node.items():
                state = request.satellites[node]
                already = trial.energy_committed_j.get(node, 0.0)
                reserve = self.battery_reserve_frac * state.battery_capacity_j
                if state.battery_j - already - energy < reserve:
                    physically_feasible = False
                    break
            if not physically_feasible:
                continue
            for node, energy in energy_by_node.items():
                trial.energy_committed_j[node] = (
                    trial.energy_committed_j.get(node, 0.0) + energy
                )

            participating = {helper, aggregator} - {task.source_sat}
            node_reliability = math.prod(
                request.satellites[node].reliability
                for node in participating
            )
            part = PartPlacement(
                helper, aggregator, route_in, route_out, route_down,
                route_down.arrival,
                route_in.reliability * route_down.reliability
                * node_reliability,
                sum(energy_by_node.values()),
            )
            if best is None or part.completion < best[0].completion:
                best = (part, trial)
        return best

    def _split_plan(self, request, task, tile, ledger, split_count):
        output_fraction = 1.0 / split_count
        # Each internal spatial boundary adds a halo.  Total transferred input
        # and compute become 1 + halo*(q-1), avoiding unrealistically free splits.
        input_fraction = (
            1.0 + self.halo_fraction * (split_count - 1)
        ) / split_count
        work_fraction = input_fraction
        trial = ledger.clone()
        parts = []
        for _ in range(split_count):
            chosen = self._part_candidate(
                request, task, tile, trial, work_fraction,
                input_fraction, output_fraction,
            )
            if chosen is None:
                return None
            part, trial = chosen
            parts.append(part)
        return SplitPlan(
            split_count, work_fraction, input_fraction, output_fraction,
            tuple(parts), max(part.completion for part in parts), trial,
        )

    def _best_plan(self, request, task, tile, ledger):
        plans = [
            self._split_plan(request, task, tile, ledger, count)
            for count in self.split_options
            if count > 0
        ]
        feasible = [plan for plan in plans if plan is not None]
        return min(feasible, key=lambda plan: plan.completion) if feasible else None

    def schedule(self, request):
        ledger = ResourceLedger.from_request(request)
        unscheduled = [
            (task, tile) for task in request.tasks for tile in task.tiles
        ]
        max_compute_rate = max(
            (state.compute_rate for state in request.satellites.values()
             if state.available), default=1.0
        )
        max_link_rate = max(
            (contact.rate_bps for contact in request.contacts), default=1.0
        )
        # Shortest-processing-time ordering is the greedy rule for minimizing
        # summed completion time.  Deadlines break ties but are not part of the
        # SECO time objective.
        unscheduled.sort(key=lambda pair: (
            pair[1].compute_ops / max_compute_rate
            + (pair[1].d_in_bits + pair[1].d_out_bits) / max_link_rate,
            pair[0].deadline, pair[0].task_id, pair[1].tile_id,
        ))
        assignments = []

        for task, tile in unscheduled:
            plan = self._best_plan(request, task, tile, ledger)
            if plan is None:
                continue
            ledger = plan.ledger_after
            source_pi = request.satellites[task.source_sat].reliability
            reliability = source_pi * math.prod(
                part.reliability for part in plan.parts
            )
            q = plan.split_count
            assignment = Assignment(
                task.task_id,
                tile.tile_id,
                task.source_sat,
                tuple(part.helper for part in plan.parts),
                tuple(part.aggregator for part in plan.parts),
                metadata={
                    "latency": plan.completion - request.sim_time,
                    "reliability": reliability,
                    "data_shards": q,
                    "split_count": q,
                    "partitioned": True,
                    "effective_replicas": 1.0,
                    "energy_j": sum(part.energy_j for part in plan.parts),
                    "includes_downlink_energy": True,
                    "time_objective": plan.completion - request.sim_time,
                },
                routes=tuple(
                    (part.route_in.path, part.route_out.path,
                     part.route_down.path)
                    for part in plan.parts
                ),
                work_fractions=(plan.work_fraction,) * q,
                input_fractions=(plan.input_fraction,) * q,
                output_fractions=(plan.output_fraction,) * q,
            )
            try:
                assignments.append(
                    self.resources.retime_and_reserve(request, assignment)
                )
            except InvalidDecisionError:
                continue
        return Decision(request.epoch, tuple(assignments))
