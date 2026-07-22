"""SECO-aligned processing baseline with orbital resource constraints.

This implements the processing portion of SECO in the experiment's existing
post-capture setting.  Each captured tile may be spatially split, and the
baseline jointly chooses routes, compute helpers, result aggregators, and a
ground route to minimize completion time.  Unlike ORDI it uses no utility or
reliability term and creates no replicas.

The extension over SECO's nominal processing model is explicit feasibility
against satellite availability, time-varying contacts, and residual
link/compute capacity. Basilisk alone evolves physical energy state.
"""
from __future__ import annotations

from bisect import insort
from dataclasses import dataclass, replace
import heapq
import math

from .schema import Assignment, Decision
from ._common import (
    Placement, advertisement_metadata, group_success, protocol_trace,
)
from ordi.eval.validation import InvalidDecisionError
from ordi.sim.messaging import MessageSimulator


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
    terminal_intervals: dict[str, list[tuple[float, float]]]

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
        return cls(ready, contact_ready, residual, {})

    def clone(self):
        return ResourceLedger(
            self.compute_ready_at.copy(),
            self.contact_ready_at.copy(),
            self.contact_residual_bits.copy(),
            {
                terminal: list(intervals)
                for terminal, intervals in self.terminal_intervals.items()
            },
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
    modeled_placement: Placement


@dataclass(frozen=True)
class SplitPlan:
    split_count: int
    work_fraction: float
    input_fraction: float
    output_fraction: float
    parts: tuple[PartPlacement, ...]
    completion: float
    ledger_after: ResourceLedger


_CONTACT_INDEX_CACHE = {}


def _contacts_by_source(contacts):
    """Index one immutable local contact view once for all route searches."""
    key = id(contacts)
    cached = _CONTACT_INDEX_CACHE.get(key)
    if cached is not None and cached[0] is contacts:
        return cached[1]
    indexed = {}
    for index, contact in enumerate(contacts):
        indexed.setdefault(contact.source, []).append((index, contact))
    result = {
        source: tuple(sorted(windows, key=lambda item: item[1].opens))
        for source, windows in indexed.items()
    }
    if len(_CONTACT_INDEX_CACHE) >= 128:
        _CONTACT_INDEX_CACHE.clear()
    _CONTACT_INDEX_CACHE[key] = (contacts, result)
    return result


def _speculative_terminal_slot(calendars, terminals, earliest, duration, latest):
    """Earliest common terminal gap over SECO's sorted local calendars."""
    first = calendars.get(terminals[0], ()) if terminals else ()
    second = calendars.get(terminals[1], ()) if len(terminals) > 1 else ()
    i = j = 0
    start = earliest
    while i < len(first) or j < len(second):
        if j >= len(second) or (i < len(first) and first[i] <= second[j]):
            reserved_start, reserved_finish = first[i]
            i += 1
        else:
            reserved_start, reserved_finish = second[j]
            j += 1
        if reserved_finish <= start + 1e-9:
            continue
        if start + duration <= reserved_start + 1e-9:
            break
        start = reserved_finish
    return start if start + duration <= latest + 1e-9 else None


def _route(request, ledger, source, targets, bits, start):
    """Earliest route whose contacts still have time and bit capacity."""
    targets = set(targets)
    if source in targets:
        return ReservedRoute(start, 1.0, (source,), (), (), bits)

    by_source = _contacts_by_source(request.contacts)

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
            duration = bits / max(contact.rate_bps, 1.0)
            terminals = tuple(
                endpoint for endpoint in (contact.source, contact.target)
                if endpoint in request.satellites
            )
            depart = _speculative_terminal_slot(
                ledger.terminal_intervals, terminals,
                max(now, contact.opens, ledger.contact_ready_at[index]),
                duration, contact.closes,
            )
            if depart is None:
                continue
            finish = depart + duration
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
        contact = request.contacts[index]
        start = finish - route.bits / max(contact.rate_bps, 1.0)
        if contact.source in request.satellites:
            insort(
                ledger.terminal_intervals.setdefault(contact.source, []),
                (start, finish),
            )
        if contact.target in request.satellites:
            insort(
                ledger.terminal_intervals.setdefault(contact.target, []),
                (start, finish),
            )


def _relaxed_arrivals(request, ledger, source, bits, start):
    """Layered-graph arrival bounds without speculative terminal calendars.

    This inexpensive routing layer shortlists compute nodes. It respects
    contact windows and residual per-contact capacity, while leaving terminal
    contention to the exact ledger evaluation below.
    """
    best = {source: start}
    queue = [(start, source)]
    by_source = _contacts_by_source(request.contacts)
    while queue:
        now, node = heapq.heappop(queue)
        if now != best[node]:
            continue
        for index, contact in by_source.get(node, ()):
            if ledger.contact_residual_bits[index] + 1e-9 < bits:
                continue
            duration = bits / max(contact.rate_bps, 1.0)
            finish = max(
                now, contact.opens, ledger.contact_ready_at[index]
            ) + duration
            if finish > contact.closes + 1e-9:
                continue
            if finish < best.get(contact.target, math.inf):
                best[contact.target] = finish
                heapq.heappush(queue, (finish, contact.target))
    return best


def _relaxed_reverse_costs(request, ledger, bits, targets):
    """Optimistic serialization cost from every node to any target."""
    reverse = {}
    for index, contact in enumerate(request.contacts):
        if ledger.contact_residual_bits[index] + 1e-9 < bits:
            continue
        duration = bits / max(contact.rate_bps, 1.0)
        if contact.opens + duration > contact.closes + 1e-9:
            continue
        reverse.setdefault(contact.target, []).append(
            (contact.source, duration)
        )
    best = {target: 0.0 for target in targets}
    queue = [(0.0, target) for target in targets]
    heapq.heapify(queue)
    while queue:
        cost, node = heapq.heappop(queue)
        if cost != best[node]:
            continue
        for predecessor, duration in reverse.get(node, ()):
            candidate = cost + duration
            if candidate < best.get(predecessor, math.inf):
                best[predecessor] = candidate
                heapq.heappush(queue, (candidate, predecessor))
    return best


def _relaxed_ground_costs(request, ledger, bits):
    return _relaxed_reverse_costs(
        request, ledger, bits, request.ground_stations
    )


class SECOAdapted:
    """Queue-aware, capacity-aware non-redundant shard placement."""

    name = "seco_adapted"

    def __init__(self, split_options=(1, 2, 4), halo_fraction=0.05,
                 candidate_limit=4):
        self.split_options = tuple(sorted(set(split_options)))
        self.halo_fraction = halo_fraction
        self.candidate_limit = max(1, int(candidate_limit))
        self.messages = MessageSimulator()

    def _rank_helpers(self, request, task, tile, ledger, work_fraction,
                      input_fraction, output_fraction,
                      excluded_helpers=frozenset()):
        """Rank the compute layer using relaxed input/compute/output costs."""
        header_bits = self.messages.header_bits
        request_arrivals = _relaxed_arrivals(
            request, ledger, task.source_sat, header_bits,
            request.sim_time,
        )
        response_costs = _relaxed_reverse_costs(
            request, ledger, header_bits, (task.source_sat,)
        )
        arrivals = _relaxed_arrivals(
            request, ledger, task.source_sat,
            tile.d_in_bits * input_fraction + header_bits,
            request.sim_time,
        )
        ground_costs = _relaxed_ground_costs(
            request, ledger,
            tile.d_out_bits * output_fraction + header_bits,
        )
        ranked = []
        work = tile.compute_ops * work_fraction
        for helper, state in request.satellites.items():
            if not state.available or helper in excluded_helpers:
                continue
            arrival = arrivals.get(helper)
            ground_cost = ground_costs.get(helper)
            request_arrival = request_arrivals.get(helper)
            response_cost = response_costs.get(helper)
            if (
                arrival is None or ground_cost is None
                or request_arrival is None or response_cost is None
            ):
                continue
            # Include both halves of the split handshake in the compute-layer
            # lower bound. Exact contact timing and contention remain the
            # responsibility of _part_candidate.
            arrival = max(
                arrival, request_arrival + response_cost
            )
            compute_done = max(
                arrival, ledger.compute_ready_at[helper]
            ) + work / max(state.compute_rate, 1.0)
            ranked.append((compute_done + ground_cost, helper))
        ranked.sort()
        # The layered estimate deliberately ignores terminal contention and
        # downlink-window waiting, so it is an optimistic completion bound.
        # If that bound already misses the deadline, exact route reservation
        # cannot recover the helper and would only waste planning time.
        return tuple(
            helper for completion, helper in ranked
            if completion <= task.deadline + 1e-9
        )

    def _part_candidate(self, request, task, tile, ledger, work_fraction,
                        input_fraction, output_fraction,
                        excluded_helpers=frozenset(), candidate_helpers=None):
        best = None
        source_state = request.satellites.get(task.source_sat)
        if source_state is None or not source_state.available:
            return None

        input_bits = tile.d_in_bits * input_fraction
        output_bits = tile.d_out_bits * output_fraction
        header_bits = self.messages.header_bits
        work = tile.compute_ops * work_fraction

        helpers = (
            request.satellites
            if candidate_helpers is None else candidate_helpers
        )
        for helper in helpers:
            hstate = request.satellites[helper]
            if not hstate.available or helper in excluded_helpers:
                continue
            trial = ledger.clone()
            split_request = _route(
                request, trial, task.source_sat, {helper}, header_bits,
                request.sim_time,
            )
            if split_request is None:
                continue
            _reserve_route(request, trial, split_request)
            split_response = _route(
                request, trial, helper, {task.source_sat}, header_bits,
                split_request.arrival,
            )
            if split_response is None:
                continue
            _reserve_route(request, trial, split_response)
            route_in = _route(
                request, trial, task.source_sat, {helper},
                input_bits + header_bits, split_response.arrival,
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

            # SECO computes and assembles this part at the helper. Subsequent
            # satellites on route_down are store-and-forward relays; they need
            # not expose live resource state to the source policy.
            route_down = _route(
                request, trial, helper, request.ground_stations,
                output_bits + header_bits, compute_done,
            )
            if route_down is None or route_down.arrival > task.deadline:
                continue
            _reserve_route(request, trial, route_down)
            aggregator = helper
            route_out = ReservedRoute(
                compute_done, 1.0, (helper,), (), (), output_bits
            )

            modeled = Placement(
                helper, aggregator,
                route_down.arrival - request.sim_time, 0.0, 0.0,
                route_in.path, route_out.path, route_down.path,
                request.sim_time, route_in.arrival, compute_done,
                route_out.arrival, route_down.arrival,
            )
            part = PartPlacement(
                helper, aggregator, route_in, route_out, route_down,
                route_down.arrival,
                group_success(request, task, (modeled,)), modeled,
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
        # One fixed, handshake-aware compute-layer shortlist per split plan.
        # Exact feasibility is deliberately bounded: a sparse contact graph
        # must not turn a failed first batch back into an all-helper scan.
        ranked = self._rank_helpers(
            request, task, tile, trial, work_fraction,
            input_fraction, output_fraction,
        )
        shortlist_size = max(self.candidate_limit, 2 * split_count)
        shortlist = ranked[:shortlist_size]
        if len(shortlist) < split_count:
            return None
        for _ in range(split_count):
            excluded = frozenset(part.helper for part in parts)
            candidates = tuple(
                helper for helper in shortlist if helper not in excluded
            )
            chosen = self._part_candidate(
                request, task, tile, trial, work_fraction,
                input_fraction, output_fraction, excluded, candidates,
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
        advertisements = self.messages.prepare_epoch(request)
        assignments = []
        by_source = {}
        for task in request.tasks:
            by_source.setdefault(task.source_sat, []).append(task)
        for source, tasks in by_source.items():
            local = self.messages.local_view(request, source)
            ledger = ResourceLedger.from_request(local)
            # The local planner knows the residual capacity consumed by state
            # advertisements sent before this decision.
            for index, contact in enumerate(local.contacts):
                key = (
                    contact.source, contact.target, contact.opens,
                    contact.closes, contact.rate_bps, contact.kind,
                )
                if key in self.messages.contact_residual_bits:
                    ledger.contact_residual_bits[index] = (
                        self.messages.contact_residual_bits[key]
                    )
                if key in self.messages.contact_ready_at:
                    ledger.contact_ready_at[index] = (
                        self.messages.contact_ready_at[key]
                    )
            unscheduled = [
                (task, tile) for task in tasks for tile in task.tiles
            ]
            max_compute_rate = max(
                (state.compute_rate for state in local.satellites.values()
                 if state.available), default=1.0
            )
            max_link_rate = max(
                (contact.rate_bps for contact in local.contacts), default=1.0
            )
            unscheduled.sort(key=lambda pair: (
                pair[1].compute_ops / max_compute_rate
                + (pair[1].d_in_bits + pair[1].d_out_bits) / max_link_rate,
                pair[0].deadline, pair[0].task_id, pair[1].tile_id,
            ))

            for task, tile in unscheduled:
                if task.deadline <= request.sim_time + 1e-9:
                    continue
                plan = self._best_plan(local, task, tile, ledger)
                if plan is None:
                    continue
                reliability = group_success(
                    local, task,
                    tuple(part.modeled_placement for part in plan.parts),
                )
                q = plan.split_count
                assignment = Assignment(
                    task.task_id, tile.tile_id, source,
                    tuple(part.helper for part in plan.parts),
                    tuple(part.aggregator for part in plan.parts),
                    metadata={
                        "latency": plan.completion - request.sim_time,
                        "reliability": reliability,
                        "data_shards": q,
                        "split_count": q,
                        "partitioned": q > 1,
                        "effective_replicas": 1.0,
                        "time_objective": plan.completion - request.sim_time,
                        "planning_method": "layered_graph_shortlist",
                        "candidate_limit": self.candidate_limit,
                        "helper_handshake": True,
                        "helper_request_kind": "split",
                        "state_observer": source,
                        "known_state_nodes": len(local.satellites),
                        "max_state_age_s": max(
                            local.state_age_s.values(), default=0.0
                        ),
                    },
                    routes=tuple(
                        (part.route_in.path, part.route_out.path,
                         part.route_down.path)
                        for part in plan.parts
                    ),
                    work_fractions=(plan.work_fraction,) * q,
                    input_fractions=(plan.input_fraction,) * q,
                    output_fractions=(plan.output_fraction,) * q,
                    node_decisions=protocol_trace(
                        local, task, tile, (plan.parts,),
                        plan.work_fraction, plan.input_fraction,
                        plan.output_fraction,
                    ),
                )
                try:
                    execution = self.messages.execute(
                        request, task, tile, assignment
                    )
                except InvalidDecisionError:
                    continue
                ledger = plan.ledger_after
                metadata = dict(assignment.metadata)
                metadata.update({
                    "latency": execution.delivery_time - request.sim_time,
                    "protocol_header_bits": self.messages.header_bits,
                    "protocol_message_count": execution.message_count,
                    "protocol_control_bits": execution.control_bits,
                    "protocol_ground_bits": execution.ground_bits,
                    "handshake_control_bits": sum(
                        event.bits for event in execution.events
                        if event.event == "hop_sent"
                        and event.kind in {
                            "split_request", "split_accept", "split_reject"
                        }
                    ),
                })
                assignments.append(replace(
                    assignment, metadata=metadata,
                    message_events=execution.events,
                ))
        return Decision(
            request.epoch, tuple(assignments),
            advertisement_metadata(advertisements),
            advertisements.events,
        )
