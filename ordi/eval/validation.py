"""Model-side validation of scheduler decisions against shared resources."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math

from ordi.algorithms.schema import Assignment, Decision, EpochInput


class InvalidDecisionError(ValueError):
    """Raised when a policy submits a decision the modeled system cannot run."""


def _terminal_slot(calendars, terminals, earliest, duration, latest):
    """Earliest gap shared by all participating spacecraft terminals."""
    intervals = sorted(
        interval for terminal in terminals
        for interval in calendars.get(terminal, ())
    )
    start = earliest
    for reserved_start, reserved_finish in intervals:
        if reserved_finish <= start + 1e-9:
            continue
        if start + duration <= reserved_start + 1e-9:
            break
        start = reserved_finish
    return start if start + duration <= latest + 1e-9 else None


def _contact_key(contact):
    return (
        contact.source, contact.target, contact.opens, contact.closes,
        contact.rate_bps, contact.kind,
    )


@dataclass
class DecisionFeasibilityModel:
    """Reserve network and compute resources independently of the policy.

    Reservations persist across scheduling epochs because a decision may use a
    future contact window.  Validation is transactional: an invalid decision
    does not alter the model ledger.
    """

    contact_ready_at: dict[tuple, float] = field(default_factory=dict)
    contact_residual_bits: dict[tuple, float] = field(default_factory=dict)
    compute_ready_at: dict[str, float] = field(default_factory=dict)
    ground_compute_ready_at: dict[str, float] = field(default_factory=dict)
    terminal_intervals: dict[str, list[tuple[float, float]]] = field(
        default_factory=dict
    )
    reservations: list[dict] = field(default_factory=list)

    def _copy(self):
        return DecisionFeasibilityModel(
            contact_ready_at=self.contact_ready_at.copy(),
            contact_residual_bits=self.contact_residual_bits.copy(),
            compute_ready_at=self.compute_ready_at.copy(),
            ground_compute_ready_at=self.ground_compute_ready_at.copy(),
            terminal_intervals={
                terminal: list(intervals)
                for terminal, intervals in self.terminal_intervals.items()
            },
            reservations=[record.copy() for record in self.reservations],
        )

    def _reserve_hop(self, request, source, target, bits, start, trace=None,
                     owner=None):
        candidates = sorted(
            (contact for contact in request.contacts
             if contact.source == source and contact.target == target),
            key=lambda contact: contact.opens,
        )
        for contact in candidates:
            key = _contact_key(contact)
            capacity = max(0.0, contact.closes - contact.opens) * max(
                contact.rate_bps, 0.0
            )
            residual = self.contact_residual_bits.get(key, capacity)
            if residual + 1e-9 < bits:
                continue
            duration = bits / max(contact.rate_bps, 1.0)
            terminals = tuple(
                endpoint for endpoint in (source, target)
                if endpoint in request.satellites
            )
            depart = _terminal_slot(
                self.terminal_intervals, terminals,
                max(start, contact.opens,
                    self.contact_ready_at.get(key, contact.opens)),
                duration, contact.closes,
            )
            if depart is None:
                continue
            finish = depart + duration
            self.contact_residual_bits[key] = residual - bits
            self.contact_ready_at[key] = finish
            for terminal in terminals:
                self.terminal_intervals.setdefault(terminal, []).append(
                    (depart, finish)
                )
            self.reservations.append({
                "owner": owner, "kind": "contact", "resource": key,
                "start": depart, "finish": finish, "amount": bits,
                "capacity": capacity,
                "terminal_resources": tuple(
                    endpoint for endpoint in (source, target)
                    if endpoint in request.satellites
                ),
            })
            if trace is not None:
                trace.append(
                    (source, target, depart, finish, contact.kind)
                )
            return finish
        raise InvalidDecisionError(
            f"no residual contact capacity for {source}->{target} "
            f"({bits:.3f} bits after t={start:.6f})"
        )

    def _reserve_path(self, request, path, bits, start, trace=None, owner=None):
        now = start
        for source, target in zip(path, path[1:]):
            now = self._reserve_hop(
                request, source, target, bits, now, trace=trace, owner=owner
            )
        return now

    def _reserve_compute(self, request, helper, work, start, trace=None,
                         owner=None):
        state = request.satellites.get(helper)
        if state is None or not state.available:
            raise InvalidDecisionError(
                f"compute helper {helper!r} is unavailable"
            )
        initial_ready = request.sim_time + state.queued_flops / max(
            state.compute_rate, 1.0
        )
        compute_start = max(
            start, initial_ready, self.compute_ready_at.get(helper, start)
        )
        finish = compute_start + work / max(state.compute_rate, 1.0)
        self.compute_ready_at[helper] = finish
        self.reservations.append({
            "owner": owner, "kind": "compute", "resource": helper,
            "start": compute_start, "finish": finish, "amount": work,
        })
        if trace is not None:
            trace.append((helper, compute_start, finish, work))
        return finish

    def _reserve_ground_compute(self, station, work, rate, start, owner=None):
        if not station:
            raise InvalidDecisionError("ground compute has no station")
        if rate <= 0.0:
            raise InvalidDecisionError(
                f"ground compute station {station!r} has no capacity"
            )
        compute_start = max(
            start, self.ground_compute_ready_at.get(station, start)
        )
        finish = compute_start + max(0.0, work) / rate
        self.ground_compute_ready_at[station] = finish
        self.reservations.append({
            "owner": owner, "kind": "ground_compute", "resource": station,
            "start": compute_start, "finish": finish, "amount": work,
        })
        return finish

    def cancel(self, owner, sim_time: float) -> None:
        """Release an assignment's unconsumed future reservations."""
        kept = []
        for record in self.reservations:
            if record["owner"] != owner or record["finish"] <= sim_time:
                kept.append(record)
                continue
            if record["start"] < sim_time and record["kind"] == "contact":
                duration = record["finish"] - record["start"]
                consumed = ((sim_time - record["start"]) / duration
                            if duration > 0.0 else 1.0)
                partial = record.copy()
                partial["finish"] = sim_time
                partial["amount"] *= max(0.0, min(1.0, consumed))
                kept.append(partial)
        self.reservations = kept
        self.contact_ready_at.clear()
        self.contact_residual_bits.clear()
        self.compute_ready_at.clear()
        self.ground_compute_ready_at.clear()
        self.terminal_intervals.clear()
        for record in kept:
            kind = record["kind"]
            resource = record["resource"]
            if kind == "contact":
                self.contact_residual_bits[resource] = (
                    self.contact_residual_bits.get(
                        resource, record["capacity"]
                    ) - record["amount"]
                )
                self.contact_ready_at[resource] = max(
                    self.contact_ready_at.get(resource, resource[2]),
                    record["finish"],
                )
                for terminal in record.get("terminal_resources", resource[:2]):
                    self.terminal_intervals.setdefault(terminal, []).append(
                        (record["start"], record["finish"])
                    )
            elif kind == "compute":
                self.compute_ready_at[resource] = max(
                    self.compute_ready_at.get(resource, 0.0), record["finish"]
                )
            elif kind == "ground_compute":
                self.ground_compute_ready_at[resource] = max(
                    self.ground_compute_ready_at.get(resource, 0.0),
                    record["finish"],
                )

    def validate_and_reserve(self, request: EpochInput, decision: Decision,
                             *, retime: bool = False):
        """Validate one decision and atomically reserve all accepted work."""
        trial = self._copy()
        accepted = []
        task_by_id = {task.task_id: task for task in request.tasks}
        tile_by_key = {
            (task.task_id, tile.tile_id): tile
            for task in request.tasks for tile in task.tiles
        }

        # State advertisements are physical ISL messages even when an epoch
        # has no science assignment. Reserve them before policy work so every
        # algorithm sees identical residual contact capacity.
        reserved_advertisements = set()
        for event in decision.message_events:
            reservation_key = (
                event.message_id, event.node, event.peer, event.time
            )
            if (event.kind != "state_advertisement"
                    or event.event != "hop_sent"
                    or reservation_key in reserved_advertisements):
                continue
            trial._reserve_hop(
                request, event.node, event.peer, event.bits, event.time
            )
            reserved_advertisements.add(reservation_key)

        for assignment in decision.assignments:
            key = (assignment.task_id, assignment.tile_id)
            task = task_by_id.get(assignment.task_id)
            tile = tile_by_key.get(key)
            if task is None or tile is None:
                raise InvalidDecisionError(
                    f"assignment references unknown task/tile {key}"
                )

            communication_intervals = []
            compute_intervals = []
            reserved_handshakes = set()
            for event in assignment.message_events:
                reservation_key = (
                    event.message_id, event.node, event.peer, event.time
                )
                if (event.event != "hop_sent"
                        or event.kind not in {
                            "split_request", "split_accept", "split_reject",
                            "replica_request", "replica_accept",
                            "replica_reject",
                        }
                        or reservation_key in reserved_handshakes):
                    continue
                trial._reserve_hop(
                    request, event.node, event.peer, event.bits, event.time,
                    trace=communication_intervals, owner=key,
                )
                reserved_handshakes.add(reservation_key)

            finishes = []
            replica_phase_ends = []
            replica_compute_rates = []
            source_release_time = None
            if assignment.downlink_only:
                path = tuple(assignment.metadata.get("path", ()))
                if len(path) < 2:
                    raise InvalidDecisionError(
                        f"direct-downlink assignment {key} has no route"
                    )
                bits = float(assignment.metadata.get(
                    "downlink_bits", tile.d_in_bits
                ))
                downlink_finish = trial._reserve_path(
                    request, path, bits, request.sim_time,
                    trace=communication_intervals, owner=key,
                )
                source_release_time = downlink_finish
                ground_work = float(assignment.metadata.get(
                    "ground_compute_flops", 0.0
                ))
                ground_rate = float(assignment.metadata.get(
                    "ground_compute_rate_flops_per_s", 0.0
                ))
                if ground_work <= 0.0 or ground_rate <= 0.0:
                    raise InvalidDecisionError(
                        f"direct-downlink assignment {key} does not include "
                        "ground inference"
                    )
                station = str(assignment.metadata.get(
                    "ground_station", path[-1]
                ))
                finishes.append(trial._reserve_ground_compute(
                    station, ground_work, ground_rate, downlink_finish,
                    owner=key,
                ))
            else:
                if len(assignment.routes) != len(assignment.helpers):
                    raise InvalidDecisionError(
                        f"assignment {key} does not provide one route per helper"
                    )
                if len(assignment.aggregators) != len(assignment.helpers):
                    raise InvalidDecisionError(
                        f"assignment {key} does not provide one aggregator per helper"
                    )
                if assignment.node_decisions:
                    allowed_actions = {
                        "execute_forward", "delegate", "split", "replicate"
                    }
                    for local in assignment.node_decisions:
                        if local.action not in allowed_actions:
                            raise InvalidDecisionError(
                                f"assignment {key} contains unknown node action "
                                f"{local.action!r}"
                            )
                        if local.node != local.item.current_node:
                            raise InvalidDecisionError(
                                f"assignment {key} has node {local.node!r} "
                                f"acting on work held by "
                                f"{local.item.current_node!r}"
                            )
                        if (local.item.task_id, local.item.tile_id) != key:
                            raise InvalidDecisionError(
                                f"assignment {key} contains a decision for a "
                                "different work item"
                            )
                    terminal = [
                        local for local in assignment.node_decisions
                        if local.action == "execute_forward"
                    ]
                    if len(terminal) != len(assignment.helpers):
                        raise InvalidDecisionError(
                            f"assignment {key} has {len(assignment.helpers)} "
                            f"compute operations but {len(terminal)} terminal "
                            "node decisions"
                        )
                    for index, local in enumerate(terminal):
                        work_fraction = (
                            assignment.work_fractions[index]
                            if index < len(assignment.work_fractions) else 1.0
                        )
                        input_fraction = (
                            assignment.input_fractions[index]
                            if index < len(assignment.input_fractions) else 1.0
                        )
                        output_fraction = (
                            assignment.output_fractions[index]
                            if index < len(assignment.output_fractions) else 1.0
                        )
                        route_down = assignment.routes[index][2]
                        destination = local.item.destination
                        destination_ok = (
                            not route_down
                            or destination == route_down[-1]
                            or (isinstance(destination, tuple)
                                and route_down[-1] in destination)
                        )
                        if (local.node != assignment.helpers[index]
                                or not math.isclose(
                                    local.item.work_fraction, work_fraction
                                )
                                or not math.isclose(
                                    local.item.input_fraction, input_fraction
                                )
                                or not math.isclose(
                                    local.item.output_fraction, output_fraction
                                )
                                or not destination_ok):
                            raise InvalidDecisionError(
                                f"assignment {key} terminal node decision "
                                f"{index} disagrees with its physical operation"
                            )
                for index, helper in enumerate(assignment.helpers):
                    route_in, route_out, route_down = assignment.routes[index]
                    work_fraction = (
                        assignment.work_fractions[index]
                        if index < len(assignment.work_fractions) else 1.0
                    )
                    input_fraction = (
                        assignment.input_fractions[index]
                        if index < len(assignment.input_fractions) else 1.0
                    )
                    output_fraction = (
                        assignment.output_fractions[index]
                        if index < len(assignment.output_fractions) else 1.0
                    )
                    protocol_header_bits = float(
                        assignment.metadata.get("protocol_header_bits", 0.0)
                    )
                    now = trial._reserve_path(
                        request, route_in,
                        tile.d_in_bits * input_fraction
                        + protocol_header_bits,
                        request.sim_time, trace=communication_intervals,
                        owner=key,
                    )
                    input_done = now
                    now = trial._reserve_compute(
                        request, helper,
                        tile.compute_ops * work_fraction, now,
                        trace=compute_intervals, owner=key,
                    )
                    compute_done = now
                    replica_compute_rates.append(
                        float(request.satellites[helper].compute_rate)
                    )
                    now = trial._reserve_path(
                        request, route_out,
                        tile.d_out_bits * output_fraction
                        + protocol_header_bits, now,
                        trace=communication_intervals,
                        owner=key,
                    )
                    output_done = now
                    now = trial._reserve_path(
                        request, route_down,
                        tile.d_out_bits * output_fraction
                        + protocol_header_bits, now,
                        trace=communication_intervals,
                        owner=key,
                    )
                    finishes.append(now)
                    replica_phase_ends.append(
                        (input_done, compute_done, output_done, now)
                    )

            if not finishes:
                raise InvalidDecisionError(
                    f"assignment {key} performs neither compute nor downlink"
                )

            required = int(assignment.metadata.get("data_shards", 1))
            required = max(1, required)
            if required > len(finishes):
                raise InvalidDecisionError(
                    f"assignment {key} requires {required} completions but "
                    f"provides {len(finishes)}"
                )
            modeled_finish = request.sim_time + float(
                assignment.metadata.get("latency", math.inf)
            )
            shard_groups = assignment.metadata.get("shard_groups")
            if shard_groups is not None:
                if len(shard_groups) != len(finishes):
                    raise InvalidDecisionError(
                        f"assignment {key} provides {len(finishes)} operations "
                        f"but {len(shard_groups)} shard-group labels"
                    )
                grouped = {}
                for label, finish in zip(shard_groups, finishes):
                    grouped.setdefault(label, []).append(finish)
                if any(len(group) != required for group in grouped.values()):
                    raise InvalidDecisionError(
                        f"assignment {key} must provide exactly {required} "
                        "shards in every reconstruction group"
                    )
                feasible_delivery = min(
                    sorted(group)[required - 1]
                    for group in grouped.values()
                )
            else:
                feasible_delivery = sorted(finishes)[required - 1]
            if feasible_delivery > task.deadline + 1e-9:
                raise InvalidDecisionError(
                    f"assignment {key} finishes after deadline "
                    f"t={task.deadline:.6f}"
                )
            if not retime and feasible_delivery > modeled_finish + 1e-6:
                raise InvalidDecisionError(
                    f"assignment {key} reports delivery at "
                    f"t={modeled_finish:.6f}, but shared-resource execution "
                    f"cannot deliver before t={feasible_delivery:.6f}"
                )
            if retime:
                metadata = dict(assignment.metadata)
                metadata["latency"] = feasible_delivery - request.sim_time
                metadata["delivery_time"] = feasible_delivery
                if source_release_time is not None:
                    metadata["source_release_time"] = source_release_time
                if replica_phase_ends:
                    metadata["replica_phase_ends"] = tuple(replica_phase_ends)
                    metadata["replica_compute_rates"] = tuple(
                        replica_compute_rates
                    )
                    # Recompute reliability on the retimed physical exposure.
                    # Resource contention can move a phase across reliability
                    # epochs, so retaining the policy's pre-ledger estimate is
                    # inconsistent with realized temporal scoring.
                    from ordi.algorithms._common import Placement, groups_success
                    placements = tuple(
                        Placement(
                            helper, aggregator,
                            phases[3] - request.sim_time, 0.0, 0.0,
                            *assignment.routes[index], request.sim_time,
                            *phases,
                        )
                        for index, (helper, aggregator, phases) in enumerate(zip(
                            assignment.helpers, assignment.aggregators,
                            replica_phase_ends,
                        ))
                    )
                    labels = metadata.get("shard_groups")
                    if labels is not None and len(labels) == len(placements):
                        grouped = {}
                        for label, placement in zip(labels, placements):
                            grouped.setdefault(label, []).append(placement)
                        reliability_groups = tuple(
                            tuple(group) for group in grouped.values()
                        )
                    elif required > 1:
                        reliability_groups = (placements,)
                    else:
                        reliability_groups = tuple(
                            (placement,) for placement in placements
                        )
                    metadata["reliability"] = groups_success(
                        request, task, reliability_groups
                    )
                metadata["communication_intervals"] = tuple(
                    communication_intervals
                )
                metadata["compute_intervals"] = tuple(compute_intervals)
                ground_work = float(metadata.get("ground_compute_flops", 0.0))
                ground_rate = float(metadata.get(
                    "ground_compute_rate_flops_per_s", 0.0
                ))
                ground_power = float(metadata.get(
                    "ground_compute_power_w", 0.0
                ))
                if ground_work > 0.0 and ground_rate > 0.0:
                    metadata["ground_compute_energy_j"] = (
                        ground_power * ground_work / ground_rate
                    )
                assignment = replace(assignment, metadata=metadata)
            accepted.append(assignment)

        self.contact_ready_at = trial.contact_ready_at
        self.contact_residual_bits = trial.contact_residual_bits
        self.compute_ready_at = trial.compute_ready_at
        self.ground_compute_ready_at = trial.ground_compute_ready_at
        self.terminal_intervals = trial.terminal_intervals
        self.reservations = trial.reservations
        return (replace(decision, assignments=tuple(accepted))
                if retime else decision)

    def retime_and_reserve(self, request: EpochInput,
                           assignment: Assignment) -> Assignment:
        """Commit one policy placement and replace optimism with model time.

        Policies still choose helpers, replicas, and paths.  This method only
        serializes those submitted operations on the shared resource ledger;
        an assignment that cannot meet its deadline remains invalid.
        """
        decision = self.validate_and_reserve(
            request, Decision(request.epoch, (assignment,)), retime=True
        )
        return decision.assignments[0]
