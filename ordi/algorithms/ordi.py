"""Self-organizing ORDI policy.

Each source injects a ``WorkItem(job, destination)`` into the protocol.  The
receiving node evaluates local execute/delegate, split, and redundancy actions.
The resulting node decisions are retained on the Assignment as an auditable
protocol trace; the model-side resource ledger remains authoritative.
"""

from copy import copy
from dataclasses import dataclass, replace
import math

from .schema import Assignment, Decision, NodeDecision, WorkItem
from ._common import enumerate_placements, plane
from ordi.eval.validation import DecisionFeasibilityModel, InvalidDecisionError
from ordi.sim.messaging import MessageSimulator


@dataclass(frozen=True)
class _LocalPlan:
    shard_count: int
    placements: tuple
    work_fraction: float
    input_fraction: float
    output_fraction: float
    latency: float
    reliability: float
    energy_j: float
    communication_bits: float
    value: float


@dataclass(frozen=True)
class _RetryState:
    next_retry_time: float
    attempts: int
    reason: str


class ORDI:
    name = "ordi"

    def __init__(self, max_replicas=2, split_options=(1, 2, 4),
                 halo_fraction=0.05, shard_count=None):
        # shard_count is retained as a compatibility hook for controlled tests
        # and ablations. Normal evaluation uses dynamic split_options.
        if shard_count is not None:
            split_options = (shard_count,)
        self.max_replicas = max(1, max_replicas)
        self.split_options = tuple(sorted({
            max(1, int(count)) for count in split_options
        }))
        self.halo_fraction = max(0.0, halo_fraction)
        self.resources = DecisionFeasibilityModel()
        self.messages = MessageSimulator()
        self.waiting: dict[tuple[int, int], _RetryState] = {}

    @staticmethod
    def _contact_key(contact):
        return (
            contact.source, contact.target, contact.opens, contact.closes,
            contact.rate_bps, contact.kind,
        )

    def _resource_aware_view(self, request):
        """Project committed protocol work into the next placement search."""
        contacts = []
        for contact in request.contacts:
            key = self._contact_key(contact)
            capacity = max(0.0, contact.closes - contact.opens) * max(
                contact.rate_bps, 0.0
            )
            residual = self.messages.contact_residual_bits.get(key, capacity)
            opens = max(
                contact.opens,
                self.messages.contact_ready_at.get(key, contact.opens),
            )
            closes = min(
                contact.closes,
                opens + residual / max(contact.rate_bps, 1.0),
            )
            if closes > opens + 1e-9:
                contacts.append(replace(
                    contact, opens=opens, closes=closes
                ))

        satellites = {}
        for sat_id, state in request.satellites.items():
            ready = self.messages.compute_ready_at.get(sat_id)
            queued = state.queued_flops
            if ready is not None:
                queued = max(
                    queued,
                    max(0.0, ready - request.sim_time)
                    * max(state.compute_rate, 1.0),
                )
            satellites[sat_id] = replace(state, queued_flops=queued)
        return replace(
            request, contacts=tuple(contacts), satellites=satellites
        )

    def _defer(self, request, task, tile, reason):
        """Wait one scheduling epoch before retrying an uncommitted tile."""
        key = (task.task_id, tile.tile_id)
        previous = self.waiting.get(key)
        next_retry = request.sim_time + request.epoch_length
        if next_retry >= task.deadline - 1e-9:
            self.waiting.pop(key, None)
            return
        self.waiting[key] = _RetryState(
            next_retry,
            1 if previous is None else previous.attempts + 1,
            reason,
        )

    @staticmethod
    def _placement_utility(request, task, tile, placement):
        weights = request.weights
        source_p = request.satellites[task.source_sat].reliability
        return (
            tile.utility * source_p * placement.reliability
            * math.exp(-weights.freshness * placement.latency)
            - weights.energy * placement.energy_j
            - weights.communication * placement.communication_bits
        )

    def _local_plan(self, request, task, tile, shard_count):
        """Let the receiving source node assess one local split action."""
        input_fraction = (
            1.0 + self.halo_fraction * (shard_count - 1)
        ) / shard_count
        work_fraction = input_fraction
        output_fraction = 1.0 / shard_count
        choices = enumerate_placements(
            request, task, tile,
            work_fraction=work_fraction,
            input_fraction=input_fraction,
            output_fraction=output_fraction,
            protocol_header_bits=self.messages.header_bits,
        )

        # One shard group cannot assign two required shards to the same compute
        # node. The receiving node keeps its best reachable path per helper.
        best_by_helper = {}
        for choice in choices:
            current = best_by_helper.get(choice.helper)
            if (current is None
                    or self._placement_utility(request, task, tile, choice)
                    > self._placement_utility(
                        request, task, tile, current
                    )):
                best_by_helper[choice.helper] = choice
        ranked = sorted(
            best_by_helper.values(),
            key=lambda placement: self._placement_utility(
                request, task, tile, placement
            ),
            reverse=True,
        )
        if len(ranked) < shard_count:
            return None

        placements = tuple(ranked[:shard_count])
        latency = max(p.latency for p in placements)
        source_p = request.satellites[task.source_sat].reliability
        reliability = source_p * math.prod(
            p.reliability for p in placements
        )
        energy = sum(p.energy_j for p in placements)
        communication = sum(p.communication_bits for p in placements)
        value = (
            tile.utility * reliability
            * math.exp(-request.weights.freshness * latency)
            - request.weights.energy * energy
            - request.weights.communication * communication
        )
        return _LocalPlan(
            shard_count, placements, work_fraction, input_fraction,
            output_fraction, latency, reliability, energy, communication,
            value,
        )

    def _decide_primary(self, request, task, tile):
        """Choose execute/delegate versus a two- or four-way local split."""
        candidates = [
            self._local_plan(request, task, tile, count)
            for count in self.split_options
        ]
        feasible = [plan for plan in candidates if plan is not None]
        if not feasible:
            return None
        plan = max(feasible, key=lambda item: item.value)
        return plan if plan.value > 0 else None

    @staticmethod
    def _group_probability(placements):
        return math.prod(p.reliability for p in placements)

    def _decide_backup(self, request, task, tile, primary):
        """Let the receiving node optionally create a full redundant group."""
        if min(self.max_replicas,
               getattr(tile, "n_replicas_max", 1)) <= 1:
            return None
        q = primary.shard_count
        choices = enumerate_placements(
            request, task, tile,
            work_fraction=primary.work_fraction,
            input_fraction=primary.input_fraction,
            output_fraction=primary.output_fraction,
            protocol_header_bits=self.messages.header_bits,
        )
        used_helpers = {p.helper for p in primary.placements}
        used_planes = {plane(p.helper) for p in primary.placements}
        used_nodes = {
            node for p in primary.placements
            for node in p.route_in + p.route_out
        }
        best_by_helper = {}
        for candidate in choices:
            candidate_nodes = set(candidate.route_in + candidate.route_out)
            if (candidate.helper in used_helpers
                    or plane(candidate.helper) in used_planes
                    or used_nodes.intersection(
                        candidate_nodes - {task.source_sat}
                    )):
                continue
            current = best_by_helper.get(candidate.helper)
            if (current is None
                    or candidate.reliability > current.reliability):
                best_by_helper[candidate.helper] = candidate
        ranked = sorted(
            best_by_helper.values(),
            key=lambda placement: self._placement_utility(
                request, task, tile, placement
            ),
            reverse=True,
        )
        if len(ranked) < q:
            return None
        backup = tuple(ranked[:q])
        source_p = request.satellites[task.source_sat].reliability
        primary_p = self._group_probability(primary.placements)
        backup_p = self._group_probability(backup)
        reliability_gain = source_p * (1.0 - primary_p) * backup_p
        gain = (
            tile.utility * reliability_gain
            * math.exp(-request.weights.freshness * min(
                primary.latency, max(p.latency for p in backup)
            ))
            - request.weights.energy * sum(p.energy_j for p in backup)
            - request.weights.communication
            * sum(p.communication_bits for p in backup)
            - request.weights.replication
        )
        return backup if gain > 0 else None

    @staticmethod
    def _leaf_item(task, tile, placement, group_id, work_fraction,
                   input_fraction, output_fraction):
        destination = (
            placement.route_down[-1]
            if placement.route_down else placement.aggregator
        )
        return WorkItem(
            task.task_id, tile.tile_id, destination, placement.helper,
            work_fraction, input_fraction, output_fraction, group_id, 1,
        )

    def _decision_trace(self, request, task, tile, groups, primary):
        root = WorkItem(
            task.task_id, tile.tile_id,
            tuple(sorted(request.ground_stations)), task.source_sat,
        )
        decisions = []
        for group_id, group in enumerate(groups):
            leaves = tuple(self._leaf_item(
                task, tile, placement, group_id,
                primary.work_fraction, primary.input_fraction,
                primary.output_fraction,
            ) for placement in group)
            if group_id == 0:
                if len(group) > 1:
                    action = "split"
                elif group[0].helper == task.source_sat:
                    action = "execute_forward"
                else:
                    action = "delegate"
            else:
                action = "replicate"
            decisions.append(NodeDecision(
                task.source_sat, action, root,
                () if action == "execute_forward" else leaves,
                reason=f"locally selected {len(group)} shard(s)",
            ))
            for leaf, placement in zip(leaves, group):
                # When the source executes an unsplit job, the root decision is
                # already the terminal action; avoid recording it twice.
                if (action == "execute_forward"
                        and placement.helper == task.source_sat):
                    continue
                decisions.append(NodeDecision(
                    placement.helper, "execute_forward", leaf, (),
                    reason="terminal shard; forward result to destination",
                ))
        return tuple(decisions)

    def schedule(self, request):
        assignments = []
        advertisements = self.messages.prepare_epoch(request)
        for task in request.tasks:
            for tile in task.tiles:
                # Rebuild the effective view after every committed tile so
                # subsequent candidates see newly consumed contacts/compute.
                local_request = self._resource_aware_view(
                    self.messages.local_view(request, task.source_sat)
                )
                key = (task.task_id, tile.tile_id)
                retry = self.waiting.get(key)
                if (retry is not None
                        and request.sim_time + 1e-9
                        < retry.next_retry_time):
                    continue
                primary = self._decide_primary(
                    local_request, task, tile
                )
                if primary is None:
                    self._defer(
                        request, task, tile, "no_primary_plan"
                    )
                    continue
                groups = [primary.placements]
                backup = self._decide_backup(
                    local_request, task, tile, primary
                )
                if backup is not None:
                    groups.append(backup)

                selected = tuple(
                    placement for group in groups for placement in group
                )
                group_labels = tuple(
                    group_id for group_id, group in enumerate(groups)
                    for _ in group
                )
                group_probabilities = [
                    self._group_probability(group) for group in groups
                ]
                source_p = local_request.satellites[
                    task.source_sat
                ].reliability
                reliability = source_p * (
                    1.0 - math.prod(1.0 - p for p in group_probabilities)
                )
                latency = min(max(p.latency for p in group) for group in groups)
                energy = sum(p.energy_j for p in selected)
                communication = sum(p.communication_bits for p in selected)
                objective = (
                    tile.utility * reliability
                    * math.exp(-request.weights.freshness * latency)
                    - request.weights.energy * energy
                    - request.weights.communication * communication
                    - request.weights.replication * (len(groups) - 1)
                )
                assignment = Assignment(
                    task.task_id, tile.tile_id, task.source_sat,
                    tuple(p.helper for p in selected),
                    tuple(p.aggregator for p in selected),
                    metadata={
                        "latency": latency,
                        "reliability": reliability,
                        "selective_redundancy": True,
                        "self_organized": True,
                        "partitioned": primary.shard_count > 1,
                        "data_shards": primary.shard_count,
                        "split_count": primary.shard_count,
                        "shard_groups": group_labels,
                        "effective_replicas": len(groups),
                        "energy_j": energy,
                        "objective": objective,
                        "state_observer": task.source_sat,
                        "known_state_nodes": len(local_request.satellites),
                        "max_state_age_s": max(
                            local_request.state_age_s.values(), default=0.0
                        ),
                    },
                    routes=tuple(
                        (p.route_in, p.route_out, p.route_down)
                        for p in selected
                    ),
                    work_fractions=(primary.work_fraction,) * len(selected),
                    input_fractions=(primary.input_fraction,) * len(selected),
                    output_fractions=(primary.output_fraction,) * len(selected),
                    node_decisions=self._decision_trace(
                        local_request, task, tile, groups, primary
                    ),
                )
                # Commit protocol and model ledgers together. A route can
                # become infeasible after earlier tiles consume a future
                # contact even when the placement was nominally reachable.
                trial_messages = copy(self.messages)
                trial_resources = copy(self.resources)
                try:
                    execution = trial_messages.execute(
                        request, task, tile, assignment
                    )
                except InvalidDecisionError:
                    self._defer(
                        request, task, tile, "protocol_rejected"
                    )
                    continue
                metadata = dict(assignment.metadata)
                metadata.update({
                    "latency": execution.delivery_time - request.sim_time,
                    "protocol_header_bits": trial_messages.header_bits,
                    "protocol_message_count": execution.message_count,
                    "protocol_control_bits": execution.control_bits,
                    "protocol_ground_bits": execution.ground_bits,
                })
                assignment = replace(
                    assignment, metadata=metadata,
                    message_events=execution.events,
                )
                # The protocol runtime and model-side ledger serialize the
                # same work independently. Keep the later completion time so
                # small differences in their reservation order cannot make an
                # otherwise feasible assignment under-report its latency.
                try:
                    modeled = trial_resources.retime_and_reserve(
                        request, assignment
                    )
                except InvalidDecisionError:
                    self._defer(
                        request, task, tile, "model_rejected"
                    )
                    continue
                modeled_latency = float(modeled.metadata["latency"])
                if modeled_latency > float(assignment.metadata["latency"]):
                    metadata = dict(assignment.metadata)
                    metadata["latency"] = modeled_latency
                    assignment = replace(assignment, metadata=metadata)
                self.messages = trial_messages
                self.resources = trial_resources
                self.waiting.pop(key, None)
                assignments.append(assignment)
        return Decision(
            request.epoch, tuple(assignments),
            metadata={
                "protocol_message_count": advertisements.message_count,
                "protocol_control_bits": advertisements.control_bits,
                "advertisement_control_bits": advertisements.control_bits,
                "waiting_tiles": len(self.waiting),
                "retry_attempts": sum(
                    retry.attempts for retry in self.waiting.values()
                ),
            },
            message_events=advertisements.events,
        )
