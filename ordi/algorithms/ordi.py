"""Self-organizing ORDI policy.

Each source injects a ``WorkItem(job, destination)`` into the protocol.  The
receiving node evaluates local execute/delegate, split, and redundancy actions.
The resulting node decisions are retained on the Assignment as an auditable
protocol trace; the model-side resource ledger remains authoritative.
"""

from copy import copy
from dataclasses import dataclass, replace
import math
import random

from .schema import Assignment, Decision, NodeDecision, WorkItem
from ._common import (
    deadline_expired, enumerate_placements, group_success, groups_success,
    plane,
)
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
                 halo_fraction=0.05, shard_count=None,
                 fault_risk_alpha=0.5, fault_risk_beta=0.5,
                 fault_risk_discount=0.98, cold_start_backup_budget=1,
                 min_fault_outcomes=3, rng_seed=0):
        # shard_count is retained as a compatibility hook for controlled tests
        # and ablations. Normal evaluation uses dynamic split_options.
        if shard_count is not None:
            split_options = (shard_count,)
        self.max_replicas = max(1, max_replicas)
        self.split_options = tuple(sorted({
            max(1, int(count)) for count in split_options
        }))
        self.halo_fraction = max(0.0, halo_fraction)
        if fault_risk_alpha <= 0.0 or fault_risk_beta <= 0.0:
            raise ValueError(
                "fault-risk Beta parameters must be positive"
            )
        if not 0.0 < fault_risk_discount <= 1.0:
            raise ValueError("fault_risk_discount must be in (0, 1]")
        if cold_start_backup_budget < 0:
            raise ValueError("cold_start_backup_budget must be non-negative")
        if min_fault_outcomes < 0:
            raise ValueError("min_fault_outcomes must be non-negative")
        # Online Beta-Bernoulli estimate. Jeffreys' Beta(1/2, 1/2) prior is
        # invariant and noninformative; it does not encode an error rate.
        # Each observed orbital plane contributes one
        # availability sample per epoch; helper and plane outages are thus
        # learned from policy-visible state rather than supplied in advance.
        self._fault_risk_alpha = float(fault_risk_alpha)
        self._fault_risk_beta = float(fault_risk_beta)
        self._fault_domain_failures = 0.0
        self._fault_domain_successes = 0.0
        self._fault_domain_counts = {}
        self._observed_fault_events = set()
        self._fault_risk_discount = float(fault_risk_discount)
        self._cold_start_backup_budget = int(cold_start_backup_budget)
        self._cold_start_backups_used = 0
        self._min_fault_outcomes = int(min_fault_outcomes)
        self._fault_rng = random.Random(rng_seed)
        self._last_fault_risk_sample = self.fault_domain_failure_estimate
        self.resources = DecisionFeasibilityModel()
        self.messages = MessageSimulator()
        self.waiting: dict[tuple[int, int], _RetryState] = {}

    @property
    def fault_domain_failure_estimate(self):
        return (
            self._fault_risk_alpha + self._fault_domain_failures
        ) / (
            self._fault_risk_alpha + self._fault_risk_beta
            + self._fault_domain_failures + self._fault_domain_successes
        )

    @property
    def fault_domain_sample_count(self):
        return self._fault_domain_failures + self._fault_domain_successes

    def observe_fault_domain_sample(self, failed: bool, domain="global",
                                    event_id=None):
        """Update the online risk estimate from one observed domain outcome."""
        event_key = None if event_id is None else (domain, event_id)
        if event_key is not None and event_key in self._observed_fault_events:
            return False
        if event_key is not None:
            self._observed_fault_events.add(event_key)
        failures, successes = self._fault_domain_counts.get(domain, (0.0, 0.0))
        if failed:
            self._fault_domain_failures += 1.0
            failures += 1.0
        else:
            self._fault_domain_successes += 1.0
            successes += 1.0
        self._fault_domain_counts[domain] = (failures, successes)
        return True

    def sample_fault_domain_failure_risk(self, domains=None):
        """Draw a Thompson sample from the learned fault-risk posterior."""
        domains = tuple(domains or ("global",))
        samples = []
        global_counts = self._fault_domain_counts.get(
            "global", (0.0, 0.0)
        )
        for domain in domains:
            failures, successes = self._fault_domain_counts.get(
                domain, global_counts
            )
            samples.append(self._fault_rng.betavariate(
                self._fault_risk_alpha + failures,
                self._fault_risk_beta + successes,
            ))
        self._last_fault_risk_sample = max(samples)
        return self._last_fault_risk_sample

    def observe_assignment_outcome(self, outcome: str, *, domains=None,
                                   event_id=None, **_context):
        """Learn from one completed or failed scheduled assignment.

        A backup recovery and an unrecoverable hard fault are evidence that the
        primary fault domain failed. Ordinary primary delivery is a success.
        Contact/queue misses never become Bernoulli fault samples.
        """
        if outcome not in {
            "primary_success", "backup_recovery", "fault_failure",
            "nonfault_failure",
        }:
            raise ValueError(f"unknown assignment outcome {outcome!r}")
        if outcome == "nonfault_failure":
            return
        failed = outcome in {"backup_recovery", "fault_failure"}
        domains = tuple(dict.fromkeys(domains or ("global",)))
        unobserved = tuple(
            domain for domain in domains
            if event_id is None
            or (domain, event_id) not in self._observed_fault_events
        )
        if not unobserved:
            return
        self._fault_domain_failures *= self._fault_risk_discount
        self._fault_domain_successes *= self._fault_risk_discount
        self._fault_domain_counts = {
            key: (failures * self._fault_risk_discount,
                  successes * self._fault_risk_discount)
            for key, (failures, successes)
            in self._fault_domain_counts.items()
        }
        for domain in unobserved:
            self.observe_fault_domain_sample(
                failed, domain=domain, event_id=event_id
            )

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
        return (
            tile.utility * group_success(request, task, (placement,))
            * math.exp(-weights.freshness * placement.latency)
            - weights.communication * placement.communication_bits
        )

    def _local_plan(self, request, task, tile, shard_count):
        """Assess an ordinary split whose every shard must complete."""
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
        reliability = group_success(request, task, placements)
        communication = sum(p.communication_bits for p in placements)
        value = (
            tile.utility * reliability
            * math.exp(-request.weights.freshness * latency)
            - request.weights.communication * communication
        )
        return _LocalPlan(
            shard_count, placements,
            work_fraction, input_fraction,
            output_fraction, latency, reliability, communication,
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
    def _route_satellite_nodes(placement, source, ground_stations):
        """Satellites whose failure can couple this placement to a backup."""
        return {
            node
            for path in (
                placement.route_in,
                placement.route_out,
                placement.route_down,
            )
            for node in path
            if node != source and node not in ground_stations
        }

    def _decide_backup(self, request, task, tile, primary):
        """Let the receiving node optionally create a full redundant group."""
        if min(self.max_replicas,
               getattr(tile, "n_replicas_max", 1)) <= 1:
            return None
        cold_start = (
            self.fault_domain_sample_count < self._min_fault_outcomes
        )
        if (cold_start
                and self._cold_start_backups_used
                >= self._cold_start_backup_budget):
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
            node
            for placement in primary.placements
            for node in self._route_satellite_nodes(
                placement, task.source_sat, request.ground_stations
            )
        }
        best_by_helper = {}
        for candidate in choices:
            candidate_nodes = self._route_satellite_nodes(
                candidate, task.source_sat, request.ground_stations
            )
            if (candidate.helper in used_helpers
                    or plane(candidate.helper) in used_planes
                    or used_nodes.intersection(candidate_nodes)):
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
        primary_p = group_success(request, task, primary.placements)
        backup_p = group_success(request, task, backup)
        sampled_domain_risk = self.sample_fault_domain_failure_risk(
            {plane(p.helper) for p in primary.placements}
        )
        modeled_gain = (
            groups_success(
                request, task, (primary.placements, backup)
            ) - primary_p
        )
        reliability_gain = max(
            modeled_gain, sampled_domain_risk * backup_p
        )
        backup_latency = max(p.latency for p in backup)
        gain = (
            tile.utility * reliability_gain
            * math.exp(-request.weights.freshness * backup_latency)
            - request.weights.communication
            * sum(p.communication_bits for p in backup)
            - request.weights.replication
        )
        if gain <= 0:
            return None
        if cold_start:
            self._cold_start_backups_used += 1
        return backup

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
            if deadline_expired(request, task):
                continue
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
                reliability = groups_success(
                    local_request, task, groups
                )
                latency = min(
                    max(p.latency for p in group)
                    for group in groups
                )
                communication = sum(p.communication_bits for p in selected)
                redundancy_factor = (
                    sum(len(group) for group in groups)
                    / primary.shard_count
                )
                objective = (
                    tile.utility * reliability
                    * math.exp(-request.weights.freshness * latency)
                    - request.weights.communication * communication
                    - request.weights.replication
                    * (redundancy_factor - 1.0)
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
                        "effective_replicas": redundancy_factor,
                        "objective": objective,
                        "state_observer": task.source_sat,
                        "known_state_nodes": len(local_request.satellites),
                        "max_state_age_s": max(
                            local_request.state_age_s.values(), default=0.0
                        ),
                        "fault_domain_failure_estimate": (
                            self.fault_domain_failure_estimate
                        ),
                        "fault_domain_samples": self.fault_domain_sample_count,
                        "fault_domain_risk_sample": (
                            self._last_fault_risk_sample
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
