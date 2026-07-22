"""Random-replication baseline with native local discovery and messaging."""

from dataclasses import replace
import random

from .schema import Assignment, Decision
from ._common import (
    advertisement_metadata, deadline_expired, enumerate_placements, protocol_trace,
    tile_success,
)
from ordi.eval.validation import InvalidDecisionError
from ordi.sim.messaging import MessageSimulator


def _placement_rank(placement):
    """Neutral deterministic route quality for one randomly chosen helper."""
    return (
        placement.latency,
        placement.communication_bits,
        -placement.reliability,
        placement.aggregator,
        tuple(placement.route_in),
        tuple(placement.route_out),
        tuple(placement.route_down),
    )


def _best_placements_by_helper(placements):
    """Keep each helper's fastest placement and remove enumeration-order bias."""
    best = {}
    for placement in placements:
        current = best.get(placement.helper)
        if current is None or _placement_rank(placement) < _placement_rank(
                current):
            best[placement.helper] = placement
    return tuple(best[helper] for helper in sorted(best))


class RandomReplication:
    name = "random_replication"

    def __init__(self, seed=0):
        self.seed = seed
        self.messages = MessageSimulator()

    def schedule(self, request):
        advertisements = self.messages.prepare_epoch(request)
        rng = random.Random(self.seed + request.epoch)
        assignments = []
        for task in request.tasks:
            if deadline_expired(request, task):
                continue
            local = self.messages.local_view(request, task.source_sat)
            for tile in task.tiles:
                pool = _best_placements_by_helper(
                    enumerate_placements(local, task, tile)
                )
                count = min(tile.n_replicas_max, len(pool))
                selected = rng.sample(pool, count)
                if not selected:
                    continue
                groups = tuple((placement,) for placement in selected)
                assignment = Assignment(
                    task.task_id, tile.tile_id, task.source_sat,
                    tuple(p.helper for p in selected),
                    tuple(p.aggregator for p in selected),
                    metadata={
                        "latency": min(p.latency for p in selected),
                        "reliability": tile_success(local, task, selected),
                        "replication": "random",
                        "data_shards": 1,
                        "shard_groups": tuple(range(len(selected))),
                        "effective_replicas": len(selected),
                        "helper_handshake": True,
                        "helper_request_kind": "replica",
                        "state_observer": task.source_sat,
                        "known_state_nodes": len(local.satellites),
                        "max_state_age_s": max(
                            local.state_age_s.values(), default=0.0
                        ),
                    },
                    routes=tuple(
                        (p.route_in, p.route_out, p.route_down)
                        for p in selected
                    ),
                    node_decisions=protocol_trace(
                        local, task, tile, groups
                    ),
                )
                try:
                    execution = self.messages.execute(
                        request, task, tile, assignment
                    )
                except InvalidDecisionError:
                    continue
                accepted = [
                    selected[index] for index in execution.executed_shards
                ]
                accepted_groups = tuple(
                    (placement,) for placement in accepted
                )
                assignment = replace(
                    assignment,
                    helpers=tuple(p.helper for p in accepted),
                    aggregators=tuple(p.aggregator for p in accepted),
                    routes=tuple(
                        (p.route_in, p.route_out, p.route_down)
                        for p in accepted
                    ),
                    node_decisions=protocol_trace(
                        local, task, tile, accepted_groups
                    ),
                )
                metadata = dict(assignment.metadata)
                metadata.update({
                    "latency": execution.delivery_time - request.sim_time,
                    "reliability": tile_success(local, task, accepted),
                    "shard_groups": tuple(range(len(accepted))),
                    "effective_replicas": len(accepted),
                    "protocol_header_bits": self.messages.header_bits,
                    "protocol_message_count": execution.message_count,
                    "protocol_control_bits": execution.control_bits,
                    "protocol_ground_bits": execution.ground_bits,
                    "handshake_control_bits": sum(
                        event.bits for event in execution.events
                        if event.event == "hop_sent"
                        and event.kind.startswith("replica_")
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
