"""Full-replication baseline with native local discovery and messaging."""

from dataclasses import replace

from .schema import Assignment, Decision
from ._common import (
    advertisement_metadata, deadline_expired, enumerate_placements, protocol_trace,
    tile_success,
)
from ordi.eval.validation import InvalidDecisionError
from ordi.sim.messaging import MessageSimulator


class FullReplication:
    name = "full_replication"

    def __init__(self):
        self.messages = MessageSimulator()

    def schedule(self, request):
        advertisements = self.messages.prepare_epoch(request)
        assignments = []
        for task in request.tasks:
            if deadline_expired(request, task):
                continue
            local = self.messages.local_view(request, task.source_sat)
            for tile in task.tiles:
                choices = sorted(
                    enumerate_placements(local, task, tile),
                    key=lambda placement: placement.reliability,
                    reverse=True,
                )
                unique = []
                for placement in choices:
                    if placement.helper not in {
                            item.helper for item in unique}:
                        unique.append(placement)
                    if len(unique) >= tile.n_replicas_max:
                        break
                if not unique:
                    continue
                groups = tuple((placement,) for placement in unique)
                assignment = Assignment(
                    task.task_id, tile.tile_id, task.source_sat,
                    tuple(p.helper for p in unique),
                    tuple(p.aggregator for p in unique),
                    metadata={
                        "latency": min(p.latency for p in unique),
                        "reliability": tile_success(local, task, unique),
                        "replication": "full",
                        "data_shards": 1,
                        "shard_groups": tuple(range(len(unique))),
                        "effective_replicas": len(unique),
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
                        for p in unique
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
                    unique[index] for index in execution.executed_shards
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
