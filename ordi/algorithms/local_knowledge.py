"""Policy-agnostic local-knowledge boundary for fair baseline evaluation."""

from __future__ import annotations

from dataclasses import replace

from .schema import Decision
from ordi.eval.validation import DecisionFeasibilityModel
from ordi.sim.messaging import MessageSimulator


class LocalKnowledgeAdapter:
    """Run a scheduler once per task source using only that node's cache.

    The wrapped algorithm and objective are unchanged. This adapter only
    replaces the impossible global live-state map with the same delayed state
    advertisements used by ORDI.
    """

    def __init__(self, policy, message_simulator=None):
        self.policy = policy
        self.name = policy.name
        self.messages = message_simulator or MessageSimulator()
        self.resources = DecisionFeasibilityModel()

    def schedule(self, request):
        advertisements = self.messages.prepare_epoch(request)
        by_source = {}
        for task in request.tasks:
            by_source.setdefault(task.source_sat, []).append(task)

        assignments = []
        for source, tasks in by_source.items():
            local_request = replace(
                self.messages.local_view(request, source),
                tasks=tuple(tasks),
            )
            local_result = self.policy.schedule(local_request)
            max_age = max(
                local_request.state_age_s.values(), default=0.0
            )
            for assignment in local_result.assignments:
                metadata = dict(assignment.metadata)
                metadata.update({
                    "state_observer": source,
                    "known_state_nodes": len(local_request.satellites),
                    "max_state_age_s": max_age,
                    "local_knowledge": True,
                })
                assignments.append(replace(
                    assignment, metadata=metadata
                ))

        decision = Decision(
            request.epoch, tuple(assignments),
            metadata={
                "protocol_message_count": advertisements.message_count,
                "protocol_control_bits": advertisements.control_bits,
                "advertisement_control_bits": advertisements.control_bits,
            },
            message_events=advertisements.events,
        )
        # Validate stale local decisions against actual current physical state.
        return self.resources.validate_and_reserve(request, decision)


__all__ = ["LocalKnowledgeAdapter"]
