"""Policy-independent monitoring of in-flight assignments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

from .schema import Assignment

STRAGGLER_FACTOR = 1.5
HIGH_PRIORITY_THRESHOLD = 1.5


@dataclass(frozen=True)
class InFlightTile:
    task_id: int
    tile_id: int
    helper: str
    aggregator: str
    expected_done_time: float
    assigned_epoch: int


@dataclass(frozen=True)
class ReplanEvent:
    trigger: str
    epoch: int
    affected_tiles: List[Tuple[int, int]]
    failed_helpers: Set[str]


class ReplanMonitor:
    def __init__(self):
        self._in_flight: List[InFlightTile] = []
        self._completed: Set[Tuple[int, int]] = set()

    def register(self, assignment: Assignment, expected_done_time: float, epoch: int):
        for helper, aggregator in zip(assignment.helpers, assignment.aggregators):
            self._in_flight.append(InFlightTile(
                assignment.task_id, assignment.tile_id, helper, aggregator,
                expected_done_time, epoch,
            ))

    def mark_completed(self, task_id: int, tile_id: int):
        self._completed.add((task_id, tile_id))
        self._in_flight = [item for item in self._in_flight
                           if (item.task_id, item.tile_id) != (task_id, tile_id)]

    def check_failures(self, epoch: int, current_time: float,
                       failed_helpers: Set[str], available_contacts: set
                       ) -> Optional[ReplanEvent]:
        affected = set()
        for item in self._in_flight:
            key = (item.task_id, item.tile_id)
            if key in self._completed:
                continue
            if (item.helper in failed_helpers
                    or (item.helper != item.aggregator
                        and (item.helper, item.aggregator) not in available_contacts)
                    or current_time > item.expected_done_time * STRAGGLER_FACTOR):
                affected.add(key)
        if not affected:
            return None
        return ReplanEvent("failure_or_straggler", epoch, sorted(affected), failed_helpers)

    def check_high_priority_arrival(self, epoch: int, new_tasks,
                                    base_utility: float = 1.0):
        affected = [(task.task_id, tile.tile_id) for task in new_tasks
                    for tile in task.tiles
                    if tile.utility >= base_utility * HIGH_PRIORITY_THRESHOLD]
        return (ReplanEvent("high_priority", epoch, affected, set())
                if affected else None)
