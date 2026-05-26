"""
Replanning triggers for ORDI.

Monitors in-flight assignments and fires replanning when:
  - A helper's A_i flips to 0 (failure / thermal / battery)
  - A contact window is missed (link not available at scheduled time)
  - A straggler is detected (tile not returned within 1.5× expected latency)
  - A new high-priority task arrives (utility above threshold)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from ordi.scheduler.ordi import TileAssignment


STRAGGLER_FACTOR = 1.5          # 1.5× expected latency → straggler
HIGH_PRIORITY_THRESHOLD = 1.5   # utility multiplier threshold for urgent replan


@dataclass
class InFlightTile:
    task_id: int
    tile_id: int
    helper: str
    aggregator: str
    expected_done_time: float   # epoch_start + L_kvia
    assigned_epoch: int


@dataclass
class ReplanEvent:
    trigger: str    # "failure" | "missed_contact" | "straggler" | "high_priority"
    epoch: int
    affected_tiles: List[Tuple[int, int]]  # (task_id, tile_id)
    failed_helpers: Set[str]


class ReplanMonitor:
    """
    Tracks in-flight tile assignments and fires replan events as needed.
    """

    def __init__(self):
        self._in_flight: List[InFlightTile] = []
        self._completed: Set[Tuple[int, int]] = set()

    def register(self, assignment: TileAssignment, expected_done_time: float, epoch: int):
        for r in assignment.replicas:
            self._in_flight.append(InFlightTile(
                task_id=assignment.task_id,
                tile_id=assignment.tile_id,
                helper=r.helper,
                aggregator=r.aggregator,
                expected_done_time=expected_done_time,
                assigned_epoch=epoch,
            ))

    def mark_completed(self, task_id: int, tile_id: int):
        self._completed.add((task_id, tile_id))
        self._in_flight = [
            t for t in self._in_flight
            if not (t.task_id == task_id and t.tile_id == tile_id)
        ]

    def check_failures(
        self,
        epoch: int,
        current_time: float,
        failed_helpers: Set[str],
        available_contacts: set,   # set of (node_a, node_b) available this epoch
    ) -> Optional[ReplanEvent]:
        """
        Check for failure/straggler/missed-contact conditions.
        Returns a ReplanEvent if replanning is needed, else None.
        """
        affected: List[Tuple[int, int]] = []

        for tile in self._in_flight:
            if (tile.task_id, tile.tile_id) in self._completed:
                continue

            # Helper failure
            if tile.helper in failed_helpers:
                affected.append((tile.task_id, tile.tile_id))
                continue

            # Missed contact: the scheduled ISL/downlink no longer available
            if (tile.helper, tile.aggregator) not in available_contacts:
                affected.append((tile.task_id, tile.tile_id))
                continue

            # Straggler: overdue by straggler factor
            if current_time > tile.expected_done_time * STRAGGLER_FACTOR:
                affected.append((tile.task_id, tile.tile_id))

        if not affected:
            return None

        # Deduplicate
        affected = list(set(affected))
        return ReplanEvent(
            trigger="failure_or_straggler",
            epoch=epoch,
            affected_tiles=affected,
            failed_helpers=failed_helpers,
        )

    def check_high_priority_arrival(
        self,
        epoch: int,
        new_tasks,           # List[EOTask]
        base_utility: float = 1.0,
    ) -> Optional[ReplanEvent]:
        """Fire replan if any new task has utility above threshold."""
        high_priority = []
        for task in new_tasks:
            max_u = max((t.utility for t in task.tiles), default=0.0)
            if max_u >= base_utility * HIGH_PRIORITY_THRESHOLD:
                high_priority.extend((task.task_id, t.tile_id) for t in task.tiles)

        if not high_priority:
            return None
        return ReplanEvent(
            trigger="high_priority",
            epoch=epoch,
            affected_tiles=high_priority,
            failed_helpers=set(),
        )
