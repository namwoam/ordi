"""ORDI policy implemented directly against Basilisk/BSK-RL state.

The policy deliberately has no dependency on ``ordi.scheduler``.  Contact
windows and resources are supplied by the Basilisk adapter; this file owns
only tile placement and replica selection.
"""
from dataclasses import dataclass, field
from typing import Any

@dataclass
class Decision:
    epoch: int
    assignments: list[dict[str, Any]] = field(default_factory=list)

class ORDI:
    name = "ordi"
    def __init__(self, backend, max_replicas=2):
        self.backend = backend
        self.max_replicas = max_replicas

    def schedule_epoch(self, epoch, opportunities, tasks):
        states = self.backend.states
        decisions = []
        for task in tasks:
            candidates = [s for s in opportunities.get(task.source_sat, ()) if states.get(s) and states[s].A_i]
            if not candidates and states.get(task.source_sat, None) and states[task.source_sat].A_i:
                candidates = [task.source_sat]
            for tile in task.tiles:
                replicas = candidates[:self.max_replicas]
                if replicas:
                    decisions.append({"task_id": task.task_id, "tile_id": tile.tile_id,
                                      "helpers": replicas, "source": task.source_sat})
        return Decision(epoch, decisions)
