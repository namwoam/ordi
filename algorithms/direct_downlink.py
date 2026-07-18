"""Direct-downlink policy against Basilisk/BSK-RL access state."""
from dataclasses import dataclass, field

@dataclass
class Decision:
    epoch: int
    assignments: list[dict] = field(default_factory=list)

class DirectDownlink:
    name = "direct_downlink"
    def __init__(self, backend): self.backend = backend
    def schedule_epoch(self, epoch, opportunities, tasks):
        assignments=[]
        for task in tasks:
            if task.source_sat not in opportunities: continue
            for tile in task.tiles:
                if opportunities[task.source_sat]:
                    assignments.append({"task_id":task.task_id,"tile_id":tile.tile_id,
                                        "source":task.source_sat,"helpers":[]})
        return Decision(epoch, assignments)
