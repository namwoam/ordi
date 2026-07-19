"""Greedy cooperative placement without redundancy.

This is an implementation-defined control policy, not a reproduction of SECO.
Each tile independently selects its lowest-latency feasible placement, using
energy as the tie breaker, and creates exactly one replica.
"""

from ._common import enumerate_placements
from .schema import Assignment, Decision


class GreedyNonredundant:
    name = "greedy_nonredundant"

    def schedule(self, request):
        assignments = []
        for task in request.tasks:
            for tile in task.tiles:
                choices = enumerate_placements(request, task, tile)
                if not choices:
                    continue
                placement = min(
                    choices, key=lambda item: (item.latency, item.energy_j)
                )
                assignments.append(Assignment(
                    task.task_id,
                    tile.tile_id,
                    task.source_sat,
                    (placement.helper,),
                    (placement.aggregator,),
                    metadata={
                        "latency": placement.latency,
                        "reliability": (
                            placement.reliability
                            * request.satellites[task.source_sat].reliability
                        ),
                        "energy_j": placement.energy_j,
                    },
                    routes=((placement.route_in, placement.route_out,
                             placement.route_down),),
                ))
        return Decision(request.epoch, tuple(assignments))
