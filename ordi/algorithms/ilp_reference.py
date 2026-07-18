"""Small-instance MILP reference using the same policy schema as every baseline."""
from __future__ import annotations

import math

from ._common import enumerate_placements, tile_success
from .schema import Assignment, Decision


class ILPReference:
    """Maximize the linearized ORDI score subject to compute and replica caps.

    This policy is intended only for the small E8 instances. It depends on no
    ORDI implementation details: candidate placements come from the shared
    Basilisk state/contact snapshot and the result is a normal ``Decision``.
    """

    name = "ilp_reference"

    def __init__(self, time_limit_s: float = 30.0, threads: int = 2):
        self.time_limit_s = time_limit_s
        self.threads = threads

    def schedule(self, request):
        import pulp

        candidates = []
        tiles = {}
        for task in request.tasks:
            for tile in task.tiles:
                key = (task.task_id, tile.tile_id)
                tiles[key] = (task, tile)
                for placement in enumerate_placements(request, task, tile):
                    candidates.append((key, placement))

        if not candidates:
            return Decision(request.epoch)

        model = pulp.LpProblem("ordi_reference", pulp.LpMaximize)
        choose = {
            i: pulp.LpVariable(f"x_{i}", cat="Binary")
            for i in range(len(candidates))
        }
        scores = {}
        for i, (key, placement) in enumerate(candidates):
            _task, tile = tiles[key]
            w = request.weights
            scores[i] = (
                tile.utility * request.satellites[_task.source_sat].reliability
                * placement.reliability
                * math.exp(-w.freshness * placement.latency)
                - w.energy * placement.energy_j
                - w.communication * placement.communication_bits
            )
        active = {key: pulp.LpVariable(
            f"active_{key[0]}_{key[1]}", cat="Binary"
        ) for key in tiles}
        model += (pulp.lpSum(scores[i] * choose[i] for i in choose)
                  - request.weights.replication * pulp.lpSum(
                      pulp.lpSum(choose[i] for i, (candidate_key, _p)
                                 in enumerate(candidates) if candidate_key == key)
                      - active[key]
                      for key in tiles
                  ))

        for key, (_task, tile) in tiles.items():
            indices = [i for i, (candidate_key, _p) in enumerate(candidates)
                       if candidate_key == key]
            model += pulp.lpSum(choose[i] for i in indices) <= tile.n_replicas_max
            model += pulp.lpSum(choose[i] for i in indices) >= active[key]
            model += pulp.lpSum(choose[i] for i in indices) <= len(indices) * active[key]
            for helper in request.satellites:
                helper_indices = [i for i in indices
                                  if candidates[i][1].helper == helper]
                if helper_indices:
                    model += pulp.lpSum(choose[i] for i in helper_indices) <= 1

        for sat_id, state in request.satellites.items():
            indices = [i for i, (key, p) in enumerate(candidates)
                       if p.helper == sat_id]
            model += pulp.lpSum(
                tiles[candidates[i][0]][1].compute_ops * choose[i]
                for i in indices
            ) <= max(0.0, state.compute_rate * request.epoch_length)

        solver = pulp.HiGHS(
            msg=False, timeLimit=self.time_limit_s, threads=self.threads
        )
        model.solve(solver)

        selected = {}
        for i, (key, placement) in enumerate(candidates):
            if choose[i].value() is not None and choose[i].value() > 0.5:
                selected.setdefault(key, []).append(placement)

        assignments = []
        for key, placements in selected.items():
            task, _tile = tiles[key]
            assignments.append(Assignment(
                key[0], key[1], task.source_sat,
                tuple(p.helper for p in placements),
                tuple(p.aggregator for p in placements),
                metadata={
                    "latency": min(p.latency for p in placements),
                    "reliability": tile_success(request, task, placements),
                    "energy_j": sum(p.energy_j for p in placements),
                    "objective": (sum(scores[i] for i, (candidate_key, p)
                                      in enumerate(candidates)
                                      if candidate_key == key and p in placements)
                                  - request.weights.replication
                                  * max(0, len(placements) - 1)),
                    "solver": "highs",
                },
                routes=tuple((p.route_in,p.route_out,p.route_down)
                             for p in placements),
            ))
        return Decision(request.epoch, tuple(assignments))
