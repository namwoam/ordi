import random
from .schema import Assignment, Decision
from ._common import enumerate_placements, tile_success
from ordi.eval.validation import DecisionFeasibilityModel, InvalidDecisionError

class RandomReplication:
    name = "random_replication"
    def __init__(self,seed=0):
        self.seed=seed
        self.resources=DecisionFeasibilityModel()
    def schedule(self, request):
        rng=random.Random(self.seed+request.epoch); out=[]
        for task in request.tasks:
            for tile in task.tiles:
                choices=enumerate_placements(request,task,tile); per_helper={}
                for p in choices: per_helper.setdefault(p.helper,p)
                pool=list(per_helper.values()); n=min(tile.n_replicas_max,len(pool)); selected=rng.sample(pool,n)
                if selected:
                    assignment=Assignment(task.task_id,tile.tile_id,task.source_sat,
                    tuple(p.helper for p in selected),tuple(p.aggregator for p in selected),
                    metadata={"latency":min(p.latency for p in selected),
                    "reliability":tile_success(request,task,selected),
                    "energy_j":sum(p.energy_j for p in selected),"replication":"random"},
                    routes=tuple((p.route_in,p.route_out,p.route_down) for p in selected))
                    try:
                        out.append(self.resources.retime_and_reserve(request,assignment))
                    except InvalidDecisionError:
                        continue
        return Decision(request.epoch,tuple(out))
