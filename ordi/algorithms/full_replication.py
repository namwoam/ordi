from .schema import Assignment, Decision
from ._common import enumerate_placements, tile_success

class FullReplication:
    name = "full_replication"
    def schedule(self, request):
        out=[]
        for task in request.tasks:
            for tile in task.tiles:
                choices=sorted(enumerate_placements(request,task,tile),key=lambda p:p.reliability,reverse=True)
                unique=[]
                for p in choices:
                    if p.helper not in {x.helper for x in unique}: unique.append(p)
                    if len(unique)>=tile.n_replicas_max: break
                if unique: out.append(Assignment(task.task_id,tile.tile_id,task.source_sat,
                    tuple(p.helper for p in unique),tuple(p.aggregator for p in unique),
                    metadata={"latency":min(p.latency for p in unique),
                    "reliability":tile_success(request,task,unique),
                    "energy_j":sum(p.energy_j for p in unique),"replication":"full"},
                    routes=tuple((p.route_in,p.route_out,p.route_down) for p in unique)))
        return Decision(request.epoch,tuple(out))
