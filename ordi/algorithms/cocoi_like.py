from .schema import Assignment, Decision
from ._common import enumerate_placements
import math

def _at_least_k(probabilities,k):
    distribution=[1.0]+[0.0]*len(probabilities)
    for probability in probabilities:
        for successes in range(len(probabilities),0,-1):
            distribution[successes]=distribution[successes]*(1-probability)+distribution[successes-1]*probability
        distribution[0]*=1-probability
    return sum(distribution[k:])

class CoCoILike:
    name = "cocoi_like"
    def schedule(self, request):
        out=[]
        for task in request.tasks:
            for tile in task.tiles:
                choices=sorted(enumerate_placements(request,task,tile),key=lambda p:(p.latency,p.energy_j)); unique=[]
                for p in choices:
                    if p.helper not in {x.helper for x in unique}: unique.append(p)
                    if len(unique)>=tile.n_replicas_max: break
                if unique:
                    k=max(1,math.ceil(len(unique)/2)); probs=sorted((p.reliability for p in unique),reverse=True)
                    reconstruction=(request.satellites[task.source_sat].reliability
                                    *_at_least_k(probs,k))
                    out.append(Assignment(task.task_id,tile.tile_id,task.source_sat,
                        tuple(p.helper for p in unique),tuple(p.aggregator for p in unique),
                        metadata={"coding":"mds","data_shards":k,"total_shards":len(unique),
                        "latency":max(sorted(p.latency for p in unique)[:k]),
                        "energy_j":sum(p.energy_j for p in unique),
                        "reconstruction_probability":reconstruction},
                        routes=tuple((p.route_in,p.route_out,p.route_down) for p in unique)))
        return Decision(request.epoch,tuple(out))
