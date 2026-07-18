from .schema import Assignment, Decision
import math
from ._common import enumerate_placements, independent_success, plane, tile_success

class ORDI:
    name = "ordi"
    def __init__(self,max_replicas=2): self.max_replicas=max_replicas
    def _utility(self,request,task,tile,p):
        w=request.weights
        source_p=request.satellites[task.source_sat].reliability
        return tile.utility*source_p*p.reliability*math.exp(-w.freshness*p.latency)-w.energy*p.energy_j-w.communication*p.communication_bits
    def schedule(self, request):
        out=[]
        for task in request.tasks:
            for tile in task.tiles:
                choices=enumerate_placements(request,task,tile)
                if not choices: continue
                primary=max(choices,key=lambda p:self._utility(request,task,tile,p))
                if self._utility(request,task,tile,primary)<=0: continue
                selected=[primary]; old_p=primary.reliability
                remaining=sorted((p for p in choices if p.helper!=primary.helper),
                    key=lambda p:p.reliability,reverse=True)
                for candidate in remaining:
                    if len(selected)>=min(self.max_replicas,tile.n_replicas_max): break
                    used_nodes={n for p in selected for n in p.route_in+p.route_out}
                    candidate_nodes=set(candidate.route_in+candidate.route_out)
                    if plane(candidate.helper)==plane(primary.helper) or used_nodes.intersection(candidate_nodes-{task.source_sat}): continue
                    new_p=independent_success(selected+[candidate]); gain=(tile.utility*request.satellites[task.source_sat].reliability*(new_p-old_p)*
                        math.exp(-request.weights.freshness*min(p.latency for p in selected+[candidate]))-
                        request.weights.energy*candidate.energy_j-
                        request.weights.communication*candidate.communication_bits-
                        request.weights.replication)
                    if gain>0: selected.append(candidate); old_p=new_p
                out.append(Assignment(task.task_id,tile.tile_id,task.source_sat,
                    tuple(p.helper for p in selected),tuple(p.aggregator for p in selected),
                    metadata={"latency":min(p.latency for p in selected),
                    "reliability":tile_success(request,task,selected),"selective_redundancy":True,
                    "energy_j":sum(p.energy_j for p in selected),
                    "objective":sum(self._utility(request,task,tile,p) for p in selected)},
                    routes=tuple((p.route_in,p.route_out,p.route_down) for p in selected)))
        return Decision(request.epoch,tuple(out))
