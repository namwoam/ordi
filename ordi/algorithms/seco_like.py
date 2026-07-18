from .schema import Assignment, Decision
from ._common import enumerate_placements

class SECOLike:
    name = "seco_like"
    def schedule(self, request):
        out=[]
        for task in request.tasks:
            for tile in task.tiles:
                choices=enumerate_placements(request,task,tile)
                if not choices: continue
                p=min(choices,key=lambda x:(x.latency,x.energy_j))
                out.append(Assignment(task.task_id,tile.tile_id,task.source_sat,
                    (p.helper,),(p.aggregator,),metadata={"latency":p.latency,
                    "reliability":p.reliability*request.satellites[task.source_sat].reliability,
                    "energy_j":p.energy_j},routes=((p.route_in,p.route_out,p.route_down),)))
        return Decision(request.epoch,tuple(out))
