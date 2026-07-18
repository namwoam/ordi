from .schema import Assignment, Decision
from ._common import earliest_direct_downlink

class DirectDownlink:
    name = "direct_downlink"
    def schedule(self, request):
        out=[]
        for task in request.tasks:
            state=request.satellites.get(task.source_sat)
            if not state or not state.available: continue
            for tile in task.tiles:
                route=earliest_direct_downlink(request,task.source_sat,tile.d_in_bits)
                if route and route.arrival<=task.deadline:
                    out.append(Assignment(task.task_id,tile.tile_id,task.source_sat,
                        downlink_only=True,metadata={"latency":route.arrival-request.sim_time,
                        "reliability":route.reliability*state.reliability,"path":route.path,
                        "downlink_bits":tile.d_in_bits}))
        return Decision(request.epoch,tuple(out))
