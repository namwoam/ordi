from .schema import Assignment, Decision
from ._common import earliest_direct_downlink

class CompressionOnly:
    name = "compression_only"
    def __init__(self, ratio=0.15): self.ratio=ratio
    def schedule(self, request):
        out=[]
        for task in request.tasks:
            state=request.satellites.get(task.source_sat)
            if state and state.available:
                for tile in task.tiles:
                    compute=(state.queued_flops+0.1*tile.compute_ops)/max(state.compute_rate,1.0)
                    bits=tile.d_in_bits*self.ratio
                    route=earliest_direct_downlink(request,task.source_sat,bits,request.sim_time+compute)
                    if route and route.arrival<=task.deadline:
                        out.append(Assignment(task.task_id,tile.tile_id,task.source_sat,
                            (task.source_sat,),(task.source_sat,),metadata={
                            "compression_ratio":self.ratio,"downlink_bits":bits,
                            "latency":route.arrival-request.sim_time,
                            "reliability":route.reliability*state.reliability,
                            "compute_flops":0.1*tile.compute_ops,
                            "energy_j":state.compute_power_w*0.1*tile.compute_ops
                                       / max(state.compute_rate,1.0)}))
        return Decision(request.epoch,tuple(out))
