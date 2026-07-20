from .schema import Assignment, Decision
from ._common import earliest_direct_downlink, source_only_view
from ordi.eval.validation import DecisionFeasibilityModel, InvalidDecisionError
from ordi.sim.ground import H100_SXM_PROFILE

class DirectDownlink:
    name = "direct_downlink"
    def __init__(self): self.resources=DecisionFeasibilityModel()
    def schedule(self, request):
        out=[]
        for task in request.tasks:
            local=source_only_view(request,task.source_sat)
            state=local.satellites.get(task.source_sat)
            if not state or not state.available: continue
            for tile in task.tiles:
                route=earliest_direct_downlink(local,task.source_sat,tile.d_in_bits)
                if route and route.arrival<=task.deadline:
                    assignment=Assignment(task.task_id,tile.tile_id,task.source_sat,
                        downlink_only=True,metadata={"latency":route.arrival-request.sim_time,
                        "reliability":route.reliability*state.reliability,"path":route.path,
                        "downlink_bits":tile.d_in_bits,
                        "scheduled_at":request.sim_time,
                        "ground_station":route.path[-1],
                        "ground_compute_profile":H100_SXM_PROFILE.name,
                        "ground_compute_flops":tile.compute_ops,
                        "ground_compute_rate_flops_per_s":
                            H100_SXM_PROFILE.compute_rate_flops_per_s,
                        "ground_compute_power_w":H100_SXM_PROFILE.active_power_w,
                        "state_observer":task.source_sat,
                        "known_state_nodes":1,"max_state_age_s":0.0})
                    try:
                        out.append(self.resources.retime_and_reserve(request,assignment))
                    except InvalidDecisionError:
                        continue
        return Decision(request.epoch,tuple(out))
