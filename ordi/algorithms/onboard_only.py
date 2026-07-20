from .schema import Assignment, Decision
from ._common import enumerate_placements, source_only_view
from ordi.eval.validation import DecisionFeasibilityModel, InvalidDecisionError

class OnboardOnly:
    name = "onboard_only"
    def __init__(self): self.resources=DecisionFeasibilityModel()
    def schedule(self, request):
        out=[]
        for task in request.tasks:
            local=source_only_view(request,task.source_sat)
            state=local.satellites.get(task.source_sat)
            if state and state.available:
                for tile in task.tiles:
                    choices=[p for p in enumerate_placements(local,task,tile)
                             if p.helper==task.source_sat and p.aggregator==task.source_sat]
                    if choices:
                        p=min(choices,key=lambda x:x.latency)
                        assignment=Assignment(task.task_id,tile.tile_id,task.source_sat,
                            (p.helper,),(p.aggregator,),metadata={"latency":p.latency,
                            "reliability":p.reliability*state.reliability,
                            "state_observer":task.source_sat,
                            "known_state_nodes":1,"max_state_age_s":0.0},
                            routes=((p.route_in,p.route_out,p.route_down),))
                        try:
                            out.append(self.resources.retime_and_reserve(request,assignment))
                        except InvalidDecisionError:
                            continue
        return Decision(request.epoch,tuple(out))
