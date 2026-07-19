from .schema import Assignment, Decision
from ._common import enumerate_placements
from ordi.eval.validation import DecisionFeasibilityModel, InvalidDecisionError

class OnboardOnly:
    name = "onboard_only"
    def __init__(self): self.resources=DecisionFeasibilityModel()
    def schedule(self, request):
        out=[]
        for task in request.tasks:
            state=request.satellites.get(task.source_sat)
            if state and state.available:
                for tile in task.tiles:
                    choices=[p for p in enumerate_placements(request,task,tile)
                             if p.helper==task.source_sat and p.aggregator==task.source_sat]
                    if choices:
                        p=min(choices,key=lambda x:x.latency)
                        assignment=Assignment(task.task_id,tile.tile_id,task.source_sat,
                            (p.helper,),(p.aggregator,),metadata={"latency":p.latency,
                            "reliability":p.reliability*state.reliability,
                            "energy_j":p.energy_j},routes=((p.route_in,p.route_out,p.route_down),))
                        try:
                            out.append(self.resources.retime_and_reserve(request,assignment))
                        except InvalidDecisionError:
                            continue
        return Decision(request.epoch,tuple(out))
