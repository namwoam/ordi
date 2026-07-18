from .schema import Assignment, Decision
from ._common import enumerate_placements

class ServalLike:
    name = "serval_like"
    def schedule(self, request):
        out=[]
        for task in sorted(request.tasks,key=lambda x:(x.deadline,-max(t.utility for t in x.tiles))):
            by_helper={}
            for tile in task.tiles:
                for p in enumerate_placements(request,task,tile):
                    current=by_helper.setdefault(p.helper,{})
                    if tile.tile_id not in current or p.latency<current[tile.tile_id].latency:
                        current[tile.tile_id]=p
            complete=[(max(p.latency for p in tiles.values()),helper,tiles)
                      for helper,tiles in by_helper.items() if len(tiles)==len(task.tiles)]
            if not complete: continue
            _,helper,tiles=min(complete)
            for tile in sorted(task.tiles,key=lambda t:-t.utility):
                p=tiles[tile.tile_id]
                out.append(Assignment(task.task_id,tile.tile_id,task.source_sat,
                    (helper,),(p.aggregator,),metadata={"latency":p.latency,
                    "reliability":p.reliability*request.satellites[task.source_sat].reliability,
                    "energy_j":p.energy_j,
                    "task_level":True},routes=((p.route_in,p.route_out,p.route_down),)))
        return Decision(request.epoch,tuple(out))
