"""Time-dependent routing and placement shared fairly by all policies."""
from __future__ import annotations
from dataclasses import dataclass
import heapq
import math

@dataclass(frozen=True)
class Route:
    arrival: float
    reliability: float
    path: tuple[str, ...]
    bits: float

@dataclass(frozen=True)
class Placement:
    helper: str
    aggregator: str
    latency: float
    reliability: float
    energy_j: float
    communication_bits: float
    route_in: tuple[str, ...]
    route_out: tuple[str, ...]
    route_down: tuple[str, ...]

def available_candidates(request, source):
    reachable={source}
    for contact in request.contacts:
        if contact.closes >= request.sim_time:
            reachable.add(contact.source); reachable.add(contact.target)
    return [sid for sid,state in request.satellites.items()
            if state.available and sid in reachable]

def load_key(state): return state.queued_flops/max(state.compute_rate,1.0)

def plane(sat_id):
    parts=sat_id.split("_"); return parts[1] if len(parts)>2 else sat_id

_CONTACT_INDEX_CACHE={}

def _contacts_by_source(contacts):
    """Index one immutable epoch contact set once without re-hashing it."""
    key=id(contacts)
    cached=_CONTACT_INDEX_CACHE.get(key)
    if cached is not None and cached[0] is contacts:
        return cached[1]
    indexed={}
    for contact in sorted(contacts,key=lambda c:c.opens):
        indexed.setdefault(contact.source,[]).append(contact)
    result={source:tuple(windows) for source,windows in indexed.items()}
    if len(_CONTACT_INDEX_CACHE)>=128:
        _CONTACT_INDEX_CACHE.clear()
    _CONTACT_INDEX_CACHE[key]=(contacts,result)
    return result

def earliest_route(request, source, targets, bits, start=None):
    targets=set(targets)
    if source in targets: return Route(start or request.sim_time,1.0,(source,),bits)
    start=request.sim_time if start is None else start
    contacts_by_source=_contacts_by_source(tuple(request.contacts))
    best={source:start}; rel={source:1.0}; paths={source:(source,)}
    queue=[(start,source)]
    while queue:
        now,node=heapq.heappop(queue)
        if now!=best[node]: continue
        if node in targets: return Route(now,rel[node],paths[node],bits)
        for c in contacts_by_source.get(node,()):
            if c.closes<now: continue
            depart=max(now,c.opens); finish=depart+bits/max(c.rate_bps,1.0)
            if finish>c.closes: continue
            if finish<best.get(c.target,math.inf):
                best[c.target]=finish; rel[c.target]=rel[node]*c.reliability
                paths[c.target]=paths[node]+(c.target,); heapq.heappush(queue,(finish,c.target))
    return None

def earliest_downlink(request, source, bits, start=None):
    return earliest_route(request,source,request.ground_stations,bits,start)

def earliest_direct_downlink(request, source, bits, start=None):
    """Earliest source-to-ground contact without an ISL relay."""
    start=request.sim_time if start is None else start
    feasible=[]
    for contact in request.contacts:
        if contact.source!=source or contact.target not in request.ground_stations:
            continue
        depart=max(start,contact.opens)
        finish=depart+bits/max(contact.rate_bps,1.0)
        if finish<=contact.closes:
            feasible.append(Route(finish,contact.reliability,
                                  (source,contact.target),bits))
    return min(feasible,key=lambda route:route.arrival) if feasible else None

def enumerate_placements(request, task, tile, allow_source=True):
    deadline=task.deadline; out=[]
    source_state=request.satellites.get(task.source_sat)
    if source_state is None or not source_state.available:
        return out
    for helper,hstate in request.satellites.items():
        if not hstate.available or (not allow_source and helper==task.source_sat): continue
        route_in=earliest_route(request,task.source_sat,{helper},tile.d_in_bits)
        if route_in is None: continue
        compute_start=route_in.arrival
        compute_time=(hstate.queued_flops+tile.compute_ops)/max(hstate.compute_rate,1.0)
        compute_done=compute_start+compute_time
        for agg,astate in request.satellites.items():
            if not astate.available: continue
            route_out=earliest_route(request,helper,{agg},tile.d_out_bits,compute_done)
            if route_out is None: continue
            down=earliest_downlink(request,agg,tile.d_out_bits,route_out.arrival)
            if down is None or down.arrival>deadline: continue
            participating={helper,agg}-{task.source_sat}
            node_reliability=math.prod(request.satellites[node].reliability
                                      for node in participating)
            p=(route_in.reliability*route_out.reliability*down.reliability
               *node_reliability)
            compute_e=(hstate.compute_power_w * tile.compute_ops
                       / max(hstate.compute_rate, 1.0))
            isl_bits=(tile.d_in_bits*max(len(route_in.path)-1,0)
                      + tile.d_out_bits*max(len(route_out.path)-1,0))
            comm_bits=isl_bits+tile.d_out_bits*max(len(down.path)-1,0)
            # Ground transmit energy is accounted once by the evaluator.
            comm_e=hstate.comms_power_w*isl_bits/200e6
            out.append(Placement(helper,agg,down.arrival-request.sim_time,p,
                compute_e+comm_e,comm_bits,route_in.path,route_out.path,down.path))
    return out

def independent_success(placements):
    failure=1.0
    for p in placements: failure*=1.0-p.reliability
    return 1.0-failure

def tile_success(request, task, placements):
    """Replica-union reliability with the source node counted exactly once."""
    return request.satellites[task.source_sat].reliability*independent_success(placements)
