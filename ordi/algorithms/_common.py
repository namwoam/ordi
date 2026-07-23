"""Time-dependent routing and placement shared fairly by all policies."""
from __future__ import annotations
from dataclasses import dataclass, replace
from itertools import combinations
import heapq
import math
from .schema import NodeDecision, WorkItem

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
    communication_bits: float
    route_in: tuple[str, ...]
    route_out: tuple[str, ...]
    route_down: tuple[str, ...]
    active_start: float = 0.0
    input_done: float = 0.0
    compute_done: float = 0.0
    output_done: float = 0.0
    delivery_done: float = 0.0

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

def deadline_expired(request, task):
    """Whether no on-time delivery remains possible for a task."""
    return task.deadline <= request.sim_time + 1e-9


def earliest_route(request, source, targets, bits, start=None, latest=None):
    targets=set(targets)
    if source in targets: return Route(start or request.sim_time,1.0,(source,),bits)
    start=request.sim_time if start is None else start
    contacts_by_source=_contacts_by_source(tuple(request.contacts))
    best={source:start}; rel={source:1.0}; paths={source:(source,)}
    queue=[(start,source)]
    while queue:
        now,node=heapq.heappop(queue)
        if now!=best[node]: continue
        if latest is not None and now > latest + 1e-9: continue
        if node in targets: return Route(now,rel[node],paths[node],bits)
        for c in contacts_by_source.get(node,()):
            if c.closes<now: continue
            depart=max(now,c.opens); finish=depart+bits/max(c.rate_bps,1.0)
            if finish>c.closes: continue
            if latest is not None and finish > latest + 1e-9: continue
            if finish<best.get(c.target,math.inf):
                best[c.target]=finish; rel[c.target]=rel[node]*c.reliability
                paths[c.target]=paths[node]+(c.target,); heapq.heappush(queue,(finish,c.target))
    return None

def earliest_downlink(request, source, bits, start=None, latest=None):
    return earliest_route(
        request, source, request.ground_stations, bits, start, latest
    )

def earliest_direct_downlink(request, source, bits, start=None, latest=None):
    """Earliest source-to-ground contact without an ISL relay."""
    start=request.sim_time if start is None else start
    feasible=[]
    for contact in request.contacts:
        if contact.source!=source or contact.target not in request.ground_stations:
            continue
        depart=max(start,contact.opens)
        finish=depart+bits/max(contact.rate_bps,1.0)
        if latest is not None and finish > latest + 1e-9:
            continue
        if finish<=contact.closes:
            feasible.append(Route(finish,contact.reliability,
                                  (source,contact.target),bits))
    return min(feasible,key=lambda route:route.arrival) if feasible else None

def enumerate_placements(request, task, tile, allow_source=True,
                         work_fraction=1.0, input_fraction=1.0,
                         output_fraction=1.0,
                         protocol_header_bits=0.0):
    deadline=task.deadline; out=[]
    if deadline_expired(request, task):
        return out
    source_state=request.satellites.get(task.source_sat)
    if source_state is None or not source_state.available:
        return out
    for helper,hstate in request.satellites.items():
        if not hstate.available or (not allow_source and helper==task.source_sat): continue
        input_bits=tile.d_in_bits*input_fraction
        output_bits=tile.d_out_bits*output_fraction
        input_transfer_bits=input_bits+protocol_header_bits
        output_transfer_bits=output_bits+protocol_header_bits
        work=tile.compute_ops*work_fraction
        optimistic_compute_done = (
            request.sim_time
            + (hstate.queued_flops + work)
            / max(hstate.compute_rate, 1.0)
        )
        if optimistic_compute_done > deadline + 1e-9:
            continue
        route_in=earliest_route(
            request, task.source_sat, {helper}, input_transfer_bits,
            latest=deadline,
        )
        if route_in is None: continue
        compute_start=route_in.arrival
        compute_time=(hstate.queued_flops+work)/max(hstate.compute_rate,1.0)
        compute_done=compute_start+compute_time
        if compute_done > deadline + 1e-9:
            continue
        for agg,astate in request.satellites.items():
            if not astate.available: continue
            route_out=earliest_route(
                request, helper, {agg}, output_transfer_bits, compute_done,
                deadline,
            )
            if route_out is None: continue
            down=earliest_downlink(
                request, agg, output_transfer_bits, route_out.arrival,
                deadline,
            )
            if down is None or down.arrival>deadline: continue
            isl_bits=(input_transfer_bits*max(len(route_in.path)-1,0)
                      + output_transfer_bits*max(len(route_out.path)-1,0))
            comm_bits=(
                isl_bits
                + output_transfer_bits*max(len(down.path)-1,0)
            )
            placement = Placement(
                helper, agg, down.arrival-request.sim_time, 0.0,
                comm_bits, route_in.path, route_out.path, down.path,
                request.sim_time, route_in.arrival, compute_done,
                route_out.arrival, down.arrival,
            )
            # Placement reliability excludes the shared source component for
            # compatibility with callers that compare individual placements.
            components = placement_components(request, task, placement)
            p = math.prod(
                probability for component, probability in components.items()
                if not (component[0] == "node"
                        and component[1] == task.source_sat)
            )
            out.append(replace(placement, reliability=p))
    return out

def _edge_reliability(request, source, target, kind):
    """Reliability of the contact class used by a selected route edge."""
    values = [
        contact.reliability for contact in request.contacts
        if contact.source == source and contact.target == target
        and contact.kind == kind
    ]
    return max(values, default=1.0)


def placement_components(request, task, placement):
    """Independent physical components required by one placement.

    Keys identify shared components, allowing a group of shards or replicas to
    count a shared node/link/downlink exactly once rather than once per path.
    Source survival is keyed by reliability epoch here so it is shared once
    across replicas while still accumulating over the source's exposure time.
    """
    components = {}

    epoch_length = max(float(request.epoch_length), 1e-9)

    def periods(start, end):
        first = int(math.floor(max(0.0, start) / epoch_length))
        last = int(math.floor(max(start, end - 1e-9) / epoch_length))
        return range(first, last + 1)

    def add_nodes(path, start, end):
        for node in path:
            state = request.satellites.get(node)
            if state is None:
                continue
            for period in periods(start, end):
                components[("node", node, period)] = state.reliability

    route_in, route_out, route_down = (
        placement.route_in, placement.route_out, placement.route_down
    )
    add_nodes(route_in, placement.active_start, placement.input_done)
    add_nodes(
        (placement.helper,), placement.input_done, placement.compute_done
    )
    add_nodes(route_out, placement.compute_done, placement.output_done)
    add_nodes(route_down, placement.output_done, placement.delivery_done)
    for path in (route_in, route_out):
        for source, target in zip(path, path[1:]):
            components[("isl", source, target)] = _edge_reliability(
                request, source, target, "isl"
            )
    for source, target in zip(route_down, route_down[1:]):
        kind = "downlink" if target in request.ground_stations else "isl"
        components[(kind, source, target)] = _edge_reliability(
            request, source, target, kind
        )
    return components


def group_success(request, task, placements):
    """Probability that every required shard in one group succeeds."""
    components = {}
    for placement in placements:
        components.update(placement_components(request, task, placement))
    return math.prod(components.values())


def groups_success(request, task, groups):
    """Exact union probability for complete reconstruction groups.

    Inclusion-exclusion is inexpensive here (normally one primary and one
    backup) and correctly retains components shared by multiple groups.
    """
    groups = tuple(tuple(group) for group in groups if group)
    if not groups:
        return 0.0
    total = 0.0
    for count in range(1, len(groups) + 1):
        sign = 1.0 if count % 2 else -1.0
        for subset in combinations(groups, count):
            components = {}
            for group in subset:
                for placement in group:
                    components.update(
                        placement_components(request, task, placement)
                    )
            total += sign * math.prod(components.values())
    return max(0.0, min(1.0, total))


def independent_success(placements):
    """Compatibility helper for callers without component information."""
    failure=1.0
    for p in placements: failure*=1.0-p.reliability
    return 1.0-failure


def tile_success(request, task, placements):
    """Replica-union reliability using shared-component inclusion-exclusion."""
    return groups_success(
        request, task, ((placement,) for placement in placements)
    )


def protocol_trace(request, task, tile, groups, work_fraction=1.0,
                   input_fraction=1.0, output_fraction=1.0):
    """Build node-local work messages for a policy-selected placement.

    This is a shared wire-format helper, not a scheduling policy: each
    algorithm remains responsible for choosing groups, splits, and replicas.
    """
    root = WorkItem(
        task.task_id, tile.tile_id,
        tuple(sorted(request.ground_stations)), task.source_sat,
    )
    decisions = []
    for group_id, group in enumerate(groups):
        leaves = []
        for placement in group:
            route_down = placement.route_down
            if hasattr(route_down, "path"):
                route_down = route_down.path
            leaves.append(WorkItem(
                task.task_id, tile.tile_id,
                route_down[-1] if route_down else placement.aggregator,
                placement.helper, work_fraction, input_fraction,
                output_fraction, group_id, 1,
            ))
        leaves = tuple(leaves)
        if group_id > 0:
            action = "replicate"
        elif len(group) > 1:
            action = "split"
        elif group[0].helper == task.source_sat:
            action = "execute_forward"
        else:
            action = "delegate"
        decisions.append(NodeDecision(
            task.source_sat, action, root,
            () if action == "execute_forward" else leaves,
            reason=f"{action} selected by {task.source_sat}",
        ))
        for leaf, placement in zip(leaves, group):
            if action == "execute_forward" and placement.helper == task.source_sat:
                continue
            decisions.append(NodeDecision(
                placement.helper, "execute_forward", leaf, (),
                reason="terminal policy-selected work item",
            ))
    return tuple(decisions)


def advertisement_metadata(batch):
    return {
        "protocol_message_count": batch.message_count,
        "protocol_control_bits": batch.control_bits,
        "advertisement_control_bits": batch.control_bits,
    }


def source_only_view(request, source):
    """Local state for policies that never discover or delegate to helpers."""
    known = ({source: request.satellites[source]}
             if source in request.satellites else {})
    allowed = {source} | set(request.ground_stations)
    return replace(
        request,
        satellites=known,
        contacts=tuple(
            contact for contact in request.contacts
            if contact.source in allowed and contact.target in allowed
        ),
        opportunities={source: ()},
        state_age_s={source: 0.0},
        observer=source,
    )
