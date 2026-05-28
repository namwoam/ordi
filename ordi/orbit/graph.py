"""
Time-expanded contact graph for earliest-arrival routing.

Nodes: (node_id, epoch_index)
Edges: contact windows snapped to epoch boundaries.

earliest_arrival(src, dst, t_epoch, data_bits) returns transfer latency in seconds
using Dijkstra on the time-expanded graph.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import heapq
import math

from ordi.orbit.contacts import ContactEvent


@dataclass
class EpochContactGraph:
    """
    Precomputed contact graph for a single scheduling epoch [t_start, t_end].

    Attributes
    ----------
    epoch : int
    t_start, t_end : float  unix timestamps
    edges : list of (node_a, node_b, rate_bps, capacity_bits, link_type)
    nodes : set of node IDs (satellite names + ground station names)
    """
    epoch: int
    t_start: float
    t_end: float
    edges: List[Tuple[str, str, float, float, str]]  # (a, b, rate, capacity, type)
    nodes: set

    def capacity_between(self, a: str, b: str) -> float:
        """Total bits available from a→b in this epoch."""
        return sum(cap for (na, nb, _, cap, _) in self.edges
                   if na == a and nb == b)

    def rate_between(self, a: str, b: str) -> float:
        """Peak rate from a→b (first matching edge)."""
        for (na, nb, rate, _, _) in self.edges:
            if na == a and nb == b:
                return rate
        return 0.0


def build_epoch_graphs(
    contacts: List[ContactEvent],
    t_sim_start: float,
    epoch_length: float,
    n_epochs: int,
) -> List[EpochContactGraph]:
    """
    Snap contact windows to epoch buckets.
    A contact (t0, t1) contributes to every epoch it overlaps.
    The available capacity in an epoch is the overlap duration × rate.
    """
    graphs: List[EpochContactGraph] = []
    for ep in range(n_epochs):
        ep_start = t_sim_start + ep * epoch_length
        ep_end   = ep_start + epoch_length
        edges = []
        nodes: set = set()
        for c in contacts:
            overlap_start = max(c.t_start, ep_start)
            overlap_end   = min(c.t_end,   ep_end)
            if overlap_end <= overlap_start:
                continue
            overlap_bits = (overlap_end - overlap_start) * c.rate_bps
            edges.append((c.node_a, c.node_b, c.rate_bps, overlap_bits, c.link_type))
            nodes.add(c.node_a)
            nodes.add(c.node_b)
        graphs.append(EpochContactGraph(ep, ep_start, ep_end, edges, nodes))
    return graphs


# ── earliest-arrival routing ─────────────────────────────────────────────────

def earliest_arrival(
    src: str,
    dst: str,
    epoch: int,
    graphs: List[EpochContactGraph],
    data_bits: float,
    max_search_epochs: Optional[int] = None,
) -> float:
    """
    Compute the earliest-arrival transfer time (seconds) to send `data_bits`
    from `src` to `dst` starting at the given epoch, using a multi-hop path
    through the time-expanded contact graph.

    max_search_epochs: if set, limit search to this many epochs ahead (e.g.
    ceil(tau_k / epoch_length) + 1). Paths beyond that window would violate
    the deadline anyway, so this is correct for feasibility checks.

    Returns math.inf if no path exists within the remaining epochs.
    """
    n_epochs = len(graphs)
    if max_search_epochs is not None:
        n_epochs = min(n_epochs, epoch + max_search_epochs)

    # Dijkstra over (epoch, node) states.
    # State cost = wall-clock time elapsed from the start of `epoch`.
    INF = math.inf
    dist: Dict[Tuple[int, str], float] = {}
    # heap: (elapsed_seconds, current_epoch, current_node, bits_remaining)
    heap = [(0.0, epoch, src, data_bits)]

    while heap:
        elapsed, ep, node, bits_left = heapq.heappop(heap)

        if node == dst and bits_left <= 0:
            return elapsed

        state = (ep, node)
        if dist.get(state, INF) < elapsed:
            continue
        dist[state] = elapsed

        if ep >= n_epochs:
            continue

        g = graphs[ep]
        ep_elapsed_so_far = elapsed - (g.t_start - graphs[epoch].t_start)

        for (na, nb, rate, cap, _) in g.edges:
            if na != node:
                continue
            # Time to transfer bits_left over this link in this epoch
            time_to_send = bits_left / rate if rate > 0 else INF
            # How much of the epoch is left?
            ep_remaining = g.t_end - g.t_start - max(0.0, ep_elapsed_so_far)
            if ep_remaining <= 0:
                continue
            bits_sent = min(bits_left, min(cap, ep_remaining * rate))
            new_bits_left = max(0.0, bits_left - bits_sent)
            time_used = bits_sent / rate if rate > 0 else 0.0
            new_elapsed = elapsed + time_used

            if new_bits_left <= 0:
                # Delivery complete on this hop
                next_state = (ep, nb)
                if dist.get(next_state, INF) > new_elapsed:
                    heapq.heappush(heap, (new_elapsed, ep, nb, 0.0))
            else:
                # Carry over to next epoch
                next_ep = ep + 1
                if next_ep < n_epochs:
                    next_elapsed = new_elapsed + (graphs[next_ep].t_start - g.t_end)
                    next_state = (next_ep, nb)
                    if dist.get(next_state, INF) > next_elapsed:
                        heapq.heappush(heap, (next_elapsed, next_ep, nb, new_bits_left))

        # Also allow waiting at current node for next epoch
        if ep + 1 < n_epochs:
            wait = graphs[ep + 1].t_start - g.t_start - max(0.0, ep_elapsed_so_far)
            next_state = (ep + 1, node)
            new_elapsed = elapsed + max(0.0, wait)
            if dist.get(next_state, INF) > new_elapsed:
                heapq.heappush(heap, (new_elapsed, ep + 1, node, bits_left))

    return INF


def earliest_downlink(
    aggregator: str,
    epoch: int,
    graphs: List[EpochContactGraph],
    data_bits: float,
    ground_stations: set,
    max_search_epochs: Optional[int] = None,
) -> float:
    """Earliest transfer time from `aggregator` to any reachable ground station."""
    best = math.inf
    for gs in ground_stations:
        t = earliest_arrival(aggregator, gs, epoch, graphs, data_bits,
                             max_search_epochs=max_search_epochs)
        if t < best:
            best = t
    return best
