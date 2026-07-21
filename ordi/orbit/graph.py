"""
Time-expanded contact graph for earliest-arrival routing.

Nodes: (node_id, epoch_index)
Edges: contact windows snapped to epoch boundaries.

earliest_arrival(src, dst, t_epoch, data_bits) returns transfer latency in seconds
using Dijkstra on the time-expanded graph.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple
import heapq
import math

import numpy as np

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
    # Exact overlap represented by each edge.  Keeping this parallel to ``edges``
    # preserves the public edge tuple while avoiding the old lossy reconstruction
    # that moved every partial contact to the start of its epoch.
    edge_windows: List[Tuple[float, float]] = field(default_factory=list)
    # Adjacency list (src → [(dst, rate, capacity)]) so earliest_arrival walks a
    # node's out-edges instead of scanning all edges per pop. Same result, faster.
    adj: Dict[str, List[Tuple[str, float, float]]] = field(
        default_factory=dict, compare=False, repr=False)

    def __post_init__(self):
        if not self.edge_windows:
            self.edge_windows = [
                (self.t_start, min(self.t_end, self.t_start + cap / max(rate, 1.0)))
                for (_na, _nb, rate, cap, _t) in self.edges
            ]
        if len(self.edge_windows) != len(self.edges):
            raise ValueError("edge_windows must have one entry per edge")
        adj: Dict[str, List[Tuple[str, float, float]]] = {}
        for (na, nb, rate, cap, _t) in self.edges:
            adj.setdefault(na, []).append((nb, rate, cap))
        self.adj = adj

    def capacity_between(self, a: str, b: str) -> float:
        """Total bits available from a→b in this epoch."""
        return sum(cap for (nb, _r, cap) in self.adj.get(a, ()) if nb == b)

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
        edge_windows = []
        nodes: set = set()
        for c in contacts:
            overlap_start = max(c.t_start, ep_start)
            overlap_end   = min(c.t_end,   ep_end)
            if overlap_end <= overlap_start:
                continue
            overlap_bits = (overlap_end - overlap_start) * c.rate_bps
            edges.append((c.node_a, c.node_b, c.rate_bps, overlap_bits, c.link_type))
            edge_windows.append((overlap_start, overlap_end))
            nodes.add(c.node_a)
            nodes.add(c.node_b)
        graphs.append(EpochContactGraph(
            ep, ep_start, ep_end, edges, nodes, edge_windows
        ))
    return graphs


# ── earliest-arrival routing ─────────────────────────────────────────────────

def earliest_arrival(
    src: str,
    dst: str,
    epoch: int,
    graphs: List[EpochContactGraph],
    data_bits: float,
    max_search_epochs: Optional[int] = None,
    node_index: Optional[Dict[str, int]] = None,
) -> float:
    """
    Compute the earliest-arrival transfer time (seconds) to send `data_bits`
    from `src` to `dst` starting at the given epoch, using a multi-hop path
    through the time-expanded contact graph.

    node_index: optional name→int mapping. When provided, the dist table is a
    numpy array (no Python arena allocations) and heap entries use integer node
    indices — eliminating the main source of arena fragmentation for large N.

    max_search_epochs: limit search to this many epochs ahead (correct for
    feasibility checks since paths beyond tau_k are infeasible).

    Returns math.inf if no path exists within the remaining epochs.
    """
    n_epochs = len(graphs)
    if max_search_epochs is not None:
        n_epochs = min(n_epochs, epoch + max_search_epochs)

    INF = math.inf

    if node_index is not None:
        # ── fast path: numpy dist array, integer node indices in heap ──────────
        src_idx = node_index.get(src, -1)
        dst_idx = node_index.get(dst, -1)
        if src_idx < 0 or dst_idx < 0:
            return INF

        n_ep = n_epochs - epoch
        n_nodes = len(node_index)
        dist = np.full((n_ep, n_nodes), INF)

        # idx→name reverse map for adjacency lookup (built once per call).
        idx2name: List[Optional[str]] = [None] * n_nodes
        for _nm, _ix in node_index.items():
            idx2name[_ix] = _nm

        # heap: (elapsed, ep_offset, node_idx, bits_remaining)
        heap = [(0.0, 0, src_idx, data_bits)]

        while heap:
            elapsed, ep_off, n_idx, bits_left = heapq.heappop(heap)
            ep = epoch + ep_off

            if n_idx == dst_idx and bits_left <= 0:
                return elapsed
            if ep_off >= n_ep:
                continue
            if dist[ep_off, n_idx] < elapsed:
                continue
            dist[ep_off, n_idx] = elapsed

            if ep >= n_epochs:
                continue

            g = graphs[ep]
            ep_elapsed_so_far = elapsed - (g.t_start - graphs[epoch].t_start)

            _name = idx2name[n_idx]
            for (nb, rate, cap) in g.adj.get(_name, ()):
                nb_idx = node_index.get(nb, -1)
                if nb_idx < 0:
                    continue

                ep_remaining = g.t_end - g.t_start - max(0.0, ep_elapsed_so_far)
                if ep_remaining <= 0:
                    continue
                bits_sent = min(bits_left, min(cap, ep_remaining * rate))
                new_bits_left = max(0.0, bits_left - bits_sent)
                time_used = bits_sent / rate if rate > 0 else 0.0
                new_elapsed = elapsed + time_used

                if new_bits_left <= 0:
                    if dist[ep_off, nb_idx] > new_elapsed:
                        dist[ep_off, nb_idx] = new_elapsed
                        heapq.heappush(heap, (new_elapsed, ep_off, nb_idx, 0.0))
                else:
                    next_ep_off = ep_off + 1
                    if next_ep_off < n_ep:
                        next_elapsed = new_elapsed + (graphs[ep + 1].t_start - g.t_end)
                        if dist[next_ep_off, nb_idx] > next_elapsed:
                            dist[next_ep_off, nb_idx] = next_elapsed
                            heapq.heappush(heap, (next_elapsed, next_ep_off, nb_idx, new_bits_left))

            next_ep_off = ep_off + 1
            if next_ep_off < n_ep:
                wait = graphs[ep + 1].t_start - g.t_start - max(0.0, ep_elapsed_so_far)
                new_elapsed = elapsed + max(0.0, wait)
                if dist[next_ep_off, n_idx] > new_elapsed:
                    dist[next_ep_off, n_idx] = new_elapsed
                    heapq.heappush(heap, (new_elapsed, next_ep_off, n_idx, bits_left))

        return INF

    # ── fallback: original dict-based Dijkstra ──────────────────────────────
    dist_d: Dict[Tuple[int, str], float] = {}
    heap_d = [(0.0, epoch, src, data_bits)]

    while heap_d:
        elapsed, ep, node, bits_left = heapq.heappop(heap_d)

        if node == dst and bits_left <= 0:
            return elapsed

        state = (ep, node)
        if dist_d.get(state, INF) < elapsed:
            continue
        dist_d[state] = elapsed

        if ep >= n_epochs:
            continue

        g = graphs[ep]
        ep_elapsed_so_far = elapsed - (g.t_start - graphs[epoch].t_start)

        for (nb, rate, cap) in g.adj.get(node, ()):
            ep_remaining = g.t_end - g.t_start - max(0.0, ep_elapsed_so_far)
            if ep_remaining <= 0:
                continue
            bits_sent = min(bits_left, min(cap, ep_remaining * rate))
            new_bits_left = max(0.0, bits_left - bits_sent)
            time_used = bits_sent / rate if rate > 0 else 0.0
            new_elapsed = elapsed + time_used

            if new_bits_left <= 0:
                next_state = (ep, nb)
                if dist_d.get(next_state, INF) > new_elapsed:
                    heapq.heappush(heap_d, (new_elapsed, ep, nb, 0.0))
            else:
                next_ep = ep + 1
                if next_ep < n_epochs:
                    next_elapsed = new_elapsed + (graphs[next_ep].t_start - g.t_end)
                    next_state = (next_ep, nb)
                    if dist_d.get(next_state, INF) > next_elapsed:
                        heapq.heappush(heap_d, (next_elapsed, next_ep, nb, new_bits_left))

        if ep + 1 < n_epochs:
            wait = graphs[ep + 1].t_start - g.t_start - max(0.0, ep_elapsed_so_far)
            next_state = (ep + 1, node)
            new_elapsed = elapsed + max(0.0, wait)
            if dist_d.get(next_state, INF) > new_elapsed:
                heapq.heappush(heap_d, (new_elapsed, ep + 1, node, bits_left))

    return INF


def earliest_arrival_all(
    src: str,
    epoch: int,
    graphs: List[EpochContactGraph],
    data_bits: float,
    node_index: Dict[str, int],
    max_search_epochs: Optional[int] = None,
    targets: Optional[Iterable[int]] = None,
) -> np.ndarray:
    """
    Single-source variant of earliest_arrival: one Dijkstra sweep returns the
    earliest-arrival time (seconds) from `src` to every node in `node_index`,
    as an array indexed by node_index values (math.inf where unreachable).

    Pop order is identical to earliest_arrival, and a node's arrival time is
    recorded at its first complete-delivery pop — exactly the state at which
    earliest_arrival would have returned for that destination — so
    arrival[node_index[v]] == earliest_arrival(src, v, ...) for every v, at
    the cost of one search instead of one per destination.

    targets: optional node indices; the search stops once all of them have an
    arrival time (other nodes may remain inf).
    """
    n_epochs = len(graphs)
    if max_search_epochs is not None:
        n_epochs = min(n_epochs, epoch + max_search_epochs)

    INF = math.inf
    n_nodes = len(node_index)
    arrival = np.full(n_nodes, INF)

    src_idx = node_index.get(src, -1)
    if src_idx < 0:
        return arrival

    pending = set(targets) if targets is not None else set(range(n_nodes))

    n_ep = n_epochs - epoch
    dist = np.full((n_ep, n_nodes), INF)

    idx2name: List[Optional[str]] = [None] * n_nodes
    for _nm, _ix in node_index.items():
        idx2name[_ix] = _nm

    # heap: (elapsed, ep_offset, node_idx, bits_remaining)
    heap = [(0.0, 0, src_idx, data_bits)]

    while heap:
        elapsed, ep_off, n_idx, bits_left = heapq.heappop(heap)
        ep = epoch + ep_off

        if bits_left <= 0 and math.isinf(arrival[n_idx]):
            arrival[n_idx] = elapsed
            pending.discard(n_idx)
            if not pending:
                return arrival
        if ep_off >= n_ep:
            continue
        if dist[ep_off, n_idx] < elapsed:
            continue
        dist[ep_off, n_idx] = elapsed

        if ep >= n_epochs:
            continue

        g = graphs[ep]
        ep_elapsed_so_far = elapsed - (g.t_start - graphs[epoch].t_start)

        _name = idx2name[n_idx]
        for (nb, rate, cap) in g.adj.get(_name, ()):
            nb_idx = node_index.get(nb, -1)
            if nb_idx < 0:
                continue

            ep_remaining = g.t_end - g.t_start - max(0.0, ep_elapsed_so_far)
            if ep_remaining <= 0:
                continue
            bits_sent = min(bits_left, min(cap, ep_remaining * rate))
            new_bits_left = max(0.0, bits_left - bits_sent)
            time_used = bits_sent / rate if rate > 0 else 0.0
            new_elapsed = elapsed + time_used

            if new_bits_left <= 0:
                if dist[ep_off, nb_idx] > new_elapsed:
                    dist[ep_off, nb_idx] = new_elapsed
                    heapq.heappush(heap, (new_elapsed, ep_off, nb_idx, 0.0))
            else:
                next_ep_off = ep_off + 1
                if next_ep_off < n_ep:
                    next_elapsed = new_elapsed + (graphs[ep + 1].t_start - g.t_end)
                    if dist[next_ep_off, nb_idx] > next_elapsed:
                        dist[next_ep_off, nb_idx] = next_elapsed
                        heapq.heappush(heap, (next_elapsed, next_ep_off, nb_idx, new_bits_left))

        next_ep_off = ep_off + 1
        if next_ep_off < n_ep:
            wait = graphs[ep + 1].t_start - g.t_start - max(0.0, ep_elapsed_so_far)
            new_elapsed = elapsed + max(0.0, wait)
            if dist[next_ep_off, n_idx] > new_elapsed:
                dist[next_ep_off, n_idx] = new_elapsed
                heapq.heappush(heap, (new_elapsed, next_ep_off, n_idx, bits_left))

    return arrival


def earliest_downlink(
    aggregator: str,
    epoch: int,
    graphs: List[EpochContactGraph],
    data_bits: float,
    ground_stations: set,
    max_search_epochs: Optional[int] = None,
    node_index: Optional[Dict[str, int]] = None,
) -> float:
    """Earliest transfer time from `aggregator` to any reachable ground station."""
    best = math.inf
    for gs in ground_stations:
        t = earliest_arrival(aggregator, gs, epoch, graphs, data_bits,
                             max_search_epochs=max_search_epochs,
                             node_index=node_index)
        if t < best:
            best = t
    return best
