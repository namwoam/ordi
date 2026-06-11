"""Numba-accelerated backend for earliest_arrival_all.

The epoch graphs are flattened once per simulation into CSR-style numpy
arrays (build_numeric_graphs); the Dijkstra sweep then runs as an @njit
kernel with a manual binary heap (~27x faster than the pure-Python sweep).

The kernel mirrors the arithmetic and visit order of
ordi.orbit.graph.earliest_arrival_all, so the arrival times of the queried
`targets` are bit-for-bit identical. Non-target entries may legally differ:
the kernel's heap breaks elapsed-time ties arbitrarily while heapq compares
full tuples, which can shift the early-termination point — target values are
tie-order independent (first qualifying pop is minimal either way), but which
incidental non-target nodes got recorded before termination is not.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from numba import njit

from ordi.orbit.graph import EpochContactGraph

NumericGraphs = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                      np.ndarray, np.ndarray]


def build_numeric_graphs(
    graphs: List[EpochContactGraph],
    node_index: Dict[str, int],
) -> NumericGraphs:
    """Flatten epoch adjacency lists into CSR arrays for the kernel.

    indptr[ep, i] .. indptr[ep, i+1] slices the edge arrays for node i's
    out-edges in epoch ep. Edges to nodes outside node_index are dropped
    (parity with the nb_idx < 0 skip in the Python sweep).
    """
    n_nodes = len(node_index)
    n_epochs = len(graphs)
    idx2name: List[Optional[str]] = [None] * n_nodes
    for nm, ix in node_index.items():
        idx2name[ix] = nm

    indptr = np.zeros((n_epochs, n_nodes + 1), dtype=np.int64)
    nb_l: List[int] = []
    rate_l: List[float] = []
    cap_l: List[float] = []
    pos = 0
    for ep, g in enumerate(graphs):
        for i in range(n_nodes):
            indptr[ep, i] = pos
            for (nb, rate, cap) in g.adj.get(idx2name[i], ()):
                nb_idx = node_index.get(nb, -1)
                if nb_idx < 0:
                    continue
                nb_l.append(nb_idx)
                rate_l.append(rate)
                cap_l.append(cap)
                pos += 1
        indptr[ep, n_nodes] = pos
    return (indptr,
            np.array(nb_l, dtype=np.int64),
            np.array(rate_l), np.array(cap_l),
            np.array([g.t_start for g in graphs]),
            np.array([g.t_end for g in graphs]))


@njit(cache=True)
def _heap_push(h_el, h_ep, h_nd, h_bt, size, el, ep, nd, bt):
    """Push onto the parallel-array binary min-heap, growing 2x when full."""
    if size == h_el.shape[0]:
        nc = size * 2
        t1 = np.empty(nc); t1[:size] = h_el[:size]; h_el = t1
        t2 = np.empty(nc, np.int64); t2[:size] = h_ep[:size]; h_ep = t2
        t3 = np.empty(nc, np.int64); t3[:size] = h_nd[:size]; h_nd = t3
        t4 = np.empty(nc); t4[:size] = h_bt[:size]; h_bt = t4
    j = size
    h_el[j] = el; h_ep[j] = ep; h_nd[j] = nd; h_bt[j] = bt
    size += 1
    while j > 0:
        par = (j - 1) // 2
        if h_el[par] <= h_el[j]:
            break
        h_el[par], h_el[j] = h_el[j], h_el[par]
        h_ep[par], h_ep[j] = h_ep[j], h_ep[par]
        h_nd[par], h_nd[j] = h_nd[j], h_nd[par]
        h_bt[par], h_bt[j] = h_bt[j], h_bt[par]
        j = par
    return h_el, h_ep, h_nd, h_bt, size


@njit(cache=True)
def _ea_all_kernel(src_idx, epoch, n_epochs_lim, data_bits,
                   indptr, edge_nb, edge_rate, edge_cap,
                   t_start, t_end, targets_mask, n_targets):
    n_nodes = indptr.shape[1] - 1
    n_ep = n_epochs_lim - epoch
    INF = np.inf
    arrival = np.full(n_nodes, INF)
    if n_ep <= 0:
        return arrival
    dist = np.full((n_ep, n_nodes), INF)

    # parallel-array binary min-heap keyed on elapsed
    h_el = np.empty(4096)
    h_ep = np.empty(4096, np.int64)
    h_nd = np.empty(4096, np.int64)
    h_bt = np.empty(4096)
    h_el[0] = 0.0; h_ep[0] = 0; h_nd[0] = src_idx; h_bt[0] = data_bits
    size = 1
    n_found = 0
    t_epoch0 = t_start[epoch]

    while size > 0:
        # pop min
        elapsed = h_el[0]; ep_off = h_ep[0]; n_idx = h_nd[0]; bits_left = h_bt[0]
        size -= 1
        h_el[0] = h_el[size]; h_ep[0] = h_ep[size]
        h_nd[0] = h_nd[size]; h_bt[0] = h_bt[size]
        i = 0
        while True:
            l = 2 * i + 1
            r = l + 1
            sm = i
            if l < size and h_el[l] < h_el[sm]:
                sm = l
            if r < size and h_el[r] < h_el[sm]:
                sm = r
            if sm == i:
                break
            h_el[i], h_el[sm] = h_el[sm], h_el[i]
            h_ep[i], h_ep[sm] = h_ep[sm], h_ep[i]
            h_nd[i], h_nd[sm] = h_nd[sm], h_nd[i]
            h_bt[i], h_bt[sm] = h_bt[sm], h_bt[i]
            i = sm

        if bits_left <= 0 and np.isinf(arrival[n_idx]):
            arrival[n_idx] = elapsed
            if targets_mask[n_idx]:
                n_found += 1
                if n_found == n_targets:
                    return arrival
        if ep_off >= n_ep:
            continue
        if dist[ep_off, n_idx] < elapsed:
            continue
        dist[ep_off, n_idx] = elapsed

        ep = epoch + ep_off
        g_start = t_start[ep]
        g_end = t_end[ep]
        ep_elapsed = elapsed - (g_start - t_epoch0)
        ep_remaining = g_end - g_start - max(0.0, ep_elapsed)

        for e in range(indptr[ep, n_idx], indptr[ep, n_idx + 1]):
            if ep_remaining <= 0:
                continue
            nb = edge_nb[e]; rate = edge_rate[e]; cap = edge_cap[e]
            bits_sent = min(bits_left, min(cap, ep_remaining * rate))
            new_bits_left = max(0.0, bits_left - bits_sent)
            time_used = bits_sent / rate if rate > 0 else 0.0
            new_elapsed = elapsed + time_used

            if new_bits_left <= 0:
                if dist[ep_off, nb] > new_elapsed:
                    dist[ep_off, nb] = new_elapsed
                    h_el, h_ep, h_nd, h_bt, size = _heap_push(
                        h_el, h_ep, h_nd, h_bt, size,
                        new_elapsed, ep_off, nb, 0.0)
            else:
                if ep_off + 1 < n_ep:
                    next_elapsed = new_elapsed + (t_start[ep + 1] - g_end)
                    if dist[ep_off + 1, nb] > next_elapsed:
                        dist[ep_off + 1, nb] = next_elapsed
                        h_el, h_ep, h_nd, h_bt, size = _heap_push(
                            h_el, h_ep, h_nd, h_bt, size,
                            next_elapsed, ep_off + 1, nb, new_bits_left)

        if ep_off + 1 < n_ep:
            wait = t_start[ep + 1] - g_start - max(0.0, ep_elapsed)
            new_elapsed = elapsed + max(0.0, wait)
            if dist[ep_off + 1, n_idx] > new_elapsed:
                dist[ep_off + 1, n_idx] = new_elapsed
                h_el, h_ep, h_nd, h_bt, size = _heap_push(
                    h_el, h_ep, h_nd, h_bt, size,
                    new_elapsed, ep_off + 1, n_idx, bits_left)

    return arrival


def earliest_arrival_all_numba(
    src: str,
    epoch: int,
    graphs: List[EpochContactGraph],
    data_bits: float,
    node_index: Dict[str, int],
    numeric: NumericGraphs,
    max_search_epochs: Optional[int] = None,
    targets: Optional[Iterable[int]] = None,
) -> np.ndarray:
    """Drop-in equivalent of graph.earliest_arrival_all for the entries in
    `targets`, backed by the @njit kernel.

    numeric: result of build_numeric_graphs(graphs, node_index).
    """
    n_epochs = len(graphs)
    if max_search_epochs is not None:
        n_epochs = min(n_epochs, epoch + max_search_epochs)
    n_nodes = len(node_index)
    src_idx = node_index.get(src, -1)
    if src_idx < 0:
        return np.full(n_nodes, np.inf)
    mask = np.zeros(n_nodes, dtype=np.bool_)
    if targets is None:
        mask[:] = True
    else:
        for t in targets:
            mask[t] = True
    n_targets = int(mask.sum())
    if n_targets == 0:
        return np.full(n_nodes, np.inf)
    indptr, edge_nb, edge_rate, edge_cap, t_start, t_end = numeric
    return _ea_all_kernel(src_idx, epoch, n_epochs, float(data_bits),
                          indptr, edge_nb, edge_rate, edge_cap,
                          t_start, t_end, mask, n_targets)
