"""
Epoch-level routing-table cache shared by ORDI and the multi-satellite
baselines (B5-B8).

Any scheduler with `cfg`, `graphs`, `states`, and `ground_stations` attributes
can mix this in and call _build_epoch_caches once per epoch, then feed the
resulting numpy tables to compute_candidates instead of letting it fall back
to one Dijkstra per (source, destination) pair.
"""

from __future__ import annotations

import math
from typing import Dict, List, Set, Tuple

import numpy as np

from ordi.orbit.graph import earliest_arrival_all
from ordi.tasks.generator import EOTask

try:
    from ordi.orbit._dijkstra_numba import (
        build_numeric_graphs, earliest_arrival_all_numba,
    )
    _HAVE_NUMBA = True
except ImportError:
    _HAVE_NUMBA = False


class EpochRoutingCacheMixin:
    # CSR graph arrays for the numba Dijkstra kernel, built lazily on the
    # first _build_epoch_caches call (node_index is identical every epoch).
    _numeric_graphs = None

    def _ea_all(self, src, epoch, data_bits, node_index, max_search_ep, targets):
        """Single-source earliest-arrival sweep; numba kernel when available."""
        if _HAVE_NUMBA:
            if self._numeric_graphs is None:
                self._numeric_graphs = build_numeric_graphs(self.graphs, node_index)
            return earliest_arrival_all_numba(
                src, epoch, self.graphs, data_bits, node_index,
                self._numeric_graphs,
                max_search_epochs=max_search_ep, targets=targets,
            )
        return earliest_arrival_all(
            src, epoch, self.graphs, data_bits, node_index,
            max_search_epochs=max_search_ep, targets=targets,
        )

    def _build_epoch_caches(
        self,
        epoch: int,
        epoch_start: float,
        pending_tasks: List[EOTask],
    ) -> Tuple[
        Dict[str, int],                     # sat_index: name → array row/col
        Dict[float, np.ndarray],            # ell_down_caches[d_out] shape (n,)
        Dict[float, np.ndarray],            # ell_ia_caches[d_out]   shape (n, n)
        Dict[float, Dict[str, np.ndarray]], # ell_ski_caches[d_in][src] shape (n,)
    ]:
        """
        Precompute three routing tables for this epoch as numpy arrays.

        Storing results in np.ndarray (managed by numpy's own allocator, not
        Python's arena system) prevents arena fragmentation — the fix for the
        tens-of-GB RSS growth seen with pure-Python dict caches.

        Reduction for n=60: ~62M Dijkstra calls per run → ~3M (20× fewer).
        """
        cfg = self.cfg
        sat_ids_active = [sid for sid, st in self.states.items() if st.A_i]
        n = len(sat_ids_active)
        sat_index: Dict[str, int] = {sid: i for i, sid in enumerate(sat_ids_active)}

        # node_index covers ALL sats (active or not) + ground stations.
        # Inactive satellites can still act as ISL relay nodes; A_i only gates
        # whether a satellite is eligible as a helper/aggregator (checked in
        # compute_candidates). Using all nodes here matches baseline routing
        # and prevents route-blocking when a satellite temporarily has A_i=0.
        all_sat_ids = list(self.states.keys())
        n_all = len(all_sat_ids)
        node_index: Dict[str, int] = {
            **{sid: i for i, sid in enumerate(all_sat_ids)},
            **{gs: n_all + i for i, gs in enumerate(sorted(self.ground_stations))},
        }

        unique_d_out: Set[float] = set()
        unique_d_in:  Set[float] = set()
        unique_sources: Set[str] = set()
        max_tau_k = cfg.epoch_length

        for task in pending_tasks:
            tau_k = task.deadline - epoch_start
            if tau_k <= 0:
                continue
            max_tau_k = max(max_tau_k, tau_k)
            unique_sources.add(task.source_sat)
            for tile in task.tiles:
                unique_d_out.add(tile.d_out_bits)
                unique_d_in.add(tile.d_in_bits)

        max_search_ep = int(math.ceil(max_tau_k / cfg.epoch_length)) + 1

        # All three tables are filled from single-source Dijkstra sweeps
        # (earliest_arrival_all): one search per source node instead of one per
        # (source, destination) pair — same arrival times, ~n× fewer searches.
        gs_idx = [node_index[gs] for gs in self.ground_stations]
        active_idx = [node_index[sid] for sid in sat_ids_active]

        # ell_down_caches[d_out] → (n,) array: ell_down_caches[d_out][agg_idx]
        ell_down_caches: Dict[float, np.ndarray] = {}
        for d_out in unique_d_out:
            arr = np.full(n, np.inf)
            for i, agg in enumerate(sat_ids_active):
                t_all = self._ea_all(agg, epoch, d_out, node_index,
                                     max_search_ep, gs_idx)
                arr[i] = t_all[gs_idx].min()
            ell_down_caches[d_out] = arr

        # ell_ia_caches[d_out] → (n, n) array: ell_ia_caches[d_out][h_idx, a_idx]
        # Only fill cells where agg can downlink (inf ell_down → skip).
        ell_ia_caches: Dict[float, np.ndarray] = {}
        for d_out in unique_d_out:
            dc = ell_down_caches[d_out]
            arr = np.full((n, n), np.inf)
            reachable = [i for i in range(n) if not np.isinf(dc[i])]
            for hi, helper in enumerate(sat_ids_active):
                targets = [active_idx[ai] for ai in reachable if ai != hi]
                if not targets:
                    continue
                t_all = self._ea_all(helper, epoch, d_out, node_index,
                                     max_search_ep, targets)
                for ai in reachable:
                    if ai == hi:
                        continue
                    arr[hi, ai] = t_all[active_idx[ai]]
            ell_ia_caches[d_out] = arr

        # ell_ski_caches[d_in][source] → (n,) array: arr[h_idx]
        ell_ski_caches: Dict[float, Dict[str, np.ndarray]] = {}
        for d_in in unique_d_in:
            per_src: Dict[str, np.ndarray] = {}
            for source in unique_sources:
                targets = [active_idx[hi] for hi, h in enumerate(sat_ids_active)
                           if h != source]
                t_all = self._ea_all(source, epoch, d_in, node_index,
                                     max_search_ep, targets)
                arr = np.full(n, np.inf)
                for hi, helper in enumerate(sat_ids_active):
                    if helper == source:
                        continue
                    arr[hi] = t_all[active_idx[hi]]
                per_src[source] = arr
            ell_ski_caches[d_in] = per_src

        return sat_index, ell_down_caches, ell_ia_caches, ell_ski_caches
