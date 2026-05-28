"""
Feasibility checker: δ_kvia and p_kvia for each (task, tile, helper, aggregator) candidate.

δ_kvia = 1  iff  L_kvia(t) ≤ τ_k(t)  AND  A_i(t)=1  AND  A_a(t)=1
L_kvia = ℓ_ski + c_kv/C_i + ℓ_ia + ℓ_down_a
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from ordi.orbit.graph import EpochContactGraph, earliest_arrival, earliest_downlink
from ordi.sim.satellite import SatelliteState
from ordi.sim.reliability import ReliabilityModel
from ordi.tasks.generator import EOTask, Tile


@dataclass
class ReplicaCandidate:
    task_id: int
    tile_id: int
    helper: str
    aggregator: str
    epoch: int
    latency: float          # L_kvia (seconds)
    p_success: float        # p_kvia
    e_compute: float        # Joules on helper
    e_rx: float             # Joules receive on helper
    e_tx: float             # Joules transmit helper→aggregator
    feasible: bool          # δ_kvia
    d_in_bits: float
    d_out_bits: float


def compute_candidates(
    task: EOTask,
    tile: Tile,
    epoch: int,
    epoch_start: float,
    graphs: List[EpochContactGraph],
    states: Dict[str, SatelliteState],
    reliability: ReliabilityModel,
    ground_stations: set,
    tau_k: float,           # remaining deadline budget (seconds)
    sat_index: Optional[Dict[str, int]] = None,
    ell_down_cache: Optional[np.ndarray] = None,  # shape (n_active,)
    ell_ia_cache: Optional[np.ndarray] = None,    # shape (n_active, n_active)
    ell_ski_cache: Optional[np.ndarray] = None,   # shape (n_active,)
) -> List[ReplicaCandidate]:
    """
    Enumerate all feasible (helper, aggregator) pairs for tile (k, v) at epoch t.

    When sat_index + numpy arrays are provided (normal path from schedule_epoch),
    all routing lookups are O(1) array accesses with no Python dict overhead.
    Falls back to computing Dijkstra inline when caches are absent.
    """
    candidates = []
    sat_ids = [sid for sid, st in states.items() if st.A_i == 1]

    epoch_length = graphs[0].t_end - graphs[0].t_start if graphs else 60.0
    max_search_epochs = int(math.ceil(tau_k / epoch_length)) + 1

    # Fallback dict-based ell_down (only used when numpy cache absent)
    _ell_down_dict: Optional[Dict[str, float]] = None
    if ell_down_cache is None or sat_index is None:
        _ell_down_dict = {
            agg: earliest_downlink(
                agg, epoch, graphs, tile.d_out_bits, ground_stations,
                max_search_epochs=max_search_epochs,
            )
            for agg in sat_ids
        }

    for helper in sat_ids:
        if helper == task.source_sat:
            continue  # source is not a helper

        h_state = states[helper]
        if not h_state.A_i:
            continue

        h_idx = sat_index.get(helper) if sat_index is not None else None

        # ℓ_ski: source → helper transfer time for input tile
        if ell_ski_cache is not None and h_idx is not None:
            ell_ski = float(ell_ski_cache[h_idx])
        else:
            ell_ski = earliest_arrival(
                task.source_sat, helper, epoch, graphs, tile.d_in_bits,
                max_search_epochs=max_search_epochs,
            )
        if math.isinf(ell_ski):
            continue

        # compute time on helper
        t_compute = tile.compute_ops / max(h_state.C_i, 1.0)

        for aggregator in sat_ids:
            if aggregator == helper:
                continue

            a_state = states[aggregator]
            if not a_state.A_i:
                continue

            a_idx = sat_index.get(aggregator) if sat_index is not None else None

            # Check downlink feasibility before computing ell_ia (cheaper first)
            if ell_down_cache is not None and a_idx is not None:
                ell_down = float(ell_down_cache[a_idx])
            else:
                ell_down = _ell_down_dict.get(aggregator, math.inf)  # type: ignore[union-attr]
            if math.isinf(ell_down):
                continue

            # ℓ_ia: helper → aggregator transfer time for output
            if ell_ia_cache is not None and h_idx is not None and a_idx is not None:
                ell_ia = float(ell_ia_cache[h_idx, a_idx])
            else:
                ell_ia = earliest_arrival(
                    helper, aggregator, epoch, graphs, tile.d_out_bits,
                    max_search_epochs=max_search_epochs,
                )
            if math.isinf(ell_ia):
                continue

            L_kvia = ell_ski + t_compute + ell_ia + ell_down
            feasible = (L_kvia <= tau_k) and bool(h_state.A_i) and bool(a_state.A_i)

            p = reliability.replica_success_prob(
                helper_id=helper,
                source_id=task.source_sat,
                aggregator_id=aggregator,
                downlink_pi=reliability.default_downlink_pi,
            )

            e_comp = h_state.energy_for_compute(tile.compute_ops)
            e_rx   = h_state.energy_for_rx(tile.d_in_bits)
            e_tx   = h_state.energy_for_tx(tile.d_out_bits)

            candidates.append(ReplicaCandidate(
                task_id=task.task_id,
                tile_id=tile.tile_id,
                helper=helper,
                aggregator=aggregator,
                epoch=epoch,
                latency=L_kvia,
                p_success=p,
                e_compute=e_comp,
                e_rx=e_rx,
                e_tx=e_tx,
                feasible=feasible,
                d_in_bits=tile.d_in_bits,
                d_out_bits=tile.d_out_bits,
            ))

    return [c for c in candidates if c.feasible]
