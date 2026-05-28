"""
Feasibility checker: δ_kvia and p_kvia for each (task, tile, helper, aggregator) candidate.

δ_kvia = 1  iff  L_kvia(t) ≤ τ_k(t)  AND  A_i(t)=1  AND  A_a(t)=1
L_kvia = ℓ_ski + c_kv/C_i + ℓ_ia + ℓ_down_a
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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
    ell_down_cache: Optional[Dict[str, float]] = None,
    ell_ia_cache: Optional[Dict[Tuple[str, str], float]] = None,
    ell_ski_cache: Optional[Dict[str, float]] = None,
) -> List[ReplicaCandidate]:
    """
    Enumerate all feasible (helper, aggregator) pairs for tile (k, v) at epoch t.

    Callers should pass epoch-level precomputed caches (ell_down_cache,
    ell_ia_cache, ell_ski_cache) to avoid redundant Dijkstra calls across tiles.
    If caches are absent they are computed locally (correct but slow for large N).
    """
    candidates = []
    sat_ids = [sid for sid, st in states.items() if st.A_i == 1]

    epoch_length = graphs[0].t_end - graphs[0].t_start if graphs else 60.0
    max_search_epochs = int(math.ceil(tau_k / epoch_length)) + 1

    if ell_down_cache is None:
        ell_down_cache = {
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

        # ℓ_ski: source → helper transfer time for input tile
        if ell_ski_cache is not None:
            ell_ski = ell_ski_cache.get(helper, math.inf)
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

            # Check downlink feasibility before computing ell_ia (cheaper first)
            ell_down = ell_down_cache.get(aggregator, math.inf)
            if math.isinf(ell_down):
                continue

            # ℓ_ia: helper → aggregator transfer time for output
            if ell_ia_cache is not None:
                ell_ia = ell_ia_cache.get((helper, aggregator), math.inf)
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
