"""
Exact ILP reference solver for small instances (≤5 tasks, ≤20 satellites).

Formulates a fully-linearized MILP from the proposal using PuLP + HiGHS.

Decision variables
------------------
  x[k,v,i,a]  ∈ {0,1}   select replica of tile (k,v) on helper i, agg a
  yP[k,a]     ∈ {0,1}   a is primary aggregator for task k
  yB[k,a]     ∈ {0,1}   a is backup  aggregator for task k
  s[k,v]      ≥ 0       replication-penalty aux: linearises [n_kv - 1]+

Objective (fully linear)
------------------------
  max  Σ_{k,v,i,a}  [u_kv · p_kvia · exp(-α · L_kvia)
                      - λ_E · (e_comp + e_rx + e_tx)
                      - λ_C · ρ_mn · q_mn] · x_{kvia}
       - λ_R · Σ_{k,v} s_{kv}

  where ρ_mn = 1 / Q_mn  (congestion price per proposal §4)
  and   q_mn aggregates d_in on the source→helper link
               and     d_out on the helper→aggregator link.

Constraints
-----------
  (C1) x_kvia ≤ yP_{k,a} + yB_{k,a}                  aggregator consistency
  (C2) Σ_a yP_{k,a} ≤ 1                               at most one primary / task
  (C3) Σ_a yB_{k,a} ≤ 1                               at most one backup  / task
  (C4) yP_{k,a} + yB_{k,a} ≤ 1                        primary ≠ backup
  (C5) Σ_{i,a} x_{k,v,i,a} ≤ r^max_{kv}              replica cap
  (C6) s_{kv} ≥ Σ_{i,a} x_{k,v,i,a} - 1              linearise [n-1]+
  (C7) Σ_{k,v,i,a: i=h} (e_comp+e_rx+e_tx) x ≤ B_h   energy budget per helper
  (C8) Σ_{k,v,i,a: uses (m,n)} bits · x ≤ Q_{mn}     link capacity

Used for:
  - validating greedy optimality gap (E8)
  - ablation experiments
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple

try:
    import pulp
    _PULP_AVAILABLE = True
except ImportError:
    _PULP_AVAILABLE = False

from ordi.orbit.graph import EpochContactGraph, earliest_arrival, earliest_downlink
from ordi.sim.satellite import SatelliteState
from ordi.sim.reliability import ReliabilityModel
from ordi.tasks.generator import EOTask, Tile
from ordi.scheduler.feasibility import compute_candidates
from ordi.scheduler.ordi import ORDIConfig, TileAssignment, SchedulerResult


def _pick_solver(time_limit_s: float, threads: int = 8) -> "pulp.LpSolver":
    """Return HiGHS if available, else CBC."""
    try:
        solver = pulp.HiGHS(msg=False, timeLimit=time_limit_s, threads=threads)
        if not solver.available():
            raise RuntimeError("HiGHS not available")
        return solver
    except Exception:
        return pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit_s, threads=threads)


def solve_ilp(
    epoch: int,
    t_sim_start: float,
    pending_tasks: List[EOTask],
    graphs: List[EpochContactGraph],
    states: Dict[str, SatelliteState],
    reliability: ReliabilityModel,
    ground_stations: set,
    cfg: ORDIConfig,
    time_limit_s: float = 60.0,
    threads: int = 8,
) -> Optional[SchedulerResult]:
    """
    Solve the MILP exactly for one epoch.
    Returns None if PuLP is unavailable or no feasible solution is found.
    """
    if not _PULP_AVAILABLE:
        return None

    sat_ids = list(states.keys())
    epoch_start = t_sim_start + epoch * cfg.epoch_length
    g = graphs[epoch]

    # ── enumerate all feasible (k, v, i, a) candidates ────────────────────────
    all_candidates: Dict[Tuple[int, int, str, str], object] = {}
    task_tile_pairs: List[Tuple[EOTask, Tile]] = []
    task_map: Dict[int, EOTask] = {}

    for task in pending_tasks:
        tau_k = task.deadline - epoch_start
        if tau_k <= 0:
            continue
        task_map[task.task_id] = task
        for tile in task.tiles:
            cands = compute_candidates(
                task, tile, epoch, epoch_start,
                graphs, states, reliability, ground_stations, tau_k,
            )
            for c in cands:
                key = (task.task_id, tile.tile_id, c.helper, c.aggregator)
                all_candidates[key] = c
            task_tile_pairs.append((task, tile))

    if not all_candidates:
        return SchedulerResult(
            epoch=epoch, assignments=[], total_utility=0.0,
            energy_penalty=0.0, comm_penalty=0.0, rep_penalty=0.0,
            objective=0.0, link_utilization={},
        )

    # ── derive unique keys ─────────────────────────────────────────────────────
    tile_map: Dict[Tuple[int, int], Tile] = {
        (task.task_id, tile.tile_id): tile
        for task, tile in task_tile_pairs
    }
    task_ids = {k for (k, v, i, a) in all_candidates}
    tile_ids = {(k, v) for (k, v, i, a) in all_candidates}
    agg_per_task: Dict[int, set] = {k: set() for k in task_ids}
    for (k, v, i, a) in all_candidates:
        agg_per_task[k].add(a)

    # ── build link → capacity map for constraint (C8) ─────────────────────────
    link_cap: Dict[Tuple[str, str], float] = {}
    for (na, nb, _, cap, _) in g.edges:
        link_cap[(na, nb)] = link_cap.get((na, nb), 0.0) + cap

    # ── problem ───────────────────────────────────────────────────────────────
    prob = pulp.LpProblem("ORDI_epoch", pulp.LpMaximize)

    # Decision variables
    x = {key: pulp.LpVariable(f"x_{key[0]}_{key[1]}_{key[2]}_{key[3]}", cat="Binary")
         for key in all_candidates}

    # yP[(k,a)], yB[(k,a)] — one per unique (task, aggregator) pair
    yP: Dict[Tuple[int, str], pulp.LpVariable] = {}
    yB: Dict[Tuple[int, str], pulp.LpVariable] = {}
    for k, aggs in agg_per_task.items():
        for a in aggs:
            yP[(k, a)] = pulp.LpVariable(f"yP_{k}_{a}", cat="Binary")
            yB[(k, a)] = pulp.LpVariable(f"yB_{k}_{a}", cat="Binary")

    # s[(k,v)] — replication-penalty aux (linearises [n_kv - 1]+)
    s: Dict[Tuple[int, int], pulp.LpVariable] = {
        kv: pulp.LpVariable(f"s_{kv[0]}_{kv[1]}", lowBound=0.0)
        for kv in tile_ids
    }

    # ── objective (fully linear) ───────────────────────────────────────────────
    obj_terms = []
    for (k, v, i, a), c in all_candidates.items():
        tile = tile_map[(k, v)]

        # Freshness-weighted utility per replica
        u_term = tile.utility * c.p_success * math.exp(-cfg.alpha * c.latency)

        # Energy penalty
        e_term = cfg.lambda_E * (c.e_compute + c.e_rx + c.e_tx)

        # Communication penalty  (ρ_mn = 1/Q_mn, q_mn = bits on that link)
        src = task_map[k].source_sat
        q_in  = c.d_in_bits
        q_out = c.d_out_bits
        rho_in  = 1.0 / max(link_cap.get((src, i), 1.0), 1.0)
        rho_out = 1.0 / max(link_cap.get((i, a),   1.0), 1.0)
        c_term = cfg.lambda_C * (rho_in * q_in + rho_out * q_out)

        obj_terms.append((u_term - e_term - c_term) * x[(k, v, i, a)])

    prob += (
        pulp.lpSum(obj_terms)
        - cfg.lambda_R * pulp.lpSum(s.values())
    )

    # ── constraints ───────────────────────────────────────────────────────────

    # (C1) aggregator consistency
    for (k, v, i, a) in all_candidates:
        prob += x[(k, v, i, a)] <= yP[(k, a)] + yB[(k, a)]

    # (C2) at most one primary aggregator per task
    for k, aggs in agg_per_task.items():
        prob += pulp.lpSum(yP[(k, a)] for a in aggs) <= 1

    # (C3) at most one backup aggregator per task
    for k, aggs in agg_per_task.items():
        prob += pulp.lpSum(yB[(k, a)] for a in aggs) <= 1

    # (C4) primary ≠ backup for same (task, aggregator)
    for (k, a) in yP:
        if (k, a) in yB:
            prob += yP[(k, a)] + yB[(k, a)] <= 1

    # (C5) replica cap per tile
    for (k, v) in tile_ids:
        tile = tile_map[(k, v)]
        prob += (
            pulp.lpSum(
                x[(kk, vv, i, a)]
                for (kk, vv, i, a) in all_candidates
                if kk == k and vv == v
            ) <= tile.n_replicas_max
        )

    # (C6) replication-penalty aux: s_kv ≥ n_kv - 1  (s_kv ≥ 0 from lower bound)
    for (k, v) in tile_ids:
        n_kv = pulp.lpSum(
            x[(kk, vv, i, a)]
            for (kk, vv, i, a) in all_candidates
            if kk == k and vv == v
        )
        prob += s[(k, v)] >= n_kv - 1

    # (C7) energy budget per helper satellite
    for helper in sat_ids:
        h_state = states[helper]
        energy_vars = [
            (c.e_compute + c.e_rx + c.e_tx) * x[key]
            for key, c in all_candidates.items()
            if c.helper == helper
        ]
        if energy_vars:
            prob += (
                pulp.lpSum(energy_vars)
                <= h_state.B_i - h_state.params.battery_min_j
            )

    # (C8) link capacity: each ISL cannot be overloaded across all replicas
    # Collect which candidates use each link
    link_usage: Dict[Tuple[str, str], list] = {}
    for (k, v, i, a), c in all_candidates.items():
        src = task_map[k].source_sat
        for link, bits in [((src, i), c.d_in_bits), ((i, a), c.d_out_bits)]:
            link_usage.setdefault(link, []).append(bits * x[(k, v, i, a)])

    for link, terms in link_usage.items():
        cap = link_cap.get(link, 0.0)
        if cap > 0:
            prob += pulp.lpSum(terms) <= cap

    # ── solve ─────────────────────────────────────────────────────────────────
    solver = _pick_solver(time_limit_s, threads=threads)
    try:
        status = prob.solve(solver)
    except Exception:
        # Solver binary unavailable at solve time — retry with CBC
        solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit_s, threads=1)
        try:
            status = prob.solve(solver)
        except Exception:
            return None

    lp_status = pulp.LpStatus[status]
    if lp_status == "Infeasible":
        return None
    # Accept Optimal or (time-limited) best-feasible ("Not Solved" in CBC,
    # "Solution Limit" / similar in HiGHS)

    # ── extract solution ───────────────────────────────────────────────────────
    assignments = []
    total_utility = 0.0
    energy_penalty = 0.0
    comm_penalty = 0.0
    link_utilization: Dict[Tuple[str, str], float] = {}

    for (k, v) in tile_ids:
        tile = tile_map[(k, v)]
        a_obj = TileAssignment(task_id=k, tile_id=v)
        replica_probs = []
        best_latency = math.inf

        for (kk, vv, i, a), c in all_candidates.items():
            if kk != k or vv != v:
                continue
            if (pulp.value(x[(kk, vv, i, a)]) or 0.0) < 0.5:
                continue

            a_obj.replicas.append(c)
            replica_probs.append(c.p_success)
            best_latency = min(best_latency, c.latency)
            energy_penalty += cfg.lambda_E * (c.e_compute + c.e_rx + c.e_tx)

            src = task_map[k].source_sat
            for link, bits in [((src, i), c.d_in_bits), ((i, a), c.d_out_bits)]:
                link_utilization[link] = link_utilization.get(link, 0.0) + bits
                cap = link_cap.get(link, 1.0)
                comm_penalty += cfg.lambda_C * bits / max(cap, 1.0)

            if (pulp.value(yP.get((k, a), 0)) or 0.0) > 0.5:
                a_obj.primary_aggregator = a
            elif (pulp.value(yB.get((k, a), 0)) or 0.0) > 0.5:
                a_obj.backup_aggregator = a

        # Source survival factored at tile level (see tile_delivery_prob).
        a_obj.z_kv = reliability.tile_delivery_prob(
            replica_probs, reliability.node_pi(task_map[k].source_sat))
        a_obj.L_hat = best_latency if not math.isinf(best_latency) else 0.0

        if a_obj.z_kv > 0:
            total_utility += tile.utility * a_obj.z_kv * math.exp(
                -cfg.alpha * a_obj.L_hat
            )

        assignments.append(a_obj)

    rep_penalty = cfg.lambda_R * sum(
        max(0, len(a.replicas) - 1) for a in assignments
    )
    objective = total_utility - energy_penalty - comm_penalty - rep_penalty

    return SchedulerResult(
        epoch=epoch,
        assignments=assignments,
        total_utility=total_utility,
        energy_penalty=energy_penalty,
        comm_penalty=comm_penalty,
        rep_penalty=rep_penalty,
        objective=objective,
        link_utilization=link_utilization,
    )
