"""
Experiment runner for E1–E8.

Each experiment returns a dict: {algorithm_name: List[EpochMetrics]}.
Results are saved to results/<experiment_id>.csv.

Simulation setup (shared):
  - Synthetic Walker constellation: 6 planes × 6 sats = 36 sats (default)
  - 10 ground stations
  - Simulation horizon: 2 orbits = 10,800 s
  - Epoch length: 60 s → 180 epochs
  - Tasks: Poisson arrivals, 3 tasks/orbit, deadline 300 s
"""

from __future__ import annotations
import csv
import math
import os
import time
from copy import deepcopy
from typing import Dict, List, Optional

from ordi.orbit.contacts import build_synthetic_walker, compute_contact_windows, DEFAULT_GROUND_STATIONS
from ordi.orbit.graph import build_epoch_graphs
from ordi.sim.satellite import make_constellation_states
from ordi.sim.reliability import ReliabilityModel
from ordi.tasks.generator import generate_tasks
from ordi.scheduler.ordi import ORDIScheduler, ORDIConfig
from ordi.baselines.baselines import build_all_baselines
from ordi.faults.injector import FaultInjector, random_fault_schedule, FaultEvent
from ordi.eval.metrics import compute_metrics, aggregate_metrics, EpochMetrics

RESULTS_DIR = "results"
SIM_DURATION_S = 10_800       # 2 orbits
EPOCH_LENGTH_S = 60.0
N_EPOCHS = int(SIM_DURATION_S / EPOCH_LENGTH_S)
T_SIM_START = 0.0


# ── simulation bootstrap ──────────────────────────────────────────────────────

def _build_sim(n_planes=6, sats_per_plane=6, seed=0, deadline_slack=300.0,
               arrival_rate=3.0):
    """Build all shared simulation objects."""
    sats = build_synthetic_walker(n_planes=n_planes, sats_per_plane=sats_per_plane)
    sat_ids = [s.name for s in sats]
    gs_names = {gs[0] for gs in DEFAULT_GROUND_STATIONS}

    print(f"  Computing contact windows for {len(sats)} sats × {len(DEFAULT_GROUND_STATIONS)} GS ...")
    t0 = time.time()
    contacts = compute_contact_windows(
        sats,
        t_start_unix=0.0,
        t_end_unix=SIM_DURATION_S,
        dt_seconds=60.0,   # coarser sampling for speed
    )
    print(f"  {len(contacts)} contact events in {time.time()-t0:.1f}s")

    graphs = build_epoch_graphs(contacts, T_SIM_START, EPOCH_LENGTH_S, N_EPOCHS)
    states = make_constellation_states(sat_ids, seed=seed)
    reliability = ReliabilityModel()

    tasks = generate_tasks(
        sat_ids, SIM_DURATION_S,
        arrival_rate_per_orbit=arrival_rate,
        deadline_slack_s=deadline_slack,
        seed=seed,
    )
    cfg = ORDIConfig(epoch_length=EPOCH_LENGTH_S)

    return sats, sat_ids, gs_names, contacts, graphs, states, reliability, tasks, cfg


def _run_algorithm(name, scheduler_or_baseline, tasks, graphs, states, reliability,
                   sat_ids, gs_names, cfg, faults: Optional[List[FaultEvent]] = None,
                   seed=0) -> List[EpochMetrics]:
    """
    Run one algorithm for all N_EPOCHS. Returns per-epoch metrics.
    States are deep-copied so each algorithm run is independent.
    """
    local_states = deepcopy(states)
    local_rel = deepcopy(reliability)

    injector = None
    if faults:
        from ordi.orbit.contacts import DEFAULT_GROUND_STATIONS
        injector = FaultInjector(local_states, local_rel, [], rng_seed=seed)
        for f in faults:
            injector.schedule(f)

    # Rebuild scheduler with local copies
    if name == "ORDI":
        sched = ORDIScheduler(cfg, sat_ids, gs_names, graphs, local_states, local_rel)
    else:
        sched = scheduler_or_baseline.__class__(
            graphs, local_states, gs_names, local_rel, cfg
        )

    sat_cap = {s: local_states[s].C_i * EPOCH_LENGTH_S for s in sat_ids}

    epoch_metrics = []
    for epoch in range(N_EPOCHS):
        ep_start = T_SIM_START + epoch * EPOCH_LENGTH_S

        if injector:
            injector.apply_epoch(epoch)

        pending = [t for t in tasks if t.release_time <= ep_start < t.deadline]

        if name == "ORDI":
            result = sched.schedule_epoch(epoch, T_SIM_START, pending)
        else:
            result = sched.schedule(epoch, T_SIM_START, pending)

        m = compute_metrics(result, pending, ep_start, sat_cap, cfg.alpha)
        epoch_metrics.append(m)

        if injector:
            injector.withdraw_epoch(epoch + 1)

    return epoch_metrics


def _save_csv(exp_id: str, results: Dict[str, List[EpochMetrics]]):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{exp_id}.csv")
    rows = []
    for alg_name, metrics in results.items():
        agg = aggregate_metrics(metrics)
        row = {"algorithm": alg_name, **agg}
        rows.append(row)
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {path}")


# ── E1: Core performance (ORDI vs all baselines, no faults) ──────────────────

def run_E1(seed=0) -> Dict[str, List[EpochMetrics]]:
    print("E1: Core performance comparison (no faults)")
    sats, sat_ids, gs_names, contacts, graphs, states, reliability, tasks, cfg = _build_sim(seed=seed)
    baselines = build_all_baselines(graphs, states, gs_names, reliability, cfg)

    results = {}
    ordi = ORDIScheduler(cfg, sat_ids, gs_names, graphs, deepcopy(states), deepcopy(reliability))
    results["ORDI"] = _run_algorithm("ORDI", ordi, tasks, graphs, states, reliability,
                                     sat_ids, gs_names, cfg)

    for name, baseline in baselines.items():
        print(f"  Running {name} ...")
        results[name] = _run_algorithm(name, baseline, tasks, graphs, states, reliability,
                                       sat_ids, gs_names, cfg)

    _save_csv("E1_core", results)
    return results


# ── E2: Fault type profile (each of 7 fault types, ORDI only) ────────────────

def run_E2(seed=0) -> Dict[str, List[EpochMetrics]]:
    print("E2: Fault type profile (ORDI)")
    sats, sat_ids, gs_names, contacts, graphs, states, reliability, tasks, cfg = _build_sim(seed=seed)

    fault_scenarios = {
        "no_fault": [],
        "isl_disruption": [FaultEvent("isl_disruption", 30, 10, [f"{sat_ids[0]}:{sat_ids[1]}"])],
        "plane_outage":   [FaultEvent("plane_outage", 20, 8, sat_ids[:6])],
        "helper_failure": [FaultEvent("helper_failure", 15, 5, [sat_ids[3]])],
        "straggler":      [FaultEvent("straggler", 10, 3, [sat_ids[4]], {"factor": 0.1})],
        "ground_miss":    [FaultEvent("ground_contact_miss", 25, 5, [sat_ids[2]])],
        "battery":        [FaultEvent("battery_shortage", 20, 4, [sat_ids[5]])],
        "thermal":        [FaultEvent("thermal_throttle", 18, 3, [sat_ids[6]])],
    }

    results = {}
    for scenario_name, faults in fault_scenarios.items():
        print(f"  Scenario: {scenario_name}")
        ordi = ORDIScheduler(cfg, sat_ids, gs_names, graphs, deepcopy(states), deepcopy(reliability))
        results[scenario_name] = _run_algorithm(
            "ORDI", ordi, tasks, graphs, states, reliability,
            sat_ids, gs_names, cfg, faults=faults, seed=seed,
        )

    _save_csv("E2_fault_types", results)
    return results


# ── E3: Fault intensity sweep (ORDI vs B5 vs B6) ─────────────────────────────

def run_E3(seed=0) -> Dict[str, List[EpochMetrics]]:
    print("E3: Fault intensity sweep")
    sats, sat_ids, gs_names, contacts, graphs, states, reliability, tasks, cfg = _build_sim(seed=seed)
    baselines = build_all_baselines(graphs, states, gs_names, reliability, cfg)

    fault_rates = [0.0, 0.05, 0.10, 0.20, 0.35, 0.50]
    results = {}

    for rate in fault_rates:
        faults = random_fault_schedule(sat_ids, N_EPOCHS, fault_rate=rate, seed=seed)
        for alg_name in ["ORDI", "B5_seco_like", "B6_full_replication"]:
            key = f"{alg_name}@fault={rate:.2f}"
            print(f"  {key}")
            if alg_name == "ORDI":
                sched = ORDIScheduler(cfg, sat_ids, gs_names, graphs,
                                      deepcopy(states), deepcopy(reliability))
                results[key] = _run_algorithm(
                    "ORDI", sched, tasks, graphs, states, reliability,
                    sat_ids, gs_names, cfg, faults=faults, seed=seed,
                )
            else:
                baseline = baselines[alg_name]
                results[key] = _run_algorithm(
                    alg_name, baseline, tasks, graphs, states, reliability,
                    sat_ids, gs_names, cfg, faults=faults, seed=seed,
                )

    _save_csv("E3_fault_intensity", results)
    return results


# ── E4: Scalability (constellation size 10–100 sats) ─────────────────────────

def run_E4(seed=0) -> Dict[str, List[EpochMetrics]]:
    print("E4: Scalability sweep")
    results = {}

    for n_sats in [12, 24, 36, 60]:
        planes = 6
        per_plane = n_sats // planes
        print(f"  {n_sats} sats ({planes}p × {per_plane})")
        sats, sat_ids, gs_names, contacts, graphs, states, reliability, tasks, cfg = \
            _build_sim(n_planes=planes, sats_per_plane=per_plane, seed=seed)
        baselines = build_all_baselines(graphs, states, gs_names, reliability, cfg)

        for alg_name in ["ORDI", "B5_seco_like"]:
            key = f"{alg_name}@n={n_sats}"
            if alg_name == "ORDI":
                sched = ORDIScheduler(cfg, sat_ids, gs_names, graphs,
                                      deepcopy(states), deepcopy(reliability))
                results[key] = _run_algorithm(
                    "ORDI", sched, tasks, graphs, states, reliability,
                    sat_ids, gs_names, cfg,
                )
            else:
                results[key] = _run_algorithm(
                    alg_name, baselines[alg_name], tasks, graphs, states, reliability,
                    sat_ids, gs_names, cfg,
                )

    _save_csv("E4_scalability", results)
    return results


# ── E5: Deadline tightness sweep ──────────────────────────────────────────────

def run_E5(seed=0) -> Dict[str, List[EpochMetrics]]:
    print("E5: Deadline tightness sweep")
    results = {}

    for slack in [60, 120, 180, 300, 600]:
        sats, sat_ids, gs_names, contacts, graphs, states, reliability, tasks, cfg = \
            _build_sim(seed=seed, deadline_slack=float(slack))
        baselines = build_all_baselines(graphs, states, gs_names, reliability, cfg)

        for alg_name in ["ORDI", "B2_onboard_only", "B4_serval_like"]:
            key = f"{alg_name}@slack={slack}s"
            if alg_name == "ORDI":
                sched = ORDIScheduler(cfg, sat_ids, gs_names, graphs,
                                      deepcopy(states), deepcopy(reliability))
                results[key] = _run_algorithm(
                    "ORDI", sched, tasks, graphs, states, reliability,
                    sat_ids, gs_names, cfg,
                )
            else:
                results[key] = _run_algorithm(
                    alg_name, baselines[alg_name], tasks, graphs, states, reliability,
                    sat_ids, gs_names, cfg,
                )

    _save_csv("E5_deadline", results)
    return results


# ── E6: λ_R penalty sweep ─────────────────────────────────────────────────────

def run_E6(seed=0) -> Dict[str, List[EpochMetrics]]:
    print("E6: λ_R (replication penalty) sweep")
    sats, sat_ids, gs_names, contacts, graphs, states, reliability, tasks, base_cfg = \
        _build_sim(seed=seed)
    results = {}

    for lambda_R in [0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0]:
        cfg = deepcopy(base_cfg)
        cfg.lambda_R = lambda_R
        key = f"ORDI@lambda_R={lambda_R}"
        sched = ORDIScheduler(cfg, sat_ids, gs_names, graphs,
                              deepcopy(states), deepcopy(reliability))
        results[key] = _run_algorithm(
            "ORDI", sched, tasks, graphs, states, reliability,
            sat_ids, gs_names, cfg,
        )

    _save_csv("E6_lambda_R", results)
    return results


# ── E7: Correlated failures (orbital-plane outage) ────────────────────────────

def run_E7(seed=0) -> Dict[str, List[EpochMetrics]]:
    print("E7: Correlated failures (plane outage) — ORDI vs B6")
    sats, sat_ids, gs_names, contacts, graphs, states, reliability, tasks, cfg = \
        _build_sim(seed=seed)
    baselines = build_all_baselines(graphs, states, gs_names, reliability, cfg)

    # Knock out entire planes (6 sats each)
    plane_sizes = [6, 12]  # 1 plane, 2 planes
    results = {}

    for n_plane_sats in plane_sizes:
        plane_sats = sat_ids[:n_plane_sats]
        faults = [FaultEvent("plane_outage", 20, 10, plane_sats)]
        label = f"plane_{n_plane_sats}_sats"

        for alg_name in ["ORDI", "B6_full_replication"]:
            key = f"{alg_name}@{label}"
            if alg_name == "ORDI":
                sched = ORDIScheduler(cfg, sat_ids, gs_names, graphs,
                                      deepcopy(states), deepcopy(reliability))
                results[key] = _run_algorithm(
                    "ORDI", sched, tasks, graphs, states, reliability,
                    sat_ids, gs_names, cfg, faults=faults, seed=seed,
                )
            else:
                results[key] = _run_algorithm(
                    alg_name, baselines[alg_name], tasks, graphs, states, reliability,
                    sat_ids, gs_names, cfg, faults=faults, seed=seed,
                )

    _save_csv("E7_correlated", results)
    return results


# ── E8: ILP vs greedy optimality gap ─────────────────────────────────────────

def run_E8(seed=0) -> Dict[str, List[EpochMetrics]]:
    print("E8: ILP vs greedy optimality gap (small instances)")
    from ordi.scheduler.ilp import solve_ilp

    # Small constellation: 3 planes × 4 sats = 12 sats, 2 tasks
    sats, sat_ids, gs_names, contacts, graphs, states, reliability, tasks, cfg = \
        _build_sim(n_planes=3, sats_per_plane=4, arrival_rate=1.0, seed=seed)

    results = {"ORDI_greedy": [], "ORDI_ILP": []}
    sat_cap = {s: states[s].C_i * EPOCH_LENGTH_S for s in sat_ids}

    greedy_sched = ORDIScheduler(cfg, sat_ids, gs_names, graphs,
                                 deepcopy(states), deepcopy(reliability))

    for epoch in range(min(20, N_EPOCHS)):  # first 20 epochs only
        ep_start = T_SIM_START + epoch * EPOCH_LENGTH_S
        pending = [t for t in tasks if t.release_time <= ep_start < t.deadline]
        if not pending:
            continue

        # Greedy
        g_result = greedy_sched.schedule_epoch(epoch, T_SIM_START, pending)
        results["ORDI_greedy"].append(
            compute_metrics(g_result, pending, ep_start, sat_cap, cfg.alpha)
        )

        # ILP
        ilp_result = solve_ilp(
            epoch, T_SIM_START, pending, graphs, deepcopy(states),
            deepcopy(reliability), gs_names, cfg, time_limit_s=30.0,
        )
        if ilp_result:
            results["ORDI_ILP"].append(
                compute_metrics(ilp_result, pending, ep_start, sat_cap, cfg.alpha)
            )

    _save_csv("E8_ilp_gap", results)
    return results


# ── master runner ─────────────────────────────────────────────────────────────

ALL_EXPERIMENTS = {
    "E1": run_E1, "E2": run_E2, "E3": run_E3, "E4": run_E4,
    "E5": run_E5, "E6": run_E6, "E7": run_E7, "E8": run_E8,
}


def run_all(seed=0):
    for exp_id, fn in ALL_EXPERIMENTS.items():
        print(f"\n{'='*50}\n{exp_id}\n{'='*50}")
        fn(seed=seed)
