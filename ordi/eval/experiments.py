"""
Experiment runner for E1–E8.

Each experiment returns a dict: {algorithm_name: List[EpochMetrics]}.
Results are saved to results/<experiment_id>.csv.

Shared realistic LEO-EO simulation setup (all experiments unless overridden):
  - Synthetic Walker constellation: 6 planes × 6 sats = 36 sats (default)
  - 2 Northern-hemisphere ground stations (Fairbanks, Greenwich) — typical
    for a Northern-focus EO mission; creates genuine routing pressure since
    Southern-hemisphere source sats have no direct downlink within 300 s.
  - FOV-constrained task generation: tasks arise only when a satellite is
    within 600 km of one of 100 random ground targets — physically correct
    for an EO system whose camera covers a finite swath.
  - Orbital period: 5760 s (96 min) — correct for 550 km LEO altitude.
  - Simulation horizon: 10 800 s (~1.875 orbits); 180 × 60 s epochs.
  - Tasks: Poisson arrivals, 6 tasks/orbit, deadline 300 s.
"""

from __future__ import annotations
import csv
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

from ordi.orbit.contacts import (
    build_synthetic_walker, compute_contact_windows,
    compute_sat_groundtracks, DEFAULT_GROUND_STATIONS,
)
from ordi.orbit.graph import build_epoch_graphs
from ordi.sim.satellite import make_constellation_states
from ordi.sim.reliability import ReliabilityModel
from ordi.tasks.generator import generate_tasks
from ordi.scheduler.ordi import ORDIScheduler, ORDIConfig
from ordi.baselines.baselines import build_all_baselines
from ordi.faults.injector import FaultInjector, random_fault_schedule, FaultEvent
from ordi.eval.metrics import compute_metrics, aggregate_metrics, EpochMetrics

RESULTS_DIR = "results"
SIM_DURATION_S = 10_800       # ~1.875 orbits at 550 km
EPOCH_LENGTH_S = 60.0
N_EPOCHS = int(SIM_DURATION_S / EPOCH_LENGTH_S)
T_SIM_START = 0.0

# Two Northern-hemisphere GS used as the shared ground segment across all
# experiments — Fairbanks (65°N) and Greenwich (51°N).
_NORTHERN_GS = [
    gs for gs in DEFAULT_GROUND_STATIONS
    if gs[0] in {"fairbanks", "greenwich"}
]


# ── simulation bootstrap ──────────────────────────────────────────────────────

def _build_sim(n_planes=6, sats_per_plane=6, seed=0, deadline_slack=300.0,
               arrival_rate=6.0, ground_stations=None,
               orbit_period_s=5760.0,
               use_fov=True, fov_range_km=600.0, n_targets=100,
               min_elevation_deg=5.0):
    """Build all shared simulation objects.

    orbit_period_s : realistic LEO period at 550 km altitude (~5760 s / 96 min).
    use_fov        : FOV-constrained task generation — tasks arise only when a
                     satellite is within fov_range_km of a ground target.
    fov_range_km   : camera footprint radius (600 km ≈ ±42° off-nadir at 550 km
                     altitude; realistic for a wide-field EO imager).
    n_targets      : number of random ground targets (uniformly in ±60° lat).
    """
    import random as _rng_mod
    if ground_stations is None:
        ground_stations = _NORTHERN_GS
    sats = build_synthetic_walker(n_planes=n_planes, sats_per_plane=sats_per_plane)
    sat_ids = [s.name for s in sats]
    gs_names = {gs[0] for gs in ground_stations}

    print(f"  Computing contact windows for {len(sats)} sats × {len(ground_stations)} GS ...")
    t0 = time.time()
    contacts = compute_contact_windows(
        sats,
        t_start_unix=0.0,
        t_end_unix=SIM_DURATION_S,
        dt_seconds=60.0,
        ground_stations=ground_stations,
        min_elevation_deg=min_elevation_deg,
    )
    print(f"  {len(contacts)} contact events in {time.time()-t0:.1f}s")

    graphs = build_epoch_graphs(contacts, T_SIM_START, EPOCH_LENGTH_S, N_EPOCHS)
    states = make_constellation_states(sat_ids, seed=seed)
    reliability = ReliabilityModel()

    sat_groundtrack = None
    ground_targets = None
    if use_fov:
        print("  Computing satellite groundtracks for FOV task generation ...")
        t1 = time.time()
        sat_groundtrack = compute_sat_groundtracks(
            sats, 0.0, SIM_DURATION_S, dt_seconds=60.0
        )
        # Random ground targets uniformly distributed in [-60°, 60°] latitude
        rng_t = _rng_mod.Random(seed + 99)
        ground_targets = [
            (rng_t.uniform(-60.0, 60.0), rng_t.uniform(-180.0, 180.0))
            for _ in range(n_targets)
        ]
        print(f"  Groundtracks done in {time.time()-t1:.1f}s, {n_targets} targets")

    tasks = generate_tasks(
        sat_ids, SIM_DURATION_S,
        arrival_rate_per_orbit=arrival_rate,
        orbit_period_s=orbit_period_s,
        deadline_slack_s=deadline_slack,
        seed=seed,
        sat_groundtrack=sat_groundtrack,
        ground_targets=ground_targets,
        fov_range_km=fov_range_km,
    )
    cfg = ORDIConfig(epoch_length=EPOCH_LENGTH_S)

    return sats, sat_ids, gs_names, contacts, graphs, states, reliability, tasks, cfg


# ── parallel worker (module-level so multiprocessing can pickle it) ───────────

def _parallel_run_algorithm(args: Tuple) -> Tuple[str, List[EpochMetrics]]:
    """
    Worker for ProcessPoolExecutor.
    args = (result_key, sched_name, scheduler_class,
            tasks, graphs, states, reliability,
            sat_ids, gs_names, cfg, faults, seed)

    result_key  : key stored in the results dict
    sched_name  : "ORDI" or the baseline name (controls dispatch)
    scheduler_class : class to instantiate
    """
    (result_key, sched_name, scheduler_class,
     tasks, graphs, states, reliability,
     sat_ids, gs_names, cfg, faults, seed) = args

    local_states = deepcopy(states)
    local_rel = deepcopy(reliability)

    injector = None
    if faults:
        injector = FaultInjector(local_states, local_rel, [], rng_seed=seed)
        for f in faults:
            injector.schedule(f)

    if sched_name == "ORDI":
        sched = ORDIScheduler(cfg, sat_ids, gs_names, graphs, local_states, local_rel)
    else:
        sched = scheduler_class(graphs, local_states, gs_names, local_rel, cfg)

    sat_cap = {s: local_states[s].C_i * EPOCH_LENGTH_S for s in sat_ids}
    epoch_metrics = []

    for epoch in range(N_EPOCHS):
        ep_start = T_SIM_START + epoch * EPOCH_LENGTH_S

        if injector:
            injector.apply_epoch(epoch)

        pending = [t for t in tasks if t.release_time <= ep_start < t.deadline]

        if sched_name == "ORDI":
            result = sched.schedule_epoch(epoch, T_SIM_START, pending)
        else:
            result = sched.schedule(epoch, T_SIM_START, pending)

        epoch_metrics.append(compute_metrics(result, pending, ep_start, sat_cap, cfg.alpha))

        if injector:
            injector.withdraw_epoch(epoch + 1)

    return result_key, epoch_metrics


def _run_parallel(job_args: List[Tuple], desc: str = "") -> Dict[str, List[EpochMetrics]]:
    """Submit a list of _parallel_run_algorithm arg-tuples and collect results."""
    n = len(job_args)
    results: Dict[str, List[EpochMetrics]] = {}
    n_workers = min(n, 16)
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_parallel_run_algorithm, args): args[0] for args in job_args}
        for fut in tqdm(as_completed(futures), total=n, desc=desc, unit="job"):
            key, metrics = fut.result()
            results[key] = metrics
    return results


_CSV_FIELDS = [
    "algorithm", "deadline_miss_ratio", "delivered_utility", "partial_coverage",
    "recovery_latency", "isl_traffic_bits", "downlink_volume_bits",
    "energy_joules", "helper_utilization", "objective",
]


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
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS,
                                extrasaction="ignore", restval="")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {path}")


# ── E1: Core performance (ORDI vs all baselines) ─────────────────────────────

def run_E1(seed=0) -> Dict[str, List[EpochMetrics]]:
    """
    Core performance comparison using the shared realistic LEO-EO setup.

    Uses a 6×4 Walker (24 sats, fewer sats than default 36 to stress routing)
    with arrival_rate=8 so the FOV filter still yields ~13 tasks.

    25° minimum elevation angle for ground-station contacts, realistic for
    Ka-band dishes (Starlink's original 25° threshold; industry range 10–25°).
    At 550 km altitude this cuts each GS pass from ~10 min (at 5°) to ~4.5 min,
    so a given source satellite is in direct view only ~9 % of the time.

    B1 (DirectDownlink) is a direct-only baseline: it must wait for the source
    satellite itself to enter a GS contact window — no ISL relay.  With narrow
    windows and a 120 s deadline, most source satellites are not in view of a
    ground station and B1 misses heavily.  ORDI (and B5/B6) route processed
    results via ISL to whichever satellite is currently in GS contact, so they
    remain feasible.
    """
    print("E1: Core performance (6×4 Walker, 2 Northern GS, 300 s deadline, 25° GS elevation)")
    sats, sat_ids, gs_names, contacts, graphs, states, reliability, tasks, cfg = \
        _build_sim(n_planes=6, sats_per_plane=4, seed=seed,
                   arrival_rate=8.0, deadline_slack=300.0, min_elevation_deg=25.0)
    baselines = build_all_baselines(graphs, states, gs_names, reliability, cfg)

    job_args = [
        ("ORDI", "ORDI", ORDIScheduler,
         tasks, graphs, states, reliability, sat_ids, gs_names, cfg, None, seed)
    ]
    for name, baseline in baselines.items():
        job_args.append((
            name, name, baseline.__class__,
            tasks, graphs, states, reliability, sat_ids, gs_names, cfg, None, seed,
        ))

    results = _run_parallel(job_args, desc="E1 algorithms")
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

    # result_key = scenario name; sched_name = "ORDI" for all
    job_args = [
        (scenario_name, "ORDI", ORDIScheduler,
         tasks, graphs, states, reliability, sat_ids, gs_names, cfg, faults, seed)
        for scenario_name, faults in fault_scenarios.items()
    ]

    results = _run_parallel(job_args, desc="E2 scenarios")
    _save_csv("E2_fault_types", results)
    return results


# ── E3: Fault intensity sweep (ORDI vs B5 vs B6) ─────────────────────────────

def run_E3(seed=0) -> Dict[str, List[EpochMetrics]]:
    print("E3: Fault intensity sweep")
    sats, sat_ids, gs_names, contacts, graphs, states, reliability, tasks, cfg = _build_sim(seed=seed)
    baselines = build_all_baselines(graphs, states, gs_names, reliability, cfg)

    fault_rates = [0.0, 0.05, 0.10, 0.20, 0.35, 0.50]
    alg_names = ["ORDI", "B5_seco_like", "B6_full_replication"]

    job_args = []
    for rate in fault_rates:
        faults = random_fault_schedule(sat_ids, N_EPOCHS, fault_rate=rate, seed=seed)
        for alg_name in alg_names:
            key = f"{alg_name}@fault={rate:.2f}"
            cls = ORDIScheduler if alg_name == "ORDI" else baselines[alg_name].__class__
            job_args.append((
                key, alg_name, cls,
                tasks, graphs, states, reliability, sat_ids, gs_names, cfg, faults, seed,
            ))

    results = _run_parallel(job_args, desc="E3 rate×alg")
    _save_csv("E3_fault_intensity", results)
    return results


# ── E4: Scalability (constellation size 10–100 sats) ─────────────────────────

_E4_CONFIGS = {
    12: (3, 4),   # 3 planes × 4 sats: enough inter-plane contact density
    24: (4, 6),   # 4 planes × 6 sats
    36: (6, 6),   # 6 planes × 6 sats (matches E1 baseline)
    60: (6, 10),  # 6 planes × 10 sats
}


def run_E4(seed=0) -> Dict[str, List[EpochMetrics]]:
    print("E4: Scalability sweep")
    results = {}

    # Sim is rebuilt per constellation size → keep outer loop sequential,
    # parallelize the 2 algorithms per size.
    for n_sats in tqdm([12, 24, 36, 60], desc="E4 constellation sizes", unit="size"):
        planes, per_plane = _E4_CONFIGS[n_sats]
        sats, sat_ids, gs_names, contacts, graphs, states, reliability, tasks, cfg = \
            _build_sim(n_planes=planes, sats_per_plane=per_plane, seed=seed)
        baselines = build_all_baselines(graphs, states, gs_names, reliability, cfg)

        job_args = []
        for alg_name in ["ORDI", "B5_seco_like"]:
            key = f"{alg_name}@n={n_sats}"
            cls = ORDIScheduler if alg_name == "ORDI" else baselines[alg_name].__class__
            job_args.append((
                key, alg_name, cls,
                tasks, graphs, states, reliability, sat_ids, gs_names, cfg, None, seed,
            ))

        results.update(_run_parallel(job_args, desc=f"E4 n={n_sats}"))

    _save_csv("E4_scalability", results)
    return results


# ── E5: Deadline tightness sweep ──────────────────────────────────────────────

def run_E5(seed=0) -> Dict[str, List[EpochMetrics]]:
    print("E5: Deadline tightness sweep")
    results = {}

    # Sim is rebuilt per slack → keep outer loop sequential,
    # parallelize the 3 algorithms per slack.
    for slack in tqdm([60, 120, 180, 300, 600], desc="E5 deadline slacks", unit="slack"):
        sats, sat_ids, gs_names, contacts, graphs, states, reliability, tasks, cfg = \
            _build_sim(seed=seed, deadline_slack=float(slack))
        baselines = build_all_baselines(graphs, states, gs_names, reliability, cfg)

        job_args = []
        for alg_name in ["ORDI", "B2_onboard_only", "B4_serval_like"]:
            key = f"{alg_name}@slack={slack}s"
            cls = ORDIScheduler if alg_name == "ORDI" else baselines[alg_name].__class__
            job_args.append((
                key, alg_name, cls,
                tasks, graphs, states, reliability, sat_ids, gs_names, cfg, None, seed,
            ))

        results.update(_run_parallel(job_args, desc=f"E5 slack={slack}s"))

    _save_csv("E5_deadline", results)
    return results


# ── E6: λ_R penalty sweep ─────────────────────────────────────────────────────

def run_E6(seed=0) -> Dict[str, List[EpochMetrics]]:
    print("E6: λ_R (replication penalty) sweep")
    sats, sat_ids, gs_names, contacts, graphs, states, reliability, tasks, base_cfg = \
        _build_sim(seed=seed)

    lambda_Rs = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
    job_args = []
    for lambda_R in lambda_Rs:
        cfg = deepcopy(base_cfg)
        cfg.lambda_R = lambda_R
        key = f"ORDI@lambda_R={lambda_R}"
        job_args.append((
            key, "ORDI", ORDIScheduler,
            tasks, graphs, states, reliability, sat_ids, gs_names, cfg, None, seed,
        ))

    results = _run_parallel(job_args, desc="E6 lambda_R")
    _save_csv("E6_lambda_R", results)
    return results


# ── E7: Correlated failures (orbital-plane outage) ────────────────────────────

def run_E7(seed=0) -> Dict[str, List[EpochMetrics]]:
    print("E7: Correlated failures (plane outage) — ORDI vs B6")
    sats, sat_ids, gs_names, contacts, graphs, states, reliability, tasks, cfg = \
        _build_sim(seed=seed)
    baselines = build_all_baselines(graphs, states, gs_names, reliability, cfg)

    plane_sizes = [6, 12]  # 1 plane, 2 planes
    job_args = []
    for n_plane_sats in plane_sizes:
        plane_sats = sat_ids[:n_plane_sats]
        faults = [FaultEvent("plane_outage", 20, 10, plane_sats)]
        label = f"plane_{n_plane_sats}_sats"
        for alg_name in ["ORDI", "B6_full_replication"]:
            key = f"{alg_name}@{label}"
            cls = ORDIScheduler if alg_name == "ORDI" else baselines[alg_name].__class__
            job_args.append((
                key, alg_name, cls,
                tasks, graphs, states, reliability, sat_ids, gs_names, cfg, faults, seed,
            ))

    results = _run_parallel(job_args, desc="E7 configs")
    _save_csv("E7_correlated", results)
    return results


# ── E8: ILP vs greedy optimality gap ─────────────────────────────────────────
# Left sequential: greedy and ILP are compared epoch-by-epoch, and HiGHS
# already uses 8 threads internally.

def run_E8(seed=0) -> Dict[str, List[EpochMetrics]]:
    print("E8: ILP vs greedy optimality gap (small instances)")
    from ordi.scheduler.ilp import solve_ilp

    # Small constellation: 3 planes × 4 sats = 12 sats
    sats, sat_ids, gs_names, contacts, graphs, states, reliability, tasks, cfg = \
        _build_sim(n_planes=3, sats_per_plane=4, arrival_rate=3.0, seed=seed)

    results = {"ORDI_greedy": [], "ORDI_ILP": []}
    sat_cap = {s: states[s].C_i * EPOCH_LENGTH_S for s in sat_ids}

    greedy_sched = ORDIScheduler(cfg, sat_ids, gs_names, graphs,
                                 deepcopy(states), deepcopy(reliability))

    for epoch in range(N_EPOCHS):
        ep_start = T_SIM_START + epoch * EPOCH_LENGTH_S
        pending = [t for t in tasks if t.release_time <= ep_start < t.deadline]
        if not pending:
            continue

        # Greedy
        g_result = greedy_sched.schedule_epoch(epoch, T_SIM_START, pending)
        results["ORDI_greedy"].append(
            compute_metrics(g_result, pending, ep_start, sat_cap, cfg.alpha)
        )

        # ILP (HiGHS uses 8 threads internally)
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
    for exp_id, fn in tqdm(ALL_EXPERIMENTS.items(), desc="Experiments", unit="exp"):
        print(f"\n{'='*50}\n{exp_id}\n{'='*50}")
        fn(seed=seed)
