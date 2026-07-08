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
  - Simulation horizon: 17 280 s (3 orbits); 288 × 60 s epochs.
  - Tasks: Poisson arrivals, 6 tasks/orbit, log-normal deadlines (σ=0.6)
    with per-type medians: wildfire 300 s, change 480 s, ship 600 s,
    cloud_filter 1200 s (EO SLA tiers; Lemaître et al. 2002).
"""

from __future__ import annotations
import csv
import math
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import copy, deepcopy
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

from ordi.orbit.contacts import (
    build_synthetic_walker, compute_contact_windows,
    compute_sat_groundtracks, DEFAULT_GROUND_STATIONS,
)
from ordi.orbit.graph import build_epoch_graphs
from ordi.sim.satellite import make_constellation_states
from ordi.sim.cots_measurements import (
    atlas_200dk_bupt1_params, load_cots_measurement_profile,
)
from ordi.sim.reliability import ReliabilityModel
from ordi.tasks.generator import generate_tasks
from ordi.scheduler.ordi import (
    ORDIScheduler, ORDIConfig, SchedulerResult, TileAssignment,
)
from ordi.baselines.baselines import (
    build_all_baselines, ALL_BASELINES, OnboardOnly, SECOLike, ServalLike,
    FullReplication, RandomReplication,
)
from ordi.faults.injector import FaultInjector, random_fault_schedule, FaultEvent
from ordi.eval.metrics import (
    compute_metrics, compute_realized_metrics, aggregate_metrics, EpochMetrics,
)

RESULTS_DIR = "results"
SIM_DURATION_S = 17_280       # 3 complete orbits at 550 km (3 × 5760 s)
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

def _build_sim(n_planes=6, sats_per_plane=6, seed=0, deadline_slack=600.0,
               deadline_lognorm_sigma=0.6,
               arrival_rate=6.0, ground_stations=None,
               orbit_period_s=5760.0,
               use_fov=True, fov_range_km=600.0, n_targets=100,
               min_elevation_deg=5.0, satellite_params_factory=None):
    """Build all shared simulation objects.

    orbit_period_s        : realistic LEO period at 550 km altitude (~5760 s / 96 min).
    deadline_slack        : global deadline scale (reference = 600 s).  Per-type medians
                            (wildfire 300 s, change 480 s, ship 600 s, cloud_filter 1200 s)
                            are multiplied by deadline_slack / 600.
    deadline_lognorm_sigma: log-space std-dev for per-task deadline sampling (σ=0.6).
    use_fov               : FOV-constrained task generation — tasks arise only when a
                            satellite is within fov_range_km of a ground target.
    fov_range_km          : camera footprint radius (600 km ≈ ±42° off-nadir at 550 km
                            altitude; realistic for a wide-field EO imager).
    n_targets             : number of random ground targets (uniformly in ±60° lat).
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
    states = make_constellation_states(
        sat_ids, seed=seed, params_factory=satellite_params_factory
    )
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
        deadline_lognorm_sigma=deadline_lognorm_sigma,
        seed=seed,
        sat_groundtrack=sat_groundtrack,
        ground_targets=ground_targets,
        fov_range_km=fov_range_km,
    )
    cfg = ORDIConfig(epoch_length=EPOCH_LENGTH_S)

    return sats, sat_ids, gs_names, contacts, graphs, states, reliability, tasks, cfg


# ── stateful rolling-horizon helpers ──────────────────────────────────────────

def _assignment_viable(a: TileAssignment, states) -> bool:
    """In-flight tile survives if any replica's helper+aggregator are both alive
    (a surviving backup avoids re-transmission). No-helper assignments never replan."""
    if not a.replicas:
        return True
    for r in a.replicas:
        h = states.get(r.helper)
        g = states.get(r.aggregator)
        if h is not None and g is not None and h.A_i and g.A_i:
            return True
    return False


def _uncommitted_tasks(pending, committed):
    """Shallow task copies holding only not-yet-committed tiles."""
    out = []
    for task in pending:
        rem = [t for t in task.tiles if (task.task_id, t.tile_id) not in committed]
        if not rem:
            continue
        sub = copy(task)
        sub.tiles = rem
        out.append(sub)
    return out


def _simulate_stateful(schedule_fn, tasks, sat_ids, states, cfg, injector=None,
                       reliability=None, realized_trials=500, realized_seed=0):
    """Run one stateful rolling-horizon simulation and return a lifetime
    EpochMetrics.  schedule_fn(epoch, todo_tasks) -> SchedulerResult dispatches
    to ORDI / a baseline / the ILP.  A committed tile stays in-flight (not
    re-scheduled, not re-charged) until a fault invalidates all its replicas.

    When a reliability model is supplied, the final lifetime assignment set is
    also scored by Monte Carlo (compute_realized_metrics): links and nodes are
    sampled from their π values with draws shared across a tile's replicas, so
    the realized_* fields report delivery outcomes the modeled z_kv assumes
    away.  Hard outages already pruned infeasible candidates during scheduling;
    this layer adds the soft stochastic loss the reliability model specifies."""
    sat_cap = {s: states[s].C_i * EPOCH_LENGTH_S for s in sat_ids}
    all_tiles = [(t.task_id, tile.tile_id) for t in tasks for tile in t.tiles]
    committed: Dict[Tuple[int, int], TileAssignment] = {}

    for epoch in range(N_EPOCHS):
        ep_start = T_SIM_START + epoch * EPOCH_LENGTH_S
        if injector:
            injector.apply_epoch(epoch)
        # Drop committed tiles whose every replica lost a node → re-planned.
        for key in list(committed.keys()):
            if not _assignment_viable(committed[key], states):
                del committed[key]
        pending = [t for t in tasks if t.release_time <= ep_start < t.deadline]
        todo = _uncommitted_tasks(pending, committed)
        result = schedule_fn(epoch, todo)
        for a in result.assignments:
            if a.z_kv > 0 and not math.isinf(a.L_hat):
                committed[(a.task_id, a.tile_id)] = a
        if injector:
            injector.withdraw_epoch(epoch + 1)

    final = [committed.get(k) or TileAssignment(task_id=k[0], tile_id=k[1])
             for k in all_tiles]
    res = SchedulerResult(
        epoch=N_EPOCHS - 1, assignments=final, total_utility=0.0,
        energy_penalty=0.0, comm_penalty=0.0, rep_penalty=0.0,
        objective=0.0, link_utilization={},
    )
    m = compute_metrics(res, tasks, 0.0, sat_cap, cfg.alpha)
    r_total = sum(max(0, len(a.replicas) - 1) for a in final)
    m.objective = (m.delivered_utility
                   - cfg.lambda_E * m.energy_joules
                   - cfg.lambda_R * r_total)
    if reliability is not None and realized_trials > 0:
        rm = compute_realized_metrics(
            res, tasks, reliability, cfg.alpha,
            n_trials=realized_trials, seed=realized_seed,
        )
        m.realized_miss_ratio = rm.realized_miss_ratio
        m.realized_utility = rm.realized_utility
        m.realized_coverage = rm.realized_coverage
    return m


# ── parallel worker (module-level so multiprocessing can pickle it) ───────────

# Simulation environment shared by all jobs of one _run_parallel call, shipped
# once per worker via the pool initializer instead of pickled into every job.
_WORKER_SHARED: Optional[Tuple] = None


def _init_worker_shared(shared: Tuple) -> None:
    global _WORKER_SHARED
    _WORKER_SHARED = shared


def _parallel_run_algorithm(args: Tuple) -> Tuple[str, List[EpochMetrics]]:
    """
    Worker for ProcessPoolExecutor.
    args = (result_key, sched_name, scheduler_class, cfg, faults, seed);
    (tasks, graphs, states, reliability, sat_ids, gs_names) come from the
    worker-global _WORKER_SHARED. Workers never mutate the shared objects:
    states/reliability are deepcopied per job and tasks are shallow-copied.

    Stateful rolling-horizon: a committed tile stays in-flight and is not
    re-scheduled (nor re-charged) until a fault invalidates all its replicas.
    Metrics are aggregated once per tile lifetime, so ISL/energy count one
    transfer per tile instead of once per epoch it stayed pending.
    """
    (result_key, sched_name, scheduler_class, cfg, faults, seed) = args
    (tasks, graphs, states, reliability, sat_ids, gs_names) = _WORKER_SHARED

    local_states = deepcopy(states)
    local_rel = deepcopy(reliability)

    # ground_contact_miss and isl_disruption mutate epoch graph edges;
    # deepcopy graphs when either is present to keep the shared object clean.
    _GRAPH_MUTATING = {"ground_contact_miss", "isl_disruption"}
    mutates_graph = any(f.fault_type in _GRAPH_MUTATING for f in faults)
    local_graphs = deepcopy(graphs) if mutates_graph else graphs

    injector = None
    if faults:
        injector = FaultInjector(local_states, local_rel, [], rng_seed=seed,
                                 graphs=local_graphs, gs_names=gs_names)
        for f in faults:
            injector.schedule(f)

    is_ordi = isinstance(scheduler_class, type) and issubclass(scheduler_class, ORDIScheduler)
    if is_ordi:
        sched = scheduler_class(cfg, sat_ids, gs_names, local_graphs, local_states, local_rel)
    else:
        sched = scheduler_class(local_graphs, local_states, gs_names, local_rel, cfg)

    def schedule_fn(ep, td):
        if is_ordi:
            return sched.schedule_epoch(ep, T_SIM_START, td)
        return sched.schedule(ep, T_SIM_START, td)

    m = _simulate_stateful(schedule_fn, tasks, sat_ids, local_states, cfg, injector,
                           reliability=local_rel, realized_seed=seed)
    return result_key, [m]


def _run_parallel(shared: Tuple, job_args: List[Tuple],
                  desc: str = "") -> Dict[str, List[EpochMetrics]]:
    """Submit a list of _parallel_run_algorithm arg-tuples and collect results.

    shared = (tasks, graphs, states, reliability, sat_ids, gs_names) is shipped
    once per worker via the pool initializer instead of pickled into every job.
    """
    # Populate the on-disk numba JIT cache before workers spawn so they load
    # the compiled kernel instead of each paying the compile.
    try:
        from ordi.orbit._dijkstra_numba import warmup_jit
        warmup_jit()
    except ImportError:
        pass

    n = len(job_args)
    results: Dict[str, List[EpochMetrics]] = {}
    n_workers = min(n, 16)
    with ProcessPoolExecutor(max_workers=n_workers,
                             initializer=_init_worker_shared,
                             initargs=(shared,)) as pool:
        futures = {pool.submit(_parallel_run_algorithm, args): args[0] for args in job_args}
        for fut in tqdm(as_completed(futures), total=n, desc=desc, unit="job"):
            key, metrics = fut.result()
            results[key] = metrics
    # as_completed yields in finish order; reorder to submission order so CSV
    # rows (and therefore plot bars) are stable across runs.
    return {args[0]: results[args[0]] for args in job_args}


def _resolve_fault_specs(fault_specs, sat_ids, tasks) -> List[FaultEvent]:
    """Translate declarative fault specs into FaultEvents post-build.

    Spec forms (build-dependent targets cannot be resolved by the caller):
      ("random_schedule", fault_rate, rng_seed)
          → random_fault_schedule over the built constellation
      (fault_type, start, duration, targets[, params]) with targets one of:
          "hot_sources"   → the half of satellites sourcing the most tasks
                            (ISL faults get the corresponding link strings)
          (int, ...)      → orbital-plane numbers → that plane's satellites
          [str, ...]      → used verbatim
    """
    faults: List[FaultEvent] = []
    hot: Optional[List[str]] = None
    for spec in fault_specs:
        if spec[0] == "random_schedule":
            _tag, rate, rng_seed = spec
            faults.extend(random_fault_schedule(sat_ids, N_EPOCHS,
                                                fault_rate=rate, seed=rng_seed))
            continue
        ft, start, dur, targets = spec[:4]
        params = spec[4] if len(spec) > 4 else {}
        if targets == "hot_sources":
            if hot is None:
                src_counts = Counter(t.source_sat for t in tasks)
                k = max(1, len(src_counts) // 2)
                hot = [s for s, _ in src_counts.most_common(k)]
            targets = ([f"{a}:{b}" for a in hot for b in sat_ids if a != b]
                       if ft == "isl_disruption" else hot)
        elif targets and all(isinstance(t, int) for t in targets):
            targets = [sid for sid in sat_ids
                       if any(sid.startswith(f"SAT_{p:02d}_") for p in targets)]
        faults.append(FaultEvent(ft, start, dur, targets, params))
    return faults


def _build_and_run_config(args: Tuple) -> List[Tuple[str, List[EpochMetrics]]]:
    """Worker for sweeps that rebuild the sim per configuration (E1-E5, E7):
    build the environment in-process, then run that config's algorithms
    sequentially via _parallel_run_algorithm (sharing through _WORKER_SHARED).
    Keeping build + jobs in one worker avoids shipping the big sim objects.

    jobs: (key, alg_name, scheduler_class[, fault_specs[, cfg_overrides]]);
    see _resolve_fault_specs for the spec forms. cfg_overrides is an optional
    dict of ORDIConfig attributes applied to a per-job copy of the built cfg."""
    build_kwargs, jobs, seed = args
    sats, sat_ids, gs_names, contacts, graphs, states, reliability, tasks, cfg = \
        _build_sim(seed=seed, **build_kwargs)
    _init_worker_shared((tasks, graphs, states, reliability, sat_ids, gs_names))
    out = []
    for job in jobs:
        key, alg_name, cls, fault_specs, cfg_overrides = (tuple(job) + (None, None))[:5]
        faults = (None if fault_specs is None
                  else _resolve_fault_specs(fault_specs, sat_ids, tasks))
        job_cfg = cfg
        if cfg_overrides:
            job_cfg = deepcopy(cfg)
            for k, v in cfg_overrides.items():
                setattr(job_cfg, k, v)
        out.append(_parallel_run_algorithm((key, alg_name, cls, job_cfg, faults, seed)))
    return out


def _run_configs_parallel(config_args: List[Tuple],
                          desc: str = "") -> Dict[str, List[EpochMetrics]]:
    """Run _build_and_run_config over configurations concurrently.
    config_args items: (build_kwargs, [(key, alg_name, scheduler_class)], seed)."""
    try:
        from ordi.orbit._dijkstra_numba import warmup_jit
        warmup_jit()
    except ImportError:
        pass

    results: Dict[str, List[EpochMetrics]] = {}
    n = len(config_args)
    with ProcessPoolExecutor(max_workers=min(n, 16)) as pool:
        futures = [pool.submit(_build_and_run_config, args) for args in config_args]
        for fut in tqdm(as_completed(futures), total=n, desc=desc, unit="config"):
            for key, metrics in fut.result():
                results[key] = metrics
    # Reorder finish-order results to submission order for stable CSV rows.
    return {job[0]: results[job[0]]
            for (_bk, jobs, _seed) in config_args
            for job in jobs}


_CSV_METRIC_KEYS = [
    "deadline_miss_ratio", "delivered_utility", "partial_coverage",
    "recovery_latency", "isl_traffic_bits", "downlink_volume_bits",
    "energy_joules", "helper_utilization", "objective", "n_replicas_avg",
    "realized_miss_ratio", "realized_utility", "realized_coverage",
]
_CSV_FIELDS = (["algorithm"] + _CSV_METRIC_KEYS
               + [f"{k}_std" for k in _CSV_METRIC_KEYS])


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

# The stressed reference scenario (shared by E1, E5, E6): a sparser Walker and
# realistic 25° Ka-band GS elevation so deadlines actually bind.
_E1_BUILD_KWARGS = dict(n_planes=6, sats_per_plane=4,
                        arrival_rate=8.0, deadline_slack=600.0,
                        deadline_lognorm_sigma=0.6, min_elevation_deg=25.0)

def run_E1(seed=0, n_seeds=60) -> Dict[str, List[EpochMetrics]]:
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
    windows and log-normal deadlines whose medians range from 300 s (wildfire)
    to 1200 s (cloud_filter), source satellites are often not in direct GS view
    and B1 misses heavily.  ORDI (and B5/B6) route via ISL to whichever
    satellite is currently in GS contact, so they remain feasible.

    Deadline distribution: log-normal σ=0.6, per-type medians at scale 600 s
    (wildfire→300 s, change→480 s, ship→600 s, cloud_filter→1200 s) matching
    empirical EO SLA tiers (Lemaître et al. 2002; Globus et al. 2004).

    Each seed rebuilds the full environment (orbital phasing, ground targets,
    task arrivals, deadline draws); the CSV reports across-seed mean and std.
    """
    print(f"E1: Core performance (6×4 Walker, 2 Northern GS, lognormal deadlines, "
          f"25° GS elevation, {n_seeds} seeds)")
    build_kwargs = dict(_E1_BUILD_KWARGS)
    alg_classes = [("ORDI", ORDIScheduler)] + [(c.name, c) for c in ALL_BASELINES]

    config_args = []
    for s in range(n_seeds):
        jobs = [(f"{alg}#s{s}", alg, cls) for alg, cls in alg_classes]
        config_args.append((build_kwargs, jobs, seed + s))

    raw = _run_configs_parallel(config_args, desc="E1 seeds")

    # Collapse per-seed records so _save_csv reports mean ± std per algorithm.
    results: Dict[str, List[EpochMetrics]] = {}
    for alg, _cls in alg_classes:
        results[alg] = [m for s in range(n_seeds)
                        for m in raw[f"{alg}#s{s}"]]

    _save_csv("E1_core", results)
    return results


# ── E2: Fault type profile (each of 7 fault types, ORDI only) ────────────────

def run_E2(seed=0, n_seeds=16) -> Dict[str, List[EpochMetrics]]:
    print(f"E2: Fault type profile (ORDI, {n_seeds} seeds)")

    # Every scenario hits the SAME satellites for the SAME duration, so the
    # differences across bars reflect each fault type's intrinsic severity, not
    # its scale.  Targets are the satellites that actually carry the most tasks
    # (FOV sources, resolved post-build by _resolve_fault_specs); hitting idle
    # satellites would be invisible.  Sustained 40 epochs so ORDI's re-planning
    # cannot fully absorb the loss.  Repeated across environment seeds.
    S, D = 10, 40
    fault_scenarios = {
        "no_fault": [],
        "isl_disruption": [("isl_disruption", S, D, "hot_sources")],
        "plane_outage":   [("plane_outage", S, D, "hot_sources")],
        "helper_failure": [("helper_failure", S, D, "hot_sources")],
        "straggler":      [("straggler", S, D, "hot_sources", {"factor": 0.1})],
        "ground_miss":    [("ground_contact_miss", S, D, "hot_sources")],
        "downlink_adv":   [("downlink_adverse", S, D, "hot_sources")],
        "battery":        [("battery_shortage", S, D, "hot_sources")],
        "thermal":        [("thermal_throttle", S, D, "hot_sources")],
    }

    config_args = []
    for s in range(n_seeds):
        jobs = [(f"{name}#s{s}", "ORDI", ORDIScheduler, specs)
                for name, specs in fault_scenarios.items()]
        config_args.append(({}, jobs, seed + s))

    raw = _run_configs_parallel(config_args, desc="E2 seeds")

    results: Dict[str, List[EpochMetrics]] = {}
    for name in fault_scenarios:
        results[name] = [m for s in range(n_seeds) for m in raw[f"{name}#s{s}"]]

    _save_csv("E2_fault_types", results)
    return results


# ── E3: Fault intensity sweep (ORDI vs B5 vs B6) ─────────────────────────────

def run_E3(seed=0, n_seeds=48) -> Dict[str, List[EpochMetrics]]:
    """
    Fault intensity sweep averaging over BOTH randomness sources: each seed
    rebuilds the environment (orbits, tasks, deadlines) AND draws a fresh
    random fault schedule, so the curves carry across-world error bars rather
    than fault-draw jitter on one fixed world.
    """
    print(f"E3: Fault intensity sweep ({n_seeds} seeds)")
    fault_rates = [0.0, 0.05, 0.10, 0.20, 0.35, 0.50]
    alg_classes = [("ORDI", ORDIScheduler),
                   ("B5_seco_like", SECOLike),
                   ("B6_full_replication", FullReplication)]

    config_args = []
    for s in range(n_seeds):
        jobs = []
        for rate in fault_rates:
            specs = [("random_schedule", rate, seed + s)]
            for alg_name, cls in alg_classes:
                jobs.append((f"{alg_name}@fault={rate:.2f}#s{s}",
                             alg_name, cls, specs))
        config_args.append(({}, jobs, seed + s))

    raw = _run_configs_parallel(config_args, desc="E3 seeds")

    # Collapse the per-seed lifetime records into one list per (rate, alg) so
    # _save_csv's aggregate_metrics reports across-seed mean ± std.
    results: Dict[str, List[EpochMetrics]] = {}
    for rate in fault_rates:
        for alg_name, _cls in alg_classes:
            results[f"{alg_name}@fault={rate:.2f}"] = [
                m for s in range(n_seeds)
                for m in raw[f"{alg_name}@fault={rate:.2f}#s{s}"]
            ]

    _save_csv("E3_fault_intensity", results)
    return results


# ── E4: Scalability (constellation size 10–100 sats) ─────────────────────────

_E4_CONFIGS = {
    12: (3, 4),   # 3 planes × 4 sats: enough inter-plane contact density
    24: (4, 6),   # 4 planes × 6 sats
    36: (6, 6),   # 6 planes × 6 sats (matches E1 baseline)
    60: (6, 10),  # 6 planes × 10 sats
}


def run_E4(seed=0, n_seeds=16) -> Dict[str, List[EpochMetrics]]:
    print(f"E4: Scalability sweep ({n_seeds} seeds)")

    # Sim is rebuilt per (constellation size, seed); each config worker chains
    # the 2 algorithms behind one build. With 32 configs the cores stay
    # saturated, so chaining (which halves the builds) beats per-alg splitting.
    alg_classes = [("ORDI", ORDIScheduler), ("B5_seco_like", SECOLike)]
    sizes = [12, 24, 36, 60]
    config_args = []
    for n_sats in sizes:
        planes, per_plane = _E4_CONFIGS[n_sats]
        for s in range(n_seeds):
            jobs = [(f"{alg_name}@n={n_sats}#s{s}", alg_name, cls)
                    for alg_name, cls in alg_classes]
            config_args.append(
                ({"n_planes": planes, "sats_per_plane": per_plane}, jobs, seed + s))

    raw = _run_configs_parallel(config_args, desc="E4 size×seed")

    results: Dict[str, List[EpochMetrics]] = {}
    for n_sats in sizes:
        for alg_name, _cls in alg_classes:
            results[f"{alg_name}@n={n_sats}"] = [
                m for s in range(n_seeds)
                for m in raw[f"{alg_name}@n={n_sats}#s{s}"]
            ]

    _save_csv("E4_scalability", results)
    return results


# ── E5: Deadline tightness sweep ──────────────────────────────────────────────

def run_E5(seed=0, n_seeds=16) -> Dict[str, List[EpochMetrics]]:
    print(f"E5: Deadline tightness sweep (log-normal σ=0.6, {n_seeds} seeds)")

    # Sweep the deadline_slack scale.  At scale=600 (reference), per-type medians
    # are wildfire→300 s, change→480 s, ship→600 s, cloud_filter→1200 s.
    # Smaller scales compress all medians proportionally, increasing miss rate.
    # Uses the E1 stressed setup (6×4 Walker, 25° GS elevation) — under the
    # default 5° regime downlink windows are so plentiful that deadline
    # tightness never bites and the sweep is flat.
    # Sim is rebuilt per (scale, seed) → one config worker each, all concurrent.
    alg_classes = [("ORDI", ORDIScheduler),
                   ("B2_onboard_only", OnboardOnly),
                   ("B4_serval_like", ServalLike)]
    slacks = [150, 300, 450, 600, 900]
    config_args = []
    for slack in slacks:
        for s in range(n_seeds):
            jobs = [(f"{alg_name}@slack={slack}s#s{s}", alg_name, cls)
                    for alg_name, cls in alg_classes]
            config_args.append(
                ({**_E1_BUILD_KWARGS, "deadline_slack": float(slack)},
                 jobs, seed + s))

    raw = _run_configs_parallel(config_args, desc="E5 scale×seed")

    results: Dict[str, List[EpochMetrics]] = {}
    for slack in slacks:
        for alg_name, _cls in alg_classes:
            results[f"{alg_name}@slack={slack}s"] = [
                m for s in range(n_seeds)
                for m in raw[f"{alg_name}@slack={slack}s#s{s}"]
            ]

    _save_csv("E5_deadline", results)
    return results


# ── E6: λ_R penalty sweep ─────────────────────────────────────────────────────

def run_E6(seed=0, n_seeds=16) -> Dict[str, List[EpochMetrics]]:
    print(f"E6: λ_R (replication penalty) sweep ({n_seeds} seeds)")

    # Uses the stressed E1 scenario so the penalty has consequences to trade
    # against; the interesting output is how λ_R throttles backup placement
    # (n_replicas_avg) and what that costs in utility.
    lambda_Rs = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
    config_args = []
    for s in range(n_seeds):
        jobs = [(f"ORDI@lambda_R={l}#s{s}", "ORDI", ORDIScheduler,
                 None, {"lambda_R": l})
                for l in lambda_Rs]
        config_args.append((dict(_E1_BUILD_KWARGS), jobs, seed + s))

    raw = _run_configs_parallel(config_args, desc="E6 seeds")

    results: Dict[str, List[EpochMetrics]] = {}
    for l in lambda_Rs:
        results[f"ORDI@lambda_R={l}"] = [
            m for s in range(n_seeds) for m in raw[f"ORDI@lambda_R={l}#s{s}"]
        ]

    _save_csv("E6_lambda_R", results)
    return results


# ── E7: Correlated failures (orbital-plane outage) ────────────────────────────

def run_E7(seed=0, n_seeds=16) -> Dict[str, List[EpochMetrics]]:
    """
    Correlated orbital-plane outages probing replica-placement quality.

    Sustained outage (epochs 10-50, matching E2's severity) hits one plane or
    two adjacent planes; the affected plane is swept over all positions so the
    results cover planes that host many primaries and planes that host few,
    and the whole sweep is repeated across environment seeds.

    Algorithms: ORDI, B6 (full replication), B7 (random replication).
    Differences isolate how much backup placement and count buy under
    correlated failure.  (Measured property, stated rather than ablated:
    ORDI's greedy scoring already places 100% of backups in a different
    orbital plane than the primary in this constellation.)
    """
    N_PLANES = 6  # default _build_sim constellation is a 6-plane Walker
    print(f"E7: Correlated plane outages (placement quality, {n_seeds} seeds)")
    alg_classes = [("ORDI", ORDIScheduler),
                   ("B6_full_replication", FullReplication),
                   ("B7_random_replication", RandomReplication)]

    # scale → list of affected-plane tuples covering all positions
    scenarios = {
        "1plane":  [(p,) for p in range(N_PLANES)],
        "2planes": [(p, p + 1) for p in range(0, N_PLANES, 2)],
    }

    config_args = []
    for s in range(n_seeds):
        jobs = []
        for label, plane_sets in scenarios.items():
            for planes in plane_sets:
                spec = [("plane_outage", 10, 40, planes)]
                for alg_name, cls in alg_classes:
                    key = f"{alg_name}@{label}#p{planes[0]}s{s}"
                    jobs.append((key, alg_name, cls, spec))
        config_args.append(({}, jobs, seed + s))

    raw = _run_configs_parallel(config_args, desc="E7 seeds")

    # Collapse over plane positions and seeds → mean ± std per (alg, scale).
    results: Dict[str, List[EpochMetrics]] = {}
    for alg_name, _cls in alg_classes:
        for label, plane_sets in scenarios.items():
            results[f"{alg_name}@{label}"] = [
                m for s in range(n_seeds) for planes in plane_sets
                for m in raw[f"{alg_name}@{label}#p{planes[0]}s{s}"]
            ]

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

    # Same stateful lifetime accounting as E1-E7 so E8 is directly comparable.
    greedy_states = deepcopy(states)
    greedy_sched = ORDIScheduler(cfg, sat_ids, gs_names, graphs,
                                 greedy_states, deepcopy(reliability))
    def greedy_fn(ep, td):
        return greedy_sched.schedule_epoch(ep, T_SIM_START, td)

    def ilp_fn(ep, td):
        r = solve_ilp(ep, T_SIM_START, td, graphs, deepcopy(states),
                      deepcopy(reliability), gs_names, cfg, time_limit_s=30.0)
        return r if r else SchedulerResult(
            epoch=ep, assignments=[], total_utility=0.0, energy_penalty=0.0,
            comm_penalty=0.0, rep_penalty=0.0, objective=0.0, link_utilization={})

    m_greedy = _simulate_stateful(greedy_fn, tasks, sat_ids, greedy_states, cfg,
                                  reliability=deepcopy(reliability), realized_seed=seed)
    m_ilp = _simulate_stateful(ilp_fn, tasks, sat_ids, deepcopy(states), cfg,
                               reliability=deepcopy(reliability), realized_seed=seed)

    results = {"ORDI_greedy": [m_greedy], "ORDI_ILP": [m_ilp]}
    _save_csv("E8_ilp_gap", results)
    return results


# ── COTS measurement-backed evaluation ────────────────────────────────────────

def run_COTS(seed=0, n_seeds=60) -> Dict[str, List[EpochMetrics]]:
    """Evaluate ORDI with BUPT-1 Atlas 200DK measurements from MobiCom24.

    This keeps the E1 orbital/task setup (and its 30-seed protocol) fixed and
    replaces the generic Jetson-class satellite parameters with
    measurement-derived Atlas 200DK-B power, battery, solar, and effective
    throughput values.
    """
    print(f"COTS: MobiCom24/BUPT-1 Atlas 200DK payload model (E1 scenario, {n_seeds} seeds)")
    profile = load_cots_measurement_profile()
    print(f"  Loading SatelliteCOTS logs from {profile.source_root}")
    print(f"  Atlas log: {profile.inference_log}")
    print(f"  Measured payload: {profile.compute_rate_gflops:.2f} GFLOP/s, "
          f"idle={profile.idle_power_w:.2f} W, active={profile.active_power_w:.2f} W, "
          f"battery={profile.battery_wh:.1f} Wh")
    build_kwargs = {**_E1_BUILD_KWARGS,
                    "satellite_params_factory": atlas_200dk_bupt1_params}
    alg_classes = [("ORDI", ORDIScheduler)] + [(c.name, c) for c in ALL_BASELINES]

    config_args = []
    for s in range(n_seeds):
        jobs = [(f"{alg}#s{s}", alg, cls) for alg, cls in alg_classes]
        config_args.append((build_kwargs, jobs, seed + s))

    raw = _run_configs_parallel(config_args, desc="COTS seeds")

    results: Dict[str, List[EpochMetrics]] = {}
    for alg, _cls in alg_classes:
        results[alg] = [m for s in range(n_seeds) for m in raw[f"{alg}#s{s}"]]

    _save_csv("COTS_mobicom24", results)
    return results


# ── master runner ─────────────────────────────────────────────────────────────

ALL_EXPERIMENTS = {
    "E1": run_E1, "E2": run_E2, "E3": run_E3, "E4": run_E4,
    "E5": run_E5, "E6": run_E6, "E7": run_E7, "E8": run_E8,
    "COTS": run_COTS,
}


def run_all(seed=0):
    for exp_id, fn in tqdm(ALL_EXPERIMENTS.items(), desc="Experiments", unit="exp"):
        print(f"\n{'='*50}\n{exp_id}\n{'='*50}")
        fn(seed=seed)
