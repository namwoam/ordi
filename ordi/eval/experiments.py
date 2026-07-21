"""
Focused experiment runner for E1–E4.

Each experiment returns a dict: {algorithm_name: List[EpochMetrics]}.
Results are saved to results/<experiment_id>.csv.

The evaluation is a notional next-generation architecture, not a reproduction
of one operator's deployed system. It composes a PlanetScope/SuperDove-class EO
workload with Kepler-class optical edge compute, an SDA-PWSA-inspired
fault-tolerant transport topology, and the autonomous scheduling direction
represented by ESA TASCNET.

Shared realistic LEO-EO simulation setup (all experiments unless overridden):
  - Synthetic Walker constellation: 6 planes × 6 sats = 36 sats (default).
    E1 uses a 3 × 12 scaled near-polar mesh so fore/aft neighbors remain
    mutually visible at the modeled optical-terminal range.
  - 6 geographically distributed ground stations: Fairbanks, Greenwich,
    Singapore, Nairobi, Hawaii, and Punta Arenas.
  - E1 acquisition events construct feasible near-nadir PlanetScope-class
    regions along sampled ground tracks. Other experiments retain the legacy
    fixed-target access generator.
  - Orbital period: 5760 s (96 min) — correct for 550 km LEO altitude.
  - Simulation horizon: 17 280 s (3 orbits); 288 × 60 s epochs.
  - E1 tasks: mean 20 requests/orbit in a clustered hot-source process. Sixty
    percent of parent events generate 3–6 same-source requests within 60 s.
  - E1 expands one same-area burst per orbit to ten requests within 30 s and
    adds 15% recurring background accelerator demand.
  - Each request is a 4096×4096 PlanetScope-class inference ROI split into
    sixteen 1024×1024 tiles, with workload-specific spectral band counts.
  - Deadlines are log-normal (σ=0.6), with medians wildfire 600 s, ship
    900 s, change 1800 s, and cloud_filter 5760 s (one orbit).
"""

from __future__ import annotations
import csv
import hashlib
import math
import os
import random
import time
from dataclasses import replace
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
from ordi.sim.satellite import SatelliteParams, make_constellation_states
from ordi.sim.reliability import ReliabilityModel
from ordi.tasks.generator import generate_tasks
from ordi.algorithms import (
    ORDI, DirectDownlink, OnboardOnly, SECOAdapted,
    FullReplication, RandomReplication, EpochInput, SatelliteView,
    ContactWindow, PolicyWeights, ExperimentConfig, Assignment, Decision,
)
ORDIConfig = ExperimentConfig
CORE_BASELINES = [DirectDownlink, OnboardOnly, SECOAdapted, FullReplication]
from ordi.faults.injector import FaultInjector, random_fault_schedule, FaultEvent
from ordi.eval.metrics import (
    compute_metrics, compute_realized_metrics, aggregate_metrics, EpochMetrics,
)
from ordi.eval.validation import DecisionFeasibilityModel, InvalidDecisionError
from ordi.algorithms._common import earliest_route

RESULTS_DIR = "results"
SIM_DURATION_S = 17_280       # 3 complete orbits at 550 km (3 × 5760 s)
EPOCH_LENGTH_S = 60.0
N_EPOCHS = int(SIM_DURATION_S / EPOCH_LENGTH_S)
T_SIM_START = 0.0

# Six geographically distributed stations avoid making relay policies win
# merely because the direct baseline has an artificially sparse ground segment.
_EVALUATION_GS = [
    gs for gs in DEFAULT_GROUND_STATIONS
    if gs[0] in {
        "fairbanks", "greenwich", "singapore",
        "nairobi", "hawaii", "punta_arenas",
    }
]


def _worker_count(n_jobs: int) -> int:
    """Bound process concurrency to avoid memory/CPU starvation in orbit builds."""
    configured = int(os.environ.get("ORDI_MAX_WORKERS", "8"))
    return max(1, min(n_jobs, configured))


# ── simulation bootstrap ──────────────────────────────────────────────────────

def _intensify_one_area_burst(tasks, compute_multiplier,
                              request_count=None, window_s=None):
    """Expand and amplify one same-area burst for compute contention.

    Any added requests copy an existing request's tiles, preserving data sizes,
    task type, source satellite, and deadline slack.  Only compute demand is
    multiplied.  Release times can optionally be compressed into ``window_s``.
    """
    if compute_multiplier <= 0.0:
        raise ValueError("compute_multiplier must be positive")
    if (compute_multiplier == 1.0 and request_count is None) or not tasks:
        return None
    grouped = {}
    for task in tasks:
        grouped.setdefault(task.burst_id, []).append(task)
    candidates = [
        group for group in grouped.values() if len(group) > 1
    ] or list(grouped.values())

    def pressure(group):
        work = sum(
            tile.compute_ops for task in group for tile in task.tiles
        )
        slack = min(max(task.deadline_slack, 60.0) for task in group)
        return work / slack

    selected = max(
        candidates,
        key=lambda group: (
            pressure(group), len(group), -group[0].burst_id
        ),
    )

    if request_count is not None:
        if request_count < len(selected):
            raise ValueError(
                "intense_area_request_count cannot shrink the selected burst"
            )
        next_task_id = max(task.task_id for task in tasks) + 1
        templates = list(selected)
        while len(selected) < request_count:
            template = templates[(len(selected) - len(templates))
                                 % len(templates)]
            clone = deepcopy(template)
            clone.task_id = next_task_id
            for tile in clone.tiles:
                tile.task_id = next_task_id
            tasks.append(clone)
            selected.append(clone)
            next_task_id += 1

    if window_s is not None:
        if window_s < 0.0:
            raise ValueError("intense_area_window_s must be non-negative")
        release_start = min(task.release_time for task in selected)
        denominator = max(len(selected) - 1, 1)
        for index, task in enumerate(selected):
            deadline_slack = task.deadline_slack
            task.release_time = release_start + window_s * index / denominator
            task.deadline = task.release_time + deadline_slack

    for task in selected:
        for tile in task.tiles:
            tile.compute_ops *= compute_multiplier
        task.intense_area = True
    return selected[0].burst_id, len(selected)


def _intensify_repeated_area_bursts(
    tasks, compute_multiplier, request_count, window_s,
    orbit_period_s, bursts_per_orbit,
):
    """Expand high-pressure bursts independently in each orbital interval."""
    if bursts_per_orbit <= 0 or not tasks:
        return []
    selected = []
    orbit_count = max(
        1, int(math.ceil(max(task.release_time for task in tasks)
                         / orbit_period_s)),
    )
    for orbit_index in range(orbit_count):
        start = orbit_index * orbit_period_s
        end = start + orbit_period_s
        period_tasks = [
            task for task in tasks if start <= task.release_time < end
        ]
        for _ in range(bursts_per_orbit):
            available = [
                task for task in period_tasks
                if task.burst_id not in {item[0] for item in selected}
            ]
            if not available:
                break
            before_objects = {id(task) for task in available}
            result = _intensify_one_area_burst(
                available, compute_multiplier,
                request_count=request_count, window_s=window_s,
            )
            if result is None:
                break
            # The helper expands its supplied list. Move clones into the full
            # task list and assign globally unique identifiers.
            for clone in available:
                if id(clone) in before_objects:
                    continue
                clone.task_id = max(task.task_id for task in tasks) + 1
                for tile in clone.tiles:
                    tile.task_id = clone.task_id
                tasks.append(clone)
                period_tasks.append(clone)
            selected.append(result)
    return selected


def _four_neighbor_walker_pairs(n_planes, sats_per_plane):
    """Fore/aft plus adjacent-plane optical neighbors for a Walker mesh."""
    pairs = set()
    for plane in range(n_planes):
        for slot in range(sats_per_plane):
            here = f"SAT_{plane:02d}_{slot:02d}"
            neighbors = {
                f"SAT_{plane:02d}_{(slot - 1) % sats_per_plane:02d}",
                f"SAT_{plane:02d}_{(slot + 1) % sats_per_plane:02d}",
                f"SAT_{(plane - 1) % n_planes:02d}_{slot:02d}",
                f"SAT_{(plane + 1) % n_planes:02d}_{slot:02d}",
            }
            pairs.update(frozenset((here, neighbor)) for neighbor in neighbors)
    return pairs


def _build_sim(n_planes=6, sats_per_plane=6, seed=0, deadline_slack=600.0,
               deadline_lognorm_sigma=0.6,
               arrival_rate=16.0, ground_stations=None,
               orbit_period_s=5760.0,
               burst_probability=0.5, burst_size_range=(2, 4),
               burst_window_s=60.0,
               intense_area_compute_multiplier=1.0,
               intense_area_request_count=None,
               intense_area_window_s=None,
               use_fov=True, fov_range_km=600.0, n_targets=100,
               min_elevation_deg=25.0, satellite_params_factory=None,
               reliability_model_factory=None,
               orbit_altitude_km=550.0, orbit_inclination_deg=53.0,
               isl_topology="range",
               acquisition_mode="targets", input_band_counts=None,
               intense_bursts_per_orbit=0,
               background_compute_utilization=0.0,
               n_replicas_max=2):
    """Build all shared simulation objects.

    orbit_period_s        : realistic LEO period at 550 km altitude (~5760 s / 96 min).
    burst_probability     : share of parent events that create hot-source bursts.
    burst_size_range      : inclusive request-count range for a burst.
    burst_window_s        : release-time spread within a burst.
    intense_area_compute_multiplier:
                            multiplier applied to one highest-pressure clustered
                            burst; 1.0 disables the compute hotspot.
    intense_area_request_count:
                            optional exact request count for the selected hotspot.
    intense_area_window_s:  optional release-time spread for the selected hotspot.
    deadline_slack        : global deadline scale (reference = 600 s).  Per-type medians
                            (wildfire 600 s, ship 900 s, change 1800 s,
                            cloud_filter 5760 s)
                            are multiplied by deadline_slack / 600.
    deadline_lognorm_sigma: log-space std-dev for per-task deadline sampling (σ=0.6).
    use_fov               : FOV-constrained task generation — tasks arise only when a
                            satellite is within fov_range_km of a ground target.
    fov_range_km          : camera footprint radius (600 km ≈ ±42° off-nadir at 550 km
                            altitude; realistic for a wide-field EO imager).
    n_targets             : number of random ground targets (uniformly in ±60° lat).
    reliability_model_factory:
                            optional zero-argument factory for the link/node
                            reliability model used by policies and realized
                            delivery trials.
    n_replicas_max        : hard per-tile cap on total replicas.

    ``acquisition_mode='groundtrack'`` defines a feasible near-nadir ROI at a
    sampled satellite subpoint. ``isl_topology='four_neighbor'`` limits the
    range/line-of-sight contact plan to fore/aft and adjacent-plane pairs.
    ``background_compute_utilization`` is recurring physical queued work.
    """
    import random as _rng_mod
    if ground_stations is None:
        ground_stations = _EVALUATION_GS
    sats = build_synthetic_walker(
        n_planes=n_planes, sats_per_plane=sats_per_plane,
        alt_km=orbit_altitude_km, inc_deg=orbit_inclination_deg,
    )
    sat_ids = [s.name for s in sats]
    gs_names = {gs[0] for gs in ground_stations}

    print(f"  Computing contact windows for {len(sats)} sats × {len(ground_stations)} GS ...")
    t0 = time.time()
    isl_pairs = (
        _four_neighbor_walker_pairs(n_planes, sats_per_plane)
        if isl_topology == "four_neighbor" else None
    )
    if isl_topology not in {"range", "four_neighbor"}:
        raise ValueError("isl_topology must be 'range' or 'four_neighbor'")
    contacts = compute_contact_windows(
        sats,
        t_start_unix=0.0,
        t_end_unix=SIM_DURATION_S,
        dt_seconds=60.0,
        ground_stations=ground_stations,
        min_elevation_deg=min_elevation_deg,
        isl_pairs=isl_pairs,
    )
    print(f"  {len(contacts)} contact events in {time.time()-t0:.1f}s")

    graphs = build_epoch_graphs(contacts, T_SIM_START, EPOCH_LENGTH_S, N_EPOCHS)
    states = make_constellation_states(
        sat_ids, seed=seed, params_factory=satellite_params_factory
    )
    reliability = (
        reliability_model_factory()
        if reliability_model_factory is not None
        else ReliabilityModel()
    )

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
        burst_probability=burst_probability,
        burst_size_range=burst_size_range,
        burst_window_s=burst_window_s,
        deadline_slack_s=deadline_slack,
        deadline_lognorm_sigma=deadline_lognorm_sigma,
        seed=seed,
        sat_groundtrack=sat_groundtrack,
        ground_targets=ground_targets,
        fov_range_km=fov_range_km,
        acquisition_mode=acquisition_mode,
        input_band_counts=input_band_counts,
        n_replicas_max=n_replicas_max,
    )
    if intense_bursts_per_orbit:
        intense = _intensify_repeated_area_bursts(
            tasks, intense_area_compute_multiplier,
            intense_area_request_count, intense_area_window_s,
            orbit_period_s, intense_bursts_per_orbit,
        )
    else:
        one = _intensify_one_area_burst(
            tasks, intense_area_compute_multiplier,
            request_count=intense_area_request_count,
            window_s=intense_area_window_s,
        )
        intense = [one] if one is not None else []
    if intense:
        print(
            f"  Intensified {len(intense)} area bursts: "
            f"{intense_area_request_count} requests each at "
            f"{intense_area_compute_multiplier:.1f}× compute"
        )
    cfg = ORDIConfig(
        epoch_length=EPOCH_LENGTH_S,
        background_compute_utilization=background_compute_utilization,
    )

    return sats, sat_ids, gs_names, contacts, graphs, states, reliability, tasks, cfg


# ── stateful rolling-horizon helpers ──────────────────────────────────────────

def _assignment_group_viability(a: Assignment, states):
    """Return reconstruction-group viability before delivery completes."""
    if not a.helpers:
        source = states.get(a.source)
        alive = source is not None and bool(source.A_i)
        return {0: alive}
    viable = []
    for helper, aggregator in zip(a.helpers, a.aggregators):
        h = states.get(helper)
        g = states.get(aggregator)
        viable.append(bool(
            h is not None and g is not None and h.A_i and g.A_i
        ))
    required = max(1, int(a.metadata.get("data_shards", 1)))
    shard_groups = a.metadata.get("shard_groups")
    if shard_groups is None:
        return {0: sum(viable) >= required}
    if len(shard_groups) != len(viable):
        return {}
    grouped = {}
    for label, alive in zip(shard_groups, viable):
        grouped.setdefault(label, []).append(alive)
    return {
        label: len(group) >= required and sum(group) >= required
        for label, group in grouped.items()
    }


def _assignment_viable(a: Assignment, states, sim_time=None) -> bool:
    """In-flight tile survives if any replica's helper+aggregator are both alive
    (a surviving backup avoids re-transmission). Completed deliveries are final;
    an unfinished direct downlink still depends on its source satellite."""
    delivery_time = float(a.metadata.get("delivery_time", math.inf))
    if sim_time is not None and delivery_time <= sim_time + 1e-9:
        return True
    if not a.helpers:
        source_release_time = float(a.metadata.get(
            "source_release_time", delivery_time
        ))
        if (sim_time is not None
                and source_release_time <= sim_time + 1e-9):
            return True
        source = states.get(a.source)
        return source is not None and bool(source.A_i)
    return any(_assignment_group_viability(a, states).values())


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


def _advance_synthetic_states(assignments, tasks, states, epoch_length_s,
                              protocol_events=()):
    """Apply newly committed assignment load to every satellite state.

    Assignment energy records are converted back to compute work using the
    helper's current rate and compute power.  Communication load includes the
    source-to-helper input, helper-to-aggregator result, and aggregator-to-ground
    output.  Idle satellites are also advanced so solar harvest, platform idle
    power, and thermal relaxation apply constellation-wide.

    Real-telemetry experiments do not call this helper: their state driver is
    authoritative and overwrites B_i/Theta_i/C_i/A_i each epoch.
    """
    tile_by_key = {
        (task.task_id, tile.tile_id): (task.source_sat, tile)
        for task in tasks
        for tile in task.tiles
    }
    compute_work = {sat_id: 0.0 for sat_id in states}
    isl_tx_bits = {sat_id: 0.0 for sat_id in states}
    rx_bits = {sat_id: 0.0 for sat_id in states}
    downlink_bits = {sat_id: 0.0 for sat_id in states}

    for assignment in assignments:
        source_and_tile = tile_by_key.get((assignment.task_id, assignment.tile_id))
        if source_and_tile is None:
            continue
        source, tile = source_and_tile

        # Direct-downlink assignments have no compute replica.
        if not assignment.helpers:
            bits = assignment.metadata.get("downlink_bits")
            if bits is not None and source in downlink_bits:
                downlink_bits[source] += float(bits)
            continue

        for index, (helper, aggregator) in enumerate(
                zip(assignment.helpers, assignment.aggregators)):
            helper_state = states.get(helper)
            if helper_state is None:
                continue

            work_fraction = (assignment.work_fractions[index]
                             if index < len(assignment.work_fractions) else 1.0)
            input_fraction = (assignment.input_fractions[index]
                              if index < len(assignment.input_fractions) else 1.0)
            output_fraction = (assignment.output_fractions[index]
                               if index < len(assignment.output_fractions) else 1.0)
            protocol_header = float(
                assignment.metadata.get("protocol_header_bits", 0.0)
            )
            per_helper_work = float(assignment.metadata.get(
                "compute_flops", tile.compute_ops * work_fraction
            ))
            compute_work[helper] += per_helper_work

            bits = float(assignment.metadata.get(
                "downlink_bits", tile.d_out_bits * output_fraction
            )) + protocol_header
            if index < len(assignment.routes):
                route_in, route_out, route_down = assignment.routes[index]
                for path, path_bits in (
                        (route_in, tile.d_in_bits * input_fraction
                         + protocol_header),
                        (route_out, tile.d_out_bits * output_fraction
                         + protocol_header)):
                    for sender, receiver in zip(path, path[1:]):
                        if sender in isl_tx_bits:
                            isl_tx_bits[sender] += path_bits
                        if receiver in rx_bits:
                            rx_bits[receiver] += path_bits
                for sender, receiver in zip(route_down, route_down[1:]):
                    if receiver not in states:
                        if sender in downlink_bits:
                            downlink_bits[sender] += bits
                    else:
                        if sender in isl_tx_bits:
                            isl_tx_bits[sender] += bits
                        rx_bits[receiver] += bits
            else:
                if source != helper and source in isl_tx_bits:
                    isl_tx_bits[source] += (
                        tile.d_in_bits * input_fraction + protocol_header
                    )
                    rx_bits[helper] += (
                        tile.d_in_bits * input_fraction + protocol_header
                    )
                if helper != aggregator and aggregator in states:
                    isl_tx_bits[helper] += (
                        tile.d_out_bits * output_fraction + protocol_header
                    )
                    rx_bits[aggregator] += (
                        tile.d_out_bits * output_fraction + protocol_header
                    )
                if aggregator in downlink_bits:
                    downlink_bits[aggregator] += bits

        for event in assignment.message_events:
            if (event.event == "hop_sent"
                    and event.kind in {
                        "split_request", "split_accept", "split_reject",
                        "replica_request", "replica_accept",
                        "replica_reject",
                    }):
                if event.node in isl_tx_bits:
                    isl_tx_bits[event.node] += event.bits
                if event.peer in rx_bits:
                    rx_bits[event.peer] += event.bits

    # Resource advertisements occur even in epochs without science work.
    for event in protocol_events:
        if (event.kind == "state_advertisement"
                and event.event == "hop_sent"):
            if event.node in isl_tx_bits:
                isl_tx_bits[event.node] += event.bits
            if event.peer in rx_bits:
                rx_bits[event.peer] += event.bits

    # Physical evolution is deliberately delegated to Basilisk/BSK-RL.  Keep
    # this function as a workload translator for callers that own a backend.
    from ordi.sim.basilisk_backend import Workload
    return {
        sid: Workload(compute_flops=compute_work[sid], tx_bits=isl_tx_bits[sid],
                      rx_bits=rx_bits[sid], downlink_bits=downlink_bits[sid])
        for sid in states
    }


def _simulate_stateful(schedule_fn, tasks, sat_ids, states, cfg, injector=None,
                       reliability=None, realized_trials=500, realized_seed=0,
                       state_driver=None, outcome_callback=None):
    """Run one stateful rolling-horizon simulation and return a lifetime
    EpochMetrics.  schedule_fn(epoch, todo_tasks) -> Decision dispatches to an
    algorithm policy. A committed tile stays in-flight (not
    re-scheduled, not re-charged) until a fault invalidates all its replicas.

    When a reliability model is supplied, the final lifetime assignment set is
    also scored by Monte Carlo (compute_realized_metrics): links and nodes are
    sampled from their π values with draws shared across a tile's replicas, so
    the realized_* fields report delivery outcomes the modeled z_kv assumes
    away.  Hard outages already pruned infeasible candidates during scheduling;
    this layer adds the soft stochastic loss the reliability model specifies."""
    # Accumulate the compute capacity actually available in each epoch.  This
    # keeps helper utilization consistent with thermal/straggler rate changes
    # and excludes epochs in which a satellite is unavailable.
    sat_cap = {s: 0.0 for s in sat_ids}
    # Basilisk/BSK-RL is the sole owner of orbit, eclipse, power, battery,
    # thermal, availability, and data-state evolution for synthetic runs.
    physical_backend = None
    physical_energy_j = 0.0
    if state_driver is None:
        from ordi.sim.basilisk_backend import BasiliskBackend
        physical_backend = BasiliskBackend(
            sat_ids, states, cfg.epoch_length, seed=realized_seed
        )
    all_tiles = [(t.task_id, tile.tile_id) for t in tasks for tile in t.tiles]
    committed: Dict[Tuple[int, int], Assignment] = {}
    feedback_reported = set()
    backup_recovery = set()
    fault_impacted = set()
    rejection_causes = {}
    protocol_message_count = 0.0
    protocol_control_bits = 0.0

    for epoch in range(N_EPOCHS):
        ep_start = T_SIM_START + epoch * EPOCH_LENGTH_S
        if injector:
            injector.apply_epoch(epoch)
        # Overwrite satellite state from a real-telemetry trace (if driving the
        # real pipeline) after fault injection so injected failures still win.
        if state_driver is not None:
            state_driver(epoch, ep_start, states)
        if injector:
            injector.refresh_active_state()
        background_fraction = cfg.background_compute_utilization
        if not 0.0 <= background_fraction < 1.0:
            raise ValueError(
                "background_compute_utilization must be in [0, 1)"
            )
        for state in states.values():
            state.Q_i += (
                background_fraction * state.C_i * cfg.epoch_length
            )
        for sat_id in sat_ids:
            state = states[sat_id]
            if state.A_i:
                sat_cap[sat_id] += state.C_i * cfg.epoch_length
        # Report actual scheduled-work outcomes. Idle healthy domains are not
        # samples: only primary delivery, backup recovery, or hard failure
        # updates a learning policy.
        for key in list(committed.keys()):
            assignment = committed[key]
            delivery_time = float(assignment.metadata.get(
                "delivery_time", math.inf
            ))
            if delivery_time <= ep_start + 1e-9:
                if outcome_callback and key not in feedback_reported:
                    outcome_callback(
                        "backup_recovery" if key in backup_recovery
                        else "primary_success"
                    )
                    feedback_reported.add(key)
                continue
            groups = _assignment_group_viability(assignment, states)
            primary_alive = groups.get(0, next(iter(groups.values()), False))
            any_alive = any(groups.values())
            if not primary_alive and any_alive:
                backup_recovery.add(key)
            if not any_alive:
                if outcome_callback:
                    outcome_callback("fault_failure")
                feedback_reported.add(key)
                fault_impacted.add(key)
                del committed[key]
        pending = [t for t in tasks if t.release_time <= ep_start < t.deadline]
        todo = _uncommitted_tasks(pending, committed)
        result = schedule_fn(epoch, todo)
        rejection_causes.update(result.metadata.get("rejection_causes", {}))
        protocol_message_count += float(
            result.metadata.get("protocol_message_count", 0.0)
        )
        protocol_control_bits += float(
            result.metadata.get("advertisement_control_bits", 0.0)
        )
        newly_committed = []
        for a in result.assignments:
            reliability_estimate = float(a.metadata.get(
                "reliability", a.metadata.get("reconstruction_probability", 0.0)
            ))
            latency = float(a.metadata.get("latency", math.inf))
            if reliability_estimate > 0 and not math.isinf(latency):
                committed[(a.task_id, a.tile_id)] = a
                newly_committed.append(a)
        if state_driver is None:
            epoch_energy_j = physical_backend.submit(_advance_synthetic_states(
                newly_committed, tasks, states, cfg.epoch_length,
                protocol_events=result.message_events,
            ))
            physical_energy_j += float(epoch_energy_j or 0.0)
        if injector:
            injector.withdraw_epoch(epoch + 1)

    source_by_key = {(task.task_id, tile.tile_id): task.source_sat
                     for task in tasks for tile in task.tiles}
    final = [committed.get(k) or Assignment(k[0], k[1], source_by_key[k])
             for k in all_tiles]
    if outcome_callback:
        for key in all_tiles:
            if key not in committed and key not in feedback_reported:
                outcome_callback("nonfault_failure")
    res = Decision(
        N_EPOCHS - 1, tuple(final),
        metadata={
            "protocol_message_count": protocol_message_count,
            "advertisement_control_bits": protocol_control_bits,
        },
    )
    m = compute_metrics(
        res, tasks, 0.0, sat_cap, cfg.alpha,
        physical_energy_j=physical_energy_j,
    )
    missing = {key for key, assignment in zip(all_tiles, final)
               if not assignment.helpers and not assignment.downlink_only}
    total_tiles = max(len(all_tiles), 1)
    hard_fault_misses = missing.intersection(fault_impacted)
    compute_misses = {
        key for key in missing - hard_fault_misses
        if rejection_causes.get(key) == "compute_queue"
    }
    policy_misses = {
        key for key in missing - hard_fault_misses - compute_misses
        if rejection_causes.get(key) == "policy"
    }
    contact_misses = (
        missing - hard_fault_misses - compute_misses - policy_misses
    )
    m.hard_fault_miss_ratio = len(hard_fault_misses) / total_tiles
    m.compute_queue_miss_ratio = len(compute_misses) / total_tiles
    m.policy_miss_ratio = len(policy_misses) / total_tiles
    m.contact_miss_ratio = len(contact_misses) / total_tiles
    r_total = sum(max(0, float(a.metadata.get(
        "effective_replicas", len(a.helpers))) - 1) for a in final)
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
        m.soft_failure_miss_ratio = max(
            0.0, m.realized_miss_ratio - m.deadline_miss_ratio
        )
    return m


# ── parallel worker (module-level so multiprocessing can pickle it) ───────────

# Simulation environment shared by all jobs of one _run_parallel call, shipped
# once per worker via the pool initializer instead of pickled into every job.
_WORKER_SHARED: Optional[Tuple] = None


def _init_worker_shared(shared: Tuple) -> None:
    global _WORKER_SHARED
    _WORKER_SHARED = shared


def _epoch_input(ep, tasks, sat_ids, states, reliability, graphs, gs_names, cfg):
    """Translate Basilisk state and contact graphs into the common policy API."""
    adjacency = {sid: [] for sid in sat_ids}
    for a, b, _rate, _cap, _kind in graphs[ep].edges:
        if a in adjacency:
            adjacency[a].append(b)
    views = {sid: SatelliteView(
        sid, bool(state.A_i), state.C_i, state.B_i,
        state.params.battery_j, state.Theta_i, state.Q_i,
        reliability.node_pi(sid),
    ) for sid, state in states.items()}
    contacts = []
    # A policy cannot use a contact after every pending task has expired. The
    # old all-horizon projection made each route lookup scan the remaining
    # three-orbit graph, even for tasks due within a few minutes.
    # Always expose the current epoch's contacts so decentralized policies can
    # exchange state advertisements even when no science task is pending.
    latest_deadline = max(
        (task.deadline for task in tasks),
        default=T_SIM_START + (ep + 1) * cfg.epoch_length,
    )
    for future in graphs[ep:]:
        if future.t_start > latest_deadline:
            break
        for a, b, rate, capacity, kind in future.edges:
            duration = capacity / max(rate, 1.0)
            contacts.append(ContactWindow(
                a, b, future.t_start, future.t_start + duration,
                rate, kind, reliability.link_pi(a, b, kind),
            ))
    return EpochInput(
        ep, T_SIM_START + ep * cfg.epoch_length, tasks, views,
        adjacency, frozenset(gs_names), tuple(contacts), cfg.epoch_length,
        PolicyWeights(cfg.alpha, cfg.lambda_E, cfg.lambda_C, cfg.lambda_R),
    )


def _validate_feasible_subset(feasibility, request, decision):
    """Admit policy assignments independently through the model ledger.

    State advertisements are reserved once for the epoch. Each science
    assignment is then transactional: an invalid tile is dropped without
    rolling back earlier feasible tiles or aborting the complete experiment.
    """
    feasibility.validate_and_reserve(
        request,
        Decision(
            decision.epoch, (),
            decision.metadata, decision.message_events,
        ),
    )
    accepted = []
    for assignment in decision.assignments:
        try:
            result = feasibility.validate_and_reserve(
                request,
                Decision(decision.epoch, (assignment,)),
                retime=True,
            )
        except InvalidDecisionError:
            continue
        accepted.append(result.assignments[0])
    return Decision(
        decision.epoch, tuple(accepted),
        decision.metadata, decision.message_events,
    )


def _classify_rejected_tiles(request, decision):
    """Shared lightweight contact/queue/admission diagnosis.

    This is a lower-bound counterfactual used only for reporting: contact asks
    whether any source-to-ground result path exists; queue compares optimistic
    completion with current queues against the same zero-queue bound.
    """
    assigned = {
        (assignment.task_id, assignment.tile_id)
        for assignment in decision.assignments
    }
    causes = {}
    for task in request.tasks:
        # Earlier rejections may be retried and are not misses. Diagnose only
        # the final scheduling opportunity before this task expires.
        if task.deadline > request.sim_time + request.epoch_length + 1e-9:
            continue
        for tile in task.tiles:
            key = (task.task_id, tile.tile_id)
            if key in assigned:
                continue
            source = request.satellites.get(task.source_sat)
            if source is None or not source.available:
                causes[key] = "fault"
                continue
            route = earliest_route(
                request, task.source_sat, request.ground_stations,
                tile.d_out_bits,
            )
            if route is None or route.arrival > task.deadline:
                causes[key] = "contact"
                continue
            available = [
                state for state in request.satellites.values()
                if state.available
            ]
            zero_compute = min(
                (tile.compute_ops / max(state.compute_rate, 1.0)
                 for state in available),
                default=math.inf,
            )
            queued_compute = min(
                ((state.queued_flops + tile.compute_ops)
                 / max(state.compute_rate, 1.0)
                 for state in available),
                default=math.inf,
            )
            zero_finish = max(
                route.arrival, request.sim_time + zero_compute
            )
            queued_finish = max(
                route.arrival, request.sim_time + queued_compute
            )
            causes[key] = (
                "compute_queue"
                if (zero_finish <= task.deadline + 1e-9
                    and queued_finish > task.deadline + 1e-9)
                else "policy"
            )
    metadata = dict(decision.metadata)
    metadata["rejection_causes"] = causes
    return replace(decision, metadata=metadata)


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
    mutates_graph = any(f.fault_type in _GRAPH_MUTATING for f in (faults or []))
    local_graphs = deepcopy(graphs) if mutates_graph else graphs

    injector = None
    if faults:
        injector = FaultInjector(local_states, local_rel, [], rng_seed=seed,
                                 graphs=local_graphs, gs_names=gs_names)
        for f in faults:
            injector.schedule(f)

    if scheduler_class is ORDI:
        sched = scheduler_class(
            max_replicas=cfg.max_backups + 1,
            split_options=cfg.ordi_split_options,
            coded_options=cfg.ordi_coded_options,
            halo_fraction=cfg.split_halo_fraction,
            rng_seed=seed,
        )
    elif scheduler_class is SECOAdapted:
        sched = scheduler_class(
            split_options=cfg.seco_split_options,
            halo_fraction=cfg.split_halo_fraction,
        )
    else:
        try:
            sched = scheduler_class(seed=seed)
        except TypeError:
            sched = scheduler_class()

    feasibility = DecisionFeasibilityModel()

    def schedule_fn(ep, td):
        request = _epoch_input(
            ep, td, sat_ids, local_states, local_rel, local_graphs, gs_names, cfg
        )
        decision = sched.schedule(request)
        try:
            validated = _validate_feasible_subset(
                feasibility, request, decision
            )
            return _classify_rejected_tiles(request, validated)
        except InvalidDecisionError as error:
            raise InvalidDecisionError(
                f"{sched.name} submitted an invalid decision in epoch {ep}: "
                f"{error}"
            ) from error

    m = _simulate_stateful(
        schedule_fn, tasks, sat_ids, local_states, cfg, injector,
        reliability=local_rel, realized_seed=seed,
        outcome_callback=getattr(sched, "observe_assignment_outcome", None),
    )
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
    n_workers = _worker_count(n)
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
    """Worker for synthetic sweeps that rebuild the sim per configuration:
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
    if _worker_count(n) == 1:
        for args in tqdm(config_args, desc=desc, unit="config"):
            for key, metrics in _build_and_run_config(args):
                results[key] = metrics
        return results
    with ProcessPoolExecutor(max_workers=_worker_count(n)) as pool:
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


def _save_csv(exp_id: str, results: Dict[str, List[EpochMetrics]],
              metric_keys: Optional[List[str]] = None):
    """Write aggregate results, optionally restricted to comparable metrics."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{exp_id}.csv")
    keys = _CSV_METRIC_KEYS if metric_keys is None else metric_keys
    fields = ["algorithm"] + keys + [f"{key}_std" for key in keys]
    rows = []
    for alg_name, metrics in results.items():
        agg = aggregate_metrics(metrics)
        row = {"algorithm": alg_name, **agg}
        rows.append(row)
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields,
                                extrasaction="ignore", restval="")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {path}")


# ── E1: Core performance (ORDI vs all baselines) ─────────────────────────────

# The composed reference scenario uses PlanetScope-class acquisitions on a
# degree-limited optical edge-compute mesh. Rates are sustained workload
# throughput assumptions, not accelerator nameplate specifications.
def _e1_satellite_params(sat_id):
    digest = hashlib.sha256(sat_id.encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    rate = min(8.0, max(3.0, rng.lognormvariate(math.log(5.0), 0.25)))
    return SatelliteParams(sat_id=sat_id, compute_rate_gflops=rate)


def _e1_reliability_model():
    """Clear-sky reliability assumptions for compute-placement E1.

    Fault robustness remains covered by the matched 2% random schedule and
    by E2/E3.  E1 uses high-availability links and nodes so stochastic delivery
    loss does not dominate the compute-placement comparison.
    """
    return ReliabilityModel(
        default_isl_pi=0.995,
        default_downlink_pi=0.98,
        default_node_pi=0.999,
    )


_E1_BUILD_KWARGS = dict(n_planes=3, sats_per_plane=12,
                        orbit_altitude_km=475.0,
                        orbit_inclination_deg=97.4,
                        orbit_period_s=5670.0,
                        arrival_rate=20.0, deadline_slack=600.0,
                        deadline_lognorm_sigma=0.6,
                        burst_probability=0.6,
                        burst_size_range=(3, 6), burst_window_s=60.0,
                        intense_area_request_count=10,
                        intense_area_compute_multiplier=1.0,
                        intense_area_window_s=30.0,
                        intense_bursts_per_orbit=1,
                        ground_stations=DEFAULT_GROUND_STATIONS,
                        min_elevation_deg=10.0,
                        isl_topology="four_neighbor",
                        acquisition_mode="groundtrack",
                        fov_range_km=16.25,
                        input_band_counts={
                            "ship": 3, "wildfire": 4,
                            "change": 8, "cloud_filter": 8,
                        },
                        background_compute_utilization=0.15,
                        satellite_params_factory=_e1_satellite_params,
                        reliability_model_factory=_e1_reliability_model)

E1_METRIC_KEYS = [
    "realized_miss_ratio",
    "contact_miss_ratio",
    "compute_queue_miss_ratio",
    "policy_miss_ratio",
    "hard_fault_miss_ratio",
    "soft_failure_miss_ratio",
    "delivery_latency_p50_s",
    "delivery_latency_p95_s",
    "isl_traffic_bits_per_delivered_tile",
    "control_traffic_bits_per_delivered_tile",
    "control_traffic_ratio",
    "protocol_messages_per_delivered_tile",
    "energy_j_per_delivered_tile",
    "downlink_bits_per_delivered_tile",
    "helper_utilization",
    "active_helper_fraction",
    "compute_load_balance",
    "helper_request_count",
    "helper_acceptance_ratio",
    "state_age_mean_s",
    "state_age_p95_s",
]

def run_E1(seed=0, n_seeds=8,
           fault_rate=0.02) -> Dict[str, List[EpochMetrics]]:
    """
    Core performance comparison using the shared realistic LEO-EO setup.

    Uses a scaled 3×12, 475 km near-polar Walker testbed. Twelve satellites
    per plane keep fore/aft neighbors within the 4,000 km optical range; using
    four satellites per plane would make those nominal links invisible.
    with a mean of 20 requests per orbit. Sixty percent of parent events create
    3–6 same-source requests within 60 s, producing realistic hot-source queues
    while preserving the requested mean rate.

    One same-area burst per orbit is expanded to ten requests within 30 s.
    Normal model FLOPs are retained; 15% recurring background work and
    heterogeneous 3–8 GFLOP/s sustained accelerators create queue pressure.

    E1 uses ten globally distributed ground stations and a 10° minimum
    elevation angle. Optical terminals form a fore/aft and adjacent-plane
    four-neighbor mesh. E1 assumes clear-sky 99.5% ISL, 98%
    downlink, and 99.9% node reliability for realized delivery trials.

    B1 (DirectDownlink) is an end-to-end ground-processing baseline: it waits
    for the source satellite to enter a GS contact, downlinks the raw tile, and
    completes inference on that station's queued H100 SXM. ORDI and cooperative
    controls may instead compute in orbit and route compact products through an
    ISL-connected satellite.

    Each acquisition is a feasible near-nadir 4096² PlanetScope-class ROI at
    3.7 m native GSD. Ship uses RGB, wildfire RGB+NIR, and cloud/change use
    eight total input bands. Deadline distribution: log-normal σ=0.6, with medians wildfire→600 s,
    ship→900 s, change→1800 s, and cloud_filter→5760 s (one orbit).

    Each seed rebuilds the full environment (orbital phasing, ground targets,
    task arrivals, deadline draws) and draws a deterministic random fault
    schedule shared by every policy in that seed. The default 0.02 per-epoch
    fault probability retains fault exposure without allowing robustness to
    dominate the compute-placement comparison. E2 and E3 retain the stronger
    fault-stress settings. The CSV reports across-seed mean and std.
    """
    print(f"E1: Core performance (3×12 Walker at 475 km, 10 global GS, "
          f"3–8 GFLOP/s, 10° GS elevation, 10-request/orbit hotspots, "
          f"fault rate {fault_rate:.2f}, {n_seeds} seeds)")
    build_kwargs = dict(_E1_BUILD_KWARGS)
    alg_classes = [("ORDI", ORDI)] + [(c.name, c) for c in CORE_BASELINES]

    config_args = []
    for s in range(n_seeds):
        fault_specs = [("random_schedule", fault_rate, seed + s)]
        jobs = [(f"{alg}#s{s}", alg, cls, fault_specs)
                for alg, cls in alg_classes]
        config_args.append((build_kwargs, jobs, seed + s))

    raw = _run_configs_parallel(config_args, desc="E1 seeds")

    # Collapse per-seed records so _save_csv reports mean ± std per algorithm.
    results: Dict[str, List[EpochMetrics]] = {}
    for alg, _cls in alg_classes:
        results[alg] = [m for s in range(n_seeds)
                        for m in raw[f"{alg}#s{s}"]]

    # Utility and objective values encode ORDI's own preference function and
    # are therefore not algorithm-neutral E1 outcomes. Report only operational
    # delivery reliability and network cost in the core comparison.
    _save_csv("E1_core", results,
              metric_keys=E1_METRIC_KEYS)
    return results


# ── E2: Fault intensity sweep ────────────────────────────────────────────────

def run_E2(seed=0, n_seeds=2) -> Dict[str, List[EpochMetrics]]:
    """
    Fault intensity sweep averaging over BOTH randomness sources: each seed
    rebuilds the environment (orbits, tasks, deadlines) AND draws a fresh
    random fault schedule, so the curves carry across-world error bars rather
    than fault-draw jitter on one fixed world.
    """
    print(f"E2: Fault intensity sweep ({n_seeds} seeds)")
    fault_rates = [0.0, 0.25, 0.50]
    alg_classes = [("ORDI", ORDI),
                   ("seco_adapted", SECOAdapted),
                   ("full_replication", FullReplication)]

    config_args = []
    for s in range(n_seeds):
        jobs = []
        for rate in fault_rates:
            specs = [("random_schedule", rate, seed + s)]
            for alg_name, cls in alg_classes:
                jobs.append((f"{alg_name}@fault={rate:.2f}#s{s}",
                             alg_name, cls, specs))
        config_args.append(({}, jobs, seed + s))

    raw = _run_configs_parallel(config_args, desc="E2 seeds")

    # Collapse the per-seed lifetime records into one list per (rate, alg) so
    # _save_csv's aggregate_metrics reports across-seed mean ± std.
    results: Dict[str, List[EpochMetrics]] = {}
    for rate in fault_rates:
        for alg_name, _cls in alg_classes:
            results[f"{alg_name}@fault={rate:.2f}"] = [
                m for s in range(n_seeds)
                for m in raw[f"{alg_name}@fault={rate:.2f}#s{s}"]
            ]

    _save_csv("E2_fault_intensity", results,
              metric_keys=["realized_miss_ratio", "isl_traffic_bits"])
    return results


# ── E4: Scalability (constellation size 12–36 sats) ─────────────────────────

_E4_CONFIGS = {
    12: (3, 4),   # 3 planes × 4 sats: enough inter-plane contact density
    24: (4, 6),   # 4 planes × 6 sats
    36: (6, 6),   # 6 planes × 6 sats (matches E1 baseline)
}


def run_E4(seed=0, n_seeds=1) -> Dict[str, List[EpochMetrics]]:
    print(f"E4: Scalability sweep ({n_seeds} seeds)")

    # Sim is rebuilt per (constellation size, seed); each config worker chains
    # the 2 algorithms behind one build. Chaining halves repeated orbit builds.
    alg_classes = [("ORDI", ORDI),
                   ("seco_adapted", SECOAdapted)]
    sizes = list(_E4_CONFIGS)
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

    _save_csv("E4_scalability", results,
              metric_keys=["realized_miss_ratio", "isl_traffic_bits"])
    return results


# ── E3: Correlated failures (orbital-plane outage) ────────────────────────────

def run_E3(seed=0, n_seeds=2) -> Dict[str, List[EpochMetrics]]:
    """
    Correlated orbital-plane outages probing replica-placement quality.

    Sustained outage (epochs 10-50, matching E2's severity) hits one plane or
    two adjacent planes. Matched environment seeds vary orbital phasing, task
    sources, and deadlines while keeping the compact evaluation tractable.

    Algorithms: ORDI, full replication, and random replication.
    Differences isolate how much backup placement and count buy under
    correlated failure.  (Measured property, stated rather than ablated:
    ORDI's greedy scoring already places 100% of backups in a different
    orbital plane than the primary in this constellation.)
    """
    print(f"E3: Correlated plane outages (placement quality, {n_seeds} seeds)")
    alg_classes = [("ORDI", ORDI),
                   ("full_replication", FullReplication),
                   ("random_replication", RandomReplication)]

    # One fixed representative placement per scale; matched seeds provide the
    # uncertainty samples without multiplying every seed by nine plane sweeps.
    scenarios = {
        "1plane":  [(0,)],
        "2planes": [(0, 1)],
    }

    config_args = []
    for s in range(n_seeds):
        jobs = []
        for label, plane_sets in scenarios.items():
            for planes in plane_sets:
                spec = [("plane_outage", 10, 40, planes)]
                for alg_name, cls in alg_classes:
                    key = f"{alg_name}@{label}#p{planes[0]}s{s}"
                    # ORDI enforces plane-disjoint backups under this
                    # correlated-failure threat model (matches the abstract).
                    overrides = ({"plane_disjoint_backup": True}
                                 if alg_name == "ORDI" else None)
                    jobs.append((key, alg_name, cls, spec, overrides))
        config_args.append(({}, jobs, seed + s))

    raw = _run_configs_parallel(config_args, desc="E3 seeds")

    # Collapse over plane positions and seeds → mean ± std per (alg, scale).
    results: Dict[str, List[EpochMetrics]] = {}
    for alg_name, _cls in alg_classes:
        for label, plane_sets in scenarios.items():
            results[f"{alg_name}@{label}"] = [
                m for s in range(n_seeds) for planes in plane_sets
                for m in raw[f"{alg_name}@{label}#p{planes[0]}s{s}"]
            ]

    _save_csv("E3_correlated", results,
              metric_keys=["realized_miss_ratio", "isl_traffic_bits"])
    return results


# ── master runner ─────────────────────────────────────────────────────────────

ALL_EXPERIMENTS = {
    "E1": run_E1, "E2": run_E2, "E3": run_E3, "E4": run_E4,
}


def run_all(seed=0):
    for exp_id, fn in tqdm(ALL_EXPERIMENTS.items(), desc="Experiments", unit="exp"):
        print(f"\n{'='*50}\n{exp_id}\n{'='*50}")
        fn(seed=seed)
