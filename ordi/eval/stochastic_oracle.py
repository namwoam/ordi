"""Multi-epoch stochastic oracle for small satellite-compute instances.

This evaluation module leaves the production policies unchanged.  It performs
an exact branch-and-bound search over a declared, bounded action set containing
primary-only and fault-disjoint primary-plus-backup placements.  Resource
reservations persist across release epochs, and every plan is scored against
the same finite set of correlated component-fault scenarios.

The result is an exact stochastic optimum for the enumerated actions, not for
the full E1 action space.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import random
from copy import deepcopy
from dataclasses import dataclass, replace

from ordi.algorithms import (
    Assignment, ContactWindow, Decision, EpochInput, ORDI, PolicyWeights,
    SatelliteView, SECOAdapted,
)
from ordi.algorithms._common import enumerate_placements, plane
from ordi.eval.validation import DecisionFeasibilityModel, InvalidDecisionError
from ordi.tasks.generator import EOTask, Tile
from ordi.tasks.profiles import PROFILES


@dataclass(frozen=True)
class FaultScenario:
    name: str
    weight: float = 1.0
    node_failures: tuple[tuple[int, frozenset[str]], ...] = ()
    link_failures: tuple[
        tuple[int, frozenset[tuple[str, str]]], ...
    ] = ()

    def node_failed(self, node: str, epoch: int) -> bool:
        return any(
            epoch >= starts and node in nodes
            for starts, nodes in self.node_failures
        )

    def link_failed(self, source: str, target: str, epoch: int) -> bool:
        return any(
            epoch >= starts and (source, target) in links
            for starts, links in self.link_failures
        )


@dataclass(frozen=True)
class MultiEpochInstance:
    requests: tuple[EpochInput, ...]
    scenarios: tuple[FaultScenario, ...]


@dataclass(frozen=True)
class StochasticOracleResult:
    objective: float
    expected_miss_ratio: float
    replication_cost: float
    assignments: tuple[tuple[int, Assignment], ...]
    search_nodes: int
    candidate_count: int


@dataclass(frozen=True)
class StochasticGapRecord:
    seed: int
    algorithm: str
    oracle_objective: float
    policy_objective: float
    optimality_gap: float
    oracle_expected_miss_ratio: float
    policy_expected_miss_ratio: float
    oracle_replication_cost: float
    policy_replication_cost: float
    oracle_assignments: int
    policy_assignments: int
    search_nodes: int


def _task_maps(instance: MultiEpochInstance):
    requests = {}
    tasks = {}
    tiles = {}
    for request in instance.requests:
        for task in request.tasks:
            for tile in task.tiles:
                key = (request.epoch, task.task_id, tile.tile_id)
                requests[key] = request
                tasks[key] = task
                tiles[key] = tile
    return requests, tasks, tiles


def _route_nodes(placement, source: str, ground_stations) -> frozenset[str]:
    ground_stations = set(ground_stations)
    return frozenset(
        node
        for route in (
            placement.route_in, placement.route_out, placement.route_down
        )
        for node in route
        if node != source and node not in ground_stations
    )


def _placements_disjoint(request, task, primary, backup) -> bool:
    if primary.helper == backup.helper:
        return False
    if plane(primary.helper) == plane(backup.helper):
        return False
    return not _route_nodes(
        primary, task.source_sat, request.ground_stations
    ).intersection(_route_nodes(
        backup, task.source_sat, request.ground_stations
    ))


def _make_action(request, task, tile, placements) -> Assignment:
    placements = tuple(placements)
    source_reliability = request.satellites[task.source_sat].reliability
    conditional_failure = math.prod(
        1.0 - placement.reliability for placement in placements
    )
    return Assignment(
        task.task_id,
        tile.tile_id,
        task.source_sat,
        helpers=tuple(item.helper for item in placements),
        aggregators=tuple(item.aggregator for item in placements),
        metadata={
            "latency": min(item.latency for item in placements),
            "reliability": source_reliability * (1.0 - conditional_failure),
            "data_shards": 1,
            "split_count": 1,
            "shard_groups": tuple(range(len(placements))),
            "effective_replicas": float(len(placements)),
            "replica_latencies": tuple(item.latency for item in placements),
            "stochastic_oracle_candidate": True,
        },
        routes=tuple((
            item.route_in, item.route_out, item.route_down
        ) for item in placements),
        work_fractions=(1.0,) * len(placements),
        input_fractions=(1.0,) * len(placements),
        output_fractions=(1.0,) * len(placements),
    )


def _action_key(assignment: Assignment):
    return (
        assignment.helpers,
        assignment.aggregators,
        assignment.routes,
        tuple(assignment.metadata.get("shard_groups", ())),
    )


def _centralize_action(assignment: Assignment) -> Assignment:
    """Strip protocol overhead while preserving a policy's placement choice."""
    labels = assignment.metadata.get("shard_groups")
    if labels is None:
        labels = tuple(range(len(assignment.helpers)))
    metadata = dict(assignment.metadata)
    metadata.update({
        "data_shards": 1,
        "split_count": 1,
        "shard_groups": tuple(labels),
        "effective_replicas": float(len(set(labels))),
        "replica_latencies": tuple(
            float(assignment.metadata.get("latency", math.inf))
            for _ in assignment.helpers
        ),
        "protocol_header_bits": 0.0,
        "oracle_policy_candidate": True,
    })
    return Assignment(
        assignment.task_id,
        assignment.tile_id,
        assignment.source,
        assignment.helpers,
        assignment.aggregators,
        False,
        metadata,
        assignment.routes,
        assignment.work_fractions or (1.0,) * len(assignment.helpers),
        assignment.input_fractions or (1.0,) * len(assignment.helpers),
        assignment.output_fractions or (1.0,) * len(assignment.helpers),
    )


def _replica_survives(assignment: Assignment, index: int,
                      scenario: FaultScenario, epoch: int,
                      ground_stations) -> bool:
    source = assignment.source
    if scenario.node_failed(source, epoch):
        return False
    ground_stations = set(ground_stations)
    routes = assignment.routes[index]
    for route in routes:
        for node in route:
            if (node not in ground_stations
                    and scenario.node_failed(node, epoch)):
                return False
        for left, right in zip(route, route[1:]):
            if scenario.link_failed(left, right, epoch):
                return False
    return True


def _action_statistics(instance: MultiEpochInstance, request: EpochInput,
                       task, tile, assignment: Assignment):
    replica_latencies = tuple(float(value) for value in assignment.metadata.get(
        "replica_latencies",
        (float(assignment.metadata.get("latency", math.inf)),)
        * len(assignment.helpers),
    ))
    modeled_latency = float(assignment.metadata.get("latency", math.inf))
    finite = [value for value in replica_latencies if math.isfinite(value)]
    shift = modeled_latency - min(finite) if finite else 0.0
    adjusted = tuple(value + shift for value in replica_latencies)

    utility_total = 0.0
    misses = 0
    total_weight = sum(scenario.weight for scenario in instance.scenarios)
    for scenario in instance.scenarios:
        surviving = []
        for index, latency in enumerate(adjusted):
            delivery_time = request.sim_time + latency
            delivery_epoch = max(
                request.epoch,
                int(delivery_time // max(request.epoch_length, 1.0)),
            )
            if (delivery_time <= task.deadline + 1e-9
                    and _replica_survives(
                        assignment, index, scenario, delivery_epoch,
                        request.ground_stations,
                    )):
                surviving.append(latency)
        if not surviving:
            misses += scenario.weight
            continue
        latency = min(surviving)
        utility_total += scenario.weight * tile.utility * math.exp(
            -request.weights.freshness * latency
        )

    total_weight = max(total_weight, 1e-12)
    # ``misses`` is accumulated as a weight, not a scenario count.
    expected_utility = utility_total / total_weight
    miss_probability = misses / total_weight
    replica_count = len(set(assignment.metadata.get(
        "shard_groups", range(len(assignment.helpers))
    )))
    replication_cost = request.weights.replication * max(
        0, replica_count - 1
    )
    return expected_utility - replication_cost, miss_probability, replication_cost


def _action_upper_bound(request, tile, assignment):
    latencies = assignment.metadata.get("replica_latencies", ())
    latency = min(latencies, default=assignment.metadata.get(
        "latency", math.inf
    ))
    labels = assignment.metadata.get(
        "shard_groups", range(len(assignment.helpers))
    )
    replication_cost = request.weights.replication * max(
        0, len(set(labels)) - 1
    )
    return max(0.0, tile.utility * math.exp(
        -request.weights.freshness * float(latency)
    ) - replication_cost)


def enumerate_stochastic_actions(
    instance: MultiEpochInstance,
    primary_cap: int = 4,
    backup_cap: int = 3,
):
    """Enumerate bounded primary and disjoint-backup choices per request."""
    requests, tasks, tiles = _task_maps(instance)
    actions = {}
    for key, request in requests.items():
        task = tasks[key]
        tile = tiles[key]
        placements = enumerate_placements(request, task, tile)
        unique = {}
        for placement in placements:
            placement_key = (
                placement.helper, placement.aggregator,
                placement.route_in, placement.route_out,
                placement.route_down,
            )
            unique.setdefault(placement_key, placement)
        ranked = sorted(
            unique.values(),
            key=lambda item: tile.utility * math.exp(
                -request.weights.freshness * item.latency
            ),
            reverse=True,
        )
        primaries = ranked[:primary_cap]
        selected = [_make_action(request, task, tile, (item,))
                    for item in primaries]
        pairs = []
        for primary in primaries:
            for backup in ranked:
                if _placements_disjoint(request, task, primary, backup):
                    action = _make_action(
                        request, task, tile, (primary, backup)
                    )
                    pairs.append(action)
        pairs.sort(
            key=lambda item: _action_upper_bound(request, tile, item),
            reverse=True,
        )
        selected.extend(pairs[:backup_cap])
        deduplicated = {}
        for action in selected:
            deduplicated.setdefault(_action_key(action), action)
        actions[key] = tuple(deduplicated.values())
    return actions, requests, tasks, tiles


def solve_stochastic_oracle(
    instance: MultiEpochInstance,
    primary_cap: int = 4,
    backup_cap: int = 3,
) -> StochasticOracleResult:
    """Solve the bounded multi-epoch stochastic placement problem exactly."""
    actions, requests, tasks, tiles = enumerate_stochastic_actions(
        instance, primary_cap, backup_cap
    )
    keys = tuple(sorted(actions))
    upper = {
        key: max(
            (_action_upper_bound(requests[key], tiles[key], action)
             for action in actions[key]),
            default=0.0,
        )
        for key in keys
    }
    best_score = 0.0
    best_assignments = ()
    search_nodes = 0

    def search(remaining, feasibility, chosen, score):
        nonlocal best_score, best_assignments, search_nodes
        search_nodes += 1
        if score + sum(upper[key] for key in remaining) <= best_score + 1e-12:
            return
        if score > best_score + 1e-12:
            best_score = score
            best_assignments = chosen
        if not remaining:
            return

        earliest_epoch = remaining[0][0]
        available = tuple(key for key in remaining if key[0] == earliest_epoch)
        first = available[0]
        search(
            tuple(key for key in remaining if key != first),
            feasibility, chosen, score,
        )
        for key in available:
            rest = tuple(item for item in remaining if item != key)
            request = requests[key]
            for action in actions[key]:
                trial = deepcopy(feasibility)
                try:
                    accepted = trial.validate_and_reserve(
                        request,
                        Decision(request.epoch, (action,)),
                        retime=True,
                    ).assignments[0]
                except InvalidDecisionError:
                    continue
                value, _miss, _cost = _action_statistics(
                    instance, request, tasks[key], tiles[key], accepted
                )
                search(
                    rest, trial,
                    chosen + ((request.epoch, accepted),),
                    score + value,
                )

    search(keys, DecisionFeasibilityModel(), (), 0.0)
    selected = {
        (epoch, item.task_id, item.tile_id): item
        for epoch, item in best_assignments
    }
    miss_total = 0.0
    replication_cost = 0.0
    for key in keys:
        action = selected.get(key)
        if action is None:
            miss_total += 1.0
            continue
        _value, miss, cost = _action_statistics(
            instance, requests[key], tasks[key], tiles[key], action
        )
        miss_total += miss
        replication_cost += cost
    return StochasticOracleResult(
        best_score,
        miss_total / max(len(keys), 1),
        replication_cost,
        best_assignments,
        search_nodes,
        sum(len(items) for items in actions.values()),
    )


def build_fault_scenarios(
    sat_ids, fault_rate: float = 0.10
) -> tuple[FaultScenario, ...]:
    """Build a finite distribution matched to E1's injected fault mass.

    E1 samples one of seven random fault families with probability
    ``fault_rate`` per epoch. The reduced model groups helper, straggler,
    battery, and thermal effects into compute-domain risk; ISL, ground-contact,
    and adverse-downlink effects into route-domain risk. A quarter of each
    group is reserved for correlated plane/cut stress cases.
    """
    if not 0.0 <= fault_rate <= 1.0:
        raise ValueError("fault_rate must be between zero and one")
    sat_ids = tuple(sat_ids)
    planes = {}
    for sat_id in sat_ids:
        planes.setdefault(plane(sat_id), set()).add(sat_id)
    plane_names = sorted(planes)
    cross_links = frozenset(
        (left, right)
        for left in sat_ids for right in sat_ids
        if plane(left) != plane(right)
    )
    sparse_cut = frozenset(sorted(cross_links)[:max(2, len(sat_ids))])
    compute_mass = fault_rate * 4.0 / 7.0
    route_mass = fault_rate * 3.0 / 7.0
    plane_mass = compute_mass * 0.25
    correlated_mass = route_mass * 0.25
    return (
        FaultScenario("nominal", weight=1.0 - fault_rate),
        FaultScenario(
            "plane_early", weight=plane_mass / 2.0,
            node_failures=((1, frozenset(planes[plane_names[0]])),),
        ),
        FaultScenario(
            "plane_late", weight=plane_mass / 2.0,
            node_failures=((2, frozenset(planes[plane_names[1]])),),
        ),
        FaultScenario(
            "helper_failure", weight=compute_mass - plane_mass,
            node_failures=((1, frozenset({sat_ids[-1]})),),
        ),
        FaultScenario(
            "isl_cut", weight=route_mass - correlated_mass,
            link_failures=((1, sparse_cut),),
        ),
        FaultScenario(
            "correlated_plane_isl", weight=correlated_mass,
            node_failures=((2, frozenset(planes[plane_names[-1]])),),
            link_failures=((2, sparse_cut),),
        ),
    )


def build_multi_epoch_instance(
    seed: int = 0,
    n_sats: int = 6,
    n_requests: int = 5,
    n_epochs: int = 3,
    fault_rate: float = 0.10,
) -> MultiEpochInstance:
    if not 4 <= n_sats <= 6:
        raise ValueError("n_sats must be between 4 and 6")
    if not 4 <= n_requests <= 6:
        raise ValueError("n_requests must be between 4 and 6")
    if not 2 <= n_epochs <= 4:
        raise ValueError("n_epochs must be between 2 and 4")
    rng = random.Random(seed)
    n_planes = 3 if n_sats >= 6 else 2
    sat_ids = tuple(
        f"SAT_{index % n_planes:02d}_{index // n_planes:02d}"
        for index in range(n_sats)
    )
    states = {}
    for sat_id in sat_ids:
        rate = rng.uniform(3.0, 8.0) * 1e9
        queued_seconds = rng.uniform(10.0, 50.0)
        states[sat_id] = SatelliteView(
            sat_id, True, rate, 80_000.0, 100_000.0,
            rng.uniform(32.0, 58.0), rate * queued_seconds,
            reliability=rng.uniform(0.97, 0.995),
        )

    epoch_length = 60.0
    horizon = 720.0
    contacts = []
    for left in sat_ids:
        for right in sat_ids:
            if left != right:
                contacts.append(ContactWindow(
                    left, right, 0.0, horizon,
                    200e6, "isl", 0.99,
                ))
    ground = "ground"
    for index, sat_id in enumerate(sat_ids):
        for pass_index in range(2):
            opens = 75.0 + 38.0 * index + 280.0 * pass_index
            contacts.append(ContactWindow(
                sat_id, ground, opens, opens + 150.0,
                100e6, "downlink", 0.985,
            ))

    profile_names = ("wildfire", "ship", "change", "cloud_filter")
    tasks_by_epoch = {epoch: [] for epoch in range(n_epochs)}
    source_pool = sat_ids[:min(3, len(sat_ids))]
    for index in range(n_requests):
        epoch = min(index * n_epochs // n_requests, n_epochs - 1)
        release = epoch * epoch_length
        profile = PROFILES[profile_names[index % len(profile_names)]]
        multiplier = 12.0 if index < max(2, n_requests // 2) else 2.0
        tile = Tile(
            index, 0, profile,
            profile.d_in_bits * rng.uniform(0.95, 1.05),
            profile.d_out_bits * rng.uniform(0.95, 1.05),
            profile.compute_ops * multiplier * rng.uniform(0.9, 1.1),
            profile.base_utility * rng.uniform(0.9, 1.1),
            0, 0, 2,
        )
        task = EOTask(
            index, source_pool[index % len(source_pool)], release,
            release + rng.uniform(260.0, 500.0),
            profile.name, [tile], 1, index // 2,
        )
        tasks_by_epoch[epoch].append(task)

    opportunities = {
        sat_id: tuple(other for other in sat_ids if other != sat_id)
        for sat_id in sat_ids
    }
    weights = PolicyWeights(
        freshness=0.002, energy=1e-5,
        communication=1e-12, replication=0.05,
    )
    requests = tuple(EpochInput(
        epoch, epoch * epoch_length, tuple(tasks_by_epoch[epoch]),
        states, opportunities, frozenset({ground}), tuple(contacts),
        epoch_length, weights,
    ) for epoch in range(n_epochs))
    return MultiEpochInstance(
        requests, build_fault_scenarios(sat_ids, fault_rate=fault_rate)
    )


def _scenario_request(request: EpochInput, scenario: FaultScenario):
    """Expose only faults active by this decision epoch to an online policy."""
    states = {
        sat_id: replace(
            state,
            available=(
                state.available
                and not scenario.node_failed(sat_id, request.epoch)
            ),
        )
        for sat_id, state in request.satellites.items()
    }
    contacts = tuple(
        contact for contact in request.contacts
        if (not scenario.node_failed(contact.source, request.epoch)
            and not scenario.node_failed(contact.target, request.epoch)
            and not scenario.link_failed(
                contact.source, contact.target, request.epoch
            ))
    )
    return replace(request, satellites=states, contacts=contacts)


def _scenario_instance(instance: MultiEpochInstance,
                       scenario: FaultScenario):
    normalized = replace(scenario, weight=1.0)
    return MultiEpochInstance(
        tuple(_scenario_request(request, scenario)
              for request in instance.requests),
        (normalized,),
    )


def _scenario_group_status(assignment: Assignment,
                           scenario: FaultScenario, epoch: int,
                           ground_stations):
    alive = [
        _replica_survives(
            assignment, index, scenario, epoch, ground_stations
        )
        for index in range(len(assignment.helpers))
    ]
    required = max(1, int(assignment.metadata.get("data_shards", 1)))
    labels = assignment.metadata.get("shard_groups")
    if labels is None:
        return {0: sum(alive) >= required}
    grouped = {}
    for label, survives in zip(labels, alive):
        grouped.setdefault(label, []).append(survives)
    return {
        label: len(group) >= required and sum(group) >= required
        for label, group in grouped.items()
    }


def _policy_plan(instance: MultiEpochInstance, scheduler,
                 scenario: FaultScenario | None = None):
    actions = {}
    pending_feedback = []
    observe = getattr(scheduler, "observe_assignment_outcome", None)
    for request in instance.requests:
        if scenario is not None:
            still_pending = []
            for original_request, assignment in pending_feedback:
                delivery_time = original_request.sim_time + float(
                    assignment.metadata.get("latency", math.inf)
                )
                delivery_epoch = int(
                    delivery_time // max(request.epoch_length, 1.0)
                )
                status = _scenario_group_status(
                    assignment, scenario, request.epoch,
                    request.ground_stations,
                )
                if not any(status.values()):
                    if observe:
                        observe("fault_failure")
                elif request.epoch >= delivery_epoch:
                    if observe:
                        observe(
                            "primary_success" if status.get(0, False)
                            else "backup_recovery"
                        )
                else:
                    still_pending.append((original_request, assignment))
            pending_feedback = still_pending
        for task in request.tasks:
            scheduler.messages.seed_knowledge(
                task.source_sat, request.satellites,
                generated_at=request.sim_time - request.epoch_length,
                delivered_at=request.sim_time,
            )
        decision = scheduler.schedule(request)
        for assignment in decision.assignments:
            key = (request.epoch, assignment.task_id, assignment.tile_id)
            action = _centralize_action(assignment)
            actions[key] = action
            if scenario is not None:
                pending_feedback.append((request, action))
    if scenario is not None:
        for request, assignment in pending_feedback:
            delivery_time = request.sim_time + float(
                assignment.metadata.get("latency", math.inf)
            )
            delivery_epoch = int(
                delivery_time // max(request.epoch_length, 1.0)
            )
            status = _scenario_group_status(
                assignment, scenario, delivery_epoch,
                request.ground_stations,
            )
            if observe:
                observe(
                    "primary_success" if status.get(0, False)
                    else ("backup_recovery" if any(status.values())
                          else "fault_failure")
                )
    return actions


def evaluate_fixed_plan(instance: MultiEpochInstance, actions: dict):
    requests, tasks, tiles = _task_maps(instance)
    feasibility = DecisionFeasibilityModel()
    objective = 0.0
    miss_total = 0.0
    replication_cost = 0.0
    accepted = 0
    for key in sorted(requests):
        action = actions.get(key)
        if action is None:
            miss_total += 1.0
            continue
        try:
            action = feasibility.validate_and_reserve(
                requests[key], Decision(key[0], (action,)), retime=True
            ).assignments[0]
        except InvalidDecisionError:
            miss_total += 1.0
            continue
        value, miss, cost = _action_statistics(
            instance, requests[key], tasks[key], tiles[key], action
        )
        objective += value
        miss_total += miss
        replication_cost += cost
        accepted += 1
    return (
        objective,
        miss_total / max(len(requests), 1),
        replication_cost,
        accepted,
    )


def compare_stochastic_oracle(
    seed: int = 0,
    n_sats: int = 6,
    n_requests: int = 5,
    n_epochs: int = 3,
    primary_cap: int = 4,
    backup_cap: int = 3,
    fault_rate: float = 0.10,
) -> tuple[StochasticGapRecord, ...]:
    instance = build_multi_epoch_instance(
        seed, n_sats, n_requests, n_epochs, fault_rate
    )
    total_weight = sum(item.weight for item in instance.scenarios)
    oracle_objective = 0.0
    oracle_misses = 0.0
    oracle_cost = 0.0
    oracle_assignments = 0.0
    oracle_search_nodes = 0
    for scenario in instance.scenarios:
        visible = _scenario_instance(instance, scenario)
        scenario_oracle = solve_stochastic_oracle(
            visible, primary_cap, backup_cap
        )
        oracle_objective += scenario.weight * scenario_oracle.objective
        oracle_misses += (
            scenario.weight * scenario_oracle.expected_miss_ratio
        )
        oracle_cost += scenario.weight * scenario_oracle.replication_cost
        oracle_assignments += (
            scenario.weight * len(scenario_oracle.assignments)
        )
        oracle_search_nodes += scenario_oracle.search_nodes
    oracle_objective /= total_weight
    oracle_misses /= total_weight
    oracle_cost /= total_weight
    oracle_assignments /= total_weight
    records = []
    policy_factories = {
        "ORDI": lambda: ORDI(
            max_replicas=2, split_options=(1,), rng_seed=seed
        ),
        "seco_adapted": lambda: SECOAdapted(split_options=(1,)),
    }
    for algorithm, factory in policy_factories.items():
        objective = misses = cost = accepted = 0.0
        for scenario in instance.scenarios:
            visible = _scenario_instance(instance, scenario)
            scheduler = factory()
            plan = _policy_plan(visible, scheduler, scenario)
            values = evaluate_fixed_plan(visible, plan)
            objective += scenario.weight * values[0]
            misses += scenario.weight * values[1]
            cost += scenario.weight * values[2]
            accepted += scenario.weight * values[3]
        objective /= total_weight
        misses /= total_weight
        cost /= total_weight
        accepted /= total_weight
        gap = (
            (oracle_objective - objective) / oracle_objective
            if oracle_objective > 0.0 else 0.0
        )
        records.append(StochasticGapRecord(
            seed, algorithm, oracle_objective, objective, gap,
            oracle_misses, misses,
            oracle_cost, cost,
            oracle_assignments, accepted, oracle_search_nodes,
        ))
    return tuple(records)


def run_stochastic_oracle_comparison(
    n_seeds: int = 8,
    output_path: str = "results/stochastic_oracle_gap.csv",
    **kwargs,
):
    records = [
        record
        for seed in range(n_seeds)
        for record in compare_stochastic_oracle(seed=seed, **kwargs)
    ]
    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(output_path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=tuple(StochasticGapRecord.__dataclass_fields__)
        )
        writer.writeheader()
        writer.writerows(record.__dict__ for record in records)
    return records


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--satellites", type=int, default=6)
    parser.add_argument("--requests", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--primary-cap", type=int, default=4)
    parser.add_argument("--backup-cap", type=int, default=3)
    parser.add_argument("--fault-rate", type=float, default=0.10)
    parser.add_argument(
        "--output", default="results/stochastic_oracle_gap.csv"
    )
    args = parser.parse_args()
    records = run_stochastic_oracle_comparison(
        args.seeds, args.output,
        n_sats=args.satellites,
        n_requests=args.requests,
        n_epochs=args.epochs,
        primary_cap=args.primary_cap,
        backup_cap=args.backup_cap,
        fault_rate=args.fault_rate,
    )
    for algorithm in ("ORDI", "seco_adapted"):
        rows = [row for row in records if row.algorithm == algorithm]
        mean_gap = sum(row.optimality_gap for row in rows) / len(rows)
        mean_miss = sum(
            row.policy_expected_miss_ratio for row in rows
        ) / len(rows)
        print(
            f"{algorithm}: mean gap={mean_gap:.2%}, "
            f"expected miss={mean_miss:.2%}"
        )
    oracle_miss = sum(
        row.oracle_expected_miss_ratio
        for row in records[::2]
    ) / max(len(records[::2]), 1)
    print(f"oracle: expected miss={oracle_miss:.2%}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    _main()
