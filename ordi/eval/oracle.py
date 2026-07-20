"""Exact reduced-instance placement oracle.

The production schedulers are intentionally left untouched.  This module
builds a small deterministic workload, enumerates a bounded set of unsplit
placements using the shared routing model, and exhaustively searches placement
choice, admission, and execution order with the physical feasibility ledger.

The result is an exact optimum only for the declared reduced action space.  It
is an evaluation bound, not a claim of global optimality for full E1.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import random
from copy import deepcopy
from dataclasses import dataclass

from ordi.algorithms import (
    Assignment, ContactWindow, Decision, EpochInput, ORDI, PolicyWeights,
    SatelliteView, SECOAdapted,
)
from ordi.algorithms._common import enumerate_placements
from ordi.eval.validation import DecisionFeasibilityModel, InvalidDecisionError
from ordi.tasks.generator import EOTask, Tile
from ordi.tasks.profiles import PROFILES


@dataclass(frozen=True)
class OracleResult:
    objective: float
    assignments: tuple[Assignment, ...]
    search_nodes: int
    candidate_count: int


@dataclass(frozen=True)
class GapRecord:
    seed: int
    algorithm: str
    oracle_objective: float
    policy_objective: float
    optimality_gap: float
    oracle_deliveries: int
    policy_deliveries: int
    search_nodes: int


def _assignment_value(request: EpochInput, tile: Tile,
                      assignment: Assignment) -> float:
    """Reliability- and freshness-weighted delivered utility."""
    reliability = float(assignment.metadata.get("reliability", 0.0))
    latency = float(assignment.metadata.get("latency", math.inf))
    if reliability <= 0.0 or not math.isfinite(latency):
        return 0.0
    return tile.utility * reliability * math.exp(
        -request.weights.freshness * latency
    )


def _candidate_assignment(request: EpochInput, task: EOTask, tile: Tile,
                          placement) -> Assignment:
    reliability = (
        request.satellites[task.source_sat].reliability
        * placement.reliability
    )
    return Assignment(
        task.task_id,
        tile.tile_id,
        task.source_sat,
        helpers=(placement.helper,),
        aggregators=(placement.aggregator,),
        metadata={
            "latency": placement.latency,
            "reliability": reliability,
            "data_shards": 1,
            "split_count": 1,
            "effective_replicas": 1.0,
            "oracle_candidate": True,
        },
        routes=((
            placement.route_in,
            placement.route_out,
            placement.route_down,
        ),),
        work_fractions=(1.0,),
        input_fractions=(1.0,),
        output_fractions=(1.0,),
    )


def enumerate_oracle_candidates(request: EpochInput,
                                candidate_cap: int = 6):
    """Return the best distinct unsplit routes for every tile."""
    if candidate_cap < 1:
        raise ValueError("candidate_cap must be positive")
    candidates = {}
    tile_lookup = {}
    for task in request.tasks:
        for tile in task.tiles:
            key = (task.task_id, tile.tile_id)
            tile_lookup[key] = tile
            unique = {}
            for placement in enumerate_placements(request, task, tile):
                route_key = (
                    placement.helper, placement.aggregator,
                    placement.route_in, placement.route_out,
                    placement.route_down,
                )
                assignment = _candidate_assignment(
                    request, task, tile, placement
                )
                current = unique.get(route_key)
                if (current is None
                        or _assignment_value(request, tile, assignment)
                        > _assignment_value(request, tile, current)):
                    unique[route_key] = assignment
            ranked = sorted(
                unique.values(),
                key=lambda item: _assignment_value(request, tile, item),
                reverse=True,
            )
            candidates[key] = tuple(ranked[:candidate_cap])
    return candidates, tile_lookup


def solve_reduced_oracle(request: EpochInput,
                         candidate_cap: int = 6,
                         max_tiles: int = 6) -> OracleResult:
    """Exhaustively optimize a small request set under shared resources.

    Branches cover every candidate, every request ordering, and admission or
    rejection.  Candidate generation is deliberately capped, so the returned
    result is exact for that bounded set and nowhere else.
    """
    n_tiles = sum(len(task.tiles) for task in request.tasks)
    if n_tiles > max_tiles:
        raise ValueError(
            f"reduced oracle accepts at most {max_tiles} tiles, got {n_tiles}"
        )
    candidates, tile_lookup = enumerate_oracle_candidates(
        request, candidate_cap
    )
    keys = tuple(candidates)
    optimistic = {
        key: max(
            (_assignment_value(request, tile_lookup[key], assignment)
             for assignment in candidates[key]),
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
        if score + sum(optimistic[key] for key in remaining) <= best_score + 1e-12:
            return
        if score > best_score + 1e-12:
            best_score = score
            best_assignments = chosen
        if not remaining:
            return

        # Skipping the canonical first item covers every admitted subset once.
        first = remaining[0]
        search(remaining[1:], feasibility, chosen, score)

        # Selecting any remaining item next covers every execution order.
        for key in remaining:
            rest = tuple(item for item in remaining if item != key)
            tile = tile_lookup[key]
            for assignment in candidates[key]:
                trial = deepcopy(feasibility)
                try:
                    accepted = trial.validate_and_reserve(
                        request,
                        Decision(request.epoch, (assignment,)),
                        retime=True,
                    ).assignments[0]
                except InvalidDecisionError:
                    continue
                value = _assignment_value(request, tile, accepted)
                search(rest, trial, chosen + (accepted,), score + value)

    search(keys, DecisionFeasibilityModel(), (), 0.0)
    return OracleResult(
        best_score,
        best_assignments,
        search_nodes,
        sum(len(items) for items in candidates.values()),
    )


def build_reduced_instance(seed: int = 0, n_tiles: int = 4) -> EpochInput:
    """Build a small compute/contact-contention snapshot for exact search."""
    if not 1 <= n_tiles <= 6:
        raise ValueError("n_tiles must be between 1 and 6")
    rng = random.Random(seed)
    sat_ids = tuple(f"SAT_{plane:02d}_{slot:02d}"
                    for plane in range(2) for slot in range(2))
    source = sat_ids[0]
    states = {}
    for index, sat_id in enumerate(sat_ids):
        rate = rng.uniform(3.0, 8.0) * 1e9
        queued_seconds = rng.uniform(5.0, 55.0)
        states[sat_id] = SatelliteView(
            sat_id, True, rate, 80_000.0, 100_000.0,
            rng.uniform(30.0, 55.0), rate * queued_seconds,
            reliability=rng.uniform(0.965, 0.995),
        )

    contacts = []
    for left in sat_ids:
        for right in sat_ids:
            if left != right:
                contacts.append(ContactWindow(
                    left, right, 0.0, 1_200.0, 200e6, "isl", 0.99
                ))
    ground = "ground"
    for index, sat_id in enumerate(sat_ids):
        opens = 80.0 + 55.0 * index + rng.uniform(0.0, 35.0)
        contacts.append(ContactWindow(
            sat_id, ground, opens, opens + 480.0,
            100e6, "downlink", 0.985,
        ))

    profile_names = ("wildfire", "ship", "change", "cloud_filter")
    tasks = []
    for index in range(n_tiles):
        profile = PROFILES[profile_names[index % len(profile_names)]]
        compute_multiplier = 16.0 if index < max(2, n_tiles // 2) else 1.0
        tile = Tile(
            index, 0, profile,
            profile.d_in_bits * rng.uniform(0.95, 1.05),
            profile.d_out_bits * rng.uniform(0.95, 1.05),
            profile.compute_ops * compute_multiplier * rng.uniform(0.9, 1.1),
            profile.base_utility * rng.uniform(0.9, 1.1),
            0, 0, 1,
        )
        deadline = rng.uniform(330.0, 700.0)
        tasks.append(EOTask(
            index, source, 0.0, deadline, profile.name, [tile], 1, 0
        ))

    opportunities = {
        sat_id: tuple(other for other in sat_ids if other != sat_id)
        for sat_id in sat_ids
    }
    return EpochInput(
        0, 0.0, tuple(tasks), states, opportunities,
        frozenset({ground}), tuple(contacts), 60.0,
        PolicyWeights(freshness=0.002, energy=1e-5,
                      communication=1e-12, replication=0.05),
    )


def _policy_result(request: EpochInput, scheduler) -> Decision:
    sources = {task.source_sat for task in request.tasks}
    for source in sources:
        scheduler.messages.seed_knowledge(
            source, request.satellites,
            generated_at=-request.epoch_length,
            delivered_at=request.sim_time,
        )
    return scheduler.schedule(request)


def _decision_value(request: EpochInput, decision: Decision) -> float:
    tiles = {
        (task.task_id, tile.tile_id): tile
        for task in request.tasks for tile in task.tiles
    }
    return sum(
        _assignment_value(request, tiles[(item.task_id, item.tile_id)], item)
        for item in decision.assignments
    )


def compare_reduced_oracle(seed: int = 0, n_tiles: int = 4,
                           candidate_cap: int = 6) -> tuple[GapRecord, ...]:
    request = build_reduced_instance(seed, n_tiles)
    oracle = solve_reduced_oracle(request, candidate_cap, max_tiles=n_tiles)
    policies = (
        ("ORDI", ORDI(max_replicas=1, split_options=(1,))),
        ("seco_adapted", SECOAdapted(split_options=(1,))),
    )
    records = []
    for name, scheduler in policies:
        decision = _policy_result(request, scheduler)
        value = _decision_value(request, decision)
        gap = (
            max(0.0, (oracle.objective - value) / oracle.objective)
            if oracle.objective > 0.0 else 0.0
        )
        records.append(GapRecord(
            seed, name, oracle.objective, value, gap,
            len(oracle.assignments), len(decision.assignments),
            oracle.search_nodes,
        ))
    return tuple(records)


def run_reduced_oracle_comparison(n_seeds: int = 8, n_tiles: int = 4,
                                  candidate_cap: int = 6,
                                  output_path: str = "results/oracle_gap.csv"):
    records = [
        record
        for seed in range(n_seeds)
        for record in compare_reduced_oracle(seed, n_tiles, candidate_cap)
    ]
    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(output_path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=tuple(GapRecord.__dataclass_fields__)
        )
        writer.writeheader()
        writer.writerows(record.__dict__ for record in records)
    return records


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--tiles", type=int, default=4)
    parser.add_argument("--candidate-cap", type=int, default=6)
    parser.add_argument("--output", default="results/oracle_gap.csv")
    args = parser.parse_args()
    records = run_reduced_oracle_comparison(
        args.seeds, args.tiles, args.candidate_cap, args.output
    )
    for algorithm in ("ORDI", "seco_adapted"):
        rows = [row for row in records if row.algorithm == algorithm]
        mean_gap = sum(row.optimality_gap for row in rows) / len(rows)
        mean_deliveries = sum(row.policy_deliveries for row in rows) / len(rows)
        print(
            f"{algorithm}: mean optimality gap={mean_gap:.2%}, "
            f"mean deliveries={mean_deliveries:.2f}"
        )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    _main()
