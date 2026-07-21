from dataclasses import replace
from types import SimpleNamespace

import pytest

from ordi.algorithms import (
    ContactWindow, Decision, EpochInput, ORDI, PolicyWeights, SatelliteView,
)
from ordi.algorithms._common import plane
from ordi.eval.experiments import _advance_synthetic_states, _assignment_viable
from ordi.eval.validation import DecisionFeasibilityModel, InvalidDecisionError


def _view(name):
    return SatelliteView(
        name, True, 1_000.0, 9_000.0, 10_000.0, 25.0, 0.0,
        reliability=0.99,
    )


def test_ordi_splits_one_tile_across_two_distinct_helpers():
    states = {name: _view(name) for name in ("src", "helper")}
    contacts = (
        ContactWindow("src", "helper", 0.0, 10.0, 1e9, "isl"),
        ContactWindow("src", "ground", 0.0, 10.0, 1e9, "downlink"),
        ContactWindow("helper", "ground", 0.0, 10.0, 1e9, "downlink"),
    )
    tile = SimpleNamespace(
        tile_id=0, n_replicas_max=1, d_in_bits=100.0,
        d_out_bits=10.0, compute_ops=1_000.0, utility=1.0,
    )
    task = SimpleNamespace(
        task_id=1, source_sat="src", deadline=10.0, tiles=[tile]
    )
    request = EpochInput(
        0, 0.0, [task], states, {}, frozenset({"ground"}), contacts
    )

    scheduler = ORDI(
        max_replicas=1, split_options=(2,)
    )
    scheduler.messages.seed_knowledge(
        "src", states, generated_at=-60.0, delivered_at=0.0
    )
    assignment = scheduler.schedule(request).assignments[0]

    assert len(set(assignment.helpers)) == 2
    assert assignment.metadata["data_shards"] == 2
    assert assignment.metadata["shard_groups"] == (0, 0)
    assert assignment.metadata["effective_replicas"] == 1
    assert sum(assignment.work_fractions) == pytest.approx(1.05)
    assert assignment.output_fractions == pytest.approx((0.5, 0.5))
    assert [decision.action for decision in assignment.node_decisions] == [
        "split", "execute_forward", "execute_forward",
    ]

    physical_states = {
        name: SimpleNamespace(
            params=SimpleNamespace(compute_power_w=10.0, comms_power_w=5.0)
        )
        for name in states
    }
    workloads = _advance_synthetic_states(
        [assignment], [task], physical_states, 60.0
    )
    assert workloads["src"].compute_flops == pytest.approx(525.0)
    assert workloads["helper"].compute_flops == pytest.approx(525.0)

    inconsistent = replace(
        assignment,
        node_decisions=(
            replace(assignment.node_decisions[0], node="not-the-holder"),
            *assignment.node_decisions[1:],
        ),
    )
    with pytest.raises(InvalidDecisionError, match="acting on work held"):
        DecisionFeasibilityModel().validate_and_reserve(
            request, Decision(0, (inconsistent,))
        )


def test_receiving_node_dynamically_selects_parallelism():
    states = {name: _view(name) for name in ("src", "helper")}
    contacts = (
        ContactWindow("src", "helper", 0.0, 10.0, 1e9, "isl"),
        ContactWindow("src", "ground", 0.0, 10.0, 1e9, "downlink"),
        ContactWindow("helper", "ground", 0.0, 10.0, 1e9, "downlink"),
    )
    tile = SimpleNamespace(
        tile_id=0, n_replicas_max=1, d_in_bits=100.0,
        d_out_bits=10.0, compute_ops=1_000.0, utility=1.0,
    )
    task = SimpleNamespace(
        task_id=1, source_sat="src", deadline=10.0, tiles=[tile]
    )
    request = EpochInput(
        0, 0.0, [task], states, {}, frozenset({"ground"}), contacts,
        weights=PolicyWeights(
            freshness=0.5, energy=0.0, communication=0.0
        ),
    )

    parallel_scheduler = ORDI(
        max_replicas=1, split_options=(1, 2, 4)
    )
    parallel_scheduler.messages.seed_knowledge(
        "src", states, generated_at=-60.0, delivered_at=0.0
    )
    parallel = parallel_scheduler.schedule(request).assignments[0]
    assert parallel.metadata["split_count"] == 2

    local_only = ORDI(
        max_replicas=1, split_options=(1, 2, 4)
    ).schedule(EpochInput(
        0, 0.0, [task], {"src": states["src"]}, {},
        frozenset({"ground"}),
        (ContactWindow(
            "src", "ground", 0.0, 10.0, 1e9, "downlink"
        ),),
        weights=request.weights,
    )).assignments[0]
    assert local_only.metadata["split_count"] == 1
    assert local_only.node_decisions[0].action == "execute_forward"


def test_fault_viability_cannot_mix_shards_from_different_groups():
    assignment = SimpleNamespace(
        helpers=("h0", "h1", "h2", "h3"),
        aggregators=("h0", "h1", "h2", "h3"),
        metadata={"data_shards": 2, "shard_groups": (0, 0, 1, 1)},
    )
    states = {
        "h0": SimpleNamespace(A_i=True),
        "h1": SimpleNamespace(A_i=False),
        "h2": SimpleNamespace(A_i=True),
        "h3": SimpleNamespace(A_i=False),
    }

    assert not _assignment_viable(assignment, states)

    states["h3"].A_i = True
    assert _assignment_viable(assignment, states)


def test_fault_viability_accepts_any_two_of_three_coded_shards():
    assignment = SimpleNamespace(
        helpers=("h0", "h1", "h2"),
        aggregators=("h0", "h1", "h2"),
        metadata={"data_shards": 2, "shard_groups": (0, 0, 0)},
    )
    states = {
        "h0": SimpleNamespace(A_i=True),
        "h1": SimpleNamespace(A_i=False),
        "h2": SimpleNamespace(A_i=True),
    }

    assert _assignment_viable(assignment, states)
    states["h2"].A_i = False
    assert not _assignment_viable(assignment, states)


def test_direct_downlink_requires_source_only_until_delivery():
    assignment = SimpleNamespace(
        source="src", helpers=(), aggregators=(),
        metadata={"source_release_time": 100.0, "delivery_time": 120.0},
    )
    states = {"src": SimpleNamespace(A_i=False)}

    assert not _assignment_viable(assignment, states, sim_time=60.0)
    assert _assignment_viable(assignment, states, sim_time=100.0)


def test_ordi_can_select_two_of_three_coded_fanout():
    states = {
        "src": _view("src"),
        "h1": _view("h1"),
        "h2": _view("h2"),
    }
    contacts = (
        ContactWindow("src", "h1", 0.0, 10.0, 1e9, "data"),
        ContactWindow("src", "h2", 0.0, 10.0, 1e9, "data"),
        ContactWindow("src", "ground", 0.0, 10.0, 1e9, "downlink"),
        ContactWindow("h1", "ground", 0.0, 10.0, 1e9, "downlink"),
        ContactWindow("h2", "ground", 0.0, 10.0, 1e9, "downlink"),
    )
    tile = SimpleNamespace(
        tile_id=0, n_replicas_max=1, d_in_bits=100.0,
        d_out_bits=10.0, compute_ops=1_000.0, utility=1.0,
    )
    task = SimpleNamespace(
        task_id=1, source_sat="src", deadline=10.0, tiles=[tile]
    )
    request = EpochInput(
        0, 0.0, [task], states, {}, frozenset({"ground"}), contacts,
        weights=PolicyWeights(
            freshness=0.0, energy=0.0, communication=0.0,
            replication=0.0,
        ),
    )
    scheduler = ORDI(
        max_replicas=1,
        split_options=(1,),
        coded_options=((2, 3),),
    )
    scheduler.messages.seed_knowledge(
        "src", states, generated_at=0.0, delivered_at=0.0
    )

    assignment = scheduler.schedule(request).assignments[0]

    assert assignment.metadata["coded"]
    assert assignment.metadata["data_shards"] == 2
    assert assignment.metadata["fanout_shards"] == 3
    assert assignment.metadata["shard_groups"] == (0, 0, 0)
    assert assignment.metadata["effective_replicas"] == pytest.approx(1.5)
    assert len(assignment.helpers) == 3
    assert sum(assignment.work_fractions) == pytest.approx(1.575)


def test_two_of_three_probability():
    assert ORDI._at_least_k_probability(
        [0.5, 0.5, 0.5], 2
    ) == pytest.approx(0.5)


def test_backup_disjointness_includes_downlink_relays():
    placement = SimpleNamespace(
        route_in=("src", "helper"),
        route_out=("helper", "agg"),
        route_down=("agg", "relay", "ground"),
    )

    nodes = ORDI._route_satellite_nodes(
        placement, "src", frozenset({"ground"})
    )

    assert nodes == {"helper", "agg", "relay"}


def test_sampled_fault_risk_selects_a_cost_effective_disjoint_backup():
    states = {
        name: _view(name)
        for name in ("SAT_00_00", "SAT_01_00")
    }
    contacts = (
        ContactWindow(
            "SAT_00_00", "SAT_01_00", 0.0, 20.0, 1e9, "isl", 0.99
        ),
        ContactWindow(
            "SAT_01_00", "SAT_00_00", 0.0, 20.0, 1e9, "isl", 0.99
        ),
        ContactWindow(
            "SAT_00_00", "ground", 0.0, 20.0, 1e9,
            "downlink", 0.99,
        ),
        ContactWindow(
            "SAT_01_00", "ground", 0.0, 20.0, 1e9,
            "downlink", 0.99,
        ),
    )
    tile = SimpleNamespace(
        tile_id=0, n_replicas_max=2, d_in_bits=100.0,
        d_out_bits=10.0, compute_ops=1_000.0, utility=1.0,
    )
    task = SimpleNamespace(
        task_id=1, source_sat="SAT_00_00", deadline=20.0, tiles=[tile]
    )
    request = EpochInput(
        0, 0.0, [task], states, {}, frozenset({"ground"}), contacts,
        weights=PolicyWeights(
            freshness=0.0, energy=0.0, communication=0.0,
            replication=0.05,
        ),
    )

    low_risk = ORDI(split_options=(1,))
    risk_aware = ORDI(split_options=(1,))
    for _ in range(20):
        low_risk.observe_fault_domain_sample(False)
    for scheduler in (low_risk, risk_aware):
        scheduler.messages.seed_knowledge(
            "SAT_00_00", states, generated_at=-60.0, delivered_at=0.0
        )

    assert len(low_risk.schedule(request).assignments[0].helpers) == 1
    assignment = risk_aware.schedule(request).assignments[0]
    assert len(assignment.helpers) == 2
    assert assignment.metadata["shard_groups"] == (0, 1)
    assert plane(assignment.helpers[0]) != plane(assignment.helpers[1])
    assert assignment.metadata["fault_domain_samples"] == 0


def test_fault_domain_estimate_learns_from_samples():
    scheduler = ORDI()
    assert scheduler.fault_domain_failure_estimate == pytest.approx(0.5)

    for failed in (False, False, False, True):
        scheduler.observe_fault_domain_sample(failed)

    assert scheduler.fault_domain_sample_count == 4
    assert scheduler.fault_domain_failure_estimate == pytest.approx(1.5 / 5)


def test_fault_domain_thompson_samples_are_seeded_and_bounded():
    left = ORDI(rng_seed=7)
    right = ORDI(rng_seed=7)

    samples = [left.sample_fault_domain_failure_risk() for _ in range(4)]

    assert samples == [
        right.sample_fault_domain_failure_risk() for _ in range(4)
    ]
    assert all(0.0 <= sample <= 1.0 for sample in samples)


def test_fault_learning_uses_assignment_outcomes_and_ignores_nonfault_misses():
    scheduler = ORDI(fault_risk_discount=0.5)

    scheduler.observe_assignment_outcome("primary_success")
    scheduler.observe_assignment_outcome("backup_recovery")
    before = scheduler.fault_domain_sample_count
    scheduler.observe_assignment_outcome("nonfault_failure")

    assert scheduler.fault_domain_sample_count == pytest.approx(before)
    assert scheduler._fault_domain_failures == pytest.approx(1.0)
    assert scheduler._fault_domain_successes == pytest.approx(0.5)


def test_fault_risk_beta_parameters_are_positive():
    with pytest.raises(ValueError, match="must be positive"):
        ORDI(fault_risk_alpha=0.0)


def test_cold_start_backup_exploration_is_budgeted():
    scheduler = ORDI(cold_start_backup_budget=1)

    assert scheduler._cold_start_backup_budget == 1
    with pytest.raises(ValueError, match="must be non-negative"):
        ORDI(cold_start_backup_budget=-1)
    with pytest.raises(ValueError, match="must be non-negative"):
        ORDI(min_fault_outcomes=-1)
