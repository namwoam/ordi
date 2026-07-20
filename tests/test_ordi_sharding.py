from dataclasses import replace
from types import SimpleNamespace

import pytest

from ordi.algorithms import (
    ContactWindow, Decision, EpochInput, ORDI, PolicyWeights, SatelliteView,
)
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
