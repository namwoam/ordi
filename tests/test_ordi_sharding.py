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
        compute_power_w=10.0, comms_power_w=5.0, reliability=0.99,
    )


def test_ordi_splits_one_tile_across_two_distinct_helpers():
    states = {name: _view(name) for name in ("src", "helper")}
    contacts = (
        ContactWindow("src", "helper", 0.0, 10.0, 1_000.0, "isl"),
        ContactWindow("src", "ground", 0.0, 10.0, 1_000.0, "downlink"),
        ContactWindow("helper", "ground", 0.0, 10.0, 1_000.0, "downlink"),
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
        ContactWindow("src", "helper", 0.0, 10.0, 1_000.0, "isl"),
        ContactWindow("src", "ground", 0.0, 10.0, 1_000.0, "downlink"),
        ContactWindow("helper", "ground", 0.0, 10.0, 1_000.0, "downlink"),
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
            "src", "ground", 0.0, 10.0, 1_000.0, "downlink"
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
