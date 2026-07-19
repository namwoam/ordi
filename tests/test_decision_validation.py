from dataclasses import replace
from types import SimpleNamespace

import pytest

from ordi.algorithms import (
    Assignment, ContactWindow, Decision, DirectDownlink, EpochInput,
    FullReplication, OnboardOnly, ORDI, RandomReplication, SatelliteView,
)
from ordi.eval.experiments import _validate_feasible_subset
from ordi.eval.validation import DecisionFeasibilityModel, InvalidDecisionError


def _request(contact_bits=100.0):
    states = {
        sid: SatelliteView(sid, True, 1_000.0, 9_000.0, 10_000.0, 25.0, 0.0)
        for sid in ("src", "relay", "h1", "h2")
    }
    contacts = (
        ContactWindow("src", "relay", 0.0, 1.0, contact_bits, "isl"),
        ContactWindow("relay", "h1", 0.0, 10.0, 1_000.0, "isl"),
        ContactWindow("relay", "h2", 0.0, 10.0, 1_000.0, "isl"),
        ContactWindow("h1", "ground", 0.0, 10.0, 1_000.0, "downlink"),
        ContactWindow("h2", "ground", 0.0, 10.0, 1_000.0, "downlink"),
    )
    tile = SimpleNamespace(
        tile_id=0, d_in_bits=60.0, d_out_bits=10.0,
        compute_ops=10.0,
    )
    task = SimpleNamespace(
        task_id=1, source_sat="src", deadline=10.0, tiles=[tile]
    )
    return EpochInput(
        0, 0.0, [task], states, {}, frozenset({"ground"}), contacts
    )


def _replicated_decision():
    return Decision(0, (Assignment(
        1, 0, "src", helpers=("h1", "h2"), aggregators=("h1", "h2"),
        metadata={"latency": 1.0, "reliability": 1.0},
        routes=(
            (("src", "relay", "h1"), ("h1",), ("h1", "ground")),
            (("src", "relay", "h2"), ("h2",), ("h2", "ground")),
        ),
    ),))


def test_model_rejects_replication_that_oversubscribes_a_shared_contact():
    with pytest.raises(InvalidDecisionError, match="no residual contact capacity"):
        DecisionFeasibilityModel().validate_and_reserve(
            _request(contact_bits=100.0), _replicated_decision()
        )


def test_model_accepts_replication_when_shared_contact_has_capacity():
    decision = _replicated_decision()
    accepted = DecisionFeasibilityModel().validate_and_reserve(
        _request(contact_bits=200.0), decision
    )
    assert accepted is decision


def test_retiming_preserves_decision_metadata_and_events():
    decision = replace(
        _replicated_decision(),
        metadata={"protocol_message_count": 3},
        message_events=(),
    )

    accepted = DecisionFeasibilityModel().validate_and_reserve(
        _request(contact_bits=200.0), decision, retime=True
    )

    assert accepted.metadata == decision.metadata
    assert accepted.message_events == decision.message_events


def test_model_side_admission_drops_only_invalid_assignments():
    decision = replace(
        _replicated_decision(),
        metadata={"protocol_message_count": 3},
    )

    accepted = _validate_feasible_subset(
        DecisionFeasibilityModel(),
        _request(contact_bits=100.0),
        decision,
    )

    assert not accepted.assignments
    assert accepted.metadata == decision.metadata


@pytest.mark.parametrize(
    "algorithm_type",
    [ORDI, DirectDownlink, OnboardOnly, FullReplication, RandomReplication],
)
def test_policies_retime_multiple_tiles_before_model_validation(algorithm_type):
    states = {
        sid: SatelliteView(
            sid, True, 1_000.0, 9_000.0, 10_000.0, 25.0, 0.0,
            reliability=0.99,
        )
        for sid in ("src", "h1", "h2")
    }
    contacts = tuple(
        [ContactWindow("src", helper, 0.0, 10.0, 1_000.0, "isl")
         for helper in ("h1", "h2")]
        + [ContactWindow(sid, "ground", 0.0, 10.0, 1_000.0, "downlink")
           for sid in states]
    )
    tiles = [
        SimpleNamespace(
            tile_id=index, n_replicas_max=2, d_in_bits=100.0,
            d_out_bits=10.0, compute_ops=10.0, utility=1.0,
        )
        for index in range(2)
    ]
    task = SimpleNamespace(
        task_id=1, source_sat="src", deadline=10.0, tiles=tiles
    )
    request = EpochInput(
        0, 0.0, [task], states, {}, frozenset({"ground"}), contacts
    )

    decision = algorithm_type().schedule(request)

    assert len(decision.assignments) == 2
    DecisionFeasibilityModel().validate_and_reserve(request, decision)
