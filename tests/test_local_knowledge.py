from dataclasses import replace
from types import SimpleNamespace

import pytest

from ordi.algorithms import (
    ContactWindow, EpochInput, FullReplication, LocalKnowledgeAdapter,
    SatelliteView,
)
from ordi.eval.validation import InvalidDecisionError


def _request(epoch=0, sim_time=0.0, helper_available=True):
    states = {
        name: SatelliteView(
            name, helper_available if name == "helper" else True,
            1_000.0, 9_000.0, 10_000.0, 25.0, 0.0,
            reliability=0.99,
        )
        for name in ("src", "helper")
    }
    contacts = (
        ContactWindow("src", "helper", 0.0, 10.0, 10_000.0, "isl"),
        ContactWindow("helper", "src", 0.0, 10.0, 10_000.0, "isl"),
        ContactWindow("src", "ground", 0.0, 10.0, 10_000.0, "downlink"),
        ContactWindow("helper", "ground", 0.0, 10.0, 10_000.0, "downlink"),
    )
    tile = SimpleNamespace(
        tile_id=0, n_replicas_max=2, d_in_bits=100.0,
        d_out_bits=10.0, compute_ops=1_000.0, utility=1.0,
    )
    task = SimpleNamespace(
        task_id=1, source_sat="src", deadline=10.0, tiles=[tile]
    )
    return EpochInput(
        epoch, sim_time, [task], states, {},
        frozenset({"ground"}), contacts,
    )


def test_replication_baseline_only_uses_advertised_helpers():
    scheduler = LocalKnowledgeAdapter(FullReplication())

    first = scheduler.schedule(_request())
    first_assignment = first.assignments[0]
    assert first_assignment.helpers == ("src",)
    assert first_assignment.metadata["known_state_nodes"] == 1

    informed = scheduler.schedule(_request(epoch=1, sim_time=1.0))
    informed_assignment = informed.assignments[0]
    assert set(informed_assignment.helpers) == {"src", "helper"}
    assert informed_assignment.metadata["known_state_nodes"] == 2
    assert informed_assignment.metadata["max_state_age_s"] == pytest.approx(1.0)


def test_stale_baseline_knowledge_is_checked_against_actual_state():
    scheduler = LocalKnowledgeAdapter(FullReplication())
    scheduler.schedule(_request())

    with pytest.raises(InvalidDecisionError, match="unavailable"):
        scheduler.schedule(_request(
            epoch=1, sim_time=1.0, helper_available=False
        ))


def test_idle_baseline_epoch_still_transmits_advertisements():
    scheduler = LocalKnowledgeAdapter(FullReplication())
    idle = replace(_request(), tasks=[])

    decision = scheduler.schedule(idle)

    assert not decision.assignments
    assert decision.metadata["protocol_message_count"] == 2
    assert decision.metadata["advertisement_control_bits"] == 2048.0
    assert all(
        event.kind == "state_advertisement"
        for event in decision.message_events
    )
