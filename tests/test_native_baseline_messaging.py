from dataclasses import replace
from types import SimpleNamespace

import pytest

from ordi.algorithms import (
    ContactWindow, EpochInput, FullReplication, RandomReplication,
    SatelliteView, SECOAdapted,
)


def _request(epoch=0, sim_time=0.0, helper_available=True):
    states = {
        name: SatelliteView(
            name, helper_available if name == "helper" else True,
            1_000.0, 9_000.0, 10_000.0, 25.0, 0.0,
            compute_power_w=10.0, comms_power_w=5.0, reliability=0.99,
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


@pytest.mark.parametrize("policy_type", [FullReplication, RandomReplication])
def test_replication_policies_natively_discover_and_message_helpers(policy_type):
    scheduler = policy_type()

    first = scheduler.schedule(_request()).assignments[0]
    assert first.helpers == ("src",)
    assert first.metadata["known_state_nodes"] == 1

    informed = scheduler.schedule(
        _request(epoch=1, sim_time=1.0)
    ).assignments[0]
    assert set(informed.helpers) == {"src", "helper"}
    assert informed.metadata["known_state_nodes"] == 2
    kinds = {event.kind for event in informed.message_events}
    assert {
        "replica_request", "replica_accept",
        "image_shard", "result_shard",
    } <= kinds


def test_replication_helper_rejects_stale_assignment_before_image_transfer():
    scheduler = FullReplication()
    scheduler.schedule(_request())

    stale = scheduler.schedule(_request(
        epoch=1, sim_time=1.0, helper_available=False
    )).assignments[0]

    assert stale.helpers == ("src",)
    rejected = [
        event for event in stale.message_events
        if event.kind == "replica_reject" and event.event == "delivered"
    ]
    assert rejected
    assert not any(
        event.kind == "image_shard" and event.node == "src"
        and event.peer == "helper" and event.event == "sent"
        for event in stale.message_events
    )


def test_seco_acceptance_precedes_split_image_transfer():
    request = _request()
    scheduler = SECOAdapted(split_options=(2,))
    scheduler.messages.seed_knowledge(
        "src", request.satellites,
        generated_at=-60.0, delivered_at=0.0,
    )

    assignment = scheduler.schedule(request).assignments[0]
    events = assignment.message_events
    accept_time = max(
        event.time for event in events
        if event.kind == "split_accept" and event.event == "delivered"
    )
    first_remote_image = min(
        event.time for event in events
        if event.kind == "image_shard" and event.event == "sent"
        and event.peer == "helper"
    )

    assert accept_time <= first_remote_image
    assert assignment.metadata["helper_request_kind"] == "split"
