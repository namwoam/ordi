from dataclasses import replace
import random
from types import SimpleNamespace

import pytest

from ordi.algorithms import (
    ContactWindow, EpochInput, FullReplication, RandomReplication,
    SatelliteView, SECOAdapted,
)
from ordi.algorithms._common import Placement
from ordi.algorithms.random_replication import _best_placements_by_helper


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


def _placement(helper, aggregator, latency, comm_bits=100.0,
               reliability=0.99):
    return Placement(
        helper, aggregator, latency, reliability, comm_bits,
        ("src", helper), (helper, aggregator), (aggregator, "ground"),
    )


def test_random_replication_normalizes_routes_before_sampling_helpers():
    slow_first = _placement("h1", "a0", 8.0, comm_bits=50.0)
    fast_later = _placement("h1", "a1", 3.0, comm_bits=100.0)
    other = _placement("h2", "a2", 4.0)
    third = _placement("h3", "a3", 5.0)

    forward = _best_placements_by_helper(
        (slow_first, third, other, fast_later)
    )
    reverse = _best_placements_by_helper(
        (fast_later, other, third, slow_first)
    )

    assert tuple(item.helper for item in forward) == ("h1", "h2", "h3")
    assert forward == reverse
    assert forward[0] is fast_later
    selected_forward = random.Random(7).sample(forward, 2)
    selected_reverse = random.Random(7).sample(reverse, 2)
    assert selected_forward == selected_reverse
    assert len({item.helper for item in selected_forward}) == 2


def test_random_replication_route_ties_use_communication_then_reliability():
    high_traffic = _placement("h1", "a0", 3.0, comm_bits=200.0)
    unreliable = _placement(
        "h1", "a1", 3.0, comm_bits=100.0, reliability=0.8
    )
    reliable = _placement(
        "h1", "a2", 3.0, comm_bits=100.0, reliability=0.95
    )

    selected, = _best_placements_by_helper(
        (high_traffic, unreliable, reliable)
    )

    assert selected is reliable


@pytest.mark.parametrize("policy_type", [FullReplication, RandomReplication])
def test_replication_policies_use_shared_current_state(policy_type):
    scheduler = policy_type()

    first = scheduler.schedule(_request()).assignments[0]
    assert set(first.helpers) == {"src", "helper"}
    assert first.metadata["known_state_nodes"] == 2

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


def test_replication_avoids_currently_unavailable_helper():
    scheduler = FullReplication()
    scheduler.schedule(_request())

    stale = scheduler.schedule(_request(
        epoch=1, sim_time=1.0, helper_available=False
    )).assignments[0]

    assert stale.helpers == ("src",)
    assert not any(
        event.kind == "replica_request" and event.peer == "helper"
        for event in stale.message_events
    )
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


def test_seco_keeps_unknown_downlink_satellites_as_route_only_relays():
    request = _request()
    states = dict(request.satellites)
    states["src"] = replace(states["src"], compute_rate=1.0)
    contacts = (
        ContactWindow("src", "helper", 0.0, 10.0, 10_000.0, "isl"),
        ContactWindow("helper", "src", 0.0, 10.0, 10_000.0, "isl"),
        ContactWindow("helper", "relay", 0.0, 10.0, 10_000.0, "data"),
        ContactWindow("relay", "ground", 0.0, 10.0, 10_000.0, "downlink"),
    )
    request = replace(request, satellites=states, contacts=contacts)
    scheduler = SECOAdapted(split_options=(1,))
    scheduler.messages.seed_knowledge(
        "src", request.satellites,
        generated_at=0.0, delivered_at=0.0,
    )

    assignment = scheduler.schedule(request).assignments[0]

    assert assignment.aggregators == ("helper",)
    assert assignment.routes[0][2] == (
        "helper", "relay", "ground"
    )
