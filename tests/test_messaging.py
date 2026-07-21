from dataclasses import replace
from types import SimpleNamespace

import pytest

from ordi.algorithms import (
    ContactWindow, EpochInput, ORDI, PolicyWeights, SatelliteView,
)
from ordi.eval.metrics import compute_metrics
from ordi.eval.validation import InvalidDecisionError
from ordi.sim.messaging import MessageSimulator


def _request(*, link_rate=1_000.0, link_close=10.0, include_helper=True):
    names = ("src", "helper") if include_helper else ("src",)
    states = {
        name: SatelliteView(
            name, True, 1_000.0, 9_000.0, 10_000.0, 25.0, 0.0,
            reliability=0.99,
        )
        for name in names
    }
    contacts = []
    if include_helper:
        contacts.append(ContactWindow(
            "src", "helper", 0.0, link_close, link_rate, "isl"
        ))
        contacts.append(ContactWindow(
            "helper", "ground", 0.0, link_close, link_rate, "downlink"
        ))
    contacts.append(ContactWindow(
        "src", "ground", 0.0, link_close, link_rate, "downlink"
    ))
    tile = SimpleNamespace(
        tile_id=0, n_replicas_max=1, d_in_bits=100.0,
        d_out_bits=10.0, compute_ops=1_000.0, utility=1.0,
    )
    task = SimpleNamespace(
        task_id=1, source_sat="src", deadline=10.0, tiles=[tile]
    )
    request = EpochInput(
        0, 0.0, [task], states, {}, frozenset({"ground"}), tuple(contacts),
        weights=PolicyWeights(
            freshness=0.5, energy=0.0, communication=0.0
        ),
    )
    return request, task, tile


def _informed_scheduler(request, **kwargs):
    scheduler = ORDI(**kwargs)
    scheduler.messages.seed_knowledge(
        "src", request.satellites, generated_at=-60.0, delivered_at=0.0
    )
    return scheduler


def test_ordi_executes_node_decisions_through_message_events():
    request, task, tile = _request()

    result = _informed_scheduler(
        request,
        max_replicas=1, split_options=(2,)
    ).schedule(request)
    assignment = result.assignments[0]

    events = assignment.message_events
    root_delivery = next(
        index for index, event in enumerate(events)
        if event.event == "delivered" and event.bits == 0.0
    )
    first_child_send = next(
        index for index, event in enumerate(events)
        if event.event == "sent" and event.kind == "image_shard"
    )
    assert root_delivery < first_child_send
    assert {event.event for event in events} >= {
        "sent", "delivered", "compute_started", "compute_finished",
    }
    assert assignment.metadata["protocol_message_count"] == 4
    assert result.metadata["protocol_message_count"] == 1
    assert assignment.metadata["protocol_control_bits"] > 0
    image_payloads = {
        event.shard_id: event.bits
        for event in events
        if event.event == "sent" and event.kind == "image_shard"
    }
    assert image_payloads == pytest.approx({
        0: tile.d_in_bits * 0.525 + 2048.0,
        1: tile.d_in_bits * 0.525 + 2048.0,
    })

    metrics = compute_metrics(
        result,
        [task], 0.0, {name: 60_000.0 for name in request.satellites},
    )
    assert metrics.protocol_messages == 5
    assert metrics.control_traffic_bits > 0
    assert metrics.downlink_volume_bits == pytest.approx(
        assignment.metadata["protocol_ground_bits"]
    )


def test_message_headers_can_make_nominal_data_route_invalid():
    # The 1,000-bit contact can carry the ten-bit model result, but not the
    # result plus ORDI's 2,048-bit protocol header.
    request, _task, _tile = _request(
        link_rate=100.0, link_close=10.0, include_helper=False
    )

    result = ORDI(
        max_replicas=1, split_options=(1,)
    ).schedule(request)
    assert not result.assignments


def test_duplicate_job_attempt_is_suppressed_within_one_epoch():
    request, _task, _tile = _request()
    scheduler = _informed_scheduler(
        request, max_replicas=1, split_options=(2,)
    )
    scheduler.schedule(request)

    assert not scheduler.schedule(request).assignments


def test_protocol_enforces_hop_and_split_depth_limits():
    request, task, tile = _request()
    assignment = _informed_scheduler(
        request,
        max_replicas=1, split_options=(2,)
    ).schedule(request).assignments[0]

    with pytest.raises(InvalidDecisionError, match="exceeds hop limit"):
        MessageSimulator(max_hops=0).execute(
            request, task, tile, assignment
        )

    too_deep = replace(
        assignment,
        node_decisions=(
            replace(
                assignment.node_decisions[0],
                item=replace(assignment.node_decisions[0].item, depth=4),
            ),
            *assignment.node_decisions[1:],
        ),
    )
    with pytest.raises(InvalidDecisionError, match="split-depth limit"):
        MessageSimulator(max_split_depth=3).execute(
            request, task, tile, too_deep
        )


def test_state_advertisements_are_delayed_and_then_used_locally():
    request, task, _tile = _request(link_rate=10_000.0)
    reverse = ContactWindow(
        "helper", "src", 0.0, 10.0, 10_000.0, "isl"
    )
    warmup = replace(
        request, epoch=0, sim_time=0.0, tasks=[],
        contacts=request.contacts + (reverse,),
    )

    before_arrival = ORDI(
        max_replicas=1, split_options=(1, 2)
    )
    before_arrival.schedule(warmup)
    early = before_arrival.schedule(replace(
        request, epoch=1, sim_time=0.05,
        contacts=warmup.contacts,
    )).assignments[0]
    assert early.metadata["split_count"] == 1
    assert early.metadata["known_state_nodes"] == 1

    after_arrival = ORDI(max_replicas=1, split_options=(2,))
    after_arrival.schedule(warmup)
    informed_result = after_arrival.schedule(replace(
        request, epoch=1, sim_time=0.3,
        contacts=warmup.contacts,
    ))
    informed = informed_result.assignments[0]
    assert informed.metadata["split_count"] == 2
    assert informed.metadata["known_state_nodes"] == 2
    assert informed.metadata["max_state_age_s"] == pytest.approx(0.3)
    assert any(
        event.kind == "state_advertisement"
        for event in informed_result.message_events
    )


def test_local_view_preserves_contacts_through_unknown_relays():
    states = {
        name: SatelliteView(
            name, True, 1_000.0, 9_000.0, 10_000.0, 25.0, 0.0,
            reliability=0.99,
        )
        for name in ("src", "helper")
    }
    contacts = (
        ContactWindow("src", "relay", 0.0, 10.0, 10_000.0, "data"),
        ContactWindow("relay", "helper", 0.0, 10.0, 10_000.0, "data"),
        ContactWindow(
            "helper", "ground", 0.0, 10.0, 10_000.0, "downlink"
        ),
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
    scheduler = ORDI(max_replicas=1, split_options=(1,))
    scheduler.messages.seed_knowledge(
        "src", {"helper": states["helper"]},
        generated_at=0.0, delivered_at=0.0,
    )

    assignment = scheduler.schedule(request).assignments[0]

    assert "relay" in {
        node for route in assignment.routes[0] for node in route
    }
    assert "relay" not in scheduler.messages.local_view(
        request, "src"
    ).satellites


def test_ordi_waits_until_next_epoch_before_retrying():
    state = SatelliteView(
        "src", True, 1_000.0, 9_000.0, 10_000.0, 25.0, 0.0,
        reliability=0.99,
    )
    tile = SimpleNamespace(
        tile_id=0, n_replicas_max=1, d_in_bits=100.0,
        d_out_bits=10.0, compute_ops=1_000.0, utility=1.0,
    )
    task = SimpleNamespace(
        task_id=1, source_sat="src", deadline=180.0, tiles=[tile]
    )
    base = EpochInput(
        0, 0.0, [task], {"src": state}, {},
        frozenset({"ground"}), (),
    )
    scheduler = ORDI(max_replicas=1, split_options=(1,))

    assert not scheduler.schedule(base).assignments
    assert scheduler.waiting[(1, 0)].next_retry_time == pytest.approx(60.0)

    contact = ContactWindow(
        "src", "ground", 30.0, 120.0, 10_000.0, "downlink"
    )
    assert not scheduler.schedule(replace(
        base, epoch=1, sim_time=30.0, contacts=(contact,)
    )).assignments

    retried = scheduler.schedule(replace(
        base, epoch=2, sim_time=60.0, contacts=(contact,)
    ))
    assert retried.assignments
    assert (1, 0) not in scheduler.waiting


def test_ordi_planning_uses_residual_contact_capacity():
    states = {
        "src": SatelliteView(
            "src", True, 1.0, 9_000.0, 10_000.0, 25.0, 0.0,
            reliability=0.99,
        ),
        "helper": SatelliteView(
            "helper", True, 1_000.0, 9_000.0, 10_000.0, 25.0, 0.0,
            reliability=0.99,
        ),
    }
    contacts = (
        # One 100-bit image plus the 2,048-bit header fits; two do not.
        ContactWindow("src", "helper", 0.0, 2.2, 1_000.0, "data"),
        ContactWindow(
            "helper", "ground", 0.0, 20.0, 10_000.0, "downlink"
        ),
    )
    tiles = [
        SimpleNamespace(
            tile_id=index, n_replicas_max=1, d_in_bits=100.0,
            d_out_bits=10.0, compute_ops=1_000.0, utility=1.0,
        )
        for index in range(2)
    ]
    task = SimpleNamespace(
        task_id=1, source_sat="src", deadline=120.0, tiles=tiles
    )
    request = EpochInput(
        0, 0.0, [task], states, {}, frozenset({"ground"}), contacts
    )
    scheduler = ORDI(max_replicas=1, split_options=(1,))
    scheduler.messages.seed_knowledge(
        "src", {"helper": states["helper"]},
        generated_at=0.0, delivered_at=0.0,
    )

    result = scheduler.schedule(request)

    assert len(result.assignments) == 1
    assert scheduler.waiting[(1, 1)].reason == "no_primary_plan"


def test_stale_advertisement_does_not_hide_physical_failure():
    request, _task, _tile = _request()
    reverse = ContactWindow(
        "helper", "src", 0.0, 10.0, 1_000.0, "isl"
    )
    warmup = replace(
        request, epoch=0, sim_time=0.0, tasks=[],
        contacts=request.contacts + (reverse,),
    )
    scheduler = ORDI(max_replicas=1, split_options=(2,))
    scheduler.schedule(warmup)

    failed_helper = replace(request.satellites["helper"], available=False)
    actual_states = dict(request.satellites)
    actual_states["helper"] = failed_helper
    stale_request = replace(
        request, epoch=1, sim_time=2.0, satellites=actual_states,
        contacts=warmup.contacts,
    )

    assert not scheduler.schedule(stale_request).assignments


def test_ordi_keeps_later_model_side_completion_time():
    request, _task, _tile = _request()
    scheduler = _informed_scheduler(
        request, max_replicas=1, split_options=(2,)
    )

    class LaterModel:
        def retime_and_reserve(self, _request, assignment):
            metadata = dict(assignment.metadata)
            metadata["latency"] = float(metadata["latency"]) + 0.5
            return replace(assignment, metadata=metadata)

    scheduler.resources = LaterModel()
    assignment = scheduler.schedule(request).assignments[0]

    protocol_delivery = max(
        event.time for event in assignment.message_events
        if event.event == "delivered" and event.kind == "result_shard"
    )
    assert assignment.metadata["latency"] == pytest.approx(
        protocol_delivery - request.sim_time + 0.5
    )
