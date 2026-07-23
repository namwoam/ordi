from types import SimpleNamespace

import ordi.algorithms._common as common
from ordi.algorithms import ContactWindow, EpochInput, SatelliteView


def _request():
    satellites = {
        name: SatelliteView(
            name, True, 1_000_000.0, 9_000.0, 10_000.0, 25.0, 0.0,
            reliability=0.99,
        )
        for name in ("src", "h1", "h2")
    }
    contacts = (
        ContactWindow("src", "h1", 0.0, 20.0, 1_000_000.0, "isl"),
        ContactWindow("src", "h2", 0.0, 20.0, 1_000_000.0, "isl"),
        ContactWindow("h1", "h2", 0.0, 20.0, 1_000_000.0, "isl"),
        ContactWindow("h2", "h1", 0.0, 20.0, 1_000_000.0, "isl"),
        ContactWindow("h1", "src", 0.0, 20.0, 1_000_000.0, "isl"),
        ContactWindow("h2", "src", 0.0, 20.0, 1_000_000.0, "isl"),
        *(
            ContactWindow(
                name, "ground", 0.0, 20.0, 1_000_000.0, "downlink"
            )
            for name in satellites
        ),
    )
    return EpochInput(
        0, 0.0, [], satellites, {}, frozenset({"ground"}), contacts,
    )


def test_batched_routes_match_independent_target_searches():
    request = _request()
    targets = set(request.satellites)

    batched = common.earliest_routes(
        request, "src", targets, 1_000.0, latest=20.0
    )

    assert batched == {
        target: common.earliest_route(
            request, "src", {target}, 1_000.0, latest=20.0
        )
        for target in targets
    }


def test_enumeration_batches_input_and_aggregator_route_searches(monkeypatch):
    request = _request()
    tile = SimpleNamespace(
        tile_id=0, d_in_bits=1_000.0, d_out_bits=100.0,
        compute_ops=1_000.0, utility=1.0,
    )
    task = SimpleNamespace(
        task_id=1, source_sat="src", deadline=20.0, tiles=[tile],
    )
    calls = []
    original = common.earliest_routes

    def traced(request, source, targets, bits, start=None, latest=None):
        calls.append((source, frozenset(targets)))
        return original(request, source, targets, bits, start, latest)

    monkeypatch.setattr(common, "earliest_routes", traced)

    placements = common.enumerate_placements(request, task, tile)

    all_satellites = frozenset(request.satellites)
    assert len(placements) == 9
    assert calls == [
        ("src", all_satellites),
        ("src", all_satellites),
        ("h1", all_satellites),
        ("h2", all_satellites),
    ]


def test_identical_batched_route_query_reuses_cached_result(monkeypatch):
    request = _request()
    common._ROUTE_CACHE.clear()
    calls = 0
    original = common._contacts_by_source

    def traced(contacts):
        nonlocal calls
        calls += 1
        return original(contacts)

    monkeypatch.setattr(common, "_contacts_by_source", traced)
    arguments = (
        request, "src", frozenset(request.satellites), 1_000.0
    )

    first = common.earliest_routes(*arguments, latest=20.0)
    second = common.earliest_routes(*arguments, latest=20.0)

    assert second == first
    assert calls == 1
