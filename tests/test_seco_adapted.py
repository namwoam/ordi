from types import SimpleNamespace

import pytest

from ordi.algorithms import (
    Assignment, ContactWindow, EpochInput, SatelliteView, SECOAdapted,
)
from ordi.eval.experiments import _advance_synthetic_states


def _state(name, rate=1e9, battery=10_000.0):
    return SatelliteView(
        name, True, rate, battery, 10_000.0, 25.0, 0.0,
        reliability=0.99,
    )


def _tile(tile_id=0, d_in=100.0, d_out=10.0, work=2e9):
    return SimpleNamespace(
        tile_id=tile_id, n_replicas_max=2, d_in_bits=d_in,
        d_out_bits=d_out, compute_ops=work, utility=1.0,
    )


def test_seco_chooses_parallel_split_without_counting_parts_as_replicas():
    states = {
        "src": _state("src", rate=1e6),
        "h1": _state("h1"),
        "h2": _state("h2"),
    }
    contacts = (
        ContactWindow("src", "h1", 0, 20, 1e6, "isl"),
        ContactWindow("src", "h2", 0, 20, 1e6, "isl"),
        ContactWindow("h1", "src", 0, 20, 1e6, "isl"),
        ContactWindow("h2", "src", 0, 20, 1e6, "isl"),
        ContactWindow("h1", "ground", 0, 20, 1e6, "downlink"),
        ContactWindow("h2", "ground", 0, 20, 1e6, "downlink"),
    )
    task = SimpleNamespace(
        task_id=1, source_sat="src", deadline=20.0, tiles=[_tile()]
    )
    request = EpochInput(
        0, 0.0, [task], states, {}, frozenset({"ground"}), contacts
    )

    scheduler = SECOAdapted()
    scheduler.messages.seed_knowledge(
        "src", states, generated_at=-60.0, delivered_at=0.0
    )
    assignment = scheduler.schedule(request).assignments[0]

    assert assignment.metadata["split_count"] == 2
    assert assignment.metadata["data_shards"] == 2
    assert assignment.metadata["effective_replicas"] == 1.0
    assert len(assignment.helpers) == 2
    assert sum(assignment.work_fractions) == pytest.approx(1.05)


def test_seco_reserves_contact_capacity_between_tiles():
    states = {
        "src": _state("src", rate=1.0),
        "helper": _state("helper"),
    }
    contacts = (
        # One handshake + 800-bit image fits, but two do not.
        ContactWindow("src", "helper", 0, 4.0, 4_000, "isl"),
        ContactWindow("helper", "src", 0, 4.0, 4_000, "isl"),
        ContactWindow("helper", "ground", 0, 20, 1_000, "downlink"),
    )
    task = SimpleNamespace(
        task_id=1, source_sat="src", deadline=20.0,
        tiles=[_tile(0, d_in=800, work=1e6),
               _tile(1, d_in=800, work=1e6)],
    )
    request = EpochInput(
        0, 0.0, [task], states, {}, frozenset({"ground"}), contacts
    )

    scheduler = SECOAdapted(split_options=(1,))
    scheduler.messages.seed_knowledge(
        "src", states, generated_at=-60.0, delivered_at=0.0
    )
    result = scheduler.schedule(request)

    assert len(result.assignments) == 1


def test_partition_fractions_drive_physical_workload():
    # Use lightweight state doubles because this test targets workload
    # translation, not Basilisk state integration.
    params = SimpleNamespace(compute_power_w=10.0, comms_power_w=5.0)
    states = {
        name: SimpleNamespace(params=params)
        for name in ("src", "h1", "h2")
    }
    tile = _tile(work=2e9)
    task = SimpleNamespace(task_id=1, source_sat="src", tiles=[tile])
    assignment = Assignment(
        1, 0, "src", helpers=("h1", "h2"),
        aggregators=("h1", "h2"),
        metadata={"latency": 1.0, "reliability": 1.0,
                  "data_shards": 2, "partitioned": True},
        work_fractions=(0.525, 0.525),
        input_fractions=(0.525, 0.525),
        output_fractions=(0.5, 0.5),
    )

    workloads = _advance_synthetic_states(
        [assignment], [task], states, 60.0
    )

    assert workloads["h1"].compute_flops == pytest.approx(1.05e9)
    assert workloads["h2"].compute_flops == pytest.approx(1.05e9)
    assert workloads["src"].tx_bits == pytest.approx(1.05 * tile.d_in_bits)
