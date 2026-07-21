from types import SimpleNamespace

import pytest

from ordi.algorithms import Assignment, Decision
from ordi.eval.metrics import compute_realized_metrics
from ordi.faults.injector import FaultEvent, FaultInjector
from ordi.sim.reliability import ReliabilityModel


def test_node_survival_is_sampled_per_reliability_epoch():
    tile = SimpleNamespace(
        tile_id=0, compute_ops=1.0, d_in_bits=1.0,
        d_out_bits=1.0, utility=1.0,
    )
    task = SimpleNamespace(task_id=1, source_sat="src", tiles=[tile])
    assignment = Assignment(
        1, 0, "src", downlink_only=True,
        metadata={
            "reliability": 1.0, "latency": 120.0,
            "scheduled_at": 0.0, "delivery_time": 120.0,
        },
    )
    model = ReliabilityModel(
        default_isl_pi=1.0, default_downlink_pi=1.0,
        default_node_pi=0.5,
    )

    temporal = compute_realized_metrics(
        Decision(0, (assignment,)), [task], model,
        n_trials=20_000, seed=3, reliability_epoch_s=60.0,
    )
    whole_run = compute_realized_metrics(
        Decision(0, (assignment,)), [task], model,
        n_trials=20_000, seed=3, reliability_epoch_s=1_000.0,
    )

    assert temporal.realized_miss_ratio == pytest.approx(0.75, abs=0.015)
    assert whole_run.realized_miss_ratio == pytest.approx(0.5, abs=0.015)


def test_withdrawn_downlink_fault_remains_in_temporal_scoring_history():
    model = ReliabilityModel(
        default_isl_pi=1.0, default_downlink_pi=1.0,
        default_node_pi=1.0,
    )
    injector = FaultInjector({}, model, [])
    injector.schedule(FaultEvent(
        "downlink_adverse", 0, 1, ["agg"], {"pi": 0.0}
    ))
    injector.apply_epoch(0)
    injector.withdraw_epoch(1)
    assert model.downlink_pi("agg") == 1.0
    assert model.downlink_pi("agg", epoch=0) == 0.0

    tile = SimpleNamespace(
        tile_id=0, compute_ops=1.0, d_in_bits=1.0,
        d_out_bits=1.0, utility=1.0,
    )
    task = SimpleNamespace(task_id=1, source_sat="src", tiles=[tile])
    assignment = Assignment(
        1, 0, "src", downlink_only=True, aggregators=("agg",),
        metadata={
            "reliability": 1.0, "latency": 30.0,
            "scheduled_at": 0.0, "delivery_time": 30.0,
        },
    )
    realized = compute_realized_metrics(
        Decision(0, (assignment,)), [task], model,
        n_trials=10, seed=1, reliability_epoch_s=60.0,
    )
    assert realized.realized_miss_ratio == 1.0
