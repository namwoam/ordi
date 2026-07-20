from types import SimpleNamespace

import pytest

from ordi.algorithms import Assignment, Decision, MessageEvent
from ordi.eval.experiments import E1_METRIC_KEYS
from ordi.eval.metrics import compute_metrics
from ordi.eval.plots import E1_PLOT_METRICS


def _event(kind, event, shard):
    return MessageEvent(
        0.0, event, f"m-{kind}-{shard}", kind, "src", "helper",
        10.0, 1, shard, 0, shard,
    )


def test_e1_operational_metrics_are_normalized_and_distribution_aware():
    tiles = [SimpleNamespace(
        tile_id=index, compute_ops=1_000.0, d_in_bits=100.0,
        d_out_bits=20.0, utility=1.0,
    ) for index in range(2)]
    task = SimpleNamespace(
        task_id=1, source_sat="src", tiles=tiles
    )
    assignments = []
    for index, (helper, latency, age) in enumerate((
            ("h1", 10.0, 5.0), ("h2", 30.0, 15.0))):
        events = [_event("replica_request", "sent", index)]
        if index == 0:
            events.append(_event("replica_accept", "delivered", index))
        else:
            events.append(_event("replica_reject", "delivered", index))
        assignments.append(Assignment(
            1, index, "src", helpers=(helper,), aggregators=(helper,),
            metadata={
                "reliability": 1.0, "latency": latency,
                "protocol_header_bits": 10.0,
                "protocol_message_count": 4,
                "protocol_ground_bits": 30.0,
                "handshake_control_bits": 20.0,
                "max_state_age_s": age,
            },
            routes=((
                ("src", helper), (helper,), (helper, "ground"),
            ),),
            message_events=tuple(events),
        ))
    decision = Decision(
        0, tuple(assignments),
        metadata={
            "protocol_message_count": 2,
            "advertisement_control_bits": 20.0,
        },
    )

    metrics = compute_metrics(
        decision, [task], 0.0,
        {"src": 10_000.0, "h1": 10_000.0, "h2": 10_000.0},
        physical_energy_j=20.0,
    )

    assert metrics.delivery_latency_p50_s == pytest.approx(20.0)
    assert metrics.delivery_latency_p95_s == pytest.approx(29.0)
    assert metrics.isl_traffic_bits_per_delivered_tile == pytest.approx(140.0)
    assert metrics.control_traffic_bits_per_delivered_tile == pytest.approx(50.0)
    assert metrics.protocol_messages_per_delivered_tile == pytest.approx(5.0)
    assert metrics.energy_j_per_delivered_tile == pytest.approx(10.0)
    assert metrics.control_traffic_ratio == pytest.approx(100.0 / 340.0)
    assert metrics.active_helper_fraction == pytest.approx(2.0 / 3.0)
    assert metrics.compute_load_balance == pytest.approx(2.0 / 3.0)
    assert metrics.helper_request_count == 2
    assert metrics.helper_acceptance_ratio == pytest.approx(0.5)
    assert metrics.state_age_mean_s == pytest.approx(10.0)
    assert metrics.state_age_p95_s == pytest.approx(14.5)


def test_e1_exports_reliability_latency_cost_and_decentralization_metrics():
    required = {
        "realized_miss_ratio", "delivery_latency_p95_s",
        "isl_traffic_bits_per_delivered_tile", "control_traffic_ratio",
        "energy_j_per_delivered_tile", "compute_load_balance",
        "helper_acceptance_ratio", "state_age_p95_s",
    }
    assert required <= set(E1_METRIC_KEYS)


def test_e1_plot_exposes_important_operational_metrics():
    plotted = {metric for metric, _scale, _title in E1_PLOT_METRICS}

    assert plotted == {
        "realized_miss_ratio",
        "delivery_latency_p95_s",
        "isl_traffic_bits_per_delivered_tile",
        "downlink_bits_per_delivered_tile",
        "energy_j_per_delivered_tile",
        "compute_load_balance",
    }
    assert "isl_traffic_bits" not in plotted
