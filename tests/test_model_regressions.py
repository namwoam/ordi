from types import SimpleNamespace

import pytest

from ordi.algorithms import (
    Assignment, ContactWindow, Decision, EpochInput, ExperimentConfig,
    MessageEvent, SatelliteView,
)
from ordi.algorithms._common import Placement, group_success
from ordi.eval.experiments import (
    _assignment_group_viability, _consumed_attempt_costs, _epoch_input,
    _simulate_stateful,
)
from ordi.eval.metrics import compute_metrics
from ordi.faults.injector import FaultEvent, FaultInjector
from ordi.orbit._contact_types import ContactEvent
from ordi.orbit.graph import build_epoch_graphs
from ordi.sim.basilisk_backend import _service_timed_compute
from ordi.sim.reliability import ReliabilityModel
from ordi.sim.satellite import make_constellation_states
from ordi.tasks.generator import EOTask, Tile
from ordi.tasks.profiles import PROFILES


def _view(name, reliability=1.0):
    return SatelliteView(
        name, True, 1e9, 1e5, 1e5, 25.0, 0.0, reliability
    )


def test_epoch_input_preserves_partial_contact_position():
    event = ContactEvent(50.0, 60.0, "sat", "ground", 1.0, "downlink")
    graphs = build_epoch_graphs([event], 0.0, 60.0, 1)
    states = make_constellation_states(["sat"])

    request = _epoch_input(
        0, [], ["sat"], states, ReliabilityModel(), graphs,
        {"ground"}, ExperimentConfig(),
    )

    assert (request.contacts[0].opens, request.contacts[0].closes) == (50.0, 60.0)


def test_policy_contact_uses_aggregator_downlink_override():
    event = ContactEvent(0.0, 60.0, "sat", "ground", 1.0, "downlink")
    graphs = build_epoch_graphs([event], 0.0, 60.0, 1)
    states = make_constellation_states(["sat"])
    reliability = ReliabilityModel(default_downlink_pi=0.98)
    reliability.set_downlink_pi("sat", 0.1)

    request = _epoch_input(
        0, [], ["sat"], states, reliability, graphs,
        {"ground"}, ExperimentConfig(),
    )

    assert request.contacts[0].reliability == pytest.approx(0.1)


def test_overlapping_failures_do_not_restore_target_early():
    states = make_constellation_states(["sat"])
    reliability = ReliabilityModel()
    injector = FaultInjector(states, reliability, [])
    injector.schedule(FaultEvent("helper_failure", 0, 2, ["sat"]))
    injector.schedule(FaultEvent("helper_failure", 1, 2, ["sat"]))

    injector.apply_epoch(0)
    injector.apply_epoch(1)
    injector.withdraw_epoch(2)

    assert states["sat"].A_i == 0
    assert reliability.node_pi("sat") == 0.0
    injector.withdraw_epoch(3)
    assert states["sat"].A_i == 1
    assert reliability.node_pi("sat") == reliability.default_node_pi


def test_inflight_compute_is_invalidated_when_rate_drops():
    assignment = Assignment(
        1, 0, "src", helpers=("helper",), aggregators=("helper",),
        routes=((('src', 'helper'), ('helper',), ('helper', 'ground')),),
        metadata={
            "delivery_time": 180.0,
            "replica_phase_ends": ((30.0, 120.0, 120.0, 180.0),),
            "replica_compute_rates": (1e9,),
        },
    )
    states = {
        "src": SimpleNamespace(A_i=1, C_i=1e9),
        "helper": SimpleNamespace(A_i=1, C_i=0.5e9),
    }

    assert not any(
        _assignment_group_viability(
            assignment, states, sim_time=60.0
        ).values()
    )


def test_unavailable_timed_compute_keeps_all_flops_queued():
    schedule = [(0.0, 60.0, 60e9, (1, 0))]

    remaining, active = _service_timed_compute(
        schedule, 0.0, 60.0, 1e9, available=False
    )

    assert active == 0.0
    assert remaining[0][2] == pytest.approx(60e9)


def test_canceled_attempt_costs_are_retained_in_metrics():
    assignment = Assignment(
        1, 0, "src", helpers=("helper",), aggregators=("helper",),
        metadata={
            "data_transfer_records": (
                (100.0, 5.0,
                 (("src", "helper", 0.0, 10.0, "isl"),)),
                (50.0, 5.0,
                 (("helper", "ground", 100.0, 110.0, "downlink"),)),
            ),
            "control_transfer_records": (
                (10.0, (("src", "helper", 0.0, 10.0, "isl"),)),
            ),
            "compute_intervals": (("helper", 30.0, 90.0, 60.0),),
        },
        message_events=(MessageEvent(
            0.0, "hop_sent", "m1", "replica_request", "src", "helper", 10.0
        ),),
    )
    consumed = _consumed_attempt_costs(assignment, 60.0)
    assert consumed["isl_traffic_bits"] == pytest.approx(110.0)
    assert consumed["downlink_volume_bits"] == 0.0
    assert consumed["control_traffic_bits"] == pytest.approx(15.0)
    assert consumed["compute_by_helper"]["helper"] == pytest.approx(30.0)

    profile = PROFILES["wildfire"]
    tile = Tile(1, 0, profile, 1.0, 1.0, 1.0, 1.0, 0, 0)
    task = EOTask(1, "src", 0.0, 120.0, "wildfire", [tile], 1)
    delivered = Assignment(
        1, 0, "src", downlink_only=True,
        metadata={"reliability": 1.0, "latency": 1.0, "downlink_bits": 0.0},
    )
    metrics = compute_metrics(
        Decision(1, (delivered,), metadata={"abandoned_costs": consumed}),
        [task], 0.0, {"src": 0.0, "helper": 100.0},
    )
    assert metrics.isl_traffic_bits == pytest.approx(110.0)
    assert metrics.control_traffic_bits == pytest.approx(15.0)
    assert metrics.protocol_messages == pytest.approx(1.0)
    assert metrics.helper_utilization == pytest.approx(0.3)


def test_stateful_replan_carries_canceled_attempt_traffic():
    profile = PROFILES["wildfire"]
    tile = Tile(1, 0, profile, 100.0, 10.0, 80.0, 1.0, 0, 0)
    task = EOTask(1, "src", 0.0, 180.0, "wildfire", [tile], 1)
    states = make_constellation_states(["src", "helper"])
    helper_rate = states["helper"].C_i

    first = Assignment(
        1, 0, "src", helpers=("helper",), aggregators=("helper",),
        metadata={
            "reliability": 1.0, "latency": 120.0,
            "delivery_time": 120.0, "scheduled_at": 0.0,
            "replica_phase_ends": ((10.0, 90.0, 90.0, 120.0),),
            "replica_compute_rates": (helper_rate,),
            "data_transfer_records": (
                (100.0, (("src", "helper", 0.0, 10.0, "isl"),)),
                (10.0, (("helper", "ground", 100.0, 110.0,
                         "downlink"),)),
            ),
            "compute_intervals": (("helper", 10.0, 90.0, 80.0),),
        },
        routes=((('src', 'helper'), ('helper',), ('helper', 'ground')),),
    )
    replacement = Assignment(
        1, 0, "src", helpers=("src",), aggregators=("src",),
        metadata={
            "reliability": 1.0, "latency": 0.0,
            "delivery_time": 60.0, "scheduled_at": 60.0,
        },
        routes=((('src',), ('src',), ('src', 'ground')),),
    )

    def schedule(epoch, tasks):
        if not tasks:
            return Decision(epoch)
        return Decision(epoch, (first if epoch == 0 else replacement,))

    def drive(epoch, _sim_time, current_states):
        if epoch == 1:
            current_states["helper"].inject_failure()

    metrics = _simulate_stateful(
        schedule, [task], ["src", "helper"], states,
        ExperimentConfig(epoch_length=60.0, simulation_epochs=2),
        realized_trials=0, state_driver=drive,
    )

    assert metrics.deadline_miss_ratio == 0.0
    assert metrics.isl_traffic_bits == pytest.approx(100.0)


def test_post_arrival_drain_epochs_schedule_late_tasks():
    profile = PROFILES["wildfire"]
    tile = Tile(1, 0, profile, 1.0, 1.0, 1.0, 1.0, 0, 0)
    task = EOTask(1, "sat", 60.0, 120.0, "wildfire", [tile], 1)
    states = make_constellation_states(["sat"])
    seen_epochs = []

    def schedule(epoch, tasks):
        seen_epochs.append((epoch, len(tasks)))
        if not tasks:
            return Decision(epoch)
        assignment = Assignment(
            1, 0, "sat", helpers=("sat",), aggregators=("sat",),
            metadata={
                "reliability": 1.0, "latency": 0.0,
                "delivery_time": 60.0, "scheduled_at": 60.0,
            },
            routes=((('sat',), ('sat',), ('sat', 'ground')),),
        )
        return Decision(epoch, (assignment,))

    metrics = _simulate_stateful(
        schedule, [task], ["sat"], states,
        ExperimentConfig(epoch_length=60.0, simulation_epochs=2),
        realized_trials=0,
        state_driver=lambda *_args: None,
    )

    assert seen_epochs == [(0, 0), (1, 1)]
    assert metrics.deadline_miss_ratio == 0.0


def test_modeled_reliability_counts_time_and_intermediate_relays():
    states = {
        "src": _view("src"),
        "relay": _view("relay", reliability=0.5),
        "helper": _view("helper"),
    }
    contacts = (
        ContactWindow("src", "relay", 0, 120, 1e9, "isl", 1.0),
        ContactWindow("relay", "helper", 0, 120, 1e9, "isl", 1.0),
        ContactWindow("helper", "ground", 0, 120, 1e9, "downlink", 1.0),
    )
    request = EpochInput(
        0, 0.0, [], states, {}, frozenset({"ground"}), contacts,
        epoch_length=60.0,
    )
    task = SimpleNamespace(source_sat="src")
    placement = Placement(
        "helper", "helper", 120.0, 1.0, 0.0,
        ("src", "relay", "helper"), ("helper",),
        ("helper", "ground"),
        0.0, 120.0, 120.0, 120.0, 120.0,
    )

    assert group_success(request, task, (placement,)) == pytest.approx(0.25)
