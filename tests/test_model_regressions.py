from types import SimpleNamespace

import pytest

from ordi.algorithms import (
    Assignment, ContactWindow, Decision, EpochInput, ExperimentConfig,
    SatelliteView,
)
from ordi.algorithms._common import Placement, group_success
from ordi.eval.experiments import (
    _assignment_group_viability, _epoch_input, _simulate_stateful,
)
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
