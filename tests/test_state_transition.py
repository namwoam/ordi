from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ordi.eval.experiments import _advance_synthetic_states, _simulate_stateful
from ordi.faults.injector import (
    RANDOM_FAULT_TYPES, FaultEvent, FaultInjector,
)
from ordi.algorithms import Assignment, Decision, ExperimentConfig
from ordi.sim.reliability import ReliabilityModel
from ordi.sim.satellite import SatelliteParams, SatelliteState
from ordi.sim.basilisk_backend import Workload, _communication_power_w
from ordi.orbit._contact_types import DOWNLINK_RATE_BPS, ISL_RATE_BPS
from ordi.sim.basilisk_backend import BasiliskBackend, Workload


def _state(sat_id: str, rate_gflops: float = 1.0) -> SatelliteState:
    params = SatelliteParams(
        sat_id=sat_id,
        compute_rate_gflops=rate_gflops,
        battery_wh=10.0,
        battery_min_frac=0.1,
        thermal_max_c=1_000_000.0,
        solar_power_w=0.0,
        idle_power_w=0.0,
        compute_power_w=10.0,
        comms_power_w=5.0,
        thermal_ambient_c=20.0,
    )
    return SatelliteState(params)


class StateTransitionTests(unittest.TestCase):
    def test_packet_workload_drives_basilisk_communication_power(self):
        params = _state("sat").params
        workload = Workload(
            tx_bits=2.0 * ISL_RATE_BPS,
            rx_bits=3.0 * ISL_RATE_BPS,
            downlink_bits=1.0 * DOWNLINK_RATE_BPS,
        )

        power_w = _communication_power_w(workload, params, 10.0)

        expected_active_seconds = 3.0 + params.rx_power_fraction * 3.0
        self.assertAlmostEqual(
            power_w, params.comms_power_w * expected_active_seconds / 10.0
        )

    def test_straggler_multiplier_is_applied_to_projected_state(self):
        state = _state("sat")
        injector = FaultInjector(
            {"sat": state}, ReliabilityModel(), [], rng_seed=0
        )
        fault = FaultEvent("straggler", 0, 2, ["sat"], {"factor": 0.1})
        injector.schedule(fault)
        injector.apply_epoch(0)

        self.assertAlmostEqual(state.C_i, 0.1e9)
        injector.withdraw_epoch(2)
        self.assertAlmostEqual(state.C_i, 1e9)

    def test_thermal_fault_reduces_and_restores_compute_rate(self):
        state = _state("sat")
        injector = FaultInjector(
            {"sat": state}, ReliabilityModel(), [], rng_seed=0
        )
        fault = FaultEvent(
            "thermal_throttle", 0, 2, ["sat"], {"factor": 0.25}
        )
        injector.schedule(fault)

        injector.apply_epoch(0)
        self.assertAlmostEqual(state.C_i, 0.25e9)
        injector.refresh_active_state()
        self.assertAlmostEqual(state.C_i, 0.25e9)

        injector.withdraw_epoch(2)
        self.assertAlmostEqual(state.C_i, 1e9)

    def test_random_fault_mix_covers_compute_network_and_ground_domains(self):
        assert set(RANDOM_FAULT_TYPES) == {
            "helper_failure", "straggler", "battery_shortage",
            "thermal_throttle", "isl_disruption", "ground_contact_miss",
            "downlink_adverse",
        }

    def test_assignment_load_advances_all_participating_satellites(self):
        states = {name: _state(name) for name in ("src", "helper", "agg")}
        tile = SimpleNamespace(
            tile_id=0, compute_ops=3e9, d_in_bits=200e6, d_out_bits=100e6
        )
        task = SimpleNamespace(task_id=1, source_sat="src", tiles=[tile])
        assignment = Assignment(
            task_id=1, tile_id=0, source="src",
            helpers=("helper",), aggregators=("agg",),
            metadata={"latency": 1.0, "reliability": 1.0},
        )

        batteries_before = {name: state.B_i for name, state in states.items()}
        workloads = _advance_synthetic_states([assignment], [task], states, 1.0)

        self.assertEqual(workloads["helper"].compute_flops, tile.compute_ops)
        self.assertEqual(workloads["src"].tx_bits, tile.d_in_bits)
        self.assertEqual(workloads["agg"].rx_bits, tile.d_out_bits)
        self.assertEqual(workloads["agg"].downlink_bits, tile.d_out_bits)
        self.assertEqual(
            {name: state.B_i for name, state in states.items()}, batteries_before
        )

    def test_backend_compute_queue_persists_and_drives_later_epoch_power(self):
        state = _state("sat", rate_gflops=1.0)
        compute_sink = SimpleNamespace(nodePowerOut=0.0, powerStatus=0)
        communication_sink = SimpleNamespace(
            nodePowerOut=0.0, powerStatus=0,
        )
        thermal_sensor = SimpleNamespace(
            sensorPowerDraw=0.0, sensorPowerStatus=0,
        )
        dynamics = SimpleNamespace(
            computeSink=compute_sink,
            communicationSink=communication_sink,
            thermalSensor=thermal_sensor,
            battery_charge=state.B_i,
            temperature_c=state.Theta_i,
        )

        class FakeSatellite:
            name = "sat"

            def __init__(self):
                self.dynamics = dynamics

            @staticmethod
            def is_alive():
                return True

        fake_satellite = FakeSatellite()
        backend = BasiliskBackend.__new__(BasiliskBackend)
        backend.epoch_length_s = 60.0
        backend.states = {"sat": state}
        backend.sat_ids = ["sat"]
        backend._work = {"sat": Workload()}
        backend.env = SimpleNamespace(
            satellites=[fake_satellite], step=lambda _actions: None,
        )

        backend.submit({"sat": Workload(compute_flops=90e9)})

        self.assertAlmostEqual(state.Q_i, 30e9)
        self.assertAlmostEqual(compute_sink.nodePowerOut, -10.0)
        self.assertAlmostEqual(thermal_sensor.sensorPowerDraw, 10.0)

        backend.submit({})

        self.assertAlmostEqual(state.Q_i, 0.0)
        self.assertAlmostEqual(compute_sink.nodePowerOut, -5.0)
        self.assertAlmostEqual(thermal_sensor.sensorPowerDraw, 5.0)

    def test_synthetic_experiment_loop_calls_state_transition(self):
        state = _state("sat")
        state.params.idle_power_w = 1.0
        battery_before = state.B_i

        submitted = []

        class FakeBackend:
            def __init__(self, *_args, **_kwargs):
                pass

            def submit(self, workloads):
                submitted.append(workloads)

        def empty_schedule(epoch, _tasks):
            return Decision(epoch)

        with patch("ordi.eval.experiments.N_EPOCHS", 2), patch(
            "ordi.sim.basilisk_backend.BasiliskBackend", FakeBackend
        ):
            _simulate_stateful(
                empty_schedule,
                tasks=[],
                sat_ids=["sat"],
                states={"sat": state},
                cfg=ExperimentConfig(epoch_length=60.0),
                realized_trials=0,
            )

        self.assertEqual(len(submitted), 2)
        self.assertEqual(state.B_i, battery_before)


if __name__ == "__main__":
    unittest.main()
