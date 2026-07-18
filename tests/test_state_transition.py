from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ordi.eval.experiments import _advance_synthetic_states, _simulate_stateful
from ordi.faults.injector import FaultEvent, FaultInjector
from ordi.algorithms import Assignment, Decision, ExperimentConfig
from ordi.sim.reliability import ReliabilityModel
from ordi.sim.satellite import SatelliteParams, SatelliteState


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
