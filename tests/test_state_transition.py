from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ordi.eval.experiments import _advance_synthetic_states, _simulate_stateful
from ordi.faults.injector import FaultEvent, FaultInjector
from ordi.scheduler.feasibility import ReplicaCandidate
from ordi.scheduler.ordi import ORDIConfig, SchedulerResult, TileAssignment
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
        thermal_rc=300.0,
        thermal_resistance_c_per_w=4.0,
        thermal_ambient_c=20.0,
    )
    return SatelliteState(params)


class StateTransitionTests(unittest.TestCase):
    def test_advance_epoch_keeps_excess_work_queued(self):
        state = _state("sat")
        battery_before = state.B_i

        state.advance_epoch(2.0, compute_flops=3e9)

        self.assertAlmostEqual(state.Q_i, 1e9)
        self.assertAlmostEqual(state.B_i, battery_before - 20.0)

    def test_straggler_multiplier_survives_epoch_advance(self):
        state = _state("sat")
        injector = FaultInjector(
            {"sat": state}, ReliabilityModel(), [], rng_seed=0
        )
        fault = FaultEvent("straggler", 0, 2, ["sat"], {"factor": 0.1})
        injector.schedule(fault)
        injector.apply_epoch(0)

        state.advance_epoch(60.0)

        self.assertAlmostEqual(state.C_i, 0.1e9)
        injector.withdraw_epoch(2)
        self.assertAlmostEqual(state.C_i, 1e9)

    def test_assignment_load_advances_all_participating_satellites(self):
        states = {name: _state(name) for name in ("src", "helper", "agg")}
        tile = SimpleNamespace(
            tile_id=0, compute_ops=3e9, d_in_bits=200e6, d_out_bits=100e6
        )
        task = SimpleNamespace(task_id=1, source_sat="src", tiles=[tile])
        helper = states["helper"]
        replica = ReplicaCandidate(
            task_id=1,
            tile_id=0,
            helper="helper",
            aggregator="agg",
            epoch=0,
            latency=1.0,
            p_success=1.0,
            e_compute=helper.energy_for_compute(tile.compute_ops),
            e_rx=helper.energy_for_rx(tile.d_in_bits),
            e_tx=helper.energy_for_tx(tile.d_out_bits),
            feasible=True,
            d_in_bits=tile.d_in_bits,
            d_out_bits=tile.d_out_bits,
        )
        assignment = TileAssignment(
            task_id=1,
            tile_id=0,
            replicas=[replica],
            primary_aggregator="agg",
            z_kv=1.0,
            L_hat=1.0,
        )

        batteries_before = {name: state.B_i for name, state in states.items()}
        _advance_synthetic_states([assignment], [task], states, 1.0)

        self.assertAlmostEqual(states["helper"].Q_i, 2e9)
        self.assertLess(states["src"].B_i, batteries_before["src"])
        self.assertLess(states["helper"].B_i, batteries_before["helper"])
        self.assertLess(states["agg"].B_i, batteries_before["agg"])

    def test_synthetic_experiment_loop_calls_state_transition(self):
        state = _state("sat")
        state.params.idle_power_w = 1.0
        battery_before = state.B_i

        def empty_schedule(epoch, _tasks):
            return SchedulerResult(
                epoch=epoch,
                assignments=[],
                total_utility=0.0,
                energy_penalty=0.0,
                comm_penalty=0.0,
                rep_penalty=0.0,
                objective=0.0,
                link_utilization={},
            )

        with patch("ordi.eval.experiments.N_EPOCHS", 2):
            _simulate_stateful(
                empty_schedule,
                tasks=[],
                sat_ids=["sat"],
                states={"sat": state},
                cfg=ORDIConfig(epoch_length=60.0),
                realized_trials=0,
            )

        self.assertAlmostEqual(state.B_i, battery_before - 120.0)


if __name__ == "__main__":
    unittest.main()
