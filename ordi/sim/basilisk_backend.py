"""Basilisk/BSK-RL execution adapter used by ORDI.

ORDI deliberately does not implement a second orbit, power, battery, or thermal
model.  This module is the narrow boundary between the scheduler's state-vector
view and the independently maintained Basilisk simulation exposed by BSK-RL.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping

import numpy as np

from Basilisk.simulation import sensorThermal, simplePowerSink
from Basilisk.utilities import orbitalMotion
from bsk_rl import GeneralSatelliteTasking, act, obs
from bsk_rl.sats import ImagingSatellite
from bsk_rl.sim import dyn, fsw, world
from bsk_rl.utils.functional import default_args

from ordi.sim.satellite import SatelliteState


class OrdiDynamics(dyn.GroundStationDynModel):
    """BSK-RL imaging dynamics plus Basilisk workload/thermal instrumentation."""

    @default_args(
        thermal_area_m2=0.01, thermal_absorptivity=0.2,
        thermal_emissivity=0.8, thermal_mass_kg=0.5,
        thermal_specific_heat=900.0, thermal_init_c=25.0,
    )
    def _setup_dynamics_objects(self, **kwargs):
        super()._setup_dynamics_objects(**kwargs)
        self.computeSink = simplePowerSink.SimplePowerSink()
        self.computeSink.ModelTag = "ordiCompute" + self.satellite.name
        self.computeSink.nodePowerOut = 0.0
        self.simulator.AddModelToTask(self.task_name, self.computeSink, ModelPriority=895)
        self.powerMonitor.addPowerNodeToModel(self.computeSink.nodePowerOutMsg)

        self.thermalSensor = sensorThermal.SensorThermal()
        self.thermalSensor.ModelTag = "ordiThermal" + self.satellite.name
        self.thermalSensor.T_0 = kwargs.get("thermal_init_c", 25.0)
        self.thermalSensor.nHat_B = [0, 0, 1]
        self.thermalSensor.sensorArea = kwargs.get("thermal_area_m2", 0.01)
        self.thermalSensor.sensorAbsorptivity = kwargs.get("thermal_absorptivity", 0.2)
        self.thermalSensor.sensorEmissivity = kwargs.get("thermal_emissivity", 0.8)
        self.thermalSensor.sensorMass = kwargs.get("thermal_mass_kg", 0.5)
        self.thermalSensor.sensorSpecificHeat = kwargs.get("thermal_specific_heat", 900.0)
        self.thermalSensor.sensorPowerDraw = 0.0
        self.thermalSensor.sunInMsg.subscribeTo(
            self.world.gravFactory.spiceObject.planetStateOutMsgs[self.world.sun_index]
        )
        self.thermalSensor.stateInMsg.subscribeTo(self.scObject.scStateOutMsg)
        self.thermalSensor.sunEclipseInMsg.subscribeTo(
            self.world.eclipseObject.eclipseOutMsgs[self.eclipse_index]
        )
        self.simulator.AddModelToTask(self.task_name, self.thermalSensor, ModelPriority=894)

    @property
    def temperature_c(self) -> float:
        return float(self.thermalSensor.temperatureOutMsg.read().temperature)

    @property
    def position_m(self):
        return np.asarray(self.r_BN_P, dtype=float)


class OrdiSatellite(ImagingSatellite):
    """BSK-RL satellite definition for an ORDI scheduling policy."""

    observation_spec = [
        obs.SatProperties(
            dict(prop="battery_charge_fraction"),
            dict(prop="storage_level_fraction"),
            dict(prop="temperature_c"),
            dict(prop="position_m", norm=7.0e6),
        ),
        obs.Eclipse(),
        obs.Time(observe_time_remaining=True),
    ]
    action_spec = [act.Charge(duration=60.0), act.Drift(duration=60.0)]
    dyn_type = OrdiDynamics
    fsw_type = fsw.ImagingFSWModel


@dataclass
class Workload:
    compute_flops: float = 0.0
    tx_bits: float = 0.0
    rx_bits: float = 0.0
    downlink_bits: float = 0.0
    stored_bits: float = 0.0


class BasiliskBackend:
    """Own one BSK-RL loop and project its state into ORDI's scheduler view."""

    def __init__(self, sat_ids: Iterable[str], states: Dict[str, SatelliteState],
                 epoch_length_s: float = 60.0, seed: int = 0,
                 ground_stations: list[dict] | None = None):
        self.epoch_length_s = float(epoch_length_s)
        self.states = states
        self.sat_ids = list(sat_ids)
        self._work = {sid: Workload() for sid in self.sat_ids}
        stations = ground_stations or [
            dict(name="fairbanks", lat=64.8378, long=-147.7164, elev=138),
            dict(name="greenwich", lat=51.4769, long=0.0, elev=46),
        ]
        satellites = []
        for index, sid in enumerate(self.sat_ids):
            oe = orbitalMotion.ClassicElements()
            oe.a = (6371.0 + 550.0) * 1e3
            oe.e = 0.001
            oe.i = np.radians(53.0)
            oe.Omega = 2 * np.pi * (index % max(1, len(self.sat_ids))) / max(1, len(self.sat_ids))
            oe.omega = 0.0
            oe.f = 2 * np.pi * index / max(1, len(self.sat_ids))
            p = states[sid].params
            sat = OrdiSatellite(sid, sat_args={
                "oe": oe, "batteryStorageCapacity": p.battery_j,
                "storedCharge_Init": states[sid].B_i,
                "basePowerDraw": -p.idle_power_w,
                "panelArea": max(0.01, p.solar_power_w / (1361.0 * 0.2)),
                "panelEfficiency": 0.2,
                "instrumentPowerDraw": 0.0,
                "transmitterPowerDraw": 0.0,
                "transmitterBaudRate": -100e6,
                "dataStorageCapacity": 20 * 8e9,
                "thermal_area_m2": p.thermal_area_m2,
                "thermal_absorptivity": p.thermal_absorptivity,
                "thermal_emissivity": p.thermal_emissivity,
                "thermal_mass_kg": p.thermal_mass_kg,
                "thermal_specific_heat": p.thermal_specific_heat_j_kg_k,
                "thermal_init_c": states[sid].Theta_i,
            })
            satellites.append(sat)
        self.env = GeneralSatelliteTasking(
            satellites, world_type=(world.GroundStationWorldModel,
                                    world.EclipseWorldModel,
                                    world.AtmosphereWorldModel),
            world_args={"groundStationsData": stations,
                        "gsMinimumElevation": np.radians(25.0)},
            sim_rate=min(1.0, self.epoch_length_s),
            max_step_duration=self.epoch_length_s,
            # BSK-RL precomputes access opportunities through the time limit;
            # keep this finite and aligned with ORDI's configured horizon.
            time_limit=self.epoch_length_s * 512,
            terminate_on_time_limit=False,
            failure_penalty=0.0,
            log_level="ERROR",
        )
        self.env.reset(seed=seed)
        self._sync_projection()

    @property
    def satellites(self):
        return {sat.name: sat for sat in self.env.satellites}

    def _sync_projection(self):
        for sid, sat in self.satellites.items():
            st = self.states[sid]
            d = sat.dynamics
            st.B_i = float(d.battery_charge)
            st.Theta_i = float(d.temperature_c)
            st.C_i = st.params.compute_rate_flops_per_s
            st.A_i = int(sat.is_alive())

    def submit(self, workload: Mapping[str, Workload]):
        """Submit one epoch of compute and packet work, then advance BSK-RL."""
        for sid in self.sat_ids:
            w = workload.get(sid, Workload())
            self._work[sid] = w
            sat = self.satellites[sid]
            d = sat.dynamics
            rate = max(self.states[sid].params.compute_rate_flops_per_s, 1.0)
            compute_time = min(self.epoch_length_s, max(0.0, w.compute_flops) / rate)
            d.computeSink.nodePowerOut = -self.states[sid].params.compute_power_w * compute_time / self.epoch_length_s
            d.computeSink.powerStatus = int(compute_time > 0.0)
            d.thermalSensor.sensorPowerDraw = max(0.0, -d.computeSink.nodePowerOut)
            d.thermalSensor.sensorPowerStatus = int(compute_time > 0.0)
        # Charge is an ordinary BSK-RL action; ORDI itself remains the policy.
        self.env.step(tuple(0 for _ in self.sat_ids))
        self._sync_projection()

    def close(self):
        self.env.close()


__all__ = ["BasiliskBackend", "OrdiSatellite", "OrdiDynamics", "Workload"]
