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

from ordi.orbit._contact_types import DOWNLINK_RATE_BPS, ISL_RATE_BPS
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

        # ORDI packet traffic does not use BSK-RL's ground-only downlink action.
        # A dedicated Basilisk node makes all packet work part of the native
        # battery integration without teaching the scheduler any power model.
        self.communicationSink = simplePowerSink.SimplePowerSink()
        self.communicationSink.ModelTag = "ordiCommunication" + self.satellite.name
        self.communicationSink.nodePowerOut = 0.0
        self.simulator.AddModelToTask(
            self.task_name, self.communicationSink, ModelPriority=894
        )
        self.powerMonitor.addPowerNodeToModel(
            self.communicationSink.nodePowerOutMsg
        )

        self.faultSink = simplePowerSink.SimplePowerSink()
        self.faultSink.ModelTag = "ordiFault" + self.satellite.name
        self.faultSink.nodePowerOut = 0.0
        self.simulator.AddModelToTask(
            self.task_name, self.faultSink, ModelPriority=893
        )
        self.powerMonitor.addPowerNodeToModel(self.faultSink.nodePowerOutMsg)

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
        self.simulator.AddModelToTask(self.task_name, self.thermalSensor, ModelPriority=892)

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
    fault_power_w: float = 0.0
    fault_heat_w: float = 0.0
    # Absolute simulation-time intervals reserved by the shared contact ledger.
    # They remain pending in BasiliskBackend until their epoch is simulated.
    tx_intervals: tuple[tuple[float, float], ...] = ()
    rx_intervals: tuple[tuple[float, float], ...] = ()
    tx_interval_owners: tuple[object, ...] = ()
    rx_interval_owners: tuple[object, ...] = ()
    # (start, finish, FLOPs, owner). Compute is not eligible before start,
    # which is the shared ledger's input-delivery completion time.
    compute_intervals: tuple[tuple[float, float, float, object], ...] = ()


def _service_compute_queue(queued_flops: float, incoming_flops: float,
                           compute_rate_flops_per_s: float,
                           epoch_length_s: float,
                           available: bool = True) -> tuple[float, float]:
    """Return ``(remaining_flops, active_seconds)`` for one FIFO epoch.

    Work admitted during the current scheduling epoch joins unfinished work
    from earlier epochs.  An unavailable accelerator performs no service, and
    otherwise can retire at most ``rate * epoch_length`` FLOPs.  The active
    duration drives Basilisk's average compute power and thermal input.
    """
    backlog = max(0.0, queued_flops) + max(0.0, incoming_flops)
    rate = max(0.0, compute_rate_flops_per_s)
    duration = max(0.0, epoch_length_s)
    if not available or rate <= 0.0 or duration <= 0.0:
        return backlog, 0.0
    completed = min(backlog, rate * duration)
    return max(0.0, backlog - completed), completed / rate


def _interval_active_seconds(intervals, start: float, end: float) -> float:
    """Union length of intervals overlapping [start, end)."""
    clipped = sorted(
        (max(start, interval[0]), min(end, interval[1]))
        for interval in intervals
        if interval[1] > start and interval[0] < end
    )
    total = 0.0
    cursor_start = cursor_end = None
    for begin, finish in clipped:
        if finish <= begin:
            continue
        if cursor_end is None or begin > cursor_end:
            if cursor_end is not None:
                total += cursor_end - cursor_start
            cursor_start, cursor_end = begin, finish
        else:
            cursor_end = max(cursor_end, finish)
    if cursor_end is not None:
        total += cursor_end - cursor_start
    return total


def _communication_power_w(workload: Workload, params,
                           epoch_length_s: float,
                           epoch_start_s: float = 0.0) -> float:
    """Average radio/terminal load presented to Basilisk for one epoch."""
    duration = max(0.0, epoch_length_s)
    if duration == 0.0:
        return 0.0
    epoch_end = epoch_start_s + duration
    if workload.tx_intervals or workload.rx_intervals:
        tx_time = _interval_active_seconds(
            workload.tx_intervals, epoch_start_s, epoch_end
        )
        rx_time = _interval_active_seconds(
            workload.rx_intervals, epoch_start_s, epoch_end
        )
    else:
        # Compatibility path for callers without contact-ledger timings.
        tx_time = min(
            duration,
            max(0.0, workload.tx_bits) / ISL_RATE_BPS
            + max(0.0, workload.downlink_bits) / DOWNLINK_RATE_BPS,
        )
        rx_time = min(
            duration,
            max(0.0, workload.rx_bits) / ISL_RATE_BPS,
        )
    return params.comms_power_w * (
        tx_time + params.rx_power_fraction * rx_time
    ) / duration


class BasiliskBackend:
    """Own one BSK-RL loop and project its state into ORDI's scheduler view."""

    def __init__(self, sat_ids: Iterable[str], states: Dict[str, SatelliteState],
                 epoch_length_s: float = 60.0, seed: int = 0,
                 ground_stations=None, n_planes: int = 6,
                 sats_per_plane: int = 6, orbit_altitude_km: float = 550.0,
                 orbit_inclination_deg: float = 53.0,
                 min_elevation_deg: float = 25.0):
        self.epoch_length_s = float(epoch_length_s)
        self.states = states
        self.sat_ids = list(sat_ids)
        self._work = {sid: Workload() for sid in self.sat_ids}
        self._communication_schedule = {
            sid: {"tx": [], "rx": []} for sid in self.sat_ids
        }
        self._compute_schedule = {sid: [] for sid in self.sat_ids}
        self._scheduled_compute_remaining = {sid: 0.0 for sid in self.sat_ids}
        self._sim_time_s = 0.0
        if ground_stations:
            stations = [
                dict(name=name, lat=lat, long=lon, elev=0.0)
                for name, lat, lon in ground_stations
            ]
        else:
            stations = [
                dict(name="fairbanks", lat=64.8378, long=-147.7164, elev=138),
                dict(name="greenwich", lat=51.4769, long=0.0, elev=46),
            ]
        satellites = []
        total = max(1, n_planes * sats_per_plane)
        for index, sid in enumerate(self.sat_ids):
            try:
                _prefix, plane_text, slot_text = sid.rsplit("_", 2)
                plane = int(plane_text)
                slot = int(slot_text)
            except (TypeError, ValueError):
                plane = index // max(1, sats_per_plane)
                slot = index % max(1, sats_per_plane)
            oe = orbitalMotion.ClassicElements()
            oe.a = (6371.0 + orbit_altitude_km) * 1e3
            oe.e = 0.001
            oe.i = np.radians(orbit_inclination_deg)
            oe.Omega = 2 * np.pi * plane / max(1, n_planes)
            oe.omega = 0.0
            oe.f = np.radians(
                (360.0 * slot / max(1, sats_per_plane)
                 + 360.0 * plane / total) % 360.0
            )
            p = states[sid].params
            sat = OrdiSatellite(sid, sat_args={
                "oe": oe, "batteryStorageCapacity": p.battery_j,
                "storedCharge_Init": states[sid].B_i,
                "basePowerDraw": -p.idle_power_w,
                "panelArea": max(0.01, p.solar_power_w / (1361.0 * 0.2)),
                "panelEfficiency": 0.2,
                "instrumentPowerDraw": 0.0,
                # BSK-RL's native sink is tied only to its ground-downlink
                # action. ORDI drives the dedicated Basilisk communicationSink.
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
            world_args={
                "groundStationsData": stations,
                "gsMinimumElevation": np.radians(min_elevation_deg),
                # Contact generation uses Unix t=0 and a matching deterministic
                # SGP4 epoch; fixing Basilisk's epoch aligns eclipse/solar phase.
                "utc_init": "1970 JAN 01 00:00:00.000 (UTC)",
            },
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
            st.project_environment(
                float(d.battery_charge), float(d.temperature_c), sat.is_alive()
            )

    def submit(self, workload: Mapping[str, Workload]) -> float:
        """Enqueue one epoch of work, advance BSK-RL, and return load energy.

        Compute exceeding the effective capacity of this epoch remains in
        ``SatelliteState.Q_i``.  Later epochs continue servicing that backlog
        even when they admit no new work, so battery and thermal state reflect
        sustained accelerator duty rather than only the newest assignments.
        Contact-ledger intervals drive a persistent Basilisk communication
        power node; untimed compatibility workloads are queued by duration.
        No scheduling policy estimates communication or compute joules.
        """
        workload_energy_j = 0.0
        if not hasattr(self, "_communication_schedule"):
            self._communication_schedule = {
                sid: {"tx": [], "rx": []} for sid in self.sat_ids
            }
        if not hasattr(self, "_compute_schedule"):
            self._compute_schedule = {sid: [] for sid in self.sat_ids}
        if not hasattr(self, "_scheduled_compute_remaining"):
            self._scheduled_compute_remaining = {
                sid: 0.0 for sid in self.sat_ids
            }
        epoch_start = getattr(self, "_sim_time_s", 0.0)
        epoch_end = epoch_start + self.epoch_length_s
        for sid in self.sat_ids:
            w = workload.get(sid, Workload())
            self._work[sid] = w
            sat = self.satellites[sid]
            d = sat.dynamics
            state = self.states[sid]
            compute_schedule = self._compute_schedule[sid]
            previous_scheduled = self._scheduled_compute_remaining[sid]
            external_backlog = max(0.0, state.Q_i - previous_scheduled)
            if w.compute_intervals:
                compute_schedule.extend(w.compute_intervals)
            else:
                external_backlog += max(0.0, w.compute_flops)
            external_remaining, external_time = _service_compute_queue(
                external_backlog,
                0.0,
                state.C_i,
                self.epoch_length_s,
                available=bool(state.A_i),
            )
            timed_compute_time = _interval_active_seconds(
                compute_schedule, epoch_start, epoch_end
            ) if state.A_i else 0.0
            compute_time = min(
                self.epoch_length_s, external_time + timed_compute_time
            )
            remaining_schedule = []
            scheduled_remaining = 0.0
            for interval in compute_schedule:
                start, finish, flops = interval[:3]
                if finish <= epoch_end + 1e-9:
                    continue
                duration = max(finish - start, 1e-12)
                remaining_fraction = max(
                    0.0, min(1.0, (finish - max(epoch_end, start)) / duration)
                )
                remaining_schedule.append(interval)
                scheduled_remaining += flops * remaining_fraction
            self._compute_schedule[sid] = remaining_schedule
            self._scheduled_compute_remaining[sid] = scheduled_remaining
            state.Q_i = external_remaining + scheduled_remaining
            duty_cycle = (
                compute_time / self.epoch_length_s
                if self.epoch_length_s > 0.0 else 0.0
            )
            d.computeSink.nodePowerOut = (
                -state.params.compute_power_w * duty_cycle
            )
            d.computeSink.powerStatus = int(compute_time > 0.0)
            schedule = self._communication_schedule[sid]
            if w.tx_intervals or w.rx_intervals:
                schedule["tx"].extend(
                    (*interval, owner)
                    for interval, owner in zip(
                        w.tx_intervals,
                        w.tx_interval_owners
                        or (None,) * len(w.tx_intervals),
                    )
                )
                schedule["rx"].extend(
                    (*interval, owner)
                    for interval, owner in zip(
                        w.rx_intervals,
                        w.rx_interval_owners
                        or (None,) * len(w.rx_intervals),
                    )
                )
            else:
                # Untimed callers still get persistent service instead of
                # losing traffic beyond the current epoch.
                cursor = max(
                    epoch_start,
                    max((interval[1] for interval in schedule["tx"]),
                        default=epoch_start),
                )
                tx_seconds = max(0.0, w.tx_bits) / ISL_RATE_BPS
                if tx_seconds:
                    schedule["tx"].append((cursor, cursor + tx_seconds))
                    cursor += tx_seconds
                down_seconds = (
                    max(0.0, w.downlink_bits) / DOWNLINK_RATE_BPS
                )
                if down_seconds:
                    schedule["tx"].append((cursor, cursor + down_seconds))
                rx_seconds = max(0.0, w.rx_bits) / ISL_RATE_BPS
                if rx_seconds:
                    rx_start = max(
                        epoch_start,
                        max((interval[1] for interval in schedule["rx"]),
                            default=epoch_start),
                    )
                    schedule["rx"].append(
                        (rx_start, rx_start + rx_seconds)
                    )
            scheduled_work = Workload(
                tx_intervals=tuple(schedule["tx"]),
                rx_intervals=tuple(schedule["rx"]),
            )
            communication_power = _communication_power_w(
                scheduled_work, state.params, self.epoch_length_s,
                epoch_start_s=epoch_start,
            )
            d.communicationSink.nodePowerOut = -communication_power
            d.communicationSink.powerStatus = int(communication_power > 0.0)
            fault_sink = getattr(d, "faultSink", None)
            if fault_sink is not None:
                fault_sink.nodePowerOut = -max(0.0, w.fault_power_w)
                fault_sink.powerStatus = int(w.fault_power_w > 0.0)
            payload_power = max(0.0, -d.computeSink.nodePowerOut) + max(
                0.0, -d.communicationSink.nodePowerOut
            )
            thermal_power = payload_power + max(0.0, w.fault_heat_w)
            d.thermalSensor.sensorPowerDraw = thermal_power
            d.thermalSensor.sensorPowerStatus = int(thermal_power > 0.0)
            workload_energy_j += (
                payload_power + max(0.0, w.fault_power_w)
            ) * self.epoch_length_s
            schedule["tx"] = [
                interval for interval in schedule["tx"]
                if interval[1] > epoch_end + 1e-9
            ]
            schedule["rx"] = [
                interval for interval in schedule["rx"]
                if interval[1] > epoch_end + 1e-9
            ]
        # Charge is an ordinary BSK-RL action; ORDI itself remains the policy.
        self.env.step(tuple(0 for _ in self.sat_ids))
        self._sim_time_s = epoch_end
        self._sync_projection()
        return workload_energy_j

    def cancel(self, owner, sim_time: float | None = None) -> None:
        """Remove an abandoned assignment's unexecuted physical workload."""
        cutoff = self._sim_time_s if sim_time is None else float(sim_time)
        for sid in self.sat_ids:
            for channel in ("tx", "rx"):
                self._communication_schedule[sid][channel] = [
                    interval for interval in self._communication_schedule[sid][channel]
                    if len(interval) < 3 or interval[2] != owner
                    or interval[0] < cutoff
                ]
            self._compute_schedule[sid] = [
                interval for interval in self._compute_schedule[sid]
                if len(interval) < 4 or interval[3] != owner
                or interval[0] < cutoff
            ]
            remaining = 0.0
            for interval in self._compute_schedule[sid]:
                start, finish, flops = interval[:3]
                duration = max(finish - start, 1e-12)
                remaining += flops * max(
                    0.0, min(1.0, (finish - max(cutoff, start)) / duration)
                )
            old_remaining = self._scheduled_compute_remaining.get(sid, 0.0)
            self.states[sid].Q_i = max(
                0.0, self.states[sid].Q_i - old_remaining + remaining
            )
            self._scheduled_compute_remaining[sid] = remaining

    def close(self):
        self.env.close()


__all__ = ["BasiliskBackend", "OrdiSatellite", "OrdiDynamics", "Workload"]
