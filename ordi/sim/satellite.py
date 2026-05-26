"""
Per-satellite state vector σ_i(t) = (C_i, B_i, Θ_i, Q_i, A_i).

Models compute throttling, battery energy balance, and thermal dynamics
consistent with a COTS-equipped LEO nanosatellite (Jetson-class payload).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional
import math


# ── default hardware parameters (Jetson Orin NX class) ───────────────────────
DEFAULT_COMPUTE_RATE_GFLOPS = 20.0       # nominal TOPS (FP16)
DEFAULT_BATTERY_WH          = 25.0       # 25 Wh typical 3U cubesat
DEFAULT_BATTERY_MIN_FRAC    = 0.15       # keep 15% reserve
DEFAULT_THERMAL_INIT_C      = 25.0       # initial chip temperature
DEFAULT_THERMAL_MAX_C       = 80.0       # throttle threshold
DEFAULT_SOLAR_POWER_W       = 8.0        # average harvested power (sunlit fraction ~55%)
DEFAULT_IDLE_POWER_W        = 3.0        # platform power when not computing
DEFAULT_COMPUTE_POWER_W     = 15.0       # incremental power during full compute
DEFAULT_COMMS_POWER_W       = 5.0        # incremental ISL/downlink Tx power
DEFAULT_THERMAL_RC          = 300.0      # thermal time constant (seconds)
DEFAULT_THERMAL_AMBIENT_C   = 20.0       # effective ambient (radiative sink)


@dataclass
class SatelliteParams:
    """Fixed hardware parameters for a satellite."""
    sat_id: str
    compute_rate_gflops: float = DEFAULT_COMPUTE_RATE_GFLOPS
    battery_wh: float = DEFAULT_BATTERY_WH
    battery_min_frac: float = DEFAULT_BATTERY_MIN_FRAC
    thermal_max_c: float = DEFAULT_THERMAL_MAX_C
    solar_power_w: float = DEFAULT_SOLAR_POWER_W
    idle_power_w: float = DEFAULT_IDLE_POWER_W
    compute_power_w: float = DEFAULT_COMPUTE_POWER_W
    comms_power_w: float = DEFAULT_COMMS_POWER_W
    thermal_rc: float = DEFAULT_THERMAL_RC
    thermal_ambient_c: float = DEFAULT_THERMAL_AMBIENT_C

    @property
    def battery_j(self) -> float:
        return self.battery_wh * 3600.0

    @property
    def battery_min_j(self) -> float:
        return self.battery_j * self.battery_min_frac

    @property
    def compute_rate_cycles_per_s(self) -> float:
        # 1 GFLOP ≈ 2e9 FP operations → treat as CPU cycles for scheduling
        return self.compute_rate_gflops * 1e9


@dataclass
class SatelliteState:
    """
    Mutable per-satellite state for one scheduling epoch.

    C_i : available compute rate (cycles/s) — may be throttled
    B_i : current battery energy (Joules)
    Θ_i : current chip temperature (°C)
    Q_i : queued compute load (cycles) already committed
    A_i : availability flag (0/1)
    """
    params: SatelliteParams
    C_i: float = field(init=False)   # computed from params + throttle
    B_i: float = field(init=False)
    Theta_i: float = field(init=False)
    Q_i: float = 0.0
    A_i: int = 1
    _injected_failure: bool = False

    def __post_init__(self):
        self.C_i = self.params.compute_rate_cycles_per_s
        self.B_i = self.params.battery_j * 0.85   # start at 85% charge
        self.Theta_i = DEFAULT_THERMAL_INIT_C

    # ── availability ─────────────────────────────────────────────────────────

    def _update_availability(self):
        if self._injected_failure:
            self.A_i = 0
            return
        if self.B_i < self.params.battery_min_j:
            self.A_i = 0
        elif self.Theta_i >= self.params.thermal_max_c:
            self.A_i = 0
        else:
            self.A_i = 1

    # ── thermal throttling ───────────────────────────────────────────────────

    def _throttled_compute_rate(self) -> float:
        if self.Theta_i >= self.params.thermal_max_c:
            return self.params.compute_rate_cycles_per_s * 0.25
        if self.Theta_i >= self.params.thermal_max_c * 0.90:
            return self.params.compute_rate_cycles_per_s * 0.60
        return self.params.compute_rate_cycles_per_s

    # ── epoch advance ─────────────────────────────────────────────────────────

    def advance_epoch(
        self,
        epoch_length_s: float,
        compute_cycles: float = 0.0,
        tx_bits: float = 0.0,
        rx_bits: float = 0.0,
        in_sunlight: bool = True,
    ):
        """
        Advance state by one epoch given the committed workload.

        compute_cycles : total compute cycles assigned this epoch
        tx_bits        : bits transmitted (ISL + downlink)
        rx_bits        : bits received (ISL uplink)
        in_sunlight    : whether satellite is in sunlight (for solar harvest)
        """
        dt = epoch_length_s

        # ── energy balance ───────────────────────────────────────────────────
        solar = self.params.solar_power_w * dt if in_sunlight else 0.0
        compute_time = min(compute_cycles / max(self.C_i, 1.0), dt)
        e_compute = self.params.compute_power_w * compute_time
        tx_time = tx_bits / max(200e6, 1.0)  # assume 200 Mbps ISL
        e_tx = self.params.comms_power_w * min(tx_time, dt)
        e_idle = self.params.idle_power_w * dt
        self.B_i = min(
            self.params.battery_j,
            self.B_i + solar - e_idle - e_compute - e_tx
        )

        # ── thermal: first-order RC ──────────────────────────────────────────
        P_dissipated = e_compute / dt if dt > 0 else 0.0
        Theta_ss = self.params.thermal_ambient_c + P_dissipated * self.params.thermal_rc
        tau = self.params.thermal_rc
        self.Theta_i = Theta_ss + (self.Theta_i - Theta_ss) * math.exp(-dt / tau)

        # ── update derived fields ────────────────────────────────────────────
        self.C_i = self._throttled_compute_rate()
        self._update_availability()

        # Drain committed queue
        self.Q_i = max(0.0, self.Q_i - compute_cycles)

    # ── energy cost estimation ────────────────────────────────────────────────

    def energy_for_compute(self, cycles: float) -> float:
        """Joules to compute `cycles` on this satellite."""
        t = cycles / max(self.C_i, 1.0)
        return self.params.compute_power_w * t

    def energy_for_rx(self, bits: float) -> float:
        """Joules to receive `bits` (antenna + baseband)."""
        rx_power_w = self.params.comms_power_w * 0.4
        return rx_power_w * (bits / max(200e6, 1.0))

    def energy_for_tx(self, bits: float) -> float:
        """Joules to transmit `bits`."""
        return self.params.comms_power_w * (bits / max(200e6, 1.0))

    def thermal_increase(self, cycles: float, epoch_length_s: float) -> float:
        """Approximate chip temperature increase (°C) from compute load."""
        P = self.params.compute_power_w * (cycles / max(self.C_i, 1.0)) / max(epoch_length_s, 1.0)
        delta_ss = P * self.params.thermal_rc / (self.params.thermal_rc + 1)
        return max(0.0, delta_ss)

    def inject_failure(self):
        self._injected_failure = True
        self.A_i = 0

    def recover(self):
        self._injected_failure = False
        self._update_availability()

    def __repr__(self):
        return (f"SatState({self.params.sat_id}: "
                f"C={self.C_i/1e9:.1f}G cycles/s, "
                f"B={self.B_i/3600:.1f}Wh, "
                f"Θ={self.Theta_i:.1f}°C, "
                f"Q={self.Q_i/1e9:.1f}G, A={self.A_i})")


def make_constellation_states(sat_ids: list, seed: int = 42) -> Dict[str, SatelliteState]:
    """
    Create a dict of SatelliteState for each satellite ID with mild random variation
    in initial battery and thermal state to break symmetry.
    """
    import random
    rng = random.Random(seed)
    states = {}
    for sid in sat_ids:
        p = SatelliteParams(sat_id=sid)
        s = SatelliteState(params=p)
        # slight variation
        s.B_i = p.battery_j * rng.uniform(0.70, 0.95)
        s.Theta_i = rng.uniform(18.0, 35.0)
        s.C_i = s._throttled_compute_rate()
        s._update_availability()
        states[sid] = s
    return states
