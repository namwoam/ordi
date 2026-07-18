"""
Per-satellite state vector σ_i(t) = (C_i, B_i, Θ_i, Q_i, A_i).

This module contains scheduler-facing state and hardware parameters. It does
not evolve physical state; :mod:`ordi.sim.basilisk_backend` projects Basilisk
and BSK-RL outputs into these objects after every environment step.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional


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
DEFAULT_DOWNLINK_RATE_BPS   = 100e6      # 100 Mbps RF ground downlink
DEFAULT_THERMAL_AMBIENT_C   = 20.0       # effective ambient (radiative sink)
DEFAULT_THERMAL_AREA_M2     = 0.01       # exposed payload radiator area
DEFAULT_THERMAL_MASS_KG     = 0.5        # thermally active payload mass
DEFAULT_SPECIFIC_HEAT       = 900.0      # aluminium-like heat capacity (J/kg/K)


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
    thermal_ambient_c: float = DEFAULT_THERMAL_AMBIENT_C
    thermal_area_m2: float = DEFAULT_THERMAL_AREA_M2
    thermal_absorptivity: float = 0.2
    thermal_emissivity: float = 0.8
    thermal_mass_kg: float = DEFAULT_THERMAL_MASS_KG
    thermal_specific_heat_j_kg_k: float = DEFAULT_SPECIFIC_HEAT

    @property
    def battery_j(self) -> float:
        return self.battery_wh * 3600.0

    @property
    def battery_min_j(self) -> float:
        return self.battery_j * self.battery_min_frac

    @property
    def compute_rate_flops_per_s(self) -> float:
        """Convert the configured GFLOP/s rate to FLOP/s."""
        return self.compute_rate_gflops * 1e9


@dataclass
class SatelliteState:
    """
    Mutable per-satellite state for one scheduling epoch.

    C_i : available compute rate (FLOP/s) — may be throttled
    B_i : current battery energy (Joules)
    Θ_i : current chip temperature (°C)
    Q_i : queued compute load (FLOPs) already committed
    A_i : availability flag (0/1)
    """
    params: SatelliteParams
    C_i: float = field(init=False)   # computed from params + throttle
    B_i: float = field(init=False)
    Theta_i: float = field(init=False)
    Q_i: float = 0.0
    D_i: float = 0.0
    A_i: int = 1
    _injected_failure: bool = False
    _compute_rate_multiplier: float = 1.0

    def __post_init__(self):
        self.C_i = self.params.compute_rate_flops_per_s
        self.B_i = self.params.battery_j * 0.85   # start at 85% charge
        self.Theta_i = DEFAULT_THERMAL_INIT_C

    # ── availability ─────────────────────────────────────────────────────────

    def _update_availability(self):
        if self._injected_failure:
            self.A_i = 0
            return
        if self.B_i < self.params.battery_min_j:
            self.A_i = 0
        else:
            self.A_i = 1

    def _effective_compute_rate(self) -> float:
        """Scheduler capacity after injected software straggler faults.

        Thermal throttling is intentionally absent here; Basilisk owns that
        physical state.  This multiplier represents only an ORDI fault event.
        """
        return self.params.compute_rate_flops_per_s * self._compute_rate_multiplier

    # ── energy cost estimation ────────────────────────────────────────────────

    def energy_for_compute(self, flops: float) -> float:
        """Joules to execute ``flops`` on this satellite."""
        t = flops / max(self.C_i, 1.0)
        return self.params.compute_power_w * t

    def energy_for_rx(self, bits: float) -> float:
        """Joules to receive `bits` (antenna + baseband)."""
        rx_power_w = self.params.comms_power_w * 0.4
        return rx_power_w * (bits / max(200e6, 1.0))

    def energy_for_tx(self, bits: float, rate_bps: float = 200e6) -> float:
        """Joules to transmit ``bits`` at ``rate_bps``.

        ISLs use the historical 200 Mbps default.  Ground downlinks pass the
        100 Mbps contact rate explicitly; keeping the rate in this calculation
        avoids making a raw-image downlink look energetically free.
        """
        return self.params.comms_power_w * (bits / max(rate_bps, 1.0))

    def energy_for_downlink(self, bits: float) -> float:
        """Joules to transmit ``bits`` over the modeled ground downlink."""
        return self.energy_for_tx(bits, DEFAULT_DOWNLINK_RATE_BPS)

    def inject_failure(self):
        self._injected_failure = True
        self.A_i = 0

    def recover(self):
        self._injected_failure = False
        self._update_availability()

    def __repr__(self):
        return (f"SatState({self.params.sat_id}: "
                f"C={self.C_i/1e9:.1f} GFLOP/s, "
                f"B={self.B_i/3600:.1f}Wh, "
                f"Θ={self.Theta_i:.1f}°C, "
                f"Q={self.Q_i/1e9:.1f} GFLOP, A={self.A_i})")


def make_constellation_states(
    sat_ids: list,
    seed: int = 42,
    params_factory: Optional[Callable[[str], SatelliteParams]] = None,
) -> Dict[str, SatelliteState]:
    """
    Create a dict of SatelliteState for each satellite ID with mild random variation
    in initial battery and thermal state to break symmetry.
    """
    import random
    rng = random.Random(seed)
    states = {}
    for sid in sat_ids:
        p = params_factory(sid) if params_factory else SatelliteParams(sat_id=sid)
        s = SatelliteState(params=p)
        # slight variation
        s.B_i = p.battery_j * rng.uniform(0.70, 0.95)
        s.Theta_i = rng.uniform(18.0, 35.0)
        s.C_i = p.compute_rate_flops_per_s
        s._update_availability()
        states[sid] = s
    return states
