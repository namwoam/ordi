"""
Per-satellite state vector σ_i(t) = (C_i, B_i, Θ_i, Q_i, A_i).

This module contains scheduler-facing state and hardware parameters. It does
not evolve physical state; :mod:`ordi.sim.basilisk_backend` projects Basilisk
and BSK-RL outputs into these objects after every environment step.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional


# ── default hardware parameters ──────────────────────────────────────────────
# Workload-level sustained throughput assumption, not device nameplate TOPS.
# Individual experiments should override it with workload measurements.
DEFAULT_COMPUTE_RATE_GFLOPS = 20.0
# Representative small-CubeSat capacity between EnduroSat's 20.8 Wh flight
# pack and AAC Clyde Space's 30 Wh smallest OPTIMUS configuration.
# https://one.endurosat.com/
# https://www.aac-clyde.space/what-we-do/space-products-components/cubesat-batteries
DEFAULT_BATTERY_WH          = 25.0
# Explicit ORDI operational-reserve policy; not a battery hardware limit.
DEFAULT_BATTERY_MIN_FRAC    = 0.15
# Controlled initial condition used before Basilisk evolves thermal state.
DEFAULT_THERMAL_INIT_C      = 25.0
# Conservative design limit below NVIDIA's 99 C software throttle point.
# https://docs.nvidia.com/jetson/archives/r36.2/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html
DEFAULT_THERMAL_MAX_C       = 80.0
# EnduroSat 3U body panel specifies up to 8.4 W in LEO.
# https://www.endurosat.com/products/3u-solar-panel/
DEFAULT_SOLAR_POWER_W       = 8.0
# Representative 3U platform load; Alen Space specifies 0.5--5.9 W average.
# https://www.cubesatshop.com/wp-content/uploads/2023/05/Alen-Space_Platforms-1U-6U.pdf
DEFAULT_IDLE_POWER_W        = 3.0
# NVIDIA Orin NX supports a documented 15 W reference power mode.
# https://developer.nvidia.com/blog/nvidia-jetpack-6-2-brings-super-mode-to-nvidia-jetson-orin-nano-and-jetson-orin-nx-modules/
DEFAULT_COMPUTE_POWER_W     = 15.0
# DLR OSIRIS4CubeSat: 8.5 W operating power at 100 Mbit/s.
# https://elib.dlr.de/187010/1/2022_ICSOS_CubeSat.pdf
DEFAULT_COMMS_POWER_W       = 8.5        # active optical terminal / Tx power
# O4C does not publish a mode-separated Rx value; use the ESA COPINS S-band
# ISL reference (1 W Rx versus 3 W Tx) as an explicit cross-technology proxy.
# https://www.esa.int/Enabling_Support/Preparing_for_the_Future/Discovery_and_Preparation/Announcement_of_opportunity_3_AIM_CubeSat_opportunity_payloads_COPINS
DEFAULT_RX_POWER_FRACTION   = 1.0 / 3.0  # receive power relative to Tx power
# DLR OSIRIS4CubeSat specifies 100 Mbit/s optical downlink operation.
# https://elib.dlr.de/187010/1/2022_ICSOS_CubeSat.pdf
DEFAULT_DOWNLINK_RATE_BPS   = 100e6
# The following three are explicit lumped-thermal-model assumptions.
DEFAULT_THERMAL_AMBIENT_C   = 20.0       # effective radiative sink
DEFAULT_THERMAL_AREA_M2     = 0.01       # 10 cm x 10 cm radiator
DEFAULT_THERMAL_MASS_KG     = 0.5        # thermally active payload mass
# NIST room-temperature aluminum specific heat is about 900 J/(kg K).
# https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=101567
DEFAULT_SPECIFIC_HEAT       = 900.0


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
    rx_power_fraction: float = DEFAULT_RX_POWER_FRACTION
    thermal_ambient_c: float = DEFAULT_THERMAL_AMBIENT_C
    thermal_area_m2: float = DEFAULT_THERMAL_AREA_M2
    # Assumed surface properties; these must be replaced for a known coating.
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
    A_i: int = 1
    _injected_failure: bool = False
    _compute_rate_multiplier: float = 1.0
    _thermal_rate_multiplier: float = 1.0

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
        """Scheduler capacity after software and thermal rate multipliers."""
        return (
            self.params.compute_rate_flops_per_s
            * self._compute_rate_multiplier
            * self._thermal_rate_multiplier
        )

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
