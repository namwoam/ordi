"""
Measurement-backed COTS payload profiles.

The constants in this module are derived from the public MobiCom '24
SatelliteCOTS artifact:
https://github.com/TiansuanConstellation/MobiCom24-SatelliteCOTS

Dataset folders used:
  - Energy-Efficiency/Data/sat_atlas_infer
  - Temperature-HeatingRate/Data/sat_atlas_infer
  - Energy-Available/Data and Energy-Available/battery_curve
"""

from __future__ import annotations

from ordi.sim.satellite import SatelliteParams


# Atlas 200DK-B on BUPT-1, FULL 4-thread satellite inference run:
# active power 10.19 W, idle power 7.78 W.  The ORDI model stores compute
# power as the incremental compute load because idle power is charged per epoch.
ATLAS_200DK_IDLE_POWER_W = 7.78
ATLAS_200DK_COMPUTE_POWER_W = 10.19 - ATLAS_200DK_IDLE_POWER_W

# FULL 4-thread run: 89 index rounds over 10,770 s.  The artifact's
# Energy-Efficiency/sat_atlas.py treats each 4T index as 13 groups of 100
# pictures, i.e. 1,300 inference images per index.  Mapping one EO tile to one
# inference image and using the ship-detection profile's 0.9 GFLOP/tile gives
# an effective scheduling rate of roughly 9.7 GFLOP/s.
ATLAS_200DK_EFFECTIVE_GFLOPS = 9.7

# Battery discharge curves show about 119 Wh usable energy.  The available
# energy table reports 22.62 Wh solar harvest per 1.568 h orbit on average.
BUPT1_BATTERY_WH = 118.9
BUPT1_SOLAR_POWER_W = 14.43

# Energy-Available reports 3.91 Wh communication energy per 1.568 h orbit.
BUPT1_COMMS_POWER_W = 2.49

# The measured full satellite run reached 57 C after sustained inference; keep
# an 80 C cutoff to match the artifact's overheating plots and Atlas envelope.
ATLAS_200DK_THERMAL_MAX_C = 80.0


def atlas_200dk_bupt1_params(sat_id: str) -> SatelliteParams:
    """Return BUPT-1 Atlas 200DK-B parameters for one simulated satellite."""
    return SatelliteParams(
        sat_id=sat_id,
        compute_rate_gflops=ATLAS_200DK_EFFECTIVE_GFLOPS,
        battery_wh=BUPT1_BATTERY_WH,
        battery_min_frac=0.15,
        thermal_max_c=ATLAS_200DK_THERMAL_MAX_C,
        solar_power_w=BUPT1_SOLAR_POWER_W,
        idle_power_w=ATLAS_200DK_IDLE_POWER_W,
        compute_power_w=ATLAS_200DK_COMPUTE_POWER_W,
        comms_power_w=BUPT1_COMMS_POWER_W,
        thermal_rc=300.0,
        thermal_ambient_c=17.0,
    )
