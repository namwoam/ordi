"""
Measurement-backed COTS payload profiles loaded from SatelliteCOTS logs.

The parser reads the public MobiCom '24 SatelliteCOTS artifact at runtime:
https://github.com/TiansuanConstellation/MobiCom24-SatelliteCOTS

Set MOBICOM24_COTS_ROOT to the cloned artifact root.  If unset, the local
/tmp/MobiCom24-SatelliteCOTS clone is used.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from statistics import mean

from ordi.sim.satellite import SatelliteParams

DEFAULT_COTS_ROOT = Path("/tmp/MobiCom24-SatelliteCOTS")
ATLAS_FULL_4T_LOG = Path("Energy-Efficiency/Data/sat_atlas_infer/200B_Sun_FULL_4T_180.csv")
AVAILABLE_ENERGY_LOG = Path("Energy-Available/Data/longest_rounds_energy_df.csv")
BATTERY_CURVE_DIR = Path("Energy-Available/battery_curve")
SHIP_TILE_GFLOPS = 0.9
IMAGES_PER_4T_INDEX = 13 * 100
ATLAS_THERMAL_MAX_C = 80.0


@dataclass(frozen=True)
class COTSMeasurementProfile:
    compute_rate_gflops: float
    idle_power_w: float
    compute_power_w: float
    battery_wh: float
    solar_power_w: float
    comms_power_w: float
    thermal_ambient_c: float
    active_power_w: float
    measured_max_temp_c: float
    source_root: str
    inference_log: str


def _dataset_root() -> Path:
    root = Path(os.environ.get("MOBICOM24_COTS_ROOT", DEFAULT_COTS_ROOT)).expanduser()
    if not root.exists():
        raise FileNotFoundError(
            "SatelliteCOTS dataset not found. Clone "
            "https://github.com/TiansuanConstellation/MobiCom24-SatelliteCOTS "
            "and set MOBICOM24_COTS_ROOT to its root."
        )
    return root


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in {"", None} else 0.0


def _parse_time(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")


def _atlas_inference_stats(root: Path) -> dict[str, float]:
    path = root / ATLAS_FULL_4T_LOG
    rows = _read_rows(path)
    active = [r for r in rows if _float(r, "INDEX") >= 0]
    idle = [r for r in rows if _float(r, "INDEX") < 0]
    if not active or not idle:
        raise ValueError(f"Atlas inference log has no active/idle samples: {path}")

    active_power = mean(_float(r, "I_Atlas200DK-B") * 12.0 / 1000.0 for r in active)
    idle_power = mean(_float(r, "I_Atlas200DK-B") * 12.0 / 1000.0 for r in idle)
    duration_s = max(_float(r, "TIME") for r in active) - min(_float(r, "TIME") for r in active)
    n_index = max(_float(r, "INDEX") for r in active)
    n_images = n_index * IMAGES_PER_4T_INDEX
    compute_rate_gflops = n_images * SHIP_TILE_GFLOPS / max(duration_s, 1.0)

    return {
        "active_power_w": active_power,
        "idle_power_w": idle_power,
        "compute_power_w": max(active_power - idle_power, 0.0),
        "compute_rate_gflops": compute_rate_gflops,
        "thermal_ambient_c": min(_float(r, "TEMP") for r in active),
        "measured_max_temp_c": max(_float(r, "TEMP") for r in active),
        "inference_log": str(path),
    }


def _battery_wh(root: Path) -> float:
    values = []
    for path in (root / BATTERY_CURVE_DIR).glob("discharging_df_*.csv"):
        rows = _read_rows(path)
        if rows:
            values.append(max(_float(r, "Energy(Wh)") for r in rows))
    if not values:
        raise ValueError(f"No battery discharge logs found under {root / BATTERY_CURVE_DIR}")
    return mean(values)


def _orbit_energy_stats(root: Path) -> dict[str, float]:
    rows = _read_rows(root / AVAILABLE_ENERGY_LOG)
    if not rows:
        raise ValueError(f"No available-energy rows found in {root / AVAILABLE_ENERGY_LOG}")

    durations_h = [
        (_parse_time(r["round_end"]) - _parse_time(r["round_start"])).total_seconds() / 3600.0
        for r in rows
    ]
    avg_duration_h = mean(durations_h)
    return {
        "solar_power_w": mean(_float(r, "solar_harvested_energy") for r in rows) / avg_duration_h,
        "comms_power_w": mean(_float(r, "comm_energy") for r in rows) / avg_duration_h,
    }


@lru_cache(maxsize=1)
def load_cots_measurement_profile() -> COTSMeasurementProfile:
    """Load BUPT-1 Atlas 200DK profile directly from SatelliteCOTS logs."""
    root = _dataset_root()
    atlas = _atlas_inference_stats(root)
    energy = _orbit_energy_stats(root)
    return COTSMeasurementProfile(
        compute_rate_gflops=atlas["compute_rate_gflops"],
        idle_power_w=atlas["idle_power_w"],
        compute_power_w=atlas["compute_power_w"],
        battery_wh=_battery_wh(root),
        solar_power_w=energy["solar_power_w"],
        comms_power_w=energy["comms_power_w"],
        thermal_ambient_c=atlas["thermal_ambient_c"],
        active_power_w=atlas["active_power_w"],
        measured_max_temp_c=atlas["measured_max_temp_c"],
        source_root=str(root),
        inference_log=atlas["inference_log"],
    )


def atlas_200dk_bupt1_params(sat_id: str) -> SatelliteParams:
    """Return BUPT-1 Atlas 200DK-B parameters from actual SatelliteCOTS logs."""
    p = load_cots_measurement_profile()
    return SatelliteParams(
        sat_id=sat_id,
        compute_rate_gflops=p.compute_rate_gflops,
        battery_wh=p.battery_wh,
        battery_min_frac=0.15,
        thermal_max_c=ATLAS_THERMAL_MAX_C,
        solar_power_w=p.solar_power_w,
        idle_power_w=p.idle_power_w,
        compute_power_w=p.compute_power_w,
        comms_power_w=p.comms_power_w,
        thermal_rc=300.0,
        thermal_ambient_c=p.thermal_ambient_c,
    )
