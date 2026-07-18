"""Real hardware-state replay from BUPT-1 in-orbit telemetry.

Instead of the synthetic RC-thermal + energy-balance model in
the Basilisk/BSK-RL state projection, this
drives battery energy (B_i), chip temperature (Theta_i), and the compute-throttle
(C_i) from the real measured telemetry stream of the BUPT-1 Atlas 200DK payload
(SatelliteCOTS ``CommonData-Telemetries/telemetry_all.csv``, ~10M rows spanning
2023-03-22 → 2023-07-24).

Pipeline:
  1. Stream-parse the 1.5 GB telemetry CSV once, downsample to a 60 s reduced
     trace, and cache it (``data/bupt1_telemetry_60s.csv``) so later runs are fast.
  2. Derive per-epoch (t_rel, soc, temp_c): SOC from measured battery voltage,
     temperature directly from the measured Atlas-B chip sensor.
  3. Build an epoch state-driver that, each epoch, indexes the trace at
     ``epoch_start + per_sat_offset`` (nearest sample, wrapping the 4-month trace)
     and overwrites each satellite's B_i / Theta_i / C_i, then recomputes A_i.

Per-satellite time offsets phase-shift the single real satellite's trace across
the whole constellation so states are decorrelated (an explicit approximation:
one real satellite standing in for many).
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

from ordi.sim.satellite import SatelliteState

# Telemetry column names (SatelliteCOTS schema).
_COL_TIME = "Time"
_COL_BATT_U = ("BATTERY1_U", "BATTERY2_U")   # mV
_COL_ATLAS_TEMP = "ATLAS_B_TEMP"             # deg C (0 when payload B powered off)

# Battery voltage → state-of-charge mapping (mV).  The BUPT-1 pack reads
# ~14.6–16.2 V operationally; use a slightly wider empty/full envelope so SOC
# stays in a sane interior range rather than saturating at the observed extremes.
_V_EMPTY_MV = 14400.0
_V_FULL_MV = 16800.0

_TELEMETRY_REL = Path("CommonData-Telemetries/telemetry_all.csv")
_REDUCED_NAME = "bupt1_telemetry_60s.csv"
_EPOCH_S = 60.0
_TEMP_IDLE_FALLBACK_C = 20.0   # payload-off samples report 0 °C; use ambient


@dataclass
class TelemetryTrace:
    """Downsampled BUPT-1 telemetry: parallel arrays at 60 s spacing."""
    t_rel: List[float]     # seconds from trace start
    soc: List[float]       # battery state-of-charge in [0, 1]
    temp_c: List[float]    # measured chip temperature (deg C)
    source_file: str
    span_start: str
    span_end: str

    def __len__(self) -> int:
        return len(self.t_rel)

    @property
    def duration_s(self) -> float:
        return self.t_rel[-1] - self.t_rel[0] if self.t_rel else 0.0


def _data_dir() -> Path:
    root = Path(os.environ.get("ORDI_DATA_DIR", Path(__file__).resolve().parents[2] / "data"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def _cots_root() -> Path:
    from ordi.sim.cots_measurements import _dataset_root
    return _dataset_root()


def _voltage_to_soc(v_mv: float) -> float:
    soc = (v_mv - _V_EMPTY_MV) / (_V_FULL_MV - _V_EMPTY_MV)
    return max(0.0, min(1.0, soc))


def _mean_float(row: Dict[str, str], keys) -> Optional[float]:
    vals = []
    for k in keys:
        v = row.get(k, "")
        if v not in ("", None):
            try:
                vals.append(float(v))
            except ValueError:
                pass
    return sum(vals) / len(vals) if vals else None


def _build_reduced_trace(src: Path, out: Path) -> None:
    """Stream the full telemetry CSV, emit one row per 60 s wall-clock bucket."""
    with src.open(newline="") as f_in, out.open("w", newline="") as f_out:
        reader = csv.DictReader(f_in)
        writer = csv.writer(f_out)
        writer.writerow(["t_rel", "soc", "temp_c", "abs_time"])
        t0: Optional[datetime] = None
        next_emit = 0.0
        for row in reader:
            ts = row.get(_COL_TIME, "")
            if not ts:
                continue
            try:
                dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
            if t0 is None:
                t0 = dt
            t_rel = (dt - t0).total_seconds()
            if t_rel < next_emit:
                continue
            next_emit = t_rel + _EPOCH_S
            v = _mean_float(row, _COL_BATT_U)
            if v is None:
                continue
            temp = row.get(_COL_ATLAS_TEMP, "")
            try:
                temp_c = float(temp)
            except (ValueError, TypeError):
                temp_c = 0.0
            if temp_c <= 0.0:
                temp_c = _TEMP_IDLE_FALLBACK_C
            writer.writerow([f"{t_rel:.0f}", f"{_voltage_to_soc(v):.4f}", f"{temp_c:.1f}", ts])


def load_bupt1_telemetry(refresh: bool = False) -> TelemetryTrace:
    """Load (building+caching if needed) the reduced 60 s BUPT-1 telemetry trace."""
    reduced = _data_dir() / _REDUCED_NAME
    if refresh or not reduced.exists():
        src = _cots_root() / _TELEMETRY_REL
        if not src.exists():
            raise FileNotFoundError(
                f"BUPT-1 telemetry not found at {src}. Unzip "
                "CommonData-Telemetries/telemetry_all.csv.zip in the SatelliteCOTS dataset."
            )
        _build_reduced_trace(src, reduced)

    t_rel: List[float] = []
    soc: List[float] = []
    temp_c: List[float] = []
    abs_first = abs_last = ""
    with reduced.open(newline="") as f:
        for r in csv.DictReader(f):
            t_rel.append(float(r["t_rel"]))
            soc.append(float(r["soc"]))
            temp_c.append(float(r["temp_c"]))
            abs_last = r.get("abs_time", abs_last)
            if not abs_first:
                abs_first = abs_last
    if not t_rel:
        raise ValueError(f"Reduced telemetry trace is empty: {reduced}")
    return TelemetryTrace(t_rel, soc, temp_c, str(reduced), abs_first, abs_last)


def make_telemetry_state_driver(
    trace: TelemetryTrace,
    sat_ids: List[str],
    seed: int = 0,
) -> Callable[[int, float, Dict[str, SatelliteState]], None]:
    """Return ``driver(epoch, epoch_start_s, states)`` that overwrites each
    satellite's B_i / Theta_i / C_i from the measured trace and recomputes A_i.

    Each satellite gets a fixed random offset into the 4-month trace so the one
    real satellite's telemetry is phase-shifted across the constellation.
    Indexing wraps modulo the trace length (nearest 60 s sample).
    """
    import random
    rng = random.Random(seed)
    n = len(trace)
    dur = max(trace.duration_s, _EPOCH_S)
    offsets = {sid: rng.uniform(0.0, dur) for sid in sat_ids}

    def _index_at(t_rel: float) -> int:
        # trace is ~uniform 60 s spacing; nearest index by division, wrapped.
        return int(round((t_rel % dur) / _EPOCH_S)) % n

    def driver(epoch: int, epoch_start_s: float, states: Dict[str, SatelliteState]) -> None:
        for sid, st in states.items():
            idx = _index_at(epoch_start_s + offsets.get(sid, 0.0))
            st.B_i = trace.soc[idx] * st.params.battery_j
            st.Theta_i = trace.temp_c[idx]
            st.C_i = st._effective_compute_rate()
            st._update_availability()

    return driver
