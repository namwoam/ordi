"""Reproducible agreement checks for ORDI's satellite-state model.

The public SatelliteCOTS logs do not contain synchronized workload,
illumination, and communication labels over the full four-month telemetry
trace.  We therefore evaluate one 60 s transition at a time:

* temperature against the one-second Atlas inference trace, using the measured
  active fraction in each interval; and
* battery SOC against the reduced 60 s telemetry trace, under the model's idle,
  orbit-average-harvest assumption.

These are in-sample consistency checks for a measurement-calibrated model, not
an independent or open-loop flight validation.
"""

from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

from ordi.sim.cots_measurements import load_cots_measurement_profile
from ordi.sim.telemetry_replay import load_bupt1_telemetry


EPOCH_S = 60


@dataclass(frozen=True)
class StateModelFit:
    thermal_intervals: int
    thermal_mae_c: float
    thermal_rmse_c: float
    thermal_within_5c_pct: float
    battery_intervals: int
    battery_mae_pp: float
    battery_rmse_pp: float
    battery_within_1pp_pct: float
    battery_unchanged_pct: float


def _errors(observed: list[float], predicted: list[float]) -> tuple[float, float]:
    """Return MAE and RMSE for equal-length numeric sequences."""
    if len(observed) != len(predicted) or not observed:
        raise ValueError("observed and predicted must have equal, non-zero length")
    residuals = [p - y for y, p in zip(observed, predicted)]
    mae = mean(abs(e) for e in residuals)
    rmse = math.sqrt(mean(e * e for e in residuals))
    return mae, rmse


def _thermal_predictions() -> tuple[list[float], list[float]]:
    profile = load_cots_measurement_profile()
    with Path(profile.inference_log).open(newline="") as f:
        rows = list(csv.DictReader(f))

    resistance = (
        (profile.measured_max_temp_c - profile.thermal_ambient_c)
        / max(profile.compute_power_w, 1e-9)
    )
    decay = math.exp(-EPOCH_S / 300.0)
    observed: list[float] = []
    predicted: list[float] = []
    # Non-overlapping intervals avoid counting almost-identical sliding windows.
    for start in range(0, len(rows) - EPOCH_S, EPOCH_S):
        interval = rows[start : start + EPOCH_S]
        active_fraction = sum(float(r["INDEX"]) >= 0 for r in interval) / EPOCH_S
        theta_0 = float(rows[start]["TEMP"])
        theta_ss = (
            profile.thermal_ambient_c
            + resistance * profile.compute_power_w * active_fraction
        )
        predicted.append(theta_ss + (theta_0 - theta_ss) * decay)
        observed.append(float(rows[start + EPOCH_S]["TEMP"]))
    return observed, predicted


def _battery_predictions() -> tuple[list[float], list[float]]:
    profile = load_cots_measurement_profile()
    trace = load_bupt1_telemetry()
    delta_soc = (
        (profile.solar_power_w - profile.idle_power_w) * EPOCH_S
        / (profile.battery_wh * 3600.0)
    )
    observed = [100.0 * soc for soc in trace.soc[1:]]
    predicted = [
        100.0 * min(1.0, max(0.0, soc + delta_soc))
        for soc in trace.soc[:-1]
    ]
    return observed, predicted


def compute_state_model_fit() -> StateModelFit:
    thermal_obs, thermal_pred = _thermal_predictions()
    thermal_mae, thermal_rmse = _errors(thermal_obs, thermal_pred)
    thermal_within = mean(
        abs(p - y) <= 5.0 for y, p in zip(thermal_obs, thermal_pred)
    )

    battery_obs, battery_pred = _battery_predictions()
    battery_mae, battery_rmse = _errors(battery_obs, battery_pred)
    battery_within = mean(
        abs(p - y) <= 1.0 for y, p in zip(battery_obs, battery_pred)
    )
    trace = load_bupt1_telemetry()
    battery_unchanged = mean(a == b for a, b in zip(trace.soc, trace.soc[1:]))
    return StateModelFit(
        thermal_intervals=len(thermal_obs),
        thermal_mae_c=thermal_mae,
        thermal_rmse_c=thermal_rmse,
        thermal_within_5c_pct=100.0 * thermal_within,
        battery_intervals=len(battery_obs),
        battery_mae_pp=battery_mae,
        battery_rmse_pp=battery_rmse,
        battery_within_1pp_pct=100.0 * battery_within,
        battery_unchanged_pct=100.0 * battery_unchanged,
    )


def write_state_model_fit(path: Path) -> StateModelFit:
    fit = compute_state_model_fit()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(fit)))
        writer.writeheader()
        writer.writerow(asdict(fit))
    return fit


if __name__ == "__main__":
    output = Path(__file__).resolve().parents[2] / "results" / "state_model_fit.csv"
    result = write_state_model_fit(output)
    print(f"wrote {output}")
    for name, value in asdict(result).items():
        print(f"{name}: {value}")
