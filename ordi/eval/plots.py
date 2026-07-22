"""
Figure generation for all experiments.

Reads CSVs from results/ and writes PNG figures to figure/.
"""

from __future__ import annotations
import csv
import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = "results"
FIGURES_DIR = "figure"

ALG_COLORS = {
    "ORDI":                   "#e63946",
    "direct_downlink":        "#457b9d",
    "onboard_only":           "#1d3557",
    "seco_adapted":           "#e9c46a",
    "full_replication":       "#f4a261",
    "random_replication":     "#264653",
}

ALG_LABELS = {
    "ORDI":                   "ORDI",
    "direct_downlink":        "Direct Downlink",
    "onboard_only":           "Onboard-Only",
    "seco_adapted":           "SECO-Adapted",
    "full_replication":       "Full Replication",
    "random_replication":     "Random Replication",
}

E1_PLOT_METRICS = (
    ("deadline_miss_ratio", 1.0, "Deadline Miss Ratio (↓)"),
    (
        "delivery_latency_p95_s",
        60.0,
        "P95 Delivery Latency (min) (↓)",
    ),
    (
        "isl_traffic_bits_per_delivered_tile",
        1e6,
        "ISL Traffic / Delivered Tile (Mbit) (↓)",
    ),
    (
        "downlink_bits_per_delivered_tile",
        1e6,
        "Downlink / Delivered Tile (Mbit) (↓)",
    ),
    (
        "energy_j_per_delivered_tile",
        1.0,
        "Energy / Delivered Tile (J) (↓)",
    ),
    (
        "compute_load_balance",
        1.0,
        "Compute Load Balance (Jain Index) (↑)",
    ),
)


def _read_csv(exp_id: str) -> List[Dict]:
    path = os.path.join(RESULTS_DIR, f"{exp_id}.csv")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _ensure_figures():
    os.makedirs(FIGURES_DIR, exist_ok=True)


def _float(row, key):
    try:
        return float(row[key])
    except (KeyError, ValueError):
        return 0.0


def _std(row, key):
    """Across-seed std column written by aggregate_metrics (0.0 if absent)."""
    return _float(row, f"{key}_std")


def _ci95(row, key):
    """Normal-approximation 95% CI from across-run sample dispersion."""
    n = max(_float(row, "sample_count"), 1.0)
    return 1.96 * _std(row, key) / np.sqrt(n)


# ── E1: Core performance bar chart ───────────────────────────────────────────

def plot_E1():
    rows = _read_csv("E1_core")
    if not rows:
        print("No E1 data"); return

    # E1 reports only algorithm-neutral operational outcomes. Utility and the
    # composite objective are ORDI-defined preference functions.
    algs = [r["algorithm"] for r in rows]
    colors = [ALG_COLORS.get(a, "#888") for a in algs]
    labels = [ALG_LABELS.get(a, a) for a in algs]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()

    for ax, (metric, scale, title) in zip(axes, E1_PLOT_METRICS):
        vals = [_float(r, metric) / scale for r in rows]
        errs = [_std(r, metric) / scale for r in rows]
        bars = ax.bar(range(len(algs)), vals, color=colors,
                      yerr=errs, capsize=2, error_kw={"linewidth": 0.8})
        ax.set_title(title, fontsize=9)
        ax.set_xticks(range(len(algs)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        if metric in {"deadline_miss_ratio", "compute_load_balance"}:
            ax.set_ylim(0.0, 1.05)
        else:
            upper = max(
                (value + error for value, error in zip(vals, errs)),
                default=0.0,
            )
            ax.set_ylim(0.0, max(1.0, upper * 1.18))
        value_labels = [
            "0" if abs(value) < 1e-12
            else f"{value:.2f}" if value < 10.0
            else f"{value:.1f}" if value < 100.0
            else f"{value:.0f}"
            for value in vals
        ]
        ax.bar_label(bars, labels=value_labels, padding=2, fontsize=6)
        # Highlight ORDI bar
        for bar, alg in zip(bars, algs):
            if alg == "ORDI":
                bar.set_edgecolor("black")
                bar.set_linewidth(2)

    fig.suptitle(
        "E1 Core Performance: Reliability, Latency, and Resource Cost",
        fontsize=12,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    path = os.path.join(FIGURES_DIR, "E1_core.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")
    plot_E1_miss_decomposition(rows)


def plot_E1_miss_decomposition(rows=None):
    """Stack E1 miss causes so irreducible contact loss stays visible."""
    rows = _read_csv("E1_core") if rows is None else rows
    if not rows:
        print("No E1 data")
        return
    _ensure_figures()
    labels = [ALG_LABELS.get(row["algorithm"], row["algorithm"])
              for row in rows]
    causes = (
        ("contact_miss_ratio", "Contact", "#457b9d"),
        ("compute_queue_miss_ratio", "Compute queue", "#e9c46a"),
        ("policy_miss_ratio", "Policy/admission", "#2a9d8f"),
        ("hard_fault_miss_ratio", "Hard fault", "#e76f51"),
        ("source_fault_miss_ratio", "Source fault", "#9b5de5"),
    )
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bottom = np.zeros(len(rows))
    for metric, label, color in causes:
        values = np.array([_float(row, metric) for row in rows])
        ax.bar(range(len(rows)), values, bottom=bottom,
               label=label, color=color)
        bottom += values
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Deadline miss ratio")
    ax.set_title("E1 Miss-Cause Decomposition")
    ax.set_ylim(0.0, max(0.5, float(max(bottom, default=0.0)) * 1.15))
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.legend(ncol=5, fontsize=8, loc="upper center")
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "E1_miss_decomposition.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  Saved {path}")


# ── E2: Fault intensity ──────────────────────────────────────────────────────

def plot_E2():
    rows = _read_csv("E2_fault_intensity")
    if not rows:
        print("No E2 data"); return

    algs = ["ORDI", "seco_adapted", "full_replication"]
    fault_rates = sorted(set(
        float(r["algorithm"].split("fault=")[1]) for r in rows
        if "fault=" in r["algorithm"]
    ))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    for alg in algs:
        series = {metric: [] for metric in (
            "deadline_miss_ratio", "hard_fault_miss_ratio",
            "source_fault_miss_ratio",
            "isl_traffic_bits_per_delivered_tile",
            "energy_j_per_delivered_tile",
        )}
        errors = {metric: [] for metric in series}
        for rate in fault_rates:
            key = f"{alg}@fault={rate:.2f}"
            row = next((r for r in rows if r["algorithm"] == key), None)
            for metric in series:
                series[metric].append(_float(row, metric) if row else 0)
                errors[metric].append(
                    _ci95(row, metric) if row else 0
                )
        style = dict(label=ALG_LABELS.get(alg, alg),
                     color=ALG_COLORS.get(alg, "#888"), marker="o", capsize=3,
                     linewidth=2.5 if alg == "ORDI" else 1.5)
        axes[0].errorbar(
            fault_rates, series["deadline_miss_ratio"],
            yerr=errors["deadline_miss_ratio"], **style,
        )
        fault_miss = [
            hard + source for hard, source in zip(
                series["hard_fault_miss_ratio"],
                series["source_fault_miss_ratio"],
            )
        ]
        fault_err = [
            hard + source for hard, source in zip(
                errors["hard_fault_miss_ratio"],
                errors["source_fault_miss_ratio"],
            )
        ]
        axes[1].errorbar(fault_rates, fault_miss, yerr=fault_err, **style)
        axes[2].errorbar(
            fault_rates, np.array(series["isl_traffic_bits_per_delivered_tile"]) / 1e6,
            yerr=np.array(errors["isl_traffic_bits_per_delivered_tile"]) / 1e6,
            **style,
        )
        axes[3].errorbar(
            fault_rates, series["energy_j_per_delivered_tile"],
            yerr=errors["energy_j_per_delivered_tile"], **style,
        )

    titles = (
        "Operational Deadline Miss Ratio (↓)",
        "Hard + Source-Fault Miss Ratio (↓)",
        "ISL Traffic / Delivered Tile (Mbit) (↓)",
        "Energy / Delivered Tile (J) (↓)",
    )
    for ax, title in zip(axes, titles):
        ax.set_xlabel("Fault-event probability per 120 s epoch")
        ax.set_ylabel(title)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "E2_fault_intensity.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ── E3: Correlated failures ──────────────────────────────────────────────────

def plot_E3():
    rows = _read_csv("E3_correlated")
    if not rows:
        print("No E3 data"); return

    algs = ["ORDI", "full_replication", "random_replication"]
    labels = ("0plane", "1plane", "2planes")
    x = np.arange(3)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    metrics = (
        ("deadline_miss_ratio", 1.0, "Operational Deadline Miss Ratio (↓)"),
        ("hard_plus_source_fault", 1.0,
         "Hard + Source-Fault Miss Ratio (↓)"),
        ("isl_traffic_bits_per_delivered_tile", 1e6,
         "ISL Traffic / Delivered Tile (Mbit) (↓)"),
        ("energy_j_per_delivered_tile", 1.0,
         "Energy / Delivered Tile (J) (↓)"),
        )
    for alg in algs:
        matched = [
            next((row for row in rows
                  if row["algorithm"] == f"{alg}@{label}"), None)
            for label in labels
        ]
        for ax, (metric, scale, title) in zip(axes, metrics):
            if metric == "hard_plus_source_fault":
                values = [
                    (_float(row, "hard_fault_miss_ratio")
                     + _float(row, "source_fault_miss_ratio")) / scale
                    if row else 0 for row in matched
                ]
                errors = [
                    (_ci95(row, "hard_fault_miss_ratio")
                     + _ci95(row, "source_fault_miss_ratio")) / scale
                    if row else 0 for row in matched
                ]
            else:
                values = [
                    _float(row, metric) / scale if row else 0
                    for row in matched
                ]
                errors = [
                    _ci95(row, metric) / scale if row else 0
                    for row in matched
                ]
            ax.errorbar(
                x, values, yerr=errors, marker="o", capsize=3,
                color=ALG_COLORS.get(alg, "#888"),
                label=ALG_LABELS.get(alg, alg),
                linewidth=2.5 if alg == "ORDI" else 1.5,
            )
            ax.set_title(title)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(("0", "1", "2"))
        ax.set_xlabel("Failed orbital planes")
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "E3_correlated.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ── E4: Scalability ───────────────────────────────────────────────────────────

def plot_E4():
    rows = _read_csv("E4_scalability")
    if not rows:
        print("No E4 data"); return

    algs = ["ORDI", "seco_adapted"]
    request_rates = sorted(set(
        int(r["algorithm"].split("requests=")[1])
        for r in rows if "requests=" in r["algorithm"]
    ))

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()

    for alg in algs:
        matched = []
        for request_rate in request_rates:
            key = f"{alg}@requests={request_rate}"
            row = next((r for r in rows if r["algorithm"] == key), None)
            matched.append(row)
        offered = [_float(row, "offered_requests_per_orbit") for row in matched]
        style = dict(label=ALG_LABELS.get(alg, alg),
                     color=ALG_COLORS.get(alg, "#888"), linewidth=2,
                     marker="s", capsize=3)
        plot_metrics = (
            ("deadline_miss_ratio", 1.0),
            ("delivered_tiles_per_orbit", 1.0),
            ("scheduling_time_p95_s", 1.0),
            ("helper_utilization", 1.0),
            ("isl_traffic_bits_per_delivered_tile", 1e6),
            ("energy_j_per_delivered_tile", 1.0),
        )
        for ax, (metric, scale) in zip(axes, plot_metrics):
            values = [_float(row, metric) / scale for row in matched]
            errors = [
                _ci95(row, metric) / scale
                for row in matched
            ]
            ax.errorbar(offered, values, yerr=errors, **style)

    reference_rows = [
        next(row for row in rows
             if row["algorithm"] == f"ORDI@requests={request_rate}")
        for request_rate in request_rates
    ]
    axes[1].plot(
        [_float(row, "offered_requests_per_orbit") for row in reference_rows],
        [_float(row, "offered_tiles_per_orbit") for row in reference_rows],
        color="#555", linestyle="--", linewidth=1.2,
        label="All offered tiles delivered",
    )

    titles = (
        "Realized Deadline Miss Ratio (↓)",
        "Delivered Tiles / Orbit (↑)",
        "P95 Scheduling Time / Epoch (s) (↓)",
        "Helper Utilization (↑)",
        "ISL Traffic / Delivered Tile (Mbit) (↓)",
        "Energy / Delivered Tile (J) (↓)",
    )
    for ax, title in zip(axes, titles):
        ax.set_xlabel("Actual Offered Requests / Orbit")
        ax.set_ylabel(title)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "E4_scalability.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def plot_all():
    _ensure_figures()
    for fn in [plot_E1, plot_E2, plot_E3, plot_E4]:
        fn()
