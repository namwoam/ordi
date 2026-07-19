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
    ("realized_miss_ratio", 1.0, "Deadline Miss Ratio (↓)"),
    (
        "isl_traffic_bits_per_delivered_tile",
        1e6,
        "ISL Traffic / Delivered Tile (Mbit) (↓)",
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

    fig, axes = plt.subplots(1, len(E1_PLOT_METRICS), figsize=(10, 4))

    for ax, (metric, scale, title) in zip(axes, E1_PLOT_METRICS):
        vals = [_float(r, metric) / scale for r in rows]
        errs = [_std(r, metric) / scale for r in rows]
        bars = ax.bar(range(len(algs)), vals, color=colors,
                      yerr=errs, capsize=2, error_kw={"linewidth": 0.8})
        ax.set_title(title, fontsize=9)
        ax.set_xticks(range(len(algs)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        # Highlight ORDI bar
        for bar, alg in zip(bars, algs):
            if alg == "ORDI":
                bar.set_edgecolor("black")
                bar.set_linewidth(2)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "E1_core.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    for alg in algs:
        miss, miss_err, traffic, traffic_err = [], [], [], []
        for rate in fault_rates:
            key = f"{alg}@fault={rate:.2f}"
            row = next((r for r in rows if r["algorithm"] == key), None)
            miss.append(_float(row, "realized_miss_ratio") if row else 0)
            miss_err.append(_std(row, "realized_miss_ratio") if row else 0)
            traffic.append(_float(row, "isl_traffic_bits") if row else 0)
            traffic_err.append(_std(row, "isl_traffic_bits") if row else 0)
        style = dict(label=ALG_LABELS.get(alg, alg),
                     color=ALG_COLORS.get(alg, "#888"), marker="o", capsize=3,
                     linewidth=2.5 if alg == "ORDI" else 1.5)
        ax1.errorbar(fault_rates, miss, yerr=miss_err, **style)
        ax2.errorbar(fault_rates, traffic, yerr=traffic_err, **style)

    ax1.set_xlabel("Fault Rate"); ax1.set_ylabel("Deadline Miss Ratio (↓)")
    ax2.set_xlabel("Fault Rate"); ax2.set_ylabel("ISL Traffic (bits) (↓)")
    ax1.legend(fontsize=8); ax2.legend(fontsize=8)
    ax1.set_ylim(bottom=0); ax2.set_ylim(bottom=0)

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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    labels = [r["algorithm"] for r in rows]
    x = np.arange(len(labels))
    colors = [ALG_COLORS.get(label.split("@")[0], "#888") for label in labels]
    for ax, metric, title in (
        (ax1, "realized_miss_ratio", "Deadline Miss Ratio (↓)"),
        (ax2, "isl_traffic_bits", "ISL Traffic (bits) (↓)"),
    ):
        vals = [_float(row, metric) for row in rows]
        errs = [_std(row, metric) for row in rows]
        ax.bar(x, vals, color=colors, yerr=errs, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_title(title)
        ax.set_ylim(bottom=0)

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
    sizes = sorted(set(
        int(r["algorithm"].split("n=")[1]) for r in rows if "n=" in r["algorithm"]
    ))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    for alg in algs:
        miss, miss_err, traffic, traffic_err = [], [], [], []
        for n in sizes:
            key = f"{alg}@n={n}"
            row = next((r for r in rows if r["algorithm"] == key), None)
            miss.append(_float(row, "realized_miss_ratio") if row else 0)
            miss_err.append(_std(row, "realized_miss_ratio") if row else 0)
            traffic.append(_float(row, "isl_traffic_bits") if row else 0)
            traffic_err.append(_std(row, "isl_traffic_bits") if row else 0)
        style = dict(label=ALG_LABELS.get(alg, alg),
                     color=ALG_COLORS.get(alg, "#888"), linewidth=2,
                     marker="s", capsize=3)
        ax1.errorbar(sizes, miss, yerr=miss_err, **style)
        ax2.errorbar(sizes, traffic, yerr=traffic_err, **style)

    ax1.set_xlabel("Number of Satellites")
    ax1.set_ylabel("Deadline Miss Ratio (↓)")
    ax2.set_xlabel("Number of Satellites")
    ax2.set_ylabel("ISL Traffic (bits) (↓)")
    for ax in (ax1, ax2):
        ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "E4_scalability.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def plot_all():
    _ensure_figures()
    for fn in [plot_E1, plot_E2, plot_E3, plot_E4]:
        fn()
