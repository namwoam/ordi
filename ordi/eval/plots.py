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
    "B1_direct_downlink":     "#457b9d",
    "B2_onboard_only":        "#1d3557",
    "B3_compression_only":    "#a8dadc",
    "B4_serval_like":         "#2a9d8f",
    "B5_seco_like":           "#e9c46a",
    "B6_full_replication":    "#f4a261",
    "B7_random_replication":  "#264653",
    "B8_cocoi_like":          "#8ecae6",
}

ALG_LABELS = {
    "ORDI":                   "ORDI",
    "B1_direct_downlink":     "B1: Direct Downlink",
    "B2_onboard_only":        "B2: Onboard-Only",
    "B3_compression_only":    "B3: Compress-Only",
    "B4_serval_like":         "B4: Serval-like",
    "B5_seco_like":           "B5: SECO-like",
    "B6_full_replication":    "B6: Full Replication",
    "B7_random_replication":  "B7: Random Repl.",
    "B8_cocoi_like":          "B8: CoCoI-like",
}


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


# ── E1: Core performance bar chart ───────────────────────────────────────────

def plot_E1():
    rows = _read_csv("E1_core")
    if not rows:
        print("No E1 data"); return

    metrics = ["deadline_miss_ratio", "delivered_utility", "partial_coverage",
               "energy_joules", "isl_traffic_bits"]
    titles  = ["Deadline Miss Ratio (↓)", "Delivered Utility (↑)",
               "Partial Coverage (↑)", "Energy (J) (↓)", "ISL Traffic (bits) (↓)"]

    algs = [r["algorithm"] for r in rows]
    colors = [ALG_COLORS.get(a, "#888") for a in algs]
    labels = [ALG_LABELS.get(a, a) for a in algs]

    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 4))
    fig.suptitle("E1: Core Performance — ORDI vs. Baselines", fontsize=12, fontweight="bold")

    for ax, metric, title in zip(axes, metrics, titles):
        vals = [_float(r, metric) for r in rows]
        bars = ax.bar(range(len(algs)), vals, color=colors)
        ax.set_title(title, fontsize=9)
        ax.set_xticks(range(len(algs)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        # Highlight ORDI bar
        for i, (bar, alg) in enumerate(zip(bars, algs)):
            if alg == "ORDI":
                bar.set_edgecolor("black")
                bar.set_linewidth(2)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "E1_core.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ── E2: Fault type radar / bar ────────────────────────────────────────────────

def plot_E2():
    rows = _read_csv("E2_fault_types")
    if not rows:
        print("No E2 data"); return

    scenarios = [r["algorithm"] for r in rows]
    miss_ratios = [_float(r, "deadline_miss_ratio") for r in rows]
    utilities   = [_float(r, "delivered_utility")   for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("E2: ORDI Robustness Across Fault Types", fontsize=12, fontweight="bold")

    ax1.bar(scenarios, miss_ratios, color="#e63946")
    ax1.set_ylabel("Deadline Miss Ratio")
    ax1.set_xticklabels(scenarios, rotation=30, ha="right", fontsize=8)
    ax1.set_title("Deadline Miss Ratio (↓)")

    ax2.bar(scenarios, utilities, color="#2a9d8f")
    ax2.set_ylabel("Delivered Utility")
    ax2.set_xticklabels(scenarios, rotation=30, ha="right", fontsize=8)
    ax2.set_title("Delivered Utility (↑)")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "E2_faults.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ── E3: Fault intensity line plots ────────────────────────────────────────────

def plot_E3():
    rows = _read_csv("E3_fault_intensity")
    if not rows:
        print("No E3 data"); return

    algs = ["ORDI", "B5_seco_like", "B6_full_replication"]
    fault_rates = sorted(set(
        float(r["algorithm"].split("fault=")[1]) for r in rows
        if "fault=" in r["algorithm"]
    ))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("E3: Graceful Degradation Under Increasing Fault Rate", fontsize=11, fontweight="bold")

    for alg in algs:
        miss_vals = []
        util_vals = []
        for rate in fault_rates:
            key = f"{alg}@fault={rate:.2f}"
            row = next((r for r in rows if r["algorithm"] == key), None)
            miss_vals.append(_float(row, "deadline_miss_ratio") if row else 0)
            util_vals.append(_float(row, "delivered_utility")   if row else 0)
        label = ALG_LABELS.get(alg, alg)
        color = ALG_COLORS.get(alg, "#888")
        lw = 2.5 if alg == "ORDI" else 1.5
        ax1.plot(fault_rates, miss_vals, label=label, color=color, linewidth=lw, marker="o")
        ax2.plot(fault_rates, util_vals, label=label, color=color, linewidth=lw, marker="o")

    ax1.set_xlabel("Fault Rate"); ax1.set_ylabel("Deadline Miss Ratio (↓)")
    ax1.legend(fontsize=8); ax1.set_title("Deadline Miss Ratio")
    ax2.set_xlabel("Fault Rate"); ax2.set_ylabel("Delivered Utility (↑)")
    ax2.legend(fontsize=8); ax2.set_title("Delivered Utility")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "E3_intensity.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ── E4: Scalability ───────────────────────────────────────────────────────────

def plot_E4():
    rows = _read_csv("E4_scalability")
    if not rows:
        print("No E4 data"); return

    algs = ["ORDI", "B5_seco_like"]
    sizes = sorted(set(
        int(r["algorithm"].split("n=")[1]) for r in rows if "n=" in r["algorithm"]
    ))

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.suptitle("E4: Scalability vs. Constellation Size", fontsize=11, fontweight="bold")

    for alg in algs:
        util_vals = []
        for n in sizes:
            key = f"{alg}@n={n}"
            row = next((r for r in rows if r["algorithm"] == key), None)
            util_vals.append(_float(row, "delivered_utility") if row else 0)
        ax.plot(sizes, util_vals, label=ALG_LABELS.get(alg, alg),
                color=ALG_COLORS.get(alg, "#888"), linewidth=2, marker="s")

    ax.set_xlabel("Number of Satellites"); ax.set_ylabel("Delivered Utility (↑)")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "E4_scalability.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ── E5: Deadline tightness ────────────────────────────────────────────────────

def plot_E5():
    rows = _read_csv("E5_deadline")
    if not rows:
        print("No E5 data"); return

    algs = ["ORDI", "B2_onboard_only", "B4_serval_like"]
    scales = sorted(set(
        int(r["algorithm"].split("slack=")[1].rstrip("s"))
        for r in rows if "slack=" in r["algorithm"]
    ))

    # Wildfire-median equivalent (most urgent task type) at each scale:
    # wildfire_median = scale × (300/600) = scale/2
    wildfire_medians = [s // 2 for s in scales]

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.suptitle("E5: Deadline Miss Rate vs. Deadline Scale\n"
                 "(log-normal σ=0.6; wildfire median = scale/2)",
                 fontsize=11, fontweight="bold")

    for alg in algs:
        miss_vals = []
        for sl in scales:
            key = f"{alg}@slack={sl}s"
            row = next((r for r in rows if r["algorithm"] == key), None)
            miss_vals.append(_float(row, "deadline_miss_ratio") if row else 1.0)
        ax.plot(wildfire_medians, miss_vals, label=ALG_LABELS.get(alg, alg),
                color=ALG_COLORS.get(alg, "#888"), linewidth=2, marker="^")

    ax.set_xlabel("Wildfire-task Deadline Median (s)  [scale × ½]")
    ax.set_ylabel("Deadline Miss Ratio (↓)")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "E5_deadline.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ── E6: λ_R sweep ────────────────────────────────────────────────────────────

def plot_E6():
    rows = _read_csv("E6_lambda_R")
    if not rows:
        print("No E6 data"); return

    lambdas = sorted(set(
        float(r["algorithm"].split("lambda_R=")[1])
        for r in rows if "lambda_R=" in r["algorithm"]
    ))

    miss_vals = [_float(next((r for r in rows if f"lambda_R={l}" in r["algorithm"]), {}),
                        "deadline_miss_ratio") for l in lambdas]
    util_vals = [_float(next((r for r in rows if f"lambda_R={l}" in r["algorithm"]), {}),
                        "delivered_utility")   for l in lambdas]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("E6: Effect of Replication Penalty λ_R", fontsize=11, fontweight="bold")
    ax1.plot(lambdas, miss_vals, color="#e63946", linewidth=2, marker="o")
    ax1.set_xlabel("λ_R"); ax1.set_ylabel("Deadline Miss Ratio (↓)")
    ax2.plot(lambdas, util_vals, color="#2a9d8f", linewidth=2, marker="o")
    ax2.set_xlabel("λ_R"); ax2.set_ylabel("Delivered Utility (↑)")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "E6_lambda_R.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ── E7: Correlated failures ───────────────────────────────────────────────────

def plot_E7():
    rows = _read_csv("E7_correlated")
    if not rows:
        print("No E7 data"); return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("E7: Correlated Orbital-Plane Outage", fontsize=11, fontweight="bold")

    for ax, metric, title in zip(axes,
                                  ["deadline_miss_ratio", "delivered_utility"],
                                  ["Deadline Miss Ratio (↓)", "Delivered Utility (↑)"]):
        labels = [r["algorithm"] for r in rows]
        vals   = [_float(r, metric) for r in rows]
        colors = ["#e63946" if "ORDI" in l else "#f4a261" for l in labels]
        ax.bar(range(len(labels)), vals, color=colors)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_title(title)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "E7_correlated.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ── E8: ILP vs greedy ─────────────────────────────────────────────────────────

def plot_E8():
    rows = _read_csv("E8_ilp_gap")
    if not rows:
        print("No E8 data"); return

    greedy_row = next((r for r in rows if "greedy" in r["algorithm"]), None)
    ilp_row    = next((r for r in rows if "ILP"    in r["algorithm"]), None)
    if not greedy_row or not ilp_row:
        return

    metrics = ["delivered_utility", "deadline_miss_ratio"]
    labels  = ["Delivered Utility", "Deadline Miss Ratio"]
    g_vals  = [_float(greedy_row, m) for m in metrics]
    i_vals  = [_float(ilp_row,    m) for m in metrics]

    x = np.arange(len(metrics))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.suptitle("E8: Greedy vs. ILP Optimality Gap", fontsize=11, fontweight="bold")
    ax.bar(x - w/2, g_vals, w, label="Greedy", color="#e63946")
    ax.bar(x + w/2, i_vals, w, label="ILP",    color="#457b9d")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.legend()
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "E8_ilp_gap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ── COTS: MobiCom24/BUPT-1 measurement-backed evaluation ────────────────────

def plot_COTS():
    rows = _read_csv("COTS_mobicom24")
    if not rows:
        print("No COTS data"); return

    metrics = ["deadline_miss_ratio", "delivered_utility", "objective",
               "energy_joules", "isl_traffic_bits"]
    titles = ["Deadline Miss Ratio (↓)", "Delivered Utility (↑)",
              "Objective (↑)", "Energy (J) (↓)", "ISL Traffic (bits) (↓)"]

    preferred_order = [
        "ORDI", "B5_seco_like", "B6_full_replication", "B8_cocoi_like",
        "B3_compression_only", "B2_onboard_only", "B4_serval_like",
        "B7_random_replication", "B1_direct_downlink",
    ]
    by_alg = {r["algorithm"]: r for r in rows}
    ordered = [by_alg[a] for a in preferred_order if a in by_alg]
    ordered += [r for r in rows if r["algorithm"] not in preferred_order]

    algs = [r["algorithm"] for r in ordered]
    colors = [ALG_COLORS.get(a, "#888") for a in algs]
    labels = [ALG_LABELS.get(a, a) for a in algs]

    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 4))
    fig.suptitle("COTS: ORDI with MobiCom24/BUPT-1 Atlas 200DK Measurements",
                 fontsize=12, fontweight="bold")

    for ax, metric, title in zip(axes, metrics, titles):
        vals = [_float(r, metric) for r in ordered]
        bars = ax.bar(range(len(algs)), vals, color=colors)
        ax.set_title(title, fontsize=9)
        ax.set_xticks(range(len(algs)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.tick_params(axis="y", labelsize=8)
        for bar, alg in zip(bars, algs):
            if alg == "ORDI":
                bar.set_edgecolor("black")
                bar.set_linewidth(2)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "COTS_mobicom24.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def plot_all():
    _ensure_figures()
    for fn in [plot_E1, plot_E2, plot_E3, plot_E4,
               plot_E5, plot_E6, plot_E7, plot_E8, plot_COTS]:
        fn()
