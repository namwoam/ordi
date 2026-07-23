"""Paired statistical comparison across seed-matched experiment runs.

Every ``run_E*`` function in :mod:`ordi.eval.experiments` shares one random
fault/environment draw per seed across every algorithm evaluated under that
seed. This module exploits that pairing instead of treating each algorithm's
per-seed samples as independent: for each condition (e.g. one E2 fault rate)
it compares every non-baseline algorithm against a baseline (ORDI by default)
using the per-seed difference, and reports a bootstrap confidence interval,
a paired-sample effect size, and paired significance tests alongside it.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy import stats

from ordi.eval.metrics import EpochMetrics

RESULTS_DIR = "results"


@dataclass(frozen=True)
class PairedComparison:
    """One metric's paired comparison of ``algorithm`` against ``baseline``.

    ``mean_diff`` and its CI are ``algorithm - baseline``; a metric where
    lower is better (e.g. deadline_miss_ratio) shows ORDI winning as negative.
    """
    algorithm: str
    baseline: str
    condition: str
    metric: str
    n: int
    mean_diff: float
    ci_lo: float
    ci_hi: float
    effect_size: float
    t_p_value: float
    wilcoxon_p_value: float


def cohens_d_paired(diff: np.ndarray) -> float:
    """Standardized paired effect size: mean difference over its std."""
    diff = np.asarray(diff, dtype=float)
    if len(diff) < 2:
        return 0.0
    sd = diff.std(ddof=1)
    return 0.0 if sd == 0.0 else float(diff.mean() / sd)


def bootstrap_mean_ci(values: Sequence[float], n_resamples: int = 10000,
                      confidence: float = 0.95, seed: int = 0):
    """Bootstrap CI for a sample mean; falls back to percentile if BCa fails.

    BCa's jackknife acceleration term is undefined for degenerate (zero- or
    near-zero-variance) samples, which arises often at n=4-8 with clipped
    metrics like a miss ratio pinned at 0.
    """
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        point = float(values.mean()) if len(values) else 0.0
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    try:
        res = stats.bootstrap(
            (values,), np.mean, n_resamples=n_resamples,
            confidence_level=confidence, method="BCa", random_state=rng,
        )
    except Exception:
        res = stats.bootstrap(
            (values,), np.mean, n_resamples=n_resamples,
            confidence_level=confidence, method="percentile", random_state=rng,
        )
    return (float(values.mean()), float(res.confidence_interval.low),
            float(res.confidence_interval.high))


def paired_comparison(baseline_name: str, other_name: str, condition: str,
                      metric: str, baseline_metrics: List[EpochMetrics],
                      other_metrics: List[EpochMetrics],
                      n_resamples: int = 10000, confidence: float = 0.95,
                      seed: int = 0) -> Optional[PairedComparison]:
    """Compare one metric between two seed-matched runs of equal length.

    Runs are matched by position: both lists must have been built by the same
    seed-ordered loop (true for every ``run_E*`` collapse in experiments.py),
    so index i in both lists shares the same seed (and, for E3, plane case).
    """
    n = min(len(baseline_metrics), len(other_metrics))
    if n < 2:
        return None
    baseline = np.array(
        [getattr(m, metric) for m in baseline_metrics[:n]], dtype=float
    )
    other = np.array(
        [getattr(m, metric) for m in other_metrics[:n]], dtype=float
    )
    diff = other - baseline
    mean_diff, ci_lo, ci_hi = bootstrap_mean_ci(
        diff, n_resamples, confidence, seed
    )
    effect = cohens_d_paired(diff)
    try:
        t_p_value = float(stats.ttest_rel(other, baseline).pvalue)
    except Exception:
        t_p_value = float("nan")
    try:
        wilcoxon_p_value = (
            float(stats.wilcoxon(other, baseline).pvalue)
            if np.any(diff != 0.0) else float("nan")
        )
    except Exception:
        wilcoxon_p_value = float("nan")
    return PairedComparison(
        other_name, baseline_name, condition, metric, n,
        mean_diff, ci_lo, ci_hi, effect, t_p_value, wilcoxon_p_value,
    )


def compare_all(results: Dict[str, List[EpochMetrics]],
                metric_keys: Sequence[str], baseline_alg: str = "ORDI",
                n_resamples: int = 10000, confidence: float = 0.95,
                seed: int = 0) -> List[PairedComparison]:
    """Compare every non-baseline algorithm against ``baseline_alg``.

    ``results`` keys are grouped by sweep condition (the text after the first
    ``@``, or "" when a run has none, as in E1) so E2/E3/E4's per-condition
    algorithm sets are compared independently.
    """
    grouped: Dict[str, Dict[str, List[EpochMetrics]]] = {}
    for key, metrics in results.items():
        alg_name, _, condition = key.partition("@")
        grouped.setdefault(condition, {})[alg_name] = metrics

    records = []
    for condition, algorithms in grouped.items():
        baseline_metrics = algorithms.get(baseline_alg)
        if not baseline_metrics:
            continue
        for alg_name, metrics in algorithms.items():
            if alg_name == baseline_alg:
                continue
            for metric in metric_keys:
                record = paired_comparison(
                    baseline_alg, alg_name, condition, metric,
                    baseline_metrics, metrics, n_resamples, confidence, seed,
                )
                if record is not None:
                    records.append(record)
    return records


def save_comparison_csv(exp_id: str, records: List[PairedComparison],
                        results_dir: str = RESULTS_DIR):
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"{exp_id}.csv")
    fields = [
        "condition", "baseline", "algorithm", "metric", "n",
        "mean_diff", "ci95_lo", "ci95_hi", "effect_size_dz",
        "paired_t_pvalue", "wilcoxon_pvalue",
    ]
    rows = [{
        "condition": r.condition, "baseline": r.baseline,
        "algorithm": r.algorithm, "metric": r.metric, "n": r.n,
        "mean_diff": r.mean_diff, "ci95_lo": r.ci_lo, "ci95_hi": r.ci_hi,
        "effect_size_dz": r.effect_size,
        "paired_t_pvalue": r.t_p_value,
        "wilcoxon_pvalue": r.wilcoxon_p_value,
    } for r in records]
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {path}")


__all__ = [
    "PairedComparison", "bootstrap_mean_ci", "cohens_d_paired",
    "compare_all", "paired_comparison", "save_comparison_csv",
]
