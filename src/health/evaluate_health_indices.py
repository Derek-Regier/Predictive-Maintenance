"""
src/health/evaluate_health_indices.py

Scores every health index in health_indices.csv against the standard
prognostic-suitability metrics, so "which index should we trust" is
answered with a number instead of an opinion.

Outputs
-------
reports/{dataset}/health_index_metrics.csv — one row per index:

    index                    column name from health_indices.csv
    monotonicity             does it move consistently in one direction
    trendability             do all engines trend the same way
    prognosability           is the value at failure consistent across engines
    mean_engine_spearman     mean within-engine rank correlation with RUL
    frac_correct_direction   fraction of engines where the sign is right
    fleet_spearman           pooled correlation across all windows
    composite                mean of the three Coble metrics
    n_engines                engines the metrics were computed over

Why these three metrics
-----------------------
They come from Coble & Hines (2009), "Identifying Optimal Prognostic
Parameters from Data" — the standard reference for choosing a health
indicator in PHM, and the framing an interviewer in this field will
recognise. Each one catches a different way an index can be useless:

  monotonicity   An index that oscillates cannot support extrapolation
                 to a failure threshold, however well it correlates
                 with RUL on average.

  trendability   An index can be beautifully monotone on each engine
                 while trending UP on some and DOWN on others. Pooled
                 correlation hides this completely. On the original
                 FD001 run, reconstruction error trended the right way
                 on 46 engines and the wrong way on 34 — a coin flip
                 that its fleet-level correlation of -0.32 concealed.

  prognosability If engines fail at wildly different index values,
                 there is no fixed threshold to alarm on, so the index
                 cannot drive a maintenance decision no matter how
                 cleanly it trends.

A high composite score does not by itself make an index a good RUL
predictor — these measure suitability as a *degradation indicator*, not
predictive accuracy. That distinction is worth keeping straight.

Usage
-----
    python src/health/evaluate_health_indices.py --dataset FD001
    python src/health/evaluate_health_indices.py --dataset all
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]

# Every index the health monitor may emit. Missing columns are skipped,
# so this file works against both the old and new CSV schemas.
CANDIDATE_INDICES = [
    "mahalanobis",
    "fisher_rao",
    "kl_div",
    "js_div",
    "wasserstein",
    "mahalanobis_self",
    "fisher_rao_self",
    "recon_error",
    "latent_mu_norm",
    "latent_mu_centered",
    "health_score",
]

# Indices where a HIGHER value means healthier. Their expected
# correlation with RUL is positive, and their monotone direction is
# reversed relative to everything else.
HIGHER_IS_HEALTHIER = {"health_score"}

MIN_WINDOWS_PER_ENGINE = 10


def monotonicity(series_by_engine: list[np.ndarray]) -> float:
    """
    Coble & Hines monotonicity:

        M = mean_engines | #(positive deltas) - #(negative deltas) | / (n - 1)

    Ranges from 0 (equal up and down movement — pure noise) to 1
    (strictly monotone). The absolute value means direction does not
    matter here; trendability is what catches inconsistent direction.

    Note this is sensitive to high-frequency noise: an index with a
    strong underlying trend plus jitter scores poorly because the
    jitter flips the sign of many individual deltas. That is arguably
    the correct behaviour for a signal you intend to extrapolate, but
    it does mean a lightly smoothed version of an index will always
    score higher than its raw form.
    """
    vals = []
    for arr in series_by_engine:
        if len(arr) < 2:
            continue
        d = np.diff(arr)
        n_pos = int((d > 0).sum())
        n_neg = int((d < 0).sum())
        vals.append(abs(n_pos - n_neg) / len(d))
    return float(np.mean(vals)) if vals else float("nan")


def trendability(series_by_engine: list[np.ndarray]) -> float:
    """
    Coble & Hines trendability: the MINIMUM absolute correlation between
    the index and time, taken across engines.

        T = min_engines | corr(index, time) |

    Using the minimum rather than the mean is deliberate and harsh — it
    asks whether EVERY engine shows the trend, because a fleet monitor
    that works on 90% of units and is silent on the rest is not a fleet
    monitor. With ~90 engines the minimum is dominated by the single
    worst unit, so the mean is also reported separately in the output
    CSV via mean_engine_spearman.

    Correlation is against cycle index rather than RUL so this stays a
    measure of trend shape and does not become a second copy of the
    RUL correlation.
    """
    vals = []
    for arr in series_by_engine:
        if len(arr) < 3 or np.ptp(arr) == 0:
            continue
        t = np.arange(len(arr), dtype=np.float64)
        r = np.corrcoef(arr, t)[0, 1]
        if np.isfinite(r):
            vals.append(abs(r))
    return float(np.min(vals)) if vals else float("nan")


def prognosability(series_by_engine: list[np.ndarray]) -> float:
    """
    Coble & Hines prognosability: how tightly the index clusters at
    failure, relative to how far it travels over life.

        P = exp( - std(failure values)
                 / mean| failure value - start value | )

    Ranges from 0 (failure values scattered so widely that no threshold
    separates them) to 1 (every engine fails at the same value).

    The denominator is what makes this scale-free: an index whose
    failure values vary by 0.5 is excellent if it travels 50 over its
    life and useless if it travels 0.5.
    """
    finals, deltas = [], []
    for arr in series_by_engine:
        if len(arr) < 2:
            continue
        finals.append(arr[-1])
        deltas.append(abs(arr[-1] - arr[0]))

    if len(finals) < 2:
        return float("nan")

    finals = np.asarray(finals, dtype=np.float64)
    mean_delta = float(np.mean(deltas))
    if mean_delta <= 0 or not np.isfinite(mean_delta):
        return float("nan")

    return float(np.exp(-float(np.std(finals)) / mean_delta))


def evaluate_dataset(dataset_key: str) -> pd.DataFrame:
    """Compute all metrics for every available index in one dataset."""
    path = _ROOT / "reports" / dataset_key / "health_indices.csv"
    if not path.exists():
        print(f"  {dataset_key}: health_indices.csv not found — "
              f"run health_monitor.py first. Skipping.")
        return pd.DataFrame()

    df = pd.read_csv(path)
    available = [c for c in CANDIDATE_INDICES if c in df.columns]
    if not available:
        print(f"  {dataset_key}: no recognised index columns found. Skipping.")
        return pd.DataFrame()

    print(f"  {dataset_key}: {len(df):,} windows, "
          f"{df['unit_number'].nunique()} engines, "
          f"{len(available)} indices")

    # Group once and reuse. Engines with too few windows are dropped
    # rather than included with noisy estimates — a 4-window engine can
    # trivially score 1.0 on monotonicity and pollute the average.
    groups = [
        g.sort_values("cycle")
        for _, g in df.groupby("unit_number")
        if len(g) >= MIN_WINDOWS_PER_ENGINE
    ]
    if not groups:
        print(f"  {dataset_key}: no engine has >= {MIN_WINDOWS_PER_ENGINE} windows.")
        return pd.DataFrame()

    rows = []
    for col in available:
        series = [g[col].to_numpy(dtype=np.float64) for g in groups]

        # Per-engine rank correlation with RUL.
        rhos = []
        for g in groups:
            if g[col].nunique() < 3 or g["true_rul"].nunique() < 3:
                continue
            rho = g[col].corr(g["true_rul"], method="spearman")
            if np.isfinite(rho):
                rhos.append(rho)
        rhos = np.asarray(rhos, dtype=np.float64)

        expect_positive = col in HIGHER_IS_HEALTHIER
        if len(rhos):
            frac_correct = float((rhos > 0).mean() if expect_positive
                                 else (rhos < 0).mean())
            mean_rho = float(rhos.mean())
        else:
            frac_correct = float("nan")
            mean_rho = float("nan")

        mono = monotonicity(series)
        trend = trendability(series)
        prog = prognosability(series)

        fleet_rho = df[col].corr(df["true_rul"], method="spearman")

        composite_parts = [v for v in (mono, trend, prog) if np.isfinite(v)]
        composite = float(np.mean(composite_parts)) if composite_parts else float("nan")

        rows.append({
            "index": col,
            "monotonicity": mono,
            "trendability": trend,
            "prognosability": prog,
            "mean_engine_spearman": mean_rho,
            "frac_correct_direction": frac_correct,
            "fleet_spearman": float(fleet_rho) if np.isfinite(fleet_rho) else float("nan"),
            "composite": composite,
            "n_engines": int(len(series)),
        })

    out = pd.DataFrame(rows).sort_values("composite", ascending=False)

    out_path = _ROOT / "reports" / dataset_key / "health_index_metrics.csv"
    out.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")

    print()
    print(out.to_string(
        index=False,
        float_format=lambda v: f"{v:.4f}",
        columns=["index", "monotonicity", "trendability", "prognosability",
                 "mean_engine_spearman", "frac_correct_direction", "composite"],
    ))
    print()

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Score health indices on prognostic-suitability metrics."
    )
    parser.add_argument(
        "--dataset",
        choices=["FD001", "FD002", "FD003", "FD004", "all"],
        default="FD001",
    )
    args = parser.parse_args()

    datasets = (
        ["FD001", "FD002", "FD003", "FD004"]
        if args.dataset == "all" else [args.dataset]
    )

    print("Health index quality evaluation")
    print("(Coble & Hines prognostic metrics — higher is better on all three)")
    print()

    for ds in datasets:
        evaluate_dataset(ds)