"""
All data loading for the dashboard — every function is decorated with
@st.cache_data so files are only read from disk once, then cached in
memory. On widget interaction Streamlit reruns the page script but
cache_data functions return instantly from the in-memory cache.

ttl=600 means the cache expires after 10 minutes. After expiry the
next call re-reads the file. This matters when evaluate.py or
health_monitor.py is run and writes new CSVs while the dashboard is
open it will pick up updates within 10 minutes automatically.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st
import yaml

# Project root: data_loader.py lives at dashboard/utils/data_loader.py
#   parents[0] = dashboard/utils/
#   parents[1] = dashboard/
#   parents[2] = project root
ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
CONFIG = ROOT / "config"


# PREDICTION DATA

@st.cache_data(ttl=600)
def load_predictions_last(dataset: str) -> pd.DataFrame:
    """
    One row per test engine at its final observed cycle.
    This is the primary fleet table data — used on Fleet Overview and
    as the summary row in Engine Deep Dive.

    Columns: unit_number, true_rul, pred_rul, pred_std, lower_90, upper_90,
             prob_failure_20, prob_failure_50, alert_tier, residual, padded
    """
    path = REPORTS / dataset / "predictions_last.csv"
    if not path.exists():
        st.warning(f"predictions_last.csv not found for {dataset}. "
                   "Run evaluate.py first.")
        return pd.DataFrame()
    df = pd.read_csv(path)
    # Add display-friendly alert column with emoji
    if "alert_tier" in df.columns:
        _emoji = {"CRITICAL": "🔴", "WARNING": "🟡",
                  "MONITOR": "🔵", "Normal": "🟢"}
        df["alert_display"] = df["alert_tier"].map(
            lambda t: f"{_emoji.get(t, '')} {t}"
        )
    return df


@st.cache_data(ttl=600)
def load_predictions_all(dataset: str) -> pd.DataFrame:
    """
    Every sliding window prediction across all test engines.
    Used for the RUL trajectory and failure probability charts on
    Engine Deep Dive — filter this df by unit_number to get one engine.

    Columns: unit_number, cycle, true_rul, pred_rul, pred_std,
             lower_90, upper_90, prob_failure_20, prob_failure_50,
             alert_tier, residual
    """
    path = REPORTS / dataset / "predictions_all.csv"
    if not path.exists():
        st.warning(f"predictions_all.csv not found for {dataset}.")
        return pd.DataFrame()
    return pd.read_csv(path)


# METRICS AND EVALUATION SUMMARIES

@st.cache_data(ttl=600)
def load_metrics_summary(dataset: str) -> dict:
    """
    All scalar evaluation metrics for one dataset — RMSE, MAE, NLL,
    NASA score, within-N cycles, AUC, fleet risk counts, calibration
    coverage dict.

    Used on Fleet Overview (fleet risk numbers) and Model Performance
    (comparison table).
    """
    path = REPORTS / dataset / "metrics_summary.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


@st.cache_data(ttl=600)
def load_calibration(dataset: str) -> pd.DataFrame:
    """
    Coverage table: expected_coverage, actual_coverage, mean_pi_width, error.
    One row per confidence level (0.50, 0.60, 0.70, 0.80, 0.90, 0.95).
    Used for the calibration plot on Model Performance.
    """
    path = REPORTS / dataset / "calibration.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(ttl=600)
def load_bucket_metrics(dataset: str) -> pd.DataFrame:
    """
    Metrics split by RUL bucket: early_life (>80), mid_life (30-80),
    end_of_life (≤30). Columns: bucket, n_samples, rmse, mae,
    nasa_score, within_10_pct, mean_pi_width_90.
    Used for the bucket RMSE chart on Model Performance.
    """
    path = REPORTS / dataset / "bucket_metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)

# HEALTH MONITORING DATA 

@st.cache_data(ttl=600)
def load_health_indices(dataset: str) -> pd.DataFrame:
    """
    Per-window VAE health metrics for all test engines.
    Returns an empty DataFrame if health_monitor.py has not been run yet.

    Columns: unit_number, cycle, true_rul, kl_div, js_div, wasserstein,
             recon_error, drift_flag, op_cluster, latent_mu_norm
    """
    path = REPORTS / dataset / "health_indices.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(ttl=600)
def load_health_summary(dataset: str) -> dict:
    """
    Fleet-level VAE statistics: healthy baseline mean/std per metric,
    drift thresholds per cluster, n_engines, n_drifted_engines.
    """
    path = REPORTS / dataset / "health_summary.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


# CONFIGURATION

@st.cache_data(ttl=3600)   # registry rarely changes — 1 hour TTL
def load_registry() -> dict:
    """
    The full model_registry.yaml — backbone type, metrics, calibration
    scale, VAE config, artifact paths for all four datasets.
    Used on Model Performance to show configuration alongside results.
    """
    path = CONFIG / "model_registry.yaml"
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}



# LOAD ALL DATASETS AT ONCE
DATASETS = ["FD001", "FD002", "FD003", "FD004"]

def load_all_metrics() -> dict[str, dict]:
    """
    Load metrics_summary.json for all four datasets in one call.
    Returns {dataset_key: metrics_dict}.
    Used on Model Performance for the cross-dataset comparison table.
    """
    return {ds: load_metrics_summary(ds) for ds in DATASETS}

def load_all_calibration() -> dict[str, pd.DataFrame]:
    """Load calibration.csv for all datasets."""
    return {ds: load_calibration(ds) for ds in DATASETS}

def load_all_bucket_metrics() -> dict[str, pd.DataFrame]:
    """Load bucket_metrics.csv for all datasets."""
    return {ds: load_bucket_metrics(ds) for ds in DATASETS}