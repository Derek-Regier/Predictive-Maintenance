"""
src/evaluation/evaluate.py

Comprehensive evaluation of the trained backbone + NGBoost pipeline on the
held-out test set. Produces structured outputs consumed by the Streamlit
and Power BI dashboards.

Changes from v1
---------------
- calibration_scale is now read from the registry and applied at inference
  so all coverage metrics reflect post-calibration sigma.
- Short engines (fewer rows than seq_length) are padded by repeating their
  earliest row rather than silently skipped — they still receive a prediction,
  flagged with padded=True in the output CSV.

Two evaluation modes
--------------------
ALL-SEQUENCE    every sliding window in the test set (engine trajectories)
LAST-TIMESTEP   one prediction per engine at its final observed cycle
                (the operationally honest held-out metric)

Outputs written to reports/{dataset}/
--------------------------------------
metrics_summary.json    all scalar metrics
predictions_all.csv     every window: mu, sigma, alert tier, etc.
predictions_last.csv    one row per test engine (last-timestep)
calibration.csv         actual vs expected coverage at each confidence level
bucket_metrics.csv      RMSE / NASA split by RUL bucket

Usage
-----
    python src/evaluation/evaluate.py --dataset FD001
    python src/evaluation/evaluate.py --dataset all
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
import sys
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score
import torch
import yaml

# Path resolution 
_ROOT = Path(__file__).resolve().parents[2]
_TRAINING = _ROOT / "src" / "training"
if str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))

from shared import (
    DEVICE,
    TARGET_COL,
    NON_FEATURE_COLS,
    build_backbone,
)
# SERIALISATION HELPERS

def _to_python(obj):
    """Recursively convert numpy scalars/arrays to native Python types."""
    if isinstance(obj, dict): return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, list): return [_to_python(v) for v in obj]
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    return obj

def _safe_json_write(path: Path, data: dict) -> None:
    """Write JSON atomically via a .tmp file."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(_to_python(data), f, indent=2)
    shutil.move(str(tmp), str(path))

# METRIC FUNCTIONS
def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    NASA asymmetric scoring function. Penalises late predictions (d>0,
    predicting more remaining life than exists) more than early ones.
    Lower is better; perfect prediction = 0.
    """
    d = y_pred - y_true
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d  / 10) - 1)))

def within_n_cycles(y_true: np.ndarray, y_pred: np.ndarray, n: int) -> float:
    return float(np.mean(np.abs(y_pred - y_true) <= n))

def binary_failure_auc(
    y_true: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    horizon: int,
) -> Optional[float]:
    binary_true = (y_true < horizon).astype(int)
    if binary_true.sum() == 0 or binary_true.sum() == len(binary_true):
        return None
    probs = norm.cdf(horizon, loc=mu, scale=sigma)
    return float(roc_auc_score(binary_true, probs))

def calibration_table(
    y_true: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> pd.DataFrame:
    """
    Coverage at each confidence level vs expected. After calibration_scale
    is applied, the 90% row should read close to 0.90.
    """
    rows = []
    for alpha in [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]:
        z = norm.ppf((1 + alpha) / 2)
        in_band  = (y_true >= mu - z*sigma) & (y_true <= mu + z*sigma)
        coverage = float(in_band.mean())
        width = float(np.mean(2 * z * sigma))
        rows.append({
            "expected_coverage": alpha,
            "actual_coverage": coverage,
            "mean_pi_width": round(width, 3),
            "error": round(coverage - alpha, 4),
        })
    return pd.DataFrame(rows)

def bucket_metrics( y_true: np.ndarray, y_pred: np.ndarray, mu: np.ndarray, sigma: np.ndarray,
) -> pd.DataFrame:
    """Metrics split by RUL zone: early_life >80, mid_life 30-80, end_of_life ≤30."""
    buckets = {
        "early_life":y_true > 80,
        "mid_life": (y_true > 30) & (y_true <= 80),
        "end_of_life": y_true <= 30,
    }
    rows = []
    for name, mask in buckets.items():
        if mask.sum() == 0:
            continue
        yt, yp, m, s = y_true[mask], y_pred[mask], mu[mask], sigma[mask]
        rows.append({
            "bucket": name,
            "n_samples": int(mask.sum()),
            "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
            "mae": float(mean_absolute_error(yt, yp)),
            "nasa_score": nasa_score(yt, yp),
            "within_10_pct": within_n_cycles(yt, yp, 10),
            "mean_pi_width_90": float(np.mean(2 * norm.ppf(0.95) * s)),
        })
    return pd.DataFrame(rows)

def alert_tier(
    prob_failure_20: np.ndarray,
    prob_failure_50: np.ndarray,
) -> np.ndarray:
    tiers = np.full(len(prob_failure_20), "NOMINAL", dtype=object)
    tiers[prob_failure_50 >= 0.50] = "MONITOR"
    tiers[prob_failure_50 >= 0.75] = "WARNING"
    tiers[prob_failure_20 >= 0.90] = "CRITICAL"
    return tiers

# SEQUENCE CONSTRUCTION

def create_sequences_tracked( df: pd.DataFrame, seq_length: int,) -> tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sliding-window sequences with unit_number and cycle tracking.
    Returns: (X_tensor, y_array, unit_ids, cycle_ids)
    """
    X_seq, y_final, unit_ids, cycle_ids = [], [], [], []
    cols_to_drop = [c for c in NON_FEATURE_COLS if c in df.columns]

    for unit in df["unit_number"].unique():
        unit_data = df[df["unit_number"] == unit].sort_values("time_in_cycles")
        X_vals = unit_data.drop(columns=cols_to_drop).values
        y_vals = unit_data[TARGET_COL].values
        cycles = unit_data["time_in_cycles"].values

        if len(X_vals) >= seq_length:
            for i in range(len(X_vals) - seq_length + 1):
                X_seq.append(X_vals[i : i + seq_length])
                y_final.append(y_vals[i + seq_length - 1])
                unit_ids.append(unit)
                cycle_ids.append(cycles[i + seq_length - 1])

    if not X_seq:
        n_feat = len(df.columns) - len(cols_to_drop)
        return (torch.empty((0, seq_length, n_feat)),
                np.array([]), np.array([]), np.array([]))

    return (torch.tensor(np.array(X_seq), dtype=torch.float32),
        np.array(y_final),
        np.array(unit_ids),
        np.array(cycle_ids),
    )

def create_last_timestep(df:pd.DataFrame, seq_length: int,) -> tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
    """
    Final window per engine with padding for short engines.

    Engines shorter than seq_length are padded by repeating their earliest
    available row at the front of the window. This is imperfect (synthetic
    early history) but far better than silently dropping the engine from
    the fleet dashboard. A `padded` flag marks these predictions.

    Returns: (X_tensor, y_array, unit_ids, padded_flags)
    """
    X_last, y_last, unit_ids, padded = [], [], [], []
    cols_to_drop = [c for c in NON_FEATURE_COLS if c in df.columns]

    for unit in df["unit_number"].unique():
        unit_data = df[df["unit_number"] == unit].sort_values("time_in_cycles")
        X_vals = unit_data.drop(columns=cols_to_drop).values
        y_vals = unit_data[TARGET_COL].values

        if len(X_vals) >= seq_length:
            # Normal case: take the last seq_length rows
            X_last.append(X_vals[-seq_length:])
            padded.append(False)
        else:
            # Short engine: pad the front with the earliest row
            # so the model sees a full seq_length window.
            # The prediction is less reliable but the engine still appears
            # in the fleet table rather than being silently excluded.
            n_pad   = seq_length - len(X_vals)
            padded_rows = np.repeat(X_vals[:1], n_pad, axis=0)
            X_last.append(np.vstack([padded_rows, X_vals]))
            padded.append(True)
            print(f"Padded engine {unit}: {len(X_vals)} rows → {seq_length} "
                  f"(first row repeated {n_pad} times)")

        y_last.append(y_vals[-1])
        unit_ids.append(unit)

    return (
        torch.tensor(np.array(X_last), dtype=torch.float32),
        np.array(y_last),
        np.array(unit_ids),
        np.array(padded),
    )

# PIPELINE LOADING

def load_pipeline(dataset_key: str, registry: dict) -> tuple:
    """
    Reconstruct backbone and NGBoost from registry.
    Also reads calibration_scale — the sigma multiplier computed by
    train.py --calibrate_only. Defaults to 1.0 (no change) if missing.

    Returns: (backbone_model, ngb_model, seq_length, sigma_scale)
    """
    champion = registry[dataset_key]["champion"]
    bb_cfg = champion["backbone_config"]
    seq_length = bb_cfg["seq_length"]
    input_dim = bb_cfg["input_dim"]
    bb_kwargs = {k: v for k, v in bb_cfg.items()
                     if k not in ("input_dim", "seq_length")}

    # sigma_scale: applied to raw NGBoost sigma before computing probabilities.
    # >1 widens intervals (fixes overconfidence), <1 narrows them.
    sigma_scale = float(champion.get("calibration_scale", 1.0))
    if "calibration_scale" not in champion:
        print(f"  Warning: calibration_scale missing for {dataset_key}. "
              "Run: python src/training/train.py --calibrate_only")

    model = build_backbone(champion["backbone"], input_dim, bb_kwargs)
    model.load_state_dict(
        torch.load(
            Path(champion["artifacts"]["backbone"]),
            map_location=DEVICE,
            weights_only=True,
        )
    )
    model.eval()

    ngb_model = joblib.load(Path(champion["artifacts"]["meta_ngboost"]))

    print(f"  Loaded {champion['backbone']} backbone + NGBoost  "
          f"sigma_scale={sigma_scale:.4f}")
    return model, ngb_model, seq_length, sigma_scale
# INFERENCE

def run_inference(
    model: torch.nn.Module,
    ngb_model,
    X_tensor: torch.Tensor,
    sigma_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    End-to-end inference with sigma calibration.

    Step 1  Backbone.encode() -> latent feature vectors
    Step 2  NGBoost.pred_dist() -> Normal(mu, sigma_raw) per prediction
    Step 3  sigma_calibrated = sigma_raw * sigma_scale
    Step 4  Compute P(RUL<20) and P(RUL<50) from calibrated distribution

    sigma_scale > 1: widens intervals -> reduces P(RUL<N) slightly
    sigma_scale < 1: narrows intervals -> increases P(RUL<N)

    Returns: (mu, sigma_calibrated, prob_20, prob_50, X_features)
    X_features is returned to avoid a second backbone forward pass for NLL.
    """
    from torch.utils.data import DataLoader, TensorDataset

    loader   = DataLoader(TensorDataset(X_tensor), batch_size=256, shuffle=False)
    features = []
    model.eval()
    with torch.no_grad():
        for (x_batch,) in loader:
            features.append(model.encode(x_batch.to(DEVICE)).cpu().numpy())
    X_feat = np.concatenate(features, axis=0)

    dist = ngb_model.pred_dist(X_feat)
    mu  = dist.loc
    sigma_raw = dist.scale
    sigma = sigma_raw * sigma_scale   
    # apply calibration
    prob_20 = norm.cdf(20, loc=mu, scale=sigma)
    prob_50 = norm.cdf(50, loc=mu, scale=sigma)

    return mu, sigma, prob_20, prob_50, X_feat

# MAIN EVALUATION PIPELINE

def evaluate_dataset(dataset_key: str, registry_path: Path) -> None:
    print(f"  Evaluating {dataset_key}")

    with open(registry_path, "r") as f:
        registry = yaml.safe_load(f) or {}

    if dataset_key not in registry or "champion" not in registry[dataset_key]:
        print(f"No champion found for {dataset_key}. Run train.py first.")
        return

    model, ngb_model, seq_length, sigma_scale = load_pipeline(dataset_key, registry)
    backbone_name = registry[dataset_key]["champion"]["backbone"]

    test_path = _ROOT / "data" / "processed" / dataset_key / "test_features.csv"
    if not test_path.exists():
        raise FileNotFoundError(f"Test features not found: {test_path}")

    test_df = pd.read_csv(test_path)
    print(f"  Test set: {test_df['unit_number'].nunique()} engines  "
          f"({len(test_df):,} rows)")
    print(f"  Sigma calibration scale: {sigma_scale:.4f}")

    # ALL-SEQUENCE EVALUATION
    print("\n  Running all-sequence inference...")
    X_all, y_all, unit_ids, cycle_ids = create_sequences_tracked(test_df, seq_length)

    mu_all, sigma_all, prob20_all, prob50_all, X_feat_all = run_inference(
        model, ngb_model, X_all, sigma_scale
    )
    preds_all = mu_all   # NGBoost point prediction == distribution mean
    tiers_all = alert_tier(prob20_all, prob50_all)
    z90 = norm.ppf(0.95)
    lower90_all = mu_all  - z90 * sigma_all
    upper90_all = mu_all  + z90 * sigma_all
    residuals_all = preds_all - y_all

    predictions_all_df = pd.DataFrame({
        "unit_number": unit_ids,
        "cycle": cycle_ids,
        "true_rul": y_all,
        "pred_rul": mu_all.round(2),
        "pred_std": sigma_all.round(2),
        "lower_90": lower90_all.round(2),
        "upper_90": upper90_all.round(2),
        "prob_failure_20": prob20_all.round(4),
        "prob_failure_50": prob50_all.round(4),
        "alert_tier": tiers_all,
        "residual": residuals_all.round(2),
    })

    # Scalar metrics
    rmse_all = float(np.sqrt(mean_squared_error(y_all, preds_all)))
    mae_all = float(mean_absolute_error(y_all, preds_all))
    nll_all = float(-ngb_model.pred_dist(X_feat_all).logpdf(y_all).mean())
    nasa_all = nasa_score(y_all, preds_all)
    w5 = within_n_cycles(y_all, preds_all, 5)
    w10 = within_n_cycles(y_all, preds_all, 10)
    w15 = within_n_cycles(y_all, preds_all, 15)
    auc_20 = binary_failure_auc(y_all, mu_all, sigma_all, horizon=20)
    auc_50 = binary_failure_auc(y_all, mu_all, sigma_all, horizon=50)
    pi_w90 = float(np.mean(2 * z90 * sigma_all))

    print(f"  All-seq RMSE     : {rmse_all:.4f}")
    print(f"  All-seq MAE      : {mae_all:.4f}")
    print(f"  All-seq NLL      : {nll_all:.4f}")
    print(f"  NASA score       : {nasa_all:.1f}")
    print(f"  Within 10 cycles : {w10*100:.1f}%")
    if auc_20: print(f"  AUC (20-cycle)   : {auc_20:.4f}")
    if auc_50: print(f"  AUC (50-cycle)   : {auc_50:.4f}")

    cal_df = calibration_table(y_all, mu_all, sigma_all)
    print(f"\n  Calibration (post sigma_scale={sigma_scale:.4f}):")
    print(cal_df.to_string(index=False))

    bkt_df = bucket_metrics(y_all, preds_all, mu_all, sigma_all)
    print("\n  Metrics by RUL bucket:")
    print(bkt_df.to_string(index=False))

    # LAST-TIMESTEP EVALUATION ─
    print("\n  Running last-timestep inference")
    X_last, y_last, unit_last, padded_last = create_last_timestep(test_df, seq_length)

    mu_last, sigma_last, prob20_last, prob50_last, _ = run_inference(
        model, ngb_model, X_last, sigma_scale
    )
    preds_last = mu_last
    tiers_last = alert_tier(prob20_last, prob50_last)
    lower90_last = mu_last - z90 * sigma_last
    upper90_last = mu_last + z90 * sigma_last
    residuals_last = preds_last - y_last

    predictions_last_df = pd.DataFrame({
        "unit_number": unit_last,
        "true_rul": y_last,
        "pred_rul": mu_last.round(2),
        "pred_std": sigma_last.round(2),
        "lower_90": lower90_last.round(2),
        "upper_90": upper90_last.round(2),
        "prob_failure_20": prob20_last.round(4),
        "prob_failure_50": prob50_last.round(4),
        "alert_tier": tiers_last,
        "residual": residuals_last.round(2),
        "padded": padded_last,   # flag for short engines
    })

    # Separate metrics for non-padded engines (the clean headline numbers)
    clean_mask = ~padded_last
    y_clean  = y_last[clean_mask]
    preds_clean = preds_last[clean_mask]

    rmse_last = float(np.sqrt(mean_squared_error(y_clean, preds_clean)))
    mae_last = float(mean_absolute_error(y_clean, preds_clean))
    nasa_last = nasa_score(y_clean, preds_clean)
    w10_last = within_n_cycles(y_clean, preds_clean, 10)
    n_padded = int(padded_last.sum())

    print(f"  Last-timestep RMSE : {rmse_last:.4f}  "
          f"({len(y_clean)} engines, {n_padded} padded excluded from metrics)")
    print(f"  Last-timestep MAE  : {mae_last:.4f}")
    print(f"  Last-timestep NASA : {nasa_last:.1f}")
    print(f"  Within 10 cycles   : {w10_last*100:.1f}%")
    print(f"  Alert distribution : "
          f"{pd.Series(tiers_last).value_counts().to_dict()}")
    if n_padded:
        print(f"  Padded engines     : {n_padded} (short history, "
              "predictions less reliable)")

    # Fleet risk
    expected_20 = float(prob20_last.sum())
    expected_50 = float(prob50_last.sum())
    print(f"\n  Fleet risk (all {len(y_last)} engines incl. padded):")
    print(f"Expected failures in 20 cycles : {expected_20:.1f}")
    print(f"Expected failures in 50 cycles : {expected_50:.1f}")

    # Metrics summary
    metrics_summary = {
        "dataset": dataset_key,
        "backbone": backbone_name,
        "sigma_scale": sigma_scale,
        "all_seq": {
            "rmse": rmse_all, "mae": mae_all, "nll": nll_all,
            "nasa_score": nasa_all,
            "within_5_pct": w5, "within_10_pct": w10, "within_15_pct": w15,
            "auc_failure_20": auc_20, "auc_failure_50": auc_50,
            "mean_pi_width_90": pi_w90,
        },
        "last_timestep": {
            "rmse": rmse_last, "mae": mae_last,
            "nasa_score": nasa_last, "within_10_pct": w10_last,
            "n_engines_evaluated": int(clean_mask.sum()),
            "n_engines_padded": n_padded,
        },
        "fleet_risk": {
            "n_test_engines": int(len(y_last)),
            "expected_failures_20": expected_20,
            "expected_failures_50": expected_50,
            "pct_critical": float((tiers_last == "CRITICAL").mean()),
            "pct_warning": float((tiers_last == "WARNING").mean()),
            "pct_monitor": float((tiers_last == "MONITOR").mean()),
            "pct_nominal": float((tiers_last == "NOMINAL").mean()),
        },
        "calibration": cal_df.set_index("expected_coverage")
                             ["actual_coverage"].to_dict(),
    }

    # Save outputs
    out_dir = _ROOT / "reports" / dataset_key
    out_dir.mkdir(parents=True, exist_ok=True)

    _safe_json_write(out_dir / "metrics_summary.json", metrics_summary)
    predictions_all_df.to_csv(out_dir / "predictions_all.csv",  index=False)
    predictions_last_df.to_csv(out_dir / "predictions_last.csv", index=False)
    cal_df.to_csv(out_dir / "calibration.csv",    index=False)
    bkt_df.to_csv(out_dir / "bucket_metrics.csv", index=False)

    print(f"\n  Saved to {out_dir}/")

# CLI 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate trained pipeline on held-out test set."
    )
    parser.add_argument(
        "--dataset",
        choices=["FD001", "FD002", "FD003", "FD004", "all"],
        default="FD001",
    )
    args = parser.parse_args()

    registry_path = _ROOT / "config" / "model_registry.yaml"
    datasets = (
        ["FD001", "FD002", "FD003", "FD004"]
        if args.dataset == "all" else [args.dataset]
    )
    for ds in datasets:
        evaluate_dataset(ds, registry_path)