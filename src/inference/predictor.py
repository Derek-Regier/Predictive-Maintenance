"""
src/inference/predictor.py

Unified inference class for the backbone + NGBoost pipeline.

Loads all trained artifacts from model_registry.yaml and exposes two methods:
  predict()       — single engine, returns full prediction dict
  predict_fleet() — multiple engines, returns per-engine predictions + fleet summary

The sigma calibration scale is applied automatically at inference time, so
P(RUL < N) values and alert tiers reflect the corrected uncertainty estimates.

Input contract
--------------
Both methods expect a DataFrame that is already:
  1. Preprocessed  (op_cluster assigned, sensors normalised per cluster)
  2. Feature-engineered (rolling stats, lags, deltas, log_cycles applied)

This matches the format of train_features.csv and test_features.csv.
The API layer is responsible for running preprocessing before calling predictor.

Example
-------
    predictor = RULPredictor("FD001", "config/model_registry.yaml")
    engine_df = pd.read_csv("data/processed/FD001/test_features.csv")
    engine_df = engine_df[engine_df["unit_number"] == 1]

    result = predictor.predict(engine_df)
    print(result)
    # {
    #   "rul_mean": 42.3,
    #   "rul_std": 8.1,        ← calibrated sigma
    #   "lower_90": 28.9,
    #   "upper_90": 55.7,
    #   "prob_failure_20": 0.043,
    #   "prob_failure_50": 0.712,
    #   "alert_tier": "WARNING",
    #   "sigma_scale": 1.412,  ← what was applied
    # }
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
import torch
import yaml

# ── Path resolution ───────────────────────────────────────────────────────────
# predictor.py lives at src/inference/predictor.py
#   parents[0] = src/inference/
#   parents[1] = src/
#   parents[2] = project root
_ROOT     = Path(__file__).resolve().parents[2]
_TRAINING = _ROOT / "src" / "training"
if str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))

from shared import (
    DEVICE,
    NON_FEATURE_COLS,
    TARGET_COL,
    build_backbone,
)


class RULPredictor:
    """
    Loads a trained backbone + NGBoost model for one dataset and exposes
    a predict() interface. Sigma calibration is applied automatically.

    Parameters
    ----------
    dataset_key     : one of "FD001" / "FD002" / "FD003" / "FD004"
    registry_path   : path to config/model_registry.yaml
    alert_thresholds: optional dict overriding the default P(RUL < N) thresholds
                      for CRITICAL / WARNING / MONITOR tiers
    """

    # Default alert thresholds — tunable per deployment based on cost ratio
    # of unplanned failure vs unnecessary early maintenance
    DEFAULT_THRESHOLDS = {
        "critical_prob_20": 0.90,   # P(RUL < 20) ≥ 0.90 → CRITICAL
        "warning_prob_50":  0.75,   # P(RUL < 50) ≥ 0.75 → WARNING
        "monitor_prob_50":  0.50,   # P(RUL < 50) ≥ 0.50 → MONITOR
    }

    def __init__(
        self,
        dataset_key:      str,
        registry_path:    str | Path,
        alert_thresholds: Optional[dict] = None,
    ) -> None:
        self.dataset_key  = dataset_key
        self.thresholds   = alert_thresholds or self.DEFAULT_THRESHOLDS
        self._load_artifacts(dataset_key, Path(registry_path))

    # ── Artifact loading ──────────────────────────────────────────────────────

    def _load_artifacts(self, dataset_key: str, registry_path: Path) -> None:
        """
        Read model config from registry, rebuild the backbone architecture,
        load trained weights, and load the NGBoost model.

        Also reads calibration_scale — the sigma multiplier computed by
        train.py's calibrate_sigma(). Defaults to 1.0 if not yet computed.
        """
        with open(registry_path, "r") as f:
            registry = yaml.safe_load(f) or {}

        if dataset_key not in registry or "champion" not in registry[dataset_key]:
            raise ValueError(
                f"No champion config for {dataset_key} in {registry_path}. "
                "Run train.py first."
            )

        champion = registry[dataset_key]["champion"]
        bb_cfg   = champion["backbone_config"]

        # Sequence length is part of the trained config — must match inference
        self.seq_length = bb_cfg["seq_length"]

        # sigma_scale: 1.0 = uncalibrated (neutral), >1 = widen, <1 = narrow
        # calibrate_sigma() in train.py writes this value to the registry.
        # If it's missing the model hasn't been calibrated yet — we warn and
        # default to 1.0 so inference still works.
        self.sigma_scale = float(champion.get("calibration_scale", 1.0))
        if "calibration_scale" not in champion:
            print(f"  Warning: no calibration_scale found for {dataset_key}. "
                  "Run: python src/training/train.py --dataset "
                  f"{dataset_key} --calibrate_only")

        # Rebuild the backbone architecture from config and load weights
        bb_name   = champion["backbone"]
        input_dim = bb_cfg["input_dim"]
        bb_kwargs = {k: v for k, v in bb_cfg.items()
                     if k not in ("input_dim", "seq_length")}

        self.backbone = build_backbone(bb_name, input_dim, bb_kwargs)
        self.backbone.load_state_dict(
            torch.load(
                Path(champion["artifacts"]["backbone"]),
                map_location=DEVICE,
                weights_only=True,
            )
        )
        self.backbone.eval()

        # Load the fitted NGBoost meta-model
        self.ngb = joblib.load(Path(champion["artifacts"]["meta_ngboost"]))

        print(f"  RULPredictor ready: {dataset_key} | "
              f"backbone={bb_name} | "
              f"sigma_scale={self.sigma_scale:.4f}")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_window(self, engine_df: pd.DataFrame) -> torch.Tensor:
        """
        Extract the last seq_length cycles from an engine's history and
        return as a (1, seq_length, n_features) tensor.

        We always take the LAST seq_length rows (sorted by time_in_cycles)
        because in deployment we have the full observed history up to now
        and want to predict at the current moment — the most recent window.

        Raises ValueError if the engine has fewer rows than seq_length.
        """
        cols_to_drop = [c for c in NON_FEATURE_COLS if c in engine_df.columns]
        X = (engine_df
             .sort_values("time_in_cycles")
             .drop(columns=cols_to_drop)
             .values)

        if len(X) < self.seq_length:
            raise ValueError(
                f"Engine has {len(X)} rows but seq_length={self.seq_length}. "
                "Provide at least seq_length rows of sensor history."
            )

        window = X[-self.seq_length:]                              # (seq, feat)
        return torch.tensor(window[np.newaxis], dtype=torch.float32)  # (1, seq, feat)

    def _run_inference(
        self,
        X_tensor: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Forward pass through backbone.encode() then NGBoost.pred_dist().
        Applies sigma calibration scale before computing failure probabilities.

        The calibration step is:
            sigma_calibrated = sigma_raw * sigma_scale

        For overconfident models (sigma_scale > 1): intervals widen,
            P(RUL < N) decreases — fewer false CRITICAL alerts.
        For underconfident models (sigma_scale < 1): intervals narrow,
            P(RUL < N) increases — suppressed alerts (like FD004) now fire.

        Returns: (mu, sigma_calibrated, prob_failure_20, prob_failure_50)
        """
        with torch.no_grad():
            features = self.backbone.encode(X_tensor.to(DEVICE)).cpu().numpy()

        dist      = self.ngb.pred_dist(features)
        mu        = dist.loc                            # predicted mean RUL
        sigma_raw = dist.scale                          # raw NGBoost sigma
        sigma     = sigma_raw * self.sigma_scale        # calibrated sigma

        prob_20 = norm.cdf(20, loc=mu, scale=sigma)    # P(RUL < 20 cycles)
        prob_50 = norm.cdf(50, loc=mu, scale=sigma)    # P(RUL < 50 cycles)

        return mu, sigma, prob_20, prob_50

    def _assign_alert_tier(
        self,
        prob_20: float,
        prob_50: float,
    ) -> str:
        """
        Map failure probabilities to a maintenance alert tier.

        Applied in ascending severity order so the most severe label wins.
        Thresholds are checked against DEFAULT_THRESHOLDS or the custom
        dict supplied at construction.

        CRITICAL  → immediate action, failure likely within 20 cycles
        WARNING   → schedule maintenance within this window (50 cycles)
        MONITOR   → flag for next inspection
        NOMINAL   → no action required
        """
        if prob_20 >= self.thresholds["critical_prob_20"]:
            return "CRITICAL"
        if prob_50 >= self.thresholds["warning_prob_50"]:
            return "WARNING"
        if prob_50 >= self.thresholds["monitor_prob_50"]:
            return "MONITOR"
        return "NOMINAL"

    # ── Public interface ──────────────────────────────────────────────────────

    def predict(self, engine_df: pd.DataFrame) -> dict:
        """
        Full probabilistic RUL prediction for one engine.

        Parameters
        ----------
        engine_df : preprocessed + feature-engineered DataFrame for one engine.
                    Must have at least seq_length rows sorted by time_in_cycles.

        Returns
        -------
        dict with keys:
            dataset          dataset this model was trained on
            rul_mean         predicted mean RUL in cycles (μ)
            rul_std          calibrated uncertainty in cycles (σ × scale)
            rul_std_raw      uncalibrated NGBoost sigma (before scaling)
            lower_90         lower bound of 90% prediction interval
            upper_90         upper bound of 90% prediction interval
            prob_failure_20  P(true RUL < 20 cycles) — drives CRITICAL alert
            prob_failure_50  P(true RUL < 50 cycles) — drives WARNING alert
            alert_tier       CRITICAL / WARNING / MONITOR / NOMINAL
            sigma_scale      the calibration multiplier that was applied
        """
        X_tensor             = self._build_window(engine_df)
        mu, sigma, p20, p50  = self._run_inference(X_tensor)

        # Squeeze from array shape (1,) to scalar
        mu_val    = float(mu[0])
        sigma_val = float(sigma[0])
        p20_val   = float(p20[0])
        p50_val   = float(p50[0])

        z90 = norm.ppf(0.95)    # z-score for 90% two-sided interval

        return {
            "dataset":         self.dataset_key,
            "rul_mean":        round(mu_val, 2),
            "rul_std":         round(sigma_val, 2),
            "rul_std_raw":     round(sigma_val / self.sigma_scale, 2),
            "lower_90":        round(mu_val - z90 * sigma_val, 2),
            "upper_90":        round(mu_val + z90 * sigma_val, 2),
            "prob_failure_20": round(p20_val, 4),
            "prob_failure_50": round(p50_val, 4),
            "alert_tier":      self._assign_alert_tier(p20_val, p50_val),
            "sigma_scale":     self.sigma_scale,
        }

    def predict_fleet(
        self,
        fleet_df: pd.DataFrame,
    ) -> tuple[dict[int, dict], dict]:
        """
        Predict RUL for every engine in fleet_df simultaneously.

        Parameters
        ----------
        fleet_df : preprocessed + feature-engineered DataFrame containing
                   multiple engines (unit_number column required).

        Returns
        -------
        predictions : dict mapping unit_number → prediction dict
                      (same structure as predict() return value)
        fleet_risk  : aggregated fleet-level risk summary dict with keys:
            n_engines             total engines with valid predictions
            expected_failures_20  Σ P(RUL < 20) across fleet
            expected_failures_50  Σ P(RUL < 50) across fleet
            n_critical / n_warning / n_monitor / n_nominal   alert counts

        Why Σ P(RUL < N)?
        -----------------
        This is the expected number of engines that will fail within N cycles.
        It's an actuarial aggregation: if one engine has P=0.3 and another
        has P=0.7, the expected failure count is 1.0. This is the fleet
        health number shown in the Power BI stakeholder dashboard.
        """
        predictions: dict[int, dict] = {}

        for unit in fleet_df["unit_number"].unique():
            engine_df = fleet_df[fleet_df["unit_number"] == unit]
            try:
                predictions[int(unit)] = self.predict(engine_df)
            except ValueError as e:
                # Engine has fewer rows than seq_length — record error but continue
                predictions[int(unit)] = {
                    "error": str(e),
                    "alert_tier": "UNKNOWN",
                }

        # Fleet-level aggregation (only over engines with valid predictions)
        valid = {k: v for k, v in predictions.items() if "error" not in v}

        if valid:
            prob_20s = np.array([v["prob_failure_20"] for v in valid.values()])
            prob_50s = np.array([v["prob_failure_50"] for v in valid.values()])
            tiers    = [v["alert_tier"] for v in valid.values()]

            fleet_risk = {
                "n_engines":            len(valid),
                "n_errors":             len(predictions) - len(valid),
                "expected_failures_20": float(prob_20s.sum()),
                "expected_failures_50": float(prob_50s.sum()),
                "n_critical":  tiers.count("CRITICAL"),
                "n_warning":   tiers.count("WARNING"),
                "n_monitor":   tiers.count("MONITOR"),
                "n_nominal":   tiers.count("NOMINAL"),
                "pct_at_risk": float(
                    (tiers.count("CRITICAL") + tiers.count("WARNING"))
                    / len(valid)
                ),
            }
        else:
            fleet_risk = {"n_engines": 0, "n_errors": len(predictions)}

        return predictions, fleet_risk


# ── Module-level convenience loader ──────────────────────────────────────────

def load_predictor(
    dataset_key:   str,
    registry_path: str | Path = "config/model_registry.yaml",
) -> RULPredictor:
    """
    Convenience function to load a predictor from anywhere in the codebase
    without needing to import the class directly.

    The API will use this at startup to load all four dataset predictors
    into memory once and reuse them per request.

    Example
    -------
        from predictor import load_predictor
        predictor = load_predictor("FD001")
        result    = predictor.predict(engine_df)
    """
    return RULPredictor(dataset_key, Path(registry_path))


# ── Quick smoke test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Smoke test: load predictor for FD001 and run a prediction on the first
    test engine. Verifies paths, shapes, and output format without needing
    to write a full test suite.

    Run from the project root:
        python src/inference/predictor.py
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="FD001",
                        choices=["FD001", "FD002", "FD003", "FD004"])
    args = parser.parse_args()

    registry = _ROOT / "config" / "model_registry.yaml"
    test_csv = _ROOT / "data" / "processed" / args.dataset / "test_features.csv"

    print(f"\nLoading predictor for {args.dataset}...")
    pred = load_predictor(args.dataset, registry)

    print(f"\nLoading test data from {test_csv}...")
    test_df  = pd.read_csv(test_csv)
    first_unit = test_df["unit_number"].iloc[0]
    engine_df  = test_df[test_df["unit_number"] == first_unit]

    print(f"\nRunning single-engine prediction (engine {first_unit})...")
    result = pred.predict(engine_df)
    print("\nPrediction result:")
    for k, v in result.items():
        print(f"  {k:<22}: {v}")

    print(f"\nRunning fleet prediction ({test_df['unit_number'].nunique()} engines)...")
    preds, risk = pred.predict_fleet(test_df)
    print("\nFleet risk summary:")
    for k, v in risk.items():
        print(f"  {k:<26}: {v}")