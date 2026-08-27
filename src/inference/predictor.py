"""
The single inference entry point for the trained backbone + NGBoost pipeline.

Used by:
  - The FastAPI backend  (/predict, /fleet/summary endpoints)
  - The Streamlit dashboard (direct calls, or via the API)
  - The smoke test at the bottom of this file

How it works
------------
At construction, RULPredictor loads all trained artifacts from disk once:
the backbone neural network, the NGBoost meta-model, the sigma calibration
scale, and the per-dataset alert thresholds. Every subsequent call to
predict() or predict_fleet() reuses those in-memory objects — no disk reads
at inference time.

The full pipeline per prediction:
  1. Take the engine's recent sensor history (preprocessed DataFrame)
  2. Build a sliding window of the last seq_length cycles
  3. Pass through backbone.encode() -> compact latent feature vector
  4. Pass feature vector through NGBoost -> N(mu, sigma) distribution
  5. Multiply sigma by calibration_scale -> corrected uncertainty
  6. Compute P(RUL < 20) and P(RUL < 50) from the corrected distribution
  7. Assign alert tier based on per-dataset thresholds from datasets.yaml

Short engine history
--------------------
If an engine has fewer recorded cycles than seq_length (e.g. a newly
commissioned engine), the window is front-padded by repeating the earliest
available row. The prediction dict includes a `padded: True` flag so
callers know the result is less reliable.

Example
-------
    predictor = RULPredictor("FD001", "config/model_registry.yaml")
    engine_df = pd.read_csv("data/processed/FD001/test_features.csv")
    engine_df = engine_df[engine_df["unit_number"] == 1]

    result = predictor.predict(engine_df)
    # {
    #   "rul_mean": 42.3,       predicted mean remaining cycles
    #   "rul_std": 8.1,         calibrated uncertainty (1 standard deviation)
    #   "alert_tier": "WARNING",
    #   "prob_failure_20": 0.043,
    #   ...
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

# Path resolution
# predictor.py is at src/inference/predictor.py
#   parents[0] = src/inference/
#   parents[1] = src/
#   parents[2] = project root
_ROOT = Path(__file__).resolve().parents[2]
_TRAINING = _ROOT / "src" / "training"
if str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))

from shared import DEVICE, NON_FEATURE_COLS, TARGET_COL, build_backbone

# Fallback alert thresholds when datasets.yaml has no entry for a dataset.
# These are the standard values used by FD001, FD002, and FD003.
# FD004 uses lower values (set in datasets.yaml) because the Transformer
# backbone produces wider sigma estimates and the 0.90 CRITICAL threshold
# would never fire at that scale.
_DEFAULT_THRESHOLDS = {
    "critical_prob_20": 0.90,   # P(RUL < 20 cycles) ≥ 0.90 → CRITICAL
    "warning_prob_50":  0.75,   # P(RUL < 50 cycles) ≥ 0.75 → WARNING
    "monitor_prob_50":  0.50,   # P(RUL < 50 cycles) ≥ 0.50 → MONITOR
}


class RULPredictor:
    """
    Loads and wraps the full trained pipeline for one dataset.

    Parameters
    ----------
    dataset_key: which dataset's model to load — "FD001" through "FD004"
    registry_path: path to config/model_registry.yaml
    datasets_config: path to config/datasets.yaml (for alert thresholds).
                      Defaults to config/datasets.yaml in the project root.
    """

    def __init__(
        self,
        dataset_key: str,
        registry_path: str | Path,
        datasets_config: str | Path | None = None,
    ) -> None:
        self.dataset_key = dataset_key
        self._load_artifacts(dataset_key, Path(registry_path))
        self._load_thresholds(
            dataset_key,
            Path(datasets_config) if datasets_config
            else _ROOT / "config" / "datasets.yaml",
        )

    # Initialisation helpers 
    def _load_artifacts(self, dataset_key: str, registry_path: Path) -> None:
        """
        Read model_registry.yaml, rebuild the backbone architecture, load
        trained weights into it, and load the NGBoost model.

        Also reads calibration_scale — the sigma multiplier computed by
        train.py --calibrate_only. A value of 1.37 means NGBoost's raw sigma
        is multiplied by 1.37 at inference, widening prediction intervals
        until they achieve their stated coverage. Defaults to 1.0 (no change)
        if calibration hasn't been run yet.
        """
        with open(registry_path, "r") as f:
            registry = yaml.safe_load(f) or {}

        if dataset_key not in registry or "champion" not in registry[dataset_key]:
            raise ValueError(
                f"No champion config for {dataset_key} in {registry_path}.\n"
                "Run tune.py then train.py first.")

        champion = registry[dataset_key]["champion"]
        bb_cfg = champion["backbone_config"]

        # seq_length: how many consecutive cycles form one input window.
        # Must match exactly what the backbone was trained with.
        self.seq_length = bb_cfg["seq_length"]

        # sigma_scale: the post-hoc calibration multiplier from train.py.
        # Applied at inference to correct over- or under-confident uncertainty.
        self.sigma_scale = float(champion.get("calibration_scale", 1.0))
        if "calibration_scale" not in champion:
            print(f"  Warning [{dataset_key}]: calibration_scale not found. "
                  f"Run: python src/training/train.py --dataset {dataset_key} --calibrate_only")

        bb_name = champion["backbone"]    # architecture string, e.g. "gru"
        input_dim = bb_cfg["input_dim"]     # number of sensor feature columns

        # bb_kwargs: the backbone constructor arguments (e.g. hidden_dim, dropout,
        # num_layers). We strip out input_dim and seq_length because those are
        # passed separately to build_backbone(), not as keyword arguments.
        bb_kwargs = {k: v for k, v in bb_cfg.items()
                     if k not in ("input_dim", "seq_length")}

        # Rebuild the architecture and load the trained weights
        self.backbone = build_backbone(bb_name, input_dim, bb_kwargs)
        self.backbone.load_state_dict(torch.load(Path(champion["artifacts"]["backbone"]),map_location=DEVICE,weights_only=True,))
        self.backbone.eval()   # switch to inference mode (disables dropout, etc.)

        # Load the fitted NGBoost meta-model
        self.ngb = joblib.load(Path(champion["artifacts"]["meta_ngboost"]))

        print(f"RULPredictor ready: {dataset_key} | "
              f"backbone={bb_name} | sigma_scale={self.sigma_scale:.4f}")

    def _load_thresholds(self, dataset_key: str, datasets_path: Path) -> None:
        """
        Read alert_thresholds from config/datasets.yaml for this dataset.

        Each dataset can have its own thresholds because the backbone
        architectures differ in how wide their uncertainty estimates are.
        Falls back to _DEFAULT_THRESHOLDS if the file doesn't exist or
        this dataset has no threshold entry.
        """
        if not datasets_path.exists():
            print(f"  Warning: datasets.yaml not found. Using default alert thresholds.")
            self.thresholds = _DEFAULT_THRESHOLDS.copy()
            return

        with open(datasets_path, "r") as f:
            cfg = yaml.safe_load(f) or {}

        self.thresholds = cfg.get(dataset_key, {}).get(
            "alert_thresholds", _DEFAULT_THRESHOLDS.copy()
        )

        print(f"Alert thresholds loaded: "
              f"CRITICAL >= {self.thresholds['critical_prob_20']}  "
              f"WARNING >= {self.thresholds['warning_prob_50']}  "
              f"MONITOR >= {self.thresholds['monitor_prob_50']}")

    # Internal inference helpers

    def _build_window(self, engine_df: pd.DataFrame) -> tuple[torch.Tensor, bool]:
        """
        Build the input tensor for one engine from its sensor history.

        Takes the last seq_length rows (most recent cycles). If the engine
        has fewer rows than seq_length, pads the beginning by repeating the
        earliest row — the same approach used in evaluate.py.

        Returns
        -------
        tensor  : shape (1, seq_length, n_features) — ready to pass to backbone
        padded  : True if the engine history was shorter than seq_length
        """
        # Drop metadata columns (unit_number, time_in_cycles, RUL) — the model
        # only sees sensor features and engineered features, not identifiers
        cols_to_drop = [c for c in NON_FEATURE_COLS if c in engine_df.columns]
        X = (engine_df
             .sort_values("time_in_cycles")
             .drop(columns=cols_to_drop)
             .values)   # shape: (n_rows, n_features)

        padded = False
        if len(X) < self.seq_length:
            # Pad the beginning with the earliest observed row
            n_pad = self.seq_length - len(X)
            padded_rows = np.repeat(X[:1], n_pad, axis=0)
            X = np.vstack([padded_rows, X])
            padded = True

        # Take only the last seq_length rows (the most recent window)
        window = X[-self.seq_length:]

        # Add a batch dimension: (1, seq_length, n_features)
        return torch.tensor(window[np.newaxis], dtype=torch.float32), padded

    def _run_inference(self, X_tensor: torch.Tensor) -> tuple[float, float, float, float]:
        """
        Run the full backbone → NGBoost → calibration pipeline.

        Steps:
          1. backbone.encode() -> latent feature vector
          2. ngb.pred_dist() -> Normal(mu, sigma_raw) per prediction
          3. sigma_calibrated = sigma_raw * sigma_scale
          4. P(RUL < 20) and P(RUL < 50) from the calibrated distribution

        Applying sigma_scale here means every probability and interval
        produced by this class is automatically calibrated — callers never
        need to apply the scale themselves.

        Returns all values as plain Python floats (not numpy scalars).
        """
        with torch.no_grad():
            # backbone.encode() returns the latent representation of the sequence
            features = self.backbone.encode(X_tensor.to(DEVICE)).cpu().numpy()

        # NGBoost returns a distribution object, not just a point prediction
        distribution = self.ngb.pred_dist(features)
        mu = float(distribution.loc[0])     # predicted mean RUL
        sigma_raw = float(distribution.scale[0])   # raw (uncalibrated) std dev
        sigma = sigma_raw * self.sigma_scale   # calibrated std dev

        # P(RUL < N) = probability that the true remaining life is less than N cycles
        # norm.cdf(N, loc=mu, scale=sigma) gives this directly from the Normal distribution
        prob_20 = float(norm.cdf(20, loc=mu, scale=sigma))
        prob_50 = float(norm.cdf(50, loc=mu, scale=sigma))

        return mu, sigma, prob_20, prob_50

    def _assign_alert_tier(self, prob_20: float, prob_50: float) -> str:
        """
        Assign a maintenance alert tier from the failure probabilities.

        Thresholds are checked from most severe to least severe, so if
        an engine meets the CRITICAL condition it gets CRITICAL, not WARNING.

        CRITICAL: imminent failure, immediate action required
        WARNING: plan maintenance within the next 50 cycles
        MONITOR: flag for the next scheduled inspection
        NOMINAL: no action needed
        """
        if prob_20 >= self.thresholds["critical_prob_20"]:
            return "CRITICAL"
        if prob_50 >= self.thresholds["warning_prob_50"]:
            return "WARNING"
        if prob_50 >= self.thresholds["monitor_prob_50"]:
            return "MONITOR"
        return "NOMINAL"

    # Public API 

    def predict(self, engine_df: pd.DataFrame) -> dict:
        """
        Full probabilistic prediction for one engine.

        Parameters
        ----------
        engine_df : preprocessed, feature-engineered DataFrame for one engine.
                    Must contain at least one row; short engines are padded.

        Returns
        -------
        dict with the following keys:

        rul_mean        predicted mean RUL in cycles (mu)
        rul_std         calibrated uncertainty — 1 standard deviation (sigma)
        rul_std_raw     NGBoost's raw sigma before calibration scaling
        lower_90        lower bound of the 90% prediction interval
        upper_90        upper bound of the 90% prediction interval
        prob_failure_20 P(RUL < 20 cycles) — primary CRITICAL alert driver
        prob_failure_50 P(RUL < 50 cycles) — primary WARNING alert driver
        alert_tier      CRITICAL / WARNING / MONITOR / NOMINAL
        dataset         which dataset's model produced this prediction
        sigma_scale     the calibration multiplier that was applied to sigma
        padded          True if the engine history was shorter than seq_length
        """
        X_tensor, padded = self._build_window(engine_df)
        mu, sigma, prob_20, prob_50 = self._run_inference(X_tensor)

        # z90: the z-score that puts 95% of probability on each side of a
        # two-sided interval, giving a total of 90% inside → norm.ppf(0.95) ≈ 1.645
        z90 = norm.ppf(0.95)

        return {
            "dataset": self.dataset_key,
            "rul_mean": round(mu, 2),
            "rul_std": round(sigma, 2),
            "rul_std_raw": round(sigma / self.sigma_scale, 2),
            "lower_90": round(mu - z90 * sigma, 2),
            "upper_90": round(mu + z90 * sigma, 2),
            "prob_failure_20": round(prob_20, 4),
            "prob_failure_50": round(prob_50, 4),
            "alert_tier": self._assign_alert_tier(prob_20, prob_50),
            "sigma_scale": self.sigma_scale,
            "padded": padded,
        }

    def predict_fleet(self,fleet_df: pd.DataFrame,) -> tuple[dict[int, dict], dict]:
        """
        Predict RUL for every engine in a fleet DataFrame.

        Parameters
        ----------
        fleet_df : preprocessed, feature-engineered DataFrame with multiple
                   engines. Must include a `unit_number` column.

        Returns
        -------
        predictions : dict mapping engine id (int) -> prediction dict
                      Same structure as predict() return value.

        fleet_risk  : dict with fleet-level aggregations:
            n_engines             total number of engines
            n_padded              engines with short history (padded)
            expected_failures_20  Sum P(RUL < 20) across all engines
            expected_failures_50  Sum P(RUL < 50) across all engines
            n_critical / n_warning / n_monitor / n_nominal   alert counts
            pct_at_risk           fraction of engines in WARNING or CRITICAL

        Why sum probabilities instead of counting alerts?
        -------------------------------------------------
        SUM P(RUL < N) is the actuarially expected number of failures.If
        engine A has P=0.3 and engine B has P=0.7, we expect 1.0 failure
        even though neither individually triggers a CRITICAL alert. This is
        a more informative fleet health signal than binary counts alone.
        """
        predictions: dict[int, dict] = {}
        for unit in fleet_df["unit_number"].unique():
            engine_df = fleet_df[fleet_df["unit_number"] == unit]
            predictions[int(unit)] = self.predict(engine_df)

        # Aggregate across all engines for the fleet summary
        all_prob_20 = np.array([v["prob_failure_20"] for v in predictions.values()])
        all_prob_50 = np.array([v["prob_failure_50"] for v in predictions.values()])
        all_tiers = [v["alert_tier"] for v in predictions.values()]
        n_padded = sum(1 for v in predictions.values() if v["padded"])

        fleet_risk = {
            "n_engines": len(predictions),
            "n_padded": n_padded,
            "expected_failures_20": float(all_prob_20.sum()),
            "expected_failures_50": float(all_prob_50.sum()),
            "n_critical": all_tiers.count("CRITICAL"),
            "n_warning": all_tiers.count("WARNING"),
            "n_monitor": all_tiers.count("MONITOR"),
            "n_nominal": all_tiers.count("NOMINAL"),
            "pct_at_risk": float((all_tiers.count("CRITICAL") + all_tiers.count("WARNING")) / len(predictions))
            }
        return predictions, fleet_risk


# Convenience function 

def load_predictor(
    dataset_key: str,
    registry_path: str | Path = "config/model_registry.yaml",
    datasets_config: str | Path | None = None,
) -> RULPredictor:
    """
    Load a predictor without importing the class directly.

    The FastAPI backend calls this at startup to load all four predictors
    into memory once, then reuses them for every incoming request.

    Example
    -------
        from predictor import load_predictor

        pred = load_predictor("FD001")
        result = pred.predict(engine_df)
    """
    return RULPredictor(dataset_key, Path(registry_path), datasets_config)

if __name__ == "__main__":
    """
    Quick end-to-end check: load the predictor for one dataset and run
    both a single-engine and fleet-wide prediction.

    Run from the project root:
        python src/inference/predictor.py --dataset FD001
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default="FD001",
        choices=["FD001", "FD002", "FD003", "FD004"],
    )
    args = parser.parse_args()

    registry = _ROOT / "config" / "model_registry.yaml"
    datasets_cfg = _ROOT / "config" / "datasets.yaml"
    test_csv = _ROOT / "data" / "processed" / args.dataset / "test_features.csv"

    print(f"\nLoading predictor for {args.dataset}...")
    pred = load_predictor(args.dataset, registry, datasets_cfg)

    print(f"\nLoading test data from {test_csv}...")
    test_df = pd.read_csv(test_csv)
    first_unit = test_df["unit_number"].iloc[0]
    engine_df = test_df[test_df["unit_number"] == first_unit]

    print(f"\nSingle-engine prediction (engine {first_unit}, {len(engine_df)} rows)...")
    result = pred.predict(engine_df)
    print("\nResult:")
    for k, v in result.items():
        print(f"  {k:<22}: {v}")

    print(f"\nFleet prediction ({test_df['unit_number'].nunique()} engines)...")
    preds, risk = pred.predict_fleet(test_df)
    print("\nFleet risk summary:")
    for k, v in risk.items():
        print(f"  {k:<26}: {v}")