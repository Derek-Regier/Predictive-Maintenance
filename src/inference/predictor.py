"""
src/inference/predictor.py

Unified inference class for the backbone + NGBoost pipeline.

This is the single entry point for predictions used by:
  - The FastAPI backend (/predict, /fleet/summary endpoints)
  - The Streamlit dashboard (direct calls or via API)
  - The smoke test below (__main__)

Loads all trained artifacts from model_registry.yaml once at construction
and reuses them for every prediction. Sigma calibration and alert thresholds
are applied automatically — callers receive a clean, ready-to-use result dict.

Alert thresholds are read from config/datasets.yaml per dataset so each
dataset can have independently tuned thresholds (FD004 uses lower thresholds
because the Transformer backbone produces wider sigma estimates).

Short engines (fewer rows than seq_length) are padded by repeating their
earliest row rather than raising an error — same approach as evaluate.py.

Example
-------
    predictor = RULPredictor("FD001", "config/model_registry.yaml")
    engine_df = pd.read_csv("data/processed/FD001/test_features.csv")
    engine_df = engine_df[engine_df["unit_number"] == 1]

    result = predictor.predict(engine_df)
    # {"rul_mean": 42.3, "rul_std": 8.1, "alert_tier": "WARNING", ...}

    all_preds, fleet_risk = predictor.predict_fleet(engine_df)
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

# Default thresholds used when datasets.yaml has no alert_thresholds entry
_DEFAULT_THRESHOLDS = {
    "critical_prob_20": 0.90,
    "warning_prob_50":  0.75,
    "monitor_prob_50":  0.50,
}


class RULPredictor:
    """
    Loads a trained backbone + NGBoost model for one dataset and exposes
    predict() and predict_fleet() for inference.

    Parameters
    ----------
    dataset_key     : "FD001" / "FD002" / "FD003" / "FD004"
    registry_path   : path to config/model_registry.yaml
    datasets_config : path to config/datasets.yaml (for alert thresholds).
                      Defaults to config/datasets.yaml relative to project root.
    """

    def __init__(
        self,
        dataset_key:    str,
        registry_path:  str | Path,
        datasets_config: str | Path | None = None,
    ) -> None:
        self.dataset_key = dataset_key
        self._load_artifacts(dataset_key, Path(registry_path))
        self._load_thresholds(
            dataset_key,
            Path(datasets_config) if datasets_config
            else _ROOT / "config" / "datasets.yaml",
        )

    # ── Artifact loading ──────────────────────────────────────────────────────

    def _load_artifacts(self, dataset_key: str, registry_path: Path) -> None:
        """
        Rebuild the backbone architecture, load trained weights, load NGBoost,
        and read the sigma calibration scale from model_registry.yaml.

        calibration_scale is the result of train.py --calibrate_only.
        It multiplies raw NGBoost sigma so that P(RUL < N) values reflect
        corrected uncertainty. Defaults to 1.0 (no change) if missing.
        """
        with open(registry_path, "r") as f:
            registry = yaml.safe_load(f) or {}

        if dataset_key not in registry or "champion" not in registry[dataset_key]:
            raise ValueError(
                f"No champion config for {dataset_key} in {registry_path}. "
                "Run tune.py then train.py first."
            )

        champion = registry[dataset_key]["champion"]
        bb_cfg   = champion["backbone_config"]

        self.seq_length  = bb_cfg["seq_length"]
        self.sigma_scale = float(champion.get("calibration_scale", 1.0))

        if "calibration_scale" not in champion:
            print(f"  Warning [{dataset_key}]: no calibration_scale found. "
                  "Run: python src/training/train.py "
                  f"--dataset {dataset_key} --calibrate_only")

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

        self.ngb = joblib.load(Path(champion["artifacts"]["meta_ngboost"]))

        print(f"  RULPredictor ready: {dataset_key} | "
              f"backbone={bb_name} | "
              f"sigma_scale={self.sigma_scale:.4f}")

    def _load_thresholds(self, dataset_key: str, datasets_path: Path) -> None:
        """
        Read alert_thresholds from config/datasets.yaml for this dataset.

        Thresholds are per-dataset because FD004's Transformer backbone
        produces wider sigma estimates, so the standard 0.90 CRITICAL threshold
        never fires — FD004 uses 0.60 instead. All other datasets use the
        standard thresholds. Falls back to _DEFAULT_THRESHOLDS if missing.
        """
        if not datasets_path.exists():
            print(f"  Warning: datasets.yaml not found at {datasets_path}. "
                  "Using default alert thresholds.")
            self.thresholds = _DEFAULT_THRESHOLDS.copy()
            return

        with open(datasets_path, "r") as f:
            cfg = yaml.safe_load(f) or {}

        dataset_cfg      = cfg.get(dataset_key, {})
        self.thresholds  = dataset_cfg.get(
            "alert_thresholds", _DEFAULT_THRESHOLDS.copy()
        )

        print(f"  Alert thresholds: "
              f"CRITICAL P(RUL<20)≥{self.thresholds['critical_prob_20']}  "
              f"WARNING  P(RUL<50)≥{self.thresholds['warning_prob_50']}  "
              f"MONITOR  P(RUL<50)≥{self.thresholds['monitor_prob_50']}")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_window(
        self,
        engine_df: pd.DataFrame,
    ) -> tuple[torch.Tensor, bool]:
        """
        Extract the last seq_length cycles from an engine's history.

        If the engine has fewer rows than seq_length (e.g. a new engine
        just commissioned, or a test engine with short history), the window
        is padded at the front by repeating the earliest available row.
        This is the same approach used in evaluate.py.

        Returns: (tensor of shape (1, seq_length, n_features), was_padded)
        """
        cols_to_drop = [c for c in NON_FEATURE_COLS if c in engine_df.columns]
        X = (engine_df
             .sort_values("time_in_cycles")
             .drop(columns=cols_to_drop)
             .values)

        padded = False
        if len(X) < self.seq_length:
            n_pad       = self.seq_length - len(X)
            padded_rows = np.repeat(X[:1], n_pad, axis=0)
            X           = np.vstack([padded_rows, X])
            padded      = True

        window = X[-self.seq_length:]                                  # (seq, feat)
        return torch.tensor(window[np.newaxis], dtype=torch.float32), padded  # (1, seq, feat)

    def _run_inference(
        self,
        X_tensor: torch.Tensor,
    ) -> tuple[float, float, float, float]:
        """
        Full inference for a single engine window.

        Step 1  backbone.encode() → latent feature vector
        Step 2  ngb.pred_dist()   → Normal(mu, sigma_raw)
        Step 3  sigma_calibrated  = sigma_raw * sigma_scale
        Step 4  P(RUL < 20) and P(RUL < 50) from calibrated distribution

        Returns: (mu, sigma_calibrated, prob_failure_20, prob_failure_50)
        All values are plain Python floats.
        """
        with torch.no_grad():
            features = self.backbone.encode(X_tensor.to(DEVICE)).cpu().numpy()

        dist      = self.ngb.pred_dist(features)
        mu        = float(dist.loc[0])
        sigma_raw = float(dist.scale[0])
        sigma     = sigma_raw * self.sigma_scale    # apply calibration

        prob_20 = float(norm.cdf(20, loc=mu, scale=sigma))
        prob_50 = float(norm.cdf(50, loc=mu, scale=sigma))

        return mu, sigma, prob_20, prob_50

    def _assign_alert_tier(self, prob_20: float, prob_50: float) -> str:
        """
        Map failure probabilities to a maintenance alert tier using the
        per-dataset thresholds loaded from datasets.yaml.

        Applied in ascending severity order so the worst label wins.
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

        Accepts any DataFrame with at least one row of preprocessed,
        feature-engineered sensor data. Short engines are padded automatically.

        Returns dict with keys:
            dataset          dataset this model was trained on
            rul_mean         predicted mean RUL in cycles (μ)
            rul_std          calibrated uncertainty (σ × sigma_scale)
            rul_std_raw      uncalibrated NGBoost sigma (before scaling)
            lower_90         lower bound of 90% prediction interval
            upper_90         upper bound of 90% prediction interval
            prob_failure_20  P(RUL < 20) — drives CRITICAL alert
            prob_failure_50  P(RUL < 50) — drives WARNING alert
            alert_tier       CRITICAL / WARNING / MONITOR / NOMINAL
            sigma_scale      calibration multiplier applied
            padded           True if engine had fewer rows than seq_length
        """
        X_tensor, padded         = self._build_window(engine_df)
        mu, sigma, prob_20, prob_50 = self._run_inference(X_tensor)

        z90 = norm.ppf(0.95)    # z-score for 90% two-sided interval

        return {
            "dataset":         self.dataset_key,
            "rul_mean":        round(mu, 2),
            "rul_std":         round(sigma, 2),
            "rul_std_raw":     round(sigma / self.sigma_scale, 2),
            "lower_90":        round(mu - z90 * sigma, 2),
            "upper_90":        round(mu + z90 * sigma, 2),
            "prob_failure_20": round(prob_20, 4),
            "prob_failure_50": round(prob_50, 4),
            "alert_tier":      self._assign_alert_tier(prob_20, prob_50),
            "sigma_scale":     self.sigma_scale,
            "padded":          padded,
        }

    def predict_fleet(
        self,
        fleet_df: pd.DataFrame,
    ) -> tuple[dict[int, dict], dict]:
        """
        Predict RUL for every engine in fleet_df.

        Parameters
        ----------
        fleet_df : preprocessed + feature-engineered DataFrame containing
                   multiple engines (unit_number column required).

        Returns
        -------
        predictions : dict mapping unit_number (int) → prediction dict
        fleet_risk  : aggregated fleet summary:
            n_engines             total engines with valid predictions
            expected_failures_20  Σ P(RUL < 20) — actuarial failure count
            expected_failures_50  Σ P(RUL < 50)
            n_critical / warning / monitor / nominal   alert tier counts
            pct_at_risk           fraction with WARNING or CRITICAL

        Why Σ P(RUL < N)?
        -----------------
        Rather than counting hard binary outcomes, we sum probabilities across
        the fleet. If engine A has P=0.3 and engine B has P=0.7, the expected
        number of failures is 1.0. This is the actuarial fleet health number
        shown in Power BI and is meaningful even when no individual engine
        crosses the CRITICAL threshold (as with FD004).
        """
        predictions: dict[int, dict] = {}

        for unit in fleet_df["unit_number"].unique():
            engine_df = fleet_df[fleet_df["unit_number"] == unit]
            predictions[int(unit)] = self.predict(engine_df)

        # Fleet-level aggregation
        valid    = predictions   # padding means all engines always have results now
        prob_20s = np.array([v["prob_failure_20"] for v in valid.values()])
        prob_50s = np.array([v["prob_failure_50"] for v in valid.values()])
        tiers    = [v["alert_tier"] for v in valid.values()]
        n_padded = sum(1 for v in valid.values() if v["padded"])

        fleet_risk = {
            "n_engines":            len(valid),
            "n_padded":             n_padded,
            "expected_failures_20": float(prob_20s.sum()),
            "expected_failures_50": float(prob_50s.sum()),
            "n_critical":  tiers.count("CRITICAL"),
            "n_warning":   tiers.count("WARNING"),
            "n_monitor":   tiers.count("MONITOR"),
            "n_nominal":   tiers.count("NOMINAL"),
            "pct_at_risk": float(
                (tiers.count("CRITICAL") + tiers.count("WARNING")) / len(valid)
            ),
        }

        return predictions, fleet_risk


# ── Convenience loader ────────────────────────────────────────────────────────

def load_predictor(
    dataset_key:   str,
    registry_path: str | Path = "config/model_registry.yaml",
    datasets_config: str | Path | None = None,
) -> RULPredictor:
    """
    Load a predictor from anywhere in the codebase without importing the class.
    The API uses this at startup to load all four predictors into memory once.

    Example
    -------
        from predictor import load_predictor
        pred   = load_predictor("FD001")
        result = pred.predict(engine_df)
    """
    return RULPredictor(
        dataset_key,
        Path(registry_path),
        datasets_config,
    )


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Smoke test: load predictor and run single + fleet predictions on the
    test set. Verifies the full inference chain works end-to-end.

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

    registry     = _ROOT / "config" / "model_registry.yaml"
    datasets_cfg = _ROOT / "config" / "datasets.yaml"
    test_csv     = _ROOT / "data" / "processed" / args.dataset / "test_features.csv"

    print(f"\nLoading predictor for {args.dataset}...")
    pred = load_predictor(args.dataset, registry, datasets_cfg)

    print(f"\nLoading test data from {test_csv}...")
    test_df    = pd.read_csv(test_csv)
    first_unit = test_df["unit_number"].iloc[0]
    engine_df  = test_df[test_df["unit_number"] == first_unit]

    print(f"\nRunning single-engine prediction (engine {first_unit}, "
          f"{len(engine_df)} rows)...")
    result = pred.predict(engine_df)
    print("\nPrediction result:")
    for k, v in result.items():
        print(f"  {k:<22}: {v}")

    print(f"\nRunning fleet prediction "
          f"({test_df['unit_number'].nunique()} engines)...")
    preds, risk = pred.predict_fleet(test_df)
    print("\nFleet risk summary:")
    for k, v in risk.items():
        print(f"  {k:<26}: {v}")