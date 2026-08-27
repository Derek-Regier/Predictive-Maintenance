"""
tests/test_predictor.py

Tests for src/inference/predictor.py.

predictor.py ties together the backbone neural network, NGBoost, sigma
calibration, and alert tier logic into one inference class. Testing it
requires mocking the trained model files since those are gitignored and
not available in CI.

Strategy
--------
We use unittest.mock to intercept the two file-loading calls:
  - build_backbone()  →  returns a MockBackbone (tiny neural net stub)
  - joblib.load()     →  returns a MockNGBoost (returns fixed distributions)
  - torch.load()      →  returns an empty state dict

This lets us test the class's inference logic — window building, sigma
scaling, probability computation, alert tier assignment, fleet aggregation —
without needing any model weights on disk.

What we test and why:
  _build_window      — short engines get padded, not dropped
  _run_inference     — sigma_scale is applied correctly
  _assign_alert_tier — tier boundaries work, worst tier wins
  predict()          — output dict has all required keys in expected ranges
  predict_fleet()    — aggregation correct, tier counts sum to n_engines
"""

import numpy as np
import pandas as pd
import pytest
import torch
from pathlib import Path
from unittest.mock import MagicMock, patch

from predictor import RULPredictor, _DEFAULT_THRESHOLDS


# ─────────────────────────────────────────────────────────────────────────────
# MOCK MODELS
# These stubs reproduce the interface that RULPredictor expects without
# loading any real weights or doing any real computation.
# ─────────────────────────────────────────────────────────────────────────────

class MockBackbone:
    """
    Minimal backbone stub. encode() returns a tensor of zeros with the
    right batch dimension — enough for downstream NGBoost to consume.
    """
    def __init__(self, feature_dim: int = 16):
        self.feature_dim = feature_dim

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        return torch.zeros(batch_size, self.feature_dim)

    def eval(self):
        return self                         # method chaining like PyTorch

    def load_state_dict(self, state_dict):
        pass                               # no-op: nothing to load

    def to(self, device):
        return self                         # already "on device"


class MockNGBoost:
    """
    Minimal NGBoost stub. pred_dist() returns a mock distribution object
    with .loc (mean) and .scale (std) set to fixed values — mu=45, sigma=10.
    This lets us verify that probability and interval calculations are correct
    for known inputs.
    """
    def pred_dist(self, X: np.ndarray):
        n = len(X)
        dist       = MagicMock()
        dist.loc   = np.full(n, 45.0)    # predicted mean RUL = 45 cycles
        dist.scale = np.full(n, 10.0)   # predicted std = 10 cycles
        return dist

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), 45.0)


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

def _make_engine_df(n_rows: int = 60, n_features: int = 8,
                    unit: int = 1, seed: int = 42) -> pd.DataFrame:
    """
    Synthetic engine history DataFrame.
    Includes unit_number, time_in_cycles, RUL (metadata) and sensor features.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for cycle in range(1, n_rows + 1):
        row = {
            "unit_number":    unit,
            "time_in_cycles": cycle,
            "RUL":            n_rows - cycle,
        }
        for i in range(1, n_features + 1):
            row[f"sensor_{i}"] = rng.uniform(0, 1)
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def mock_predictor(tmp_path):
    """
    A fully mocked RULPredictor for FD001.

    Uses tmp_path (a pytest-provided temporary directory) to write minimal
    YAML configs so the predictor's __init__ has real files to read.
    The backbone and NGBoost are replaced with mock objects so no .pt or
    .pkl files need to exist.
    """
    # Minimal model_registry.yaml
    registry_content = """
FD001:
  champion:
    backbone: gru
    backbone_config:
      hidden_dim: 64
      num_layers: 1
      dropout: 0.2
      bidirectional: false
      input_dim: 8
      seq_length: 30
    calibration_scale: 1.25
    artifacts:
      backbone: fake_backbone.pt
      meta_ngboost: fake_ngboost.pkl
"""
    (tmp_path / "model_registry.yaml").write_text(registry_content)

    # Minimal datasets.yaml with standard alert thresholds
    datasets_content = """
FD001:
  n_clusters: 1
  max_rul: 125
  alert_thresholds:
    critical_prob_20: 0.90
    warning_prob_50: 0.75
    monitor_prob_50: 0.50
"""
    (tmp_path / "datasets.yaml").write_text(datasets_content)

    # Patch all three file-loading calls so no real files are needed
    with patch("predictor.build_backbone", return_value=MockBackbone(feature_dim=8)):
        with patch("torch.load", return_value={}):
            with patch("joblib.load", return_value=MockNGBoost()):
                predictor = RULPredictor(
                    "FD001",
                    tmp_path / "model_registry.yaml",
                    tmp_path / "datasets.yaml",
                )
    return predictor


# ─────────────────────────────────────────────────────────────────────────────
# predict() — output structure
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_KEYS = {
    "dataset", "rul_mean", "rul_std", "rul_std_raw",
    "lower_90", "upper_90",
    "prob_failure_20", "prob_failure_50",
    "alert_tier", "sigma_scale", "padded",
}

def test_predict_returns_all_required_keys(mock_predictor):
    """predict() must return every key the dashboard and API depend on."""
    result = mock_predictor.predict(_make_engine_df())
    missing = REQUIRED_KEYS - set(result.keys())
    assert not missing, f"Missing keys in predict() output: {missing}"


def test_predict_no_extra_keys(mock_predictor):
    """Unexpected extra keys signal a breaking interface change."""
    result = mock_predictor.predict(_make_engine_df())
    extra = set(result.keys()) - REQUIRED_KEYS
    assert not extra, f"Unexpected extra keys in predict() output: {extra}"


# ─────────────────────────────────────────────────────────────────────────────
# predict() — value ranges
# ─────────────────────────────────────────────────────────────────────────────

def test_predict_probabilities_in_zero_one(mock_predictor):
    """Failure probabilities must be valid probabilities — in [0, 1]."""
    result = mock_predictor.predict(_make_engine_df())
    assert 0.0 <= result["prob_failure_20"] <= 1.0, \
        f"prob_failure_20={result['prob_failure_20']} out of [0,1]"
    assert 0.0 <= result["prob_failure_50"] <= 1.0, \
        f"prob_failure_50={result['prob_failure_50']} out of [0,1]"


def test_predict_prob50_geq_prob20(mock_predictor):
    """
    P(RUL < 50) ≥ P(RUL < 20) because 50 > 20 — a wider window
    can only increase or maintain the probability of failure.
    """
    result = mock_predictor.predict(_make_engine_df())
    assert result["prob_failure_50"] >= result["prob_failure_20"], \
        "P(RUL<50) must be ≥ P(RUL<20)"


def test_predict_interval_correctly_ordered(mock_predictor):
    """The 90% PI must be: lower_90 < rul_mean < upper_90."""
    result = mock_predictor.predict(_make_engine_df())
    assert result["lower_90"] < result["rul_mean"] < result["upper_90"], \
        "Prediction interval bounds are not ordered correctly"


def test_predict_alert_tier_valid(mock_predictor):
    """Alert tier must be one of the four defined strings."""
    valid_tiers = {"CRITICAL", "WARNING", "MONITOR", "NOMINAL"}
    result = mock_predictor.predict(_make_engine_df())
    assert result["alert_tier"] in valid_tiers, \
        f"Invalid alert tier: {result['alert_tier']!r}"


def test_predict_dataset_key_correct(mock_predictor):
    """The dataset key in the output should match what the predictor was loaded for."""
    result = mock_predictor.predict(_make_engine_df())
    assert result["dataset"] == "FD001"


# ─────────────────────────────────────────────────────────────────────────────
# predict() — sigma calibration
# ─────────────────────────────────────────────────────────────────────────────

def test_sigma_scale_applied_to_std(mock_predictor):
    """
    rul_std should equal rul_std_raw × sigma_scale.
    This verifies calibration is actually being applied in _run_inference.
    """
    result = mock_predictor.predict(_make_engine_df())
    expected_std = round(result["rul_std_raw"] * result["sigma_scale"], 2)
    assert abs(result["rul_std"] - expected_std) < 0.01, \
        f"rul_std ({result['rul_std']}) ≠ rul_std_raw × sigma_scale ({expected_std})"


# ─────────────────────────────────────────────────────────────────────────────
# predict() — short engine padding
# ─────────────────────────────────────────────────────────────────────────────

def test_short_engine_gets_padded_flag(mock_predictor):
    """
    An engine with fewer rows than seq_length (30) should be padded and
    the padded flag in the output should be True.
    """
    short_df = _make_engine_df(n_rows=10)   # shorter than seq_length=30
    result   = mock_predictor.predict(short_df)
    assert result["padded"] is True, \
        "padded should be True when engine history is shorter than seq_length"


def test_full_engine_not_padded(mock_predictor):
    """An engine with enough history should not be flagged as padded."""
    long_df = _make_engine_df(n_rows=60)    # longer than seq_length=30
    result  = mock_predictor.predict(long_df)
    assert result["padded"] is False, \
        "padded should be False when engine history is longer than seq_length"


def test_short_engine_still_returns_valid_result(mock_predictor):
    """Padded engines should still produce valid (non-erroring) predictions."""
    short_df = _make_engine_df(n_rows=5)
    result   = mock_predictor.predict(short_df)
    assert set(result.keys()) == REQUIRED_KEYS
    assert 0.0 <= result["prob_failure_20"] <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# _assign_alert_tier() — tier boundary logic
# ─────────────────────────────────────────────────────────────────────────────

def test_alert_tier_critical_fires_above_threshold(mock_predictor):
    """P(RUL<20) = 0.95 should trigger CRITICAL (threshold = 0.90)."""
    assert mock_predictor._assign_alert_tier(0.95, 0.99) == "CRITICAL"


def test_alert_tier_warning_fires_above_threshold(mock_predictor):
    """P(RUL<50) = 0.80 with P(RUL<20) = 0.40 should trigger WARNING."""
    assert mock_predictor._assign_alert_tier(0.40, 0.80) == "WARNING"


def test_alert_tier_monitor_fires_above_threshold(mock_predictor):
    """P(RUL<50) = 0.60 with P(RUL<20) = 0.10 should trigger MONITOR."""
    assert mock_predictor._assign_alert_tier(0.10, 0.60) == "MONITOR"


def test_alert_tier_nominal_below_all_thresholds(mock_predictor):
    """Low probabilities should produce NOMINAL — no action needed."""
    assert mock_predictor._assign_alert_tier(0.01, 0.20) == "NOMINAL"


def test_alert_tier_critical_wins_over_warning(mock_predictor):
    """
    When both CRITICAL and WARNING conditions are met, CRITICAL should win.
    Tiers are checked from most severe to least, so the worst label wins.
    """
    # P(RUL<20) = 0.95 → CRITICAL, P(RUL<50) = 0.99 → also WARNING
    result = mock_predictor._assign_alert_tier(0.95, 0.99)
    assert result == "CRITICAL", \
        "CRITICAL should take precedence over WARNING"


def test_alert_tier_exactly_at_threshold(mock_predictor):
    """
    A value exactly at the threshold should trigger that tier.
    Off-by-one errors in >= vs > comparisons would fail this test.
    """
    # Exactly at CRITICAL threshold (0.90)
    assert mock_predictor._assign_alert_tier(0.90, 0.99) == "CRITICAL"
    # Just below CRITICAL, exactly at WARNING threshold (0.75)
    assert mock_predictor._assign_alert_tier(0.89, 0.75) == "WARNING"


# ─────────────────────────────────────────────────────────────────────────────
# predict_fleet() — aggregation
# ─────────────────────────────────────────────────────────────────────────────

def _make_fleet_df(n_engines: int = 3, n_rows: int = 60,
                   n_features: int = 8) -> pd.DataFrame:
    """Build a fleet DataFrame with multiple engines."""
    dfs = [
        _make_engine_df(n_rows=n_rows, n_features=n_features, unit=u, seed=u)
        for u in range(1, n_engines + 1)
    ]
    return pd.concat(dfs, ignore_index=True)


def test_fleet_returns_prediction_for_every_engine(mock_predictor):
    """predict_fleet() should produce one prediction dict per engine."""
    fleet_df    = _make_fleet_df(n_engines=4)
    predictions, _ = mock_predictor.predict_fleet(fleet_df)
    assert len(predictions) == 4


def test_fleet_risk_n_engines_correct(mock_predictor):
    """fleet_risk['n_engines'] should match the number of unique engines."""
    fleet_df       = _make_fleet_df(n_engines=3)
    _, fleet_risk  = mock_predictor.predict_fleet(fleet_df)
    assert fleet_risk["n_engines"] == 3


def test_fleet_risk_tier_counts_sum_to_n_engines(mock_predictor):
    """
    Every engine is in exactly one alert tier, so the four tier counts
    must add up to the total number of engines.
    """
    fleet_df      = _make_fleet_df(n_engines=5)
    _, fleet_risk = mock_predictor.predict_fleet(fleet_df)

    tier_total = (
        fleet_risk["n_critical"] + fleet_risk["n_warning"] +
        fleet_risk["n_monitor"] + fleet_risk["n_nominal"]
    )
    assert tier_total == fleet_risk["n_engines"], \
        f"Tier counts sum to {tier_total}, expected {fleet_risk['n_engines']}"


def test_fleet_risk_expected_failures_non_negative(mock_predictor):
    """Expected failure counts are sums of probabilities — never negative."""
    _, fleet_risk = mock_predictor.predict_fleet(_make_fleet_df())
    assert fleet_risk["expected_failures_20"] >= 0
    assert fleet_risk["expected_failures_50"] >= 0


def test_fleet_risk_expected_50_geq_20(mock_predictor):
    """
    Σ P(RUL<50) ≥ Σ P(RUL<20) because the 50-cycle horizon captures
    strictly more failures than the 20-cycle horizon.
    """
    _, fleet_risk = mock_predictor.predict_fleet(_make_fleet_df())
    assert fleet_risk["expected_failures_50"] >= fleet_risk["expected_failures_20"]


def test_fleet_risk_pct_at_risk_in_range(mock_predictor):
    """pct_at_risk is a fraction — must be in [0, 1]."""
    _, fleet_risk = mock_predictor.predict_fleet(_make_fleet_df())
    assert 0.0 <= fleet_risk["pct_at_risk"] <= 1.0