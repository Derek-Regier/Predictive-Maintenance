"""
tests/test_feature_engineering.py

Tests for src/data/feature_engineering.py.

Feature engineering is the step most likely to silently produce wrong data:
NaN values from lagging and rolling operations, incorrect RUL clipping,
missing feature groups. These tests catch those issues on synthetic data
so they never reach training.

All tests use a minimal synthetic DataFrame that mimics the output of
preprocessing.py — normalised sensor values, op_cluster column, RUL column.
No real CSV files are required.
"""

import numpy as np
import pandas as pd
import pytest

from feature_engineering import feature_engineering


# ─────────────────────────────────────────────────────────────────────────────
# SHARED FIXTURE
# ─────────────────────────────────────────────────────────────────────────────

def _make_preprocessed_df(n_engines: int = 4,
                           cycles_per_engine: int = 60,
                           n_sensors: int = 6,
                           seed: int = 42) -> pd.DataFrame:
    """
    Synthetic DataFrame that looks like the output of preprocessing.py.

    Values are already normalised to [0, 1] (as they would be after
    fit_scalers). Each engine has a linearly decreasing RUL.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for unit in range(1, n_engines + 1):
        for cycle in range(1, cycles_per_engine + 1):
            row = {
                "unit_number":    unit,
                "time_in_cycles": cycle,
                "op_setting_1":   0.5,                    # constant (single condition)
                "op_setting_2":   0.3,
                "op_setting_3":   0.8,
                "op_cluster":     0,
                "RUL":            cycles_per_engine - cycle,   # 0 at last cycle
            }
            for i in range(1, n_sensors + 1):
                # Slightly drifting sensors to give rolling/delta features signal
                row[f"sensor_{i}"] = rng.uniform(0, 1) + cycle * 0.001
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def preprocessed_df():
    return _make_preprocessed_df()


@pytest.fixture
def minimal_cfg():
    """Minimal config dict matching what preprocessing.py would provide."""
    return {
        "max_rul":     125,
        "window_size": 5,     # small window so tests run fast
    }


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT SHAPE AND CONTENT
# ─────────────────────────────────────────────────────────────────────────────

def test_no_nan_in_train_output(preprocessed_df, minimal_cfg):
    """
    NaN values in the training set are training bugs: they produce
    misleading gradients and silently corrupt metric calculations.
    Feature engineering must produce zero NaN values.
    """
    train_out, _ = feature_engineering(
        preprocessed_df.copy(), preprocessed_df.copy(), minimal_cfg
    )
    null_counts = train_out.isnull().sum()
    assert null_counts.sum() == 0, \
        f"NaN found in training output:\n{null_counts[null_counts > 0]}"


def test_no_nan_in_test_output(preprocessed_df, minimal_cfg):
    """Same NaN check for the test set output."""
    _, test_out = feature_engineering(
        preprocessed_df.copy(), preprocessed_df.copy(), minimal_cfg
    )
    null_counts = test_out.isnull().sum()
    assert null_counts.sum() == 0, \
        f"NaN found in test output:\n{null_counts[null_counts > 0]}"


def test_feature_count_increases(preprocessed_df, minimal_cfg):
    """
    Feature engineering exists to add features. If this test fails,
    the function returned the input unchanged — something went wrong.
    """
    n_cols_before = len(preprocessed_df.columns)
    train_out, _ = feature_engineering(
        preprocessed_df.copy(), preprocessed_df.copy(), minimal_cfg
    )
    assert len(train_out.columns) > n_cols_before, \
        "Feature engineering should add columns to the DataFrame"


def test_all_engines_present_in_output(preprocessed_df, minimal_cfg):
    """
    Engines should not silently disappear. If any engine has fewer rows
    than the window size, it should still produce at least one output row.
    """
    n_engines_in = preprocessed_df["unit_number"].nunique()
    train_out, _ = feature_engineering(
        preprocessed_df.copy(), preprocessed_df.copy(), minimal_cfg
    )
    n_engines_out = train_out["unit_number"].nunique()
    assert n_engines_out == n_engines_in, \
        f"Input had {n_engines_in} engines, output has {n_engines_out}"


# ─────────────────────────────────────────────────────────────────────────────
# RUL CLIPPING
# ─────────────────────────────────────────────────────────────────────────────

def test_rul_clipped_at_max_rul(minimal_cfg):
    """
    Early-life rows have very high RUL (e.g. 200). The model doesn't benefit
    from distinguishing RUL=200 from RUL=125 — both mean "healthy, no action
    needed." Clipping forces the model to focus on the degradation range.
    """
    df = _make_preprocessed_df(n_engines=2, cycles_per_engine=200)
    # Some rows will have RUL > 125 (max_rul)
    assert (df["RUL"] > minimal_cfg["max_rul"]).any(), \
        "Test setup issue: need some RUL values above max_rul"

    train_out, _ = feature_engineering(df.copy(), df.copy(), minimal_cfg)

    assert train_out["RUL"].max() <= minimal_cfg["max_rul"], \
        f"RUL exceeds max_rul={minimal_cfg['max_rul']} after feature engineering"


def test_rul_not_negative_after_clipping(preprocessed_df, minimal_cfg):
    """Clipping should only cap from above, never produce negative RUL."""
    train_out, _ = feature_engineering(
        preprocessed_df.copy(), preprocessed_df.copy(), minimal_cfg
    )
    assert (train_out["RUL"] >= 0).all(), \
        "RUL should remain non-negative after clipping"


# ─────────────────────────────────────────────────────────────────────────────
# EXPECTED FEATURE GROUPS
# ─────────────────────────────────────────────────────────────────────────────

def test_rolling_mean_features_present(preprocessed_df, minimal_cfg):
    """
    Rolling mean features smooth cycle-to-cycle noise. If they're missing,
    either the function name convention changed or rolling failed silently.
    """
    train_out, _ = feature_engineering(
        preprocessed_df.copy(), preprocessed_df.copy(), minimal_cfg
    )
    rolling_cols = [c for c in train_out.columns if "_rolling_mean" in c]
    assert len(rolling_cols) > 0, \
        "No rolling mean features found — check that feature_engineering adds them"


def test_rolling_std_features_present(preprocessed_df, minimal_cfg):
    """Rolling std captures sensor volatility, an independent degradation signal."""
    train_out, _ = feature_engineering(
        preprocessed_df.copy(), preprocessed_df.copy(), minimal_cfg
    )
    rolling_std_cols = [c for c in train_out.columns if "_rolling_std" in c]
    assert len(rolling_std_cols) > 0, \
        "No rolling std features found"


def test_lag_features_present(preprocessed_df, minimal_cfg):
    """Lag features give the model information about the previous cycle's readings."""
    train_out, _ = feature_engineering(
        preprocessed_df.copy(), preprocessed_df.copy(), minimal_cfg
    )
    lag_cols = [c for c in train_out.columns if "_lag" in c]
    assert len(lag_cols) > 0, \
        "No lag features found"


def test_delta_features_present(preprocessed_df, minimal_cfg):
    """Delta (cycle-on-cycle change) features capture the rate of degradation."""
    train_out, _ = feature_engineering(
        preprocessed_df.copy(), preprocessed_df.copy(), minimal_cfg
    )
    delta_cols = [c for c in train_out.columns if "_delta" in c]
    assert len(delta_cols) > 0, \
        "No delta features found"


def test_log_cycles_present(preprocessed_df, minimal_cfg):
    """
    log_cycles = log(1 + time_in_cycles) is a nonlinear cycle-progress
    feature. Its absence won't break training but is a regression from
    the expected feature set.
    """
    train_out, _ = feature_engineering(
        preprocessed_df.copy(), preprocessed_df.copy(), minimal_cfg
    )
    assert "log_cycles" in train_out.columns, \
        "log_cycles feature should be present after feature engineering"


def test_log_cycles_non_negative(preprocessed_df, minimal_cfg):
    """log(1 + cycle) is always ≥ 0 since cycle ≥ 1."""
    train_out, _ = feature_engineering(
        preprocessed_df.copy(), preprocessed_df.copy(), minimal_cfg
    )
    if "log_cycles" in train_out.columns:
        assert (train_out["log_cycles"] >= 0).all(), \
            "log_cycles should be non-negative"


# ─────────────────────────────────────────────────────────────────────────────
# COLUMN CONSISTENCY BETWEEN TRAIN AND TEST
# ─────────────────────────────────────────────────────────────────────────────

def test_train_and_test_have_same_columns(preprocessed_df, minimal_cfg):
    """
    Train and test must have identical columns for the model to be able to
    make predictions. A mismatch here would crash at inference time.
    """
    train_out, test_out = feature_engineering(
        preprocessed_df.copy(), preprocessed_df.copy(), minimal_cfg
    )
    assert set(train_out.columns) == set(test_out.columns), \
        "Train and test output should have the same columns after feature engineering"