"""
tests/test_preprocessing.py

Tests for src/data/preprocessing.py.

All tests use synthetic DataFrames that look like CMAPSS data — the same
column names and value ranges — so no real data files are required.
This means the tests run in CI without needing the NASA dataset.

What we test and why:
  compute_RUL       — correctness of label generation (RUL = 0 at last cycle)
  drop_low_variance — constant sensors get dropped, variable sensors survive
  fit_clusters      — all rows assigned, cluster count respects n_clusters
  apply_clusters    — test clusters stay within labels seen during training
  fit_scalers       — sensor values in [0, 1] after scaling
  attach_test_rul   — RUL reconstruction from provided end-point labels
"""

import numpy as np
import pandas as pd
import pytest

from preprocessing import (
    SENSOR_COLS,
    apply_clusters,
    apply_scalers,
    attach_test_rul,
    compute_RUL,
    drop_low_variance,
    fit_clusters,
    fit_scalers,
)


# ─────────────────────────────────────────────────────────────────────────────
# SHARED FIXTURES
# pytest fixtures are reusable setup blocks. A test that declares a fixture
# name as a parameter receives a fresh copy of it for every test run.
# ─────────────────────────────────────────────────────────────────────────────

def _make_cmapss_df(n_engines: int = 5, cycles_per_engine: int = 50,
                    seed: int = 42) -> pd.DataFrame:
    """
    Build a synthetic CMAPSS-like DataFrame with the correct column names.
    Uses a fixed seed for reproducibility across test runs.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for unit in range(1, n_engines + 1):
        for cycle in range(1, cycles_per_engine + 1):
            row = {
                "unit_number":    unit,
                "time_in_cycles": cycle,
                "op_setting_1":   rng.uniform(0, 42),    # altitude (ft)
                "op_setting_2":   rng.uniform(0, 1),     # Mach number
                "op_setting_3":   rng.uniform(0, 100),   # throttle resolver angle
            }
            for i in range(1, 22):                        # 21 sensor channels
                row[f"sensor_{i}"] = rng.standard_normal()
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def train_df():
    return _make_cmapss_df(n_engines=5, cycles_per_engine=50)


@pytest.fixture
def test_df():
    """Shorter test set — 3 engines, 30 cycles each."""
    return _make_cmapss_df(n_engines=3, cycles_per_engine=30, seed=99)


# ─────────────────────────────────────────────────────────────────────────────
# compute_RUL
# ─────────────────────────────────────────────────────────────────────────────

def test_compute_rul_column_exists(train_df):
    """RUL column should be present in the output."""
    result = compute_RUL(train_df.copy())
    assert "RUL" in result.columns


def test_compute_rul_zero_at_last_cycle(train_df):
    """
    The last observed cycle for each engine has zero cycles remaining,
    so its RUL should be exactly 0.
    """
    result = compute_RUL(train_df.copy())
    last_rul_per_engine = result.groupby("unit_number")["RUL"].min()
    assert (last_rul_per_engine == 0).all(), \
        "RUL should equal 0 at the last cycle of each engine"


def test_compute_rul_decreases_by_one_per_cycle(train_df):
    """
    RUL should count down by exactly 1 for each passing cycle.
    If this fails, the formula max_cycle - current_cycle is wrong.
    """
    result = compute_RUL(train_df.copy())
    for unit in result["unit_number"].unique():
        unit_data = result[result["unit_number"] == unit].sort_values("time_in_cycles")
        diffs = np.diff(unit_data["RUL"].values)
        assert (diffs == -1).all(), \
            f"Engine {unit}: RUL should decrease by 1 per cycle, got diffs={diffs}"


def test_compute_rul_non_negative(train_df):
    """RUL should never be negative."""
    result = compute_RUL(train_df.copy())
    assert (result["RUL"] >= 0).all(), "RUL should always be ≥ 0"


# ─────────────────────────────────────────────────────────────────────────────
# drop_low_variance
# ─────────────────────────────────────────────────────────────────────────────

def test_drop_low_variance_removes_constant_sensor(train_df, test_df):
    """
    A sensor with constant values (std = 0) carries no degradation signal
    and should be dropped from both train and test.
    """
    df_train = train_df.copy()
    df_test  = test_df.copy()
    df_train["sensor_1"] = 99.0    # make sensor_1 constant
    df_test["sensor_1"]  = 99.0

    train_out, test_out, dropped = drop_low_variance(df_train, df_test, threshold=0.01)

    assert "sensor_1" in dropped, \
        "Constant sensor should appear in the dropped list"
    assert "sensor_1" not in train_out.columns, \
        "Constant sensor should be removed from train"
    assert "sensor_1" not in test_out.columns, \
        "Constant sensor should be removed from test"


def test_drop_low_variance_keeps_variable_sensors(train_df, test_df):
    """Sensors with real variance should survive the filter."""
    _, train_out, dropped = drop_low_variance(
        train_df.copy(), test_df.copy(), threshold=0.01
    )
    # Sensors not in dropped should still be columns
    for col in SENSOR_COLS:
        if col not in dropped:
            assert col in train_out.columns, \
                f"{col} was dropped despite having sufficient variance"


def test_drop_low_variance_threshold_respected(train_df, test_df):
    """
    Manually set one sensor to low (but non-zero) variance and confirm
    the threshold parameter controls what gets dropped.
    """
    df_train = train_df.copy()
    df_test  = test_df.copy()
    df_train["sensor_2"] = np.random.uniform(0, 0.001, size=len(df_train))
    df_test["sensor_2"]  = np.random.uniform(0, 0.001, size=len(df_test))

    _, _, dropped_strict = drop_low_variance(df_train.copy(), df_test.copy(), threshold=0.01)
    _, _, dropped_loose  = drop_low_variance(df_train.copy(), df_test.copy(), threshold=0.0001)

    # The strict threshold should drop more than the loose one
    assert len(dropped_strict) >= len(dropped_loose), \
        "A stricter threshold should drop at least as many sensors"


# ─────────────────────────────────────────────────────────────────────────────
# fit_clusters / apply_clusters
# ─────────────────────────────────────────────────────────────────────────────

def test_fit_clusters_adds_op_cluster_column(train_df, tmp_path):
    """fit_clusters should add an 'op_cluster' column to the DataFrame."""
    result, _ = fit_clusters(train_df.copy(), n_clusters=2, artifact_dir=str(tmp_path))
    assert "op_cluster" in result.columns, \
        "fit_clusters should add an op_cluster column"


def test_fit_clusters_no_missing_assignments(train_df, tmp_path):
    """Every row should be assigned to a cluster — no NaN cluster labels."""
    result, _ = fit_clusters(train_df.copy(), n_clusters=2, artifact_dir=str(tmp_path))
    assert result["op_cluster"].isna().sum() == 0, \
        "All rows should have a cluster assignment"


def test_fit_clusters_respects_n_clusters(train_df, tmp_path):
    """The number of unique clusters should not exceed n_clusters."""
    n_clusters = 3
    result, _ = fit_clusters(train_df.copy(), n_clusters=n_clusters, artifact_dir=str(tmp_path))
    assert result["op_cluster"].nunique() <= n_clusters, \
        f"Got more clusters than n_clusters={n_clusters}"


def test_apply_clusters_uses_same_labels(train_df, test_df, tmp_path):
    """
    Test cluster labels should be a subset of training cluster labels.
    A new operating condition not seen in training should still get
    assigned to the nearest known cluster — not create a new label.
    """
    train_with_cluster, kmeans = fit_clusters(
        train_df.copy(), n_clusters=2, artifact_dir=str(tmp_path)
    )
    test_with_cluster = apply_clusters(test_df.copy(), kmeans)

    train_labels = set(train_with_cluster["op_cluster"].unique())
    test_labels  = set(test_with_cluster["op_cluster"].unique())

    assert test_labels.issubset(train_labels), \
        f"Test has cluster labels not in training: {test_labels - train_labels}"


# ─────────────────────────────────────────────────────────────────────────────
# fit_scalers / apply_scalers
# ─────────────────────────────────────────────────────────────────────────────

def test_fit_scalers_output_in_unit_range(train_df, tmp_path):
    """
    After per-cluster MinMaxScaling, all sensor values should fall in [0, 1].
    A value outside [0, 1] means the scaler was fit incorrectly or the
    wrong columns were included.
    """
    train_clustered, kmeans = fit_clusters(
        train_df.copy(), n_clusters=1, artifact_dir=str(tmp_path)
    )
    active_sensors = [c for c in SENSOR_COLS if c in train_clustered.columns]

    train_scaled, _ = fit_scalers(
        train_clustered,
        n_clusters=1,
        active_sensors=active_sensors,
        artifact_dir=str(tmp_path),
    )

    for sensor in active_sensors:
        assert train_scaled[sensor].min() >= -1e-6, \
            f"{sensor}: value below 0 after scaling (min={train_scaled[sensor].min():.4f})"
        assert train_scaled[sensor].max() <= 1 + 1e-6, \
            f"{sensor}: value above 1 after scaling (max={train_scaled[sensor].max():.4f})"


def test_apply_scalers_preserves_row_count(train_df, test_df, tmp_path):
    """Scaling should not add or remove any rows."""
    train_clustered, kmeans = fit_clusters(
        train_df.copy(), n_clusters=1, artifact_dir=str(tmp_path)
    )
    test_clustered = apply_clusters(test_df.copy(), kmeans)
    active_sensors = [c for c in SENSOR_COLS if c in train_clustered.columns]

    _, scalers = fit_scalers(
        train_clustered, n_clusters=1,
        active_sensors=active_sensors, artifact_dir=str(tmp_path)
    )
    test_scaled = apply_scalers(test_clustered, scalers, active_sensors)

    assert len(test_scaled) == len(test_df), \
        "apply_scalers should not change the number of rows"


# ─────────────────────────────────────────────────────────────────────────────
# attach_test_rul
# ─────────────────────────────────────────────────────────────────────────────

def test_attach_test_rul_column_exists(test_df):
    """attach_test_rul should add a RUL column to the test DataFrame."""
    rul_df = pd.DataFrame({
        "unit_number": [1, 2, 3],
        "RUL":         [10, 25, 5],
    })
    result = attach_test_rul(test_df.copy(), rul_df)
    assert "RUL" in result.columns


def test_attach_test_rul_correct_at_final_cycle(test_df):
    """
    The provided RUL value is the RUL at the LAST observed cycle.
    At that cycle, the reconstructed RUL should equal the provided value.
    """
    provided_rul = {"1": 10, "2": 25, "3": 5}
    rul_df = pd.DataFrame({
        "unit_number": [1, 2, 3],
        "RUL":         [10, 25, 5],
    })
    result = attach_test_rul(test_df.copy(), rul_df)

    for unit_id, expected_rul in provided_rul.items():
        unit_data = result[result["unit_number"] == int(unit_id)]
        last_row  = unit_data.sort_values("time_in_cycles").iloc[-1]
        assert last_row["RUL"] == expected_rul, \
            f"Engine {unit_id}: expected RUL={expected_rul} at last cycle, got {last_row['RUL']}"


def test_attach_test_rul_non_negative(test_df):
    """Reconstructed RUL should never be negative at any cycle."""
    rul_df = pd.DataFrame({
        "unit_number": [1, 2, 3],
        "RUL":         [10, 25, 5],
    })
    result = attach_test_rul(test_df.copy(), rul_df)
    assert (result["RUL"] >= 0).all(), \
        "Reconstructed RUL should never be negative"