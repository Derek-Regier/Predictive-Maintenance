import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import joblib
import os
import yaml
import argparse

DATASETS = ["FD001", "FD002", "FD003", "FD004"]

COLUMNS = [
    "unit_number", "time_in_cycles", "op_setting_1", "op_setting_2", "op_setting_3",
    "sensor_1", "sensor_2", "sensor_3", "sensor_4", "sensor_5", "sensor_6",
    "sensor_7", "sensor_8", "sensor_9", "sensor_10", "sensor_11", "sensor_12",
    "sensor_13", "sensor_14", "sensor_15", "sensor_16", "sensor_17", "sensor_18",
    "sensor_19", "sensor_20", "sensor_21"
]

SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
OP_COLS = ["op_setting_1", "op_setting_2", "op_setting_3"]
FEATURE_COLS = SENSOR_COLS + OP_COLS


def load_config(config_path: str, dataset_name: str) -> dict:
    """Loads the YAML config and extracts the specific dataset sub-configuration."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        full_config = yaml.safe_load(f)

    if dataset_name not in full_config:
        raise KeyError(
            f"Dataset '{dataset_name}' not found in {config_path}. "
            f"Choose from: {list(full_config.keys())}"
        )

    cfg = full_config[dataset_name]
    cfg["columns"] = COLUMNS
    cfg["sensor_cols"]  = SENSOR_COLS   # updated after drop_low_variance
    cfg["feature_cols"] = FEATURE_COLS
    cfg["dataset_name"] = dataset_name
    cfg["artifact_dir"] = os.path.join("models", dataset_name)
    return cfg


def load_data(filename: str) -> pd.DataFrame:
    df = pd.read_csv(filename, sep=r"\s+", header=None)
    df = df.dropna(axis=1, how="all")
    df.columns = COLUMNS
    return df

def load_rul(filename: str) -> pd.DataFrame:
    rul_df = pd.read_csv(filename, sep=r"\s+", header=None)
    rul_df.columns    = ["RUL"]
    rul_df["unit_number"] = rul_df.index + 1
    return rul_df

def compute_RUL(df: pd.DataFrame) -> pd.DataFrame:
    """Train data: RUL = cycles until last observed cycle per engine."""
    max_cycles  = df.groupby("unit_number")["time_in_cycles"].transform("max")
    df["RUL"]   = max_cycles - df["time_in_cycles"]
    return df

def attach_test_rul(test_df: pd.DataFrame, rul_df: pd.DataFrame) -> pd.DataFrame:
    """
    Test data: RUL_FDxxx.txt gives the RUL at the *last* observed cycle.
    Reconstruct full RUL for every row:
        RUL_at_row = provided_RUL + (last_cycle - current_cycle)
    """
    last_cycles = (test_df.groupby("unit_number")["time_in_cycles"]
                   .max()
                   .reset_index()
                   .rename(columns={"time_in_cycles": "max_cycle"}))
    last_cycles = last_cycles.merge(rul_df, on="unit_number")

    test_df = test_df.merge(last_cycles, on="unit_number")
    test_df["RUL"] = test_df["RUL"] + (test_df["max_cycle"] - test_df["time_in_cycles"])
    test_df = test_df.drop(columns=["max_cycle"])
    return test_df

def drop_low_variance(
    train_df: pd.DataFrame,
    test_df:  pd.DataFrame,
    threshold: float = 0.01,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Drop sensors whose std (measured on TRAINING data) falls below threshold.
    Returns the surviving sensor list so cfg can be updated immediately.
    """
    stds = train_df[SENSOR_COLS].std()
    cols_to_drop = stds[stds < threshold].index.tolist()

    print(f" Low-variance drop  : {len(cols_to_drop)} sensors removed {cols_to_drop}")

    train_df = train_df.drop(columns=cols_to_drop)
    test_df  = test_df.drop(columns=cols_to_drop)
    return train_df, test_df, cols_to_drop

def fit_clusters(
    train_df:     pd.DataFrame,
    n_clusters:   int,
    artifact_dir: str,
) -> tuple[pd.DataFrame, KMeans]:
    """
    Fit KMeans on raw op_settings
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    train_df["op_cluster"] = kmeans.fit_predict(train_df[OP_COLS])

    os.makedirs(artifact_dir, exist_ok=True)
    joblib.dump(kmeans, os.path.join(artifact_dir, "kmeans.pkl"))
    print(f"  KMeans saved       : {artifact_dir}/kmeans.pkl  "
          f"(n_clusters={n_clusters})")
    return train_df, kmeans

def apply_clusters(test_df: pd.DataFrame, kmeans: KMeans) -> pd.DataFrame:
    test_df["op_cluster"] = kmeans.predict(test_df[OP_COLS])
    return test_df


def fit_scalers(
    train_df:      pd.DataFrame,
    n_clusters:    int,
    active_sensors: list[str],
    artifact_dir:  str,
) -> tuple[pd.DataFrame, dict]:
    """
    Fit one MinMaxScaler per operating cluster — on SENSOR columns only.

    Op_settings are excluded: within a single cluster they are nearly constant
    (the cluster IS that operating point), so per-cluster scaling would produce
    near-zero-variance columns and meaningless normalised values.
    Op_settings remain in the dataframe in their raw form; the op_cluster
    feature already encodes the condition for the model.
    """
    scalers = {}

    for cluster_id in range(n_clusters):
        mask = train_df["op_cluster"] == cluster_id
        if not mask.any():
            print(f"  Warning: cluster {cluster_id} has no training rows — skipped")
            continue

        scaler = MinMaxScaler()
        train_df.loc[mask, active_sensors] = scaler.fit_transform(
            train_df.loc[mask, active_sensors]
        )
        scalers[cluster_id] = scaler

    os.makedirs(artifact_dir, exist_ok=True)
    joblib.dump(scalers, os.path.join(artifact_dir, "scalers.pkl"))
    print(f"  Scalers saved      : {artifact_dir}/scalers.pkl  "
          f"({len(scalers)} cluster scaler(s))")
    return train_df, scalers

def apply_scalers(
    test_df:        pd.DataFrame,
    scalers:        dict,
    active_sensors: list[str],
) -> pd.DataFrame:
    """Apply the per-cluster scalers to test data. Never fits."""
    for cluster_id, scaler in scalers.items():
        mask = test_df["op_cluster"] == cluster_id
        if mask.any():
            test_df.loc[mask, active_sensors] = scaler.transform(
                test_df.loc[mask, active_sensors]
            )
    return test_df

def process_dataset(
    dataset_name: str,
    cfg: dict,
    raw_dir: str,
    processed_dir: str,
) -> None:
    """
    Full preprocessing pipeline for one CMAPSS dataset.
    """
    print(f"\n{'='*55}")
    print(f"  Processing {dataset_name}")
    print(f"{'='*55}")

    artifact_dir = cfg["artifact_dir"]

    train_df = load_data(os.path.join(raw_dir, f"train_{dataset_name}.txt"))
    test_df  = load_data(os.path.join(raw_dir, f"test_{dataset_name}.txt"))
    rul_df   = load_rul(os.path.join(raw_dir,  f"RUL_{dataset_name}.txt"))

    print(f"  Loaded train: {train_df.shape[0]:,} rows  "
          f"{train_df['unit_number'].nunique()} engines")
    print(f"  Loaded test: {test_df.shape[0]:,} rows  "
          f"{test_df['unit_number'].nunique()} engines")

    train_df = compute_RUL(train_df)
    test_df  = attach_test_rul(test_df, rul_df)
    train_df, test_df, dropped = drop_low_variance(train_df, test_df)

    # Keep cfg in sync so feature_engineering sees the correct sensor list
    active_sensors = [c for c in SENSOR_COLS if c not in dropped]
    cfg["sensor_cols"] = active_sensors

    train_df, kmeans = fit_clusters(train_df, cfg["n_clusters"], artifact_dir)
    test_df = apply_clusters(test_df, kmeans)

    train_df, scalers = fit_scalers(
        train_df, cfg["n_clusters"], active_sensors, artifact_dir
    )
    test_df = apply_scalers(test_df, scalers, active_sensors)

    out_dir = os.path.join(processed_dir, dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    train_df.to_csv(os.path.join(out_dir, "train_preprocessed.csv"), index=False)
    test_df.to_csv(os.path.join(out_dir, "test_preprocessed.csv"),  index=False)

    print(f" Saved  {out_dir}/train_preprocessed.csv  "
          f"{train_df.shape}")
    print(f" Saved : {out_dir}/test_preprocessed.csv   "
          f"{test_df.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CMAPSS preprocessing - run one dataset or all four."
    )
    parser.add_argument(
        "--dataset",
        choices=DATASETS + ["all"],
        default="all",
        help="Dataset to process (default: all)",
    )
    parser.add_argument("--raw_dir", default="data/raw")
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--config_path", default="config/datasets.yaml")
    args = parser.parse_args()

    to_run = DATASETS if args.dataset == "all" else [args.dataset]

    for ds in to_run:
        cfg = load_config(args.config_path, ds)
        process_dataset(ds, cfg, args.raw_dir, args.processed_dir)

    print("\nDone.")