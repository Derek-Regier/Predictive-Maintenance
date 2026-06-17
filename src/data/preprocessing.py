import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

COLUMNS = [
    "unit_number",
    "time_in_cycles",
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
    "sensor_1",
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_5",
    "sensor_6",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_10",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_16",
    "sensor_17",
    "sensor_18",
    "sensor_19",
    "sensor_20",
    "sensor_21"
]

SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]

FEATURE_COLS = SENSOR_COLS + [
    "op_setting_1", "op_setting_2", "op_setting_3"
]

def load_data(filename):
    df = pd.read_csv(
        filename,
        sep=r"\s+",
        header=None
    )
    df = df.dropna(axis=1, how="all")
    df.columns = COLUMNS
    return df

def compute_RUL(df):
        # 1. Find the max cycle for each unit and map it back to every row
    max_cycles = df.groupby('unit_number')['time_in_cycles'].transform('max')

    # 2. Subtract current cycle from max cycle to get Remaining Useful Life (RUL)
    df['RUL'] = max_cycles - df['time_in_cycles']

    return df

def load_rul(filename):
    rul_df = pd.read_csv(filename, sep=r"\s+", header=None)
    rul_df.columns = ["RUL"]
    # unit_number is just its row position (1-indexed)
    rul_df["unit_number"] = rul_df.index + 1
    return rul_df

def attach_test_rul(test_df, rul_df):
    # Get the last observed row per engine
    last_cycles = test_df.groupby("unit_number")["time_in_cycles"].max().reset_index()
    last_cycles.columns = ["unit_number", "max_cycle"]

    # Merge RUL values onto those last rows
    last_cycles = last_cycles.merge(rul_df, on="unit_number")

    # Now reconstruct full RUL for every row in the test set
    # RUL at any row = (true RUL at last cycle) + (cycles remaining to last row)
    test_df = test_df.merge(last_cycles, on="unit_number")
    test_df["RUL"] = test_df["RUL"] + (test_df["max_cycle"] - test_df["time_in_cycles"])
    test_df = test_df.drop(columns=["max_cycle"])

    return test_df

def drop_low_variance(train_df, test_df, threshold=0.01):
    stds = train_df[SENSOR_COLS].std()
    cols_to_drop = stds[stds < threshold].index.tolist()
    
    print(f"Dropping {len(cols_to_drop)} low-variance sensors: {cols_to_drop}")
    
    train_df = train_df.drop(columns=cols_to_drop)
    test_df = test_df.drop(columns=cols_to_drop)
    
    return train_df, test_df

def normalize(train_df, test_df, scaler_path="models/scaler.pkl"):
    # Only scale the columns that actually exist after dropping low-variance ones
    cols_to_scale = [col for col in FEATURE_COLS if col in train_df.columns]

    scaler = MinMaxScaler()

    # Fit on training data, then transform both
    train_df[cols_to_scale] = scaler.fit_transform(train_df[cols_to_scale])
    test_df[cols_to_scale] = scaler.transform(test_df[cols_to_scale])

    # Save scaler so evaluate.py can load it later without refitting
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")

    return train_df, test_df

def process_data(test_file, train_file, RUL_file, train_dir, test_dir):
    train_df = load_data(train_file)
    test_df = load_data(test_file)
    RUL_df = load_rul(RUL_file)

    train_df = compute_RUL(train_df)
    test_df = attach_test_rul(test_df,RUL_df)

    train_df, test_df = drop_low_variance(train_df,test_df)
    print(train_df.head())
    print(test_df.head())

    train_df, test_df = normalize(train_df=train_df, test_df=test_df)

    train_df.to_csv(train_dir, index=False)
    test_df.to_csv(test_dir, index=False)
