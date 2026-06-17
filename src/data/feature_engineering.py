import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

WINDOW_SIZE = 5
N_CLUSTERS = 6  # standard for CMAPSS, covers the operating condition combinations

OP_COLS = ["op_setting_1", "op_setting_2", "op_setting_3"]
MAX_RUL = 125  # engines rarely exceed this in FD001


def add_operational_clusters(train_df, test_df):
    # Fit ONLY on training data, same principle as the scaler
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    train_df["op_cluster"] = kmeans.fit_predict(train_df[OP_COLS])
    test_df["op_cluster"] = kmeans.predict(test_df[OP_COLS])
    return train_df, test_df

def feature_engineering(train_df, test_df):
    SENSOR_COLS = [col for col in train_df.columns if col.startswith("sensor_")]

    # Operational clusters before the loop since it needs both dfs together
    train_df, test_df = add_operational_clusters(train_df, test_df)

    for df in [train_df, test_df]:
        max_cycles = df.groupby("unit_number")["time_in_cycles"].transform("max")
        df["log_cycles"] = np.log1p(df["time_in_cycles"])

        for sensor in SENSOR_COLS:
            grouped = df.groupby("unit_number")[sensor]

            # Rolling mean + std
            df[sensor + "_rolling_mean"] = grouped.transform(
                lambda x: x.rolling(window=WINDOW_SIZE, min_periods=1).mean())
            df[sensor + "_rolling_std"] = grouped.transform(
                lambda x: x.rolling(window=WINDOW_SIZE, min_periods=1).std())

            # Lag features
            df[sensor + "_lag1"] = grouped.transform(lambda x: x.shift(1))
            df[sensor + "_lag2"] = grouped.transform(lambda x: x.shift(2))

            # Delta/trend — how much did the sensor change since last cycle
            df[sensor + "_delta"] = df[sensor] - df[sensor + "_lag1"]

    train_df = train_df.dropna()
    test_df = test_df.dropna()
    train_df["RUL"] = train_df["RUL"].clip(upper=MAX_RUL)
    test_df["RUL"] = test_df["RUL"].clip(upper=MAX_RUL)
    return train_df, test_df