import pandas as pd
import numpy as np

def feature_engineering(train_df, test_df, cfg: dict):

    """
    Constructs rolling statistics and lag variables. 
    Assumes operational clusters are already populated downstream from preprocessing.
    """

    WINDOW_SIZE = cfg.get("window_size", 30)
    max_rul = cfg["max_rul"]
    # Dynamically find sensors that survived the low-variance filter step
    SENSOR_COLS = [col for col in train_df.columns if col.startswith("sensor_")]

    for df in [train_df, test_df]:
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
    
    train_df["RUL"] = train_df["RUL"].clip(upper=max_rul)
    test_df["RUL"] = test_df["RUL"].clip(upper=max_rul)
    
    return train_df, test_df