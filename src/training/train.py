import argparse
import os

import pandas as pd
import numpy as np
import joblib

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from ngboost import NGBRegressor
from sklearn.tree import DecisionTreeRegressor

# Constants 

TARGET_COL = "RUL"
NON_FEATURE_COLS = ["unit_number", "time_in_cycles", TARGET_COL]

TRAIN_PATH = "data/processed/train_FD001_features.csv"
TEST_PATH = "data/processed/test_FD001_features.csv"
MODEL_DIR = "models"

# LSTM hyperparameters
SEQ_LENGTH = 30
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.2
BATCH_SIZE = 256
EPOCHS = 50
LR = 0.001
PATIENCE = 7 # early-stopping patience

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Data loading & splitting 

def load_feature_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)
    return train_df, test_df


def split_by_engine(df, test_size=0.2, random_state=42):
    """Split whole engines into train/val — never splits a single engine across both sets."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    groups = df["unit_number"]
    train_idx, val_idx = next(splitter.split(df, df[TARGET_COL], groups))
    return df.iloc[train_idx].copy(), df.iloc[val_idx].copy()


def prep_xy(df):
    """Drop non-feature columns and return X, y. Called after split so
    unit_number is still available for grouping during the split itself."""
    cols_to_drop = [c for c in NON_FEATURE_COLS if c in df.columns]
    X = df.drop(columns=cols_to_drop)
    y = df[TARGET_COL]
    return X, y


# Sequence utilities 

class SequenceDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: np.ndarray):
        self.X = X
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_sequences(df, seq_length=SEQ_LENGTH):
    """
    Groups data by unit_number and creates sliding windows.
    Returns: (sequences tensor, targets array)
    """
    X_seq, y_final = [], []
    cols_to_drop = [c for c in NON_FEATURE_COLS if c in df.columns]

    for unit in df["unit_number"].unique():
        unit_data = df[df["unit_number"] == unit]
        X_values  = unit_data.drop(columns=cols_to_drop).values
        y_values  = unit_data[TARGET_COL].values

        if len(X_values) >= seq_length:
            for i in range(len(X_values) - seq_length + 1):
                X_seq.append(X_values[i : i + seq_length])
                # Target is the RUL at the end of the sequence window
                y_final.append(y_values[i + seq_length - 1])

    return (
        torch.tensor(np.array(X_seq), dtype=torch.float32),
        np.array(y_final),
    )


def make_loader(df, seq_length=SEQ_LENGTH, batch_size=BATCH_SIZE, shuffle=True):
    """Build a DataLoader of (sequence, target) pairs from a feature DataFrame."""
    X_seq, y_seq = create_sequences(df, seq_length)
    ds = SequenceDataset(X_seq, y_seq)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=True)


def get_lstm_features(model: nn.Module, dataloader: DataLoader) -> np.ndarray:
    """
    Passes sequences through the LSTM and returns the final hidden state as features.
    Expects batches of (X, y) — y is ignored.
    """
    model.eval()
    all_features = []
    with torch.no_grad():
        for batch in dataloader:
            x_batch = batch[0].to(DEVICE)
            # hn shape: (num_layers, batch, hidden_dim)
            _, (hn, _) = model.lstm(x_batch)
            # Take the last layer's hidden state → (batch, hidden_dim)
            all_features.append(hn[-1].cpu().numpy())
    return np.concatenate(all_features)


# LSTM model 

class LSTMRegressor(nn.Module):
    """
    Two-layer LSTM followed by a small MLP head.
    The .lstm attribute is exposed so get_lstm_features() can tap the
    hidden states directly for the stacked model.
    """
    def __init__(self, input_dim: int, hidden_dim: int = HIDDEN_DIM,
                 num_layers: int = NUM_LAYERS, dropout: float = DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hn, _) = self.lstm(x)
        return self.head(hn[-1]).squeeze(-1)


# Tabular trainers 

def train_linear(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.01,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
    )
    model.fit(X_train, y_train)
    return model


def train_ngboost(X_train, y_train):
    model = NGBRegressor(
        Base=DecisionTreeRegressor(max_depth=4),
        n_estimators=500,
        learning_rate=0.01,
    )
    model.fit(X_train, y_train)
    return model


# LSTM trainer 

def train_lstm(train_df: pd.DataFrame, val_df: pd.DataFrame) -> LSTMRegressor:
    """
    Train the LSTM on sliding-window sequences with early stopping.
    Uses GPU automatically when available.
    """
    print(f"  Device: {DEVICE}")

    train_loader = make_loader(train_df, shuffle=True)
    val_loader = make_loader(val_df, shuffle=False)

    # Infer input_dim from the first batch so this stays in sync with
    # whatever features feature_engineering.py produces
    sample_x, _ = next(iter(train_loader))
    input_dim = sample_x.shape[2]
    print(f"  Input dim: {input_dim}  |  "
          f"Train sequences: {len(train_loader.dataset)}  |  "
          f"Val sequences: {len(val_loader.dataset)}")

    model = LSTMRegressor(input_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )
    criterion = nn.MSELoss()

    best_val_rmse = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        # Training
        model.train()
        running_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            # Clip gradients to avoid exploding gradients in deep LSTMs
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * len(x_batch)

        train_rmse = np.sqrt(running_loss / len(train_loader.dataset))

        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                val_preds.append(model(x_batch.to(DEVICE)).cpu().numpy())
                val_targets.append(y_batch.numpy())

        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)
        val_rmse = np.sqrt(mean_squared_error(val_targets, val_preds))

        scheduler.step(val_rmse)

        print(f"Epoch {epoch:>3}/{EPOCHS}  "
              f"train_RMSE={train_rmse:.4f}  val_RMSE={val_rmse:.4f}")

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            epochs_no_improve = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"  Early stopping at epoch {epoch} (patience={PATIENCE})")
                break

    model.load_state_dict(best_state)
    print(f"Best val RMSE: {best_val_rmse:.4f}")
    return model


# Stacked LSTM -> XGBoost trainer 

def train_stacked(train_df: pd.DataFrame, val_df: pd.DataFrame):
    """
    Two-stage stacking:
      1. Train an LSTM backbone on sequences.
      2. Extract the final LSTM hidden state as a feature vector.
      3. Train ALL tabular meta-models on those features and compare them.

    Returns: (lstm_backbone, meta_models)
      - lstm_backbone : trained LSTMRegressor
      - meta_models   : dict[name -> fitted model] for every meta-model
    Saved as: models/lstm_backbone.pt  +  models/stacked_<name>.pkl for each meta-model
    """
    # Stage 1: LSTM backbone
    print("  Stage 1/2 - Training LSTM backbone...")
    lstm_model = train_lstm(train_df, val_df)

    # Stage 2: extract features once, then benchmark all meta-models
    print("\n  Stage 2/2 - Extracting LSTM features & training all meta-models...")

    # shuffle=False so the order of extracted features matches the sequence targets
    train_feat_loader = make_loader(train_df, shuffle=False)
    val_feat_loader = make_loader(val_df, shuffle=False)

    X_train_feat = get_lstm_features(lstm_model, train_feat_loader)
    X_val_feat = get_lstm_features(lstm_model, val_feat_loader)

    # Sequence targets are aligned with the feature rows
    _, y_train_seq = create_sequences(train_df)
    _, y_val_seq = create_sequences(val_df)

    # XGBoost benefits from eval_set / early stopping; keep that separate
    def _train_xgb_meta(X_tr, y_tr):
        m = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            early_stopping_rounds=20,
        )
        m.fit(X_tr, y_tr,
              eval_set=[(X_val_feat, y_val_seq)])
        return m

    META_TRAINERS = {
        "linear": train_linear,
        "rf": train_random_forest,
        "xgboost": _train_xgb_meta,
        "ngboost": train_ngboost,
    }

    meta_models = {}
    results = []

    for name, train_fn in META_TRAINERS.items():
        print(f"Training LSTM -> {name}...")
        meta = train_fn(X_train_feat, y_train_seq)
        val_preds = meta.predict(X_val_feat)
        rmse = np.sqrt(mean_squared_error(y_val_seq, val_preds))
        mae = mean_absolute_error(y_val_seq, val_preds)
        meta_models[name] = meta
        results.append((name, rmse, mae))

    # Comparison table
    print("\n Stacked model comparison (LSTM features-> ta-model)")
    print(f" {'Meta-model':<12}  {'val RMSE':>10}  {'val MAE':>10}")
    for name, rmse, mae in sorted(results, key=lambda r: r[1]):
        print(f"  {name:<12}  {rmse:>10.2f}  {mae:>10.2f}")

    return lstm_model, meta_models


# Persistence

def save_model(model, name: str):
    os.makedirs(MODEL_DIR, exist_ok=True)

    if isinstance(model, tuple):
        # Stacked model: (LSTMRegressor, dict[name -> meta-model])
        lstm_model, meta_models = model
        lstm_path = os.path.join(MODEL_DIR, "lstm_backbone.pt")
        torch.save(lstm_model.state_dict(), lstm_path)
        print(f" Saved LSTM backbone: {lstm_path}")
        for meta_name, meta_model in meta_models.items():
            meta_path = os.path.join(MODEL_DIR, f"stacked_{meta_name}.pkl")
            joblib.dump(meta_model, meta_path)
            print(f" Saved stacked meta-model: {meta_path}")
    elif isinstance(model, nn.Module):
        path = os.path.join(MODEL_DIR, f"{name}.pt")
        torch.save(model.state_dict(), path)
        print(f" Saved: {path}")
    else:
        path = os.path.join(MODEL_DIR, f"{name}.pkl")
        joblib.dump(model, path)
        print(f" Saved: {path}")


def score(model, X_val, y_val, name: str):
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    mae = mean_absolute_error(y_val, preds)
    print(f" {name:<20} val RMSE={rmse:.2f}  MAE={mae:.2f}")


# Entry point 

TABULAR_MODELS = {
    "linear": train_linear,
    "rf":train_random_forest,
    "xgboost": train_xgboost,
    "ngboost": train_ngboost,
}

ALL_CHOICES = ["all", "lstm", "stacked", *TABULAR_MODELS.keys()]


def main(model_choice: str = "all"):
    train_df, _ = load_feature_data(TRAIN_PATH, TEST_PATH)
    train_split, val_split = split_by_engine(train_df)

    X_train, y_train = prep_xy(train_split)
    X_val, y_val = prep_xy(val_split)

    print(f" Train engines : {train_split['unit_number'].nunique()}  ({len(X_train)} rows)")
    print(f" Val engines : {val_split['unit_number'].nunique()}   ({len(X_val)} rows)\n")

    # Flags for what to run
    run_all = model_choice == "all"
    run_stacked = model_choice in ("all", "stacked")
    run_lstm = model_choice == "lstm"
    run_tabular = model_choice in TABULAR_MODELS

    # Tabular models
    tabular_to_run = TABULAR_MODELS if run_all else (
        {model_choice: TABULAR_MODELS[model_choice]} if run_tabular else {}
    )
    for name, train_fn in tabular_to_run.items():
        print(f"Training {name}...")
        model = train_fn(X_train, y_train)
        score(model, X_val, y_val, name)
        save_model(model, name)
        print()

    # Standalone LSTM 
    if run_lstm:
        print("Training lstm...")
        lstm_model = train_lstm(train_split, val_split)
        save_model(lstm_model, "lstm")
        print()

    # Stacked LSTM -> XGBoost 
    # "all" trains the stacked model as the representative LSTM-based model;
    # standalone LSTM can be trained separately with --model lstm if needed.
    if run_stacked:
        print("Training stacked (LSTM -> XGBoost)...")
        stacked = train_stacked(train_split, val_split)
        save_model(stacked, "stacked")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train RUL prediction models on CMAPSS FD001."
    )
    parser.add_argument(
        "--model",
        default="all",
        choices=ALL_CHOICES,
    )
    args = parser.parse_args()
    main(args.model)