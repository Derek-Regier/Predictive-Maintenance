"""
src/training/shared.py

Shared utilities imported by both tune.py and train.py.
Provides sequence construction, engine-aware splitting, the backbone
factory, feature extraction, and the core training loop.

"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error
import optuna

# shared.py is at  src/training/shared.py
#   parents[0] = src/training/
#   parents[1] = src/
#   parents[2] = project root
_ROOT = Path(__file__).resolve().parents[2]
_BACKBONES = _ROOT / "src" / "models" / "backbones"

if str(_BACKBONES) not in sys.path:
    sys.path.insert(0, str(_BACKBONES))

from lstm import LSTMBackbone
from gru import GRUBackbone
from tcn import TCNBackbone
from transformer import TransformerBackbone


TARGET_COL = "RUL"
NON_FEATURE_COLS = ["unit_number", "time_in_cycles", TARGET_COL]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BACKBONE_REGISTRY: dict[str, type] = {
    "lstm": LSTMBackbone,
    "gru": GRUBackbone,
    "tcn": TCNBackbone,
    "transformer": TransformerBackbone,
}

class SequenceDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: np.ndarray) -> None:
        self.X = X
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]

def create_sequences(df, seq_length: int) -> tuple[torch.Tensor, np.ndarray]:
    """
    Sliding-window sequences grouped by engine.
    Target = RUL at the last timestep of each window.
    Returns (tensor[N, seq_length, features], array[N]).
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
                y_final.append(y_values[i + seq_length - 1])

    if not X_seq:
        n_feat = len(df.columns) - len(cols_to_drop)
        return torch.empty((0, seq_length, n_feat)), np.array([])

    return (
        torch.tensor(np.array(X_seq), dtype=torch.float32),
        np.array(y_final),
    )

def make_loader(
    df,
    seq_length: int,
    batch_size: int = 256,
    shuffle: bool = True,
) -> DataLoader:
    X_seq, y_seq = create_sequences(df, seq_length)
    ds = SequenceDataset(X_seq, y_seq)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=True)

def split_by_engine(
    df,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """
    Split whole engines into train/val so no single engine appears in both sets.
    Uses GroupShuffleSplit on unit_number.
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size,
                                 random_state=random_state)
    groups = df["unit_number"]
    train_idx, val_idx = next(splitter.split(df, df[TARGET_COL], groups))
    return df.iloc[train_idx].copy(), df.iloc[val_idx].copy()


def build_backbone(name: str, input_dim: int, backbone_cfg: dict) -> nn.Module:
    """Instantiate any backbone by name and move to DEVICE."""
    if name not in BACKBONE_REGISTRY:
        raise ValueError(
            f"Unknown backbone '{name}'. "
            f"Available: {list(BACKBONE_REGISTRY)}"
        )
    return BACKBONE_REGISTRY[name](input_dim=input_dim, **backbone_cfg).to(DEVICE)

def get_backbone_features(model: nn.Module, loader: DataLoader) -> np.ndarray:
    """
    Call model.encode() on every batch and return concatenated numpy array.
    Works for any backbone — no architecture-specific assumptions.
    """
    model.eval()
    parts: list[np.ndarray] = []
    with torch.no_grad():
        for x_batch, _ in loader:
            parts.append(model.encode(x_batch.to(DEVICE)).cpu().numpy())
    return np.concatenate(parts, axis=0)

def train_backbone(
    model:  nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: dict,
    trial: optuna.Trial | None = None,
    verbose: bool = True,
) -> tuple[nn.Module, float]:
    """
    Generic backbone training loop.

    Features
    --------
    - Adam + ReduceLROnPlateau scheduler
    - Gradient clipping (max_norm=1.0)
    - Early stopping
    - Optuna pruning when `trial` is provided
    - `verbose=False` silences per-epoch output during Optuna trials

    cfg keys
    --------
    lr, epochs, patience   (batch_size is set on the loaders, not here)

    Returns
    -------
    (model_with_best_weights, best_val_rmse)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )
    criterion = nn.MSELoss()

    epochs = cfg.get("epochs", 50)
    patience = cfg.get("patience", 7)

    best_val_rmse = float("inf")
    best_weights: dict | None = None
    epochs_no_improve = 0

    for epoch in range(epochs):

        model.train()
        running_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * len(y_batch)

        train_rmse = np.sqrt(running_loss / len(train_loader.dataset))

        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                val_preds.append(model(x_batch.to(DEVICE)).cpu().numpy())
                val_targets.append(y_batch.numpy())

        val_preds   = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)
        val_rmse    = np.sqrt(mean_squared_error(val_targets, val_preds))

        scheduler.step(val_rmse)

        if verbose:
            print(f"  Epoch {epoch + 1:>3}/{epochs}  "
                  f"train_RMSE={train_rmse:.4f}  val_RMSE={val_rmse:.4f}")

        if trial is not None:
            trial.report(val_rmse, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        if val_rmse < best_val_rmse:
            best_val_rmse     = val_rmse
            epochs_no_improve = 0
            # .clone() is required — .cpu() alone moves but does not copy
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch + 1} "
                          f"(patience={patience})")
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    return model, best_val_rmse