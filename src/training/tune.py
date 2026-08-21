"""
src/training/tune.py

Optuna hyperparameter search across backbone architectures.

Usage
-----
    python src/training/tune.py --dataset FD001 --n_trials 50

After the study completes, the best trial's config is written into
config/model_registry.yaml under the dataset key. train.py then reads
that config to do the full training run.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch

_ROOT = Path(__file__).resolve().parents[2]
_TRAINING = _ROOT / "src" / "training"
if str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))

from shared import (
    DEVICE,
    build_backbone,
    load_registry,
    make_loader,
    save_registry,
    split_by_engine,
    train_backbone,
)


def update_registry(
    registry_path: Path,
    dataset: str,
    backbone_name: str,
    best_rmse: float,
    backbone_cfg: dict,
    input_dim: int,
    seq_length: int,
    study_path: str,
) -> None:
    """Overwrite `dataset`'s champion entry in model_registry.yaml.
    All other dataset entries are left untouched."""
    registry = load_registry(registry_path)
    registry.setdefault(dataset, {})

    artifact_dir = str(_ROOT / "models" / dataset)
    registry[dataset]["champion"] = {
        "backbone": backbone_name,
        "meta_model": None,  # set by train.py after meta comparison
        "backbone_config": {**backbone_cfg, "input_dim": input_dim, "seq_length": seq_length},
        "metrics": {"val_rmse": float(best_rmse), "val_mae": None, "val_nll": None, "nasa_score": None},
        "artifacts": {
            "backbone": str(Path(artifact_dir) / "backbone.pt"),
            "meta_ngboost": str(Path(artifact_dir) / "stacked_ngboost.pkl"),
        },
        "trained_at": str(date.today()),
        "optuna_study": study_path,
    }

    save_registry(registry_path, registry)
    print(f" Registry updated: {registry_path}  [{dataset} champion: {backbone_name} RMSE={best_rmse:.4f}]")


def objective(trial: optuna.Trial, train_df: pd.DataFrame, val_df: pd.DataFrame, train_cfg: dict, dataset: str) -> float:
    backbone_name = trial.suggest_categorical("backbone", ["lstm", "gru", "tcn", "transformer"])

    # Shared hyperparameters
    hidden_dim = trial.suggest_int("hidden_dim", 64, 256, step=64)
    dropout = trial.suggest_float("dropout", 0.1, 0.4)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    seq_length = trial.suggest_int("seq_length", 20, 50, step=10)

    # Architecture-specific hyperparameters
    if backbone_name in ("lstm", "gru"):
        num_layers = trial.suggest_int("num_layers", 1, 3)
        bidirectional = trial.suggest_categorical("bidirectional", [True, False])
        backbone_cfg = dict(hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)

    elif backbone_name == "tcn":
        n_levels = trial.suggest_int("tcn_levels", 3, 6)
        kernel_size = trial.suggest_int("kernel_size", 2, 5)
        backbone_cfg = dict(num_channels=[hidden_dim] * n_levels, kernel_size=kernel_size, dropout=dropout)

    elif backbone_name == "transformer":
        nhead = trial.suggest_categorical("nhead", [2, 4, 8])
        num_layers = trial.suggest_int("tf_layers", 1, 4)
        dim_ff = trial.suggest_int("dim_feedforward", 128, 512, step=128)

        if hidden_dim % nhead != 0:  # d_model must be divisible by nhead
            raise optuna.exceptions.TrialPruned()

        backbone_cfg = dict(d_model=hidden_dim, nhead=nhead, num_layers=num_layers, dim_feedforward=dim_ff, dropout=dropout)

    # Build loaders with this trial's sequence length
    train_loader = make_loader(train_df, seq_length=seq_length, batch_size=train_cfg["batch_size"], shuffle=True)
    val_loader = make_loader(val_df, seq_length=seq_length, batch_size=train_cfg["batch_size"], shuffle=False)

    if len(train_loader.dataset) == 0 or len(val_loader.dataset) == 0:
        raise optuna.exceptions.TrialPruned()

    sample_x, _ = next(iter(train_loader))
    input_dim = sample_x.shape[2]

    try:
        model = build_backbone(backbone_name, input_dim, backbone_cfg)
    except Exception:
        raise optuna.exceptions.TrialPruned()

    _, best_val_rmse = train_backbone(
        model, train_loader, val_loader, cfg={**train_cfg, "lr": lr}, trial=trial, verbose=False,
    )

    # Track the global best and save it whenever this trial improves on it
    global_best = trial.study.user_attrs.get("global_best_rmse", float("inf"))
    if best_val_rmse < global_best:
        trial.study.set_user_attr("global_best_rmse", best_val_rmse)

        model_dir = _ROOT / "models" / dataset
        model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_dir / "backbone.pt")

        update_registry(
            registry_path=_ROOT / "config" / "model_registry.yaml",
            dataset=dataset,
            backbone_name=backbone_name,
            best_rmse=best_val_rmse,
            backbone_cfg=backbone_cfg,
            input_dim=input_dim,
            seq_length=seq_length,
            study_path=str(_ROOT / "studies" / f"{dataset}_v1.db"),
        )

    return best_val_rmse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backbone architecture search via Optuna.")
    parser.add_argument("--dataset", choices=["FD001", "FD002", "FD003", "FD004"], default="FD001")
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=50, help="Max epochs per trial")
    parser.add_argument("--patience", type=int, default=7, help="Early-stopping patience per trial")
    parser.add_argument("--batch_size", type=int, default=256)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    feature_path = _ROOT / "data" / "processed" / args.dataset / "train_features.csv"

    if not feature_path.exists():
        raise FileNotFoundError(
            f"Feature file not found: {feature_path}\n"
            f"Run preprocessing and feature engineering first:\n"
            f"  python src/data/preprocessing.py --dataset {args.dataset}\n"
            f"  (then run the feature engineering notebook)"
        )

    full_df = pd.read_csv(feature_path)
    train_df, val_df = split_by_engine(full_df)

    print(f"Dataset: {args.dataset}")
    print(f"Device: {DEVICE}")
    print(f"Train: {train_df['unit_number'].nunique()} engines  ({len(train_df):,} rows)")
    print(f"Val : {val_df['unit_number'].nunique()} engines  ({len(val_df):,} rows)")
    print(f"Trials: {args.n_trials}\n")

    train_cfg = {"batch_size": args.batch_size, "epochs": args.epochs, "patience": args.patience}

    studies_dir = _ROOT / "studies"
    studies_dir.mkdir(exist_ok=True)
    study_db = studies_dir / f"{args.dataset}_v1.db"

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        storage=f"sqlite:///{study_db}",
        study_name=f"{args.dataset}_backbone_search",
        load_if_exists=True,  # resume if interrupted
    )
    study.set_user_attr("global_best_rmse", study.user_attrs.get("global_best_rmse", float("inf")))

    study.optimize(lambda trial: objective(trial, train_df, val_df, train_cfg, args.dataset), n_trials=args.n_trials)

    print(f"Optimization complete - {args.dataset}")
    print(f"Best trial : #{study.best_trial.number}")
    print(f"Best RMSE  : {study.best_value:.4f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")