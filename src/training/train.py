"""
src/training/train.py

Trains the final NGBoost meta-model using features extracted from the
optimal backbone found during tune.py. Optimises NGBoost hyperparameters
via Optuna (minimising NLL, not RMSE) and writes results to model_registry.yaml.

Usage:
    python src/training/train.py --dataset FD001 --n_trials 30
    python src/training/train.py --dataset all   --n_trials 30
"""

from __future__ import annotations

import argparse
import datetime
from pathlib import Path
import sys
import yaml

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
import torch
import optuna
from ngboost import NGBRegressor

# ── Path resolution ───────────────────────────────────────────────────────────
_ROOT     = Path(__file__).resolve().parents[2]
_TRAINING = _ROOT / "src" / "training"
if str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))

from shared import (
    DEVICE,
    build_backbone,
    make_loader,
    split_by_engine,
    get_backbone_features,
    create_sequences,
)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train NGBoost meta-model over backbone features."
    )
    parser.add_argument(
        "--dataset",
        choices=["FD001", "FD002", "FD003", "FD004", "all"],
        default="FD001",
    )
    parser.add_argument(
        "--n_trials", type=int, default=30,
        help="Number of Optuna trials for NGBoost hyperparameter search.",
    )
    return parser.parse_args()


# ── Per-dataset training ──────────────────────────────────────────────────────

def train_meta_for_dataset(
    dataset_key:   str,
    n_trials:      int,
    registry_path: Path,
) -> None:
    print(f"\n{'='*60}")
    print(f"  {dataset_key} — NGBoost meta-model training")
    print(f"{'='*60}")

    # 1. Load champion backbone config from registry ──────────────────────────
    with open(registry_path, "r") as f:
        registry = yaml.safe_load(f) or {}

    if dataset_key not in registry or "champion" not in registry[dataset_key]:
        print(f"  No champion config for {dataset_key}. Run tune.py first.")
        return

    champion    = registry[dataset_key]["champion"]
    bb_name     = champion["backbone"]
    bb_cfg      = champion["backbone_config"]
    seq_length  = bb_cfg["seq_length"]
    input_dim   = bb_cfg["input_dim"]
    bb_kwargs   = {k: v for k, v in bb_cfg.items()
                   if k not in ("input_dim", "seq_length")}

    # 2. Load feature data ────────────────────────────────────────────────────
    feature_path = _ROOT / "data" / "processed" / dataset_key / "train_features.csv"
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature file not found: {feature_path}")

    full_df              = pd.read_csv(feature_path)
    train_df, val_df     = split_by_engine(full_df)

    # 3. Rebuild backbone and load trained weights ─────────────────────────────
    print(f"  Rebuilding {bb_name} backbone and loading weights...")
    model             = build_backbone(bb_name, input_dim=input_dim, backbone_cfg=bb_kwargs)
    backbone_path     = Path(champion["artifacts"]["backbone"])

    if not backbone_path.exists():
        raise FileNotFoundError(
            f"Backbone weights not found: {backbone_path}\n"
            "Run tune.py first."
        )

    model.load_state_dict(
        torch.load(backbone_path, map_location=DEVICE, weights_only=True)
    )

    # 4. Extract backbone features ─────────────────────────────────────────────
    print("  Extracting backbone features...")
    train_loader = make_loader(train_df, seq_length=seq_length,
                               batch_size=256, shuffle=False)
    val_loader   = make_loader(val_df,   seq_length=seq_length,
                               batch_size=256, shuffle=False)

    _, y_train = create_sequences(train_df, seq_length=seq_length)
    _, y_val   = create_sequences(val_df,   seq_length=seq_length)

    X_train = get_backbone_features(model, train_loader)
    X_val   = get_backbone_features(model, val_loader)

    print(f"  Train features : {X_train.shape}   targets: {y_train.shape}")
    print(f"  Val   features : {X_val.shape}   targets: {y_val.shape}")

    # 5. Optuna objective — minimise NLL ──────────────────────────────────────
    # NGBoost's strength is calibrated uncertainty, so we optimise its actual
    # training objective (NLL) rather than RMSE. Hyperparameters are tuned on
    # the base learner (DecisionTreeRegressor) not on the NGBoost wrapper itself.
    def objective(trial: optuna.Trial) -> float:
        n_estimators     = trial.suggest_int("n_estimators", 100, 1000, step=50)
        learning_rate    = trial.suggest_float("learning_rate", 0.005, 0.1, log=True)
        max_depth        = trial.suggest_int("max_depth", 2, 6)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)

        meta = NGBRegressor(
            n_estimators  = n_estimators,
            learning_rate = learning_rate,
            Base          = DecisionTreeRegressor(
                max_depth        = max_depth,
                min_samples_leaf = min_samples_leaf,
            ),
            verbose = False,
        )
        meta.fit(X_train, y_train)

        val_dists = meta.pred_dist(X_val)
        val_nll   = -val_dists.logpdf(y_val).mean()
        return val_nll

    # 6. Run NGBoost hyperparameter search ────────────────────────────────────
    print(f"  Optimising NGBoost via Optuna ({n_trials} trials, objective = NLL)...")
    studies_dir = _ROOT / "studies"
    studies_dir.mkdir(exist_ok=True)
    study_db    = studies_dir / f"{dataset_key}_ngboost_v2.db"

    study = optuna.create_study(
        direction      = "minimize",
        storage        = f"sqlite:///{study_db}",
        study_name     = f"{dataset_key}_ngboost_search",
        load_if_exists = True,
    )
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_nll    = study.best_value
    print(f"  Best val NLL : {best_nll:.4f}")
    print(f"  Best params  : {best_params}")

    # 7. Train final NGBoost with best hyperparameters ─────────────────────────
    print("  Training final NGBoost with best parameters...")
    final_model = NGBRegressor(
        n_estimators  = best_params["n_estimators"],
        learning_rate = best_params["learning_rate"],
        Base          = DecisionTreeRegressor(
            max_depth        = best_params["max_depth"],
            min_samples_leaf = best_params["min_samples_leaf"],
        ),
        verbose      = False,
        verbose_eval = 100,   # print progress every 100 iterations
    )
    final_model.fit(X_train, y_train)

    # 8. Evaluate — NLL, RMSE, MAE, calibration coverage ─────────────────────
    print("  Evaluating final model...")
    val_dists = final_model.pred_dist(X_val)
    mu        = val_dists.loc
    sigma     = val_dists.scale
    val_nll   = -val_dists.logpdf(y_val).mean()
    val_preds = final_model.predict(X_val)
    val_rmse  = np.sqrt(mean_squared_error(y_val, val_preds))
    val_mae   = mean_absolute_error(y_val, val_preds)

    # Calibration coverage — for each α, what fraction of true values fall
    # inside the predicted α-interval? A well-calibrated model should match
    # the diagonal (e.g. 90% PI contains ~90% of true values).
    calibration: dict[str, float] = {}
    for alpha in [0.50, 0.80, 0.90, 0.95]:
        z       = norm.ppf((1 + alpha) / 2)
        in_band = ((y_val >= mu - z * sigma) & (y_val <= mu + z * sigma))
        key     = f"coverage_{int(alpha * 100)}"
        calibration[key] = float(in_band.mean())

    print(f"  val NLL  : {val_nll:.4f}")
    print(f"  val RMSE : {val_rmse:.4f}")
    print(f"  val MAE  : {val_mae:.4f}")
    print("  Calibration coverage:")
    for k, v in calibration.items():
        expected = int(k.split("_")[1]) / 100
        flag     = "✓" if abs(v - expected) < 0.05 else "✗"
        print(f"    {k}: {v:.3f}  (expected {expected:.2f})  {flag}")

    # 9. Save artifact ─────────────────────────────────────────────────────────
    meta_path = Path(champion["artifacts"]["meta_ngboost"])
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, meta_path)
    print(f"  Saved NGBoost → {meta_path}")

    # 10. Update registry ──────────────────────────────────────────────────────
    # Re-read before writing to avoid clobbering parallel dataset runs
    with open(registry_path, "r") as f:
        current = yaml.safe_load(f) or {}

    current[dataset_key]["champion"]["meta_model"] = "ngboost"
    current[dataset_key]["champion"]["meta_model_config"] = {
        "n_estimators":     int(best_params["n_estimators"]),
        "learning_rate":    float(best_params["learning_rate"]),
        "max_depth":        int(best_params["max_depth"]),
        "min_samples_leaf": int(best_params["min_samples_leaf"]),
    }
    current[dataset_key]["champion"]["metrics"].update({
        "val_nll":     float(val_nll),
        "val_rmse":    float(val_rmse),
        "val_mae":     float(val_mae),
        "calibration": calibration,
    })
    current[dataset_key]["champion"]["trained_at"]   = str(datetime.date.today())
    current[dataset_key]["champion"]["optuna_study"] = str(study_db)

    with open(registry_path, "w") as f:
        yaml.safe_dump(current, f, default_flow_style=False, sort_keys=False)
    print(f"  Registry updated → {registry_path.name}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args         = parse_args()
    registry     = _ROOT / "config" / "model_registry.yaml"
    datasets     = (["FD001", "FD002", "FD003", "FD004"]
                    if args.dataset == "all" else [args.dataset])

    for ds in datasets:
        train_meta_for_dataset(ds, args.n_trials, registry)