"""
Five-stage training pipeline for one or all CMAPSS datasets.

  Stage 1: Retrain the champion backbone from scratch for a full epoch budget.
             (tune.py found the best architecture; this gives it proper training time.)
  Stage 2: Pass every training/validation sequence through the frozen backbone
             to extract fixed-size feature vectors for NGBoost.
  Stage 3: Search for the best NGBoost hyperparameters using Optuna.
  Stage 4: Train the final NGBoost, evaluate on the validation set.
  Stage 5: Compute and save a sigma calibration scale so prediction intervals
             hit their stated coverage.

Usage
-----
    # Full pipeline
    python src/training/train.py --dataset FD001
    python src/training/train.py --dataset all --backbone_epochs 150

    # Skip training, just recompute sigma calibration from saved models
    python src/training/train.py --dataset all --calibrate_only
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import optuna
import torch
from ngboost import NGBRegressor
from scipy.optimize import brentq
from scipy.stats import norm
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.tree import DecisionTreeRegressor

# Locate the project root from this file's position so imports work regardless
# of where the script is launched from.
#   parents[0] = src/training/   (this file's folder)
#   parents[1] = src/
#   parents[2] = project root
_ROOT     = Path(__file__).resolve().parents[2]
_TRAINING = _ROOT / "src" / "training"
if str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))

from shared import (
    DEVICE,
    build_backbone,
    create_sequences,
    get_backbone_features,
    load_registry,
    make_loader,
    save_registry,
    split_by_engine,
    train_backbone,
)


# SIGMA CALIBRATION
# NGBoost outputs a predicted Normal distribution N(mu, sigma^2) for each
# engine. "Calibration" means: when we say the true RUL falls inside a
# 90% interval, it should actually do so 90% of the time. If it only does
# so 77% of the time, sigma is too small and we need to scale it up.

def _coverage_error(
    s: float,        # the sigma scale factor we are testing
    y_true: np.ndarray,   # true RUL values on the validation set
    mu: np.ndarray,   # NGBoost predicted means
    sigma: np.ndarray,   # NGBoost predicted standard deviations (before scaling)
    alpha: float,        # target coverage level, e.g. 0.90
) -> float:
    """
    Returns (actual coverage when sigma is multiplied by s) minus alpha.

    This function equals zero when s produces exactly the right coverage.
    brentq finds that zero by trying different values of s in the bracket
    [0.01, 10.0]:
      - At s = 0.01 the interval is almost zero-width -> coverage ~= 0 → return ~= -alpha
      - At s = 10.0 the interval spans the whole range -> coverage ~= 1 → return ~= 1-alpha
    Because the function goes from negative to positive, there is exactly
    one crossing point which is the s we want.
    """
    # z is the number of standard deviations needed for an alpha-level
    # two-sided interval. For alpha=0.90, z ~= 1.645.
    z = norm.ppf((1 + alpha) / 2)

    # Check how many true values fall inside [mu - z*s*sigma, mu + z*s*sigma]
    in_band = (y_true >= mu - z * s * sigma) & (y_true <= mu + z * s * sigma)

    # Subtract alpha so the function is zero at the correct scale factor
    return float(in_band.mean()) - alpha


def calibrate_sigma(dataset_key: str, registry_path: Path, alpha: float = 0.90) -> float:
    """
    Find the sigma multiplier that gives exact `alpha` coverage on the
    validation set, then save it to the registry for use at inference time.

    Why the validation set and not the test set?
    The test set is kept completely untouched as a true holdout as we never
    use it to make any decisions during training. The validation set was
    held out during backbone and NGBoost training, so calibrating on it
    does not contaminate either of those stages. The fixed random_state=42
    in split_by_engine() guarantees this call gets the same val engines
    that were excluded during training.

    Why alpha = 0.90?
    The 90% interval is what the alert system and predictor.py use to
    compute failure probabilities. Calibrating at 90% means P(RUL < N)
    values used for CRITICAL/WARNING decisions are computed from a sigma
    that has a proven coverage guarantee.
    """
    print(f"\n  [{dataset_key}] Computing sigma calibration (target coverage={alpha:.0%})")

    registry = load_registry(registry_path)
    if dataset_key not in registry or "champion" not in registry[dataset_key]:
        print(f"No champion config for {dataset_key}. Run full training first.")
        return 1.0  # scale of 1.0 = no change

    champion = registry[dataset_key]["champion"]

    # Rebuild backbone and load its trained weights
    bb_name = champion["backbone"]   # architecture name, e.g. "gru"
    bb_cfg = champion["backbone_config"]
    seq_length = bb_cfg["seq_length"]   # number of cycles per input window
    input_dim = bb_cfg["input_dim"]    # number of sensor features

    # bb_kwargs: everything in backbone_config except the two meta-keys we
    # handle separately (input_dim and seq_length). These are the actual
    # constructor arguments for the backbone class, e.g. hidden_dim, dropout.
    bb_kwargs = {k: v for k, v in bb_cfg.items() if k not in ("input_dim", "seq_length")}

    feature_path = _ROOT / "data" / "processed" / dataset_key / "train_features.csv"
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature file not found: {feature_path}")

    full_df = pd.read_csv(feature_path)
    _, val_df = split_by_engine(full_df)   # same split as training (random_state=42)

    model = build_backbone(bb_name, input_dim=input_dim, backbone_cfg=bb_kwargs)
    model.load_state_dict(torch.load(Path(champion["artifacts"]["backbone"]), map_location=DEVICE, weights_only=True))

    # Extract backbone features and run NGBoost on the val set 
    val_loader = make_loader(val_df, seq_length=seq_length, batch_size=256, shuffle=False)
    _, y_val = create_sequences(val_df, seq_length=seq_length)   # true RUL targets
    X_val = get_backbone_features(model, val_loader)           # backbone latent vectors

    ngb_model = joblib.load(Path(champion["artifacts"]["meta_ngboost"]))

    # pred_dist() returns a Normal distribution object; .loc is the mean, .scale is std
    val_dists = ngb_model.pred_dist(X_val)
    mu, sigma = val_dists.loc, val_dists.scale

    # Show current (pre-calibration) coverage 
    z = norm.ppf((1 + alpha) / 2)
    current_coverage = float(((y_val >= mu - z * sigma) & (y_val <= mu + z * sigma)).mean())
    print(f"  Current {int(alpha*100)}% coverage : {current_coverage:.4f}  (target {alpha:.2f})")

    # Find the scale factor using brentq root-finding
    scale = brentq(
        _coverage_error,
        a=0.01, b=10.0,                 # safe bracket — see _coverage_error docstring
        args=(y_val, mu, sigma, alpha),
        xtol=1e-6,                       # stop when scale is known to 6 decimal places
    )
    direction = "↑ widen intervals" if scale > 1 else "↓ narrow intervals"
    print(f"  Scale factor found : {scale:.6f}  ({direction})")

    # Verify the result
    post_coverage = float(
        ((y_val >= mu - z * scale * sigma) & (y_val <= mu + z * scale * sigma)).mean()
    )
    print(f"  Post-scale coverage: {post_coverage:.4f}  (target {alpha:.2f})")

    # Save to registry
    current = load_registry(registry_path)
    current[dataset_key]["champion"]["calibration_scale"]        = float(scale)
    current[dataset_key]["champion"]["calibration_alpha"]        = alpha
    current[dataset_key]["champion"]["calibration_pre_coverage"] = current_coverage
    current[dataset_key]["champion"]["calibrated_at"]            = str(datetime.date.today())
    save_registry(registry_path, current)
    print(f"  Registry updated → calibration_scale={scale:.6f}")

    return float(scale)


# CLI ARGUMENT PARSING
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrain backbone + optimise NGBoost + calibrate sigma."
    )
    parser.add_argument(
        "--dataset",
        choices=["FD001", "FD002", "FD003", "FD004", "all"],
        default="FD001",
        help="Which dataset to train on. 'all' runs all four sequentially.",
    )
    parser.add_argument(
        "--n_trials", type=int, default=30,
        help="Number of Optuna trials for NGBoost hyperparameter search.",
    )
    parser.add_argument(
        "--backbone_epochs", type=int, default=150,
        help="Maximum training epochs for the backbone retrain.",
    )
    parser.add_argument(
        "--backbone_patience", type=int, default=15,
        help="Early-stopping patience: stop if val RMSE hasn't improved for this many epochs.",
    )
    parser.add_argument(
        "--backbone_lr", type=float, default=1e-3,
        help="Learning rate for the backbone Adam optimiser.",
    )
    parser.add_argument(
        "--calibrate_only", action="store_true",
        help="Skip all training. Just compute and save sigma calibration "
             "scale factors from already-trained models.",
    )
    return parser.parse_args()


# MAIN TRAINING PIPELINE
def train_meta_for_dataset(
    dataset_key: str,
    n_trials: int,
    registry_path: Path,
    backbone_epochs: int,
    backbone_patience: int,
    backbone_lr: float,
) -> None:
    """
    Run all five training stages for one dataset.

    The registry (model_registry.yaml) must already contain a champion
    entry for this dataset that is written by tune.py. This function
    reads the champion architecture and hyperparameters, retrains the
    backbone fully, stacks NGBoost on top, and saves everything.

    Parameters
    ----------
    dataset_key       : e.g. "FD001" which CMAPSS dataset to train on
    n_trials          : how many Optuna trials to run for NGBoost search
    registry_path     : path to config/model_registry.yaml
    backbone_epochs   : max epochs for the backbone retrain
    backbone_patience : early-stopping patience for the backbone
    backbone_lr       : Adam learning rate for the backbone
    """
    print(f"\n{'='*60}\n  {dataset_key}\n{'='*60}")

    # Load the champion configuration from tune.py
    registry = load_registry(registry_path)
    if dataset_key not in registry or "champion" not in registry[dataset_key]:
        print(f"No champion config for {dataset_key}. Run tune.py first.")
        return

    champion = registry[dataset_key]["champion"]
    bb_name = champion["backbone"]          # e.g. "gru", "tcn", "transformer"
    bb_cfg = champion["backbone_config"]
    seq_length = bb_cfg["seq_length"]          # sliding window length (cycles)
    input_dim = bb_cfg["input_dim"]           # number of input sensor features

    # Strip meta-keys to get the backbone constructor arguments only
    bb_kwargs = {k: v for k, v in bb_cfg.items() if k not in ("input_dim", "seq_length")}

    # Load and split data
    feature_path = _ROOT / "data" / "processed" / dataset_key / "train_features.csv"
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature file not found: {feature_path}")

    full_df = pd.read_csv(feature_path)
    train_df, val_df = split_by_engine(full_df)   # split by whole engine, not by row

    print(f"  Backbone: {bb_name}  |  seq_length={seq_length} | input_dim={input_dim}")
    print(f"  Train: {train_df['unit_number'].nunique()} engines ({len(train_df):,} rows)")
    print(f"  Val: {val_df['unit_number'].nunique()} engines ({len(val_df):,} rows)")

    # STAGE 1 — Retrain backbone from scratch
    # tune.py saved weights from a short Optuna trial (patience=7, ~20-50 epochs).
    # Here we rebuild the same architecture and train for a full epoch budget so
    # the backbone has every opportunity to converge before we extract features.
    print(f"\n  [Stage 1] Retraining {bb_name} backbone "
          f"(max {backbone_epochs} epochs, patience={backbone_patience})")

    model = build_backbone(bb_name, input_dim=input_dim, backbone_cfg=bb_kwargs)

    # shuffle=True for training (randomises sequence order each epoch)
    # shuffle=False for validation (order must be consistent for metric tracking)
    train_loader_bb = make_loader(train_df, seq_length=seq_length, batch_size=256, shuffle=True)
    val_loader_bb = make_loader(val_df,   seq_length=seq_length, batch_size=256, shuffle=False)

    # trial=None means no Optuna pruning — this is the final training run
    # verbose=True prints every epoch so you can monitor convergence
    model, best_bb_rmse = train_backbone(
        model, train_loader_bb, val_loader_bb,
        cfg={"lr": backbone_epochs, "epochs": backbone_epochs, "patience": backbone_patience},
        trial=None, verbose=True,
    )
    print(f"\n  [Stage 1] Best backbone val RMSE: {best_bb_rmse:.4f}")

    backbone_path = Path(champion["artifacts"]["backbone"])
    backbone_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), backbone_path)
    print(f"  [Stage 1] Saved at {backbone_path}")

    # STAGE 2 - Extract backbone features
    # Pass every training and validation sequence through model.encode() to get
    # a fixed-size representation vector (the backbone's final hidden state).
    # These compact vectors become the input features for NGBoost.
    #
    # IMPORTANT: shuffle=False here. The order of extracted features must match
    # the order of the RUL targets from create_sequences(). If you shuffle the
    # DataLoader, features[i] and y[i] would refer to different sequences.
    print("\n  [Stage 2] Extracting backbone features")

    # Separate loaders with shuffle=False for feature extraction
    train_loader = make_loader(train_df, seq_length=seq_length, batch_size=256, shuffle=False)
    val_loader = make_loader(val_df,   seq_length=seq_length, batch_size=256, shuffle=False)

    # y_train / y_val: true RUL targets for each sliding window sequence
    _, y_train = create_sequences(train_df, seq_length=seq_length)
    _, y_val = create_sequences(val_df,   seq_length=seq_length)

    # X_train / X_val: backbone latent feature vectors, shape (n_sequences, hidden_dim)
    X_train = get_backbone_features(model, train_loader)
    X_val = get_backbone_features(model, val_loader)

    print(f"  Train features : {X_train.shape} targets: {y_train.shape}")
    print(f"  Val features : {X_val.shape} targets: {y_val.shape}")

    # STAGE 3 - NGBoost hyperparameter search
    # NGBoost fits a gradient-boosted ensemble of decision trees to produce a
    # calibrated Normal distribution N(mu, sigma^2) per prediction.
    # We optimise NLL (negative log-likelihood) rather than RMSE because NLL
    # rewards both accurate predictions AND honest uncertainty estimates.
    # A model that predicts accurately but with overconfident sigma scores
    # badly on NLL - which is exactly the property we care about for the
    # alert system.
    # The base learner (DecisionTreeRegressor) is what we actually tune:
    # max_depth and min_samples_leaf control how complex each tree is.
    print(f"\n  [Stage 3] Searching NGBoost hyperparameters "
          f"({n_trials} Optuna trials, objective = NLL)")

    def objective(trial: optuna.Trial) -> float:
        # Suggest hyperparameters for this trial
        n_estimators = trial.suggest_int("n_estimators", 100, 1000, step=50)
        learning_rate = trial.suggest_float("learning_rate",  0.005, 0.1, log=True)
        max_depth = trial.suggest_int("max_depth", 2, 6)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)

        meta = NGBRegressor(
            n_estimators  = n_estimators,
            learning_rate = learning_rate,
            Base = DecisionTreeRegressor(max_depth = max_depth, min_samples_leaf = min_samples_leaf),
            verbose = False,
        )
        meta.fit(X_train, y_train)

        # logpdf(y) = log probability of the true value under the predicted distribution.
        # We negate and average to get NLL - lower NLL means better calibration.
        val_nll = float(-meta.pred_dist(X_val).logpdf(y_val).mean())
        return val_nll

    studies_dir = _ROOT / "studies"
    studies_dir.mkdir(exist_ok=True)
    study_db = studies_dir / f"{dataset_key}_ngboost_v2.db"  # Optuna saves trials here

    study = optuna.create_study(
        direction = "minimize",
        storage = f"sqlite:///{study_db}",   # SQLite file - resumable if interrupted
        study_name = f"{dataset_key}_ngboost_search",
        load_if_exists = True,                       # pick up where we left off
    )
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    print(f" Best val NLL : {study.best_value:.4f}")
    print(f" Best params  : {best_params}")

    # STAGE 4 - Train final NGBoost and evaluate
    print("\n  [Stage 4] Training final NGBoost with best parameters")
    final_model = NGBRegressor(
        n_estimators = best_params["n_estimators"],
        learning_rate = best_params["learning_rate"],
        Base = DecisionTreeRegressor(
            max_depth = best_params["max_depth"],
            min_samples_leaf = best_params["min_samples_leaf"],
        ),
        verbose = False,
        verbose_eval = 100,   # print a progress line every 100 boosting rounds
    )
    final_model.fit(X_train, y_train)

    # Evaluate on the validation set
    # pred_dist() returns a Normal distribution object:
    #   .loc   = predicted mean RUL (mu)
    #   .scale = predicted std deviation (sigma)
    val_dists = final_model.pred_dist(X_val)
    mu = val_dists.loc    # predicted mean RUL for each validation window
    sigma = val_dists.scale  # predicted uncertainty for each window

    val_nll = float(-val_dists.logpdf(y_val).mean())   # negative log-likelihood
    val_preds = final_model.predict(X_val)                # point predictions (= mu)
    val_rmse = float(np.sqrt(mean_squared_error(y_val, val_preds)))
    val_mae = float(mean_absolute_error(y_val, val_preds))

    # NASA asymmetric score: d = predicted - true
    #   d > 0 (predicted TOO MUCH life remaining) -> penalised with exp(d/10) - 1
    #   d < 0 (predicted TOO LITTLE life) -> penalised with exp(-d/13) - 1
    # Positive errors use divisor 10 (steeper penalty) because late predictions
    # are operationally more dangerous than early ones.
    d    = val_preds - y_val
    nasa = float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))

    # Percentage of predictions within N cycles of the true RUL
    within = {
        f"within_{n}_cycles": float(np.mean(np.abs(val_preds - y_val) <= n))
        for n in [5, 10, 15]
    }
    # Calibration coverage: what fraction of true values fall inside the alpha interval?
    # A well-calibrated model should have coverage ~= alpha at every level.
    calibration: dict[str, float] = {}
    for alpha in [0.50, 0.80, 0.90, 0.95]:
        z = norm.ppf((1 + alpha) / 2)   # z-score for a two-sided alpha interval
        in_band = (y_val >= mu - z * sigma) & (y_val <= mu + z * sigma)
        calibration[f"coverage_{int(alpha * 100)}"] = float(in_band.mean())

    # Binary failure AUC: use P(RUL < horizon) as a ranking score for the binary
    # question "will this engine fail within N cycles?". AUC near 1.0 means the
    # model correctly ranks near-failure engines above healthy ones.
    auc_metrics: dict[str, float] = {}
    for horizon in [20, 50]:
        # Create binary labels: 1 if the engine will fail within `horizon` cycles
        binary_true = (y_val < horizon).astype(int)
        # Only compute AUC when both classes (0 and 1) are present in the val set
        if 0 < binary_true.sum() < len(binary_true):
            failure_probs = norm.cdf(horizon, loc=mu, scale=sigma)   # P(RUL < horizon)
            auc_metrics[f"auc_failure_{horizon}"] = float(
                roc_auc_score(binary_true, failure_probs)
            )

    # Mean width of the 90% prediction interval across all val windows
    pi_width_90 = float(np.mean(2 * norm.ppf(0.95) * sigma))

    # --- Print summary ---
    print(f"\n  {'─'*40}")
    print(f"  val NLL: {val_nll:.4f}")
    print(f"  val RMSE: {val_rmse:.4f}")
    print(f"  val MAE: {val_mae:.4f}")
    print(f"  NASA score: {nasa:.1f}  (lower = better)")
    print(f"  Mean PI width 90%: {pi_width_90:.2f} cycles")
    for k, v in within.items():
        print(f"  {k:<22}: {v * 100:.1f}%")
    for k, v in auc_metrics.items():
        print(f"  {k:<22}: {v:.4f}")
    print("  Calibration (pre sigma-scaling):")
    for k, v in calibration.items():
        expected = int(k.split("_")[1]) / 100
        flag = "Y" if abs(v - expected) < 0.05 else "N"
        print(f" {k}: {v:.3f}  (expected {expected:.2f})  {flag}")

    # Save model artifact
    meta_path = Path(champion["artifacts"]["meta_ngboost"])
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, meta_path)
    print(f"\n  Saved NGBoost → {meta_path}")

    # Update registry
    # Re-read before writing to avoid overwriting changes from parallel runs
    current = load_registry(registry_path)
    current[dataset_key]["champion"]["meta_model"] = "ngboost"
    current[dataset_key]["champion"]["meta_model_config"] = {
        "n_estimators": int(best_params["n_estimators"]),
        "learning_rate": float(best_params["learning_rate"]),
        "max_depth": int(best_params["max_depth"]),
        "min_samples_leaf": int(best_params["min_samples_leaf"]),
    }
    current[dataset_key]["champion"]["metrics"].update({
        "val_nll": val_nll,
        "val_rmse": val_rmse,
        "val_mae": val_mae,
        "nasa_score":  nasa,
        "pi_width_90": pi_width_90,
        "calibration": calibration,
        **within,
        **auc_metrics,
    })
    current[dataset_key]["champion"]["backbone_retrain"] = {
        "epochs": backbone_epochs,
        "patience": backbone_patience,
        "lr": backbone_lr,
        "best_val_rmse": float(best_bb_rmse),
    }
    current[dataset_key]["champion"]["trained_at"] = str(datetime.date.today())
    current[dataset_key]["champion"]["optuna_study"] = str(study_db)
    save_registry(registry_path, current)
    print(f"  Registry updated → {registry_path.name}")

    # STAGE 5 — Sigma calibration
    # Always runs automatically at the end of a full training pipeline.
    # Can also be run standalone with --calibrate_only.
    calibrate_sigma(dataset_key, registry_path)


# ENTRY POINT
if __name__ == "__main__":
    args = parse_args()
    registry = _ROOT / "config" / "model_registry.yaml"
    datasets = (
        ["FD001", "FD002", "FD003", "FD004"]
        if args.dataset == "all" else [args.dataset]
    )

    if args.calibrate_only:
        # Fast path: load existing models and recompute calibration only
        print("Calibrate-only mode: computing sigma scale factors from existing models.")
        for ds in datasets:
            calibrate_sigma(ds, registry)
        print("\nDone. Re-run evaluate.py to see updated calibration metrics.")
    else:
        for ds in datasets:
            train_meta_for_dataset(
                dataset_key = ds,
                n_trials = args.n_trials,
                registry_path = registry,
                backbone_epochs = args.backbone_epochs,
                backbone_patience = args.backbone_patience,
                backbone_lr = args.backbone_lr,
            )