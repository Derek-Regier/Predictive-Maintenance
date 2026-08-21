"""
src/training/train.py

Stage 1 — Retrain the champion backbone from scratch with a full epoch budget.
Stage 2 — Extract backbone features (shuffle=False, order must match targets).
Stage 3 — Optimise NGBoost hyperparameters via Optuna (objective = NLL).
Stage 4 — Train final NGBoost, evaluate, save artifacts.
Stage 5 — Compute and save sigma calibration scale factor (brentq).

Usage
-----
    # Full pipeline (retrain backbone + NGBoost + calibrate)
    python src/training/train.py --dataset FD001
    python src/training/train.py --dataset all --backbone_epochs 150

    # Calibration only — no retraining, just compute sigma scale
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

_ROOT = Path(__file__).resolve().parents[2]
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


# ── Sigma calibration ─────────────────────────────────────────────────────────

def _coverage_error(s: float, y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray, alpha: float) -> float:
    """
    (actual coverage of the [mu ± z*s*sigma] interval at scale s) - alpha.
    Zero when s produces exactly `alpha` coverage — this is the root
    brentq searches for. s>1 widens the interval, s<1 narrows it.
    """
    z = norm.ppf((1 + alpha) / 2)
    in_band = (y_true >= mu - z * s * sigma) & (y_true <= mu + z * s * sigma)
    return float(in_band.mean()) - alpha


def calibrate_sigma(dataset_key: str, registry_path: Path, alpha: float = 0.90) -> float:
    """
    Find the sigma scale factor that brings NGBoost's predicted `alpha`
    coverage on the val set to exactly `alpha` (brentq root-finding —
    equivalent to conformal calibration). Saves the scale to the registry.

    Uses the val set (not the test set) so the test set stays a true
    holdout; split_by_engine's fixed random_state=42 reproduces the same
    val engines that were excluded during training.

    alpha=0.90 because that's the interval alert_tier() and predictor.py
    use for CRITICAL/WARNING decisions.
    """
    print(f"\n  [{dataset_key}] Computing sigma calibration scale (alpha={alpha})...")

    registry = load_registry(registry_path)
    if dataset_key not in registry or "champion" not in registry[dataset_key]:
        print(f"  No champion config for {dataset_key}. Run full training first.")
        return 1.0

    champion = registry[dataset_key]["champion"]
    bb_name = champion["backbone"]
    bb_cfg = champion["backbone_config"]
    seq_length = bb_cfg["seq_length"]
    input_dim = bb_cfg["input_dim"]
    bb_kwargs = {k: v for k, v in bb_cfg.items() if k not in ("input_dim", "seq_length")}

    feature_path = _ROOT / "data" / "processed" / dataset_key / "train_features.csv"
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature file not found: {feature_path}")

    full_df = pd.read_csv(feature_path)
    _, val_df = split_by_engine(full_df)  # random_state=42 matches training

    model = build_backbone(bb_name, input_dim=input_dim, backbone_cfg=bb_kwargs)
    model.load_state_dict(
        torch.load(Path(champion["artifacts"]["backbone"]), map_location=DEVICE, weights_only=True)
    )

    val_loader = make_loader(val_df, seq_length=seq_length, batch_size=256, shuffle=False)
    _, y_val = create_sequences(val_df, seq_length=seq_length)
    X_val = get_backbone_features(model, val_loader)

    ngb_model = joblib.load(Path(champion["artifacts"]["meta_ngboost"]))
    val_dists = ngb_model.pred_dist(X_val)
    mu, sigma = val_dists.loc, val_dists.scale

    z = norm.ppf((1 + alpha) / 2)
    current_coverage = float(((y_val >= mu - z * sigma) & (y_val <= mu + z * sigma)).mean())
    print(f"  Current {int(alpha*100)}% coverage : {current_coverage:.4f}  (target {alpha:.2f})")

    # Bracket [0.01, 10.0] is safe: coverage is monotone in s, ~0 at s=0.01
    # (too narrow) and ~1.0 at s=10 (spans the whole range) — exactly one root.
    scale = brentq(_coverage_error, a=0.01, b=10.0, args=(y_val, mu, sigma, alpha), xtol=1e-6)
    print(f"  Sigma scale factor : {scale:.6f}  ({'↑ widen intervals' if scale > 1 else '↓ narrow intervals'})")

    post_coverage = float(((y_val >= mu - z * scale * sigma) & (y_val <= mu + z * scale * sigma)).mean())
    print(f"  Post-scale coverage: {post_coverage:.4f}  (target {alpha:.2f})")

    current = load_registry(registry_path)
    current[dataset_key]["champion"]["calibration_scale"] = float(scale)
    current[dataset_key]["champion"]["calibration_alpha"] = alpha
    current[dataset_key]["champion"]["calibration_pre_coverage"] = current_coverage
    current[dataset_key]["champion"]["calibrated_at"] = str(datetime.date.today())
    save_registry(registry_path, current)
    print(f"  Registry updated → calibration_scale={scale:.6f}")

    return float(scale)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full training run: backbone retrain + NGBoost + sigma calibration.")
    parser.add_argument("--dataset", choices=["FD001", "FD002", "FD003", "FD004", "all"], default="FD001")
    parser.add_argument("--n_trials", type=int, default=30, help="Optuna trials for NGBoost hyperparameter search.")
    parser.add_argument("--backbone_epochs", type=int, default=150, help="Max epochs for backbone retrain (default 150).")
    parser.add_argument("--backbone_patience", type=int, default=15, help="Early-stopping patience for backbone retrain (default 15).")
    parser.add_argument("--backbone_lr", type=float, default=1e-3, help="Learning rate for backbone retrain (default 1e-3).")
    parser.add_argument(
        "--calibrate_only", action="store_true",
        help="Skip all training. Just compute and save sigma calibration scale for each dataset using the already-trained models.",
    )
    return parser.parse_args()


# ── Per-dataset full training pipeline ───────────────────────────────────────

def train_meta_for_dataset(
    dataset_key: str,
    n_trials: int,
    registry_path: Path,
    backbone_epochs: int,
    backbone_patience: int,
    backbone_lr: float,
) -> None:
    print(f"\n{'='*60}\n  {dataset_key}\n{'='*60}")

    registry = load_registry(registry_path)
    if dataset_key not in registry or "champion" not in registry[dataset_key]:
        print(f"  No champion config for {dataset_key}. Run tune.py first.")
        return

    champion = registry[dataset_key]["champion"]
    bb_name = champion["backbone"]
    bb_cfg = champion["backbone_config"]
    seq_length = bb_cfg["seq_length"]
    input_dim = bb_cfg["input_dim"]
    bb_kwargs = {k: v for k, v in bb_cfg.items() if k not in ("input_dim", "seq_length")}

    feature_path = _ROOT / "data" / "processed" / dataset_key / "train_features.csv"
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature file not found: {feature_path}")

    full_df = pd.read_csv(feature_path)
    train_df, val_df = split_by_engine(full_df)

    print(f"  Backbone : {bb_name}  seq_length={seq_length}  input_dim={input_dim}")
    print(f"  Train    : {train_df['unit_number'].nunique()} engines ({len(train_df):,} rows)")
    print(f"  Val      : {val_df['unit_number'].nunique()} engines ({len(val_df):,} rows)")

    # ── Stage 1: retrain backbone from scratch ──────────────────────────────
    print(f"\n  [Stage 1] Retraining {bb_name} backbone ({backbone_epochs} epochs max, patience={backbone_patience})...")

    model = build_backbone(bb_name, input_dim=input_dim, backbone_cfg=bb_kwargs)
    train_loader_bb = make_loader(train_df, seq_length=seq_length, batch_size=256, shuffle=True)
    val_loader_bb = make_loader(val_df, seq_length=seq_length, batch_size=256, shuffle=False)

    model, best_bb_rmse = train_backbone(
        model, train_loader_bb, val_loader_bb,
        cfg={"lr": backbone_lr, "epochs": backbone_epochs, "patience": backbone_patience},
        trial=None, verbose=True,
    )
    print(f"\n  [Stage 1] Best backbone val RMSE: {best_bb_rmse:.4f}")

    backbone_path = Path(champion["artifacts"]["backbone"])
    backbone_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), backbone_path)
    print(f"  [Stage 1] Saved → {backbone_path}")

    # ── Stage 2: extract backbone features ──────────────────────────────────
    print("\n  [Stage 2] Extracting backbone features...")

    train_loader = make_loader(train_df, seq_length=seq_length, batch_size=256, shuffle=False)
    val_loader = make_loader(val_df, seq_length=seq_length, batch_size=256, shuffle=False)
    _, y_train = create_sequences(train_df, seq_length=seq_length)
    _, y_val = create_sequences(val_df, seq_length=seq_length)
    X_train = get_backbone_features(model, train_loader)
    X_val = get_backbone_features(model, val_loader)

    print(f"  Train features : {X_train.shape}   targets: {y_train.shape}")
    print(f"  Val   features : {X_val.shape}   targets: {y_val.shape}")

    # ── Stage 3: NGBoost hyperparameter search ──────────────────────────────
    print(f"\n  [Stage 3] NGBoost Optuna search ({n_trials} trials, obj=NLL)...")

    def objective(trial: optuna.Trial) -> float:
        n_est = trial.suggest_int("n_estimators", 100, 1000, step=50)
        lr_ngb = trial.suggest_float("learning_rate", 0.005, 0.1, log=True)
        depth = trial.suggest_int("max_depth", 2, 6)
        min_smp = trial.suggest_int("min_samples_leaf", 1, 20)

        meta = NGBRegressor(
            n_estimators=n_est,
            learning_rate=lr_ngb,
            Base=DecisionTreeRegressor(max_depth=depth, min_samples_leaf=min_smp),
            verbose=False,
        )
        meta.fit(X_train, y_train)
        return float(-meta.pred_dist(X_val).logpdf(y_val).mean())

    studies_dir = _ROOT / "studies"
    studies_dir.mkdir(exist_ok=True)
    study_db = studies_dir / f"{dataset_key}_ngboost_v2.db"

    study = optuna.create_study(
        direction="minimize",
        storage=f"sqlite:///{study_db}",
        study_name=f"{dataset_key}_ngboost_search",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    print(f"  Best val NLL : {study.best_value:.4f}")
    print(f"  Best params  : {best_params}")

    # ── Stage 4: train final NGBoost + evaluate ─────────────────────────────
    print("\n  [Stage 4] Training final NGBoost...")
    final_model = NGBRegressor(
        n_estimators=best_params["n_estimators"],
        learning_rate=best_params["learning_rate"],
        Base=DecisionTreeRegressor(max_depth=best_params["max_depth"], min_samples_leaf=best_params["min_samples_leaf"]),
        verbose=False, verbose_eval=100,
    )
    final_model.fit(X_train, y_train)

    val_dists = final_model.pred_dist(X_val)
    mu, sigma = val_dists.loc, val_dists.scale
    val_nll = float(-val_dists.logpdf(y_val).mean())
    val_preds = final_model.predict(X_val)
    val_rmse = float(np.sqrt(mean_squared_error(y_val, val_preds)))
    val_mae = float(mean_absolute_error(y_val, val_preds))

    d = val_preds - y_val
    nasa = float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))

    within = {f"within_{n}_cycles": float(np.mean(np.abs(val_preds - y_val) <= n)) for n in [5, 10, 15]}

    calibration: dict[str, float] = {}
    for alpha in [0.50, 0.80, 0.90, 0.95]:
        z = norm.ppf((1 + alpha) / 2)
        in_band = (y_val >= mu - z * sigma) & (y_val <= mu + z * sigma)
        calibration[f"coverage_{int(alpha*100)}"] = float(in_band.mean())

    auc_metrics: dict[str, float] = {}
    for horizon in [20, 50]:
        binary_true = (y_val < horizon).astype(int)
        if 0 < binary_true.sum() < len(binary_true):
            probs = norm.cdf(horizon, loc=mu, scale=sigma)
            auc_metrics[f"auc_failure_{horizon}"] = float(roc_auc_score(binary_true, probs))

    pi_width_90 = float(np.mean(2 * norm.ppf(0.95) * sigma))

    print(f"\n  {'─'*40}")
    print(f"  val NLL          : {val_nll:.4f}")
    print(f"  val RMSE         : {val_rmse:.4f}")
    print(f"  val MAE          : {val_mae:.4f}")
    print(f"  NASA score       : {nasa:.1f}")
    print(f"  Mean PI width 90%: {pi_width_90:.2f} cycles")
    for k, v in within.items():
        print(f"  {k:<22}: {v*100:.1f}%")
    for k, v in auc_metrics.items():
        print(f"  {k:<22}: {v:.4f}")
    print("  Calibration (pre-scaling):")
    for k, v in calibration.items():
        expected = int(k.split("_")[1]) / 100
        flag = "✓" if abs(v - expected) < 0.05 else "✗"
        print(f"    {k}: {v:.3f}  (expected {expected:.2f})  {flag}")

    meta_path = Path(champion["artifacts"]["meta_ngboost"])
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, meta_path)
    print(f"\n  Saved NGBoost → {meta_path}")

    current = load_registry(registry_path)
    current[dataset_key]["champion"]["meta_model"] = "ngboost"
    current[dataset_key]["champion"]["meta_model_config"] = {
        "n_estimators": int(best_params["n_estimators"]),
        "learning_rate": float(best_params["learning_rate"]),
        "max_depth": int(best_params["max_depth"]),
        "min_samples_leaf": int(best_params["min_samples_leaf"]),
    }
    current[dataset_key]["champion"]["metrics"].update({
        "val_nll": val_nll, "val_rmse": val_rmse,
        "val_mae": val_mae, "nasa_score": nasa,
        "pi_width_90": pi_width_90,
        "calibration": calibration,
        **within, **auc_metrics,
    })
    current[dataset_key]["champion"]["backbone_retrain"] = {
        "epochs": backbone_epochs, "patience": backbone_patience,
        "lr": backbone_lr, "best_val_rmse": float(best_bb_rmse),
    }
    current[dataset_key]["champion"]["trained_at"] = str(datetime.date.today())
    current[dataset_key]["champion"]["optuna_study"] = str(study_db)
    save_registry(registry_path, current)
    print(f"  Registry updated → {registry_path.name}")

    # ── Stage 5: sigma calibration (always runs after full training) ───────
    calibrate_sigma(dataset_key, registry_path)


if __name__ == "__main__":
    args = parse_args()
    registry = _ROOT / "config" / "model_registry.yaml"
    datasets = ["FD001", "FD002", "FD003", "FD004"] if args.dataset == "all" else [args.dataset]

    if args.calibrate_only:
        print("Calibrate-only mode: computing sigma scale factors from existing models.")
        for ds in datasets:
            calibrate_sigma(ds, registry)
        print("\nDone. Re-run evaluate.py to see updated calibration metrics.")
    else:
        for ds in datasets:
            train_meta_for_dataset(
                dataset_key=ds,
                n_trials=args.n_trials,
                registry_path=registry,
                backbone_epochs=args.backbone_epochs,
                backbone_patience=args.backbone_patience,
                backbone_lr=args.backbone_lr,
            )