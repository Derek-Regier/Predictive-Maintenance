"""
src/training/train_vae.py

Trains a VAE on healthy engine windows for unsupervised health monitoring.

The VAE is trained ONLY on sequences where RUL > vae_healthy_rul_threshold
(default 80). It learns what a healthy degradation trajectory looks like.
At inference, deviations between the current encoding and the healthy
reference distribution become the information-geometric health indices
used by health_monitor.py.

After training, two artefacts are built from the healthy training data:

  healthy_reference  — per-cluster (mu_ref, sigma_ref): the mean encoded
                       distribution of healthy engine windows. Used as
                       the comparison point for KL/JS/Wasserstein distances.

  drift_thresholds   — per-cluster reconstruction error at mean + 2σ.
                       Reconstruction error above this threshold on new
                       data signals that the engine is outside the healthy
                       distribution the VAE was trained on.

Hyperparameters come from config/datasets.yaml (see NEXT_STEPS.md for
the full list to add). seq_length and input_dim are read from the champion
backbone config in model_registry.yaml so the VAE uses the same window
size as the predictive model — necessary for cycle numbers to align in
the dashboard.

Usage
-----
    python src/training/train_vae.py --dataset FD001
    python src/training/train_vae.py --dataset all
"""

from __future__ import annotations

import argparse
import datetime
import shutil
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

# ── Path resolution ───────────────────────────────────────────────────────────
_ROOT     = Path(__file__).resolve().parents[2]
_TRAINING = _ROOT / "src" / "training"
_MODELS   = _ROOT / "src" / "models"

for _p in [str(_TRAINING), str(_MODELS)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shared import DEVICE, NON_FEATURE_COLS, TARGET_COL, split_by_engine
from vae import VAE, vae_loss


# ── Serialisation helpers ─────────────────────────────────────────────────────

def _to_python(obj):
    """Recursively convert numpy scalars/arrays to native Python types."""
    if isinstance(obj, dict):             return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):             return [_to_python(v) for v in obj]
    if isinstance(obj, np.integer):       return int(obj)
    if isinstance(obj, np.floating):      return float(obj)
    if isinstance(obj, np.ndarray):       return obj.tolist()
    return obj

def _safe_yaml_write(path: Path, data: dict) -> None:
    """Atomic YAML write via .tmp rename."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        yaml.safe_dump(_to_python(data), f, default_flow_style=False, sort_keys=False)
    shutil.move(str(tmp), str(path))


# ── Sequence construction with cluster tracking ───────────────────────────────

def create_health_sequences(
    df: pd.DataFrame,
    seq_length: int,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sliding-window sequences from df, tracking unit_number, cycle,
    true_rul, and the op_cluster at the last timestep of each window.

    op_cluster is tracked per-window so healthy reference lookups can be
    per-operating-condition in multi-condition datasets (FD002/FD004).

    Returns: (X_tensor, y_array, unit_ids, cycle_ids, cluster_ids)
    """
    X_seq, y_arr, unit_ids, cycle_ids, cluster_ids = [], [], [], [], []
    cols_to_drop = [c for c in NON_FEATURE_COLS if c in df.columns]
    has_cluster  = "op_cluster" in df.columns

    for unit in df["unit_number"].unique():
        udata    = df[df["unit_number"] == unit].sort_values("time_in_cycles")
        X        = udata.drop(columns=cols_to_drop).values
        y        = udata[TARGET_COL].values
        cycles   = udata["time_in_cycles"].values
        clusters = udata["op_cluster"].values if has_cluster else np.zeros(len(udata), dtype=int)

        if len(X) >= seq_length:
            for i in range(len(X) - seq_length + 1):
                X_seq.append(X[i : i + seq_length])
                y_arr.append(y[i + seq_length - 1])
                unit_ids.append(unit)
                cycle_ids.append(cycles[i + seq_length - 1])
                cluster_ids.append(int(clusters[i + seq_length - 1]))

    if not X_seq:
        n_feat = len(df.columns) - len(cols_to_drop)
        empty  = torch.empty((0, seq_length, n_feat))
        return empty, np.array([]), np.array([]), np.array([]), np.array([])

    return (
        torch.tensor(np.array(X_seq), dtype=torch.float32),
        np.array(y_arr),
        np.array(unit_ids),
        np.array(cycle_ids),
        np.array(cluster_ids, dtype=int),
    )


# ── Beta warm-up ──────────────────────────────────────────────────────────────

def get_beta(epoch: int, max_epochs: int, target_beta: float,
             warmup_fraction: float = 0.2) -> float:
    """
    Linearly ramp β from 0 to target_beta over the first warmup_fraction
    of training, then hold at target_beta.

    Why warm-up?
    ------------
    At the start of training the decoder is uninitialised and produces poor
    reconstructions. If β is already at 1.0 the KL term dominates, forcing
    the encoder to map everything to N(0,I) before the decoder has a chance
    to learn — this is "posterior collapse". Starting β at 0 (pure
    autoencoder) lets both encoder and decoder learn meaningful
    representations before regularisation kicks in.
    """
    warmup_epochs = max(1, int(max_epochs * warmup_fraction))
    if epoch < warmup_epochs:
        return target_beta * epoch / warmup_epochs
    return target_beta


# ── Healthy reference + drift thresholds ─────────────────────────────────────

def _encode_batched(
    vae:     VAE,
    X:       torch.Tensor,
    batch:   int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Encode a tensor of sequences in batches without gradients.
    Returns (mu_all, sigma_all) both (N, latent_dim) numpy arrays.
    sigma = exp(0.5 * logvar), not logvar — geometry.py expects sigma.
    """
    loader   = DataLoader(TensorDataset(X), batch_size=batch, shuffle=False)
    mus, sigs = [], []
    vae.eval()
    with torch.no_grad():
        for (x_batch,) in loader:
            mu, logvar = vae.encode(x_batch.to(DEVICE))
            mus.append(mu.cpu().numpy())
            sigs.append(torch.exp(0.5 * logvar).cpu().numpy())
    return np.concatenate(mus), np.concatenate(sigs)


def build_healthy_reference(
    vae:          VAE,
    healthy_X:    torch.Tensor,
    cluster_ids:  np.ndarray,
    n_clusters:   int,
) -> dict[int, dict]:
    """
    Compute per-cluster reference distributions from healthy training windows.

    For each operating cluster c:
      mu_ref[c]    = mean of mu vectors for all healthy windows in cluster c
      sigma_ref[c] = mean of sigma vectors (same windows)

    This point in latent space represents "what healthy looks like at this
    operating condition". Health monitor computes KL/JS/Wasserstein between
    an engine's current encoding and this reference.
    """
    mu_all, sigma_all = _encode_batched(vae, healthy_X)
    reference = {}

    for c in range(n_clusters):
        mask = cluster_ids == c
        if mask.sum() == 0:
            continue
        reference[c] = {
            "mu":    mu_all[mask].mean(axis=0),     # (latent_dim,)
            "sigma": sigma_all[mask].mean(axis=0),  # (latent_dim,)
        }

    # Fallback for clusters with no healthy samples
    if reference:
        any_ref = next(iter(reference.values()))
        for c in range(n_clusters):
            if c not in reference:
                reference[c] = {
                    "mu":    any_ref["mu"].copy(),
                    "sigma": any_ref["sigma"].copy(),
                }
                print(f"  Warning: cluster {c} has no healthy samples — "
                      "using nearest cluster reference.")

    return reference


def build_drift_thresholds(
    vae:         VAE,
    healthy_X:   torch.Tensor,
    cluster_ids: np.ndarray,
    n_clusters:  int,
) -> dict:
    """
    Compute reconstruction error distribution on healthy training data
    and set drift threshold = mean + 2σ per cluster.

    When an engine's reconstruction error exceeds this threshold the VAE
    is struggling to reconstruct its sensor pattern from the healthy
    latent space — a strong signal that something has changed.

    mean + 2σ covers ~97.5% of healthy behaviour assuming approximately
    Gaussian error distribution. A lower multiplier (e.g. 1.5σ) would
    be more sensitive; a higher one (3σ) more conservative.
    """
    loader  = DataLoader(TensorDataset(healthy_X), batch_size=256, shuffle=False)
    errors  = []
    vae.eval()
    with torch.no_grad():
        for (x_batch,) in loader:
            errors.append(
                vae.reconstruction_error(x_batch.to(DEVICE)).cpu().numpy()
            )
    errors = np.concatenate(errors)

    thresholds: dict = {}
    for c in range(n_clusters):
        mask = cluster_ids == c
        if mask.sum() < 2:
            continue
        ec = errors[mask]
        thresholds[c] = float(ec.mean() + 2.0 * ec.std())

    # Global fallback (mean + 2σ over all healthy data)
    thresholds["global"] = float(errors.mean() + 2.0 * errors.std())

    print(f"  Drift thresholds (recon error mean+2σ):")
    for k, v in thresholds.items():
        print(f"    cluster {k}: {v:.6f}")

    return thresholds


# ── Per-dataset training pipeline ─────────────────────────────────────────────

def train_vae_for_dataset(
    dataset_key:   str,
    cfg_ds:        dict,
    registry_path: Path,
) -> None:
    print(f"\n{'='*60}")
    print(f"  {dataset_key} — VAE training")
    print(f"{'='*60}")

    # ── VAE hyperparameters from datasets.yaml ────────────────────────────────
    # These can be tuned per dataset. Sensible defaults are provided so the
    # yaml additions are optional — train_vae.py works without them.
    healthy_threshold  = int(cfg_ds.get("vae_healthy_rul_threshold", 80))
    hidden_dim         = int(cfg_ds.get("vae_hidden_dim",  64))
    latent_dim         = int(cfg_ds.get("vae_latent_dim",  16))
    num_layers         = int(cfg_ds.get("vae_num_layers",   1))
    beta               = float(cfg_ds.get("vae_beta",       1.0))
    max_epochs         = int(cfg_ds.get("vae_epochs",      100))
    patience           = int(cfg_ds.get("vae_patience",     10))
    lr                 = float(cfg_ds.get("vae_lr",        1e-3))
    batch_size         = int(cfg_ds.get("vae_batch_size",  256))
    warmup_fraction    = float(cfg_ds.get("vae_beta_warmup_fraction", 0.2))
    n_clusters         = int(cfg_ds.get("n_clusters",        1))

    # ── seq_length + input_dim from backbone champion in registry ─────────────
    # The VAE must use the same seq_length as the predictive backbone so that
    # cycle numbers in health_indices.csv align with predictions_all.csv.
    with open(registry_path, "r") as f:
        registry = yaml.safe_load(f) or {}

    if dataset_key not in registry or "champion" not in registry[dataset_key]:
        print(f"  No champion config in registry for {dataset_key}. "
              "Run tune.py + train.py first.")
        return

    bb_cfg     = registry[dataset_key]["champion"]["backbone_config"]
    seq_length = int(bb_cfg["seq_length"])
    input_dim  = int(bb_cfg["input_dim"])
    artifact_dir = _ROOT / "models" / dataset_key

    print(f"  Device         : {DEVICE}")
    print(f"  seq_length     : {seq_length}  (from backbone champion)")
    print(f"  input_dim      : {input_dim}")
    print(f"  hidden_dim     : {hidden_dim}  latent_dim: {latent_dim}")
    print(f"  beta           : {beta}  (warmup fraction: {warmup_fraction})")
    print(f"  Healthy RUL >  : {healthy_threshold}")

    # ── Load data ─────────────────────────────────────────────────────────────
    feature_path = _ROOT / "data" / "processed" / dataset_key / "train_features.csv"
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature file not found: {feature_path}")

    full_df          = pd.read_csv(feature_path)
    train_df, val_df = split_by_engine(full_df)

    # Filter to healthy windows AFTER the engine split — critical to prevent
    # val engines' healthy data from leaking into the training reference.
    healthy_train = train_df[train_df[TARGET_COL] > healthy_threshold].copy()
    healthy_val   = val_df[val_df[TARGET_COL]   > healthy_threshold].copy()

    print(f"  Train engines  : {train_df['unit_number'].nunique()} total  → "
          f"{healthy_train['unit_number'].nunique()} with healthy windows")
    print(f"  Val engines    : {val_df['unit_number'].nunique()} total  → "
          f"{healthy_val['unit_number'].nunique()} with healthy windows")

    if len(healthy_train) == 0:
        raise RuntimeError(
            f"No healthy training data for {dataset_key} at threshold RUL>{healthy_threshold}."
        )

    # ── Build sequence datasets ───────────────────────────────────────────────
    X_train, _, _, _, cluster_train = create_health_sequences(healthy_train, seq_length)
    X_val,   _, _, _, _             = create_health_sequences(healthy_val,   seq_length)

    print(f"  Train sequences: {len(X_train):,}")
    print(f"  Val   sequences: {len(X_val):,}")

    train_loader = DataLoader(
        TensorDataset(X_train), batch_size=batch_size, shuffle=True, pin_memory=True
    )
    val_loader   = DataLoader(
        TensorDataset(X_val),   batch_size=batch_size, shuffle=False, pin_memory=True
    )

    # ── Instantiate VAE ───────────────────────────────────────────────────────
    vae = VAE(
        input_dim  = input_dim,
        hidden_dim = hidden_dim,
        latent_dim = latent_dim,
        seq_length = seq_length,
        num_layers = num_layers,
        dropout    = 0.1 if num_layers > 1 else 0.0,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=4, factor=0.5
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss     = float("inf")
    best_weights      = None
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        current_beta = get_beta(epoch, max_epochs, beta, warmup_fraction)

        # Train pass
        vae.train()
        train_total = train_recon = train_kl = 0.0
        for (x_batch,) in train_loader:
            x_batch = x_batch.to(DEVICE)
            optimizer.zero_grad()
            x_hat, mu, logvar = vae(x_batch)
            loss, recon, kl   = vae_loss(x_batch, x_hat, mu, logvar, beta=current_beta)
            loss.backward()
            nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            optimizer.step()
            n = len(x_batch)
            train_total += loss.item() * n
            train_recon += recon.item() * n
            train_kl    += kl.item() * n

        n_train      = len(train_loader.dataset)
        train_total /= n_train
        train_recon /= n_train
        train_kl    /= n_train

        # Val pass
        vae.eval()
        val_total = val_recon = val_kl = 0.0
        with torch.no_grad():
            for (x_batch,) in val_loader:
                x_batch = x_batch.to(DEVICE)
                x_hat, mu, logvar = vae(x_batch)
                loss, recon, kl   = vae_loss(x_batch, x_hat, mu, logvar, beta=current_beta)
                n = len(x_batch)
                val_total += loss.item() * n
                val_recon += recon.item() * n
                val_kl    += kl.item() * n

        n_val      = max(len(val_loader.dataset), 1)
        val_total /= n_val
        val_recon /= n_val
        val_kl    /= n_val

        scheduler.step(val_total)

        print(
            f"  Epoch {epoch+1:>3}/{max_epochs}  β={current_beta:.3f}  "
            f"train [total={train_total:.4f} recon={train_recon:.4f} kl={train_kl:.4f}]  "
            f"val [total={val_total:.4f} recon={val_recon:.4f} kl={val_kl:.4f}]"
        )

        # Posterior collapse warning — KL near zero means the encoder ignores input
        if epoch > 10 and val_kl < 0.01:
            print("  ⚠ KL ≈ 0: possible posterior collapse. "
                  "Consider lowering beta or increasing warmup_fraction.")

        # Early stopping on total val loss
        if val_total < best_val_loss:
            best_val_loss     = val_total
            epochs_no_improve = 0
            best_weights      = {k: v.cpu().clone() for k, v in vae.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  Early stopping at epoch {epoch+1} (patience={patience})")
                break

    vae.load_state_dict(best_weights)
    print(f"\n  Best val loss: {best_val_loss:.6f}")

    # ── Build healthy reference + drift thresholds ────────────────────────────
    # Both use TRAINING data only — the val split is untouched.
    print("\n  Building healthy reference distributions...")
    reference = build_healthy_reference(vae, X_train, cluster_train, n_clusters)
    for c, ref in reference.items():
        print(f"    cluster {c}: "
              f"mu_ref norm={np.linalg.norm(ref['mu']):.4f}  "
              f"sigma_ref mean={ref['sigma'].mean():.4f}")

    print("\n  Building drift thresholds...")
    thresholds = build_drift_thresholds(vae, X_train, cluster_train, n_clusters)

    # ── Save artefacts ────────────────────────────────────────────────────────
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # VAE weights
    torch.save(vae.state_dict(), artifact_dir / "vae.pt")
    print(f"\n  Saved vae.pt → {artifact_dir}")

    # VAE config — needed by health_monitor.py to rebuild the architecture
    vae_cfg_save = {
        "input_dim":  input_dim,
        "hidden_dim": hidden_dim,
        "latent_dim": latent_dim,
        "seq_length": seq_length,
        "num_layers": num_layers,
        "dropout":    0.1 if num_layers > 1 else 0.0,
    }
    joblib.dump(vae_cfg_save, artifact_dir / "vae_config.pkl")

    # Healthy reference — convert numpy arrays to lists for joblib compatibility
    reference_saveable = {
        c: {"mu": ref["mu"].tolist(), "sigma": ref["sigma"].tolist()}
        for c, ref in reference.items()
    }
    joblib.dump(reference_saveable, artifact_dir / "vae_healthy_reference.pkl")

    # Drift thresholds
    joblib.dump(thresholds, artifact_dir / "vae_drift_thresholds.pkl")

    print(f"  Saved vae_config.pkl, vae_healthy_reference.pkl, "
          f"vae_drift_thresholds.pkl → {artifact_dir}")

    # ── Update registry ───────────────────────────────────────────────────────
    with open(registry_path, "r") as f:
        current = yaml.safe_load(f) or {}

    current[dataset_key]["vae"] = {
        "trained_at":    str(datetime.date.today()),
        "best_val_loss": float(best_val_loss),
        "config":        vae_cfg_save,
        "healthy_rul_threshold": healthy_threshold,
        "n_clusters":    n_clusters,
        "artifacts": {
            "vae":            str(artifact_dir / "vae.pt"),
            "config":         str(artifact_dir / "vae_config.pkl"),
            "reference":      str(artifact_dir / "vae_healthy_reference.pkl"),
            "thresholds":     str(artifact_dir / "vae_drift_thresholds.pkl"),
        },
    }

    _safe_yaml_write(registry_path, current)
    print(f"  Registry updated → vae section added for {dataset_key}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train VAE for health monitoring on healthy engine data."
    )
    parser.add_argument(
        "--dataset",
        choices=["FD001", "FD002", "FD003", "FD004", "all"],
        default="FD001",
    )
    args = parser.parse_args()

    registry_path = _ROOT / "config" / "model_registry.yaml"
    datasets_path = _ROOT / "config" / "datasets.yaml"

    with open(datasets_path, "r") as f:
        all_cfg = yaml.safe_load(f) or {}

    datasets = (
        ["FD001", "FD002", "FD003", "FD004"]
        if args.dataset == "all" else [args.dataset]
    )

    for ds in datasets:
        cfg_ds = all_cfg.get(ds, {})
        train_vae_for_dataset(ds, cfg_ds, registry_path)