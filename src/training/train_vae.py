"""
Trains a VAE on healthy engine windows for unsupervised health monitoring.

The VAE is trained ONLY on sequences where RUL > vae_healthy_rul_threshold
(default 80), so it learns what a healthy degradation trajectory looks
like. At inference, deviations between the current encoding and the
healthy reference distribution become the health indices used by
health_monitor.py.

Three artefacts are built after training:

  healthy_reference — per-cluster FULL-COVARIANCE Gaussian describing the
                      region of latent space healthy windows occupy:
                      {mu, sigma, cov, cov_inv, cov_sqrt, cov_logdet}.
                      Built on healthy TRAINING windows.

  calibration       — drift and geometry alarm thresholds, plus the
                      empirical healthy Mahalanobis quantile grid used
                      to map raw distances onto a 0-100 health score.
                      Built on healthy VALIDATION windows.

  diagnostics       — per-dimension KL and active-unit count, so we can
                      tell at a glance whether the latent space is alive.

seq_length and input_dim are read from the champion backbone config in
model_registry.yaml so the VAE uses the same window size as the
predictive model (needed for cycle numbers to line up in the dashboard).

WHAT CHANGED IN THIS REVISION
-----------------------------
1. The healthy reference is now the COVARIANCE OF HEALTHY ENCODINGS, not
   the average posterior sigma. Those are different objects: the old one
   measured "how uncertain is the encoder about one window", the new one
   measures "how much do healthy windows differ from each other". Only
   the second is a sensible yardstick for anomaly, and using the first
   is why every geometry metric came out at ~1e-5.

2. Thresholds are calibrated on healthy VALIDATION windows, not
   training windows. Training-set reconstruction error is optimistically
   low, so a threshold fitted there fires constantly on anything else.
   This also matches the project's standing rule: validation is for
   optimisation and calibration, the test set stays untouched.

3. Thresholds are QUANTILES, not mean + 2*sigma. Reconstruction errors
   are right-skewed, so mean + 2*sigma does not correspond to the ~97.5%
   coverage the old docstring claimed. A quantile makes no distributional
   assumption and means exactly what it says.

4. Posterior diagnostics (active units, per-dimension KL) are computed
   and written to the registry, so a collapsed run is visible
   immediately rather than being inferred from a val-loss number.

Usage
-----
    python src/training/train_vae.py --dataset FD001
    python src/training/train_vae.py --dataset all
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset

_ROOT = Path(__file__).resolve().parents[2]
_TRAINING = _ROOT / "src" / "training"
_MODELS = _ROOT / "src" / "models"
_HEALTH = _ROOT / "src" / "health"
for _p in [str(_TRAINING), str(_MODELS), str(_HEALTH)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shared import DEVICE, NON_FEATURE_COLS, TARGET_COL, load_registry, save_registry, split_by_engine
from vae import VAE, vae_loss, posterior_diagnostics
from geometry import build_reference_matrices, mahalanobis


# Sequence building with cluster tracking

def create_health_sequences(df: pd.DataFrame, seq_length: int) -> tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sliding-window sequences, tracking unit_number, cycle, true_rul, and
    op_cluster (at the last timestep of each window) so healthy-reference
    lookups can be per-operating-condition in multi-condition datasets
    (FD002/FD004). Returns (X_tensor, y_array, unit_ids, cycle_ids, cluster_ids).
    """
    cols_to_drop = [c for c in NON_FEATURE_COLS if c in df.columns]
    has_cluster = "op_cluster" in df.columns
    X_seq, y_arr, unit_ids, cycle_ids, cluster_ids = [], [], [], [], []

    for unit in df["unit_number"].unique():
        udata = df[df["unit_number"] == unit].sort_values("time_in_cycles")
        X = udata.drop(columns=cols_to_drop).values
        y = udata[TARGET_COL].values
        cycles = udata["time_in_cycles"].values
        clusters = udata["op_cluster"].values if has_cluster else np.zeros(len(udata), dtype=int)

        for i in range(len(X) - seq_length + 1):
            X_seq.append(X[i : i + seq_length])
            y_arr.append(y[i + seq_length - 1])
            unit_ids.append(unit)
            cycle_ids.append(cycles[i + seq_length - 1])
            cluster_ids.append(int(clusters[i + seq_length - 1]))

    if not X_seq:
        n_feat = len(df.columns) - len(cols_to_drop)
        empty = torch.empty((0, seq_length, n_feat))
        return empty, np.array([]), np.array([]), np.array([]), np.array([])

    return (
        torch.tensor(np.array(X_seq), dtype=torch.float32),
        np.array(y_arr), np.array(unit_ids), np.array(cycle_ids), np.array(cluster_ids, dtype=int),
    )


# Beta warm-up

def get_beta(epoch: int, max_epochs: int, target_beta: float, warmup_fraction: float = 0.2) -> float:
    """
    Linearly ramp beta from 0 to target_beta over the first
    warmup_fraction of training, then hold at target_beta. Without this,
    an early KL term dominates before the decoder has learned anything,
    forcing the encoder to collapse to N(0,I) ("posterior collapse").
    Starting near 0 lets the encoder/decoder learn real structure before
    regularisation kicks in.

    This was already correct — but it was fighting an effective beta
    inflated by seq_length * input_dim from the loss reduction mismatch
    in vae.py, so it never had a chance. With that fixed, the warm-up
    does what it is supposed to.
    """
    warmup_epochs = max(1, int(max_epochs * warmup_fraction))
    if epoch < warmup_epochs:
        return target_beta * epoch / warmup_epochs
    return target_beta


# ============================================================
# Healthy Reference + Calibration
# ============================================================
#
# The VAE is being used here as a HEALTH MONITOR rather than
# as part of the RUL prediction model.
#
# The basic idea is:
#
#   1. Take data representing healthy engine behavior.
#   2. Pass that data through the trained VAE.
#   3. Establish what "healthy" looks like in latent space —
#      as a REGION (mean + covariance), not a point.
#   4. Establish how far outside that region is normal.
#   5. Later, compare new engine data against these references.
#
# This gives us two independent ways to detect abnormal behavior:
#
#   - Latent-space distance (Mahalanobis / KL / Bures-Wasserstein /
#     Fisher-Rao) — measured in the geometry of the healthy fleet
#
#   - Reconstruction error — how poorly the VAE reproduces the input
#
# Operating clusters are kept separate because engines operating
# under different conditions may naturally have different behavior.
# We therefore do not want one global definition of "healthy" for
# every operating condition.


def _encode_batched(
    vae: VAE,
    X: torch.Tensor,
    batch: int = 256
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Encode a collection of input sequences through the VAE.

    The VAE encoder does not simply return one latent vector.
    It returns the parameters of a probability distribution in
    latent space:

        mu     = mean of the latent distribution
        logvar = log of the variance of the latent distribution

    We convert logvar into sigma (standard deviation) because
    the downstream geometry calculations expect mu and sigma.

    Returns (mu, sigma, logvar) — logvar is returned as well so the
    caller can feed it straight to posterior_diagnostics() without
    re-encoding.

    Processed in batches so a large number of sequences does not have to
    fit into GPU memory simultaneously. shuffle=False matters: the
    encoded outputs must stay in the same order as X, because
    cluster_ids[i] has to keep referring to the same sample.
    """
    loader = DataLoader(TensorDataset(X), batch_size=batch, shuffle=False)

    mus, sigs, logvars = [], [], []
    vae.eval()

    with torch.no_grad():
        for (x_batch,) in loader:
            mu, logvar = vae.encode(x_batch.to(DEVICE))
            mus.append(mu.cpu().numpy())
            # variance = exp(logvar); sigma = sqrt(variance) = exp(0.5 * logvar)
            sigs.append(torch.exp(0.5 * logvar).cpu().numpy())
            logvars.append(logvar.cpu().numpy())

    return np.concatenate(mus), np.concatenate(sigs), np.concatenate(logvars)


def _recon_errors_batched(vae: VAE, X: torch.Tensor, batch: int = 256) -> np.ndarray:
    """Per-window reconstruction error for every sequence in X. Returns (N,)."""
    loader = DataLoader(TensorDataset(X), batch_size=batch, shuffle=False)
    errors = []
    vae.eval()
    with torch.no_grad():
        for (x_batch,) in loader:
            errors.append(vae.reconstruction_error(x_batch.to(DEVICE)).cpu().numpy())
    return np.concatenate(errors) if errors else np.array([])


def build_healthy_reference(
    vae: VAE,
    healthy_X: torch.Tensor,
    cluster_ids: np.ndarray,
    n_clusters: int,
    ridge: float = 1e-6,
) -> dict[int, dict]:
    """
    Build a reference REGION of healthy engine behaviour, per operating
    cluster.

        healthy sequences
              |
              v
            VAE encoder
              |
              v
        latent mu/sigma per window
              |
              v
        mean AND covariance across healthy windows
              |
              v
        healthy reference N(mu_ref, Sigma_ref)

    The covariance is the important part. It describes which directions
    in latent space healthy engines naturally vary along, which lets
    Mahalanobis distance discount benign variation and amplify the
    directions healthy engines never move in. Without it, every distance
    is Euclidean and the degradation signal — which in the collapsed run
    was a perturbation of size ~0.02 on a vector of norm ~22.75 — gets
    swamped.

    Returns {cluster_id: reference_dict}, where reference_dict has the
    keys documented in geometry.build_reference_matrices().
    """
    mu_all, sigma_all, _ = _encode_batched(vae, healthy_X)

    reference: dict[int, dict] = {}
    latent_dim = mu_all.shape[1]

    for c in range(n_clusters):
        mask = cluster_ids == c
        n_c = int(mask.sum())

        # A full covariance in latent_dim dimensions needs meaningfully
        # more than latent_dim samples to be anything but noise. Below
        # that, build_reference_matrices falls back to a diagonal
        # estimate, but it is worth flagging loudly.
        if n_c == 0:
            continue
        if n_c <= latent_dim:
            print(f"    Warning: cluster {c} has only {n_c} healthy windows "
                  f"for a {latent_dim}-dim covariance — estimate will be diagonal-only.")

        reference[c] = build_reference_matrices(
            mu_all[mask], sigma_all[mask], ridge=ridge
        )

    # Fallback handling.
    # Some clusters may contain no healthy samples at all. Without a
    # fallback, monitoring an engine assigned to one of those clusters
    # would fail because no reference exists.
    #
    # NOTE this picks the FIRST available reference, not the nearest one.
    # Doing it properly would mean measuring distance between cluster
    # centroids in operating-condition space and copying the closest.
    # Left as-is deliberately: it only triggers on datasets where a whole
    # operating condition has no healthy data, which does not happen on
    # FD001-FD004 at the default RUL>80 threshold.
    if reference:
        any_c = next(iter(reference))
        for c in range(n_clusters):
            if c not in reference:
                reference[c] = {k: (v.copy() if isinstance(v, np.ndarray) else v)
                                for k, v in reference[any_c].items()}
                print(f"    Warning: cluster {c} has no healthy samples — "
                      f"copying cluster {any_c}'s reference.")

    return reference


def build_calibration(
    vae: VAE,
    healthy_val_X: torch.Tensor,
    val_cluster_ids: np.ndarray,
    reference: dict[int, dict],
    n_clusters: int,
    drift_quantile: float = 0.99,
    geo_quantile: float = 0.99,
) -> dict:
    """
    Calibrate alarm thresholds on HEALTHY VALIDATION windows.

    Why validation rather than training:
        The model has already fit the training windows, so their
        reconstruction error is optimistically low. A threshold set
        there is systematically too tight and fires on almost
        everything, which is what produced the old behaviour of 87 of 92
        engines flagged as drifted with a median first-drift RUL sitting
        at the 125 cap — i.e. flagging on the very first window.

    Why a quantile rather than mean + 2*sigma:
        mean + 2*sigma only corresponds to ~97.5% coverage if the errors
        are Gaussian. Reconstruction errors are bounded below by zero and
        right-skewed, so they are not. An empirical quantile makes no
        distributional assumption: the 0.99 quantile of healthy error is,
        by construction, exceeded by 1% of healthy windows.

    Also stores the empirical CDF grid of healthy Mahalanobis distances,
    which health_monitor.py uses to map a raw distance onto a 0-100
    health score. Under a Gaussian model d_M^2 would be chi-squared with
    latent_dim degrees of freedom; the empirical grid is used instead
    because latent encodings are not exactly Gaussian and it costs
    nothing to be honest about that.

    Returns a dict with per-cluster and global entries:
        drift_thresholds     {cluster: float, "global": float}
        geo_thresholds       {cluster: float, "global": float}
        maha_quantile_grid   {cluster: [201 values], "global": [...]}
        quantile_levels      the 201 probability levels the grid maps to
    """
    if len(healthy_val_X) == 0:
        print("    Warning: no healthy validation windows — falling back to "
              "training-set calibration. Thresholds will be optimistic.")
        return {}

    recon = _recon_errors_batched(vae, healthy_val_X)
    mu_val, sigma_val, _ = _encode_batched(vae, healthy_val_X)

    levels = np.linspace(0.0, 1.0, 201)

    drift_thresholds: dict = {}
    geo_thresholds: dict = {}
    maha_grid: dict = {}

    # Per-cluster Mahalanobis, computed against that cluster's own reference.
    maha_all = np.full(len(mu_val), np.nan, dtype=np.float64)

    for c in range(n_clusters):
        mask = val_cluster_ids == c
        if mask.sum() < 2:
            continue

        # --- reconstruction-error drift threshold
        drift_thresholds[c] = float(np.quantile(recon[mask], drift_quantile))

        # --- geometry alarm threshold
        if c in reference:
            ref = reference[c]
            m = mahalanobis(mu_val[mask], ref["mu"], ref["cov_inv"])
            maha_all[mask] = m
            geo_thresholds[c] = float(np.quantile(m, geo_quantile))
            maha_grid[str(c)] = [float(v) for v in np.quantile(m, levels)]

    drift_thresholds["global"] = float(np.quantile(recon, drift_quantile))

    finite_maha = maha_all[np.isfinite(maha_all)]
    if len(finite_maha) > 1:
        geo_thresholds["global"] = float(np.quantile(finite_maha, geo_quantile))
        maha_grid["global"] = [float(v) for v in np.quantile(finite_maha, levels)]

    print(f"    Calibrated on {len(recon):,} healthy validation windows")
    print(f"    Drift thresholds (recon error, q={drift_quantile}):")
    for k, v in drift_thresholds.items():
        print(f"      cluster {k}: {v:.6f}")
    print(f"    Geometry thresholds (Mahalanobis, q={geo_quantile}):")
    for k, v in geo_thresholds.items():
        print(f"      cluster {k}: {v:.4f}")

    return {
        "drift_thresholds": drift_thresholds,
        "geo_thresholds": geo_thresholds,
        "maha_quantile_grid": maha_grid,
        "quantile_levels": [float(v) for v in levels],
        "drift_quantile": float(drift_quantile),
        "geo_quantile": float(geo_quantile),
        "n_calibration_windows": int(len(recon)),
        "calibrated_on": "healthy_validation",
    }


# Per-dataset training pipeline

def train_vae_for_dataset(dataset_key: str, cfg_ds: dict, registry_path: Path) -> None:
    print(f"{dataset_key} - VAE training")

    # VAE hyperparameters, tunable per dataset in datasets.yaml (all optional).
    #
    # `_cfg` records WHERE each value came from. This exists because the
    # single most confusing failure mode of this script is a hyperparameter
    # that is set in config/datasets.yaml and therefore silently overrides
    # the default written here — the code looks like it says beta=0.5 while
    # the run uses beta=0.0. Note also that these values come from
    # config/datasets.yaml, NOT from config/model_registry.yaml: the
    # registry is an OUTPUT of this script (it records what was trained),
    # so editing it can never change how training behaves.
    _sources: dict[str, str] = {}

    def _cfg(key, default, cast):
        present = key in cfg_ds
        _sources[key] = "datasets.yaml" if present else "default"
        return cast(cfg_ds[key] if present else default)

    healthy_threshold = _cfg("vae_healthy_rul_threshold", 80, int)
    hidden_dim = _cfg("vae_hidden_dim", 64, int)
    latent_dim = _cfg("vae_latent_dim", 16, int)
    num_layers = _cfg("vae_num_layers", 1, int)
    # NOTE the default. With the loss reduction fixed in vae.py, beta is on
    # the true ELBO scale, where 1.0 is the honest VAE and anything above
    # that is a beta-VAE trading reconstruction for disentanglement.
    # 0.5 gives a live posterior with good reconstruction on CMAPSS; the
    # useful search range is roughly [0.05, 1.0].
    beta = _cfg("vae_beta", 0.5, float)
    free_bits = _cfg("vae_free_bits", 0.05, float)
    max_epochs = _cfg("vae_epochs", 200, int)
    patience = _cfg("vae_patience", 15, int)
    lr = _cfg("vae_lr", 1e-3, float)
    batch_size = _cfg("vae_batch_size", 256, int)
    warmup_fraction = _cfg("vae_beta_warmup_fraction", 0.2, float)
    n_clusters = _cfg("n_clusters", 1, int)
    drift_quantile = _cfg("vae_drift_quantile", 0.99, float)
    geo_quantile = _cfg("vae_geo_quantile", 0.99, float)
    # Default raised from 1e-6 after the first real run: every dataset came
    # back with cond(cov) between 3e6 and 1.8e7. The ridge is applied
    # RELATIVE to trace(cov)/d, so 1e-4 brings that down to ~2e5.
    # A badly conditioned covariance matters because Mahalanobis divides by
    # it: the near-null directions get amplified by ~sqrt(cond), so a
    # rounding error in a dead latent direction can dominate the distance.
    cov_ridge = _cfg("vae_cov_ridge", 1e-4, float)

    # seq_length/input_dim must match the predictive backbone (registry) so
    # cycle numbers in health_indices.csv align with predictions_all.csv.
    registry = load_registry(registry_path)
    if dataset_key not in registry or "champion" not in registry[dataset_key]:
        print(f"  No champion config in registry for {dataset_key}. Run tune.py + train.py first.")
        return

    bb_cfg = registry[dataset_key]["champion"]["backbone_config"]
    seq_length = int(bb_cfg["seq_length"])
    input_dim = int(bb_cfg["input_dim"])
    artifact_dir = _ROOT / "models" / dataset_key

    print(f"  Device         : {DEVICE}")
    print(f"  seq_length     : {seq_length}  (from backbone champion)")
    print(f"  input_dim      : {input_dim}")
    print(f"  hidden_dim     : {hidden_dim}  latent_dim: {latent_dim}")
    print(f"  beta           : {beta}  [{_sources['vae_beta']}]  "
          f"(warmup fraction: {warmup_fraction} [{_sources['vae_beta_warmup_fraction']}], "
          f"free_bits: {free_bits} [{_sources['vae_free_bits']}])")

    # Anything overridden by config/datasets.yaml is listed explicitly. If a
    # value here is not what you expected, that file is where to change it —
    # editing model_registry.yaml has no effect on training, because the
    # registry is written by this script rather than read by it.
    _overridden = sorted(k for k, v in _sources.items() if v == "datasets.yaml")
    if _overridden:
        print(f"  from datasets.yaml: {', '.join(_overridden)}")
    else:
        print("  from datasets.yaml: (nothing — all values are code defaults)")
    print(f"  Healthy RUL >  : {healthy_threshold}")

    # beta comes from datasets.yaml and silently overrides this module's
    # default. Worth calling out explicitly, because beta=0 was the correct
    # workaround for the OLD loss scaling and is the wrong setting now — the
    # reduction mismatch that forced it has been fixed, so beta=0 no longer
    # buys anything and costs the entire probabilistic side of the model.
    if beta == 0.0:
        print("  WARNING: beta=0 means this is a plain autoencoder, not a VAE.")
        print("           The KL term is switched off entirely, so nothing constrains")
        print("           the posterior: sigma collapses to the logvar clamp floor")
        print("           (~0.0498) and mu is free to drift to an arbitrary scale.")
        print("           Fisher-Rao and KL will be dominated by that floor rather")
        print("           than by degradation. Set vae_beta in config/datasets.yaml")
        print("           to 0.5 (or anywhere in [0.05, 1.0]) now that the loss")
        print("           reduction is fixed.")
    if warmup_fraction == 0.0 and beta > 0.0:
        print("  WARNING: warmup_fraction=0 applies full beta from epoch 1, before the")
        print("           decoder has learned anything. Set vae_beta_warmup_fraction")
        print("           to 0.2 in config/datasets.yaml.")
    print(f"  NOTE: loss is now summed over (seq_length, input_dim). Absolute")
    print(f"        loss values are ~{seq_length * input_dim}x larger than previous runs —")
    print(f"        compare KL and active units, not total loss.")

    feature_path = _ROOT / "data" / "processed" / dataset_key / "train_features.csv"
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature file not found: {feature_path}")

    full_df = pd.read_csv(feature_path)
    train_df, val_df = split_by_engine(full_df)

    # Filter to healthy windows AFTER the engine split — otherwise val
    # engines' healthy data could leak into the training reference.
    healthy_train = train_df[train_df[TARGET_COL] > healthy_threshold].copy()
    healthy_val = val_df[val_df[TARGET_COL] > healthy_threshold].copy()

    print(f"  Train engines  : {train_df['unit_number'].nunique()} total  -> {healthy_train['unit_number'].nunique()} with healthy windows")
    print(f"  Val engines    : {val_df['unit_number'].nunique()} total  -> {healthy_val['unit_number'].nunique()} with healthy windows")

    if len(healthy_train) == 0:
        raise RuntimeError(f"No healthy training data for {dataset_key} at threshold RUL>{healthy_threshold}.")

    X_train, _, _, _, cluster_train = create_health_sequences(healthy_train, seq_length)
    X_val, _, _, _, cluster_val = create_health_sequences(healthy_val, seq_length)

    print(f"  Train sequences: {len(X_train):,}")
    print(f"  Val   sequences: {len(X_val):,}")

    train_loader = DataLoader(TensorDataset(X_train), batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(TensorDataset(X_val), batch_size=batch_size, shuffle=False, pin_memory=True)

    vae = VAE(
        input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim,
        seq_length=seq_length, num_layers=num_layers,
        dropout=0.1 if num_layers > 1 else 0.0,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, factor=0.5)

    # Training loop
    best_val_loss = float("inf")
    best_weights = None
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        current_beta = get_beta(epoch, max_epochs, beta, warmup_fraction)

        vae.train()
        train_total = train_recon = train_kl = 0.0
        for (x_batch,) in train_loader:
            x_batch = x_batch.to(DEVICE)
            optimizer.zero_grad()
            x_hat, mu, logvar = vae(x_batch)
            loss, recon, kl = vae_loss(x_batch, x_hat, mu, logvar,
                                       beta=current_beta, free_bits=free_bits)
            loss.backward()
            nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            optimizer.step()
            n = len(x_batch)
            train_total += loss.item() * n
            train_recon += recon.item() * n
            train_kl += kl.item() * n

        n_train = len(train_loader.dataset)
        train_total /= n_train
        train_recon /= n_train
        train_kl /= n_train

        vae.eval()
        val_total = val_recon = val_kl = 0.0
        with torch.no_grad():
            for (x_batch,) in val_loader:
                x_batch = x_batch.to(DEVICE)
                x_hat, mu, logvar = vae(x_batch)
                loss, recon, kl = vae_loss(x_batch, x_hat, mu, logvar,
                                           beta=current_beta, free_bits=free_bits)
                n = len(x_batch)
                val_total += loss.item() * n
                val_recon += recon.item() * n
                val_kl += kl.item() * n

        n_val = max(len(val_loader.dataset), 1)
        val_total /= n_val
        val_recon /= n_val
        val_kl /= n_val

        scheduler.step(val_total)

        print(
            f"  Epoch {epoch+1:>3}/{max_epochs}  beta={current_beta:.3f}  "
            f"train [total={train_total:.2f} recon={train_recon:.2f} kl={train_kl:.3f}]  "
            f"val [total={val_total:.2f} recon={val_recon:.2f} kl={val_kl:.3f}]"
        )

        # KL sitting at the free-bits floor means every dimension is
        # riding its allowance and none is carrying information.
        floor = free_bits * latent_dim
        if epoch > 10 and val_kl <= floor * 1.05:
            print(f"  KL pinned at the free-bits floor ({floor:.3f}) — the posterior "
                  f"is not using the latent space. Lower beta or raise warmup_fraction.")

        # Early stopping is on the total loss, but note this is a moving
        # target during warm-up because beta is still changing. Only
        # start counting once beta has reached its target value,
        # otherwise a decreasing-then-increasing loss triggers a stop for
        # the wrong reason.
        warmup_epochs = max(1, int(max_epochs * warmup_fraction))
        if epoch < warmup_epochs:
            best_val_loss = val_total
            best_weights = {k: v.cpu().clone() for k, v in vae.state_dict().items()}
            continue

        if val_total < best_val_loss:
            best_val_loss = val_total
            epochs_no_improve = 0
            best_weights = {k: v.cpu().clone() for k, v in vae.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  Early stopping at epoch {epoch+1} (patience={patience})")
                break

    vae.load_state_dict(best_weights)
    print(f"\n  Best val loss: {best_val_loss:.4f}")

    # --- Posterior diagnostics: is the latent space actually alive?
    print("\n  Posterior diagnostics (healthy validation windows)")
    if len(X_val) > 0:
        mu_v, _, logvar_v = _encode_batched(vae, X_val)
        diagnostics = posterior_diagnostics(
            torch.from_numpy(mu_v), torch.from_numpy(logvar_v)
        )
    else:
        mu_v, _, logvar_v = _encode_batched(vae, X_train)
        diagnostics = posterior_diagnostics(
            torch.from_numpy(mu_v), torch.from_numpy(logvar_v)
        )
        diagnostics["note"] = "computed on training windows (no healthy validation data)"

    print(f"    Active units      : {diagnostics['active_units']} / {diagnostics['latent_dim']}"
          f"  ({diagnostics['active_fraction']:.0%})")
    print(f"    Total KL          : {diagnostics['total_kl']:.3f} nats")
    print(f"    ||mu|| mean/std   : {diagnostics['mu_norm_mean']:.3f} / {diagnostics['mu_norm_std']:.4f}")
    print(f"    mean sigma        : {diagnostics['sigma_mean']:.4f}")
    print(f"    per-dim KL        : "
          + ", ".join(f"{v:.3f}" for v in diagnostics["per_dim_kl"]))

    if diagnostics["active_units"] == 0:
        print("    LATENT SPACE IS DEAD — no dimension responds to the input. "
              "The geometry indices will be meaningless. Lower beta and retrain.")
    elif diagnostics["active_units"] < 3:
        print("    Very few active units. Geometry indices will be weak. "
              "Consider lowering beta.")
    # A large ||mu|| with a tiny std is the specific failure signature
    # from the original run: the encoder emits a near-constant vector and
    # the degradation signal hides in the trailing decimals.
    if diagnostics["mu_norm_mean"] > 0 and \
       diagnostics["mu_norm_std"] / max(diagnostics["mu_norm_mean"], 1e-9) < 1e-2:
        print("    ||mu|| is essentially constant across inputs — the encoder is "
              "ignoring its input even if per-dim KL looks non-zero.")

    # Both reference and calibration are built without ever touching the test set.
    # Reference: healthy TRAINING windows. Calibration: healthy VALIDATION windows.
    print("\n  Building healthy reference distributions (full covariance)")
    reference = build_healthy_reference(vae, X_train, cluster_train, n_clusters, ridge=cov_ridge)
    for c, ref in reference.items():
        cond = np.linalg.cond(ref["cov"])
        print(f"    cluster {c}: n={ref['n']:,}  ||mu_ref||={np.linalg.norm(ref['mu']):.4f}  "
              f"tr(cov)={np.trace(ref['cov']):.4f}  cond(cov)={cond:.1f}")
        if cond > 1e6:
            print(f"      Warning: covariance is poorly conditioned. Raise vae_cov_ridge.")

    print("\n  Calibrating thresholds on healthy validation windows")
    calibration = build_calibration(
        vae, X_val, cluster_val, reference, n_clusters,
        drift_quantile=drift_quantile, geo_quantile=geo_quantile,
    )

    # If there was no healthy validation data, fall back to training-set
    # calibration so the pipeline still produces usable artefacts.
    if not calibration:
        calibration = build_calibration(
            vae, X_train, cluster_train, reference, n_clusters,
            drift_quantile=drift_quantile, geo_quantile=geo_quantile,
        )
        calibration["calibrated_on"] = "healthy_train_fallback"

    # Save artefacts
    artifact_dir.mkdir(parents=True, exist_ok=True)

    torch.save(vae.state_dict(), artifact_dir / "vae.pt")
    print(f"\n  Saved vae.pt -> {artifact_dir}")

    vae_cfg_save = {
        "input_dim": input_dim, "hidden_dim": hidden_dim, "latent_dim": latent_dim,
        "seq_length": seq_length, "num_layers": num_layers,
        "dropout": 0.1 if num_layers > 1 else 0.0,
    }
    joblib.dump(vae_cfg_save, artifact_dir / "vae_config.pkl")

    # Reference is saved as numpy arrays (joblib handles these natively).
    # health_monitor.py accepts both this and the older list-of-floats
    # {mu, sigma} format, so an un-retrained dataset keeps working.
    reference_saveable = {
        c: {
            "mu": ref["mu"], "sigma": ref["sigma"], "cov": ref["cov"],
            "cov_inv": ref["cov_inv"], "cov_sqrt": ref["cov_sqrt"],
            "cov_logdet": ref["cov_logdet"], "n": ref["n"],
        }
        for c, ref in reference.items()
    }
    joblib.dump(reference_saveable, artifact_dir / "vae_healthy_reference.pkl")

    # vae_drift_thresholds.pkl keeps its filename and its {cluster: float,
    # "global": float} shape so nothing downstream breaks, and the richer
    # calibration lands in a new sibling file.
    joblib.dump(calibration["drift_thresholds"], artifact_dir / "vae_drift_thresholds.pkl")
    joblib.dump(calibration, artifact_dir / "vae_calibration.pkl")
    joblib.dump(diagnostics, artifact_dir / "vae_diagnostics.pkl")

    print(f"  Saved vae_config.pkl, vae_healthy_reference.pkl, vae_drift_thresholds.pkl,")
    print(f"        vae_calibration.pkl, vae_diagnostics.pkl -> {artifact_dir}")

    current = load_registry(registry_path)
    current[dataset_key]["vae"] = {
        "trained_at": str(datetime.date.today()),
        "best_val_loss": float(best_val_loss),
        "config": vae_cfg_save,
        "beta": float(beta),
        "free_bits": float(free_bits),
        "healthy_rul_threshold": healthy_threshold,
        "n_clusters": n_clusters,
        "loss_reduction": "sum_over_seq_and_features",   # marks post-fix runs
        "geometry_version": 2,                            # full-covariance reference
        "posterior": {
            "active_units": diagnostics["active_units"],
            "latent_dim": diagnostics["latent_dim"],
            "total_kl": diagnostics["total_kl"],
            "mu_norm_mean": diagnostics["mu_norm_mean"],
            "mu_norm_std": diagnostics["mu_norm_std"],
            "sigma_mean": diagnostics["sigma_mean"],
            "per_dim_kl": diagnostics["per_dim_kl"],
            "per_dim_mu_var": diagnostics["per_dim_mu_var"],
        },
        "calibration": {
            "calibrated_on": calibration.get("calibrated_on"),
            "drift_quantile": calibration.get("drift_quantile"),
            "geo_quantile": calibration.get("geo_quantile"),
            "n_calibration_windows": calibration.get("n_calibration_windows"),
        },
        "artifacts": {
            "vae": str(artifact_dir / "vae.pt"),
            "config": str(artifact_dir / "vae_config.pkl"),
            "reference": str(artifact_dir / "vae_healthy_reference.pkl"),
            "thresholds": str(artifact_dir / "vae_drift_thresholds.pkl"),
            "calibration": str(artifact_dir / "vae_calibration.pkl"),
            "diagnostics": str(artifact_dir / "vae_diagnostics.pkl"),
        },
    }
    save_registry(registry_path, current)
    print(f"  Registry updated -> vae section added for {dataset_key}")


class _StrictLoader(yaml.SafeLoader):
    """
    SafeLoader that refuses duplicate keys inside a mapping.

    Plain yaml.safe_load accepts

        vae_beta_warmup_fraction: 0.2
        ...
        vae_beta_warmup_fraction: 0.0

    without a word of complaint and silently keeps the LAST one. That is
    valid per the YAML spec (a mapping is built by successive assignment)
    but it is a genuinely dangerous default for a config file that gets
    edited by appending: the value you can see at the top of the block is
    not the value that runs, and nothing anywhere reports the difference.

    Raising here turns a silent wrong-hyperparameter run into a one-line
    error that names the key and the file.
    """

    def construct_mapping(self, node, deep=False):
        seen = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                raise yaml.constructor.ConstructorError(
                    None, None,
                    f"duplicate key {key!r} in mapping — the later value silently "
                    f"overrides the earlier one. Delete one of them.",
                    key_node.start_mark,
                )
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


_StrictLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _StrictLoader.construct_mapping,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VAE for health monitoring on healthy engine data.")
    parser.add_argument("--dataset", choices=["FD001", "FD002", "FD003", "FD004", "all"], default="FD001")
    args = parser.parse_args()

    registry_path = _ROOT / "config" / "model_registry.yaml"
    datasets_path = _ROOT / "config" / "datasets.yaml"

    with open(datasets_path, "r") as f:
        all_cfg = yaml.load(f, Loader=_StrictLoader) or {}

    datasets = ["FD001", "FD002", "FD003", "FD004"] if args.dataset == "all" else [args.dataset]

    for ds in datasets:
        train_vae_for_dataset(ds, all_cfg.get(ds, {}), registry_path)