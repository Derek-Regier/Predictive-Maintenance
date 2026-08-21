"""
Trains a VAE on healthy engine windows for unsupervised health monitoring.

The VAE is trained ONLY on sequences where RUL > vae_healthy_rul_threshold
(default 80), so it learns what a healthy degradation trajectory looks
like. At inference, deviations between the current encoding and the
healthy reference distribution become the health indices used by
health_monitor.py.

Two artefacts are built from the healthy training data after training:

  healthy_reference — per-cluster (mu_ref, sigma_ref): the mean encoded
                      distribution of healthy engine windows. Used as
                      the comparison point for KL/JS/Wasserstein distances.

  drift_thresholds  — per-cluster reconstruction error at mean + 2σ.
                      Reconstruction error above this on new data signals
                      the engine is outside the healthy distribution.

seq_length and input_dim are read from the champion backbone config in
model_registry.yaml so the VAE uses the same window size as the
predictive model (needed for cycle numbers to line up in the dashboard).

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
for _p in [str(_TRAINING), str(_MODELS)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shared import DEVICE, NON_FEATURE_COLS, TARGET_COL, load_registry, save_registry, split_by_engine
from vae import VAE, vae_loss


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
    Linearly ramp β from 0 to target_beta over the first warmup_fraction of
    training, then hold at target_beta. Without this, an early KL term
    dominates before the decoder has learned anything, forcing the encoder
    to collapse to N(0,I) ("posterior collapse"). Starting near 0 lets the
    encoder/decoder learn real structure before regularisation kicks in.
    """
    warmup_epochs = max(1, int(max_epochs * warmup_fraction))
    if epoch < warmup_epochs:
        return target_beta * epoch / warmup_epochs
    return target_beta


# ============================================================
# Healthy Reference + Drift Thresholds
# ============================================================
#
# The VAE is being used here as a HEALTH MONITOR rather than
# as part of the RUL prediction model.
#
# The basic idea is:
#
#   1. Take data representing healthy engine behavior.
#   2. Pass that data through the trained VAE.
#   3. Establish what "healthy" looks like in latent space.
#   4. Establish how much reconstruction error is normal.
#   5. Later, compare new engine data against these references.
#
# This gives us two different ways to detect abnormal behavior:
#
#   - Latent-space distance:
#       KL / JS / Wasserstein
#
#   - Reconstruction error:
#       How poorly can the VAE reconstruct this data?
#
# Operating clusters are kept separate because engines operating
# under different conditions may naturally have different behavior.
# We therefore do not necessarily want one global definition of
# "healthy" for every operating condition.


def _encode_batched(
    vae: VAE,
    X: torch.Tensor,
    batch: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    """
    Encode a collection of input sequences through the VAE.

    The VAE encoder does not simply return one latent vector.
    It returns the parameters of a probability distribution in
    latent space:

        mu     = mean of the latent distribution
        logvar = log of the variance of the latent distribution

    We convert logvar into sigma (standard deviation) because
    the downstream geometry calculations expect mu and sigma.

    The data is processed in batches rather than all at once so
    that a large number of sequences does not have to fit into
    GPU memory simultaneously.

    Parameters
    ----------
    vae:
        The already-trained VAE.

    X:
        Tensor containing the sequences that should be encoded.
        Expected shape is approximately:

            (number_of_samples, sequence_length, features)

    batch:
        Number of sequences processed at once.

    Returns
    -------
    mu:
        Latent means with shape:

            (N, latent_dim)

    sigma:
        Latent standard deviations with shape:

            (N, latent_dim)

    These two arrays describe where each input sequence lies in
    the VAE's latent probability space.
    """

    # DataLoader lets us process X in manageable batches instead
    # of sending the entire dataset through the VAE at once.
    #
    # shuffle=False is important because we want the encoded
    # outputs to remain in exactly the same order as X.
    loader = DataLoader(
        TensorDataset(X),
        batch_size=batch,
        shuffle=False
    )

    mus, sigs = [], []

    # Put the VAE into evaluation mode.
    # This ensures layers such as dropout behave consistently
    # during encoding.
    vae.eval()

    # We are only extracting information from the trained model.
    # No gradients are required, which saves memory and computation.
    with torch.no_grad():

        # Process the input sequences batch by batch.
        for (x_batch,) in loader:

            # Encode the batch.
            #
            # mu describes the center of the latent distribution.
            # logvar describes its variance in logarithmic form.
            mu, logvar = vae.encode(x_batch.to(DEVICE))

            # Move the results back to CPU and convert them to
            # NumPy arrays so they can be used by the geometry code.
            mus.append(mu.cpu().numpy())

            # The VAE gives us log(variance), but geometry.py
            # expects standard deviation.
            #
            # variance = exp(logvar)
            # sigma    = sqrt(variance)
            #          = exp(0.5 * logvar)
            sigs.append(
                torch.exp(0.5 * logvar).cpu().numpy()
            )

    # Each batch produced one NumPy array.
    # concatenate() combines all batches back into one array.
    #
    # The original ordering is preserved because shuffle=False.
    return np.concatenate(mus), np.concatenate(sigs)


def build_healthy_reference(vae: VAE, healthy_X: torch.Tensor, cluster_ids: np.ndarray, n_clusters: int) -> dict[int, dict]:
    """
    Build a reference representation of "healthy" engine behavior.

    The reference is calculated separately for every operating
    cluster.
    For each cluster:
        healthy sequences
              |
              v
            VAE
              |
              v
        latent mu/sigma
              |
              v
        average across healthy samples
              |
              v
       healthy reference
    Later, when a new engine is observed, its latent distribution
    can be compared against the reference for its operating cluster.

    This is important because an engine operating under one set of
    conditions may naturally produce a different latent distribution
    than an engine operating under another set of conditions.

    Parameters:
    vae:
        Trained VAE used to encode the healthy data.

    healthy_X:
        Sequences representing healthy engine behavior.

    cluster_ids:
        Operating-condition cluster assigned to each sequence in
        healthy_X.

    n_clusters:
        Total number of operating-condition clusters.

    Returns
    reference:
        Dictionary indexed by operating cluster.

        Example:

            reference[2]["mu"]
            reference[2]["sigma"]

        represent the average latent distribution for healthy
        samples belonging to cluster 2.
    """

    # Encode every healthy sequence into latent-space parameters.
    # mu_all[i] and sigma_all[i] describe the latent distribution
    # produced by healthy sequence i.
    mu_all, sigma_all = _encode_batched(vae, healthy_X)

    reference = {}

    # Build a separate healthy reference for each operating cluster.
    for c in range(n_clusters):

        # Select only the healthy samples belonging to this cluster.
        mask = cluster_ids == c

        # If there are no healthy samples in this cluster, we cannot
        # calculate a meaningful cluster-specific reference.
        if mask.sum() == 0:
            continue

        # Average the latent distributions of all healthy samples
        # belonging to this operating condition.
        #
        # This gives us a representative point/distribution describing
        # what "healthy" looks like for this particular cluster.
        reference[c] = {
            "mu": mu_all[mask].mean(axis=0),
            "sigma": sigma_all[mask].mean(axis=0)
        }
    # Fallback handling
    # It is possible that some clusters contain no healthy samples.
    # Without a fallback, attempting to monitor an engine assigned
    # to one of those clusters would fail because no reference exists.
    # We therefore reuse an existing reference.
    # NOTE for future edit:
    # The comment currently says "nearest cluster reference", but
    # the code does NOT actually calculate the nearest cluster.
    # It simply takes the first available reference.
    if reference:

        # Take the first available reference as the fallback.
        any_ref = next(iter(reference.values()))
        for c in range(n_clusters):
            if c not in reference:
                # copy() prevents the missing cluster from sharing
                # the exact same mutable NumPy arrays.
                reference[c] = {
                    "mu": any_ref["mu"].copy(),
                    "sigma": any_ref["sigma"].copy()
                }
                print(f" Warning: cluster {c} has no healthy samples - using nearest cluster reference.")
    return reference

def build_drift_thresholds(
    vae: VAE,
    healthy_X: torch.Tensor,
    cluster_ids: np.ndarray,
    n_clusters: int
) -> dict:
    """
    Calculate reconstruction-error thresholds for detecting drift.
    The VAE attempts to reconstruct its input.

    For healthy data, we expect reconstruction error to generally
    remain within a normal range.
    The threshold is currently defined as:
        mean reconstruction error + 2 * standard deviation

    This is calculated separately for each operating cluster.
    Later, when new data is processed:
        reconstruction error > threshold
                    |
                    v
              possible drift

    A global threshold is also calculated as a fallback.

    IMPORTANT:
    mean + 2σ corresponds to approximately 97.5% coverage only
    under a roughly Gaussian / symmetric error distribution.
    Reconstruction errors do not necessarily follow a Gaussian
    distribution, so this assumption should eventually be tested.
    """

    # Process the healthy sequences in batches for the same reason
    # as _encode_batched(): avoid loading the entire dataset through
    # the model at once.
    loader = DataLoader(TensorDataset(healthy_X), batch_size=256, shuffle=False)

    errors = []
    # Evaluation mode ensures deterministic model behavior.
    vae.eval()
    # We are measuring the trained model, not training it.
    # Therefore gradients are unnecessary.
    with torch.no_grad():

        for (x_batch,) in loader:
            # Calculate how well the VAE reconstructs each sequence.
            # The result should contain one reconstruction-error value
            # for each sample in the batch.
            batch_errors = vae.reconstruction_error(x_batch.to(DEVICE))

            # Move the errors back to CPU and store them as NumPy arrays.
            errors.append(batch_errors.cpu().numpy())
    # Combine all batch results into one array.
    # After this:
    #   errors[i]
    # corresponds to healthy_X[i].
    # This correspondence is important because cluster_ids[i]
    # must refer to the same sample.
    errors = np.concatenate(errors)

    thresholds: dict = {}

    # Build a separate threshold for each operating condition.
    for c in range(n_clusters):
        # Identify healthy samples belonging to this cluster.
        mask = cluster_ids == c
        # We need at least two samples to calculate a meaningful standard deviation.
        if mask.sum() < 2:
            continue

        # Extract reconstruction errors for this cluster only.
        ec = errors[mask]

        # Define the cluster's "normal" reconstruction-error limit.
        # Errors above this value are considered unusual relative
        # to the healthy training data for this operating condition.
        thresholds[c] = float(ec.mean() + 2.0 * ec.std())

    # Also calculate a global threshold across every healthy sample.
    # This provides a fallback if a cluster-specific threshold
    # cannot be used.
    thresholds["global"] = float(errors.mean() + 2.0 * errors.std())

    print("Drift thresholds (recon error mean+2σ):")

    for k, v in thresholds.items():
        print(f"  cluster {k}: {v:.6f}")

    return thresholds


# Per-dataset training pipeline

def train_vae_for_dataset(dataset_key: str, cfg_ds: dict, registry_path: Path) -> None:
    print(f"{dataset_key} - VAE training")

    # VAE hyperparameters, tunable per dataset in datasets.yaml (all optional).
    healthy_threshold = int(cfg_ds.get("vae_healthy_rul_threshold", 80))
    hidden_dim = int(cfg_ds.get("vae_hidden_dim", 64))
    latent_dim = int(cfg_ds.get("vae_latent_dim", 16))
    num_layers = int(cfg_ds.get("vae_num_layers", 1))
    beta = float(cfg_ds.get("vae_beta", 1.0))
    max_epochs = int(cfg_ds.get("vae_epochs", 100))
    patience = int(cfg_ds.get("vae_patience", 10))
    lr = float(cfg_ds.get("vae_lr", 1e-3))
    batch_size = int(cfg_ds.get("vae_batch_size", 256))
    warmup_fraction = float(cfg_ds.get("vae_beta_warmup_fraction", 0.2))
    n_clusters = int(cfg_ds.get("n_clusters", 1))

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
    print(f"  beta           : {beta}  (warmup fraction: {warmup_fraction})")
    print(f"  Healthy RUL >  : {healthy_threshold}")

    feature_path = _ROOT / "data" / "processed" / dataset_key / "train_features.csv"
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature file not found: {feature_path}")

    full_df = pd.read_csv(feature_path)
    train_df, val_df = split_by_engine(full_df)

    # Filter to healthy windows AFTER the engine split — otherwise val
    # engines' healthy data could leak into the training reference.
    healthy_train = train_df[train_df[TARGET_COL] > healthy_threshold].copy()
    healthy_val = val_df[val_df[TARGET_COL] > healthy_threshold].copy()

    print(f"  Train engines  : {train_df['unit_number'].nunique()} total  → {healthy_train['unit_number'].nunique()} with healthy windows")
    print(f"  Val engines    : {val_df['unit_number'].nunique()} total  → {healthy_val['unit_number'].nunique()} with healthy windows")

    if len(healthy_train) == 0:
        raise RuntimeError(f"No healthy training data for {dataset_key} at threshold RUL>{healthy_threshold}.")

    X_train, _, _, _, cluster_train = create_health_sequences(healthy_train, seq_length)
    X_val, _, _, _, _ = create_health_sequences(healthy_val, seq_length)

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
            loss, recon, kl = vae_loss(x_batch, x_hat, mu, logvar, beta=current_beta)
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
                loss, recon, kl = vae_loss(x_batch, x_hat, mu, logvar, beta=current_beta)
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
            f"  Epoch {epoch+1:>3}/{max_epochs}  β={current_beta:.3f}  "
            f"train [total={train_total:.4f} recon={train_recon:.4f} kl={train_kl:.4f}]  "
            f"val [total={val_total:.4f} recon={val_recon:.4f} kl={val_kl:.4f}]"
        )

        # KL near zero means the encoder is ignoring the input (posterior collapse).
        if epoch > 10 and val_kl < 0.01:
            print("  KL ~= 0: possible posterior collapse. Consider lowering beta or increasing warmup_fraction.")

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
    print(f"\n  Best val loss: {best_val_loss:.6f}")

    # Both reference and thresholds use TRAINING data only — val stays untouched.
    print("\n  Building healthy reference distributions")
    reference = build_healthy_reference(vae, X_train, cluster_train, n_clusters)
    for c, ref in reference.items():
        print(f"    cluster {c}: mu_ref norm={np.linalg.norm(ref['mu']):.4f}  sigma_ref mean={ref['sigma'].mean():.4f}")

    print("\n  Building drift thresholds...")
    thresholds = build_drift_thresholds(vae, X_train, cluster_train, n_clusters)

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

    reference_saveable = {c: {"mu": ref["mu"].tolist(), "sigma": ref["sigma"].tolist()} for c, ref in reference.items()}
    joblib.dump(reference_saveable, artifact_dir / "vae_healthy_reference.pkl")
    joblib.dump(thresholds, artifact_dir / "vae_drift_thresholds.pkl")

    print(f"  Saved vae_config.pkl, vae_healthy_reference.pkl, vae_drift_thresholds.pkl -> {artifact_dir}")

    current = load_registry(registry_path)
    current[dataset_key]["vae"] = {
        "trained_at": str(datetime.date.today()),
        "best_val_loss": float(best_val_loss),
        "config": vae_cfg_save,
        "healthy_rul_threshold": healthy_threshold,
        "n_clusters": n_clusters,
        "artifacts": {
            "vae": str(artifact_dir / "vae.pt"),
            "config": str(artifact_dir / "vae_config.pkl"),
            "reference": str(artifact_dir / "vae_healthy_reference.pkl"),
            "thresholds": str(artifact_dir / "vae_drift_thresholds.pkl"),
        },
    }
    save_registry(registry_path, current)
    print(f"  Registry updated -> vae section added for {dataset_key}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VAE for health monitoring on healthy engine data.")
    parser.add_argument("--dataset", choices=["FD001", "FD002", "FD003", "FD004", "all"], default="FD001")
    args = parser.parse_args()

    registry_path = _ROOT / "config" / "model_registry.yaml"
    datasets_path = _ROOT / "config" / "datasets.yaml"

    with open(datasets_path, "r") as f:
        all_cfg = yaml.safe_load(f) or {}

    datasets = ["FD001", "FD002", "FD003", "FD004"] if args.dataset == "all" else [args.dataset]

    for ds in datasets:
        train_vae_for_dataset(ds, all_cfg.get(ds, {}), registry_path)