"""
src/health/health_monitor.py

Runs the trained VAE over the test set to produce information-geometric
health indices for every sliding window across every test engine.

Outputs
-------
reports/{dataset}/health_indices.csv — one row per (engine, cycle) window:
    unit_number    engine identifier
    cycle          time_in_cycles at the last step of the window
    true_rul       ground-truth RUL at that cycle (for dashboard alignment)
    kl_div         KL(q_current || q_healthy) — asymmetric, current vs healthy ref
    js_div         Jensen-Shannon divergence — symmetric, bounded [0, log2]
    wasserstein    Squared Wasserstein-2 distance — numerically robust
    recon_error    VAE reconstruction MSE — rises as engine leaves healthy regime
    drift_flag     True when recon_error exceeds the per-cluster drift threshold
    op_cluster     operating cluster at this window (for multi-condition datasets)
    latent_mu_norm L2 norm of the latent mean vector (useful for scatter plots)

reports/{dataset}/health_summary.json — fleet-level statistics:
    healthy baseline mean/std per metric, drift threshold per cluster,
    n_engines, n_drifted_engines, first drift cycle distribution.

Usage
-----
    python src/health/health_monitor.py --dataset FD001
    python src/health/health_monitor.py --dataset all
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import yaml

# ── Path resolution ───────────────────────────────────────────────────────────
_ROOT     = Path(__file__).resolve().parents[2]
_TRAINING = _ROOT / "src" / "training"
_MODELS   = _ROOT / "src" / "models"
_HEALTH   = _ROOT / "src" / "health"

for _p in [str(_TRAINING), str(_MODELS), str(_HEALTH)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shared import DEVICE, NON_FEATURE_COLS, TARGET_COL
from vae import VAE
from geometry import all_distances


# ── JSON helper ───────────────────────────────────────────────────────────────

def _to_python(obj):
    if isinstance(obj, dict):       return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):       return [_to_python(v) for v in obj]
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating):return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    return obj

def _safe_json_write(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(_to_python(data), f, indent=2)
    shutil.move(str(tmp), str(path))


# ── Sequence construction with full tracking ──────────────────────────────────

def create_health_sequences(
    df: pd.DataFrame,
    seq_length: int,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sliding-window sequences tracking unit_number, cycle, true_rul, and
    op_cluster at the last timestep. Mirrors train_vae.py's version — kept
    local so this module has no dependency on training code.

    Returns: (X_tensor, true_rul, unit_ids, cycle_ids, cluster_ids)
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


# ── Artefact loading ──────────────────────────────────────────────────────────

def load_vae_artifacts(
    dataset_key:  str,
    registry:     dict,
) -> tuple[VAE, dict, dict, dict]:
    """
    Load VAE model + config + healthy reference + drift thresholds from
    paths recorded in model_registry.yaml under the 'vae' key.

    Returns: (vae_model, vae_cfg, reference, thresholds)

    reference  : {cluster_id: {"mu": np.ndarray, "sigma": np.ndarray}}
    thresholds : {cluster_id: float, "global": float}
    """
    if "vae" not in registry.get(dataset_key, {}):
        raise RuntimeError(
            f"No VAE entry for {dataset_key} in registry. "
            "Run train_vae.py first."
        )

    vae_reg = registry[dataset_key]["vae"]
    arts    = vae_reg["artifacts"]

    # Rebuild VAE architecture from saved config
    vae_cfg  = joblib.load(Path(arts["config"]))
    vae_model = VAE(
        input_dim  = vae_cfg["input_dim"],
        hidden_dim = vae_cfg["hidden_dim"],
        latent_dim = vae_cfg["latent_dim"],
        seq_length = vae_cfg["seq_length"],
        num_layers = vae_cfg["num_layers"],
        dropout    = vae_cfg["dropout"],
    )
    vae_model.load_state_dict(
        torch.load(
            Path(arts["vae"]),
            map_location=DEVICE,
            weights_only=True,
        )
    )
    vae_model.eval()
    vae_model.to(DEVICE) 

    # Load healthy reference — stored as {cluster_id: {"mu": list, "sigma": list}}
    reference_raw = joblib.load(Path(arts["reference"]))
    reference = {
        c: {
            "mu":    np.array(ref["mu"],    dtype=np.float32),
            "sigma": np.array(ref["sigma"], dtype=np.float32),
        }
        for c, ref in reference_raw.items()
    }

    thresholds = joblib.load(Path(arts["thresholds"]))

    print(f"  VAE loaded: latent_dim={vae_cfg['latent_dim']}  "
          f"seq_length={vae_cfg['seq_length']}  "
          f"clusters_in_reference={list(reference.keys())}")

    return vae_model, vae_cfg, reference, thresholds


# ── Batch inference ───────────────────────────────────────────────────────────

def _encode_batched(
    vae_model: VAE,
    X:         torch.Tensor,
    batch:     int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Encode all sequences in X without gradients.
    Returns (mu_all, sigma_all, recon_error_all) — all (N,) or (N, latent_dim).

    sigma = exp(0.5 * logvar) — geometry.py expects sigma not logvar.
    reconstruction_error() uses the deterministic mu path (no sampling)
    so results are reproducible across runs.
    """
    loader      = DataLoader(TensorDataset(X), batch_size=batch, shuffle=False)
    mus, sigs, errs = [], [], []

    vae_model.eval()
    with torch.no_grad():
        for (x_batch,) in loader:
            x_batch  = x_batch.to(DEVICE)
            mu, logvar = vae_model.encode(x_batch)
            sigma    = torch.exp(0.5 * logvar)
            err      = vae_model.reconstruction_error(x_batch)

            mus.append(mu.cpu().numpy())
            sigs.append(sigma.cpu().numpy())
            errs.append(err.cpu().numpy())

    return (
        np.concatenate(mus),    # (N, latent_dim)
        np.concatenate(sigs),   # (N, latent_dim)
        np.concatenate(errs),   # (N,)
    )


# ── Main health monitoring pipeline ──────────────────────────────────────────

def compute_health_indices(
    dataset_key:   str,
    registry_path: Path,
) -> None:
    print(f"\n{'='*60}")
    print(f"  Health monitoring — {dataset_key}")
    print(f"{'='*60}")

    # ── Load registry + VAE artefacts ─────────────────────────────────────────
    with open(registry_path, "r") as f:
        registry = yaml.safe_load(f) or {}

    vae_model, vae_cfg, reference, thresholds = load_vae_artifacts(
        dataset_key, registry
    )
    seq_length = vae_cfg["seq_length"]

    # ── Load test features ────────────────────────────────────────────────────
    test_path = _ROOT / "data" / "processed" / dataset_key / "test_features.csv"
    if not test_path.exists():
        raise FileNotFoundError(f"Test features not found: {test_path}")

    test_df = pd.read_csv(test_path)
    print(f"  Test set: {test_df['unit_number'].nunique()} engines  "
          f"({len(test_df):,} rows)")

    # ── Build sliding window sequences ────────────────────────────────────────
    print("  Creating sequences...")
    X_test, y_test, unit_ids, cycle_ids, cluster_ids = create_health_sequences(
        test_df, seq_length
    )
    print(f"  Total sequences: {len(X_test):,}")

    # ── Encode all sequences in batches ───────────────────────────────────────
    print("  Encoding sequences through VAE...")
    mu_all, sigma_all, recon_all = _encode_batched(vae_model, X_test)

    # ── Compute information-geometric distances ───────────────────────────────
    # For each window, look up the healthy reference for its operating cluster,
    # then compute all three distances using geometry.py.
    # Done in a vectorised loop over unique clusters for efficiency — avoids
    # calling all_distances() once per sequence.
    print("  Computing information-geometric distances...")

    n_seq    = len(X_test)
    kl_arr   = np.zeros(n_seq, dtype=np.float32)
    js_arr   = np.zeros(n_seq, dtype=np.float32)
    w2_arr   = np.zeros(n_seq, dtype=np.float32)

    unique_clusters = np.unique(cluster_ids)

    for c in unique_clusters:
        mask = cluster_ids == c

        if c in reference:
            ref = reference[c]
        else:
            # Fallback to cluster 0 if this cluster wasn't in the healthy training data
            ref = reference[0]
            print(f"  Warning: cluster {c} not in reference — using cluster 0 fallback")

        mu_ref    = ref["mu"]     # (latent_dim,)
        sigma_ref = ref["sigma"]  # (latent_dim,)

        # Broadcast reference to match batch shape (n_in_cluster, latent_dim)
        mu_batch    = mu_all[mask]     # (n, latent_dim)
        sigma_batch = sigma_all[mask]  # (n, latent_dim)

        # all_distances handles batched input (2D arrays) and returns (n,) arrays
        dists = all_distances(mu_batch, sigma_batch, mu_ref, sigma_ref)

        kl_arr[mask] = dists["kl_div"].astype(np.float32)
        js_arr[mask] = dists["js_div"].astype(np.float32)
        w2_arr[mask] = dists["wasserstein"].astype(np.float32)

    # ── Drift flag ────────────────────────────────────────────────────────────
    # An engine window is flagged as drifted if its reconstruction error
    # exceeds the per-cluster drift threshold. Falls back to the global
    # threshold if the cluster isn't in the threshold dict.
    drift_flags = np.zeros(n_seq, dtype=bool)

    for c in unique_clusters:
        mask = cluster_ids == c
        thr  = thresholds.get(c, thresholds.get("global", float("inf")))
        drift_flags[mask] = recon_all[mask] > thr

    # ── Latent space norm — useful for 2D scatter visualisations ─────────────
    # The L2 norm of mu captures how far the current encoding is from the
    # origin of latent space. Healthy encodings cluster near the prior N(0,I),
    # so norms near 0 are healthy; large norms indicate the encoder is
    # producing representations the prior wasn't trained to produce.
    latent_mu_norm = np.linalg.norm(mu_all, axis=1).astype(np.float32)

    # ── Assemble output DataFrame ─────────────────────────────────────────────
    health_df = pd.DataFrame({
        "unit_number":    unit_ids,
        "cycle":          cycle_ids,
        "true_rul":       y_test,
        "kl_div":         kl_arr.round(6),
        "js_div":         js_arr.round(6),
        "wasserstein":    w2_arr.round(6),
        "recon_error":    recon_all.round(6),
        "drift_flag":     drift_flags,
        "op_cluster":     cluster_ids,
        "latent_mu_norm": latent_mu_norm.round(4),
    })

    # ── Print summary statistics ──────────────────────────────────────────────
    print(f"\n  Health index statistics (all windows):")
    for col in ["kl_div", "js_div", "wasserstein", "recon_error"]:
        print(f"    {col:<14} mean={health_df[col].mean():.4f}  "
              f"std={health_df[col].std():.4f}  "
              f"max={health_df[col].max():.4f}")

    n_drifted = drift_flags.sum()
    pct_drift = n_drifted / len(drift_flags) * 100
    print(f"\n  Drift flags: {n_drifted:,} / {len(drift_flags):,} "
          f"windows ({pct_drift:.1f}%)")

    # Per-engine: first cycle where drift was detected
    drift_df = health_df[health_df["drift_flag"]].copy()
    if len(drift_df) > 0:
        first_drift = (drift_df.groupby("unit_number")["cycle"].min()
                       .rename("first_drift_cycle"))
        last_rul    = (health_df.groupby("unit_number")["true_rul"].min()
                       .rename("min_rul"))
        drift_summary = pd.concat([first_drift, last_rul], axis=1).dropna()
        n_drifted_engines = len(first_drift)
        print(f"  Drifted engines: {n_drifted_engines} / "
              f"{health_df['unit_number'].nunique()}")
    else:
        n_drifted_engines = 0
        drift_summary     = pd.DataFrame()
        print("  No drift detected in any engine.")

    # ── Fleet-level health summary ────────────────────────────────────────────
    health_summary = {
        "dataset":          dataset_key,
        "n_engines":        int(health_df["unit_number"].nunique()),
        "n_sequences":      int(len(health_df)),
        "n_drifted_engines": n_drifted_engines,
        "drift_thresholds": {
            str(k): float(v) for k, v in thresholds.items()
        },
        "stats": {
            col: {
                "mean": float(health_df[col].mean()),
                "std":  float(health_df[col].std()),
                "min":  float(health_df[col].min()),
                "max":  float(health_df[col].max()),
                "p90":  float(np.percentile(health_df[col], 90)),
            }
            for col in ["kl_div", "js_div", "wasserstein", "recon_error"]
        },
        "pct_windows_drifted": float(pct_drift),
    }

    # ── Save outputs ──────────────────────────────────────────────────────────
    out_dir = _ROOT / "reports" / dataset_key
    out_dir.mkdir(parents=True, exist_ok=True)

    health_path = out_dir / "health_indices.csv"
    health_df.to_csv(health_path, index=False)
    print(f"\n  Saved: {health_path}  ({len(health_df):,} rows)")

    summary_path = out_dir / "health_summary.json"
    _safe_json_write(summary_path, health_summary)
    print(f"  Saved: {summary_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute VAE health indices on the test set."
    )
    parser.add_argument(
        "--dataset",
        choices=["FD001", "FD002", "FD003", "FD004", "all"],
        default="FD001",
    )
    args = parser.parse_args()

    registry_path = _ROOT / "config" / "model_registry.yaml"
    datasets = (
        ["FD001", "FD002", "FD003", "FD004"]
        if args.dataset == "all" else [args.dataset]
    )

    for ds in datasets:
        compute_health_indices(ds, registry_path)