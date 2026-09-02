"""
Runs the trained VAE over the test set to produce information-geometric
health indices for every sliding window across every test engine.

Outputs
-------
reports/{dataset}/health_indices.csv — one row per (engine, cycle) window:

  Identity
    unit_number         engine identifier
    cycle               time_in_cycles at the last step of the window
    true_rul            ground-truth RUL at that cycle (dashboard alignment)
    op_cluster          operating cluster at this window

  Fleet-referenced geometry — distance from the healthy population
  N(mu_ref, Sigma_ref) for this window's operating cluster:
    mahalanobis         sqrt((mu-mu_ref)' Sigma_ref^-1 (mu-mu_ref))
    fisher_rao          geodesic distance on the Gaussian manifold
    kl_div              KL(current posterior || healthy population)
    js_div              symmetrised version of the above
    wasserstein         squared Bures-Wasserstein distance

  Self-referenced geometry — distance from this engine's OWN earliest
  observed state, whitened by the fleet's healthy covariance. Removes
  unit-to-unit offset, so it answers "has this engine changed" rather
  than "is this engine unusual":
    mahalanobis_self
    fisher_rao_self

  Reconstruction
    recon_error         VAE reconstruction MSE (per element)

  Alarms
    drift_raw           recon_error > per-cluster drift threshold
    drift_flag          drift_raw after a k-of-n persistence filter
    geo_alarm_raw       mahalanobis > per-cluster geometry threshold
    geo_alarm           geo_alarm_raw after the same persistence filter

  Derived
    latent_mu_norm      L2 norm of the latent mean (legacy column)
    latent_mu_centered  ||mu - mu_ref|| — the norm with the healthy
                        offset removed, which is what you actually want
                        to plot
    health_score        0-100, from the empirical healthy Mahalanobis
                        CDF. 100 = indistinguishable from healthy.

reports/{dataset}/health_summary.json — fleet-level statistics, alarm
    thresholds, posterior diagnostics, and detection lead times.

WHAT CHANGED IN THIS REVISION
-----------------------------
1. Geometry is computed against the healthy population COVARIANCE
   rather than an averaged posterior sigma. See geometry.py's module
   docstring for why the old version returned ~1e-5 for everything.

2. A per-engine self-baseline was added. On FD001 the latent norm
   tracked RUL with the right sign on 80 of 80 engines but varied only
   in the fourth decimal place, because a large constant offset
   dominated. Re-centring on the engine's own early-life encoding
   removes that offset entirely.

3. Alarms get a k-of-n persistence filter. A threshold with a 1% false
   positive rate still fires on ~1% of healthy windows, and with
   hundreds of windows per engine that guarantees a spurious early
   trigger on nearly every engine — which is why "first drift cycle"
   was previously meaningless. Requiring k of the last n consecutive
   windows drops the effective false-alarm rate by orders of magnitude
   while costing only a few cycles of detection delay.

4. Detection lead time is reported: how many cycles before end-of-life
   the first persisted alarm fires. That is the number a maintenance
   organisation actually cares about.

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

# Path resolution
_ROOT = Path(__file__).resolve().parents[2]
_TRAINING = _ROOT / "src" / "training"
_MODELS = _ROOT / "src" / "models"
_HEALTH = _ROOT / "src" / "health"

for _p in [str(_TRAINING), str(_MODELS), str(_HEALTH)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shared import DEVICE, NON_FEATURE_COLS, TARGET_COL
from vae import VAE
from geometry import (
    all_distances_full,
    fisher_rao,
    mahalanobis,
    regularise_covariance,
)

# Defaults for the self-baseline and persistence filter. Overridable per
# dataset in datasets.yaml (vae_baseline_windows / vae_persist_k / vae_persist_n).
_DEFAULT_BASELINE_WINDOWS = 20
_DEFAULT_PERSIST_K = 3
_DEFAULT_PERSIST_N = 5


# JSON helper

def _to_python(obj):
    if isinstance(obj, dict): return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, list): return [_to_python(v) for v in obj]
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating):return float(obj)
    if isinstance(obj, np.bool_): return bool(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    return obj

def _safe_json_write(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(_to_python(data), f, indent=2)
    shutil.move(str(tmp), str(path))


# Sequence construction with full tracking

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
        udata = df[df["unit_number"] == unit].sort_values("time_in_cycles")
        X = udata.drop(columns=cols_to_drop).values
        y = udata[TARGET_COL].values
        cycles = udata["time_in_cycles"].values
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


# Artefact loading

def _upgrade_legacy_reference(ref_raw: dict) -> dict:
    """
    Accept a reference saved by the OLD train_vae.py, which stored only
    {"mu": [...], "sigma": [...]} as plain lists, and synthesise the
    matrices the full-covariance geometry needs.

    The synthesised covariance is diag(sigma^2) — the average posterior
    width. That is exactly the quantity the revision moved away from, so
    the results will be as weak as the old ones; the fallback exists only
    so an un-retrained dataset still renders in the dashboard instead of
    crashing it. Retraining is what fixes the numbers.
    """
    mu = np.asarray(ref_raw["mu"], dtype=np.float64)
    sigma = np.asarray(ref_raw["sigma"], dtype=np.float64)
    cov = regularise_covariance(np.diag(sigma**2))
    evals, evecs = np.linalg.eigh(cov)

    return {
        "mu": mu,
        "sigma": sigma,
        "cov": cov,
        "cov_inv": np.linalg.inv(cov),
        "cov_sqrt": evecs @ np.diag(np.sqrt(np.clip(evals, 0.0, None))) @ evecs.T,
        "cov_logdet": float(np.linalg.slogdet(cov)[1]),
        "n": 0,
        "legacy": True,
    }


def load_vae_artifacts(
    dataset_key: str,
    registry: dict,
) -> tuple[VAE, dict, dict, dict, dict]:
    """
    Load VAE model + config + healthy reference + thresholds + calibration
    from paths recorded in model_registry.yaml under the 'vae' key.

    Returns: (vae_model, vae_cfg, reference, thresholds, calibration)

    reference   : {cluster_id: {"mu", "sigma", "cov", "cov_inv",
                                "cov_sqrt", "cov_logdet", "n"}}
    thresholds  : {cluster_id: float, "global": float}   (recon error)
    calibration : {"geo_thresholds", "maha_quantile_grid",
                   "quantile_levels", ...} — empty dict if the dataset
                   has not been retrained since this revision.
    """
    if "vae" not in registry.get(dataset_key, {}):
        raise RuntimeError(
            f"No VAE entry for {dataset_key} in registry. "
            "Run train_vae.py first."
        )
    vae_reg = registry[dataset_key]["vae"]
    arts = vae_reg["artifacts"]

    # Rebuild VAE architecture from saved config
    vae_cfg  = joblib.load(Path(arts["config"]))
    vae_model = VAE(
        input_dim  = vae_cfg["input_dim"],
        hidden_dim = vae_cfg["hidden_dim"],
        latent_dim = vae_cfg["latent_dim"],
        seq_length = vae_cfg["seq_length"],
        num_layers = vae_cfg["num_layers"],
        dropout = vae_cfg["dropout"],
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

    # Healthy reference. New format carries full covariance matrices;
    # the old {mu, sigma} format is upgraded on the fly with a warning.
    reference_raw = joblib.load(Path(arts["reference"]))
    reference = {}
    legacy_ref = False
    for c, ref in reference_raw.items():
        if "cov_inv" in ref:
            reference[c] = {
                "mu": np.asarray(ref["mu"], dtype=np.float64),
                "sigma": np.asarray(ref["sigma"], dtype=np.float64),
                "cov": np.asarray(ref["cov"], dtype=np.float64),
                "cov_inv": np.asarray(ref["cov_inv"], dtype=np.float64),
                "cov_sqrt": np.asarray(ref["cov_sqrt"], dtype=np.float64),
                "cov_logdet": float(ref["cov_logdet"]),
                "n": int(ref.get("n", 0)),
            }
        else:
            reference[c] = _upgrade_legacy_reference(ref)
            legacy_ref = True

    if legacy_ref:
        print("  WARNING: healthy reference is in the pre-revision format "
              "(diagonal posterior sigma, no population covariance).")
        print("           Geometry indices will be as weak as before. "
              "Re-run train_vae.py for this dataset.")

    thresholds = joblib.load(Path(arts["thresholds"]))

    calibration: dict = {}
    if "calibration" in arts and Path(arts["calibration"]).exists():
        calibration = joblib.load(Path(arts["calibration"]))
    else:
        print("  Note: no calibration artefact found — geometry alarms and "
              "health scores will be derived from the test distribution "
              "instead of a healthy reference. Re-run train_vae.py to fix.")

    print(f"  VAE loaded: latent_dim={vae_cfg['latent_dim']}  "
          f"seq_length={vae_cfg['seq_length']}  "
          f"clusters_in_reference={list(reference.keys())}")

    posterior = vae_reg.get("posterior", {})
    if posterior:
        print(f"  Posterior : {posterior.get('active_units', '?')}/"
              f"{posterior.get('latent_dim', '?')} active units, "
              f"total KL {posterior.get('total_kl', float('nan')):.3f}")

    return vae_model, vae_cfg, reference, thresholds, calibration


# Batch inference

def _encode_batched(
    vae_model: VAE,
    X: torch.Tensor,
    batch: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Encode all sequences in X without gradients.
    Returns (mu_all, sigma_all, recon_error_all).

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
            sigma = torch.exp(0.5 * logvar)
            err = vae_model.reconstruction_error(x_batch)

            mus.append(mu.cpu().numpy())
            sigs.append(sigma.cpu().numpy())
            errs.append(err.cpu().numpy())

    return (
        np.concatenate(mus).astype(np.float64),    # (N, latent_dim)
        np.concatenate(sigs).astype(np.float64),   # (N, latent_dim)
        np.concatenate(errs).astype(np.float64),   # (N,)
    )


# Self-referenced baseline

def compute_self_referenced(
    mu_all: np.ndarray,
    sigma_all: np.ndarray,
    unit_ids: np.ndarray,
    cycle_ids: np.ndarray,
    cluster_ids: np.ndarray,
    reference: dict,
    n_baseline: int = _DEFAULT_BASELINE_WINDOWS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Distance of every window from its OWN engine's earliest observed
    state, whitened by the fleet's healthy covariance.

    Why this exists
    ---------------
    Fleet-referenced distance conflates two things: how far this engine
    has degraded, and how far it started from the fleet average.
    Turbofans vary in initial wear, so the second term is a per-engine
    constant that adds noise to a cross-engine comparison. Subtracting
    each engine's own baseline removes it.

    Why the covariance still comes from the fleet
    ---------------------------------------------
    The baseline mean needs only ~20 windows to estimate well. A
    covariance in latent_dim dimensions does not — from 20 samples in
    16 dimensions it would be rank-deficient noise. So we take the
    centre from the engine (cheap to estimate, engine-specific) and the
    whitening structure from the fleet (expensive to estimate, and not
    engine-specific anyway). That split is the whole trick.

    The caveat to be ready for
    --------------------------
    CMAPSS test trajectories are truncated at an arbitrary point, so an
    engine's first observed window is NOT guaranteed to be healthy —
    some test units are already well into degradation when the record
    starts. This index therefore measures change relative to the
    earliest OBSERVED state, not relative to a known-healthy state. For
    an engine whose record begins late, it will understate degradation.
    The fleet-referenced indices do not have this problem, which is the
    reason to keep both rather than replace one with the other.

    Returns (mahalanobis_self, fisher_rao_self), each shape (N,).
    """
    n = len(mu_all)
    maha_self = np.zeros(n, dtype=np.float64)
    fr_self = np.zeros(n, dtype=np.float64)

    for unit in np.unique(unit_ids):
        mask = unit_ids == unit
        idx = np.where(mask)[0]

        # Order by cycle so "first k windows" means what it says — the
        # sequences were built per engine in cycle order, but relying on
        # that implicitly is the kind of assumption that breaks silently
        # if create_health_sequences is ever reordered.
        order = idx[np.argsort(cycle_ids[idx])]
        k = min(n_baseline, len(order))
        base_idx = order[:k]

        mu_self = mu_all[base_idx].mean(axis=0)
        sigma_self = sigma_all[base_idx].mean(axis=0)

        # Whitening comes from this engine's operating cluster reference.
        # Use the modal cluster over the baseline window — an engine can
        # move between clusters in FD002/FD004, but its baseline is a
        # single point so it needs a single covariance.
        base_clusters = cluster_ids[base_idx]
        modal_cluster = int(np.bincount(base_clusters).argmax())
        ref = reference.get(modal_cluster) or next(iter(reference.values()))

        maha_self[order] = mahalanobis(mu_all[order], mu_self, ref["cov_inv"])
        fr_self[order] = fisher_rao(
            mu_all[order], sigma_all[order], mu_self, sigma_self
        )

    return maha_self, fr_self


# Persistence filter

def apply_persistence_filter(
    raw_flags: np.ndarray,
    unit_ids: np.ndarray,
    cycle_ids: np.ndarray,
    k: int = _DEFAULT_PERSIST_K,
    n: int = _DEFAULT_PERSIST_N,
) -> np.ndarray:
    """
    Confirm an alarm only when at least k of the last n windows were
    raw-flagged, evaluated independently per engine in cycle order.

    A threshold calibrated at the 0.99 quantile of healthy behaviour
    fires on 1% of healthy windows by construction. With ~90 windows per
    engine, the probability that at least one healthy window trips it is
    1 - 0.99^90 = 60%. That is why the un-filtered "first drift cycle"
    statistic was worthless: the first flag was almost always noise.

    Requiring 3 of the last 5 drops the per-engine false-alarm
    probability to well under 1% (the exact figure depends on the
    autocorrelation between overlapping windows, which is high, so treat
    the independent-trials calculation as a lower bound on the true
    rate). The cost is at most k-1 windows of detection delay.

    Engines with fewer than n windows use the shorter available history,
    scaling k proportionally so short records are not silently exempt.
    """
    confirmed = np.zeros(len(raw_flags), dtype=bool)

    for unit in np.unique(unit_ids):
        idx = np.where(unit_ids == unit)[0]
        order = idx[np.argsort(cycle_ids[idx])]
        flags = raw_flags[order].astype(int)

        window = min(n, len(flags))
        if window == 0:
            continue
        required = max(1, int(round(k * window / n)))

        # Rolling count of raw flags over the trailing `window` entries.
        rolling = (
            pd.Series(flags)
            .rolling(window=window, min_periods=1)
            .sum()
            .to_numpy()
        )
        confirmed[order] = rolling >= required

    return confirmed


# Health score

def maha_to_health_score(
    maha: np.ndarray,
    cluster_ids: np.ndarray,
    calibration: dict,
) -> np.ndarray:
    """
    Map a Mahalanobis distance onto a 0-100 health score using the
    EMPIRICAL CDF of healthy validation distances.

        score = 100 * (1 - F_healthy(d_M))

    100 means the window sits at or below the healthiest observed
    distance; 0 means it is beyond anything seen in healthy data.

    Under a Gaussian model d_M^2 would be chi-squared with latent_dim
    degrees of freedom, and scipy.stats.chi2.sf would give this directly.
    The empirical CDF is used instead because VAE encodings are not
    exactly Gaussian and the empirical version costs nothing — it is
    just an np.interp against a stored quantile grid.

    Falls back to a within-dataset percentile rank if no calibration
    artefact is available. That fallback is NOT comparable across
    datasets and is flagged as such in the summary JSON, because it
    normalises against the test set rather than against healthy data.
    """
    grid = calibration.get("maha_quantile_grid", {})
    levels = np.asarray(calibration.get("quantile_levels", []), dtype=np.float64)

    if not grid or len(levels) == 0:
        ranks = pd.Series(maha).rank(pct=True).to_numpy()
        return np.clip(100.0 * (1.0 - ranks), 0.0, 100.0)

    score = np.zeros(len(maha), dtype=np.float64)

    for c in np.unique(cluster_ids):
        mask = cluster_ids == c
        q = grid.get(str(c)) or grid.get("global")
        if q is None:
            score[mask] = np.nan
            continue
        # np.interp needs a strictly increasing x; quantiles are
        # non-decreasing but can tie, which interp tolerates.
        cdf = np.interp(maha[mask], np.asarray(q, dtype=np.float64), levels)
        score[mask] = 100.0 * (1.0 - cdf)

    return np.clip(score, 0.0, 100.0)


# Main health monitoring pipeline

def compute_health_indices(
    dataset_key:   str,
    registry_path: Path,
    ds_cfg:        dict | None = None,
) -> None:
    print(f"  Health monitoring - {dataset_key}")

    ds_cfg = ds_cfg or {}
    n_baseline = int(ds_cfg.get("vae_baseline_windows", _DEFAULT_BASELINE_WINDOWS))
    persist_k = int(ds_cfg.get("vae_persist_k", _DEFAULT_PERSIST_K))
    persist_n = int(ds_cfg.get("vae_persist_n", _DEFAULT_PERSIST_N))

    # Load registry + VAE artefacts
    with open(registry_path, "r") as f:
        registry = yaml.safe_load(f) or {}

    vae_model, vae_cfg, reference, thresholds, calibration = load_vae_artifacts(
        dataset_key, registry
    )
    seq_length = vae_cfg["seq_length"]

    # Load test features
    test_path = _ROOT / "data" / "processed" / dataset_key / "test_features.csv"
    if not test_path.exists():
        raise FileNotFoundError(f"Test features not found: {test_path}")

    test_df = pd.read_csv(test_path)
    print(f"  Test set: {test_df['unit_number'].nunique()} engines  "
          f"({len(test_df):,} rows)")

    # Build sliding window sequences
    print("  Creating sequences...")
    X_test, y_test, unit_ids, cycle_ids, cluster_ids = create_health_sequences(
        test_df, seq_length
    )
    print(f"  Total sequences: {len(X_test):,}")

    if len(X_test) == 0:
        raise RuntimeError(f"No sequences built for {dataset_key} — every test "
                           f"engine has fewer than seq_length={seq_length} cycles.")

    # Encode all sequences in batches
    print("  Encoding sequences through VAE...")
    mu_all, sigma_all, recon_all = _encode_batched(vae_model, X_test)

    # A cheap sanity check that pays for itself. If the spread of ||mu||
    # is a negligible fraction of its mean, the encoder is emitting a
    # near-constant vector and every geometry index below is measuring
    # floating-point dust.
    mu_norms = np.linalg.norm(mu_all, axis=1)
    rel_spread = mu_norms.std() / max(mu_norms.mean(), 1e-12)
    print(f"  Latent check: ||mu|| = {mu_norms.mean():.4f} +/- {mu_norms.std():.4f} "
          f"(relative spread {rel_spread:.2e})")
    if rel_spread < 1e-2:
        print("  WARNING: the encoder output is nearly constant across the test set.")
        print("           Geometry indices will be near-degenerate. Check the")
        print("           active-unit count from train_vae.py and lower vae_beta.")

    # Information-geometric distances against the healthy population
    # Vectorised over unique clusters — one all_distances_full() call per
    # cluster rather than one per sequence.
    print("  Computing information-geometric distances...")

    n_seq = len(X_test)
    index_names = ["mahalanobis", "fisher_rao", "kl_div", "js_div", "wasserstein"]
    arrays = {name: np.zeros(n_seq, dtype=np.float64) for name in index_names}
    mu_centered = np.zeros(n_seq, dtype=np.float64)

    unique_clusters = np.unique(cluster_ids)

    for c in unique_clusters:
        mask = cluster_ids == c

        if c in reference:
            ref = reference[c]
        else:
            fallback_c = next(iter(reference))
            ref = reference[fallback_c]
            print(f"  Warning: cluster {c} not in reference — "
                  f"using cluster {fallback_c} fallback")

        dists = all_distances_full(mu_all[mask], sigma_all[mask], ref, include_js=True)
        for name in index_names:
            arrays[name][mask] = np.asarray(dists[name], dtype=np.float64)

        # ||mu - mu_ref||: the plain Euclidean norm with the healthy
        # offset removed. Kept alongside Mahalanobis so the dashboard can
        # show what whitening buys you.
        mu_centered[mask] = np.linalg.norm(mu_all[mask] - ref["mu"], axis=1)

    # Self-referenced geometry
    print(f"  Computing self-referenced distances (baseline = first "
          f"{n_baseline} windows per engine)...")
    maha_self, fr_self = compute_self_referenced(
        mu_all, sigma_all, unit_ids, cycle_ids, cluster_ids,
        reference, n_baseline=n_baseline,
    )

    # Alarms: raw threshold crossing, then persistence filtering
    print(f"  Applying alarms (persistence filter: {persist_k} of last {persist_n})...")

    drift_raw = np.zeros(n_seq, dtype=bool)
    for c in unique_clusters:
        mask = cluster_ids == c
        thr = thresholds.get(c, thresholds.get("global", float("inf")))
        drift_raw[mask] = recon_all[mask] > thr

    geo_thresholds = calibration.get("geo_thresholds", {})
    geo_alarm_raw = np.zeros(n_seq, dtype=bool)
    if geo_thresholds:
        for c in unique_clusters:
            mask = cluster_ids == c
            thr = geo_thresholds.get(c, geo_thresholds.get("global", float("inf")))
            geo_alarm_raw[mask] = arrays["mahalanobis"][mask] > thr
    else:
        print("  No geometry thresholds available — geo_alarm columns will be all-False.")

    drift_flag = apply_persistence_filter(
        drift_raw, unit_ids, cycle_ids, k=persist_k, n=persist_n
    )
    geo_alarm = apply_persistence_filter(
        geo_alarm_raw, unit_ids, cycle_ids, k=persist_k, n=persist_n
    )

    # Health score
    health_score = maha_to_health_score(arrays["mahalanobis"], cluster_ids, calibration)

    # Assemble output DataFrame
    health_df = pd.DataFrame({
        "unit_number": unit_ids,
        "cycle": cycle_ids,
        "true_rul": y_test,
        "op_cluster": cluster_ids,

        "mahalanobis": arrays["mahalanobis"].round(6),
        "fisher_rao": arrays["fisher_rao"].round(6),
        "kl_div": arrays["kl_div"].round(6),
        "js_div": arrays["js_div"].round(6),
        "wasserstein": arrays["wasserstein"].round(6),

        "mahalanobis_self": maha_self.round(6),
        "fisher_rao_self": fr_self.round(6),

        "recon_error": recon_all.round(6),

        "drift_raw": drift_raw,
        "drift_flag": drift_flag,
        "geo_alarm_raw": geo_alarm_raw,
        "geo_alarm": geo_alarm,

        "latent_mu_norm": mu_norms.round(6),
        "latent_mu_centered": mu_centered.round(6),
        "health_score": health_score.round(3),
    })

    # Print summary statistics
    metric_cols = ["mahalanobis", "fisher_rao", "kl_div", "js_div", "wasserstein",
                   "mahalanobis_self", "fisher_rao_self", "recon_error",
                   "latent_mu_centered", "health_score"]

    print(f"\n  Health index statistics (all windows):")
    for col in metric_cols:
        print(f"    {col:<20} mean={health_df[col].mean():>12.4f}  "
              f"std={health_df[col].std():>12.4f}  "
              f"max={health_df[col].max():>12.4f}")

    # Per-engine correlation with RUL — the quick read on whether each
    # index is tracking degradation WITHIN an engine, which is what a
    # health index is for. A global correlation can look respectable
    # while every individual engine is flat or contradictory.
    print(f"\n  Mean per-engine Spearman correlation with true RUL:")
    print(f"  (negative and consistent is what you want — the index should")
    print(f"   rise as RUL falls)")
    corr_summary = {}
    for col in metric_cols:
        rhos = []
        for _, g in health_df.groupby("unit_number"):
            if g[col].nunique() < 3 or g["true_rul"].nunique() < 3:
                continue
            rho = g[col].corr(g["true_rul"], method="spearman")
            if np.isfinite(rho):
                rhos.append(rho)
        if rhos:
            rhos = np.array(rhos)
            # health_score is inverted by design (high = healthy), so its
            # expected sign is positive while every other index is negative.
            expected_negative = col != "health_score"
            correct = (rhos < 0).mean() if expected_negative else (rhos > 0).mean()
            corr_summary[col] = {
                "mean_engine_spearman": float(rhos.mean()),
                "frac_correct_direction": float(correct),
                "n_engines": int(len(rhos)),
            }
            print(f"    {col:<20} {rhos.mean():+.4f}   "
                  f"correct direction on {correct:.0%} of engines")

    n_drifted = int(drift_flag.sum())
    pct_drift = n_drifted / len(drift_flag) * 100
    n_geo = int(geo_alarm.sum())
    pct_geo = n_geo / len(geo_alarm) * 100
    print(f"\n  Drift flags (confirmed): {n_drifted:,} / {len(drift_flag):,} "
          f"windows ({pct_drift:.1f}%)  [raw: {int(drift_raw.sum()):,}]")
    print(f"  Geometry alarms (confirmed): {n_geo:,} / {len(geo_alarm):,} "
          f"windows ({pct_geo:.1f}%)  [raw: {int(geo_alarm_raw.sum()):,}]")

    # Detection lead time: cycles between the first confirmed alarm and
    # the engine's last observed cycle. Reported per alarm type.
    def _lead_times(flag_col: str) -> dict:
        sub = health_df[health_df[flag_col]]
        if sub.empty:
            return {"n_engines_alarmed": 0}
        first = sub.groupby("unit_number")["cycle"].min()
        last = health_df.groupby("unit_number")["cycle"].max()
        # true_rul at the engine's final observed window — how much life
        # was genuinely left when monitoring stopped.
        final_rul = (health_df.sort_values("cycle")
                     .groupby("unit_number")["true_rul"].last())
        lead = (last - first).reindex(first.index)
        rul_at_alarm = (health_df.set_index(["unit_number", "cycle"])
                        .loc[list(zip(first.index, first.values)), "true_rul"]
                        .to_numpy())
        return {
            "n_engines_alarmed": int(len(first)),
            "n_engines_total": int(health_df["unit_number"].nunique()),
            "median_lead_cycles": float(np.median(lead)),
            "mean_lead_cycles": float(np.mean(lead)),
            "median_rul_at_first_alarm": float(np.median(rul_at_alarm)),
            "mean_final_rul": float(final_rul.mean()),
        }

    drift_lead = _lead_times("drift_flag")
    geo_lead = _lead_times("geo_alarm")

    print(f"\n  Detection lead time (cycles from first confirmed alarm to last observation):")
    for label, lt in [("recon drift", drift_lead), ("geometry", geo_lead)]:
        if lt.get("n_engines_alarmed"):
            print(f"    {label:<14} {lt['n_engines_alarmed']}/{lt['n_engines_total']} engines, "
                  f"median {lt['median_lead_cycles']:.0f} cycles, "
                  f"median RUL at alarm {lt['median_rul_at_first_alarm']:.0f}")
        else:
            print(f"    {label:<14} no engines alarmed")

    # Fleet-level health summary
    vae_reg = registry.get(dataset_key, {}).get("vae", {})
    health_summary = {
        "dataset": dataset_key,
        "geometry_version": vae_reg.get("geometry_version", 1),
        "n_engines": int(health_df["unit_number"].nunique()),
        "n_sequences": int(len(health_df)),

        # Kept for backward compatibility with the existing dashboard.
        "n_drifted_engines": int(drift_lead.get("n_engines_alarmed", 0)),
        "pct_windows_drifted": float(pct_drift),
        "drift_thresholds": {str(k): float(v) for k, v in thresholds.items()},

        "geo_thresholds": {str(k): float(v) for k, v in geo_thresholds.items()},
        "n_geo_alarmed_engines": int(geo_lead.get("n_engines_alarmed", 0)),
        "pct_windows_geo_alarmed": float(pct_geo),

        "alarm_config": {
            "persistence_k": persist_k,
            "persistence_n": persist_n,
            "baseline_windows": n_baseline,
            "drift_quantile": calibration.get("drift_quantile"),
            "geo_quantile": calibration.get("geo_quantile"),
            "calibrated_on": calibration.get("calibrated_on", "unknown"),
        },

        "lead_time": {"drift": drift_lead, "geometry": geo_lead},

        "index_quality": corr_summary,

        "latent": {
            "mu_norm_mean": float(mu_norms.mean()),
            "mu_norm_std": float(mu_norms.std()),
            "mu_norm_relative_spread": float(rel_spread),
            "degenerate": bool(rel_spread < 1e-2),
        },

        "posterior": vae_reg.get("posterior", {}),

        "stats": {
            col: {
                "mean": float(health_df[col].mean()),
                "std": float(health_df[col].std()),
                "min": float(health_df[col].min()),
                "max": float(health_df[col].max()),
                "p90": float(np.percentile(health_df[col], 90)),
            }
            for col in metric_cols
        },
    }

    # Save outputs
    out_dir = _ROOT / "reports" / dataset_key
    out_dir.mkdir(parents=True, exist_ok=True)

    health_path = out_dir / "health_indices.csv"
    health_df.to_csv(health_path, index=False)
    print(f"\n  Saved: {health_path}  ({len(health_df):,} rows)")

    summary_path = out_dir / "health_summary.json"
    _safe_json_write(summary_path, health_summary)
    print(f"  Saved: {summary_path}")


# CLI

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
    datasets_path = _ROOT / "config" / "datasets.yaml"

    all_cfg = {}
    if datasets_path.exists():
        with open(datasets_path, "r") as f:
            all_cfg = yaml.safe_load(f) or {}

    datasets = (
        ["FD001", "FD002", "FD003", "FD004"]
        if args.dataset == "all" else [args.dataset]
    )

    for ds in datasets:
        compute_health_indices(ds, registry_path, all_cfg.get(ds, {}))