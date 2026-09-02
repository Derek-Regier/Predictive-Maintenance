"""
src/inference/calibration.py

Distributional recalibration for the NGBoost predictive distribution.

WHY THIS EXISTS
---------------
`calibration_scale` (fitted in train.py by brentq) multiplies sigma by a
single constant. That is a one-parameter correction, and it can make
exactly ONE quantile level correct. Everything else drifts.

Measured on the FD001-FD004 test sets, the scale factor each level would
need to reach nominal coverage:

              0.5    0.6    0.7    0.8    0.9   0.95   spread
    FD001   0.840  0.909  0.978  1.067  1.170  1.228   0.39
    FD002   0.799  0.822  0.862  0.906  0.975  1.030   0.23
    FD003   0.671  0.736  0.812  0.886  0.974  1.043   0.37
    FD004   0.543  0.606  0.681  0.759  0.875  0.959   0.42

`calibration_alpha: 0.9` targeted the 0.9 column, so that column is close
and the rest are not. FD004 needs its intervals halved in the core and
left alone in the tails; no scalar does that. FD001 needs them NARROWED
at the median and WIDENED at 95% — its errors are heavier-tailed than the
Gaussian NGBoost fits, so scaling in either direction makes one end worse.

The problem is the SHAPE of the predictive distribution, not its width.

THE METHOD
----------
Kuleshov, Fenner & Ermon (2018), "Accurate Uncertainties for Deep
Learning Using Calibrated Regression".

For each calibration point, compute the Probability Integral Transform:

    u_i = F_i(y_i) = Phi( (y_i - mu_i) / sigma_i )

If the predictive distributions were perfectly calibrated, the u_i would
be Uniform(0,1). They are not. So fit an isotonic regression from the
predicted CDF value u to its own empirical CDF, giving a monotone map

    R : [0,1] -> [0,1]

and report R(F(y)) instead of F(y). Because R is monotone it never
reorders anything, so the model's ranking — and therefore AUC, which is
0.997+ on every dataset here — is completely untouched. Only the
probability scale moves.

This is strictly more general than scalar sigma scaling: sigma scaling is
the special case where R happens to be the composition of two Gaussian
CDFs. It corrects arbitrary shape misspecification (skew, kurtosis,
asymmetry) with one isotonic fit and no retraining.

RELATIONSHIP TO sigma_scale
---------------------------
This calibrator is fitted ON TOP of the existing sigma_scale pipeline —
it sees the already-scaled distribution and corrects the residual. That
keeps sigma_scale meaningful as a first-order correction, keeps every
existing artifact valid, and means code that has no calibrator loaded
behaves exactly as it does today.

WHAT IT DOES NOT FIX
--------------------
Recalibration is marginal, not conditional. It makes coverage correct
AVERAGED OVER THE FLEET. If the model is over-confident for near-failure
engines and under-confident for healthy ones, a single monotone map
cannot fix both — it will split the difference. Conditional calibration
would need the map to depend on features (e.g. binned by predicted mu).
Worth saying out loud rather than claiming more than the method delivers.

Usage
-----
    # fit on healthy validation engines and register the artifact
    python src/inference/calibration.py --dataset all

    # then in code
    cal = CDFCalibrator.load(path)
    p   = cal.cdf(20, mu, sigma)              # calibrated P(RUL < 20)
    lo, hi = cal.interval(0.90, mu, sigma)    # calibrated 90% interval
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression
import torch
import yaml

_ROOT = Path(__file__).resolve().parents[2]
_TRAINING = _ROOT / "src" / "training"
if str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))

from shared import (  # noqa: E402
    DEVICE, NON_FEATURE_COLS, TARGET_COL,
    build_backbone, load_registry, save_registry, split_by_engine,
)

# Resolution of the grid used to invert the isotonic map. 2001 points over
# [0,1] gives ~5e-4 resolution in probability, well below the precision
# anyone reads off a coverage table.
_GRID_N = 2001


class CDFCalibrator:
    """
    Monotone recalibration map for a Gaussian predictive distribution.

    Fitted on validation data, applied at inference. Holds no model — just
    an isotonic regressor and the grid used to invert it.
    """

    def __init__(
        self,
        p_grid: np.ndarray,
        q_grid: np.ndarray,
        n_cal: int,
        dataset_key: str,
        sigma_scale: float,
    ) -> None:
        """
        The calibrator holds a MONOTONE LOOKUP GRID, not the fitted
        IsotonicRegression object.

        That is deliberate. Persisting the sklearn estimator would tie the
        saved artifact to a class path and to a sklearn version, and it bit
        immediately: the first version pickled `self` from inside
        `__main__`, so the pickle recorded the class as
        `__main__.CDFCalibrator` and every later `from calibration import
        CDFCalibrator` failed to unpickle it. Storing two float arrays
        removes the whole class of problem — the artifact is pure numpy and
        `load()` does not need sklearn at all.

        The grid is a piecewise-linear reading of the isotonic step
        function at 2001 points, so it agrees with the exact estimator to
        ~5e-4 in probability. Both directions of the map use the same
        array, which guarantees `transform_prob` and `inverse_prob` are
        exact inverses of each other.
        """
        self._p_grid = np.asarray(p_grid, dtype=np.float64)
        self._q_grid = np.asarray(q_grid, dtype=np.float64)
        self.n_cal = int(n_cal)
        self.dataset_key = dataset_key
        # Recorded so a loaded calibrator can be checked against the
        # sigma_scale it was fitted under. If sigma_scale is later refitted
        # without refitting this, the two corrections stack incorrectly.
        self.sigma_scale = float(sigma_scale)

    # ---- fitting ---------------------------------------------------------

    @classmethod
    def fit(
        cls,
        y_true: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray,
        dataset_key: str,
        sigma_scale: float = 1.0,
        exclude_censored_at: float | None = None,
    ) -> "CDFCalibrator":
        """
        Fit the recalibration map.

        y_true : (n,) observed RUL on the calibration set
        mu     : (n,) predicted means
        sigma  : (n,) predicted sigmas, ALREADY multiplied by sigma_scale

        The empirical CDF target uses (rank + 0.5)/n rather than rank/n.
        Using rank/n forces the largest observation to map to exactly 1.0,
        which makes the tail of the map degenerate and produces infinite
        quantiles. The half-offset is the standard plotting-position fix
        and keeps the map strictly inside (0,1).
        """
        y_true = np.asarray(y_true, dtype=np.float64)
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.clip(np.asarray(sigma, dtype=np.float64), 1e-8, None)

        # Censored windows: those whose target sits exactly at max_rul.
        #
        # These are not observations of remaining life — they are a
        # constant, and the model predicts close to that constant, so their
        # PIT values pile up in the middle of [0,1]. On FD004 they are 62%
        # of all windows, which means a map fitted on the full set spends
        # most of its capacity correcting an artefact of the RUL cap rather
        # than correcting the model. Excluding them yields a map that
        # describes the predictive distribution where the target is real.
        #
        # This is left OFF by default. Turning it on makes the calibrator
        # honest about the model but mismatched against a test set that is
        # itself 62% censored, so the headline coverage number will look
        # worse while the uncensored one improves. Which you want depends
        # on whether the report is about the benchmark or about the model —
        # worth being explicit about that in a writeup rather than picking
        # one silently.
        n_before = len(y_true)
        if exclude_censored_at is not None:
            keep = y_true < exclude_censored_at
            if keep.sum() < 200:
                print(f"    Warning: only {int(keep.sum())} uncensored windows; "
                      f"keeping all {n_before} instead.")
            else:
                y_true, mu, sigma = y_true[keep], mu[keep], sigma[keep]
                print(f"    Excluded {n_before - int(keep.sum()):,} censored "
                      f"windows (target == {exclude_censored_at:g}); "
                      f"fitting on {int(keep.sum()):,}.")

        # PIT values: where each observation fell in its own predicted CDF
        u = norm.cdf(y_true, loc=mu, scale=sigma)

        order = np.argsort(u)
        u_sorted = u[order]
        n = len(u)
        emp = (np.arange(n) + 0.5) / n

        iso = IsotonicRegression(
            y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip"
        )
        iso.fit(u_sorted, emp)

        # Read the fitted step function onto a grid and keep only that.
        # Isotonic output is non-decreasing but can be flat; np.interp needs
        # a strictly increasing array to invert unambiguously, so add a
        # negligible ramp that breaks ties without moving anything by a
        # meaningful amount.
        p_grid = np.linspace(0.0, 1.0, _GRID_N)
        q = np.clip(np.asarray(iso.predict(p_grid), dtype=np.float64), 0.0, 1.0)
        q_grid = np.maximum.accumulate(q) + np.linspace(0.0, 1e-9, _GRID_N)

        return cls(p_grid, q_grid, n_cal=n,
                   dataset_key=dataset_key, sigma_scale=sigma_scale)

    # ---- applying --------------------------------------------------------

    def transform_prob(self, p) -> np.ndarray | float:
        """
        Map a raw Gaussian CDF value to its calibrated equivalent.

        This is the operation that fixes P(RUL < 20) and P(RUL < 50).
        Monotone, so the fleet ranking by failure probability is preserved
        exactly — only the numbers attached to that ranking change.
        """
        scalar = np.isscalar(p)
        p_arr = np.atleast_1d(np.asarray(p, dtype=np.float64))
        out = np.interp(np.clip(p_arr, 0.0, 1.0), self._p_grid, self._q_grid)
        out = np.clip(out, 0.0, 1.0)
        return float(out[0]) if scalar else out

    def inverse_prob(self, q) -> np.ndarray | float:
        """
        Inverse map: given a DESIRED calibrated probability q, return the raw
        Gaussian CDF level p such that transform_prob(p) == q.

        Needed for intervals. To build a genuinely 90% interval we cannot
        just take the raw 5th and 95th percentiles — we need the raw levels
        that RECALIBRATE to 0.05 and 0.95.
        """
        scalar = np.isscalar(q)
        q_arr = np.atleast_1d(np.asarray(q, dtype=np.float64))
        out = np.interp(np.clip(q_arr, 0.0, 1.0), self._q_grid, self._p_grid)
        return float(out[0]) if scalar else out

    def cdf(self, threshold: float, mu, sigma) -> np.ndarray | float:
        """Calibrated P(Y < threshold) for a Gaussian(mu, sigma) prediction."""
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.clip(np.asarray(sigma, dtype=np.float64), 1e-8, None)
        return self.transform_prob(norm.cdf(threshold, loc=mu, scale=sigma))

    def quantile(self, q: float, mu, sigma) -> np.ndarray | float:
        """Calibrated q-quantile of the predictive distribution."""
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.clip(np.asarray(sigma, dtype=np.float64), 1e-8, None)
        p_raw = self.inverse_prob(q)
        # Clip away from 0 and 1 so norm.ppf stays finite even if the
        # isotonic map saturates at the extremes.
        p_raw = float(np.clip(p_raw, 1e-6, 1 - 1e-6))
        return mu + sigma * norm.ppf(p_raw)

    def interval(self, level: float, mu, sigma) -> tuple:
        """
        Calibrated two-sided interval with the requested coverage.

        Note the asymmetry this can produce: the recalibrated lower and
        upper quantiles need not be equidistant from mu, because the
        isotonic map has no reason to be symmetric about 0.5. That is a
        feature — a symmetric interval is only correct if the error
        distribution is symmetric, and on CMAPSS it is not.
        """
        lo_q = (1.0 - level) / 2.0
        hi_q = (1.0 + level) / 2.0
        return self.quantile(lo_q, mu, sigma), self.quantile(hi_q, mu, sigma)

    # ---- diagnostics -----------------------------------------------------

    def coverage_table(self, levels=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95)) -> pd.DataFrame:
        """
        The raw level each nominal level maps to. Useful for showing what
        the calibrator is actually doing without needing test data:
        `raw_level_used` far from `nominal` means a large correction.
        """
        rows = []
        for lv in levels:
            lo_q, hi_q = (1 - lv) / 2, (1 + lv) / 2
            p_lo, p_hi = self.inverse_prob(lo_q), self.inverse_prob(hi_q)
            rows.append({
                "nominal": lv,
                "raw_lower_level": float(p_lo),
                "raw_upper_level": float(p_hi),
                "raw_level_used": float(p_hi - p_lo),
                "width_ratio_vs_gaussian": float(
                    (norm.ppf(np.clip(p_hi, 1e-6, 1 - 1e-6))
                     - norm.ppf(np.clip(p_lo, 1e-6, 1 - 1e-6)))
                    / (norm.ppf(hi_q) - norm.ppf(lo_q))
                ),
            })
        return pd.DataFrame(rows)

    # ---- persistence -----------------------------------------------------

    # Bumped whenever the on-disk layout changes, so a stale artifact fails
    # loudly with a message instead of silently misbehaving.
    STATE_VERSION = 1

    def to_state(self) -> dict:
        """Plain-data representation — numpy arrays and scalars only."""
        return {
            "__type__": "CDFCalibrator",
            "version": self.STATE_VERSION,
            "p_grid": self._p_grid,
            "q_grid": self._q_grid,
            "n_cal": self.n_cal,
            "dataset_key": self.dataset_key,
            "sigma_scale": self.sigma_scale,
        }

    def save(self, path: Path) -> None:
        """
        Persist as a dict, NOT as `self`.

        Pickling the instance records its class by import path. When this
        file is run as a script that path is `__main__`, so the artifact
        could only ever be loaded by another `__main__` — importing the
        class normally raised
        `AttributeError: module '__main__' has no attribute 'CDFCalibrator'`.
        Serialising state rather than the object sidesteps that entirely and
        also decouples the artifact from the sklearn version.
        """
        joblib.dump(self.to_state(), Path(path))

    @classmethod
    def from_state(cls, state: dict) -> "CDFCalibrator":
        if not isinstance(state, dict) or state.get("__type__") != "CDFCalibrator":
            raise ValueError(
                "This calibrator file was written by an older version that "
                "pickled the object itself. Delete it and re-run:\n"
                "    python src/inference/calibration.py --dataset all"
            )
        if state.get("version") != cls.STATE_VERSION:
            raise ValueError(
                f"Calibrator state version {state.get('version')} does not match "
                f"the expected {cls.STATE_VERSION}. Re-run "
                f"src/inference/calibration.py."
            )
        return cls(
            state["p_grid"], state["q_grid"],
            n_cal=state["n_cal"],
            dataset_key=state["dataset_key"],
            sigma_scale=state["sigma_scale"],
        )

    @staticmethod
    def load(path: Path) -> "CDFCalibrator":
        return CDFCalibrator.from_state(joblib.load(Path(path)))


# ==========================================================================
# Fitting script
# ==========================================================================

def _create_sequences(df: pd.DataFrame, seq_length: int):
    """
    Sliding-window sequences with targets. Local copy so this module does
    not import from src/evaluation — same convention as health_monitor.py.
    """
    cols_to_drop = [c for c in NON_FEATURE_COLS if c in df.columns]
    X_seq, y = [], []
    for unit in df["unit_number"].unique():
        u = df[df["unit_number"] == unit].sort_values("time_in_cycles")
        Xu = u.drop(columns=cols_to_drop).values
        yu = u[TARGET_COL].values
        for i in range(len(Xu) - seq_length + 1):
            X_seq.append(Xu[i:i + seq_length])
            y.append(yu[i + seq_length - 1])
    if not X_seq:
        return torch.empty((0, seq_length, 1)), np.array([])
    return torch.tensor(np.array(X_seq), dtype=torch.float32), np.array(y)


def fit_for_dataset(dataset_key: str, registry_path: Path,
                    exclude_censored: bool = False) -> None:
    """
    Fit the recalibrator on the VALIDATION split and register it.

    Validation, not test — same rule the sigma_scale calibration already
    follows. The test set stays untouched, so the coverage numbers in
    metrics_summary.json remain an honest held-out measurement of whether
    the recalibration actually worked.
    """
    print(f"\n  Distributional calibration - {dataset_key}")

    registry = load_registry(registry_path)
    if dataset_key not in registry or "champion" not in registry[dataset_key]:
        print(f"  No champion for {dataset_key}. Run train.py first. Skipping.")
        return

    champion = registry[dataset_key]["champion"]
    bb_cfg = champion["backbone_config"]
    seq_length = bb_cfg["seq_length"]
    sigma_scale = float(champion.get("calibration_scale", 1.0))

    bb_kwargs = {k: v for k, v in bb_cfg.items()
                 if k not in ("input_dim", "seq_length")}
    model = build_backbone(champion["backbone"], bb_cfg["input_dim"], bb_kwargs)
    model.load_state_dict(torch.load(Path(champion["artifacts"]["backbone"]),
                                     map_location=DEVICE, weights_only=True))
    model.eval()
    ngb = joblib.load(Path(champion["artifacts"]["meta_ngboost"]))

    feature_path = _ROOT / "data" / "processed" / dataset_key / "train_features.csv"
    full_df = pd.read_csv(feature_path)
    # Same seed and function as train.py, so this is the same held-out split
    # the model never fitted on.
    _, val_df = split_by_engine(full_df)

    X_val, y_val = _create_sequences(val_df, seq_length)
    print(f"  Calibration set: {val_df['unit_number'].nunique()} val engines, "
          f"{len(X_val):,} windows")

    from torch.utils.data import DataLoader, TensorDataset
    loader = DataLoader(TensorDataset(X_val), batch_size=256, shuffle=False)
    feats = []
    with torch.no_grad():
        for (xb,) in loader:
            feats.append(model.encode(xb.to(DEVICE)).cpu().numpy())
    dist = ngb.pred_dist(np.concatenate(feats, axis=0))

    mu = np.asarray(dist.loc, dtype=np.float64)
    sigma = np.asarray(dist.scale, dtype=np.float64) * sigma_scale

    # Read max_rul so the censoring diagnostic below has something to
    # report. Set exclude_censored_at=max_rul in the fit call to build the
    # uncensored-only variant instead (see CDFCalibrator.fit).
    ds_cfg_path = _ROOT / "config" / "datasets.yaml"
    max_rul = None
    if ds_cfg_path.exists():
        with open(ds_cfg_path, "r") as f:
            max_rul = (yaml.safe_load(f) or {}).get(dataset_key, {}).get("max_rul")

    if max_rul is not None:
        n_cens = int((y_val >= max_rul).sum())
        print(f"  Censored windows in the calibration set: {n_cens:,} / "
              f"{len(y_val):,} ({n_cens/len(y_val):.1%}) at the cap of {max_rul:g}")

    cal = CDFCalibrator.fit(y_val, mu, sigma, dataset_key, sigma_scale,
                            exclude_censored_at=(max_rul if exclude_censored else None))

    # Report before/after coverage ON THE CALIBRATION SET. These will look
    # excellent by construction — the map was fitted here. The number that
    # matters is the test-set coverage in metrics_summary.json after
    # re-running evaluate.py.
    print(f"  sigma_scale in force: {sigma_scale:.4f}")
    print(f"  {'level':>7}{'raw cov':>10}{'recal cov':>12}   (calibration set — in-sample)")
    for lv in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        lo_q, hi_q = (1 - lv) / 2, (1 + lv) / 2
        raw_lo = mu + sigma * norm.ppf(lo_q)
        raw_hi = mu + sigma * norm.ppf(hi_q)
        raw_cov = float(((y_val >= raw_lo) & (y_val <= raw_hi)).mean())

        c_lo, c_hi = cal.interval(lv, mu, sigma)
        rec_cov = float(((y_val >= c_lo) & (y_val <= c_hi)).mean())
        print(f"  {lv:>7.2f}{raw_cov:>10.3f}{rec_cov:>12.3f}")

    print("\n  What the map does to interval widths:")
    print(cal.coverage_table().round(4).to_string(index=False))

    out_dir = _ROOT / "models" / dataset_key
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cdf_calibrator.pkl"
    cal.save(out_path)
    print(f"\n  Saved: {out_path}")

    current = load_registry(registry_path)
    current[dataset_key]["champion"]["artifacts"]["cdf_calibrator"] = str(out_path)
    current[dataset_key]["champion"]["cdf_calibration"] = {
        "method": "isotonic_pit_recalibration",
        "reference": "Kuleshov et al. 2018",
        "fitted_on": "validation",
        "n_calibration_windows": int(cal.n_cal),
        "sigma_scale_at_fit": sigma_scale,
        "excluded_censored": bool(exclude_censored),
        "max_rul": max_rul,
    }
    save_registry(registry_path, current)
    print(f"  Registry updated -> cdf_calibrator registered for {dataset_key}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fit isotonic CDF recalibration on the validation split."
    )
    parser.add_argument("--dataset",
                        choices=["FD001", "FD002", "FD003", "FD004", "all"],
                        default="all")
    parser.add_argument(
        "--exclude-censored", action="store_true",
        help=("Fit the map only on windows whose true RUL is below max_rul. "
              "Between 27%% (FD001) and 62%% (FD004) of windows sit exactly at "
              "the cap, where the target is a constant rather than a remaining "
              "life; including them makes the map correct the cap instead of "
              "the model. Measured effect of NOT excluding them: marginal "
              "coverage error improves, uncensored coverage error gets worse "
              "on all four datasets."))
    args = parser.parse_args()

    registry_path = _ROOT / "config" / "model_registry.yaml"
    datasets = (["FD001", "FD002", "FD003", "FD004"]
                if args.dataset == "all" else [args.dataset])

    print("Distributional recalibration (isotonic on PIT values)")
    print("Corrects the SHAPE of the predictive distribution; sigma_scale only")
    print("corrects its width and can fix exactly one quantile level.")

    if args.exclude_censored:
        print("Fitting on UNCENSORED windows only (true RUL < max_rul).")
    else:
        print("Fitting on all windows, censored included "
              "(pass --exclude-censored to change).")

    for ds in datasets:
        fit_for_dataset(ds, registry_path, exclude_censored=args.exclude_censored)