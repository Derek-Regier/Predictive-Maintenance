"""
Model Performance

This page shows the technical validation of the trained models
across all four CMAPSS datasets, including:
  - Comparison table (RMSE, AUC, calibration, NASA score)
  - Calibration plots (actual vs expected coverage)
  - RMSE by RUL bucket (why end-of-life accuracy is what matters)
  - Within-N accuracy and AUC comparison

Intended audience: ML engineers and technically literate stakeholders.
"""

import sys
from pathlib import Path

_DASHBOARD = Path(__file__).resolve().parents[1]
if str(_DASHBOARD) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD))

import streamlit as st
import pandas as pd

from utils.data_loader import (
    load_all_metrics,
    load_all_calibration,
    load_all_bucket_metrics,
    load_registry,
    DATASETS,
)
from utils.charts import (
    build_calibration_subplots,
    build_bucket_rmse_chart,
    build_within_n_chart,
    build_auc_comparison,
)
from utils.styles import DATASET_COLORS, CUSTOM_CSS, HEADER_HTML, SIGNATURE_HTML

st.set_page_config(
    page_title="Model Performance - Predictive Maintenance",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.markdown(HEADER_HTML, unsafe_allow_html=True)
st.markdown(SIGNATURE_HTML, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Model Performance")
    st.markdown(
        "This page shows evaluation results on the held-out test set "
        "across all four CMAPSS datasets.\n\n"
        "**Last-timestep** = one prediction per engine at its final "
        "observed cycle (the operationally honest metric).\n\n"
        "**All-sequence** = every sliding window (driven by training signal)."
    )

st.title("Model Performance")
st.caption(
    "Evaluation on held-out test sets"
)

# DATA LOADING

all_metrics = load_all_metrics()
all_cal = load_all_calibration()
all_bucket = load_all_bucket_metrics()
registry = load_registry()

# COMPARISON TABLE

st.markdown("### Dataset Comparison - Last-Timestep Evaluation")
st.caption(
    "One prediction per engine at its final observed cycle. "

)

# Build summary rows from metrics_summary.json for all datasets
rows = []
for ds in DATASETS:
    m   = all_metrics.get(ds, {})
    lt  = m.get("last_timestep", {})
    aseq = m.get("all_seq", {})
    cal  = m.get("calibration", {})
    rec  = m.get("calibration_recalibrated") or {}
    unc  = m.get("calibration_uncensored") or {}
    reg  = registry.get(ds, {}).get("champion", {})

    # n_seq: total evaluated windows, used only to express the censored
    # count as a percentage. Not stored directly, so derive it from the
    # bucket table when available.
    _bkt = all_bucket.get(ds, pd.DataFrame())
    n_seq = int(_bkt["n_samples"].sum()) if not _bkt.empty else None

    rows.append({
        "Dataset": ds,
        "Backbone": reg.get("backbone", "—").upper(),
        "RMSE": round(lt.get("rmse", 0), 2),
        "MAE": round(lt.get("mae", 0), 2),
        "NASA Score": round(lt.get("nasa_score", 0), 1),
        "Within 10 cyc": f"{lt.get('within_10_pct', 0)*100:.1f}%",
        "AUC-20": round(aseq.get("auc_failure_20", 0), 4)
                          if aseq.get("auc_failure_20") else "—",
        "Coverage 90%": round(cal.get(0.9, cal.get("0.9", 0)), 3),
        "Cov 90% recal": (round(rec.get(0.9, rec.get("0.9", 0)), 3)
                          if rec else "—"),
        "Cov 90% uncens": (round(unc.get(0.9, unc.get("0.9", 0)), 3)
                           if unc else "—"),
        "% at RUL cap": (f"{m['n_windows_at_rul_cap'] / n_seq:.0%}"
                         if m.get("n_windows_at_rul_cap") and n_seq else "—"),
        "σ scale": round(m.get("sigma_scale", 1.0), 4),
    })

compare_df = pd.DataFrame(rows)

# Colour-code the RMSE column: lower = greener
def _color_rmse(val):
    """Green if RMSE ≤ 10, yellow ≤ 14, red otherwise."""
    if not isinstance(val, (int, float)):
        return ""
    if val <= 10: return "background-color: #D1FAE5; color: #065F46"
    if val <= 14: return "background-color: #FEF3C7; color: #92400E"
    return  "background-color: #FEE2E2; color: #991B1B"

styled = compare_df.style.map(_color_rmse, subset=["RMSE"])

st.dataframe(styled, use_container_width=True, hide_index=True)

# KEY METRICS EXPLANATION

with st.expander("What these metrics mean"):
    st.markdown("""
| Metric | What it measures | Better when |
|--------|-----------------|-------------|
| **RMSE** | Root mean squared error of RUL predictions at last timestep | Lower |
| **NASA Score** | Asymmetric penalty — late predictions (dangerous) penalised more heavily | Lower |
| **Within 10 cyc** | % of predictions within ±10 cycles of true RUL | Higher |
| **AUC-20** | Ability to rank near-failure engines above healthy ones (20-cycle horizon) | Higher (max 1.0) |
| **Coverage 90%** | Fraction of true RUL values inside the stated 90% prediction interval | ~0.90 |
| **σ scale** | Post-hoc calibration multiplier applied to NGBoost sigma | Closest to 1.0 |

| **Cov 90% recal** | Coverage after isotonic CDF recalibration (Kuleshov et al. 2018) | ~0.90 |
| **Cov 90% uncens** | Coverage on windows whose true RUL is below the 125 cap | ~0.90 |
| **% at RUL cap** | Share of evaluation windows whose target is the cap, not a real RUL | Context only |

**Coverage 90%** is the calibration check for NGBoost's uncertainty estimates.
A well-calibrated model should show ~0.90. Below 0.90 = overconfident (intervals too narrow).
Above 0.90 = underconfident (intervals too wide).

**Why `sigma scale` alone is not enough.** It is a single multiplier, so it can
make exactly one confidence level correct — `calibration_alpha` targeted 0.90,
and every other level drifted. Measured on the test sets, the scale factor each
level would need ranged from 0.54 to 0.96 within FD004 alone. No scalar covers
that spread, which is why the isotonic CDF map was added: it corrects the shape
of the distribution rather than only its width, and being monotone it leaves the
AUC columns untouched.

**Why the uncensored column matters more than it looks.** Coverage measured over
all windows is dominated by targets pinned at the 125 cap. On FD004 the marginal
mean coverage error is 0.142 while the uncensored figure is 0.036 — the model is
roughly four times better calibrated than the headline suggests, on the windows
where the target is a real remaining life.
    """)


st.divider()

# CALIBRATION PLOTS
st.markdown("### Calibration Plots")
st.caption(
    "Each plot shows actual coverage vs expected coverage at six confidence levels. "
    "A perfectly calibrated model has all dots on the grey diagonal. "
    "**Below the diagonal** means intervals too narrow and model is overconfident. "
    "**Above the diagonal** means intervals too wide and model is underconfident."
)

non_empty_cal = {ds: df for ds, df in all_cal.items() if not df.empty}

# Which coverage curves are available depends on how evaluate.py was run.
_SERIES_LABELS = {
    "actual_coverage": "Raw (sigma scaled)",
    "recal_coverage": "+ CDF recalibration",
    "uncensored_coverage": "Raw, uncensored only",
    "uncensored_recal_coverage": "Recal, uncensored only",
}
_available = []
for _col in _SERIES_LABELS:
    if any(_col in df.columns for df in non_empty_cal.values()):
        _available.append(_col)

if non_empty_cal:
    if len(_available) > 1:
        chosen_series = st.multiselect(
            "Coverage curves to plot",
            _available,
            default=[c for c in ("actual_coverage", "recal_coverage",
                                 "uncensored_coverage") if c in _available],
            format_func=lambda c: _SERIES_LABELS[c],
            key="calibration_series",
        )
    else:
        chosen_series = _available

    cal_fig = build_calibration_subplots(non_empty_cal, series=chosen_series or None)
    st.plotly_chart(cal_fig, use_container_width=True)

    # ---- censoring note --------------------------------------------------
    if "uncensored_coverage" in _available:
        st.info(
            "**Why there are two sets of curves.** RUL targets are capped at "
            "125, and a large share of evaluation windows sit exactly at that "
            "cap — 27% on FD001 up to 62% on FD004. For those windows the "
            "target is a constant rather than a remaining life, the model "
            "predicts close to it, and the residual is near-deterministic, so "
            "they land inside every interval and inflate coverage. The "
            "uncensored curves drop them and show how the model behaves where "
            "the target actually means something."
        )

    # ---- four-way calibration scorecard ----------------------------------
    st.markdown("**Mean absolute coverage error**")
    st.caption(
        "Averaged over the six confidence levels. Lower is better. Read the "
        "rows against each other, not in isolation — the marginal and "
        "uncensored columns are measuring different populations."
    )

    score_rows = []
    for ds, df in non_empty_cal.items():
        row = {"Dataset": ds}
        for col, label in _SERIES_LABELS.items():
            if col in df.columns:
                err = (df[col] - df["expected_coverage"]).abs().mean()
                row[label] = round(float(err), 4)
        score_rows.append(row)

    if score_rows:
        score_df = pd.DataFrame(score_rows)

        def _color_err(v):
            if not isinstance(v, (int, float)):
                return ""
            if v <= 0.03:  return "background-color: #D1FAE5; color: #065F46"
            if v <= 0.08:  return "background-color: #FEF3C7; color: #92400E"
            return "background-color: #FEE2E2; color: #991B1B"

        num_cols = [c for c in score_df.columns if c != "Dataset"]
        st.dataframe(
            score_df.style.map(_color_err, subset=num_cols),
            use_container_width=True, hide_index=True,
        )

        # Verdict computed from the table rather than asserted. Which way
        # this reads depends on whether the calibrator was fitted with
        # --exclude-censored, so hardcoding a message here goes stale the
        # moment the fit changes.
        _raw_u, _rec_u = "Raw, uncensored only", "Recal, uncensored only"
        if _raw_u in score_df.columns and _rec_u in score_df.columns:
            better = int((score_df[_rec_u] < score_df[_raw_u]).sum())
            total = int(score_df[_rec_u].notna().sum())
            if better >= total - 1:
                st.success(
                    f"**Recalibration improves the uncensored column on "
                    f"{better} of {total} datasets.** That column is the one "
                    "that describes the model rather than the RUL cap, so it "
                    "is the one to quote. The marginal column may look worse "
                    "at the same time — expected, since the map is no longer "
                    "spending its capacity fitting capped targets."
                )
            else:
                st.warning(
                    f"**Recalibration improves the uncensored column on only "
                    f"{better} of {total} datasets.** If the calibrators were "
                    "fitted on all windows, refit them on uncensored ones:\n\n"
                    "`python src/inference/calibration.py --dataset all "
                    "--exclude-censored`"
                )

        # Guard against a specific degenerate reading. When a large block of
        # censored windows has near-identical PIT values, a narrow interval
        # can land just to one side of that block and the marginal coverage
        # collapses. FD004 at level 0.50 is the live example: 0.163 marginal
        # against 0.431 uncensored implies ~0.0007 coverage on the 18,041
        # capped windows. That is an artefact of where the block sits, not a
        # model failure, and the marginal number should not be quoted.
        for _ds, _df in non_empty_cal.items():
            if "recal_coverage" not in _df.columns or _rec_u not in score_df.columns:
                continue
            if "uncensored_recal_coverage" not in _df.columns:
                continue
            _gap = (_df["uncensored_recal_coverage"] - _df["recal_coverage"]).max()
            if _gap > 0.20:
                st.caption(
                    f"Note on {_ds}: the marginal recalibrated coverage falls "
                    f"more than {_gap:.2f} below the uncensored figure at some "
                    "level. That happens when a dense block of capped targets "
                    "sits just outside a narrowed interval. Read the "
                    "uncensored curve for this dataset."
                )
else:
    st.info("No calibration data found. Run evaluate.py first.")


st.divider()


# BUCKET METRICS
st.markdown("### RMSE by RUL Bucket")
st.caption(
    "Prediction accuracy split into three zones: "
    "**Early life** (RUL > 80), **Mid life** (30–80), **End of life** (≤ 30). "
    "End-of-life is where maintenance decisions are made."
)

non_empty_bkt = {ds: df for ds, df in all_bucket.items() if not df.empty}
if non_empty_bkt:
    bkt_fig = build_bucket_rmse_chart(non_empty_bkt)
    st.plotly_chart(bkt_fig, use_container_width=True)

    # Show the raw bucket table in an expander
    with st.expander("Full bucket metrics table"):
        for ds, bkt_df in non_empty_bkt.items():
            st.markdown(f"**{ds}**")
            st.dataframe(bkt_df.round(3), use_container_width=True, hide_index=True)


st.divider()

# WITHIN-N AND AUC CHARTS

perf_col1, perf_col2 = st.columns(2)

with perf_col1:
    st.markdown("### Within-N Cycle Accuracy")
    st.caption(
        "What fraction of predictions land within ±N cycles of the true RUL. "
        "Within 10 cycles is roughly one maintenance window."
    )
    within_fig = build_within_n_chart(all_metrics)
    st.plotly_chart(within_fig, use_container_width=True)

with perf_col2:
    st.markdown("### Failure Ranking Quality (AUC)")
    st.caption(
        "AUC measures how well the model ranks near-failure engines above "
        "healthy ones using P(RUL < N) as the score. "
        "AUC = 1.0 is perfect ranking."
    )
    auc_fig = build_auc_comparison(all_metrics)
    st.plotly_chart(auc_fig, use_container_width=True)


st.divider()

# MODEL REGISTRY DETAILS

st.markdown("### Trained Model Configurations")

for ds in DATASETS:
    champion = registry.get(ds, {}).get("champion", {})
    if not champion:
        continue

    with st.expander(f"{ds} — {champion.get('backbone', '').upper()} backbone"):
        cfg_col1, cfg_col2, cfg_col3 = st.columns(3)

        bb_cfg = champion.get("backbone_config", {})
        meta   = champion.get("meta_model_config", {})
        retrain = champion.get("backbone_retrain", {})

        with cfg_col1:
            st.markdown("**Backbone**")
            for k, v in bb_cfg.items():
                st.markdown(f"- `{k}`: `{v}`")

        with cfg_col2:
            st.markdown("**NGBoost meta-model**")
            for k, v in meta.items():
                st.markdown(f"- `{k}`: `{v}`")

        with cfg_col3:
            st.markdown("**Training details**")
            st.markdown(f"- Trained: `{champion.get('trained_at', '—')}`")
            st.markdown(f"- σ scale: `{champion.get('calibration_scale', 1.0):.4f}`")
            if retrain:
                st.markdown(f"- Backbone epochs: `{retrain.get('epochs', '—')}`")
                st.markdown(f"- Best val RMSE: `{retrain.get('best_val_rmse', 0):.4f}`")