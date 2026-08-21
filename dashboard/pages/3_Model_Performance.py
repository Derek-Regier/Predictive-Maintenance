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
    reg  = registry.get(ds, {}).get("champion", {})

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

**Coverage 90%** is the calibration check for NGBoost's uncertainty estimates.
A well-calibrated model should show ~0.90. Below 0.90 = overconfident (intervals too narrow).
Above 0.90 = underconfident (intervals too wide).
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
if non_empty_cal:
    cal_fig = build_calibration_subplots(non_empty_cal)
    st.plotly_chart(cal_fig, use_container_width=True)
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