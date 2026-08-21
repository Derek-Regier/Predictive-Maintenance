"""
Engine Deep Dive.

This page shows the full degradation history of one selected engine:
  - RUL trajectory with confidence intervals
  - Failure probability evolution over time
  - VAE health index trajectory
  - Prediction residuals over time

Navigation
----------
Arrives here from Fleet Overview 

Streamlit concepts used here
-----------------------------
st.tabs() -> multiple charts in one area, user switches tabs
st.session_state -> reads selected_engine set by Fleet Overview
st.selectbox() -> manual engine override in sidebar
st.expander() -> collapsible technical detail section
"""

import sys
from pathlib import Path

_DASHBOARD = Path(__file__).resolve().parents[1]
if str(_DASHBOARD) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD))

import streamlit as st
import pandas as pd

from utils.data_loader import (
    load_predictions_all,
    load_predictions_last,
    load_health_indices,
    load_health_summary,
    DATASETS,
)
from utils.charts import (
    build_rul_trajectory,
    build_failure_probability,
    build_health_index,
    build_residual_scatter,
)
from utils.styles import ALERT_COLORS, ALERT_BG_COLORS, CUSTOM_CSS, HEADER_HTML, SIGNATURE_HTML

st.set_page_config(
    page_title="Engine Deep Dive",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.markdown(HEADER_HTML, unsafe_allow_html=True)
st.markdown(SIGNATURE_HTML, unsafe_allow_html=True)



# SIDEBAR

with st.sidebar:
    st.markdown("### Engine Selection")

    # Dataset selector
    selected_dataset = st.selectbox(
        "Dataset",
        DATASETS,
        index=DATASETS.index(st.session_state.get("selected_dataset", "FD001")),
    )
    st.session_state["selected_dataset"] = selected_dataset

    # Load fleet data to populate the engine dropdown
    pred_last = load_predictions_last(selected_dataset)
    all_units = sorted(pred_last["unit_number"].unique().tolist()) if not pred_last.empty else [1]

    # Pre-select the engine navigated from Fleet Overview, if any
    nav_engine     = st.session_state.get("selected_engine", all_units[0])
    nav_engine_idx = all_units.index(nav_engine) if nav_engine in all_units else 0

    selected_engine = st.selectbox(
        "Engine",
        all_units,
        index=nav_engine_idx,
        help="Select any test engine. Navigate here from the fleet table "
             "to pre-select an engine.",
    )
    # Update session state so returning to Fleet Overview keeps context
    st.session_state["selected_engine"] = selected_engine

    st.divider()

    # Quick navigation back to fleet table
    if st.button("← Back to Fleet Overview"):
        st.switch_page("pages/1_Fleet_Overview.py")


# DATA LOADING

pred_all = load_predictions_all(selected_dataset)
pred_last = load_predictions_last(selected_dataset)
health_all = load_health_indices(selected_dataset)
health_summ = load_health_summary(selected_dataset)

# Filter all DataFrames to the selected engine
engine_pred_all = pred_all[pred_all["unit_number"] == selected_engine].sort_values("cycle") \
                   if not pred_all.empty else pd.DataFrame()
engine_pred_last = pred_last[pred_last["unit_number"] == selected_engine] \
                   if not pred_last.empty else pd.DataFrame()
engine_health = health_all[health_all["unit_number"] == selected_engine].sort_values("cycle") \
                   if not health_all.empty else pd.DataFrame()

if engine_pred_all.empty:
    st.error(
        f"No trajectory data found for engine {selected_engine} "
        f"in dataset {selected_dataset}."
    )
    st.stop()


# PAGE HEADER

st.title(f"Engine {selected_engine} — Deep Dive")

# Pull the summary prediction for this engine (last-timestep row)
if not engine_pred_last.empty:
    row = engine_pred_last.iloc[0]
    tier = row.get("alert_tier", "NOMINAL")
    tier_color = ALERT_COLORS.get(tier, "#6B7280")
    tier_bg = ALERT_BG_COLORS.get(tier, "#F9FAFB")
    padded = bool(row.get("padded", False))

    # Status banner coloured by alert tier
    st.markdown(
        f"""
        <div style="
            background-color: {tier_bg};
            border-left: 5px solid {tier_color};
            border-radius: 6px;
            padding: 12px 18px;
            margin-bottom: 1rem;
        ">
            <b style="color:{tier_color}">
                {'🔴' if tier=='CRITICAL' else '🟡' if tier=='WARNING'
                 else '🔵' if tier=='MONITOR' else '🟢'} {tier}
            </b>
            &nbsp;·&nbsp;
            Dataset: <b>{selected_dataset}</b>
            &nbsp;·&nbsp;
            {f'<span style="color:#D97706"> Short history engine (padded)</span>' if padded else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Four summary metrics across the top
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Predicted RUL", f"{row['pred_rul']:.1f} cycles")
    with m2:
        st.metric("Uncertainty (σ)", f"±{row['pred_std']:.1f} cycles")
    with m3:
        st.metric("P(fail < 20 cycles)", f"{row['prob_failure_20']:.1%}")
    with m4:
        st.metric("P(fail < 50 cycles)", f"{row['prob_failure_50']:.1%}")

    st.caption(
        f"True RUL: **{row['true_rul']:.0f} cycles**  ·  "
        f"Residual: **{row['residual']:+.1f} cycles**  "
        f"({'late' if row['residual'] > 0 else 'early'} prediction)  ·  "
        f"90% PI: [{row['lower_90']:.1f}, {row['upper_90']:.1f}]"
    )


st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# CHART TABS
# st.tabs() creates a tab bar. Content inside each `with tab:` block is
# shown only when that tab is active — the other content is hidden but
# still rendered, so switching tabs is instant.
# ─────────────────────────────────────────────────────────────────────────────

tab_labels = ["RUL Trajectory", "Failure Probability",
              "Health Index", "Residuals"]
tab1, tab2, tab3, tab4 = st.tabs(tab_labels)

# Tab 1: RUL Trajectory
with tab1:
    st.markdown(
        "The **blue line** is the model's predicted mean RUL. "
        "The **shaded band** is the calibrated 90% prediction interval — "
        "the model believes the true RUL falls within this range 90% of the time. "
        "The **dotted grey line** is the actual true RUL. "
        "The **red zone** marks the critical end-of-life region (RUL ≤ 30 cycles)."
    )
    rul_fig = build_rul_trajectory(engine_pred_all, selected_engine)
    st.plotly_chart(rul_fig, use_container_width=True)

# Tab 2: Failure Probability 
with tab2:
    st.markdown(
        "**P(RUL < 20)** (red) and **P(RUL < 50)** (amber) computed from "
        "NGBoost's calibrated Normal distribution. "
        "Horizontal dashed lines show where alert tiers fire. "
        "A well-behaved engine should see both curves rise as it approaches failure."
    )
    prob_fig = build_failure_probability(engine_pred_all, selected_engine)
    st.plotly_chart(prob_fig, use_container_width=True)

# Tab 3: Health Index (VAE) 
with tab3:
    if engine_health.empty:
        st.info(
            "Health index data not yet available for this dataset. "
            "Run: `python src/health/health_monitor.py --dataset all`"
        )
    else:
        st.markdown(
            "Four VAE-derived health indices, each normalised to **[0, 1]** "
            "per engine so they're visually comparable. "
            "**Reconstruction error** (red) is the primary signal - it rises "
            "as the engine's sensor patterns deviate from the healthy baseline. "
            "The geometry distances (KL, JS, Wasserstein) are secondary signals "
            "derived from the VAE's latent space representation. "
            "Vertical dotted red lines mark **drift detected** windows."
        )

        # Get drift threshold for this dataset/cluster
        thresholds  = health_summ.get("drift_thresholds", {})
        cluster_0   = engine_health["op_cluster"].iloc[0] if "op_cluster" in engine_health.columns else 0
        drift_thr   = thresholds.get(str(cluster_0), thresholds.get("global"))

        health_fig = build_health_index(engine_health, selected_engine)
        st.plotly_chart(health_fig, use_container_width=True)

        # Health summary stats for this engine
        n_drift = int(engine_health["drift_flag"].sum()) if "drift_flag" in engine_health.columns else 0
        pct_drift = n_drift / len(engine_health) * 100 if len(engine_health) > 0 else 0
        st.caption(
            f"Drift-flagged windows: **{n_drift}** / {len(engine_health)} "
            f"({pct_drift:.1f}%)  ·  "
            f"Drift threshold: **{drift_thr:.5f}** (mean + 2σ of healthy training data)"
            if drift_thr else ""
        )

# Tab 4: Residuals
with tab4:
    st.markdown(
        "Prediction residual = **predicted RUL − true RUL**. "
        "🔴 Red points are **late predictions** (model predicted too much remaining life — "
        "operationally dangerous). "
        "🟢 Green points are **early predictions** (conservative — model scheduled "
        "maintenance earlier than necessary). "
        "Residuals should be centred near zero with no visible time trend."
    )
    res_fig = build_residual_scatter(engine_pred_all, selected_engine)
    st.plotly_chart(res_fig, use_container_width=True)


st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# TECHNICAL DETAILS (collapsed by default)
# st.expander() hides technical content behind a click but keeps the page
# clean while making details accessible for engineers who want them.
# ─────────────────────────────────────────────────────────────────────────────

with st.expander("Technical details and model configuration"):
    if not engine_pred_last.empty:
        row = engine_pred_last.iloc[0]
        tech_col1, tech_col2 = st.columns(2)

        with tech_col1:
            st.markdown("**Prediction**")
            st.markdown(f"- σ (raw NGBoost): `{row['pred_std'] / 1.0:.3f}` cycles")
            st.markdown(f"- σ (calibrated): `{row['pred_std']:.3f}` cycles")
            st.markdown(f"- 90% PI width: `{row['upper_90'] - row['lower_90']:.1f}` cycles")

        with tech_col2:
            st.markdown("**Engine info**")
            n_windows = len(engine_pred_all)
            n_cycles  = int(engine_pred_all["cycle"].max() - engine_pred_all["cycle"].min())
            st.markdown(f"- Observed cycles: `{n_cycles}`")
            st.markdown(f"- Sliding windows: `{n_windows}`")
            if not engine_health.empty:
                final_recon = engine_health["recon_error"].iloc[-1]
                st.markdown(f"- Final recon error: `{final_recon:.5f}`")