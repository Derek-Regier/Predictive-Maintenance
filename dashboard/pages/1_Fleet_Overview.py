"""
dashboard/pages/1_Fleet_Overview.py

Fleet Overview — the landing page engineers see first.

Layout
------
Sidebar      : dataset selector, alert tier filter, fleet risk summary
Top row      : four KPI cards (CRITICAL / WARNING / MONITOR / NOMINAL counts)
Middle row   : alert donut chart | cross-dataset risk comparison bar
Main area    : fleet engine table (clickable row → Engine Deep Dive)

Streamlit concepts used here
-----------------------------
st.columns()     — side-by-side layout
st.metric()      — KPI card with optional delta indicator
st.dataframe()   — interactive table with selection and column formatting
st.session_state — persists selected engine across page navigation
st.switch_page() — programmatic navigation to another page
"""

import sys
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
# Adds dashboard/ to sys.path so utils imports work regardless of where
# Streamlit was launched from.

_DASHBOARD = Path(__file__).resolve().parents[1]
if str(_DASHBOARD) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD))

import streamlit as st
import pandas as pd

from utils.data_loader import (
    load_predictions_last,
    load_metrics_summary,
    load_all_metrics,
    DATASETS,
)
from utils.charts import build_alert_donut, build_cross_dataset_risk_bar
from utils.styles import ALERT_COLORS, CUSTOM_CSS, HEADER_HTML, SIGNATURE_HTML

# ── Page config + CSS ─────────────────────────────────────────────────────────
# Each page needs its own set_page_config call. layout="wide" is required
# on every page — not just app.py.

st.set_page_config(
    page_title="Fleet Overview — Predictive Maintenance",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.markdown(HEADER_HTML, unsafe_allow_html=True)
st.markdown(SIGNATURE_HTML, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# Widgets in st.sidebar persist visually but DO NOT persist their values
# across page navigations unless stored in st.session_state.
# We save the selected dataset to session_state so Engine Deep Dive
# automatically shows the right dataset when navigated to.
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### Controls")

    # Dataset selector — saves to session_state so other pages inherit it
    selected_dataset = st.selectbox(
        "Dataset",
        DATASETS,
        index=DATASETS.index(st.session_state.get("selected_dataset", "FD001")),
        help=(
            "FD001/FD003: single operating condition\n"
            "FD002/FD004: six operating conditions\n"
            "FD003/FD004: includes HPC fault mode"
        ),
    )
    st.session_state["selected_dataset"] = selected_dataset

    st.divider()

    # Alert tier filter — controls which engines appear in the fleet table
    st.markdown("**Filter alert tiers**")
    show_tiers = st.multiselect(
        "Show tiers",
        ["CRITICAL", "WARNING", "MONITOR", "NOMINAL"],
        default=["CRITICAL", "WARNING", "MONITOR", "NOMINAL"],
        label_visibility="collapsed",
    )

    # Show padded engines option (engines with short history)
    show_padded = st.checkbox(
        "Show short-history engines",
        value=True,
        help="Engines with fewer cycles than the model's sequence length were "
             "padded with repeated rows. Their predictions are less reliable.",
    )

    st.divider()
    st.markdown(
        "<small>Click a row in the fleet table to open that engine's "
        "detailed trajectory on the Engine Deep Dive page.</small>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# All data loading is done through cached functions in data_loader.py.
# On the first run these read from disk; on reruns they return from cache.
# ─────────────────────────────────────────────────────────────────────────────

predictions  = load_predictions_last(selected_dataset)
metrics      = load_metrics_summary(selected_dataset)
all_metrics  = load_all_metrics()

if predictions.empty:
    st.error(
        f"No prediction data found for {selected_dataset}. "
        "Run `python src/evaluation/evaluate.py --dataset all` first."
    )
    st.stop()   # st.stop() halts execution of the rest of the page


# ─────────────────────────────────────────────────────────────────────────────
# PAGE HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.title("Fleet Overview")
st.caption(
    f"Dataset: **{selected_dataset}**  ·  "
    f"{predictions['unit_number'].nunique()} test engines  ·  "
    f"Backbone: **{metrics.get('backbone', 'N/A')}**  ·  "
    f"σ scale: **{metrics.get('sigma_scale', 1.0):.4f}**"
)


# ─────────────────────────────────────────────────────────────────────────────
# KPI CARDS — alert tier counts
# st.metric() creates a card with a large number and optional delta.
# The four columns divide the full page width equally.
# ─────────────────────────────────────────────────────────────────────────────

tier_counts = predictions["alert_tier"].value_counts()

col1, col2, col3, col4 = st.columns(4)

with col1:
    n_crit = int(tier_counts.get("CRITICAL", 0))
    st.metric(
        label="🔴 CRITICAL",
        value=n_crit,
        help="P(RUL < 20 cycles) above threshold — immediate action required",
    )

with col2:
    n_warn = int(tier_counts.get("WARNING", 0))
    st.metric(
        label="🟡 WARNING",
        value=n_warn,
        help="P(RUL < 50 cycles) above threshold — schedule maintenance",
    )

with col3:
    n_mon = int(tier_counts.get("MONITOR", 0))
    st.metric(
        label="🔵 MONITOR",
        value=n_mon,
        help="P(RUL < 50 cycles) above lower threshold — flag for inspection",
    )

with col4:
    n_nom = int(tier_counts.get("NOMINAL", 0))
    st.metric(
        label="🟢 NOMINAL",
        value=n_nom,
        help="No alert — engine operating within expected parameters",
    )


st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# CHARTS ROW — donut + cross-dataset risk bar
# st.columns([ratio1, ratio2]) creates unequal columns.
# The donut gets 1/3 of the width; the risk bar gets 2/3.
# ─────────────────────────────────────────────────────────────────────────────

chart_col1, chart_col2 = st.columns([1, 2])

with chart_col1:
    st.markdown("**Alert distribution**")
    # Build the donut chart only if there are engines in at least one tier
    if tier_counts.sum() > 0:
        donut_fig = build_alert_donut(tier_counts.to_dict())
        st.plotly_chart(donut_fig, use_container_width=True)

with chart_col2:
    st.markdown("**Expected failures across all datasets**")
    # Cross-dataset comparison — loads metrics for all four datasets
    risk_fig = build_cross_dataset_risk_bar(all_metrics)
    st.plotly_chart(risk_fig, use_container_width=True)


st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# FLEET RISK SUMMARY
# Expected failures = Σ P(RUL < N) across all fleet engines.
# This is an actuarial number: if 30 engines each have P=0.33, the
# expected number of failures is 10, not 30.
# ─────────────────────────────────────────────────────────────────────────────

fleet_risk = metrics.get("fleet_risk", {})
if fleet_risk:
    st.markdown("### Fleet Risk Summary")
    risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)

    with risk_col1:
        st.metric(
            "Expected failures (20 cycles)",
            f"{fleet_risk.get('expected_failures_20', 0):.1f}",
            help="Σ P(RUL < 20) across all fleet engines",
        )
    with risk_col2:
        st.metric(
            "Expected failures (50 cycles)",
            f"{fleet_risk.get('expected_failures_50', 0):.1f}",
            help="Σ P(RUL < 50) across all fleet engines",
        )
    with risk_col3:
        st.metric(
            "Engines evaluated",
            fleet_risk.get("n_test_engines", "—"),
        )
    with risk_col4:
        n_padded = int(predictions.get("padded", pd.Series(dtype=bool)).sum())
        st.metric(
            "Short-history engines",
            n_padded,
            help="Engines with fewer cycles than the model's sequence length. "
                 "Padded with repeated rows — predictions less reliable.",
        )

    st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# FLEET TABLE
# st.dataframe() with on_select="rerun" and selection_mode="single-row"
# makes the table interactive: clicking a row reruns the script with
# that row stored in event.selection.rows.
#
# We then:
#   1. Store the selected engine in st.session_state["selected_engine"]
#   2. Call st.switch_page() to navigate to Engine Deep Dive
#
# column_config controls how each column is displayed — labels, format,
# help text. Columns not in column_config are shown with their raw name.
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("### Engine Fleet Table")
st.caption("Click any row to open that engine's full trajectory on the Engine Deep Dive page.")

# Apply filters from sidebar
filtered = predictions.copy()
if show_tiers:
    filtered = filtered[filtered["alert_tier"].isin(show_tiers)]
if not show_padded:
    filtered = filtered[~filtered["padded"].astype(bool)]

# Sort: CRITICAL first, then WARNING, then MONITOR, then NOMINAL,
# then by predicted RUL ascending (most urgent at top)
tier_order   = {"CRITICAL": 0, "WARNING": 1, "MONITOR": 2, "NOMINAL": 3}
filtered     = filtered.copy()
filtered["tier_rank"] = filtered["alert_tier"].map(tier_order).fillna(4)
filtered     = filtered.sort_values(["tier_rank", "pred_rul"]).drop(columns="tier_rank")

# Columns to display in the table — rename for readability
display_cols = {
    "unit_number":     "Engine",
    "alert_display":   "Alert",
    "pred_rul":        "Predicted RUL",
    "pred_std":        "Uncertainty (σ)",
    "lower_90":        "90% PI Lower",
    "upper_90":        "90% PI Upper",
    "true_rul":        "True RUL",
    "prob_failure_20": "P(fail<20)",
    "prob_failure_50": "P(fail<50)",
    "residual":        "Residual",
    "padded":          "Short history",
}

# Only include columns that actually exist in the data
avail_cols   = [c for c in display_cols if c in filtered.columns]
display_df   = filtered[avail_cols].rename(columns=display_cols)

event = st.dataframe(
    display_df,
    use_container_width=True,
    selection_mode="single-row",
    on_select="rerun",                  # rerun the script when a row is selected
    hide_index=True,
    column_config={
        "Engine":          st.column_config.NumberColumn(format="%d"),
        "Predicted RUL":   st.column_config.NumberColumn(format="%.1f cycles"),
        "Uncertainty (σ)": st.column_config.NumberColumn(format="%.2f"),
        "90% PI Lower":    st.column_config.NumberColumn(format="%.1f"),
        "90% PI Upper":    st.column_config.NumberColumn(format="%.1f"),
        "True RUL":        st.column_config.NumberColumn(format="%.0f cycles"),
        "P(fail<20)":      st.column_config.ProgressColumn(
                               format="%.3f", min_value=0, max_value=1),
        "P(fail<50)":      st.column_config.ProgressColumn(
                               format="%.3f", min_value=0, max_value=1),
        "Residual":        st.column_config.NumberColumn(format="%.1f"),
        "Short history":   st.column_config.CheckboxColumn(),
    },
)

# ── Handle row selection → navigate to Engine Deep Dive ──────────────────────
if event.selection.rows:
    # event.selection.rows contains the row index in the displayed dataframe
    row_idx = event.selection.rows[0]
    selected_unit = int(display_df.iloc[row_idx]["Engine"])

    # Store both the engine ID and dataset so Engine Deep Dive can load them
    st.session_state["selected_engine"]  = selected_unit
    st.session_state["selected_dataset"] = selected_dataset

    # Navigate — this triggers a full page switch, not just a rerun
    st.switch_page("pages/2_Engine_Deep_Dive.py")


# ─────────────────────────────────────────────────────────────────────────────
# TABLE SUMMARY STATS — shown below the table
# ─────────────────────────────────────────────────────────────────────────────

if not filtered.empty:
    st.caption(
        f"Showing **{len(filtered)}** engines  ·  "
        f"Mean predicted RUL: **{filtered['pred_rul'].mean():.1f} cycles**  ·  "
        f"Min predicted RUL: **{filtered['pred_rul'].min():.1f} cycles**"
    )