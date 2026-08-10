"""
dashboard/pages/4_Health_Monitor.py

Health Monitor — unsupervised VAE-based health monitoring.

The VAE was trained on healthy engine windows (RUL > threshold) and
never saw labels. It learned what "normal" sensor patterns look like.
At inference, deviations from that learned representation surface as:
  - Reconstruction error    (primary signal — works on all datasets)
  - KL / JS / Wasserstein   (geometry distances in latent space)
  - Drift flag              (reconstruction error > healthy baseline + 2σ)

This page shows those signals at the fleet level and per-engine.
"""

import sys
from pathlib import Path

_DASHBOARD = Path(__file__).resolve().parents[1]
if str(_DASHBOARD) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD))

import streamlit as st
import pandas as pd

from utils.data_loader import (
    load_health_indices,
    load_health_summary,
    load_predictions_last,
    DATASETS,
)
from utils.charts import (
    build_health_index,
    build_recon_error_fleet,
    build_latent_norm_vs_rul,
    build_drift_engine_bar,
)
from utils.styles import CUSTOM_CSS, SIGNATURE_HTML

st.set_page_config(
    page_title="Health Monitor — Predictive Maintenance",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.markdown(SIGNATURE_HTML, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### Health Monitor")

    selected_dataset = st.selectbox(
        "Dataset",
        DATASETS,
        index=DATASETS.index(st.session_state.get("selected_dataset", "FD001")),
    )
    st.session_state["selected_dataset"] = selected_dataset

    st.divider()
    st.markdown(
        "**How VAE health monitoring works:**\n\n"
        "1. VAE trained on healthy engines (RUL > 80)\n"
        "2. At inference, each engine's sensor window is encoded\n"
        "3. Reconstruction error = how well the VAE reproduces it\n"
        "4. High error = sensor pattern outside healthy distribution\n"
        "5. Geometry distances = deviation in latent space"
    )


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

health_df   = load_health_indices(selected_dataset)
health_summ = load_health_summary(selected_dataset)
pred_last   = load_predictions_last(selected_dataset)

st.title("Health Monitor")
st.caption(f"Dataset: **{selected_dataset}** — VAE-based unsupervised degradation detection")

# ── Handle missing health data ────────────────────────────────────────────────
if health_df.empty:
    st.warning(
        "Health index data not found for this dataset.\n\n"
        "Run the following commands:\n"
        "```bash\n"
        "python src/training/train_vae.py --dataset all\n"
        "python src/health/health_monitor.py --dataset all\n"
        "```"
    )
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# FLEET SUMMARY CARDS
# ─────────────────────────────────────────────────────────────────────────────

n_engines        = health_summ.get("n_engines", health_df["unit_number"].nunique())
n_drifted        = health_summ.get("n_drifted_engines", 0)
pct_drift        = health_summ.get("pct_windows_drifted", 0)
thresholds       = health_summ.get("drift_thresholds", {})
global_threshold = thresholds.get("global")

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric("Engines monitored", n_engines)
with m2:
    st.metric(
        "Engines with drift detected",
        n_drifted,
        help="At least one window exceeded the reconstruction error threshold",
    )
with m3:
    st.metric(
        "Windows flagged as drifted",
        f"{pct_drift:.1f}%",
        help="Fraction of all test windows above the drift threshold",
    )
with m4:
    if global_threshold is not None:
        st.metric(
            "Drift threshold (global)",
            f"{global_threshold:.5f}",
            help="Reconstruction error mean + 2σ computed on healthy training data",
        )

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# FLEET-LEVEL CHARTS
# Two charts side by side: reconstruction error scatter and engine drift bar
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("### Fleet-Wide Degradation Signal")

fleet_col1, fleet_col2 = st.columns([3, 2])

with fleet_col1:
    st.markdown("**Reconstruction error vs True RUL (all engines, all windows)**")
    st.caption(
        "Each point is one sliding window. The reconstruction error should rise "
        "as true RUL falls — confirming the VAE captures degradation in input space. "
        "Red = drift flagged, grey = normal."
    )
    recon_fig = build_recon_error_fleet(health_df, global_threshold)
    st.plotly_chart(recon_fig, use_container_width=True)

with fleet_col2:
    st.markdown("**Drift fraction per engine**")
    st.caption(
        "Fraction of each engine's windows that exceeded the drift threshold. "
        "Red bars have >50% windows drifted, amber >20%, blue less."
    )
    drift_bar_fig = build_drift_engine_bar(health_df)
    st.plotly_chart(drift_bar_fig, use_container_width=True)


st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# LATENT SPACE
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("### Latent Space Representation")
st.caption(
    "L2 norm of the encoded mean vector (‖μ‖₂) vs true RUL. "
    "In a VAE, the healthy prior is N(0,I) so healthy encodings cluster near zero. "
    "As engines degrade, their encodings may drift away from the origin. "
    "Colour: green = healthy (high RUL), red = near failure (low RUL)."
)

latent_fig = build_latent_norm_vs_rul(health_df)
st.plotly_chart(latent_fig, use_container_width=True)


st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# PER-ENGINE HEALTH TRAJECTORY
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("### Per-Engine Health Index Trajectory")
st.caption(
    "Select an engine to see its full health index history. "
    "All four indices are normalised to [0, 1] per engine for visual comparison. "
    "Raw values appear in the hover tooltip."
)

# Engine selector — pre-populate from session_state if navigated from another page
all_units      = sorted(health_df["unit_number"].unique().tolist())
nav_engine     = st.session_state.get("selected_engine", all_units[0])
default_idx    = all_units.index(nav_engine) if nav_engine in all_units else 0

selected_engine = st.selectbox("Select engine", all_units, index=default_idx)

engine_health = health_df[health_df["unit_number"] == selected_engine].sort_values("cycle")

if not engine_health.empty:
    # Health index chart
    health_fig = build_health_index(engine_health, selected_engine)
    st.plotly_chart(health_fig, use_container_width=True)

    # Summary stats for this engine
    n_windows  = len(engine_health)
    n_drifted_w = int(engine_health["drift_flag"].sum()) if "drift_flag" in engine_health.columns else 0
    first_drift = engine_health[engine_health["drift_flag"]]["cycle"].min() \
                  if "drift_flag" in engine_health.columns and n_drifted_w > 0 else None

    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
    with stat_col1:
        st.metric("Windows analysed", n_windows)
    with stat_col2:
        st.metric("Drift-flagged windows", n_drifted_w)
    with stat_col3:
        st.metric(
            "First drift detected at cycle",
            int(first_drift) if first_drift is not None else "—",
        )
    with stat_col4:
        final_recon = engine_health["recon_error"].iloc[-1]
        st.metric("Final reconstruction error", f"{final_recon:.5f}")

    # Show raw health index data in expander
    with st.expander("Raw health index data"):
        show_cols = [c for c in
                     ["cycle", "true_rul", "recon_error", "kl_div",
                      "js_div", "wasserstein", "drift_flag", "op_cluster"]
                     if c in engine_health.columns]
        st.dataframe(
            engine_health[show_cols].round(6),
            use_container_width=True,
            hide_index=True,
        )


st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# METHODOLOGY NOTE
# ─────────────────────────────────────────────────────────────────────────────

with st.expander("Methodology — VAE health monitoring"):
    st.markdown(f"""
**Training setup**
- VAE trained on sequences where RUL > 80 (healthy engine windows only)
- No labels used — purely unsupervised
- Sequence length matches the predictive backbone: `{health_df['cycle'].nunique()} unique cycles observed`

**Health index interpretation**

| Metric | Source | Signal |
|--------|--------|--------|
| Reconstruction error | VAE decoder MSE | Primary — rises as sensor pattern diverges from healthy regime |
| KL divergence | Latent space distance | How surprised the healthy prior is by the current encoding |
| JS divergence | Symmetric version of KL | More stable when distributions barely overlap |
| Wasserstein | Optimal transport distance | Numerically robust alternative to KL |

**Drift detection**
- Threshold = mean + 2σ of reconstruction error on healthy training data
- Covers ~97.5% of healthy behaviour (assuming approximately Gaussian errors)
- Per operating-cluster thresholds for multi-condition datasets (FD002/FD004)

**Note on geometry distances**
The CMAPSS healthy data is highly uniform, causing the VAE encoder to map
both healthy and degraded sequences to a compact region of latent space.
Reconstruction error (input-space signal) is therefore the most reliable
health indicator on this dataset. The geometry distances provide supporting
signal, strongest on FD002 and FD004 (multi-condition, dual fault modes).
    """)