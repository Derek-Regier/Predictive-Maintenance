"""
Health Monitor.

The VAE was trained on healthy engine windows (RUL > threshold) and
never saw labels. It learned the region of latent space that healthy
engines occupy. At inference, deviations from that region surface as:

  - Information-geometric distances against the healthy fleet
    (Mahalanobis, Fisher-Rao, KL, JS, Bures-Wasserstein)
  - The same distances against the engine's OWN early-life baseline,
    which removes unit-to-unit offset
  - Reconstruction error (input-space signal)
  - Persistence-filtered alarms on both the geometry and reconstruction
    channels

SCHEMA HANDLING
---------------
This page reads whatever columns health_indices.csv actually contains
rather than assuming a fixed set. A dataset that has not been re-run
through the updated train_vae.py + health_monitor.py still renders,
with the panels that need the new columns replaced by a note. That
matters because retraining four datasets takes a while and a dashboard
that crashes halfway through the migration is worse than one that says
what is missing.
"""

import sys
from pathlib import Path

_DASHBOARD = Path(__file__).resolve().parents[1]
if str(_DASHBOARD) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD))

import numpy as np
import pandas as pd
import streamlit as st

from utils.data_loader import (
    load_health_indices,
    load_health_summary,
    load_health_index_metrics,
    load_registry,
    DATASETS,
)
from utils.charts import (
    build_health_index,
    build_geometry_fleet,
    build_index_small_multiples,
    build_health_score_trajectory,
    build_alarm_timeline,
    build_index_quality_bar,
    build_index_quality_radar,
    build_latent_diagnostics_bar,
    build_engine_alarm_bar,
    build_lead_time_hist,
    build_latent_norm_vs_rul,
)
from utils.styles import (
    CUSTOM_CSS, HEADER_HTML, SIGNATURE_HTML,
    HEALTH_LABELS, HEALTH_DESCRIPTIONS, GEOMETRY_INDICES, SCORE_BANDS,
)

st.set_page_config(
    page_title="Health Monitor — Predictive Maintenance",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.markdown(HEADER_HTML, unsafe_allow_html=True)
st.markdown(SIGNATURE_HTML, unsafe_allow_html=True)


# HELPERS

# Order matters: the first available entry becomes the default selection,
# so this list encodes "which index we would reach for first".
# Order set by measured performance across all four datasets — fraction of
# engines whose index trends the right way against RUL (0.50 = coin flip):
#
#     mahalanobis_self   0.90     recon_error         0.53
#     fisher_rao_self    0.89     latent_mu_centered  0.53
#     mahalanobis        0.64     kl_div              0.52 (sign unstable)
#                                 wasserstein         0.51
#                                 fisher_rao          0.46 (backwards)
#                                 js_div              0.42 (backwards)
#
# The two self-referenced indices are the only ones that work everywhere,
# which is the empirical case for the per-engine baseline.
_PREFERRED_INDEX_ORDER = [
    "mahalanobis_self",
    "fisher_rao_self",
    "mahalanobis",
    "recon_error",
    "latent_mu_centered",
    "wasserstein",
    "kl_div",
    "js_div",
    "fisher_rao",
    "latent_mu_norm",
]


def available_indices(df: pd.DataFrame) -> list[str]:
    """Indices present in this CSV, in preferred display order."""
    return [c for c in _PREFERRED_INDEX_ORDER if c in df.columns]


def score_band(score: float) -> tuple[str, str]:
    """Map a 0-100 health score to its (label, colour) band."""
    for lo, hi, colour, label in SCORE_BANDS:
        if lo <= score < hi:
            return label, colour
    return "Unknown", "#6B7280"


def per_engine_spearman(df: pd.DataFrame, col: str) -> tuple[float, float]:
    """
    Mean within-engine rank correlation with true RUL, and the fraction of
    engines where the sign points the expected way.

    Computed live rather than read from health_summary.json so the panel
    still works on a legacy CSV, and so it always reflects the file on
    disk rather than whatever the summary was written from.
    """
    rhos = []
    for _, g in df.groupby("unit_number"):
        if g[col].nunique() < 3 or g["true_rul"].nunique() < 3:
            continue
        rho = g[col].corr(g["true_rul"], method="spearman")
        if np.isfinite(rho):
            rhos.append(rho)
    if not rhos:
        return float("nan"), float("nan")
    rhos = np.asarray(rhos)
    expect_positive = col == "health_score"
    correct = (rhos > 0).mean() if expect_positive else (rhos < 0).mean()
    return float(rhos.mean()), float(correct)


# SIDEBAR

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
        "1. VAE trained on healthy engines only (RUL > 80), no labels\n"
        "2. Healthy encodings define a region — mean **and covariance** — "
        "in latent space\n"
        "3. Each test window is encoded and its distance from that region "
        "measured under the Fisher information metric\n"
        "4. Reconstruction error gives an independent input-space signal\n"
        "5. Alarms require persistence: k of the last n windows"
    )

    st.divider()
    st.caption(
        "The covariance in step 2 is what makes the geometry work. "
        "Comparing against an averaged posterior *width* instead of the "
        "population *spread* collapses every distance to near zero."
    )


# DATA LOADING

health_df = load_health_indices(selected_dataset)
health_summ = load_health_summary(selected_dataset)
metrics_df = load_health_index_metrics(selected_dataset)
registry = load_registry()

st.title("Health Monitor")
st.caption(f"Dataset: **{selected_dataset}** — VAE-based unsupervised degradation detection")

if health_df.empty:
    st.warning(
        "Health index data not found for this dataset.\n\n"
        "Run the following commands:\n"
        "```bash\n"
        "python src/training/train_vae.py --dataset all\n"
        "python src/health/health_monitor.py --dataset all\n"
        "python src/health/evaluate_health_indices.py --dataset all\n"
        "```"
    )
    st.stop()

indices = available_indices(health_df)
geometry_version = health_summ.get("geometry_version", 1)
has_geometry_v2 = "mahalanobis" in health_df.columns
has_score = "health_score" in health_df.columns
has_geo_alarm = "geo_alarm" in health_df.columns

if not has_geometry_v2:
    st.info(
        "This dataset is still on the original health-index schema "
        "(diagonal posterior reference). The geometry panels below will be "
        "limited until it is re-run:\n\n"
        "```bash\n"
        f"python src/training/train_vae.py --dataset {selected_dataset}\n"
        f"python src/health/health_monitor.py --dataset {selected_dataset}\n"
        f"python src/health/evaluate_health_indices.py --dataset {selected_dataset}\n"
        "```"
    )

# A degenerate latent space makes every geometry panel meaningless, so say
# so loudly at the top rather than letting someone read noise as signal.
latent_info = health_summ.get("latent", {})
if latent_info.get("degenerate"):
    st.error(
        f"**Latent space is degenerate.** ‖μ‖ = "
        f"{latent_info.get('mu_norm_mean', float('nan')):.3f} ± "
        f"{latent_info.get('mu_norm_std', float('nan')):.5f} "
        f"(relative spread {latent_info.get('mu_norm_relative_spread', float('nan')):.1e}). "
        "The encoder is emitting a near-constant vector regardless of input, so the "
        "geometry indices below are measuring floating-point noise. "
        "Lower `vae_beta` in datasets.yaml and retrain."
    )


# FLEET SUMMARY CARDS

n_engines = health_summ.get("n_engines", health_df["unit_number"].nunique())
n_drifted = health_summ.get("n_drifted_engines", 0)
pct_drift = health_summ.get("pct_windows_drifted", 0.0)
thresholds = health_summ.get("drift_thresholds", {})
geo_thresholds = health_summ.get("geo_thresholds", {})
global_threshold = thresholds.get("global")
global_geo_threshold = geo_thresholds.get("global")
lead_time = health_summ.get("lead_time", {})

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric("Engines monitored", n_engines)

with m2:
    if has_score:
        # Fleet health is summarised by each engine's LAST window, not the
        # mean over all windows — a fleet average across time is dominated
        # by however long each engine happened to be observed for.
        last_scores = (health_df.sort_values("cycle")
                       .groupby("unit_number")["health_score"].last())
        st.metric(
            "Median health score (latest window)",
            f"{last_scores.median():.0f}",
            help="Percentile against healthy validation data. 100 = "
                 "indistinguishable from healthy.",
        )
    else:
        st.metric("Engines with drift detected", n_drifted)

with m3:
    geo_lead = lead_time.get("geometry", {})
    if geo_lead.get("n_engines_alarmed"):
        st.metric(
            "Median RUL at geometry alarm",
            f"{geo_lead['median_rul_at_first_alarm']:.0f}",
            help=f"Across the {geo_lead['n_engines_alarmed']} of "
                 f"{geo_lead.get('n_engines_total', n_engines)} engines that "
                 "raised a confirmed geometry alarm. Higher = earlier warning.",
        )
    else:
        st.metric(
            "Windows flagged as drifted",
            f"{pct_drift:.1f}%",
            help="Fraction of all test windows above the drift threshold",
        )

with m4:
    if global_geo_threshold is not None:
        st.metric(
            "Geometry alarm threshold",
            f"{global_geo_threshold:.3f}",
            help="Mahalanobis distance at the calibration quantile of "
                 "healthy validation windows.",
        )
    elif global_threshold is not None:
        st.metric(
            "Drift threshold (global)",
            f"{global_threshold:.5f}",
            help="Reconstruction error threshold from healthy data",
        )

alarm_cfg = health_summ.get("alarm_config", {})
if alarm_cfg:
    st.caption(
        f"Alarms calibrated on **{alarm_cfg.get('calibrated_on', 'unknown')}** "
        f"at quantile {alarm_cfg.get('geo_quantile', '—')} (geometry) / "
        f"{alarm_cfg.get('drift_quantile', '—')} (reconstruction), confirmed by "
        f"{alarm_cfg.get('persistence_k', '—')} of the last "
        f"{alarm_cfg.get('persistence_n', '—')} windows. Self-referenced indices "
        f"use each engine's first {alarm_cfg.get('baseline_windows', '—')} windows "
        f"as its baseline."
    )

st.divider()


# TABS

tab_fleet, tab_engine, tab_quality, tab_diag = st.tabs([
    "Fleet Signal",
    "Engine Deep Dive",
    "Index Quality",
    "Model Diagnostics",
])


# ── TAB 1: FLEET SIGNAL ──────────────────────────────────────────────────

with tab_fleet:
    st.markdown("### Fleet-Wide Degradation Signal")

    sel_col, opt_col = st.columns([3, 2])
    with sel_col:
        chosen_index = st.selectbox(
            "Health index",
            indices,
            format_func=lambda c: HEALTH_LABELS.get(c, c),
            key="fleet_index_select",
        )
    with opt_col:
        log_y = st.checkbox(
            "Log scale",
            value=chosen_index in ("kl_div", "js_div", "wasserstein"),
            help="KL and Wasserstein against a tight healthy reference span "
                 "several orders of magnitude; a linear axis flattens them.",
        )

    st.caption(HEALTH_DESCRIPTIONS.get(chosen_index, ""))

    # Pick the alarm column that belongs to the selected index, so the
    # colouring means something rather than always showing recon drift.
    if chosen_index in GEOMETRY_INDICES and has_geo_alarm:
        alarm_col, threshold_val = "geo_alarm", global_geo_threshold
    elif "drift_flag" in health_df.columns:
        alarm_col, threshold_val = "drift_flag", global_threshold
    else:
        alarm_col, threshold_val = None, None

    # The threshold line only belongs on the index it was calibrated for.
    if chosen_index not in ("mahalanobis", "recon_error"):
        threshold_val = None

    fleet_fig = build_geometry_fleet(
        health_df, chosen_index,
        threshold=threshold_val, alarm_col=alarm_col, log_y=log_y,
    )
    st.plotly_chart(fleet_fig, use_container_width=True)

    mean_rho, frac_correct = per_engine_spearman(health_df, chosen_index)
    q1, q2 = st.columns(2)
    with q1:
        st.metric(
            "Mean per-engine Spearman vs RUL",
            f"{mean_rho:+.3f}",
            help="Within-engine rank correlation, averaged over engines. This "
                 "is the number that matters — a pooled correlation across all "
                 "windows can look strong while every individual engine is flat.",
        )
    with q2:
        st.metric(
            "Engines trending the right way",
            f"{frac_correct:.0%}",
            help="Fraction of engines where the sign of the correlation points "
                 "the expected direction. Near 50% means the index is a coin "
                 "flip at the engine level.",
        )

    st.divider()

    st.markdown("**All indices side by side**")
    st.caption(
        "A useful index fans out toward low RUL on the left. A dead one forms "
        "a flat horizontal band. Points are subsampled for rendering speed."
    )
    st.plotly_chart(
        build_index_small_multiples(health_df, indices),
        use_container_width=True,
    )

    st.divider()

    st.markdown("### Alarms")
    a1, a2 = st.columns(2)
    with a1:
        st.plotly_chart(
            build_lead_time_hist(health_df, "geo_alarm" if has_geo_alarm else "drift_flag"),
            use_container_width=True,
        )
        st.caption(
            "How much life was left when the monitor first spoke up. A pile "
            "against the right edge means the threshold is too tight and it "
            "fires immediately; a pile at zero means it fires too late."
        )
    with a2:
        if has_geo_alarm:
            st.plotly_chart(
                build_lead_time_hist(health_df, "drift_flag"),
                use_container_width=True,
            )
            st.caption(
                "The reconstruction channel for comparison. The two channels "
                "are independent — reconstruction error is an input-space "
                "signal that does not depend on the latent space behaving."
            )

    flag_choice = st.radio(
        "Per-engine alarm rate",
        [c for c in ["geo_alarm", "drift_flag"] if c in health_df.columns],
        format_func=lambda c: "Geometry alarm" if c.startswith("geo") else "Reconstruction drift",
        horizontal=True,
        key="fleet_flag_choice",
    )
    st.plotly_chart(build_engine_alarm_bar(health_df, flag_choice),
                    use_container_width=True)


# ── TAB 2: ENGINE DEEP DIVE ──────────────────────────────────────────────

with tab_engine:
    st.markdown("### Per-Engine Health Trajectory")

    all_units = sorted(health_df["unit_number"].unique().tolist())
    nav_engine = st.session_state.get("selected_engine", all_units[0])
    default_idx = all_units.index(nav_engine) if nav_engine in all_units else 0

    selected_engine = st.selectbox("Select engine", all_units, index=default_idx,
                                   key="health_engine_select")
    engine_health = health_df[health_df["unit_number"] == selected_engine].sort_values("cycle")

    if engine_health.empty:
        st.warning("No health windows for this engine.")
        st.stop()

    # Summary row
    n_windows = len(engine_health)
    s1, s2, s3, s4 = st.columns(4)

    with s1:
        st.metric("Windows analysed", n_windows)

    with s2:
        if has_score:
            final_score = float(engine_health["health_score"].iloc[-1])
            label, colour = score_band(final_score)
            st.metric("Health score (latest)", f"{final_score:.0f}", help=f"Band: {label}")
        else:
            st.metric("Final reconstruction error",
                      f"{engine_health['recon_error'].iloc[-1]:.5f}")

    with s3:
        if has_geo_alarm and engine_health["geo_alarm"].any():
            first_geo = int(engine_health.loc[engine_health["geo_alarm"], "cycle"].min())
            rul_at = int(engine_health.loc[engine_health["cycle"] == first_geo,
                                           "true_rul"].iloc[0])
            st.metric("First geometry alarm", f"cycle {first_geo}",
                      help=f"True RUL at that point: {rul_at} cycles")
        else:
            st.metric("First geometry alarm", "—")

    with s4:
        if "drift_flag" in engine_health.columns and engine_health["drift_flag"].any():
            first_drift = int(engine_health.loc[engine_health["drift_flag"], "cycle"].min())
            st.metric("First recon drift", f"cycle {first_drift}")
        else:
            st.metric("First recon drift", "—")

    if has_score:
        st.plotly_chart(
            build_health_score_trajectory(engine_health, selected_engine),
            use_container_width=True,
        )
        st.caption(
            "Absolute scale, comparable across engines and datasets — the score "
            "is a percentile against healthy validation data, not against this "
            "engine's own range."
        )

    st.divider()

    st.markdown("**Health indices (normalised per engine)**")
    st.caption(
        "Each index is min-max normalised to this engine's own range so indices "
        "on different scales share an axis. That means every engine's worst "
        "window maps to 1.0, including engines that never degraded — this chart "
        "shows shape, not severity. Shaded bands are confirmed alarm episodes."
    )

    default_cols = [c for c in ["mahalanobis_self", "fisher_rao_self",
                                "mahalanobis", "recon_error"]
                    if c in engine_health.columns]
    chosen_cols = st.multiselect(
        "Indices to plot",
        indices,
        default=default_cols or indices[:3],
        format_func=lambda c: HEALTH_LABELS.get(c, c),
        key="engine_index_multiselect",
    )
    if chosen_cols:
        st.plotly_chart(
            build_health_index(engine_health, selected_engine, columns=chosen_cols),
            use_container_width=True,
        )

    if "drift_raw" in engine_health.columns or "geo_alarm_raw" in engine_health.columns:
        st.divider()
        st.markdown("**Alarm timeline — raw vs persistence-filtered**")
        st.caption(
            "A threshold at the 99th healthy percentile fires on 1% of healthy "
            "windows by construction, so raw flags scatter sporadic single-window "
            "hits across the record. The confirmed rows show what survives the "
            "k-of-n requirement."
        )
        st.plotly_chart(
            build_alarm_timeline(engine_health, selected_engine),
            use_container_width=True,
        )

    with st.expander("Raw health index data"):
        show_cols = [c for c in
                     ["cycle", "true_rul", "health_score", "mahalanobis",
                      "mahalanobis_self", "fisher_rao", "kl_div", "js_div",
                      "wasserstein", "recon_error", "drift_flag", "geo_alarm",
                      "op_cluster"]
                     if c in engine_health.columns]
        st.dataframe(
            engine_health[show_cols].round(6),
            use_container_width=True,
            hide_index=True,
        )


# ── TAB 3: INDEX QUALITY ─────────────────────────────────────────────────

with tab_quality:
    st.markdown("### Which index should we trust?")
    st.caption(
        "Scored on the standard prognostic-suitability metrics from Coble & "
        "Hines (2009). Each catches a different way an index can fail, and an "
        "index can look fine on a pooled correlation while failing all three."
    )

    if metrics_df.empty:
        st.info(
            "Index quality metrics not found. Generate them with:\n\n"
            "```bash\n"
            f"python src/health/evaluate_health_indices.py --dataset {selected_dataset}\n"
            "```"
        )
    else:
        metric_choice = st.radio(
            "Metric",
            ["composite", "monotonicity", "trendability", "prognosability",
             "mean_engine_spearman", "frac_correct_direction"],
            format_func=lambda m: m.replace("_", " ").title(),
            horizontal=True,
            key="quality_metric_choice",
        )

        explain = {
            "composite":
                "Mean of the three Coble & Hines metrics. A summary, not a "
                "substitute for reading the individual columns.",
            "monotonicity":
                "Does the index move consistently in one direction? An index "
                "that oscillates cannot support extrapolation to a failure "
                "threshold, however well it correlates on average.",
            "trendability":
                "The MINIMUM absolute index-vs-time correlation across engines. "
                "Deliberately harsh: it asks whether every engine shows the "
                "trend, not whether most do.",
            "prognosability":
                "How tightly the index clusters at failure, relative to how far "
                "it travels over life. If engines fail at wildly different "
                "values there is no fixed threshold to alarm on.",
            "mean_engine_spearman":
                "Within-engine rank correlation with RUL, averaged over engines. "
                "Plotted as absolute value — the sign is a property of the "
                "index's direction, not its quality.",
            "frac_correct_direction":
                "Fraction of engines where the correlation points the expected "
                "way. Near 50% means the index is a coin flip at engine level, "
                "which a pooled correlation will happily hide.",
        }
        st.caption(explain.get(metric_choice, ""))

        q1, q2 = st.columns([3, 2])
        with q1:
            st.plotly_chart(
                build_index_quality_bar(metrics_df, metric_choice),
                use_container_width=True,
            )
        with q2:
            st.plotly_chart(
                build_index_quality_radar(metrics_df),
                use_container_width=True,
            )

        st.markdown("**Full scorecard**")
        display = metrics_df.copy()
        display["index"] = display["index"].map(lambda c: HEALTH_LABELS.get(c, c))
        st.dataframe(
            display.round(4),
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "These measure suitability as a degradation *indicator*, not "
            "predictive accuracy. A high composite means the index is a good "
            "thing to threshold and extrapolate; it does not mean it predicts "
            "RUL better than the NGBoost model."
        )


# ── TAB 4: MODEL DIAGNOSTICS ─────────────────────────────────────────────

with tab_diag:
    st.markdown("### VAE posterior diagnostics")
    st.caption(
        "Whether the latent space is alive at all. Everything on the other "
        "tabs depends on this being healthy."
    )

    posterior = health_summ.get("posterior") or \
        registry.get(selected_dataset, {}).get("vae", {}).get("posterior", {})

    if not posterior:
        st.info(
            "No posterior diagnostics recorded for this dataset. They are "
            "written by train_vae.py from this revision onward — retrain to "
            "populate them."
        )
    else:
        d1, d2, d3, d4 = st.columns(4)
        active = posterior.get("active_units")
        latent_dim = posterior.get("latent_dim")

        with d1:
            st.metric(
                "Active latent units",
                f"{active} / {latent_dim}" if active is not None else "—",
                help="Dimensions where Var_x(E[z|x]) exceeds the activity "
                     "threshold — i.e. the posterior mean actually moves when "
                     "the input changes (Burda et al. 2016).",
            )
        with d2:
            st.metric("Total KL (nats)", f"{posterior.get('total_kl', float('nan')):.3f}")
        with d3:
            st.metric("Mean posterior σ", f"{posterior.get('sigma_mean', float('nan')):.4f}")
        with d4:
            mu_mean = posterior.get("mu_norm_mean", float("nan"))
            mu_std = posterior.get("mu_norm_std", float("nan"))
            st.metric(
                "‖μ‖ mean ± std",
                f"{mu_mean:.2f} ± {mu_std:.3f}",
                help="A large mean with a tiny std is the signature of an "
                     "encoder emitting a near-constant vector.",
            )

        if active == 0:
            st.error(
                "No active latent units. The encoder is ignoring its input "
                "entirely — every geometry index is noise. Lower `vae_beta` "
                "in datasets.yaml and retrain."
            )
        elif active is not None and latent_dim and active < 3:
            st.warning(
                f"Only {active} of {latent_dim} dimensions are active. The "
                "geometry indices will be weak. Consider lowering `vae_beta`."
            )

        st.plotly_chart(
            build_latent_diagnostics_bar(posterior),
            use_container_width=True,
        )
        st.caption(
            "The two panels can disagree, and that disagreement is the point. "
            "A dimension can show healthy KL purely from a constant offset "
            "while its posterior mean never moves — a dead unit wearing a "
            "disguise. The lower panel is the honest test."
        )

    st.divider()

    st.markdown("**Latent norm vs RUL**")
    st.caption(
        "Kept as a diagnostic. If the encoder is healthy this spreads out; if "
        "it has collapsed to a constant, the whole cloud compresses into a "
        "horizontal line whose thickness is measured in trailing decimals."
    )
    st.plotly_chart(build_latent_norm_vs_rul(health_df), use_container_width=True)

    st.divider()

    with st.expander("Methodology — VAE health monitoring"):
        vae_reg = registry.get(selected_dataset, {}).get("vae", {})
        st.markdown(f"""
**Training setup**
- VAE trained only on windows where RUL > {vae_reg.get('healthy_rul_threshold', 80)} — unsupervised, no labels
- β = {vae_reg.get('beta', '—')}, free bits = {vae_reg.get('free_bits', '—')} per latent dimension
- Loss reduction: `{vae_reg.get('loss_reduction', 'unknown')}`
- Sequence length matches the predictive backbone so cycles align with `predictions_all.csv`

**Why the reduction matters**

The reconstruction term is summed over sequence and feature dimensions and averaged over
batch, matching the KL's reduction. Averaging reconstruction over all three axes — as the
original version did — divides it by `seq_length × input_dim` while leaving KL divided by
batch alone, which silently multiplies the effective β by several hundred. No warm-up
schedule or free-bits floor survives that, and the result is total posterior collapse.

**The healthy reference**

Built from healthy *training* windows as a full-covariance Gaussian N(μ_ref, Σ_ref), where
Σ_ref is the covariance of healthy encodings **across the fleet** — not the average
posterior width of individual windows. Those are different objects, and using the second
where the first is needed collapses every distance to near zero.

**Index families**

| Index | Reference | What it measures |
|---|---|---|
| Mahalanobis | Fleet healthy region | Distance whitened by healthy covariance — discounts benign directions |
| Fisher-Rao | Fleet healthy region | Geodesic distance under the Fisher information metric. A true metric, not a divergence |
| KL / JS | Fleet healthy region | Divergence of the current posterior from the healthy population |
| Bures-Wasserstein | Fleet healthy region | Optimal transport, covariance-aware. No division or log, so most numerically robust |
| Mahalanobis (self) | Engine's own baseline | Same whitening, centred on the engine — removes unit-to-unit offset |
| Reconstruction error | Healthy quantile | Input-space signal, independent of the latent space behaving |

**Calibration and alarms**

- Thresholds are quantiles of healthy **validation** windows, not `mean + 2σ` on training
  data. Reconstruction errors are right-skewed, so `mean + 2σ` does not give the coverage
  it appears to; and training-set error is optimistically low.
- Alarms require {alarm_cfg.get('persistence_k', 'k')} of the last
  {alarm_cfg.get('persistence_n', 'n')} windows. Without this, a threshold with a 1%
  false-positive rate trips at least once on roughly 60% of engines over a 90-window
  record, which makes "first alarm cycle" meaningless.

**Known caveat — the self-referenced baseline**

CMAPSS test trajectories are truncated at an arbitrary point, so an engine's first
observed window is not guaranteed to be healthy. The self-referenced indices measure
change relative to the earliest *observed* state, and will understate degradation for an
engine whose record begins late. The fleet-referenced indices do not have this problem,
which is why both are kept rather than one replacing the other.
        """)