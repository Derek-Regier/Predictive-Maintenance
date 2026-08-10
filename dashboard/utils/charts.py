"""
dashboard/utils/charts.py

All Plotly figure builders for the dashboard.

Every function follows the same contract:
  - Accepts a DataFrame or dict of data (never reads files directly)
  - Accepts optional colour/style overrides
  - Returns a plotly.graph_objects.Figure
  - Is called with st.plotly_chart(fig, use_container_width=True)

Keeping chart logic here (not in the page files) means:
  1. Page files stay readable — just function calls
  2. Charts are reusable across pages
  3. Visual changes happen in one place

To style a chart: find the relevant function, edit fig.update_layout()
or the trace parameters. Plotly docs: https://plotly.com/python/
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.styles import (
    ALERT_COLORS, DATASET_COLORS, HEALTH_COLORS, HEALTH_LABELS,
    CHART_HEIGHT, CHART_HEIGHT_TALL, CHART_HEIGHT_COMPACT, CHART_THEME,
)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _apply_theme(fig: go.Figure, height: int = CHART_HEIGHT) -> go.Figure:
    """Apply the global chart theme and height to any figure."""
    fig.update_layout(**CHART_THEME, height=height)
    return fig


def _normalise_column(series: pd.Series) -> pd.Series:
    """
    Min-max normalise a Series to [0, 1].
    Returns zeros if the series is constant (avoids division by zero).
    Used to put health metrics on the same axis regardless of scale.
    """
    mn, mx = series.min(), series.max()
    if mx > mn:
        return (series - mn) / (mx - mn)
    return pd.Series(0.0, index=series.index)


# ─────────────────────────────────────────────────────────────────────────────
# FLEET OVERVIEW CHARTS
# ─────────────────────────────────────────────────────────────────────────────

def build_alert_donut(tier_counts: dict) -> go.Figure:
    """
    Donut chart showing the distribution of alert tiers across the fleet.

    tier_counts: dict like {"CRITICAL": 11, "WARNING": 19, ...}
    The hole=0.65 creates the donut shape — reduce it for a thicker ring.
    """
    labels = list(tier_counts.keys())
    values = list(tier_counts.values())
    colors = [ALERT_COLORS.get(t, "#9CA3AF") for t in labels]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        marker_colors=colors,
        hole=0.65,                          # donut hole size (0=pie, 1=invisible)
        textinfo="label+percent",
        textfont_size=12,
        hovertemplate="%{label}: %{value} engines (%{percent})<extra></extra>",
    ))

    # Centre annotation showing total engine count
    total = sum(values)
    fig.update_layout(
        annotations=[dict(
            text=f"<b>{total}</b><br>engines",
            x=0.5, y=0.5,
            font_size=14,
            showarrow=False,
        )],
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
    )
    return _apply_theme(fig, height=CHART_HEIGHT_COMPACT)


def build_cross_dataset_risk_bar(all_metrics: dict[str, dict]) -> go.Figure:
    """
    Horizontal bar chart showing expected failures in 20 and 50 cycles
    for each dataset side by side. This gives a quick fleet-health
    comparison across all four CMAPSS conditions.

    all_metrics: {dataset_key: metrics_summary_dict}
    """
    datasets, exp20, exp50 = [], [], []

    for ds, m in all_metrics.items():
        risk = m.get("fleet_risk", {})
        if risk:
            datasets.append(ds)
            exp20.append(risk.get("expected_failures_20", 0))
            exp50.append(risk.get("expected_failures_50", 0))

    fig = go.Figure()

    # 50-cycle bar (lighter, drawn first so 20-cycle sits on top)
    fig.add_trace(go.Bar(
        y=datasets, x=exp50,
        orientation="h",
        name="Within 50 cycles",
        marker_color=[DATASET_COLORS.get(d, "#9CA3AF") for d in datasets],
        opacity=0.4,
        hovertemplate="%{y}: %{x:.1f} expected<extra>50 cycles</extra>",
    ))

    # 20-cycle bar (darker, immediate risk)
    fig.add_trace(go.Bar(
        y=datasets, x=exp20,
        orientation="h",
        name="Within 20 cycles",
        marker_color=[DATASET_COLORS.get(d, "#9CA3AF") for d in datasets],
        opacity=0.95,
        hovertemplate="%{y}: %{x:.1f} expected<extra>20 cycles</extra>",
    ))

    fig.update_layout(
        barmode="overlay",          # overlay so 20-cycle sits in front of 50-cycle
        xaxis_title="Expected number of engine failures",
        yaxis_title="",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        title="Expected fleet failures by dataset",
    )
    return _apply_theme(fig, height=CHART_HEIGHT_COMPACT)


# ─────────────────────────────────────────────────────────────────────────────
# ENGINE DEEP DIVE CHARTS
# ─────────────────────────────────────────────────────────────────────────────

def build_rul_trajectory(engine_df: pd.DataFrame, engine_id: int) -> go.Figure:
    """
    RUL trajectory for one engine showing true RUL, predicted mean,
    and the 90% prediction interval as a shaded band.

    The shaded band directly represents NGBoost's calibrated uncertainty.
    Where it's narrow the model is confident; where it's wide the model
    is uncertain. The end-of-life shading marks the critical zone.

    engine_df: rows from predictions_all.csv filtered to one engine,
               sorted by cycle.
    """
    df = engine_df.sort_values("cycle")

    fig = go.Figure()

    # ── End-of-life danger zone (RUL ≤ 30) ───────────────────────────────────
    # Find the cycle where RUL first drops to 30 to place the zone marker.
    # If no cycle reaches RUL≤30 in this engine's history we skip it.
    eol_cycles = df[df["true_rul"] <= 30]["cycle"]
    if len(eol_cycles) > 0:
        fig.add_vrect(
            x0=eol_cycles.min(), x1=df["cycle"].max(),
            fillcolor="#FEE2E2", opacity=0.25, layer="below",
            line_width=0, annotation_text="Critical zone",
            annotation_position="top left",
            annotation_font_size=10, annotation_font_color="#DC2626",
        )

    # ── 90% prediction interval (shaded band) ────────────────────────────────
    # Build as a filled polygon: trace along upper_90 forward then lower_90 backward.
    x_band = pd.concat([df["cycle"], df["cycle"][::-1]])
    y_band = pd.concat([df["upper_90"], df["lower_90"][::-1]])

    fig.add_trace(go.Scatter(
        x=x_band, y=y_band,
        fill="toself",
        fillcolor="rgba(37, 99, 235, 0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        hoverinfo="skip",
        showlegend=True,
        name="90% PI",
    ))

    # ── True RUL (ground truth — shown as context, not prediction) ───────────
    fig.add_trace(go.Scatter(
        x=df["cycle"], y=df["true_rul"],
        mode="lines",
        line=dict(color="#9CA3AF", width=1.5, dash="dot"),
        name="True RUL",
        hovertemplate="Cycle %{x}: true RUL = %{y:.0f}<extra></extra>",
    ))

    # ── Predicted mean RUL (the model's point estimate) ───────────────────────
    fig.add_trace(go.Scatter(
        x=df["cycle"], y=df["pred_rul"],
        mode="lines",
        line=dict(color="#2563EB", width=2.2),
        name="Predicted RUL",
        hovertemplate="Cycle %{x}: pred = %{y:.1f} cycles<extra></extra>",
    ))

    fig.update_layout(
        title=f"Engine {engine_id} — RUL Trajectory",
        xaxis_title="Cycle",
        yaxis_title="Remaining Useful Life (cycles)",
        hovermode="x unified",
    )
    return _apply_theme(fig, height=CHART_HEIGHT_TALL)


def build_failure_probability(engine_df: pd.DataFrame,
                               engine_id: int,
                               thresholds: dict | None = None) -> go.Figure:
    """
    Failure probability trajectory for one engine.
    Two lines: P(RUL<20) in red (CRITICAL threshold) and P(RUL<50)
    in amber (WARNING threshold). Horizontal dashed reference lines
    show where alert tiers fire.

    thresholds: optional dict with keys "critical_prob_20" and "warning_prob_50"
                loaded from datasets.yaml. Defaults to 0.90 / 0.75 if not given.
    """
    df   = engine_df.sort_values("cycle")
    thr  = thresholds or {}
    crit = thr.get("critical_prob_20", 0.90)
    warn = thr.get("warning_prob_50",  0.75)
    mon  = thr.get("monitor_prob_50",  0.50)

    fig = go.Figure()

    # P(RUL < 50) — WARNING signal (drawn first, behind P<20)
    fig.add_trace(go.Scatter(
        x=df["cycle"], y=df["prob_failure_50"],
        mode="lines",
        line=dict(color=ALERT_COLORS["WARNING"], width=2),
        name="P(fail within 50 cycles)",
        hovertemplate="Cycle %{x}: P(RUL<50) = %{y:.3f}<extra></extra>",
    ))

    # P(RUL < 20) — CRITICAL signal
    fig.add_trace(go.Scatter(
        x=df["cycle"], y=df["prob_failure_20"],
        mode="lines",
        line=dict(color=ALERT_COLORS["CRITICAL"], width=2.2),
        name="P(fail within 20 cycles)",
        hovertemplate="Cycle %{x}: P(RUL<20) = %{y:.3f}<extra></extra>",
    ))

    # Threshold reference lines
    for level, label, color in [
        (crit, f"CRITICAL threshold ({crit:.0%})", ALERT_COLORS["CRITICAL"]),
        (warn, f"WARNING threshold  ({warn:.0%})", ALERT_COLORS["WARNING"]),
        (mon,  f"MONITOR threshold  ({mon:.0%})",  ALERT_COLORS["MONITOR"]),
    ]:
        fig.add_hline(
            y=level,
            line=dict(color=color, width=1, dash="dash"),
            annotation_text=label,
            annotation_position="right",
            annotation_font_size=9,
        )

    fig.update_layout(
        title=f"Engine {engine_id} — Failure Probability",
        xaxis_title="Cycle",
        yaxis_title="Probability",
        yaxis_range=[0, 1.05],
        hovermode="x unified",
    )
    return _apply_theme(fig, height=CHART_HEIGHT)


def build_health_index(health_df: pd.DataFrame, engine_id: int) -> go.Figure:
    """
    Normalised VAE health indices for one engine over its observed cycles.

    All four metrics (recon_error, kl_div, js_div, wasserstein) are
    normalised to [0, 1] per engine so they're visually comparable on
    the same axis even though their raw scales differ significantly.

    Raw values appear in the hover tooltip so engineers can see actual numbers.
    Drift flag events are marked as vertical dashed lines.

    health_df: rows from health_indices.csv filtered to one engine.
    """
    df = health_df.sort_values("cycle").copy()

    fig = go.Figure()

    for col in ["recon_error", "kl_div", "js_div", "wasserstein"]:
        if col not in df.columns:
            continue

        # Normalise per engine for visual comparison
        norm_col = _normalise_column(df[col])

        fig.add_trace(go.Scatter(
            x=df["cycle"],
            y=norm_col,
            mode="lines",
            line=dict(color=HEALTH_COLORS[col], width=1.8),
            name=HEALTH_LABELS[col],
            hovertemplate=(
                f"Cycle %{{x}}<br>"
                f"{HEALTH_LABELS[col]}: %{{customdata:.5f}} (raw)<br>"
                f"Normalised: %{{y:.3f}}<extra></extra>"
            ),
            customdata=df[col],
        ))

    # Mark drift events as vertical dashed lines
    if "drift_flag" in df.columns:
        drift_cycles = df[df["drift_flag"]]["cycle"]
        for cyc in drift_cycles:
            fig.add_vline(
                x=cyc,
                line=dict(color="#DC2626", width=0.6, dash="dot"),
                opacity=0.4,
            )
        # Add a single legend entry for drift events
        if len(drift_cycles) > 0:
            fig.add_trace(go.Scatter(
                x=[drift_cycles.iloc[0]], y=[None],
                mode="lines",
                line=dict(color="#DC2626", width=1, dash="dot"),
                name=f"Drift detected ({len(drift_cycles)} windows)",
                showlegend=True,
            ))

    fig.update_layout(
        title=f"Engine {engine_id} — Health Indices (normalised to [0,1] per engine)",
        xaxis_title="Cycle",
        yaxis_title="Normalised health index  (1 = worst observed)",
        yaxis_range=[-0.05, 1.1],
        hovermode="x unified",
    )
    return _apply_theme(fig, height=CHART_HEIGHT)


def build_residual_scatter(engine_df: pd.DataFrame, engine_id: int) -> go.Figure:
    """
    Scatter plot of prediction residual (pred_rul − true_rul) vs cycle.

    A well-behaved model should show residuals:
      - Centred near zero (no systematic bias)
      - Random scatter (no trend over time)

    A systematic upward trend (predicting too much life) is more dangerous
    than a downward trend — the NASA asymmetric score reflects this.
    """
    df = engine_df.sort_values("cycle")

    # Colour each point by sign: red = late prediction (dangerous),
    # green = early prediction (safe / conservative)
    colors = df["residual"].apply(
        lambda r: ALERT_COLORS["CRITICAL"] if r > 0 else ALERT_COLORS["NOMINAL"]
    )

    fig = go.Figure()

    # Zero reference line
    fig.add_hline(
        y=0,
        line=dict(color="#6B7280", width=1.2, dash="dash"),
        annotation_text="Zero (perfect prediction)",
        annotation_position="right",
        annotation_font_size=9,
    )

    # Residual scatter
    fig.add_trace(go.Scatter(
        x=df["cycle"],
        y=df["residual"],
        mode="markers",
        marker=dict(color=colors, size=5, opacity=0.7),
        hovertemplate=(
            "Cycle %{x}<br>"
            "Residual: %{y:.1f} cycles<br>"
            "(positive = predicted too much life)<extra></extra>"
        ),
        name="Residual",
        showlegend=False,
    ))

    fig.update_layout(
        title=f"Engine {engine_id} — Prediction Residuals (pred − true)",
        xaxis_title="Cycle",
        yaxis_title="Residual (cycles)  —  red = late, green = early",
    )
    return _apply_theme(fig, height=CHART_HEIGHT)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL PERFORMANCE CHARTS
# ─────────────────────────────────────────────────────────────────────────────

def build_calibration_subplots(all_cal: dict[str, pd.DataFrame]) -> go.Figure:
    """
    2×2 grid of calibration plots — one per dataset.

    Each subplot shows:
      - Blue dots: measured coverage at each confidence level
      - Grey diagonal: perfect calibration reference
      - Light grey band: ±0.05 tolerance around diagonal

    A well-calibrated model has its dots sitting on or very close to
    the diagonal. Dots below = overconfident (intervals too narrow).
    Dots above = underconfident (intervals too wide).
    """
    datasets = ["FD001", "FD002", "FD003", "FD004"]
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=datasets,
        horizontal_spacing=0.12,
        vertical_spacing=0.18,
    )

    for idx, ds in enumerate(datasets):
        row = idx // 2 + 1
        col = idx %  2 + 1
        cal = all_cal.get(ds, pd.DataFrame())

        # Perfect calibration diagonal
        diag_x = [0.45, 1.0]
        fig.add_trace(go.Scatter(
            x=diag_x, y=diag_x,
            mode="lines",
            line=dict(color="#9CA3AF", width=1, dash="dash"),
            showlegend=(idx == 0),
            name="Perfect calibration",
        ), row=row, col=col)

        # ±0.05 tolerance band
        fig.add_trace(go.Scatter(
            x=diag_x + diag_x[::-1],
            y=[d + 0.05 for d in diag_x] + [d - 0.05 for d in diag_x[::-1]],
            fill="toself",
            fillcolor="rgba(156,163,175,0.12)",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=(idx == 0),
            name="±5% tolerance",
            hoverinfo="skip",
        ), row=row, col=col)

        # Actual coverage dots
        if not cal.empty:
            fig.add_trace(go.Scatter(
                x=cal["expected_coverage"],
                y=cal["actual_coverage"],
                mode="markers+lines",
                marker=dict(
                    color=DATASET_COLORS.get(ds, "#6366F1"),
                    size=8,
                ),
                line=dict(color=DATASET_COLORS.get(ds, "#6366F1"), width=1.5),
                showlegend=False,
                hovertemplate=(
                    "Expected: %{x:.0%}<br>"
                    "Actual:   %{y:.3f}<extra>" + ds + "</extra>"
                ),
            ), row=row, col=col)

    fig.update_xaxes(title_text="Expected coverage", range=[0.45, 1.0])
    fig.update_yaxes(title_text="Actual coverage",   range=[0.4,  1.05])
    return _apply_theme(fig, height=CHART_HEIGHT_TALL)


def build_bucket_rmse_chart(all_bucket: dict[str, pd.DataFrame]) -> go.Figure:
    """
    Grouped bar chart of RMSE split by RUL bucket for all four datasets.

    This is the most visually compelling result in the project: the
    end-of-life bars (where decisions are made) are dramatically shorter
    than early-life bars, showing the model is most accurate when it matters.

    Bucket order: early_life → mid_life → end_of_life (left to right).
    """
    bucket_order  = ["early_life", "mid_life", "end_of_life"]
    bucket_labels = {"early_life": "Early Life\n(RUL > 80)",
                     "mid_life":   "Mid Life\n(30 < RUL ≤ 80)",
                     "end_of_life":"End of Life\n(RUL ≤ 30)"}
    fig = go.Figure()

    for ds, bkt_df in all_bucket.items():
        if bkt_df.empty:
            continue
        rmse_by_bucket = (bkt_df.set_index("bucket")["rmse"]
                          .reindex(bucket_order))
        fig.add_trace(go.Bar(
            x=[bucket_labels[b] for b in bucket_order],
            y=rmse_by_bucket.values,
            name=ds,
            marker_color=DATASET_COLORS.get(ds, "#9CA3AF"),
            hovertemplate=f"{ds}<br>%{{x}}<br>RMSE: %{{y:.2f}} cycles<extra></extra>",
        ))

    fig.update_layout(
        barmode="group",
        xaxis_title="",
        yaxis_title="RMSE (cycles)",
        title="Prediction RMSE by RUL bucket — end-of-life is what matters",
        bargap=0.20,
        bargroupgap=0.08,
    )
    return _apply_theme(fig, height=CHART_HEIGHT)


def build_within_n_chart(all_metrics: dict[str, dict]) -> go.Figure:
    """
    Grouped bar chart of within-N-cycles accuracy for each dataset.
    Three groups: within 5 cycles, within 10 cycles, within 15 cycles.

    Operationally: within_10 is the most meaningful — it answers
    "what fraction of predictions are within one maintenance window?"
    """
    groups   = ["within_5_cycles", "within_10_cycles", "within_15_cycles"]
    labels   = ["Within ±5 cycles", "Within ±10 cycles", "Within ±15 cycles"]
    fig = go.Figure()

    for ds, m in all_metrics.items():
        all_seq = m.get("all_seq", {})
        vals = [all_seq.get(g, 0) * 100 for g in groups]   # convert to %
        fig.add_trace(go.Bar(
            x=labels, y=vals,
            name=ds,
            marker_color=DATASET_COLORS.get(ds, "#9CA3AF"),
            hovertemplate=f"{ds}: %{{y:.1f}}%<extra></extra>",
        ))

    fig.update_layout(
        barmode="group",
        xaxis_title="",
        yaxis_title="% of predictions",
        yaxis_range=[0, 105],
        title="Prediction accuracy within N cycles (higher = better)",
    )
    return _apply_theme(fig, height=CHART_HEIGHT)


def build_auc_comparison(all_metrics: dict[str, dict]) -> go.Figure:
    """
    Horizontal bar chart comparing AUC at 20 and 50 cycle horizons.
    AUC near 1.0 = model perfectly ranks near-failure engines above healthy ones.
    """
    datasets, auc20, auc50 = [], [], []
    for ds, m in all_metrics.items():
        all_seq = m.get("all_seq", {})
        if all_seq.get("auc_failure_20") is not None:
            datasets.append(ds)
            auc20.append(all_seq.get("auc_failure_20", 0))
            auc50.append(all_seq.get("auc_failure_50", 0))

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=datasets, x=auc50,
        orientation="h",
        name="AUC (50-cycle horizon)",
        marker_color=[DATASET_COLORS.get(d, "#9CA3AF") for d in datasets],
        opacity=0.45,
    ))
    fig.add_trace(go.Bar(
        y=datasets, x=auc20,
        orientation="h",
        name="AUC (20-cycle horizon)",
        marker_color=[DATASET_COLORS.get(d, "#9CA3AF") for d in datasets],
        opacity=0.95,
    ))

    fig.add_vline(x=0.95, line=dict(color="#6B7280", width=1, dash="dash"),
                  annotation_text="0.95", annotation_position="top")

    fig.update_layout(
        barmode="overlay",
        xaxis_title="AUC (area under ROC curve)",
        xaxis_range=[0.9, 1.002],
        title="Failure ranking quality by dataset — closer to 1.0 is better",
    )
    return _apply_theme(fig, height=CHART_HEIGHT_COMPACT)


# ─────────────────────────────────────────────────────────────────────────────
# HEALTH MONITOR CHARTS
# ─────────────────────────────────────────────────────────────────────────────

def build_recon_error_fleet(health_df: pd.DataFrame,
                              threshold: float | None = None) -> go.Figure:
    """
    Fleet-wide reconstruction error by RUL bucket: scatter of all
    (true_rul, recon_error) points coloured by drift_flag.

    This shows whether recon_error increases as RUL decreases — the key
    validation that the VAE is capturing degradation in input space.

    threshold: the drift threshold line (from health_summary.json)
    """
    fig = go.Figure()

    # Non-drifted points (below threshold)
    normal = health_df[~health_df["drift_flag"]] if "drift_flag" in health_df.columns \
             else health_df

    fig.add_trace(go.Scatter(
        x=normal["true_rul"],
        y=normal["recon_error"],
        mode="markers",
        marker=dict(color="#9CA3AF", size=2, opacity=0.4),
        name="Normal",
        hovertemplate="RUL: %{x:.0f}<br>Recon error: %{y:.4f}<extra></extra>",
    ))

    # Drifted points (above threshold)
    if "drift_flag" in health_df.columns:
        drifted = health_df[health_df["drift_flag"]]
        if len(drifted) > 0:
            fig.add_trace(go.Scatter(
                x=drifted["true_rul"],
                y=drifted["recon_error"],
                mode="markers",
                marker=dict(color=ALERT_COLORS["CRITICAL"], size=3, opacity=0.7),
                name="Drift detected",
                hovertemplate="RUL: %{x:.0f}<br>Recon error: %{y:.4f}<extra></extra>",
            ))

    # Drift threshold line
    if threshold is not None:
        fig.add_hline(
            y=threshold,
            line=dict(color="#DC2626", width=1.5, dash="dash"),
            annotation_text=f"Drift threshold ({threshold:.4f})",
            annotation_position="right",
            annotation_font_size=9,
        )

    fig.update_layout(
        title="Reconstruction error vs True RUL — rising error indicates degradation",
        xaxis_title="True RUL (cycles)",
        xaxis_autorange="reversed",       # right = healthy (high RUL), left = failing
        yaxis_title="Reconstruction error (MSE)",
        hovermode="closest",
    )
    return _apply_theme(fig, height=CHART_HEIGHT)


def build_latent_norm_vs_rul(health_df: pd.DataFrame) -> go.Figure:
    """
    Scatter of latent_mu_norm (L2 norm of encoded mean) vs true_rul.

    In a VAE, the prior is N(0,I) so the healthy reference is near the
    origin. As engines degrade, their encodings may move away from zero.
    This chart shows whether that pattern holds — it's the most direct
    visualisation of the information-geometric health index.
    """
    fig = go.Figure(go.Scatter(
        x=health_df["true_rul"],
        y=health_df["latent_mu_norm"],
        mode="markers",
        marker=dict(
            color=health_df["true_rul"],
            colorscale="RdYlGn",       # red = low RUL, green = high RUL
            size=2,
            opacity=0.5,
            colorbar=dict(title="True RUL"),
        ),
        hovertemplate="RUL: %{x:.0f}<br>Latent norm: %{y:.4f}<extra></extra>",
        showlegend=False,
    ))

    fig.update_layout(
        title="Latent space norm vs True RUL — divergence from healthy origin",
        xaxis_title="True RUL (cycles)",
        xaxis_autorange="reversed",
        yaxis_title="‖μ‖₂  (L2 norm of encoded mean)",
    )
    return _apply_theme(fig, height=CHART_HEIGHT)


def build_drift_engine_bar(health_df: pd.DataFrame) -> go.Figure:
    """
    Horizontal bar chart: for each engine, show the fraction of its
    windows that were flagged as drifted. Sorted by drift fraction.

    This gives a fleet-level view of which engines are most degraded
    according to the VAE's reconstruction error criterion.
    """
    if "drift_flag" not in health_df.columns:
        return go.Figure()

    drift_frac = (health_df.groupby("unit_number")["drift_flag"]
                  .mean()
                  .sort_values(ascending=True))

    # Colour bars: fraction > 0.5 = red, fraction > 0.2 = amber, else blue
    bar_colors = drift_frac.apply(
        lambda f: ALERT_COLORS["CRITICAL"] if f > 0.50
                  else (ALERT_COLORS["WARNING"] if f > 0.20
                  else ALERT_COLORS["MONITOR"])
    )

    fig = go.Figure(go.Bar(
        x=drift_frac.values,
        y=drift_frac.index.astype(str),
        orientation="h",
        marker_color=bar_colors.values,
        hovertemplate="Engine %{y}: %{x:.1%} of windows drifted<extra></extra>",
    ))

    fig.update_layout(
        title="Fraction of drifted windows per engine (sorted)",
        xaxis_title="Fraction of windows above drift threshold",
        xaxis_tickformat=".0%",
        yaxis_title="Engine",
        height=max(CHART_HEIGHT, len(drift_frac) * 8),  # scale with number of engines
    )
    return _apply_theme(fig)