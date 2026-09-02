"""
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
    HEALTH_DESCRIPTIONS, GEOMETRY_INDICES, HIGHER_IS_HEALTHIER, SCORE_BANDS,
    CHART_HEIGHT, CHART_HEIGHT_TALL, CHART_HEIGHT_COMPACT, CHART_THEME,
)


# HELPER

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

# FLEET OVERVIEW CHARTS

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

# ENGINE DEEP DIVE CHARTS

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

    # End-of-life danger zone (RUL <= 30)
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

    # 90% prediction interval (shaded band) 
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

    # True RUL (ground truth — shown as context, not prediction) 
    fig.add_trace(go.Scatter(
        x=df["cycle"], y=df["true_rul"],
        mode="lines",
        line=dict(color="#9CA3AF", width=1.5, dash="dot"),
        name="True RUL",
        hovertemplate="Cycle %{x}: true RUL = %{y:.0f}<extra></extra>",
    ))

    # Predicted mean RUL (the model's point estimate) 
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


def build_failure_probability(engine_df: pd.DataFrame, engine_id: int, thresholds: dict | None = None) -> go.Figure:
    """
    Failure probability trajectory for one engine.
    Two lines: P(RUL<20) in red (CRITICAL threshold) and P(RUL<50)
    in amber (WARNING threshold). Horizontal dashed reference lines
    show where alert tiers fire.

    thresholds: optional dict with keys "critical_prob_20" and "warning_prob_50"
                loaded from datasets.yaml. Defaults to 0.90 / 0.75 if not given.
    """
    df = engine_df.sort_values("cycle")
    thr = thresholds or {}
    crit = thr.get("critical_prob_20", 0.90)
    warn = thr.get("warning_prob_50",  0.75)
    mon = thr.get("monitor_prob_50",  0.50)

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


# Preferred plotting order for the per-engine health index chart. Only the
# columns that actually exist in the DataFrame are drawn, so this same
# function works against both the legacy and current health_indices.csv
# schemas without any branching at the call site.
_HEALTH_INDEX_ORDER = [
    "mahalanobis_self",
    "mahalanobis",
    "fisher_rao",
    "recon_error",
    "kl_div",
    "js_div",
    "wasserstein",
]


def _contiguous_runs(mask: pd.Series, x: pd.Series) -> list[tuple]:
    """
    Collapse a boolean mask into a list of (x_start, x_end) spans.

    Drawing one vertical line per flagged window means hundreds of Plotly
    shapes on a long engine record, which is slow to render and reads as
    a solid wall of red rather than as distinct alarm episodes. Shading
    contiguous runs instead gives one shape per episode.
    """
    mask = mask.to_numpy(dtype=bool)
    x = x.to_numpy()
    if not mask.any():
        return []

    runs = []
    start = None
    for k, flagged in enumerate(mask):
        if flagged and start is None:
            start = k
        elif not flagged and start is not None:
            runs.append((x[start], x[k - 1]))
            start = None
    if start is not None:
        runs.append((x[start], x[-1]))
    return runs


def build_health_index(health_df: pd.DataFrame, engine_id: int,
                       columns: list[str] | None = None) -> go.Figure:
    """
    Normalised health indices for one engine over its observed cycles.

    Each index is min-max normalised to [0, 1] PER ENGINE so indices on
    wildly different scales (a Mahalanobis distance of ~40 and a
    reconstruction error of ~0.14) can share an axis. Raw values stay in
    the hover tooltip, because the normalised value is only meaningful
    relative to this engine's own range.

    A consequence worth stating on the page: normalising per engine means
    every engine's worst window maps to 1.0, including engines that never
    actually degraded. The chart shows SHAPE, not severity. Absolute
    severity is what health_score and the fleet scatter are for.

    Confirmed alarm episodes are shaded rather than drawn as one line per
    window. Where both raw and confirmed flags exist, only confirmed
    episodes are shaded — the raw ones are shown on the alarm timeline.

    health_df: rows from health_indices.csv filtered to one engine.
    columns:   optional explicit index list; defaults to whatever of
               _HEALTH_INDEX_ORDER is present.
    """
    df = health_df.sort_values("cycle").copy()

    if columns is None:
        columns = [c for c in _HEALTH_INDEX_ORDER if c in df.columns]

    fig = go.Figure()

    for col in columns:
        if col not in df.columns:
            continue

        norm_col = _normalise_column(df[col])

        fig.add_trace(go.Scatter(
            x=df["cycle"],
            y=norm_col,
            mode="lines",
            line=dict(color=HEALTH_COLORS.get(col, "#6B7280"), width=1.8),
            name=HEALTH_LABELS.get(col, col),
            hovertemplate=(
                f"Cycle %{{x}}<br>"
                f"{HEALTH_LABELS.get(col, col)}: %{{customdata:.5f}} (raw)<br>"
                f"Normalised: %{{y:.3f}}<extra></extra>"
            ),
            customdata=df[col],
        ))

    # Shade confirmed alarm episodes.
    shade_specs = [
        ("drift_flag", "#DC2626", "Reconstruction drift"),
        ("geo_alarm", "#4F46E5", "Geometry alarm"),
    ]
    for flag_col, colour, label in shade_specs:
        if flag_col not in df.columns:
            continue
        runs = _contiguous_runs(df[flag_col].astype(bool), df["cycle"])
        for x0, x1 in runs:
            fig.add_vrect(
                x0=x0, x1=max(x1, x0 + 0.5),
                fillcolor=colour, opacity=0.10,
                line_width=0, layer="below",
            )
        if runs:
            # One invisible trace carries the legend entry for the shading.
            fig.add_trace(go.Scatter(
                x=[df["cycle"].iloc[0]], y=[None],
                mode="markers",
                marker=dict(color=colour, size=10, symbol="square", opacity=0.35),
                name=f"{label} ({len(runs)} episode{'s' if len(runs) != 1 else ''})",
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
    than a downward trend and the NASA asymmetric score reflects this.
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


# MODEL PERFORMANCE CHARTS

# (column, legend label, line dash, opacity) for the calibration subplots.
# Order matters — it is the legend order.
_CAL_SERIES = [
    ("actual_coverage",            "Raw (sigma scaled)",        "solid", 1.00),
    ("recal_coverage",             "+ CDF recalibration",       "dot",   0.95),
    ("uncensored_coverage",        "Raw, uncensored only",      "dash",  0.60),
    ("uncensored_recal_coverage",  "Recal, uncensored only",    "longdash", 0.60),
]

def build_calibration_subplots(all_cal: dict[str, pd.DataFrame],
                               series: list[str] | None = None) -> go.Figure:
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
        col_i = idx %  2 + 1
        cal = all_cal.get(ds, pd.DataFrame())

        # Perfect calibration diagonal
        diag_x = [0.45, 1.0]
        fig.add_trace(go.Scatter(
            x=diag_x, y=diag_x,
            mode="lines",
            line=dict(color="#9CA3AF", width=1, dash="dash"),
            showlegend=(idx == 0),
            name="Perfect calibration",
        ), row=row, col=col_i)

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
        ), row=row, col=col_i)

        # Coverage curves. Which columns exist depends on how evaluate.py
        # was run: `actual_coverage` is always present, the other three
        # appear once a CDF recalibrator and max_rul are configured.
        for col, label, dash, opacity in _CAL_SERIES:
            if cal.empty or col not in cal.columns:
                continue
            if series is not None and col not in series:
                continue
            fig.add_trace(go.Scatter(
                x=cal["expected_coverage"],
                y=cal[col],
                mode="markers+lines",
                marker=dict(color=DATASET_COLORS.get(ds, "#6366F1"),
                            size=7, opacity=opacity),
                line=dict(color=DATASET_COLORS.get(ds, "#6366F1"),
                          width=1.6, dash=dash),
                name=label,
                legendgroup=col,
                showlegend=(idx == 0),
                opacity=opacity,
                hovertemplate=(f"{label}<br>expected %{{x:.2f}}"
                               "<br>actual %{y:.3f}<extra></extra>"),
            ), row=row, col=col_i)

        # (legacy single-series block retained below, now unreachable)
        if False:
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
            ), row=row, col=col_i)

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
                     "mid_life": "Mid Life\n(30 < RUL ≤ 80)",
                     "end_of_life":"End of Life\n(RUL ≤ 30)"}
    fig = go.Figure()

    for ds, bkt_df in all_bucket.items():
        if bkt_df.empty:
            continue
        rmse_by_bucket = (bkt_df.set_index("bucket")["rmse"].reindex(bucket_order))
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
    Keys match what evaluate.py writes to metrics_summary.json:
      all_seq.within_5_pct, all_seq.within_10_pct, all_seq.within_15_pct
    """
    # These keys match the actual JSON structure from evaluate.py
    groups = ["within_5_pct", "within_10_pct", "within_15_pct"]
    labels = ["Within ±5 cycles", "Within ±10 cycles", "Within ±15 cycles"]
    fig    = go.Figure()
 
    for ds, m in all_metrics.items():
        all_seq = m.get("all_seq", {})
        vals    = [all_seq.get(g, 0) * 100 for g in groups]  # to %
 
        # Skip datasets where all values are zero (data not loaded yet)
        if all(v == 0 for v in vals):
            continue
 
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


# HEALTH MONITOR CHARTS

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
        yaxis_title="‖μ‖₂ (L2 norm of encoded mean)",
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

    drift_frac = (health_df.groupby("unit_number")["drift_flag"].mean().sort_values(ascending=True))

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


# ─────────────────────────────────────────────────────────────────────────────
# INFORMATION-GEOMETRY CHARTS
#
# These were added when the health monitor moved from a diagonal
# posterior reference to a full-covariance healthy population reference.
# Every function here degrades gracefully if the column it wants is
# absent, so the page still renders against a legacy health_indices.csv.
# ─────────────────────────────────────────────────────────────────────────────

def build_geometry_fleet(health_df: pd.DataFrame,
                         index_col: str,
                         threshold: float | None = None,
                         alarm_col: str | None = None,
                         log_y: bool = False) -> go.Figure:
    """
    Generic fleet scatter: any health index against true RUL, coloured by
    whether that window tripped an alarm.

    This generalises build_recon_error_fleet so the page can offer a
    selector over every available index rather than hard-coding
    reconstruction error. Being able to flip between Mahalanobis and
    reconstruction error on the same axes is the fastest way to see that
    they are not measuring the same thing.

    The x axis is reversed so the chart reads left-to-right as time:
    healthy (high RUL) on the right, failing on the left.

    log_y helps when an index is heavily right-skewed — KL divergence
    against a tight healthy population routinely spans three orders of
    magnitude, which flattens everything into the bottom of a linear axis.
    """
    if index_col not in health_df.columns:
        return go.Figure()

    fig = go.Figure()
    label = HEALTH_LABELS.get(index_col, index_col)

    has_alarm = alarm_col is not None and alarm_col in health_df.columns
    normal = health_df[~health_df[alarm_col].astype(bool)] if has_alarm else health_df

    fig.add_trace(go.Scatter(
        x=normal["true_rul"],
        y=normal[index_col],
        mode="markers",
        marker=dict(color="#9CA3AF", size=2, opacity=0.4),
        name="Normal",
        hovertemplate=f"RUL: %{{x:.0f}}<br>{label}: %{{y:.4f}}<extra></extra>",
    ))

    if has_alarm:
        alarmed = health_df[health_df[alarm_col].astype(bool)]
        if len(alarmed) > 0:
            fig.add_trace(go.Scatter(
                x=alarmed["true_rul"],
                y=alarmed[index_col],
                mode="markers",
                marker=dict(color=ALERT_COLORS["CRITICAL"], size=3, opacity=0.7),
                name="Alarm confirmed",
                hovertemplate=f"RUL: %{{x:.0f}}<br>{label}: %{{y:.4f}}<extra></extra>",
            ))

    if threshold is not None:
        fig.add_hline(
            y=threshold,
            line=dict(color="#DC2626", width=1.5, dash="dash"),
            annotation_text=f"Alarm threshold ({threshold:.4g})",
            annotation_position="right",
            annotation_font_size=9,
        )

    fig.update_layout(
        title=f"{label} vs True RUL",
        xaxis_title="True RUL (cycles)",
        xaxis_autorange="reversed",
        yaxis_title=label,
        yaxis_type="log" if log_y else "linear",
        hovermode="closest",
    )
    return _apply_theme(fig, height=CHART_HEIGHT)


def build_index_small_multiples(health_df: pd.DataFrame,
                                columns: list[str],
                                n_cols: int = 3,
                                sample: int = 6000) -> go.Figure:
    """
    A grid of index-vs-RUL scatters, one panel per index, on shared x.

    The point of showing them together rather than one at a time is that
    the differences are structural, not cosmetic: a good index forms a
    visible fan opening toward low RUL, a dead one forms a horizontal
    band. Side by side that is obvious in a second; one at a time it
    takes an argument.

    `sample` caps the number of points per panel — Plotly gets sluggish
    past ~10k markers per subplot and FD004 has far more windows than
    that. Sampling is deterministic (fixed seed) so the chart does not
    reshuffle on every Streamlit rerun.
    """
    columns = [c for c in columns if c in health_df.columns]
    if not columns:
        return go.Figure()

    df = health_df
    if len(df) > sample:
        df = df.sample(sample, random_state=0)

    n_rows = (len(columns) + n_cols - 1) // n_cols
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[HEALTH_LABELS.get(c, c) for c in columns],
        horizontal_spacing=0.08, vertical_spacing=0.14,
    )

    for k, col in enumerate(columns):
        r, c = divmod(k, n_cols)
        fig.add_trace(
            go.Scatter(
                x=df["true_rul"],
                y=df[col],
                mode="markers",
                marker=dict(
                    color=HEALTH_COLORS.get(col, "#6B7280"),
                    size=2, opacity=0.35,
                ),
                name=HEALTH_LABELS.get(col, col),
                showlegend=False,
                hovertemplate=f"RUL: %{{x:.0f}}<br>value: %{{y:.4f}}<extra></extra>",
            ),
            row=r + 1, col=c + 1,
        )
        fig.update_xaxes(autorange="reversed", row=r + 1, col=c + 1)

    fig.update_layout(
        title="Every health index against true RUL — a good index fans out toward low RUL",
    )
    fig.update_annotations(font_size=11)
    return _apply_theme(fig, height=max(CHART_HEIGHT, 230 * n_rows))


def build_health_score_trajectory(engine_df: pd.DataFrame,
                                  engine_id: int) -> go.Figure:
    """
    Health score (0-100) over an engine's life, with the score bands
    shaded behind it and confirmed alarm episodes marked.

    Unlike the normalised index chart, this axis is absolute and
    comparable across engines and datasets: the score is a percentile
    against HEALTHY VALIDATION data, not against this engine's own range.
    An engine that never leaves the green band never degraded, and the
    chart says so plainly rather than rescaling its noise to fill the
    frame.
    """
    if "health_score" not in engine_df.columns:
        return go.Figure()

    df = engine_df.sort_values("cycle")
    fig = go.Figure()

    # Score bands as background stripes.
    for lo, hi, colour, label in SCORE_BANDS:
        fig.add_hrect(
            y0=lo, y1=min(hi, 100),
            fillcolor=colour, opacity=0.07,
            line_width=0, layer="below",
            annotation_text=label,
            annotation_position="right",
            annotation_font_size=8,
        )

    fig.add_trace(go.Scatter(
        x=df["cycle"],
        y=df["health_score"],
        mode="lines",
        line=dict(color="#0F172A", width=2),
        name="Health score",
        customdata=df["true_rul"],
        hovertemplate=("Cycle %{x}<br>Health score: %{y:.1f}"
                       "<br>True RUL: %{customdata:.0f}<extra></extra>"),
    ))

    for flag_col, colour in [("geo_alarm", "#4F46E5"), ("drift_flag", "#DC2626")]:
        if flag_col not in df.columns:
            continue
        for x0, x1 in _contiguous_runs(df[flag_col].astype(bool), df["cycle"]):
            fig.add_vrect(x0=x0, x1=max(x1, x0 + 0.5), fillcolor=colour,
                          opacity=0.10, line_width=0, layer="below")

    fig.update_layout(
        title=f"Engine {engine_id} — Health Score (100 = indistinguishable from healthy)",
        xaxis_title="Cycle",
        yaxis_title="Health score",
        yaxis_range=[-2, 102],
        hovermode="x unified",
    )
    return _apply_theme(fig, height=CHART_HEIGHT)


def build_alarm_timeline(engine_df: pd.DataFrame, engine_id: int) -> go.Figure:
    """
    Raw vs confirmed alarm state over an engine's life, as four stacked
    step traces.

    This exists to make the persistence filter visible. A threshold
    calibrated at the 99th healthy percentile fires on 1% of healthy
    windows by construction, so raw flags scatter sporadic single-window
    hits across the whole record; the confirmed row shows what survives
    the k-of-n requirement. Seeing the two rows together is the clearest
    way to justify the filter — and to notice if it has been set so
    aggressively that it is swallowing real detections.
    """
    df = engine_df.sort_values("cycle")

    rows = [
        ("drift_raw", "Recon drift (raw)", "#FCA5A5"),
        ("drift_flag", "Recon drift (confirmed)", "#DC2626"),
        ("geo_alarm_raw", "Geometry (raw)", "#A5B4FC"),
        ("geo_alarm", "Geometry (confirmed)", "#4F46E5"),
    ]
    rows = [r for r in rows if r[0] in df.columns]
    if not rows:
        return go.Figure()

    fig = go.Figure()

    for level, (col, label, colour) in enumerate(rows):
        flags = df[col].astype(bool)
        # Plot only the flagged cycles as markers on this row's baseline.
        flagged_cycles = df.loc[flags, "cycle"]
        fig.add_trace(go.Scatter(
            x=flagged_cycles,
            y=[level] * len(flagged_cycles),
            mode="markers",
            marker=dict(color=colour, size=6, symbol="line-ns-open",
                        line=dict(width=2, color=colour)),
            name=label,
            hovertemplate=f"{label}<br>Cycle %{{x}}<extra></extra>",
        ))

    fig.update_layout(
        title=f"Engine {engine_id} — Alarm timeline (raw vs persistence-filtered)",
        xaxis_title="Cycle",
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(len(rows))),
            ticktext=[r[1] for r in rows],
            range=[-0.6, len(rows) - 0.4],
        ),
        showlegend=False,
        hovermode="closest",
    )
    return _apply_theme(fig, height=CHART_HEIGHT_COMPACT)


def build_index_quality_bar(metrics_df: pd.DataFrame,
                            metric: str = "composite") -> go.Figure:
    """
    Horizontal bar chart of one prognostic-quality metric across indices,
    from health_index_metrics.csv.

    metric: "composite", "monotonicity", "trendability", "prognosability",
            "mean_engine_spearman", or "frac_correct_direction".

    mean_engine_spearman is plotted as its absolute value, because the
    sign is a property of the index's direction (health_score rises with
    health, every other index rises with damage) and not a property of
    its quality. The unsigned magnitude is what is comparable.
    """
    if metrics_df.empty or metric not in metrics_df.columns:
        return go.Figure()

    df = metrics_df.copy()
    values = df[metric].abs() if metric == "mean_engine_spearman" else df[metric]
    df = df.assign(_v=values).dropna(subset=["_v"]).sort_values("_v")

    fig = go.Figure(go.Bar(
        x=df["_v"],
        y=[HEALTH_LABELS.get(i, i) for i in df["index"]],
        orientation="h",
        marker_color=[HEALTH_COLORS.get(i, "#6B7280") for i in df["index"]],
        hovertemplate="%{y}: %{x:.3f}<extra></extra>",
    ))

    pretty = metric.replace("_", " ").title()
    fig.update_layout(
        title=f"Health index quality — {pretty} (higher is better)",
        xaxis_title=pretty,
        yaxis_title="",
    )
    return _apply_theme(fig, height=max(CHART_HEIGHT_COMPACT, 34 * len(df) + 110))


def build_index_quality_radar(metrics_df: pd.DataFrame,
                              top_n: int = 5) -> go.Figure:
    """
    Radar comparison of the three Coble & Hines metrics for the best few
    indices.

    A radar is the right shape here because the three metrics are
    genuinely different axes rather than a ranking: an index can be
    highly monotone but untrendable (moves smoothly, but in different
    directions on different engines), and the polygon shape shows that
    trade-off at a glance where three separate bar charts would not.

    Kept to the top few indices — a radar with ten overlapping polygons
    is unreadable.
    """
    if metrics_df.empty:
        return go.Figure()

    axes = ["monotonicity", "trendability", "prognosability"]
    axes = [a for a in axes if a in metrics_df.columns]
    if len(axes) < 3:
        return go.Figure()

    df = metrics_df.dropna(subset=axes).nlargest(top_n, "composite")

    fig = go.Figure()
    for _, row in df.iterrows():
        name = row["index"]
        values = [float(row[a]) for a in axes]
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],            # close the polygon
            theta=[a.title() for a in axes] + [axes[0].title()],
            fill="toself",
            opacity=0.35,
            line=dict(color=HEALTH_COLORS.get(name, "#6B7280"), width=2),
            name=HEALTH_LABELS.get(name, name),
        ))

    fig.update_layout(
        title="Prognostic quality profile — top indices",
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
    )
    return _apply_theme(fig, height=CHART_HEIGHT)


def build_latent_diagnostics_bar(posterior: dict) -> go.Figure:
    """
    Per-dimension KL and posterior-mean variance, from the VAE diagnostics
    stored in model_registry.yaml.

    Two bars per latent dimension because they answer different questions
    and can disagree:

      per-dim KL      how far q(z_j|x) sits from the N(0,1) prior. Can be
                      non-zero purely from a constant offset.
      Var_x(mu_j)     how much the posterior mean MOVES as the input
                      changes. This is the honest test of whether the
                      dimension carries information (Burda et al. 2016).

    The failure this chart is built to expose: a dimension with healthy
    KL but near-zero mean variance is a dead unit wearing a disguise,
    which is exactly what the pre-fix FD001 run had across the board.
    """
    per_dim_kl = posterior.get("per_dim_kl") or []
    mu_var = posterior.get("per_dim_mu_var") or []
    if not per_dim_kl:
        return go.Figure()

    dims = list(range(len(per_dim_kl)))
    threshold = posterior.get("active_threshold", 1e-2)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
        subplot_titles=("KL from prior, per latent dimension (nats)",
                        "Var(mu) per latent dimension — the active-unit test"),
    )

    fig.add_trace(go.Bar(
        x=dims, y=per_dim_kl, marker_color="#8B5CF6",
        name="KL", hovertemplate="dim %{x}: %{y:.4f} nats<extra></extra>",
        showlegend=False,
    ), row=1, col=1)

    if mu_var:
        colours = ["#16A34A" if v > threshold else "#D1D5DB" for v in mu_var]
        fig.add_trace(go.Bar(
            x=dims, y=mu_var, marker_color=colours,
            name="Var(mu)",
            hovertemplate="dim %{x}: %{y:.5f}<extra></extra>",
            showlegend=False,
        ), row=2, col=1)
        fig.add_hline(
            y=threshold, row=2, col=1,
            line=dict(color="#DC2626", width=1, dash="dash"),
            annotation_text=f"active threshold ({threshold:g})",
            annotation_position="right", annotation_font_size=8,
        )
        fig.update_yaxes(type="log", row=2, col=1)

    fig.update_xaxes(title_text="Latent dimension", row=2, col=1)
    fig.update_layout(title="VAE posterior diagnostics", bargap=0.25)
    fig.update_annotations(font_size=11)
    return _apply_theme(fig, height=CHART_HEIGHT_TALL)


def build_engine_alarm_bar(health_df: pd.DataFrame,
                           flag_col: str = "drift_flag") -> go.Figure:
    """
    Per-engine fraction of windows under a given confirmed alarm.

    Generalises build_drift_engine_bar to any flag column so the page can
    show reconstruction drift and the geometry alarm on the same footing.
    """
    if flag_col not in health_df.columns:
        return go.Figure()

    frac = (health_df.groupby("unit_number")[flag_col]
            .mean().sort_values(ascending=True))

    bar_colors = frac.apply(
        lambda f: ALERT_COLORS["CRITICAL"] if f > 0.50
                  else (ALERT_COLORS["WARNING"] if f > 0.20
                  else ALERT_COLORS["MONITOR"])
    )

    pretty = "Geometry alarm" if flag_col.startswith("geo") else "Reconstruction drift"

    fig = go.Figure(go.Bar(
        x=frac.values,
        y=frac.index.astype(str),
        orientation="h",
        marker_color=bar_colors.values,
        hovertemplate="Engine %{y}: %{x:.1%} of windows<extra></extra>",
    ))

    fig.update_layout(
        title=f"{pretty} — fraction of windows flagged, per engine",
        xaxis_title="Fraction of windows flagged",
        xaxis_tickformat=".0%",
        yaxis_title="Engine",
        height=max(CHART_HEIGHT, len(frac) * 8),
    )
    return _apply_theme(fig)


def build_lead_time_hist(health_df: pd.DataFrame,
                         flag_col: str = "geo_alarm") -> go.Figure:
    """
    Distribution of true RUL at the first CONFIRMED alarm, across engines.

    This is the operationally meaningful summary: how much life was left
    when the monitor first spoke up. A histogram piled against the
    right-hand edge (high RUL) means the monitor is firing immediately
    and the threshold is too tight; one piled at zero means it only fires
    once the engine has already failed.

    Engines that never alarm are excluded and reported in the title,
    since they have no first-alarm RUL to plot — and quietly dropping
    them would make a monitor that alarms on three engines look excellent.
    """
    if flag_col not in health_df.columns:
        return go.Figure()

    alarmed = health_df[health_df[flag_col].astype(bool)]
    n_total = health_df["unit_number"].nunique()

    if alarmed.empty:
        fig = go.Figure()
        fig.update_layout(title=f"No engines alarmed on {flag_col}")
        return _apply_theme(fig, height=CHART_HEIGHT_COMPACT)

    first = alarmed.sort_values("cycle").groupby("unit_number").first()
    n_alarmed = len(first)

    pretty = "Geometry alarm" if flag_col.startswith("geo") else "Reconstruction drift"

    fig = go.Figure(go.Histogram(
        x=first["true_rul"],
        nbinsx=25,
        marker_color=HEALTH_COLORS.get("mahalanobis" if flag_col.startswith("geo")
                                       else "recon_error", "#6B7280"),
        hovertemplate="RUL %{x}: %{y} engines<extra></extra>",
    ))

    fig.update_layout(
        title=(f"{pretty} — true RUL at first confirmed alarm "
               f"({n_alarmed}/{n_total} engines alarmed)"),
        xaxis_title="True RUL when the alarm first fired (cycles)",
        yaxis_title="Engines",
        bargap=0.05,
    )
    return _apply_theme(fig, height=CHART_HEIGHT_COMPACT)