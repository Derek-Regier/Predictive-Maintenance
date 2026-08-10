"""
dashboard/utils/styles.py

All visual constants and CSS for the dashboard.

This file is intentionally separated from data logic so a non-technical
collaborator can adjust colours, fonts, and sizing without touching
any Python that reads or processes data.

HOW TO EDIT
-----------
- Hex colours: any colour picker (e.g. coolors.co, htmlcolorcodes.com)
  gives you the #RRGGBB code to paste in.
- CHART_HEIGHT: pixels tall each chart is. 400 is a good default.
- FONT_FAMILY: any Google Font name, or "Arial", "Helvetica", etc.
- CSS block at the bottom: standard web CSS — controls card borders,
  padding, hover effects. Each section is labelled.
"""

# ── Alert tier colours ────────────────────────────────────────────────────────
# Used in charts, badges, KPI cards, and the fleet table.
# Change these to adjust the entire dashboard's alert colour scheme.

ALERT_COLORS = {
    "CRITICAL": "#DC2626",   # red   — failure likely within 20 cycles
    "WARNING":  "#D97706",   # amber — schedule maintenance within 50 cycles
    "MONITOR":  "#2563EB",   # blue  — flag for next inspection
    "NOMINAL":  "#16A34A",   # green — no action required
}

# Lighter background versions used for table row tinting and card fills
ALERT_BG_COLORS = {
    "CRITICAL": "#FEE2E2",
    "WARNING":  "#FEF3C7",
    "MONITOR":  "#DBEAFE",
    "NOMINAL":  "#D1FAE5",
}

# Emoji prefixes for alert tiers — shown in the fleet table's Alert column
ALERT_EMOJI = {
    "CRITICAL": "🔴 CRITICAL",
    "WARNING":  "🟡 WARNING",
    "MONITOR":  "🔵 MONITOR",
    "NOMINAL":  "🟢 NOMINAL",
}

# ── Dataset colours ───────────────────────────────────────────────────────────
# One colour per dataset — used in multi-dataset comparison charts so each
# dataset is consistently represented across all pages.

DATASET_COLORS = {
    "FD001": "#6366F1",   
    "FD002": "#10B981",   
    "FD003": "#F59E0B",   
    "FD004": "#EF4444",   
}

# ── Health index colours ──────────────────────────────────────────────────────
# Used on the health monitor page for the VAE metric trajectories.

HEALTH_COLORS = {
    "recon_error": "#EF4444",   # reconstruction error — primary signal
    "kl_div":      "#8B5CF6",   # KL divergence
    "js_div":      "#06B6D4",   # Jensen-Shannon divergence
    "wasserstein": "#F97316",   # Wasserstein distance
}

HEALTH_LABELS = {
    "recon_error": "Reconstruction Error",
    "kl_div":      "KL Divergence",
    "js_div":      "JS Divergence",
    "wasserstein": "Wasserstein Distance",
}

# ── Chart sizing ──────────────────────────────────────────────────────────────
# All charts use use_container_width=True so they fill the column they're in.
# These heights control how tall each chart is in pixels.

CHART_HEIGHT         = 420   # standard chart height
CHART_HEIGHT_TALL    = 550   # taller charts
CHART_HEIGHT_COMPACT = 300   # compact charts 

# ── Typography ────────────────────────────────────────────────────────────────
# FONT_FAMILY now drives BOTH the Plotly charts (via CHART_THEME below) and the
# Streamlit page text (via CUSTOM_CSS below) — change it in one place.
#
# To use a different Google Font:
#   1. Go to fonts.google.com, pick a font, click it, copy the "@import" line
#      it gives you under the "Use on the web" tab.
#   2. Paste that url(...) into GOOGLE_FONT_IMPORT below, replacing the existing one.
#   3. Set FONT_FAMILY to that font's name, followed by the same fallback stack.

GOOGLE_FONT_IMPORT = "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"
FONT_FAMILY = "Inter, -apple-system, sans-serif"
FONT_SIZE   = 12   # base chart font size in pt

# ── Signature ─────────────────────────────────────────────────────────────────
# Small initials shown in a fixed corner of every page, in a decorative font.
# Purely cosmetic — safe to edit any value below without breaking anything.

SIGNATURE_TEXT   = "DLR"                # your initials — swap for whatever you like
SIGNATURE_FONT   = "'Pinyon Script', cursive"   # a cursive/script Google Font
SIGNATURE_IMPORT = "https://fonts.googleapis.com/css2?family=Pinyon+Script&display=swap"
SIGNATURE_COLOR  = "#9CA3AF"           # muted grey so it doesn't compete for attention
SIGNATURE_SIZE   = "1.6rem"
SIGNATURE_CORNER = {                   # pick ONE pair: (top or bottom) + (left or right)
    "bottom": "14px",
    "right":  "22px",
}

# ── Chart theme ───────────────────────────────────────────────────────────────
# Applied to every Plotly figure via fig.update_layout(**CHART_THEME).

CHART_THEME = {
    "font":         {"family": FONT_FAMILY, "size": FONT_SIZE, "color": "#374151"},
    "paper_bgcolor": "rgba(0,0,0,0)",   # transparent — inherits page background
    "plot_bgcolor":  "rgba(0,0,0,0)",
    "margin":       {"t": 50, "b": 40, "l": 50, "r": 20},
    "legend":       {"bgcolor": "rgba(0,0,0,0)", "borderwidth": 0},
    "xaxis":        {"gridcolor": "#F3F4F6", "linecolor": "#E5E7EB", "zeroline": False},
    "yaxis":        {"gridcolor": "#F3F4F6", "linecolor": "#E5E7EB", "zeroline": False},
}

# ── CSS injected into every page ──────────────────────────────────────────────
# Loaded on every page via st.markdown(CUSTOM_CSS, unsafe_allow_html=True).
# Edit this block to change the look of cards, the sidebar, metric widgets, etc.
# It's an f-string so the variables above (fonts, signature) can be dropped in —
# if you're not touching those, everything below is plain CSS.

_corner_css = "; ".join(f"{side}: {value}" for side, value in SIGNATURE_CORNER.items())

CUSTOM_CSS = f"""
<style>

/* ── Fonts ── */
@import url('{GOOGLE_FONT_IMPORT}');
@import url('{SIGNATURE_IMPORT}');

/* ── Page background ── */
[data-testid="stAppViewContainer"] {{
    background-color: #FAFAFA;   /* change this hex to recolour the whole page */
}}

/* ── Apply FONT_FAMILY to Streamlit's own text, not just charts ── */
html, body, [class*="css"] {{
    font-family: {FONT_FAMILY};
}}

/* ── General page ── */
.main .block-container {{
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}}

/* ── Signature — fixed initials in a page corner ── */
.dashboard-signature {{
    position: fixed;
    {_corner_css};
    font-family: {SIGNATURE_FONT};
    font-size: {SIGNATURE_SIZE};
    color: {SIGNATURE_COLOR};
    opacity: 0.8;
    z-index: 9999;
    pointer-events: none;   /* clicks pass through it — never blocks the UI */
    user-select: none;
}}

/* ── KPI metric cards ── */
/* Streamlit's [data-testid="metric-container"] targets st.metric() blocks */
[data-testid="metric-container"] {{
    background-color: #F9FAFB;
    border: 1px solid #E5E7EB;
    border-radius: 10px;
    padding: 16px 20px;
}}
[data-testid="metric-container"]:hover {{
    border-color: #D1D5DB;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}}

/* ── Section dividers ── */
hr {{
    border: none;
    border-top: 1px solid #E5E7EB;
    margin: 1.2rem 0;
}}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background-color: #F8FAFC;
}}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiSelect label {{
    font-weight: 600;
    font-size: 0.85rem;
    color: #374151;
}}

/* ── Dataframe header ── */
[data-testid="stDataFrame"] thead th {{
    background-color: #F3F4F6 !important;
    font-weight: 600;
    font-size: 0.82rem;
    color: #374151;
}}

/* ── Tab styling ── */
[data-testid="stTab"] {{
    font-size: 0.88rem;
    font-weight: 500;
}}

/* ── Info / warning boxes ── */
[data-testid="stInfo"],
[data-testid="stWarning"] {{
    border-radius: 8px;
}}

</style>
"""

# ── Signature HTML snippet ──────────────────────────────────────────────────
# Actually renders the initials. Import + inject this alongside CUSTOM_CSS —
# see the usage note in app.py / each page file.

SIGNATURE_HTML = f'<div class="dashboard-signature">{SIGNATURE_TEXT}</div>'