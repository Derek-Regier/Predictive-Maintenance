"""
All visual constants and CSS for the dashboard.

This file is intentionally separated from data logic so a non-technical
collaborator can adjust colours, fonts, and sizing without touching
any Python that reads or processes data.

HOW TO EDIT
-----------
- Hex colours: any colour picker (e.g. coolors.co, htmlcolorcodes.com)
  gives you the #RRGGBB code to paste in.
- CHART_HEIGHT: pixels tall each chart is. 400 is a good default.
- FONT_FAMILY: any system font name — "Arial", "Helvetica", etc.
- CSS block at the bottom: standard web CSS — controls card borders,
  padding, hover effects. Each section is labelled.
- Signature: edit SIGNATURE_TEXT, SIGNATURE_COLOR, or SIGNATURE_SIZE.
  Font is now Inter bold (same as the rest of the dashboard).
"""

# Alert tier colours

ALERT_COLORS = {
    "CRITICAL": "#DC2626",
    "WARNING":  "#D97706",
    "MONITOR":  "#2563EB",
    "NOMINAL":  "#16A34A",
}

ALERT_BG_COLORS = {
    "CRITICAL": "#FEE2E2",
    "WARNING":  "#FEF3C7",
    "MONITOR":  "#DBEAFE",
    "NOMINAL":  "#D1FAE5",
}

ALERT_EMOJI = {
    "CRITICAL": "🔴 CRITICAL",
    "WARNING":  "🟡 WARNING",
    "MONITOR":  "🔵 MONITOR",
    "NOMINAL":  "🟢 NOMINAL",
}

# Dataset colours

DATASET_COLORS = {
    "FD001": "#6366F1",
    "FD002": "#10B981",
    "FD003": "#F59E0B",
    "FD004": "#EF4444",
}

# Health index colours 

HEALTH_COLORS = {
    "recon_error": "#EF4444",
    "kl_div":      "#8B5CF6",
    "js_div":      "#06B6D4",
    "wasserstein": "#F97316",
}

HEALTH_LABELS = {
    "recon_error": "Reconstruction Error",
    "kl_div": "KL Divergence",
    "js_div": "JS Divergence",
    "wasserstein": "Wasserstein Distance",
}

# Chart sizing 

CHART_HEIGHT = 420
CHART_HEIGHT_TALL = 550
CHART_HEIGHT_COMPACT = 300

# Typography 
# FONT_FAMILY drives both Plotly charts (via CHART_THEME) and Streamlit page
# text (via CUSTOM_CSS). Change it in one place and it applies everywhere.
#
# To use a different Google Font:
#   1. Go to fonts.google.com, pick a font, copy the @import URL.
#   2. Paste it into GOOGLE_FONT_IMPORT below.
#   3. Set FONT_FAMILY to that font's name + the same fallback stack.

GOOGLE_FONT_IMPORT = "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap"
FONT_FAMILY = "Inter, -apple-system, sans-serif"
FONT_SIZE = 12

# Signature 
# Small initials shown in a fixed corner of every page.
# Font is now Inter bold (non-cursive) — same as the dashboard body font
# but at weight 800 so it reads as a distinct mark without being decorative.
# SIGNATURE_IMPORT is no longer needed (no external font) but kept as an
# empty string so any code that references it doesn't break.

SIGNATURE_TEXT = "DLR"
SIGNATURE_FONT = "Inter, -apple-system, sans-serif"   # non-cursive, matches dashboard
SIGNATURE_WEIGHT = "900"                                 # extra-bold
SIGNATURE_IMPORT = ""                                    # no external font needed
SIGNATURE_COLOR = "#9CA3AF"
SIGNATURE_SIZE = "1.0rem"                              # slightly smaller than cursive
SIGNATURE_CORNER = {
    "bottom": "14px",
    "right":  "22px",
}

# Header bar 
# Persistent top bar shown on every page. Edit the text in HEADER_HTML below.
# Colours are set in the CSS section (.pm-header-bar).

HEADER_BG_COLOR = "#0F172A"   # dark navy bar — change for a different colour
HEADER_TITLE_COLOR = "#F8FAFC"
HEADER_SUBTITLE_COLOR = "#94A3B8"
NATIVE_HEADER_HEIGHT= "3.75rem"   # Streamlit's own built-in header height (60px)

# Chart theme 

CHART_THEME = {
    "font":          {"family": FONT_FAMILY, "size": FONT_SIZE, "color": "#374151"},
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor":  "rgba(0,0,0,0)",
    "margin": {"t": 50, "b": 40, "l": 50, "r": 20},
    "legend": {"bgcolor": "rgba(0,0,0,0)", "borderwidth": 0},
    "xaxis": {"gridcolor": "#F3F4F6", "linecolor": "#E5E7EB", "zeroline": False},
    "yaxis": {"gridcolor": "#F3F4F6", "linecolor": "#E5E7EB", "zeroline": False},
}

# CSS injected into every page 
# Loaded on every page via:
#     st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
# It's an f-string so the Python variables above are inserted automatically.
# Only the CSS rules change here — SIGNATURE_HTML and HEADER_HTML (below)
# render the actual HTML elements.

_corner_css = "; ".join(f"{side}: {value}" for side, value in SIGNATURE_CORNER.items())

CUSTOM_CSS = f"""
<style>

/* ── Fonts ── */
@import url('{GOOGLE_FONT_IMPORT}');
/* Note: no second @import needed — signature now uses Inter like the rest */

/* ── Page background ── */
[data-testid="stAppViewContainer"] {{
    background-color: #FAFAFA;
}}

/* ── Apply FONT_FAMILY to all Streamlit text ── */
html, body, [class*="css"] {{
    font-family: {FONT_FAMILY};
}}

/* ── General page ── */
/* padding-top leaves room for Streamlit's native header (3.75rem)
   plus our fixed custom header bar (~2.85rem) stacked below it. */
.main .block-container {{
    padding-top: 7.5rem;
    padding-bottom: 2rem;
}}

/* ── Persistent top header bar ── */
/* Positioned just below Streamlit's own native header bar (which is
   fixed, opaque white, and sits at z-index: 999990 — a plain z-index
   bump here would still get painted over by it). NATIVE_HEADER_HEIGHT
   below is Streamlit's own header height; keep the two in sync if a
   future Streamlit version changes it. */
.pm-header-bar {{
    position: fixed;
    top: {NATIVE_HEADER_HEIGHT};
    left: 0;
    right: 0;
    z-index: 999991;
    background: {HEADER_BG_COLOR};
    padding: 10px 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 1px 6px rgba(0,0,0,0.25);
}}
.pm-header-title {{
    font-family: {FONT_FAMILY};
    font-weight: 700;
    font-size: 1.0rem;
    letter-spacing: 0.01em;
    color: {HEADER_TITLE_COLOR};
}}
.pm-header-subtitle {{
    font-family: {FONT_FAMILY};
    font-weight: 400;
    font-size: 0.78rem;
    letter-spacing: 0.03em;
    color: {HEADER_SUBTITLE_COLOR};
}}

/* ── Signature — fixed corner ── */
/* Uses Inter bold (same as dashboard body font) instead of cursive.
   font-weight: 800 (extra-bold) makes it read as a mark without
   being decorative. pointer-events: none means it never blocks clicks. */
.dashboard-signature {{
    position: fixed;
    {_corner_css};
    font-family: {SIGNATURE_FONT};
    font-weight: {SIGNATURE_WEIGHT};
    font-size: {SIGNATURE_SIZE};
    color: {SIGNATURE_COLOR};
    letter-spacing: 0.15em;
    opacity: 0.85;
    z-index: 9999;
    pointer-events: none;
    user-select: none;
}}

/* ── KPI metric cards ── */
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

# HTML snippets — imported and rendered by each page
# Each page should inject both of these alongside CUSTOM_CSS.
# Typical usage at the top of every page file:
#
#     from utils.styles import CUSTOM_CSS, SIGNATURE_HTML, HEADER_HTML
#     st.markdown(CUSTOM_CSS + HEADER_HTML + SIGNATURE_HTML, unsafe_allow_html=True)
#
# HEADER_HTML — the persistent top title bar
# Edit the text between the span tags to rename the project or subtitle.

HEADER_HTML = f"""
<div class="pm-header-bar">
    <span class="pm-header-title">Turbofan Engine Health & Predictive Maintenance</span>
    <span class="pm-header-subtitle">NASA C-MAPSS Dataset: RUL Prediction & Health Analytics</span>
</div>
"""

# SIGNATURE_HTML — the initials mark in the page corner
# Edit SIGNATURE_TEXT above to change the initials.

SIGNATURE_HTML = f'<div class="dashboard-signature">{SIGNATURE_TEXT}</div>'