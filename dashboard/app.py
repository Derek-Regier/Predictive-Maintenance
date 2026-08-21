"""
dashboard/app.py

Entry point for the Streamlit dashboard.

Run from the project root:
    streamlit run dashboard/app.py

What this file does
-------------------
1. Sets page config (must be the FIRST Streamlit call in the app)
2. Injects the custom CSS from styles.py into every page
3. Redirects to the Fleet Overview landing page

The page config here applies to all pages in the dashboard because
app.py is the entry point Streamlit loads first.

Multipage structure
-------------------
Every .py file in dashboard/pages/ automatically becomes a page.
The number prefix sets the sidebar order; underscores become spaces.
Streamlit builds the navigation automatically — no routing code needed.

    pages/
      1_Fleet_Overview.py      → "Fleet Overview"      (first in sidebar)
      2_Engine_Deep_Dive.py   → "Engine Deep Dive"
      3_Model_Performance.py  → "Model Performance"
      4_Health_Monitor.py     → "Health Monitor"
"""

import sys
from pathlib import Path

import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
# Add dashboard/ to sys.path so all page files can do:
#     from utils.data_loader import ...
#     from utils.charts import ...
#     from utils.styles import ...
# without needing the full path every time.

_DASHBOARD = Path(__file__).resolve().parent
if str(_DASHBOARD) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD))

from utils.styles import CUSTOM_CSS, HEADER_HTML, SIGNATURE_HTML

# ── Page configuration ────────────────────────────────────────────────────────
# Must be called before any other st. function.
# layout="wide" is critical — without it charts are constrained to a narrow
# centre column. page_icon can be any emoji or a path to an image file.

st.set_page_config(
    page_title          = "Predictive Maintenance",
    page_icon           = "",
    layout              = "wide",
    initial_sidebar_state = "expanded",
    menu_items={
        "About": (
            "Predictive Maintenance Dashboard — NASA CMAPSS turbofan dataset.\n"
            "Models: GRU / TCN / Transformer backbone + NGBoost meta-model.\n"
            "Health monitoring: VAE with information geometry."
        )
    }
)

# ── Inject CSS ────────────────────────────────────────────────────────────────
# unsafe_allow_html=True is required to inject a <style> block.
# The CSS lives in utils/styles.py so the non-tech collaborator can
# edit it without touching this file.

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.markdown(HEADER_HTML, unsafe_allow_html=True)
st.markdown(SIGNATURE_HTML, unsafe_allow_html=True)

# ── Redirect to landing page ──────────────────────────────────────────────────
# When users open the app they see app.py by default. We immediately
# redirect them to the Fleet Overview page which is the real landing page.

st.switch_page("pages/1_Fleet_Overview.py")