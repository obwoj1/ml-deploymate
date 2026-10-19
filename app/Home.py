import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # lets the app find the `deploymate` code

import streamlit as st
from deploymate import service  # noqa: F401  (the pages use this; Home will show status later)

st.set_page_config(page_title="ML DeployMate", layout="wide")

# --- Hero banner (matches the wireframe) ---
st.markdown(
    """
    <style>
    .hero{background:#131826;border:1px solid #232B3F;border-radius:14px;padding:40px;margin-bottom:8px}
    .hero .eyebrow{color:#B6FF3C;font-family:monospace;letter-spacing:1px;text-transform:uppercase;font-size:.8rem}
    .hero h1{font-size:3.6rem;margin:.3rem 0;padding:0}
    .hero p{color:#8A94AD;font-size:1.2rem;max-width:560px;margin:0}
    </style>
    <div class="hero">
      <div class="eyebrow">Model monitoring</div>
      <h1>ML DeployMate</h1>
      <p>Watches a deployed ML model and catches problems before users notice.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

overview = service.get_overview()
if overview["mock"]:
    st.info("Showing demo data. The real backend isn't connected yet.")

col1, col2, col3 = st.columns(3)
col1.metric("Deployment", overview["deployment"])
col2.metric("Status", overview["status"].title())
col3.metric("Active alerts", overview["active_alerts"])

# --- How to use: one card per page ---
st.subheader("How to use this dashboard")
st.write("Use the sidebar, or the cards below, to move between pages.")

PAGES = [
    ("01", "Overview", "Model health at a glance", "pages/1_Overview.py"),
    ("02", "Live Monitor", "Watch batches and drift live", "pages/2_Live_Monitor.py"),
    ("03", "Model Performance", "Accuracy, ROC, key features", "pages/3_Model_Performance.py"),
    ("04", "Alerts", "Problems, explained in plain English", "pages/4_Alerts.py"),
    ("05", "Config Generator", "Deployment files from a form", "pages/5_Config_Generator.py"),
]
for col, (num, name, blurb, path) in zip(st.columns(len(PAGES)), PAGES):
    with col, st.container(border=True):
        st.caption(num)
        st.markdown(f"**{name}**")
        st.write(blurb)
        st.page_link(path, label="Open →")

st.caption("Team SavageAI")
