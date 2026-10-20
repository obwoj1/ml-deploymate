import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))          # lets the app find the `deploymate` code
sys.path.insert(0, str(ROOT / "app"))  # lets the app find style.py

import streamlit as st
from deploymate import service
from style import inject, status_pill, stat_card, status_color

st.set_page_config(page_title="ML DeployMate", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")
inject()


def as_dict(x):
    return x if isinstance(x, dict) else vars(x)


def pick(d, *keys, default="—"):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


overview = as_dict(service.get_overview())
status = pick(overview, "status")

# --- Hero banner ---
st.markdown(
    f"""
    <div class="hero">
      <div class="eyebrow">Model monitoring {status_pill(status)}</div>
      <h1>ML DeployMate</h1>
      <p>Watches a deployed ML model and catches problems before users notice.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Live status strip ---
c1, c2, c3 = st.columns(3)
c1.markdown(stat_card("Deployment", pick(overview, "deployment", "deployment_name", "name")), unsafe_allow_html=True)
c2.markdown(stat_card("Status", str(status).title(), status_color(status)), unsafe_allow_html=True)
c3.markdown(stat_card("Active alerts", pick(overview, "active_alerts", "n_active_alerts", "alerts"), "#FF6B7A"),
            unsafe_allow_html=True)

# --- How to use: one card per page ---
st.subheader("How to use this dashboard")
st.write("Use the sidebar, or the cards below, to move between pages.")

PAGES = [
    ("01", "📊", "Overview", "Model health at a glance", "pages/1_Overview.py"),
    ("02", "📡", "Live Monitor", "Watch batches and drift live", "pages/2_Live_Monitor.py"),
    ("03", "⚠️", "Alerts", "Problems, explained in plain English", "pages/3_Alerts.py"),
    ("04", "📈", "Model Performance", "Accuracy, ROC, key features", "pages/4_Model_Performance.py"),
    ("05", "⚙️", "Config Generator", "Deployment files from a form", "pages/5_Config_Generator.py"),
]
for col, (num, icon, name, blurb, path) in zip(st.columns(len(PAGES)), PAGES):
    with col, st.container(border=True):
        st.markdown(f'<div class="num">{num}</div><div class="ico">{icon}</div>', unsafe_allow_html=True)
        st.markdown(f"**{name}**")
        st.write(blurb)
        st.page_link(path, label="Open →")

st.caption("Team SavageAI")