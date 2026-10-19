import sys
from pathlib import Path

import streamlit as st

# app directs to deplymate package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deploymate import service 

st.set_page_config(page_title="ML DeployMate", page_icon="🛰️", layout="wide")

st.title("ML DeployMate")
st.caption("Watch a deployed ML model, catch drift early, and get plain-English explanations.")

overview = service.get_overview()

if overview["mock"]:
    st.info("Showing demo data. The real backend isn't connected yet.")

col1, col2, col3 = st.columns(3)
col1.metric("Deployment", overview["deployment"])
col2.metric("Status", overview["status"].title())
col3.metric("Active alerts", overview["active_alerts"])

st.divider()
st.subheader("Explore the dashboard")
st.page_link("pages/1_Overview.py", label="Overview")
st.page_link("pages/2_Live_Monitor.py", label="Live Monitor")
st.page_link("pages/3_Alerts.py", label="Alerts and AI explanations")
st.page_link("pages/4_Model_Performance.py", label="Model Performance")
st.page_link("pages/5_Config_Generator.py", label="Config Generator")