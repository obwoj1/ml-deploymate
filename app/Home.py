import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # lets the app find the `deploymate` code

import streamlit as st
from deploymate import service

st.set_page_config(page_title="ML DeployMate", layout="wide")
st.title("ML DeployMate")
st.write("Watches a deployed ML model and catches problems before users notice.")

st.subheader("How to use this dashboard")
st.write(
    "Use the sidebar on the left to move between pages. "
    "**Overview** shows the model's overall health, **Live Monitor** tracks incoming data for drift, "
    "**Model Performance** shows accuracy and other metrics, **Alerts** lists problems the monitor "
    "has caught with plain-English explanations, and **Config Generator** creates deployment configs."
)
