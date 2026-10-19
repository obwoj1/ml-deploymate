
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # lets the page find the `deploymate` code
 
import time
 
import pandas as pd
import streamlit as st
from deploymate import service
 
def as_dict(x):
    return x if isinstance(x, dict) else vars(x)
 
 
def pick(d, *keys, default="—"):
    """Return the first key that exists in d (the mock service may name things slightly differently)."""
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default
 
 
st.title("Live Monitor")
st.caption("Pick a scenario and watch DeployMate react.")
 
c1, c2, c3 = st.columns([2, 2, 1])
scenario = c1.selectbox("Scenario", service.list_scenarios())
n_batches = c2.slider("Batches", 10, 100, 40)
start = c3.button("▶ Start", type="primary")
 
if start:
    banner = st.empty()
    left, right = st.columns(2)
    left.subheader("Accuracy per batch")
    right.subheader("Drift (PSI) per feature")
    acc_chart, psi_chart = left.empty(), right.empty()
    acc_rows, psi_rows, first = [], [], None
 
    for i, batch in enumerate(service.run_scenario(scenario, n_batches), start=1):
        b = as_dict(batch)
        first = first or b
        metrics = as_dict(pick(b, "metrics", default={}))
        drift = as_dict(pick(b, "drift", "feature_drift", "psi", default={}))
 
        acc_rows.append({"batch": i, "accuracy": pick(metrics, "accuracy", default=None)})
        psi_rows.append({"batch": i, **{f: (pick(v, "psi", default=None) if isinstance(v, dict) else v)
                                       for f, v in drift.items()}})
 
        acc_chart.line_chart(pd.DataFrame(acc_rows).set_index("batch"))
        psi_df = pd.DataFrame(psi_rows).set_index("batch")
        if psi_df.shape[1] > 0:
            psi_chart.line_chart(psi_df)
 
        if pick(b, "alert", "is_alert", "alert_fired", default=False):
            banner.error(f"ALERT: drift detected at batch {i}")
        time.sleep(0.15)
 
    with st.expander("Raw first batch (for debugging, remove later)"):
        st.write(first)