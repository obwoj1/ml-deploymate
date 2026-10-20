import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # lets the page find the `deploymate` code
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so the page can find style.py
from style import inject
inject()

import pandas as pd
import streamlit as st

from deploymate import service

st.set_page_config(page_title="Overview · ML DeployMate", layout="wide")
st.title("Overview")
st.caption("Is the deployed model healthy right now?")

with st.spinner("Loading (first load takes a few seconds)..."):
    alerts = service.list_alerts()  # first: a fresh server seeds demo alerts here
    overview = service.get_overview()
    report = service.get_model_report()

if overview["mock"]:
    st.info("Showing demo data. The real backend isn't connected.")

# --- Health banner ------------------------------------------------------------
if overview["status"] == "healthy":
    st.success("✅ **Healthy.** Incoming data looks like the data the model was trained on.")
else:
    newest = alerts[0] if alerts else None
    detail = f" Most recent: batch {newest['batch_id']}, {', '.join(newest['features'])} changed." if newest else ""
    st.error(f"🚨 **Drift detected.** {overview['active_alerts']} alert(s) so far.{detail}")

# --- Headline numbers ------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Status", overview["status"].title())
c2.metric("Test accuracy", f"{overview['test_accuracy']:.1%}")
c3.metric("Batches checked", overview["batches_processed"])
c4.metric("Active alerts", overview["active_alerts"])
st.caption(f"Deployment **{overview['deployment']}** · model **{overview['model']}** · "
           f"last updated {overview['last_updated']}")

st.divider()

# --- Recent alerts + model at a glance ------------------------------------------
left, right = st.columns([3, 2])
with left:
    st.subheader("Recent alerts")
    if alerts:
        recent = pd.DataFrame([{
            "Batch": a["batch_id"],
            "Severity": a["severity"],
            "What changed": ", ".join(a["features"]),
            "Scenario": a["scenario"].replace("_", " "),
            "Time": a["time"],
        } for a in alerts[:5]])
        st.dataframe(recent, hide_index=True, width="stretch")
        st.page_link("pages/3_Alerts.py", label="See all alerts and AI explanations →")
    else:
        st.info("No alerts yet.")
        st.page_link("pages/2_Live_Monitor.py", label="Run a scenario on Live Monitor →")

with right:
    st.subheader("The model")
    test = report["metrics"]["test"]
    m1, m2 = st.columns(2)
    m1.metric("F1 score", f"{test['f1']:.3f}")
    m2.metric("ROC-AUC", f"{test['roc_auc']:.3f}")
    st.caption(f"{report['model']} predicting whether a person earns more than $50K "
               "(UCI Adult census data).")
    st.page_link("pages/4_Model_Performance.py", label="Full model performance →")

st.divider()

# --- How it works ----------------------------------------------------------------
st.subheader("How DeployMate keeps watch")
s1, s2, s3 = st.columns(3)
s1.markdown("**1. Compare**  \nEvery incoming batch is compared with the training data, column by column.")
s2.markdown("**2. Alert**  \nIf a column shifts too much (PSI or Cramér's V over the threshold), an alert fires.")
s3.markdown("**3. Explain**  \nAI turns the alert into plain English: what happened, why, and what to do.")
