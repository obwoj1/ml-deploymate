import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # lets the page find the `deploymate` code
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so the page can find style.py
from style import inject
inject()

import streamlit as st

from deploymate import service

st.set_page_config(page_title="Alerts · ML DeployMate", layout="wide")
st.title("Alerts")
st.caption("Every time the incoming data stops looking like the training data, DeployMate raises an alert. "
           "Click one to get a plain-English explanation.")

with st.spinner("Loading alerts..."):
    alerts = service.list_alerts()
    if service.get_overview()["mock"]:
        st.info("Showing demo data. The real backend isn't connected.")

if not alerts:
    st.info("No alerts yet. Run a scenario on the **Live Monitor** page.")
    st.stop()

if st.button("Clear all alerts", help="Start fresh: removes every alert and resets the batch counter."):
    service.clear_alerts()
    st.session_state.pop("explanations", None)
    st.rerun()

# Explanations are cached per alert for this browser session, so the AI is called once per alert.
explanations = st.session_state.setdefault("explanations", {})

# --- Summary + filter ---------------------------------------------------------
high = sum(a["severity"] == "high" for a in alerts)
c1, c2, c3 = st.columns(3)
c1.metric("Total alerts", len(alerts))
c2.metric("High severity", high)
c3.metric("Medium severity", len(alerts) - high)

severity = st.segmented_control("Severity", ["all", "high", "medium"], default="all")
shown = [a for a in alerts if severity in (None, "all") or a["severity"] == severity]
st.caption(f"Showing {len(shown)} of {len(alerts)} alerts, newest first.")

# --- Alert list ---------------------------------------------------------------
for alert in shown:
    icon = "🔴" if alert["severity"] == "high" else "🟠"
    title = (f"{icon} Batch {alert['batch_id']} · {', '.join(alert['features'])} · "
             f"{alert['severity']} · {alert['scenario'].replace('_', ' ')}")
    with st.expander(title):
        d1, d2, d3 = st.columns(3)
        d1.write(f"**Time:** {alert['time']}")
        accuracy = alert.get("accuracy")
        d2.write(f"**Batch accuracy:** {accuracy:.1%}" if accuracy is not None else "**Batch accuracy:** n/a")
        d3.write(f"**Changed:** {', '.join(alert['features'])}")

        if st.button("✨ Explain with AI", key=f"explain-{alert['id']}"):
            with st.spinner("Asking the AI..."):
                explanations[alert["id"]] = service.explain_alert(alert["id"])

        info = explanations.get(alert["id"])
        if info:
            st.markdown(f"**What happened**  \n{info['summary']}")
            st.markdown(f"**Likely cause**  \n{info['likely_cause']}")
            st.markdown(f"**Suggested action**  \n{info['suggested_action']}")
            source = info.get("source", "")
            st.caption("Explained by Claude" if source == "claude"
                       else "Explained by OpenAI" if source == "openai"
                       else "Template explanation (no AI key available)")
