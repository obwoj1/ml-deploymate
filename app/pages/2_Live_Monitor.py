import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # lets the page find the `deploymate` code

import pandas as pd
import plotly.express as px
import streamlit as st

from deploymate import service

st.set_page_config(page_title="Live Monitor · ML DeployMate", layout="wide")
st.title("Live Monitor")
st.caption("Replay live traffic through the deployed model and watch DeployMate catch problems as they happen.")

with st.spinner("Starting the monitor (first load takes a few seconds)..."):
    scenarios = service.list_scenarios()
    if service.get_overview()["mock"]:
        st.info("Showing demo data. The real backend isn't connected.")

# --- Controls ---------------------------------------------------------------
names = [s["name"] for s in scenarios]
descriptions = {s["name"]: s["description"] for s in scenarios}
left, mid, right = st.columns([2, 1, 1])
choice = left.selectbox("Scenario", names, index=names.index("sudden_shift") if "sudden_shift" in names else 0,
                        format_func=lambda n: n.replace("_", " ").title())
n_batches = mid.slider("Batches", 10, 40, 30, step=5)
delay = right.slider("Speed (seconds per batch)", 0.0, 0.5, 0.15, step=0.05)
st.caption(f"**What this scenario does:** {descriptions[choice]}")

start = st.button("▶ Start", type="primary")

# --- Live area ----------------------------------------------------------------
banner = st.empty()
kpis = st.empty()
chart_left, chart_right = st.columns(2)
accuracy_chart = chart_left.empty()
drift_chart = chart_right.empty()
log_box = st.empty()


def drift_score(batch: dict) -> float:
    """The largest drift statistic in the batch (PSI or Cramér's V)."""
    return max(d["statistic"] for d in batch["drift"])


def draw(rows: list[dict], alerts: list[dict], done: bool) -> None:
    df = pd.DataFrame(rows)
    latest = rows[-1]
    with kpis.container():
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Batch", f"{latest['batch']} / {n_batches}")
        c2.metric("Accuracy", f"{latest['accuracy']:.1%}")
        c3.metric("Model confidence", f"{latest['confidence']:.1%}")
        c4.metric("Alerts", len(alerts))

    acc = px.line(df, x="batch", y="accuracy", markers=True, range_y=[0.7, 1.0],
                  title="Model accuracy per batch")
    acc.update_layout(height=320, margin=dict(t=40, b=10))
    accuracy_chart.plotly_chart(acc, width="stretch", key=f"acc-{len(rows)}-{done}")

    drift = px.bar(df, x="batch", y="drift_score", color="alert",
                   color_discrete_map={True: "#E5484D", False: "#4CC38A"},
                   title="Biggest drift score per batch (red = alert)")
    drift.update_layout(height=320, margin=dict(t=40, b=10), showlegend=False)
    drift_chart.plotly_chart(drift, width="stretch", key=f"drift-{len(rows)}-{done}")

    if alerts:
        first = alerts[0]
        msg = (f"🚨 Drift detected at batch {first['batch']}: "
               f"{', '.join(first['features'])} changed (severity: {first['severity']}).")
        banner.error(msg + (" See the **Alerts** page for the AI explanation." if done else ""))
    elif done:
        banner.success("✅ No drift detected. The incoming data looks like the training data.")

    if alerts:
        log = pd.DataFrame([{"batch": a["batch"], "severity": a["severity"],
                             "changed": ", ".join(a["features"])} for a in alerts])
        log_box.dataframe(log, hide_index=True, width="stretch", key=f"log-{len(rows)}-{done}")


if start:
    rows, alerts = [], []
    for batch in service.run_scenario(choice, n_batches=n_batches):
        drifted = [d["feature"] for d in batch["drift"] if d["drifted"]]
        rows.append({"batch": batch["batch_id"], "accuracy": batch["metrics"]["accuracy"],
                     "confidence": batch["metrics"]["mean_confidence"],
                     "drift_score": drift_score(batch), "alert": batch["alert"]})
        if batch["alert"]:
            alerts.append({"batch": batch["batch_id"], "severity": batch["severity"], "features": drifted})
        draw(rows, alerts, done=False)
        time.sleep(delay)
    draw(rows, alerts, done=True)
    st.session_state["last_run"] = {"scenario": choice, "rows": rows, "alerts": alerts}
elif "last_run" in st.session_state and st.session_state["last_run"]["scenario"] == choice:
    run = st.session_state["last_run"]
    draw(run["rows"], run["alerts"], done=True)
else:
    banner.info("Pick a scenario and press **Start**. Try `healthy` first, then `sudden_shift`.")
