import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # lets the page find the `deploymate` code
 
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
 
 
st.title("Overview")
st.caption("Is the model healthy right now?")
 
data = as_dict(service.get_overview())
cols = st.columns(4)
cols[0].metric("Status", pick(data, "status"))
cols[1].metric("Model", pick(data, "model", "model_name"))
cols[2].metric("Batches", pick(data, "n_batches", "batches", "batches_seen"))
cols[3].metric("Active alerts", pick(data, "active_alerts", "n_active_alerts", "alerts"))
 
st.subheader("Recent alerts")
alerts = [as_dict(a) for a in service.list_alerts()]
if alerts:
    st.dataframe(pd.DataFrame(alerts).head(5), hide_index=True)
else:
    st.info("No alerts yet.")
 
with st.expander("Raw data (for debugging, remove later)"):
    st.write(data)