import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # lets the page find the `deploymate` code
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so the page can find style.py
from style import inject
inject()

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from deploymate import service

st.set_page_config(page_title="Model Performance · ML DeployMate", layout="wide")
st.title("Model Performance")

with st.spinner("Loading the model report..."):
    report = service.get_model_report()

st.caption(f"The model DeployMate is monitoring: **{report['model']}**, predicting whether a person earns "
           "more than $50K from 1994 U.S. Census data (UCI Adult).")

# --- Headline scores ------------------------------------------------------------
test, train = report["metrics"]["test"], report["metrics"]["train"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Accuracy", f"{test['accuracy']:.1%}")
c2.metric("F1 score", f"{test['f1']:.3f}")
c3.metric("ROC-AUC", f"{test['roc_auc']:.3f}")
c4.metric("Recall", f"{test['recall']:.1%}")
baseline = report.get("baseline_always_negative_accuracy")
if baseline:
    st.caption(f"For comparison, always guessing \"≤$50K\" would score {baseline:.1%} accuracy, "
               "which is why F1 and ROC-AUC matter here.")

# --- Confusion matrix + ROC -----------------------------------------------------
left, right = st.columns(2)
cm = report["confusion_matrix"]
heat = px.imshow(cm["matrix"], x=[f"Predicted {l}" for l in cm["labels"]],
                 y=[f"Actually {l}" for l in cm["labels"]], text_auto=",",
                 color_continuous_scale=[[0, "#1C2F52"], [1, "#B4FF39"]], title="Confusion matrix (test set)")
heat.update_layout(height=380, coloraxis_showscale=False, margin=dict(t=50, b=10))
left.plotly_chart(heat, width="stretch")
left.caption("Diagonal = correct predictions. The model misses some high earners, "
             "which is normal when only 24% of people earn >$50K.")

roc = report["roc_curve"]
curve = go.Figure([
    go.Scatter(x=roc["fpr"], y=roc["tpr"], mode="lines", name=f"Model (AUC {test['roc_auc']:.3f})",
               line=dict(color="#B4FF39", width=3)),
    go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random guess", line=dict(dash="dash", color="#A8B5CE")),
])
curve.update_layout(title="ROC curve (test set)", xaxis_title="False positive rate",
                    yaxis_title="True positive rate", height=380, margin=dict(t=50, b=10),
                    legend=dict(x=0.45, y=0.1))
right.plotly_chart(curve, width="stretch")
right.caption("The further the curve bows toward the top-left, the better the model separates the two groups.")

# --- Feature importance -----------------------------------------------------------
importance = pd.DataFrame(report["feature_importance"]).sort_values("importance")
bars = px.bar(importance, x="importance", y="feature", orientation="h",
              color_discrete_sequence=["#B4FF39"], title="Which features the model relies on (permutation importance)")
bars.update_layout(height=420, margin=dict(t=50, b=10))
st.plotly_chart(bars, width="stretch")
st.caption("How much the test ROC-AUC drops when that feature is shuffled. Bigger = more important.")

# --- Model comparison + train vs test ---------------------------------------------
left, right = st.columns(2)
if report.get("comparison"):
    comparison = pd.DataFrame(report["comparison"]).rename(columns={
        "model": "Model", "cv_accuracy": "CV accuracy", "cv_f1": "CV F1", "cv_roc_auc": "CV ROC-AUC"})
    left.subheader("Models we compared")
    left.dataframe(comparison[["Model", "CV accuracy", "CV F1", "CV ROC-AUC"]], hide_index=True, width="stretch")
    left.caption("5-fold cross-validation on the training data. The winner is picked by ROC-AUC.")

right.subheader("Train vs. test")
split = pd.DataFrame({"Train": train, "Test": test}).rename(index=str.capitalize).round(3)
right.dataframe(split, width="stretch")
right.caption("Train and test scores are close, so the model isn't overfitting.")
