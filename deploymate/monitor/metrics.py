"""Per-batch metrics: how is the model behaving on this batch?

The LABEL-FREE metrics are what the anomaly detector watches. In production the
true answers (did this person really earn >50K?) arrive weeks later, if ever,
so a monitor can't rely on accuracy. It has to notice trouble from the inputs
and the model's own behavior. Accuracy/F1 are added only when labels are
available (as in our simulation) so we can show they move too.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from deploymate.data.load import CATEGORICAL

# Metrics the anomaly detector uses: available the moment a batch arrives.
LABEL_FREE = ["positive_rate", "mean_confidence", "low_confidence_share", "unknown_rate"]
LOW_CONFIDENCE = 0.6  # predictions where the model is less than 60% sure


def batch_metrics(model, X: pd.DataFrame, y: pd.Series | None = None) -> dict:
    probabilities = model.predict_proba(X)[:, 1]
    confidence = np.maximum(probabilities, 1 - probabilities)  # certainty in its chosen answer
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "positive_rate": float(predictions.mean()),          # share predicted >50K
        "mean_confidence": float(confidence.mean()),
        "low_confidence_share": float((confidence < LOW_CONFIDENCE).mean()),
        "unknown_rate": float((X[CATEGORICAL] == "Unknown").to_numpy().mean()),
    }
    if y is not None:
        metrics["accuracy"] = float(accuracy_score(y, predictions))
        metrics["f1"] = float(f1_score(y, predictions, zero_division=0))
    return {k: round(v, 4) for k, v in metrics.items()}
