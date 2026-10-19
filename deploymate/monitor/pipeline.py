"""The full monitor: one call per incoming batch.

Alerts come from the DRIFT TESTS only. The anomaly detector is reported as an
info signal: evaluation showed that letting it trigger alerts added ~0.6%
false alarms for +0.1% recall (see README results + PROGRESS.md, 2026-10-15).

    monitor = Monitor.from_data(model, X_reference, healthy_batches)
    report = monitor.check(batch_X, batch_y)   # y optional

The report has the same shape deploymate.service promises the dashboard:
metrics, per-feature drift, anomaly result, alert flag and severity.
"""

from __future__ import annotations

from dataclasses import asdict

import pandas as pd

from deploymate.monitor.anomaly import AnomalyDetector
from deploymate.monitor.drift import detect_drift
from deploymate.monitor.metrics import batch_metrics

# A drift statistic this far past its threshold counts as severe.
SEVERE_PSI = 0.5
SEVERE_CRAMERS_V = 0.25


def describe_shift(reference: pd.Series, current: pd.Series, kind: str) -> dict:
    """Human-readable summary of HOW a drifted feature changed (used in explanations)."""
    if kind == "numeric":
        return {"reference_mean": round(float(reference.mean()), 2),
                "current_mean": round(float(current.mean()), 2)}
    ref_share = reference.value_counts(normalize=True)
    cur_share = current.value_counts(normalize=True)
    change = cur_share.subtract(ref_share, fill_value=0)
    top = change.abs().idxmax()
    return {"category": str(top),
            "reference_share": round(float(ref_share.get(top, 0.0)), 3),
            "current_share": round(float(cur_share.get(top, 0.0)), 3)}


class Monitor:
    def __init__(self, model, reference: pd.DataFrame, detector: AnomalyDetector):
        self.model = model
        self.reference = reference
        self.detector = detector

    @classmethod
    def from_data(cls, model, reference: pd.DataFrame, healthy_batches: list[pd.DataFrame]):
        """Calibrate the anomaly detector on batches we know are healthy (~200 works well)."""
        healthy = [batch_metrics(model, X) for X in healthy_batches]
        return cls(model, reference, AnomalyDetector().fit(healthy))

    def check(self, X: pd.DataFrame, y: pd.Series | None = None, batch_id: int | None = None) -> dict:
        metrics = batch_metrics(self.model, X, y)
        drift = [asdict(d) for d in detect_drift(self.reference, X)]
        anomaly = self.detector.check(metrics)

        drifted = [d for d in drift if d["drifted"]]
        for d in drifted:
            d["shift"] = describe_shift(self.reference[d["feature"]], X[d["feature"]], d["kind"])
        alert = bool(drifted)
        severe = any(
            d["statistic"] >= (SEVERE_PSI if d["kind"] == "numeric" else SEVERE_CRAMERS_V)
            for d in drifted)
        if not alert:
            severity = "none"
        elif severe or (drifted and anomaly["anomalous"]) or len(drifted) >= 2:
            severity = "high"
        else:
            severity = "medium"

        return {
            "batch_id": batch_id,
            "metrics": metrics,
            "drift": drift,
            "drifted_features": [d["feature"] for d in drifted],
            "anomaly": anomaly,  # info signal: shown on the dashboard, doesn't trigger alerts
            "alert": alert,
            "severity": severity,
        }
