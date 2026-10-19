import numpy as np
import pandas as pd
import pytest

from deploymate.data.load import CATEGORICAL, FEATURES, NUMERIC
from deploymate.model.pipeline import build_model
from deploymate.monitor.anomaly import AnomalyDetector
from deploymate.monitor.metrics import LABEL_FREE, batch_metrics
from deploymate.monitor.pipeline import Monitor
from deploymate.simulate.scenarios import simulate

rng = np.random.default_rng(5)


def make_data(n):
    X = pd.DataFrame({col: rng.normal(40, 10, n) for col in NUMERIC})
    for col in CATEGORICAL:
        X[col] = rng.choice(["a", "b", "c"], n)
    y = ((X["age"] > 45) | (X["workclass"] == "a")).astype(int)
    return X[FEATURES], y


@pytest.fixture(scope="module")
def setup():
    X, y = make_data(6000)
    model = build_model().fit(X[:3000], y[:3000])
    X_live, y_live = X[3000:].reset_index(drop=True), y[3000:].reset_index(drop=True)
    healthy = [b.X for b in simulate("healthy", X_live, y_live, n_batches=100, batch_size=200, seed=9)]
    monitor = Monitor.from_data(model, X[:3000], healthy)
    return model, monitor, X_live, y_live


def test_batch_metrics_without_and_with_labels(setup):
    model, _, X, y = setup
    no_labels = batch_metrics(model, X[:200])
    assert set(no_labels) == set(LABEL_FREE)
    with_labels = batch_metrics(model, X[:200], y[:200])
    assert {"accuracy", "f1"} <= set(with_labels)
    assert 0.5 <= with_labels["mean_confidence"] <= 1


def test_unknown_rate_counts_missing_fields(setup):
    model, _, X, _ = setup
    batch = X[:200].copy()
    batch["workclass"] = "Unknown"  # 1 of 7 categorical columns fully missing
    assert batch_metrics(model, batch)["unknown_rate"] == pytest.approx(1 / 7, abs=0.01)


def test_anomaly_detector_flags_outlier_but_not_normal():
    healthy = [{m: v + rng.normal(0, 0.01) for m, v in zip(LABEL_FREE, [0.24, 0.85, 0.15, 0.01])}
               for _ in range(200)]
    detector = AnomalyDetector().fit(healthy)
    assert not detector.check(healthy[0])["anomalous"]
    weird = dict(healthy[0], mean_confidence=0.6)
    result = detector.check(weird)
    assert result["anomalous"] and "mean_confidence" in result["flagged_metrics"]


def test_healthy_batches_raise_no_alerts(setup):
    _, monitor, X, y = setup
    reports = [monitor.check(b.X) for b in simulate("healthy", X, y, n_batches=20, batch_size=200, seed=1)]
    assert sum(r["alert"] for r in reports) <= 1


def test_broken_batches_alert_with_severity(setup):
    _, monitor, X, y = setup
    batches = list(simulate("missing_values", X, y, n_batches=20, batch_size=200, seed=2))
    report = monitor.check(batches[-1].X, batches[-1].y, batch_id=20)
    assert report["alert"] and report["severity"] in {"medium", "high"}
    assert report["batch_id"] == 20
    assert {"metrics", "drift", "drifted_features", "anomaly", "alert", "severity"} <= set(report)
    shifted = next(d for d in report["drift"] if d["feature"] == "workclass")
    assert shifted["shift"]["category"] == "Unknown"
    assert shifted["shift"]["current_share"] > shifted["shift"]["reference_share"]


def test_anomaly_alone_does_not_trigger_alert(setup, monkeypatch):
    _, monitor, X, _ = setup
    real_check = monitor.detector.check
    monkeypatch.setattr(monitor.detector, "check", lambda row: dict(real_check(row), anomalous=True))
    report = monitor.check(X[:200])  # healthy batch: no drift, but forced anomaly
    assert report["anomaly"]["anomalous"]
    assert not report["drifted_features"]
    assert report["alert"] is False and report["severity"] == "none"
