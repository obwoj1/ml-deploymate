import numpy as np
import pandas as pd

from deploymate.data.load import CATEGORICAL, FEATURES, NUMERIC
from deploymate.monitor.evaluate import DETECTORS, evaluate

rng = np.random.default_rng(3)
X = pd.DataFrame({col: rng.normal(40, 10, 2000) for col in NUMERIC})
for col in CATEGORICAL:
    X[col] = rng.choice(["a", "b"], 2000)
X = X[FEATURES]
y = pd.Series(rng.integers(0, 2, 2000))


class StubMonitor:
    """Always (or never) alerts, so the expected scores are known exactly."""
    def __init__(self, alert):
        self.alert = alert

    def check(self, X):
        return {"drifted_features": ["age"] if self.alert else [],
                "anomaly": {"anomalous": self.alert}, "alert": self.alert}


def run(alert, scenarios):
    return evaluate(StubMonitor(alert), X, y, seeds=range(2), n_batches=10,
                    batch_size=50, scenarios=scenarios)


def test_always_alert_catches_everything_but_false_alarms_on_healthy():
    res = run(True, ["sudden_shift"])["combined"]["sudden_shift"]
    # sudden_shift: 5 healthy then 5 broken batches, x2 runs
    assert (res["tp"], res["fp"], res["fn"], res["tn"]) == (10, 10, 0, 0)
    assert res["recall"] == 1.0 and res["precision"] == 0.5
    assert res["false_alarm_rate"] == 1.0
    assert res["mean_detection_delay"] == 0.0 and res["runs_detected"] == "2/2"


def test_never_alert_misses_everything():
    res = run(False, ["sudden_shift"])["combined"]["sudden_shift"]
    assert res["recall"] == 0.0 and res["false_alarm_rate"] == 0.0
    assert res["mean_detection_delay"] is None and res["runs_detected"] == "0/2"


def test_healthy_scenario_has_no_precision_or_recall():
    res = run(False, ["healthy"])["combined"]["healthy"]
    assert res["precision"] is None and res["recall"] is None
    assert res["false_alarm_rate"] == 0.0


def test_every_detector_and_overall_reported():
    res = run(True, ["healthy", "noise"])
    assert set(res) == set(DETECTORS)
    assert {"healthy", "noise", "overall"} == set(res["combined"])
