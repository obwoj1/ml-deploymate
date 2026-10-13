"""Spot batches where the model is behaving unusually.

Two complementary checks, both learned from a set of known-healthy batches:

1. Adaptive thresholds: for each metric, learn its normal mean and spread.
   A batch is flagged on that metric if it's more than Z_THRESHOLD standard
   deviations away. Thresholds come from the data, not hand-picked numbers.
2. Isolation Forest: looks at all metrics TOGETHER and flags combinations
   that are unusual even when no single metric is extreme. Its alarm line is
   set at the strangest healthy batch seen during calibration, so a batch only
   fires if it's weirder than anything normal traffic produced. (We tried
   sklearn's `contamination` setting instead, which by design flags a fixed
   share of healthy batches: ~2% false alarms.)
"""

import numpy as np
from sklearn.ensemble import IsolationForest

from deploymate.monitor.metrics import LABEL_FREE

Z_THRESHOLD = 4.0
# Floor on the learned spread so a metric that barely moves in healthy data
# (e.g. unknown_rate) doesn't fire on tiny wiggles.
MIN_STD = {"positive_rate": 0.01, "mean_confidence": 0.005,
           "low_confidence_share": 0.01, "unknown_rate": 0.005}


class AnomalyDetector:
    def __init__(self, metrics=LABEL_FREE, seed: int = 0):
        self.metrics = list(metrics)
        self.forest = IsolationForest(n_estimators=200, random_state=seed)

    def _matrix(self, rows):
        return np.array([[row[m] for m in self.metrics] for row in rows])

    def fit(self, healthy_rows: list[dict]) -> "AnomalyDetector":
        X = self._matrix(healthy_rows)
        self.mean_ = X.mean(axis=0)
        self.std_ = np.maximum(X.std(axis=0), [MIN_STD.get(m, 1e-3) for m in self.metrics])
        self.forest.fit(X)
        self.threshold_ = float(self._score(X).max())
        return self

    def _score(self, X) -> np.ndarray:
        return -self.forest.score_samples(X)  # higher = stranger

    def check(self, row: dict) -> dict:
        x = self._matrix([row])
        z = (x[0] - self.mean_) / self.std_
        flagged = [m for m, score in zip(self.metrics, z) if abs(score) > Z_THRESHOLD]
        score = float(self._score(x)[0])
        forest_flag = score > self.threshold_
        return {
            "anomalous": bool(flagged) or forest_flag,
            "flagged_metrics": flagged,
            "isolation_forest": forest_flag,
            "anomaly_score": round(score, 4),
            "anomaly_threshold": round(self.threshold_, 4),
            "z_scores": {m: round(float(s), 2) for m, s in zip(self.metrics, z)},
        }
