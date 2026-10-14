import numpy as np
import pandas as pd
import pytest

from deploymate.data.load import CATEGORICAL, FEATURES, NUMERIC
from deploymate.model.pipeline import build_model, score

rng = np.random.default_rng(0)


@pytest.fixture
def toy_data():
    n = 400
    X = pd.DataFrame({col: rng.normal(40, 10, n) for col in NUMERIC})
    for col in CATEGORICAL:
        X[col] = rng.choice(["a", "b", "c"], n)
    # Learnable rule: older + category "a" -> positive
    y = ((X["age"] > 40) & (X["workclass"] == "a")).astype(int)
    return X[FEATURES], y


def test_model_learns_simple_rule(toy_data):
    X, y = toy_data
    model = build_model().fit(X, y)
    metrics = score(model, X, y)
    assert set(metrics) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert metrics["roc_auc"] > 0.9


def test_unseen_category_does_not_crash(toy_data):
    X, y = toy_data
    model = build_model().fit(X, y)
    new_row = X.iloc[[0]].copy()
    new_row["occupation"] = "never-seen-before"
    assert model.predict(new_row).shape == (1,)


def test_compare_models_ranks_all_candidates(toy_data):
    from deploymate.model.train import candidates, compare_models
    X, y = toy_data
    rows = compare_models(X, y, folds=3)
    assert {r["model"] for r in rows} == set(candidates())
    aucs = [r["cv_roc_auc"] for r in rows]
    assert aucs == sorted(aucs, reverse=True)


def test_model_report_has_dashboard_fields(toy_data):
    from deploymate.model.train import model_report
    X, y = toy_data
    model = build_model().fit(X, y)
    report = model_report("LogReg", model, X, y, X, y, comparison=[])
    assert {"train", "test"} == set(report["metrics"])
    assert len(report["confusion_matrix"]["matrix"]) == 2
    assert len(report["roc_curve"]["fpr"]) == len(report["roc_curve"]["tpr"])
    assert {f["feature"] for f in report["feature_importance"]} == set(FEATURES)
