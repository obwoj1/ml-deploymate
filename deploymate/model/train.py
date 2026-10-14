"""Compare candidate models with cross-validation, then build the final model report.

Used by scripts/train.py. Kept separate so it can be tested on small data.
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_validate

from deploymate.model.pipeline import build_model, score

SEED = 42


def candidates() -> dict:
    """The models we compare. Same preprocessing; only the classifier changes."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=SEED),
        "Gradient Boosting": HistGradientBoostingClassifier(random_state=SEED),
    }


def compare_models(X, y, folds: int = 5) -> list[dict]:
    """5-fold cross-validation on the TRAINING data only.

    The test set stays untouched until the winner is chosen, so the final test
    score is an honest estimate of real-world performance.
    """
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    rows = []
    for name, classifier in candidates().items():
        result = cross_validate(build_model(classifier), X, y, cv=cv,
                                scoring=["accuracy", "f1", "roc_auc"], n_jobs=1)
        rows.append({
            "model": name,
            "cv_accuracy": round(result["test_accuracy"].mean(), 4),
            "cv_f1": round(result["test_f1"].mean(), 4),
            "cv_roc_auc": round(result["test_roc_auc"].mean(), 4),
            "cv_roc_auc_std": round(result["test_roc_auc"].std(), 4),
        })
    # ROC-AUC picks the winner: it measures ranking quality across all
    # thresholds and isn't fooled by the 76/24 class imbalance.
    return sorted(rows, key=lambda r: r["cv_roc_auc"], reverse=True)


def model_report(name, model, X_train, y_train, X_test, y_test, comparison) -> dict:
    """Everything the dashboard's Model Performance page needs, as plain JSON."""
    probabilities = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probabilities)
    # Downsample the ROC curve to ~50 points so the JSON stays small.
    keep = np.unique(np.linspace(0, len(fpr) - 1, 50).astype(int))

    # Permutation importance: shuffle one ORIGINAL column at a time and measure
    # how much test ROC-AUC drops. Works the same for any model type.
    importance = permutation_importance(model, X_test, y_test, scoring="roc_auc",
                                        n_repeats=3, random_state=SEED, n_jobs=-1)
    features = sorted(
        ({"feature": col, "importance": round(float(val), 4)}
         for col, val in zip(X_test.columns, importance.importances_mean)),
        key=lambda f: f["importance"], reverse=True)

    return {
        "model": name,
        "metrics": {"train": score(model, X_train, y_train),
                    "test": score(model, X_test, y_test)},
        "baseline_always_negative_accuracy": round(float(1 - y_test.mean()), 4),
        "confusion_matrix": {"labels": ["<=50K", ">50K"],
                             "matrix": confusion_matrix(y_test, model.predict(X_test)).tolist()},
        "roc_curve": {"fpr": [round(float(v), 4) for v in fpr[keep]],
                      "tpr": [round(float(v), 4) for v in tpr[keep]]},
        "feature_importance": features,
        "comparison": comparison,
    }
