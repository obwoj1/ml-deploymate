"""Build and score the income classifier that DeployMate monitors.

Every model is a scikit-learn Pipeline: preprocessing + classifier in one
object. That way the saved model takes raw rows (the same columns
load_dataset() returns) and handles scaling and encoding itself, so the API
and the monitor never have to repeat the preprocessing.
"""

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from deploymate.data.load import CATEGORICAL, NUMERIC


def build_preprocessor() -> ColumnTransformer:
    """Scale numeric columns; one-hot encode categorical ones.

    handle_unknown="ignore" matters for monitoring: if live data contains a
    category the model never saw (e.g. a new occupation), it's encoded as all
    zeros instead of crashing the prediction.
    """
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])


def build_model(classifier=None) -> Pipeline:
    """Preprocessing + classifier. Defaults to the Logistic Regression baseline."""
    if classifier is None:
        classifier = LogisticRegression(max_iter=1000)
    return Pipeline([("preprocess", build_preprocessor()), ("classifier", classifier)])


def score(model: Pipeline, X, y) -> dict:
    """Standard metrics for a fitted model on one split.

    Accuracy alone is misleading here (76% of people earn <=50K), so we also
    report precision, recall, F1 and ROC-AUC.
    """
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    return {
        "accuracy": round(accuracy_score(y, predictions), 4),
        "precision": round(precision_score(y, predictions), 4),
        "recall": round(recall_score(y, predictions), 4),
        "f1": round(f1_score(y, predictions), 4),
        "roc_auc": round(roc_auc_score(y, probabilities), 4),
    }
