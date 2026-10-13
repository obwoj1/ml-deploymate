"""Train the model DeployMate monitors.

    python scripts/train.py

1. Cross-validates Logistic Regression, Random Forest and Gradient Boosting on the training data
2. Picks the best by ROC-AUC and refits it on all the training data
3. Scores it once on the held-out test data
4. Saves models/model.joblib (the fitted Pipeline) and models/report.json (metrics for the dashboard)
"""

import json
import sys
import time
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from deploymate.data.load import load_dataset  # noqa: E402
from deploymate.model.pipeline import build_model  # noqa: E402
from deploymate.model.train import candidates, compare_models, model_report  # noqa: E402

MODELS_DIR = ROOT / "models"


def main() -> None:
    start = time.time()
    X_train, X_test, y_train, y_test = load_dataset()
    print(f"Loaded {len(X_train):,} train / {len(X_test):,} test rows\n")

    print("Cross-validating (5 folds on the training data)...")
    comparison = compare_models(X_train, y_train)
    print(f"\n{'model':<22}{'accuracy':>10}{'F1':>8}{'ROC-AUC':>10}")
    for row in comparison:
        print(f"{row['model']:<22}{row['cv_accuracy']:>10.4f}{row['cv_f1']:>8.4f}"
              f"{row['cv_roc_auc']:>10.4f}")

    best = comparison[0]["model"]
    print(f"\nWinner: {best}. Refitting on all training data...")
    model = build_model(candidates()[best]).fit(X_train, y_train)

    report = model_report(best, model, X_train, y_train, X_test, y_test, comparison)
    test = report["metrics"]["test"]
    print(f"Test set: accuracy {test['accuracy']:.4f} | F1 {test['f1']:.4f} | ROC-AUC {test['roc_auc']:.4f}")

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODELS_DIR / "model.joblib")
    (MODELS_DIR / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\nSaved models/model.joblib and models/report.json ({time.time() - start:.0f}s total)")


if __name__ == "__main__":
    main()
