"""The ONE module the dashboard (app/) is allowed to import.

Frontend team: call these functions and nothing else from `deploymate`.
Right now they return realistic MOCK data so you can build every page today.
Later the backend swaps the internals for the real model and monitor, and the
function names and return shapes stay exactly the same (tests/test_service.py
enforces that).

All return values are plain dicts/lists, so they're easy to drop into
st.metric, st.dataframe, and Plotly.
"""

from __future__ import annotations

import math
from typing import Iterator

import numpy as np

MOCK = True  # flips to False once the real backend is wired in

FEATURES_NUMERIC = ["age", "hours_per_week", "education_num", "capital_gain"]
FEATURES_CATEGORICAL = ["workclass", "occupation", "marital_status", "native_country"]

SCENARIOS = {
    "healthy": "Normal traffic: nothing changes.",
    "gradual_shift": "Ages slowly creep upward batch after batch.",
    "sudden_shift": "Hours-per-week jumps suddenly halfway through.",
    "category_shift": "The occupation mix changes sharply.",
    "missing_values": "An upstream pipeline breaks and fields go blank.",
    "noise": "Random noise is added to numeric fields.",
}

# Which feature each scenario breaks, and from which batch on.
_SCENARIO_TARGET = {
    "gradual_shift": ("age", 0),
    "sudden_shift": ("hours_per_week", 0.5),
    "category_shift": ("occupation", 0.4),
    "missing_values": ("workclass", 0.5),
    "noise": ("capital_gain", 0.3),
}

_ALERTS = [
    {"id": 1, "time": "2026-10-08 09:14", "scenario": "sudden_shift", "batch_id": 16,
     "severity": "high", "features": ["hours_per_week"], "accuracy": 0.79, "psi_max": 0.41},
    {"id": 2, "time": "2026-10-08 10:02", "scenario": "category_shift", "batch_id": 13,
     "severity": "medium", "features": ["occupation"], "accuracy": 0.82, "psi_max": 0.18},
    {"id": 3, "time": "2026-10-08 11:47", "scenario": "missing_values", "batch_id": 15,
     "severity": "high", "features": ["workclass", "occupation"], "accuracy": 0.77, "psi_max": 0.0},
]


def get_overview() -> dict:
    """Headline numbers for the Overview page."""
    return {
        "deployment": "income-classifier-v1",
        "model": "Gradient Boosting",
        "status": "degraded" if _ALERTS else "healthy",
        "batches_processed": 180,
        "active_alerts": len(_ALERTS),
        "test_accuracy": 0.871,
        "last_updated": "2026-10-08 11:47",
        "mock": MOCK,
    }


def list_scenarios() -> list[dict]:
    """Scenarios the Live Monitor page can replay."""
    return [{"name": name, "description": desc} for name, desc in SCENARIOS.items()]


def run_scenario(name: str, n_batches: int = 30, seed: int = 0) -> Iterator[dict]:
    """Yield one batch report at a time, like live traffic arriving.

    Each report:
        batch_id, scenario,
        metrics: {accuracy, f1, positive_rate, mean_confidence, missing_rate},
        drift:   [{feature, kind, statistic, p_value, drifted}, ...],
        alert:   bool, severity: "none" | "medium" | "high"
    """
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario {name!r}. Choose from {list(SCENARIOS)}")

    rng = np.random.default_rng(seed)
    target, start = _SCENARIO_TARGET.get(name, (None, 1.0))

    for i in range(n_batches):
        progress = i / max(n_batches - 1, 1)
        # How broken this batch is: 0 = healthy, 1 = fully drifted.
        if name == "gradual_shift":
            strength = progress
        elif target and progress >= start:
            strength = 1.0
        else:
            strength = 0.0

        drift = []
        for feature in FEATURES_NUMERIC + FEATURES_CATEGORICAL:
            hit = feature == target
            if feature in FEATURES_NUMERIC:
                stat = abs(rng.normal(0.03, 0.015)) + (0.45 * strength if hit else 0)
                drifted = stat >= 0.2
                kind = "numeric"
            else:
                stat = abs(rng.normal(0.02, 0.01)) + (0.25 * strength if hit else 0)
                drifted = stat >= 0.1
                kind = "categorical"
            p_value = float(math.exp(-40 * stat))
            drift.append({"feature": feature, "kind": kind, "statistic": round(float(stat), 4),
                          "p_value": round(p_value, 6), "drifted": bool(drifted)})

        missing_rate = 0.07 + (0.25 * strength if name == "missing_values" else 0)
        accuracy = 0.87 - 0.09 * strength + rng.normal(0, 0.008)
        metrics = {
            "accuracy": round(float(accuracy), 4),
            "f1": round(float(accuracy - 0.16 + rng.normal(0, 0.01)), 4),
            "positive_rate": round(float(0.24 + 0.06 * strength + rng.normal(0, 0.01)), 4),
            "mean_confidence": round(float(0.84 - 0.08 * strength + rng.normal(0, 0.005)), 4),
            "missing_rate": round(float(missing_rate), 4),
        }

        n_drifted = sum(d["drifted"] for d in drift)
        alert = n_drifted > 0 or missing_rate > 0.15
        severity = "none" if not alert else ("high" if strength >= 0.7 else "medium")
        yield {"batch_id": i + 1, "scenario": name, "metrics": metrics,
               "drift": drift, "alert": alert, "severity": severity}


def list_alerts() -> list[dict]:
    """Alerts for the Alerts page, newest first."""
    return sorted(_ALERTS, key=lambda a: a["time"], reverse=True)


def explain_alert(alert_id: int) -> dict:
    """Plain-English explanation of one alert (the LLM's job once it's real)."""
    alert = next((a for a in _ALERTS if a["id"] == alert_id), None)
    if alert is None:
        raise ValueError(f"No alert with id {alert_id}")
    features = ", ".join(alert["features"])
    return {
        "alert_id": alert_id,
        "summary": f"Batch {alert['batch_id']} no longer looks like the training data: "
                   f"{features} shifted, and accuracy fell to {alert['accuracy']:.0%}.",
        "likely_cause": f"The incoming values for {features} changed. That usually means "
                        "a new user population or a broken upstream data source.",
        "suggested_action": "Check the upstream data pipeline for recent changes. If the "
                            "shift is real, collect fresh labeled data and retrain.",
        "source": "mock",
    }


def get_model_report() -> dict:
    """Everything the Model Performance page needs."""
    fpr = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    tpr = [0.0, 0.35, 0.55, 0.72, 0.85, 0.91, 0.96, 0.98, 1.0]
    return {
        "model": "Gradient Boosting",
        "metrics": {
            "train": {"accuracy": 0.884, "precision": 0.80, "recall": 0.66, "f1": 0.72, "roc_auc": 0.935},
            "test": {"accuracy": 0.871, "precision": 0.78, "recall": 0.63, "f1": 0.70, "roc_auc": 0.926},
        },
        "confusion_matrix": {"labels": ["<=50K", ">50K"], "matrix": [[11800, 635], [1420, 2426]]},
        "roc_curve": {"fpr": fpr, "tpr": tpr},
        "feature_importance": [
            {"feature": "relationship", "importance": 0.24},
            {"feature": "capital_gain", "importance": 0.19},
            {"feature": "education_num", "importance": 0.15},
            {"feature": "marital_status", "importance": 0.12},
            {"feature": "age", "importance": 0.10},
            {"feature": "occupation", "importance": 0.08},
            {"feature": "hours_per_week", "importance": 0.07},
            {"feature": "capital_loss", "importance": 0.05},
        ],
    }


def generate_config(form: dict) -> dict:
    """Generate deployment files from the Config Generator form.

    form keys: model_name, python_version, port, framework ("fastapi" | "streamlit")
    """
    name = form.get("model_name", "my-model")
    py = form.get("python_version", "3.11")
    port = int(form.get("port", 8000))
    if form.get("framework", "fastapi") == "streamlit":
        cmd = f'["streamlit", "run", "app.py", "--server.port", "{port}"]'
    else:
        cmd = f'["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "{port}"]'

    dockerfile = (
        f"FROM python:{py}-slim\n"
        "WORKDIR /app\n"
        "COPY requirements.txt .\n"
        "RUN pip install --no-cache-dir -r requirements.txt\n"
        "COPY . .\n"
        f"EXPOSE {port}\n"
        f"CMD {cmd}\n"
    )
    github_actions = (
        f"name: {name} CI\n"
        "on: [push, pull_request]\n"
        "jobs:\n"
        "  test:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/checkout@v4\n"
        "      - uses: actions/setup-python@v5\n"
        "        with:\n"
        f'          python-version: "{py}"\n'
        "      - run: pip install -r requirements.txt\n"
        "      - run: pytest -q\n"
    )
    env_example = f"MODEL_NAME={name}\nPORT={port}\nLOG_LEVEL=info\n"
    return {"dockerfile": dockerfile, "github_actions": github_actions,
            "env_example": env_example, "source": "mock"}
