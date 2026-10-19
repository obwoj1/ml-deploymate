"""The ONE module the dashboard (app/) is allowed to import.

Frontend team: call these functions and nothing else from `deploymate`.
They now run the REAL model, monitor and AI explanations. The function names
and return shapes are the same as the mock version (tests/test_service.py
enforces that), so pages built against the mock work unchanged.

Demo data fallback: if the real backend can't start (no dataset and no
internet, for example) or DEPLOYMATE_MOCK=true, every function returns the
mock data from mock_service.py instead, and get_overview()["mock"] is True.

First call takes a few seconds (load data + model, calibrate the monitor);
after that everything is cached in memory. If the dataset or trained model is
missing, it downloads / trains them automatically (so the live site sets
itself up).
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Iterator

from deploymate import mock_service

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "model.joblib"
REPORT_PATH = ROOT / "models" / "report.json"
DEPLOYMENT_NAME = "income-classifier-v1"
CALIBRATION_SEED = 100_000  # same as scripts/evaluate_monitor.py
MAX_ALERTS = 200

generate_config = mock_service.generate_config  # template-based; no model needed


class _Backend:
    """Everything loaded once: data, model, calibrated monitor, alert log."""

    def __init__(self):
        import joblib

        from deploymate.data.download import download, is_downloaded
        from deploymate.data.load import load_dataset
        from deploymate.monitor.pipeline import Monitor
        from deploymate.simulate.scenarios import simulate

        if not is_downloaded():
            download(verbose=False)
        X_train, self.X_test, y_train, self.y_test = load_dataset()

        if not MODEL_PATH.exists():  # live site / fresh clone: quick fit of the chosen model
            from deploymate.model.pipeline import build_model
            from deploymate.model.train import candidates
            MODEL_PATH.parent.mkdir(exist_ok=True)
            joblib.dump(build_model(candidates()["Gradient Boosting"]).fit(X_train, y_train), MODEL_PATH)
        model = joblib.load(MODEL_PATH)

        healthy = [b.X for b in simulate("healthy", self.X_test, self.y_test,
                                         n_batches=200, seed=CALIBRATION_SEED)]
        self.monitor = Monitor.from_data(model, X_train.sample(5000, random_state=0), healthy)
        self.report = json.loads(REPORT_PATH.read_text())
        self.alerts: list[dict] = []          # newest last
        self.reports: dict[int, dict] = {}    # alert id -> full monitor report
        self.batches_processed = 0
        self.last_updated = None
        self.lock = threading.Lock()          # Streamlit serves sessions on threads

    def record(self, scenario: str, report: dict) -> None:
        with self.lock:
            self.batches_processed += 1
            self.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if not report["alert"]:
                return
            alert_id = (self.alerts[-1]["id"] + 1) if self.alerts else 1
            drifted = [d for d in report["drift"] if d["drifted"]]
            self.alerts.append({
                "id": alert_id,
                "time": self.last_updated,
                "scenario": scenario,
                "batch_id": report["batch_id"],
                "severity": report["severity"],
                "features": report["drifted_features"],
                "accuracy": report["metrics"].get("accuracy"),
                "psi_max": round(max((d["statistic"] for d in drifted if d["kind"] == "numeric"),
                                     default=0.0), 4),
            })
            self.reports[alert_id] = report
            if len(self.alerts) > MAX_ALERTS:  # keep memory bounded on a long-running server
                dropped = self.alerts.pop(0)
                self.reports.pop(dropped["id"], None)


_backend: _Backend | None = None
_backend_error: str | None = None
_init_lock = threading.Lock()


def _get_backend() -> _Backend | None:
    """The real backend, or None (→ mock data) if it's disabled or failed to start."""
    global _backend, _backend_error
    if os.getenv("DEPLOYMATE_MOCK", "false").lower() == "true":
        return None
    if _backend is None and _backend_error is None:
        with _init_lock:
            if _backend is None and _backend_error is None:
                try:
                    _backend = _Backend()
                except Exception as err:  # noqa: BLE001 - any failure → demo data, never a crash
                    _backend_error = f"{type(err).__name__}: {err}"
    return _backend


def backend_status() -> dict:
    """For debugging/the Home page: is the real backend running, and if not, why?"""
    backend = _get_backend()
    return {"real": backend is not None, "error": _backend_error}


def get_overview() -> dict:
    """Headline numbers for the Overview page."""
    b = _get_backend()
    if b is None:
        return mock_service.get_overview()
    with b.lock:
        return {
            "deployment": DEPLOYMENT_NAME,
            "model": b.report["model"],
            "status": "degraded" if b.alerts else "healthy",
            "batches_processed": b.batches_processed,
            "active_alerts": len(b.alerts),
            "test_accuracy": b.report["metrics"]["test"]["accuracy"],
            "last_updated": b.last_updated or "-",
            "mock": False,
        }


def list_scenarios() -> list[dict]:
    """Scenarios the Live Monitor page can replay."""
    from deploymate.simulate.scenarios import SCENARIOS
    return [{"name": name, "description": desc} for name, (desc, _, _) in SCENARIOS.items()]


def run_scenario(name: str, n_batches: int = 30, seed: int = 0) -> Iterator[dict]:
    """Yield one monitor report per batch, like live traffic arriving.

    Each report:
        batch_id, scenario,
        metrics: {accuracy, f1, positive_rate, mean_confidence, low_confidence_share, unknown_rate},
        drift:   [{feature, kind, statistic, p_value, drifted, shift?}, ...],
        drifted_features, anomaly (info only), alert: bool, severity: "none" | "medium" | "high"
    Alerts are recorded, so they show up in list_alerts() / get_overview().
    """
    b = _get_backend()
    if b is None:
        yield from mock_service.run_scenario(name, n_batches, seed)
        return

    from deploymate.simulate.scenarios import SCENARIOS, simulate
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario {name!r}. Choose from {list(SCENARIOS)}")
    for batch in simulate(name, b.X_test, b.y_test, n_batches=n_batches, seed=seed):
        report = b.monitor.check(batch.X, batch.y, batch_id=batch.batch_id)
        report["scenario"] = name
        b.record(name, report)
        yield report


def _ensure_demo_alerts(b: _Backend) -> None:
    """On a fresh server the alert log is empty; seed it with one run of a few scenarios."""
    if b.alerts:
        return
    for scenario in ["sudden_shift", "category_shift", "missing_values"]:
        for report in run_scenario(scenario, n_batches=30, seed=7):
            if report["alert"]:
                break  # one alert per scenario is enough for the demo


def list_alerts() -> list[dict]:
    """Alerts for the Alerts page, newest first."""
    b = _get_backend()
    if b is None:
        return mock_service.list_alerts()
    _ensure_demo_alerts(b)
    with b.lock:
        return sorted(b.alerts, key=lambda a: a["id"], reverse=True)


def explain_alert(alert_id: int) -> dict:
    """Plain-English explanation of one alert (OpenAI, or a template if unavailable)."""
    b = _get_backend()
    if b is None:
        return mock_service.explain_alert(alert_id)
    with b.lock:
        report = b.reports.get(alert_id)
    if report is None:
        raise ValueError(f"No alert with id {alert_id}")
    from deploymate.llm.explain import explain_alert as llm_explain
    return {"alert_id": alert_id, **llm_explain(report)}


def get_model_report() -> dict:
    """Everything the Model Performance page needs (from models/report.json)."""
    b = _get_backend()
    if b is None:
        return mock_service.get_model_report()
    return b.report
