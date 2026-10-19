"""DeployMate REST API: the same backend as the dashboard, for other programs.

    uvicorn api.main:app --reload
    → interactive docs at http://localhost:8000/docs

The dashboard is for people; this API is how another system would use
DeployMate: send it a batch of live data, get back drift alerts and
plain-English explanations. Every endpoint is a thin wrapper around
deploymate.service, so the API and the dashboard always agree.
"""

from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from deploymate import service

app = FastAPI(
    title="ML DeployMate API",
    description="Monitor a deployed ML model: drift detection, alerts and AI explanations.",
    version="1.0.0",
)


class Person(BaseModel):
    """One row of input: the 12 features the model uses (see docs/DATASET.md)."""
    age: int = Field(ge=17, le=90, examples=[39])
    education_num: int = Field(ge=1, le=16, examples=[13], description="Years of education (13 = Bachelors)")
    capital_gain: float = Field(ge=0, examples=[0])
    capital_loss: float = Field(ge=0, examples=[0])
    hours_per_week: int = Field(ge=1, le=99, examples=[40])
    workclass: str = Field(examples=["Private"])
    marital_status: str = Field(examples=["Never-married"])
    occupation: str = Field(examples=["Adm-clerical"])
    relationship: str = Field(examples=["Not-in-family"])
    race: str = Field(examples=["White"])
    sex: str = Field(examples=["Male"])
    native_country: str = Field(examples=["United-States"])


class Batch(BaseModel):
    rows: list[Person] = Field(min_length=1, max_length=5000)


class Prediction(BaseModel):
    prediction: Literal["<=50K", ">50K"]
    probability_over_50K: float


class ConfigForm(BaseModel):
    model_name: str = "my-model"
    python_version: str = "3.11"
    framework: Literal["fastapi", "streamlit"] = "fastapi"
    port: int = Field(default=8000, ge=1, le=65535)


def _rows(batch: Batch) -> list[dict]:
    return [row.model_dump() for row in batch.rows]


def _needs_real_backend(call, *args):
    try:
        return call(*args)
    except service.BackendUnavailable as err:
        raise HTTPException(503, f"Real backend unavailable: {err}") from err


@app.get("/health", tags=["status"])
def health() -> dict:
    """Is the API up, and is it running on the real model (or demo data)?"""
    status = service.backend_status()
    return {"status": "ok", "real_backend": status["real"], "error": status["error"]}


@app.get("/overview", tags=["status"])
def overview() -> dict:
    return service.get_overview()


@app.post("/predict", tags=["model"], response_model=list[Prediction])
def predict(batch: Batch):
    """Predict whether each person earns more than $50K."""
    return _needs_real_backend(service.predict, _rows(batch))


@app.get("/model/report", tags=["model"])
def model_report() -> dict:
    """Test metrics, model comparison, confusion matrix, ROC curve, feature importance."""
    return service.get_model_report()


@app.post("/monitor/check", tags=["monitor"])
def check(batch: Batch) -> dict:
    """Check a batch of live data for drift. Raises an alert if it doesn't look like the training data."""
    return _needs_real_backend(service.check_batch, _rows(batch))


@app.get("/monitor/scenarios", tags=["monitor"])
def scenarios() -> list[dict]:
    return service.list_scenarios()


@app.post("/monitor/scenarios/{name}/run", tags=["monitor"])
def run_scenario(name: str, n_batches: int = 20) -> dict:
    """Replay a simulated scenario; returns a one-line summary per batch."""
    if not 1 <= n_batches <= 100:
        raise HTTPException(422, "n_batches must be between 1 and 100")
    try:
        reports = list(service.run_scenario(name, n_batches=n_batches))
    except ValueError as err:
        raise HTTPException(404, str(err)) from err
    return {
        "scenario": name,
        "alerts": sum(r["alert"] for r in reports),
        "batches": [{"batch_id": r["batch_id"], "alert": r["alert"], "severity": r["severity"],
                     "drifted_features": [d["feature"] for d in r["drift"] if d["drifted"]]}
                    for r in reports],
    }


@app.get("/alerts", tags=["alerts"])
def alerts() -> list[dict]:
    return service.list_alerts()


@app.get("/alerts/{alert_id}/explain", tags=["alerts"])
def explain(alert_id: int) -> dict:
    """Plain-English explanation: what happened, likely cause, suggested action."""
    try:
        return service.explain_alert(alert_id)
    except ValueError as err:
        raise HTTPException(404, str(err)) from err


@app.post("/config", tags=["tools"])
def config(form: ConfigForm) -> dict:
    """Generate a Dockerfile, GitHub Actions workflow and .env.example."""
    return service.generate_config(form.model_dump())
