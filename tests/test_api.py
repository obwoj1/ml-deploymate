import pytest
from fastapi.testclient import TestClient

from api.main import app
from deploymate.data.download import is_downloaded

client = TestClient(app)

PERSON = {"age": 39, "education_num": 13, "capital_gain": 0, "capital_loss": 0,
          "hours_per_week": 40, "workclass": "Private", "marital_status": "Never-married",
          "occupation": "Adm-clerical", "relationship": "Not-in-family", "race": "White",
          "sex": "Male", "native_country": "United-States"}

real_only = pytest.mark.skipif(not is_downloaded(), reason="real dataset not downloaded")


@pytest.fixture(autouse=True)
def no_openai(monkeypatch):
    monkeypatch.setenv("LLM_OFFLINE_MODE", "true")


@pytest.fixture
def mock_mode(monkeypatch):
    monkeypatch.setenv("DEPLOYMATE_MOCK", "true")


def test_health(mock_mode):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["real_backend"] is False


def test_overview_scenarios_report_alerts_config(mock_mode):
    assert "status" in client.get("/overview").json()
    assert "healthy" in [s["name"] for s in client.get("/monitor/scenarios").json()]
    assert "metrics" in client.get("/model/report").json()
    alerts = client.get("/alerts").json()
    assert {"summary", "likely_cause"} <= set(client.get(f"/alerts/{alerts[0]['id']}/explain").json())
    files = client.post("/config", json={"model_name": "demo", "port": 9000}).json()
    assert "9000" in files["dockerfile"]


def test_errors_are_clean(mock_mode):
    assert client.get("/alerts/999999/explain").status_code == 404
    assert client.post("/monitor/scenarios/not_real/run").status_code == 404
    assert client.post("/monitor/scenarios/healthy/run?n_batches=0").status_code == 422
    assert client.post("/predict", json={"rows": [{**PERSON, "age": 5}]}).status_code == 422  # validation
    assert client.post("/predict", json={"rows": []}).status_code == 422
    assert client.post("/config", json={"framework": "django"}).status_code == 422


def test_predict_needs_real_backend(mock_mode):
    assert client.post("/predict", json={"rows": [PERSON]}).status_code == 503


@real_only
def test_predict_real(monkeypatch):
    monkeypatch.setenv("DEPLOYMATE_MOCK", "false")
    rich = {**PERSON, "marital_status": "Married-civ-spouse", "relationship": "Husband",
            "capital_gain": 15000, "education_num": 16, "age": 50, "hours_per_week": 55}
    out = client.post("/predict", json={"rows": [PERSON, rich]}).json()
    assert [o["prediction"] for o in out] == ["<=50K", ">50K"]


@real_only
def test_monitor_check_real(monkeypatch):
    monkeypatch.setenv("DEPLOYMATE_MOCK", "false")
    from deploymate.data.load import load_dataset
    _, X_test, _, _ = load_dataset()
    healthy = X_test.sample(400, random_state=1).to_dict("records")
    assert client.post("/monitor/check", json={"rows": healthy}).json()["alert"] is False
    shifted = [{**r, "hours_per_week": min(r["hours_per_week"] + 15, 99)} for r in healthy]
    report = client.post("/monitor/check", json={"rows": shifted}).json()
    assert report["alert"] and "hours_per_week" in report["drifted_features"]
    assert client.get("/alerts").json()[0]["scenario"] == "api"
