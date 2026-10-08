"""Contract tests for deploymate.service.

These pin down the shapes the dashboard depends on. When the mock internals are
replaced by the real backend, these tests must still pass unchanged.
"""

import pytest

from deploymate import service


def test_overview_has_dashboard_fields():
    overview = service.get_overview()
    for key in ["deployment", "model", "status", "batches_processed", "active_alerts", "test_accuracy"]:
        assert key in overview
    assert overview["status"] in {"healthy", "degraded"}


def test_scenarios_have_names_and_descriptions():
    scenarios = service.list_scenarios()
    assert {"name", "description"} <= set(scenarios[0])
    assert "healthy" in [s["name"] for s in scenarios]


@pytest.mark.parametrize("name", [s["name"] for s in service.list_scenarios()])
def test_run_scenario_batch_shape(name):
    batches = list(service.run_scenario(name, n_batches=10))
    assert len(batches) == 10
    batch = batches[0]
    assert {"batch_id", "scenario", "metrics", "drift", "alert", "severity"} <= set(batch)
    assert {"accuracy", "f1", "positive_rate", "mean_confidence", "missing_rate"} <= set(batch["metrics"])
    assert {"feature", "kind", "statistic", "p_value", "drifted"} <= set(batch["drift"][0])
    assert batch["severity"] in {"none", "medium", "high"}


def test_healthy_scenario_raises_no_alerts():
    assert not any(b["alert"] for b in service.run_scenario("healthy", n_batches=30))


def test_broken_scenarios_eventually_alert():
    for name in ["gradual_shift", "sudden_shift", "category_shift", "missing_values", "noise"]:
        assert any(b["alert"] for b in service.run_scenario(name, n_batches=30)), name


def test_run_scenario_is_repeatable_with_same_seed():
    first = list(service.run_scenario("sudden_shift", seed=7))
    second = list(service.run_scenario("sudden_shift", seed=7))
    assert first == second


def test_unknown_scenario_raises():
    with pytest.raises(ValueError):
        list(service.run_scenario("not_a_scenario"))


def test_alerts_and_explanations():
    alerts = service.list_alerts()
    assert alerts
    explanation = service.explain_alert(alerts[0]["id"])
    assert {"summary", "likely_cause", "suggested_action"} <= set(explanation)
    with pytest.raises(ValueError):
        service.explain_alert(-1)


def test_model_report_shape():
    report = service.get_model_report()
    for split in ["train", "test"]:
        assert {"accuracy", "f1", "roc_auc"} <= set(report["metrics"][split])
    assert len(report["confusion_matrix"]["matrix"]) == 2
    assert len(report["roc_curve"]["fpr"]) == len(report["roc_curve"]["tpr"])
    assert {"feature", "importance"} <= set(report["feature_importance"][0])


@pytest.mark.parametrize("framework", ["fastapi", "streamlit"])
def test_generate_config_returns_all_files(framework):
    files = service.generate_config({"model_name": "demo", "port": 8501, "framework": framework})
    assert {"dockerfile", "github_actions", "env_example"} <= set(files)
    assert "8501" in files["dockerfile"]
