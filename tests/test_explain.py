from types import SimpleNamespace

import openai
import pytest

from deploymate.llm import explain
from deploymate.llm.explain import Explanation, alert_context, explain_alert, template_explanation


def make_report(drift, alert=True):
    return {
        "batch_id": 40, "alert": alert, "severity": "high" if alert else "none",
        "metrics": {"positive_rate": 0.3, "mean_confidence": 0.84, "unknown_rate": 0.01},
        "drift": drift,
        "anomaly": {"flagged_metrics": [], "anomalous": False},
    }


HOURS = {"feature": "hours_per_week", "kind": "numeric", "statistic": 0.61, "p_value": 0.0,
         "drifted": True, "shift": {"reference_mean": 40.3, "current_mean": 55.1}}
BLANK = {"feature": "workclass", "kind": "categorical", "statistic": 0.4, "p_value": 0.0,
         "drifted": True, "shift": {"category": "Unknown", "reference_share": 0.06, "current_share": 0.43}}
QUIET = {"feature": "age", "kind": "numeric", "statistic": 0.02, "p_value": 0.9, "drifted": False}


class FakeClient:
    """Stands in for openai.OpenAI: returns a canned answer or raises an error."""
    def __init__(self, answer=None, error=None):
        self.calls = []
        self.answer, self.error = answer, error
        self.chat = SimpleNamespace(completions=SimpleNamespace(parse=self._parse))

    def _parse(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        message = SimpleNamespace(parsed=self.answer, refusal=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    # Never hit the real API from tests, even if a developer has a key in .env.
    monkeypatch.setenv("LLM_OFFLINE_MODE", "true")


def test_context_only_has_drifted_features_with_plain_meanings():
    ctx = alert_context(make_report([HOURS, QUIET]))
    assert [f["feature"] for f in ctx["drifted_features"]] == ["hours_per_week"]
    f = ctx["drifted_features"][0]
    assert f["meaning"] == "hours worked per week" and f["drift_score_name"] == "PSI"
    assert f["current_mean"] == 55.1


def test_template_mentions_the_numbers():
    out = template_explanation(make_report([HOURS]))
    assert "55.1" in out["summary"] and "40.3" in out["summary"]
    assert out["source"] == "template"


def test_template_spots_blank_fields():
    out = template_explanation(make_report([BLANK]))
    assert "43%" in out["summary"]
    assert "pipeline" in out["likely_cause"]


def test_offline_mode_uses_template():
    assert explain_alert(make_report([HOURS]))["source"] == "template"


def test_openai_answer_is_used_and_prompt_is_grounded():
    answer = Explanation(summary="Hours jumped to 55.", likely_cause="Maybe a new source.",
                         suggested_action="Check the pipeline.")
    client = FakeClient(answer=answer)
    out = explain_alert(make_report([HOURS]), client=client)
    assert out == {**answer.model_dump(), "source": "openai"}
    sent = client.calls[0]
    assert sent["response_format"] is Explanation
    assert "55.1" in sent["messages"][1]["content"]          # the real numbers are in the prompt
    assert "ONLY the numbers" in sent["messages"][0]["content"]


def test_api_error_falls_back_to_template():
    error = openai.APIConnectionError(request=SimpleNamespace(method="POST", url="x"))
    out = explain_alert(make_report([HOURS]), client=FakeClient(error=error))
    assert out["source"].startswith("template (AI unavailable")
    assert "55.1" in out["summary"]


def test_refusal_falls_back_to_template():
    out = explain_alert(make_report([HOURS]), client=FakeClient(answer=None))
    assert out["source"].startswith("template")


def test_no_alert_never_calls_the_api():
    client = FakeClient(error=AssertionError("should not be called"))
    out = explain_alert(make_report([QUIET], alert=False), client=client)
    assert client.calls == [] and "No drift" in out["summary"]


def test_every_feature_has_a_plain_meaning():
    from deploymate.data.load import FEATURES
    assert set(FEATURES) <= set(explain.FEATURE_MEANINGS)
