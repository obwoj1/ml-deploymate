"""Turn a monitor alert into a plain-English explanation.

    from deploymate.llm.explain import explain_alert
    explain_alert(report)  # report = Monitor.check(...) output
    # -> {"summary", "likely_cause", "suggested_action", "source"}

Uses OpenAI when OPENAI_API_KEY is set; otherwise (or if the call fails) it
falls back to a template built from the same numbers, so the dashboard always
works: for teammates without a key, in CI, and during the live demo.

The model only sees the alert's numbers (no raw rows of data) and is told to
use only those numbers, so it explains what was measured instead of guessing.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

DEFAULT_MODEL = "gpt-4o-mini"

# What each column means, so explanations use plain words.
FEATURE_MEANINGS = {
    "age": "age in years",
    "education_num": "years of education",
    "capital_gain": "investment income",
    "capital_loss": "investment losses",
    "hours_per_week": "hours worked per week",
    "workclass": "type of employer",
    "marital_status": "marital status",
    "occupation": "job type",
    "relationship": "role in household",
    "race": "race",
    "sex": "sex",
    "native_country": "country of origin",
}

SYSTEM_PROMPT = """You explain alerts from DeployMate, a tool that monitors a deployed \
machine-learning model. The model predicts whether a person earns more than $50K/year \
from census data. An alert means a batch of incoming data no longer looks like the \
data the model was trained on (data drift).

Write for an engineer who is not an ML expert. Rules:
- Use ONLY the numbers in the alert. Never invent values, dates, or causes stated as fact.
- Name the changed features in plain words (you're given what each one means).
- likely_cause: give the most plausible real-world reason, worded as a possibility.
- suggested_action: concrete next steps (check the upstream data source, compare with \
recent pipeline changes, retrain on fresh labeled data if the shift is real).
- Each field: 1-2 short sentences. No markdown."""


class Explanation(BaseModel):
    summary: str = Field(description="What changed, with the key numbers")
    likely_cause: str = Field(description="Most plausible real-world reason, as a possibility")
    suggested_action: str = Field(description="Concrete next steps")


def alert_context(report: dict) -> dict:
    """The compact facts the explanation is based on (no raw data rows)."""
    drifted = []
    for d in report["drift"]:
        if not d["drifted"]:
            continue
        drifted.append({
            "feature": d["feature"],
            "meaning": FEATURE_MEANINGS.get(d["feature"], d["feature"]),
            "kind": d["kind"],
            "drift_score": round(d["statistic"], 3),
            "drift_score_name": "PSI" if d["kind"] == "numeric" else "Cramér's V",
            **d.get("shift", {}),
        })
    metrics = report["metrics"]
    return {
        "batch_id": report.get("batch_id"),
        "severity": report["severity"],
        "drifted_features": drifted,
        "model_behavior": {
            "share_predicted_over_50K": metrics["positive_rate"],
            "mean_confidence": metrics["mean_confidence"],
            "share_of_blank_fields": metrics["unknown_rate"],
            "unusual_behavior_flags": report["anomaly"]["flagged_metrics"],
        },
    }


def _describe(f: dict) -> str:
    if "reference_mean" in f:
        return (f"{f['meaning']} averaged {f['current_mean']:g} "
                f"(normally {f['reference_mean']:g})")
    if "category" in f:
        return (f"{f['meaning']} '{f['category']}' made up {f['current_share']:.0%} "
                f"(normally {f['reference_share']:.0%})")
    return f["meaning"]


def template_explanation(report: dict) -> dict:
    """Rule-based explanation from the same facts. No API key needed."""
    ctx = alert_context(report)
    features = ctx["drifted_features"]
    if not features:
        return {"summary": "No drift detected in this batch.", "likely_cause": "-",
                "suggested_action": "No action needed.", "source": "template"}

    changes = "; ".join(_describe(f) for f in features)
    names = ", ".join(f["meaning"] for f in features)
    blank = any(f.get("category") == "Unknown" for f in features)
    return {
        "summary": f"Batch {ctx['batch_id']} doesn't look like the training data: {changes}.",
        "likely_cause": (
            f"Many fields arriving blank suggests a broken upstream data pipeline or form."
            if blank else
            f"The incoming {names} changed, which usually means a new user population, "
            f"a change in how data is collected, or a bug in an upstream data source."),
        "suggested_action": (
            "Check recent changes to the data pipeline and source systems. If the shift is "
            "real and lasting, collect fresh labeled data and retrain the model."),
        "source": "template",
    }


def _client():
    """An OpenAI client, or None when offline (no key, or LLM_OFFLINE_MODE=true)."""
    if os.getenv("LLM_OFFLINE_MODE", "false").lower() == "true":
        return None
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    from openai import OpenAI
    return OpenAI(api_key=key, timeout=20, max_retries=1)


def explain_alert(report: dict, client=None) -> dict:
    """Explain one monitor report. Never raises because of the LLM: falls back to the template."""
    if not report.get("alert"):
        return template_explanation(report)
    client = client or _client()
    if client is None:
        return template_explanation(report)

    import openai
    try:
        completion = client.chat.completions.parse(
            model=os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Alert:\n" + json.dumps(alert_context(report), indent=2)},
            ],
            response_format=Explanation,
            temperature=0.2,
        )
        parsed = completion.choices[0].message.parsed
        if parsed is None:  # model refused or returned nothing usable
            raise ValueError("no parsed explanation")
        return {**parsed.model_dump(), "source": "openai"}
    except (openai.OpenAIError, ValueError) as err:
        fallback = template_explanation(report)
        fallback["source"] = f"template (AI unavailable: {type(err).__name__})"
        return fallback
