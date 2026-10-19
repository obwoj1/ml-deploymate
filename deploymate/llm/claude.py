"""Claude (Anthropic API) provider for alert explanations.

Used by deploymate/llm/explain.py when ANTHROPIC_API_KEY is set. Returns the
same Explanation fields as the OpenAI path, or raises so explain.py can fall
back to the template.
"""

from __future__ import annotations

import os

import anthropic

MODEL = "claude-opus-5-5"


def make_client():
    """Anthropic client, or None when no key is set."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None
    return anthropic.Anthropic(timeout=30, max_retries=1)


def explain(client, system_prompt: str, user_content: str, schema):
    """One structured-output call. Returns a `schema` instance.

    - effort "low": a short, well-specified writing task; no deep reasoning needed
    - fallbacks "default": if a safety classifier declines, the API retries on a
      fallback model inside the same call
    """
    response = client.beta.messages.parse(
        model=os.getenv("ANTHROPIC_MODEL", MODEL),
        max_tokens=16000,
        system=system_prompt,
        messages=[{"role": "user", "content": user_content}],
        output_format=schema,
        output_config={"effort": "low"},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if response.stop_reason == "refusal" or response.parsed_output is None:
        raise ValueError(f"Claude returned no explanation (stop_reason={response.stop_reason})")
    return response.parsed_output
