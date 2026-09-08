"""The grader: the only model calls the audit makes.

Two calls, both against a model drawn from the rotation by the article's key:

  adjudicate  the claims the deterministic tiers could not settle, batched once
              per article
  judge       the judgment fields, on articles that carry no claims to check

The grader's own claim set does not come from here. It is the validator's existing
reference analysis, which is already computed for every article it validates.

Forced tool calls at temperature zero. Models that refuse a forced tool call cannot be
used for this and are excluded from the rotation rather than worked around.
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional

import bittensor as bt

from alpharidge_ai import config

MAX_TOKENS = 2000
RETRIES = 1

ADJUDICATION_TOOL = {
    "type": "function",
    "function": {
        "name": "report_claims",
        "description": "Report whether the article supports each claim.",
        "parameters": {
            "type": "object",
            "properties": {
                "claims": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "i": {"type": "integer"},
                            "supported": {"type": "boolean"},
                            "evidence": {
                                "type": "string",
                                "description": ("A span copied verbatim from the "
                                                "article. Empty when unsupported."),
                            },
                        },
                        "required": ["i", "supported", "evidence"],
                    },
                }
            },
            "required": ["claims"],
        },
    },
}

JUDGMENT_TOOL = {
    "type": "function",
    "function": {
        "name": "report_judgment",
        "description": "Classify the article.",
        "parameters": {
            "type": "object",
            "properties": {
                "overall_sentiment": {"type": "string"},
                "impact_potential": {"type": "string"},
                "urgency": {"type": "string"},
                "content_type": {"type": "string"},
                "assets": {"type": "array", "items": {"type": "string"}},
                "entities": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["overall_sentiment", "impact_potential", "urgency",
                         "content_type"],
        },
    },
}

ADJUDICATION_PROMPT = (
    "Check each numeric claim against the article below.\n\n"
    "Dates, times, years and article timestamps are not numeric claims; mark any of "
    "them unsupported.\n"
    "A claim is supported only if the article states it. Copy the supporting span "
    "verbatim into `evidence`; a span that is not in the article does not count, so "
    "do not paraphrase it. Do not compute, infer, or use outside knowledge: a figure "
    "you derived is not a figure the article stated.\n\n"
    "ARTICLE:\n{article}\n\nCLAIMS:\n{claims}\n"
)

JUDGMENT_PROMPT = (
    "Classify the article below.\n\n"
    "ARTICLE:\n{article}\n"
)

ARTICLE_CHARS = 12000


class Grader:
    """Calls a named model. The model is chosen per article by the audit key."""

    def __init__(self, client=None):
        self._client = client

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(base_url=config.LLM_BASE, api_key=config.API_KEY)
        return self._client

    def _call(self, model: str, prompt: str, tool: dict) -> Optional[dict]:
        name = tool["function"]["name"]
        for attempt in range(RETRIES + 1):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    tools=[tool],
                    tool_choice={"type": "function", "function": {"name": name}},
                    temperature=0,
                    max_tokens=MAX_TOKENS,
                )
                calls = response.choices[0].message.tool_calls
                if not calls:
                    continue
                return json.loads(calls[0].function.arguments)
            except Exception as e:
                if attempt >= RETRIES:
                    bt.logging.warning(f"[GRADER] {model} {name} failed: {e}")
        return None

    def adjudicate(self, article_text: str, claims: List[dict], model: str) -> List[dict]:
        """Verdicts on the residual claims. An empty result leaves them unsupported."""
        if not claims or not model:
            return []
        prompt = ADJUDICATION_PROMPT.format(
            article=(article_text or "")[:ARTICLE_CHARS],
            claims=json.dumps(claims, ensure_ascii=False))
        result = self._call(model, prompt, ADJUDICATION_TOOL)
        return (result or {}).get("claims") or []

    def judge(self, article_text: str, model: str) -> Dict:
        """Judgment fields for the keeper path."""
        if not model:
            return {}
        prompt = JUDGMENT_PROMPT.format(article=(article_text or "")[:ARTICLE_CHARS])
        return self._call(model, prompt, JUDGMENT_TOOL) or {}
