"""The grader must share the submission's judgment vocabulary.

keeper_agreement scores the four judgment fields by exact string match. An
unconstrained grader answers in prose ("neutral to positive") and every field
scores zero for every miner, which is what shipped and what this guards.
"""
from alpharidge_ai.models.article_intelligence import (
    Sentiment, ImpactPotential, Urgency, ArticleContentType)
from alpharidge_ai.oracle import grader


def _enum_of(field):
    return grader.JUDGMENT_TOOL["function"]["parameters"]["properties"][field].get("enum")


def test_every_judgment_field_is_constrained():
    for field in ("overall_sentiment", "impact_potential", "urgency", "content_type"):
        assert _enum_of(field), f"{field} has no enum constraint"


def test_the_vocabulary_matches_the_submission_schema():
    assert set(_enum_of("overall_sentiment")) == {e.value for e in Sentiment}
    assert set(_enum_of("impact_potential")) == {e.value for e in ImpactPotential}
    assert set(_enum_of("urgency")) == {e.value for e in Urgency}
    assert set(_enum_of("content_type")) == {e.value for e in ArticleContentType}


def test_the_prompt_pins_assets_to_tickers():
    assert "ticker" in grader.JUDGMENT_PROMPT.lower()


def test_a_prose_answer_would_have_scored_zero():
    """Why this matters, stated as a test rather than a comment."""
    from alpharidge_ai.mechanism import scoring
    mine = {"overall_sentiment": "neutral", "impact_potential": "negligible",
            "urgency": "evergreen", "content_type": "other"}
    prose = {"overall_sentiment": "neutral to positive", "impact_potential": "moderate",
             "urgency": "low", "content_type": "energy industry commentary"}
    assert scoring.keeper_agreement(mine, prose) == 0.0
    assert scoring.keeper_agreement(mine, dict(mine)) == 1.0
