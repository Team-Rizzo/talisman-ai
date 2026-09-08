"""Schema 1.2.0 fields, extraction caps, and the grace window."""

import pytest
from pydantic import ValidationError

from alpharidge_ai.models import article_intelligence as ai
from alpharidge_ai.oracle import schema_gate


def test_schema_version_is_bumped():
    assert ai.SCHEMA_VERSION == "1.2.0"
    assert ai.SCHEMA_VERSION_PREV == "1.1.0"


# ---- new fields -------------------------------------------------------------------

def test_numeric_claim_carries_confidence():
    c = ai.NumericClaim(metric_name="revenue", value=1.2e9, unit="USD", confidence=0.93)
    assert c.confidence == 0.93


def test_quote_carries_confidence_and_offsets():
    q = ai.QuoteExtraction(speaker="CEO", text="we are pleased",
                           confidence=0.8, start_offset=10, end_offset=24)
    assert (q.confidence, q.start_offset, q.end_offset) == (0.8, 10, 24)


@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_confidence_is_bounded(bad):
    with pytest.raises(ValidationError):
        ai.NumericClaim(metric_name="m", value=1.0, unit="USD", confidence=bad)


def test_offsets_cannot_be_negative():
    with pytest.raises(ValidationError):
        ai.QuoteExtraction(speaker="s", text="t", start_offset=-1)


def test_older_submissions_still_parse():
    """The grace window needs 1.1.0 payloads to load, not to be rejected outright."""
    c = ai.NumericClaim(metric_name="revenue", value=1.0, unit="USD")
    q = ai.QuoteExtraction(speaker="CEO", text="hello")
    assert c.confidence is None
    assert q.confidence is None and q.start_offset is None


# ---- caps -------------------------------------------------------------------------

def test_extraction_lists_are_capped():
    assert ai.MAX_NUMERIC_CLAIMS == 40
    assert ai.MAX_QUOTES == 20
    fields = ai.ArticleIntelligence.model_fields
    for name, cap in (("assets", ai.MAX_ASSETS),
                      ("entities", ai.MAX_ENTITIES),
                      ("economic_data", ai.MAX_ECONOMIC_DATA),
                      ("numeric_claims", ai.MAX_NUMERIC_CLAIMS),
                      ("quotes", ai.MAX_QUOTES),
                      ("contagion_links", ai.MAX_CONTAGION_LINKS)):
        limits = [m for m in fields[name].metadata if hasattr(m, "max_length")]
        assert limits and limits[0].max_length == cap, name


def test_over_cap_submission_is_rejected():
    """Reporting more than the cap fails validation rather than buying a bigger set."""
    from tests.test_cheat_probe import make_reference_intel
    base = make_reference_intel(1).model_dump(mode="json")

    def with_claims(n):
        payload = dict(base)
        payload["numeric_claims"] = [
            {"metric_name": "m", "value": float(i), "unit": "USD", "confidence": 0.5}
            for i in range(n)]
        return payload

    ai.ArticleIntelligence.model_validate(with_claims(ai.MAX_NUMERIC_CLAIMS))
    with pytest.raises(ValidationError):
        ai.ArticleIntelligence.model_validate(with_claims(ai.MAX_NUMERIC_CLAIMS + 1))


def test_a_current_schema_submission_round_trips():
    from tests.test_cheat_probe import make_reference_intel
    payload = make_reference_intel(2).model_dump(mode="json")
    payload["numeric_claims"] = [{"metric_name": "revenue", "value": 1.2e9,
                                  "unit": "USD", "confidence": 0.93}]
    payload["quotes"] = [{"speaker": "CEO", "text": "we are pleased",
                          "confidence": 0.8, "start_offset": 4, "end_offset": 18}]
    intel = ai.ArticleIntelligence.model_validate(payload)
    assert intel.numeric_claims[0].confidence == 0.93
    assert intel.quotes[0].start_offset == 4
    assert intel.schema_version == "1.2.0"


# ---- grace window -----------------------------------------------------------------

CUTOVER = 5_000_000


def test_current_schema_is_scored_on_confidence():
    v = schema_gate.evaluate("1.2.0", block=1, cutover_block=CUTOVER)
    assert v.accepted and v.score_confidence


def test_previous_schema_is_accepted_before_the_cutover():
    v = schema_gate.evaluate("1.1.0", block=CUTOVER - 1, cutover_block=CUTOVER)
    assert v.accepted and not v.score_confidence and v.reason == "grace"


def test_previous_schema_fails_after_the_cutover():
    v = schema_gate.evaluate("1.1.0", block=CUTOVER, cutover_block=CUTOVER)
    assert not v.accepted


def test_current_schema_keeps_working_after_the_cutover():
    v = schema_gate.evaluate("1.2.0", block=CUTOVER + 10, cutover_block=CUTOVER)
    assert v.accepted and v.score_confidence


def test_unknown_schema_is_refused_at_any_block():
    for block in (0, CUTOVER + 1):
        assert not schema_gate.evaluate("0.9.0", block=block, cutover_block=CUTOVER).accepted
    assert not schema_gate.evaluate(None, block=0, cutover_block=CUTOVER).accepted


# ---- confidence extraction --------------------------------------------------------

def test_confidences_are_read_by_position():
    claims = [ai.NumericClaim(metric_name="m", value=1.0, unit="USD", confidence=0.9),
              ai.NumericClaim(metric_name="m", value=2.0, unit="USD", confidence=0.2)]
    assert schema_gate.confidences(claims) == {0: 0.9, 1: 0.2}


def test_missing_confidences_are_omitted_not_invented():
    claims = [ai.NumericClaim(metric_name="m", value=1.0, unit="USD"),
              ai.NumericClaim(metric_name="m", value=2.0, unit="USD", confidence=0.7)]
    assert schema_gate.confidences(claims) == {1: 0.7}


def test_grace_scoring_matches_the_neutral_baseline():
    """A 1.1.0 submission scores as one that stated 0.5 on everything."""
    from alpharidge_ai.mechanism import scoring
    gold = {"a", "b", "c"}
    graced = scoring.article_score(sorted(gold), gold, (), {}, score_confidence=False)
    neutral = scoring.article_score(sorted(gold), gold, (), {k: 0.5 for k in gold})
    assert graced.normalized == pytest.approx(neutral.normalized)
