"""The reference miner emits what 1.2.0 requires, and passes its own gate."""

import pytest

from alpharidge_ai.analyzer.article_intelligence_analyzer import (
    ArticleIntelligenceAnalyzer as A)
from alpharidge_ai.models import article_intelligence as ai
from alpharidge_ai.oracle import floor, schema_gate

TEXT = ('Revenue rose to $1.2 billion in the quarter. The chief executive said '
        '"margins should recover in the second half" on Monday.')


def claims(raw):
    return A._build_numeric_claims(A, raw)


def quotes(raw, content=TEXT):
    return A._build_quotes(A, raw, content)


# ---- confidence --------------------------------------------------------------------

def test_a_stated_claim_confidence_is_carried():
    c = claims([{"metric_name": "revenue", "value": 1.2e9, "unit": "USD",
                 "confidence": 0.93}])[0]
    assert c.confidence == pytest.approx(0.93)


def test_a_missing_confidence_becomes_neutral_not_optimistic():
    c = claims([{"metric_name": "revenue", "value": 1.2e9, "unit": "USD"}])[0]
    assert c.confidence == pytest.approx(0.5)


@pytest.mark.parametrize("bad", [-1.0, 2.0, "high", None])
def test_an_unusable_confidence_is_clamped_or_neutral(bad):
    c = claims([{"metric_name": "m", "value": 1.0, "unit": "USD",
                 "confidence": bad}])[0]
    assert 0.0 <= c.confidence <= 1.0


def test_quote_confidence_is_carried():
    q = quotes([{"speaker": "CEO", "text": "margins should recover in the second half",
                 "confidence": 0.8}])[0]
    assert q.confidence == pytest.approx(0.8)


# ---- offsets -----------------------------------------------------------------------

def test_quote_offsets_locate_the_quote_in_the_served_text():
    q = quotes([{"speaker": "CEO", "text": "margins should recover in the second half",
                 "confidence": 0.9}])[0]
    assert TEXT[q.start_offset:q.end_offset] == \
        "margins should recover in the second half"


def test_offsets_are_computed_not_taken_from_the_model():
    """A model-supplied offset is ignored; the text decides."""
    q = quotes([{"speaker": "CEO", "text": "margins should recover in the second half",
                 "confidence": 0.9, "start_offset": 9999, "end_offset": 10000}])[0]
    assert TEXT[q.start_offset:q.end_offset].startswith("margins should recover")


def test_a_quote_not_in_the_article_is_not_emitted():
    """It cannot carry offsets, and a 1.2.0 submission missing them fails the cutover
    gate for the whole article. Dropping the quote costs less than the article."""
    assert quotes([{"speaker": "CEO", "text": "we will double the dividend",
                    "confidence": 0.9}]) == []


def test_every_emitted_quote_carries_offsets():
    out = quotes([{"speaker": "CEO", "text": "margins should recover in the second half",
                   "confidence": 0.9},
                  {"speaker": "CFO", "text": "not in the article at all",
                   "confidence": 0.9}])
    assert out and all(q.start_offset is not None and q.end_offset is not None
                       for q in out)


def test_no_article_text_is_survivable():
    """Nothing can be located against nothing, so nothing is emitted."""
    assert quotes([{"speaker": "CEO", "text": "anything", "confidence": 0.5}],
                  content="") == []


def test_the_miner_and_validator_agree_on_the_span():
    """Both sides run the same alignment, so the keys match by construction."""
    q = quotes([{"speaker": "CEO", "text": "margins should recover in the second half",
                 "confidence": 0.9}])[0]
    hit = floor.align_quote(floor.normalize(TEXT), q.text, q.start_offset, q.end_offset)
    assert (hit.start, hit.end) == (q.start_offset, q.end_offset)


# ---- caps --------------------------------------------------------------------------

def test_emission_respects_the_schema_caps():
    many_claims = [{"metric_name": f"m{i}", "value": float(i), "unit": "USD",
                    "confidence": 0.5} for i in range(200)]
    assert len(claims(many_claims)) <= ai.MAX_NUMERIC_CLAIMS

    many_quotes = [{"speaker": "s", "text": f"quote {i}", "confidence": 0.5}
                   for i in range(200)]
    assert len(quotes(many_quotes)) <= ai.MAX_QUOTES


# ---- the gate ----------------------------------------------------------------------

def test_the_reference_miner_passes_its_own_cutover_gate():
    """Before this, the reference implementation claimed 1.2.0 and would have failed."""
    import types
    intel = types.SimpleNamespace(
        numeric_claims=claims([{"metric_name": "revenue", "value": 1.2e9,
                                "unit": "USD", "confidence": 0.93}]),
        quotes=quotes([{"speaker": "CEO",
                        "text": "margins should recover in the second half",
                        "confidence": 0.9}]))
    verdict = schema_gate.evaluate(ai.SCHEMA_VERSION, block=10_000,
                                   cutover_block=1_000, intel=intel)
    assert verdict.accepted and verdict.score_confidence
