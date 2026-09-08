"""Adjudication: deterministic tiers first, a checked model call only for the rest."""

import types

import pytest

from alpharidge_ai.oracle import audit, floor

ARTICLE = ('Revenue rose to $1.2 billion in the quarter, up 12.5% year on year. '
           'The company said margins held steady near 40%.')


def claim(metric, value, unit="USD"):
    return types.SimpleNamespace(metric_name=metric, value=value, unit=unit,
                                 context="")


def run(miner, grader=(), adjudicator=None, article=ARTICLE):
    result = floor.evaluate(types.SimpleNamespace(numeric_claims=list(miner),
                                                  quotes=[], assets=[]), article)
    return audit.adjudicate(miner, grader, result.grounded, article, adjudicator)


# ---- tier 1: the floor already settled it -----------------------------------------

def test_a_grounded_claim_needs_no_model():
    got = run([claim("revenue", 1.2e9)])
    assert got.valid == {("m", 0)}
    assert got.residual == []
    assert got.llm_calls == 0


def test_an_ungrounded_claim_becomes_residual():
    got = run([claim("revenue", 8.8e9)])
    assert got.residual == [0]
    assert got.valid == set()


# ---- tier 2: the reference run found the same claim --------------------------------

def test_a_claim_matching_the_reference_is_settled_without_a_call():
    got = run([claim("revenue", 1.2e9)], grader=[claim("revenue", 1.2e9)])
    assert got.miner_keys == [("g", 0)]
    assert got.residual == []
    assert got.llm_calls == 0


def test_matching_tolerates_small_value_differences():
    assert audit.claims_match(claim("revenue", 1.2e9), claim("revenue", 1.2004e9))
    assert not audit.claims_match(claim("revenue", 1.2e9), claim("revenue", 1.9e9))


def test_matching_requires_the_same_metric_and_unit():
    assert not audit.claims_match(claim("revenue", 12.5), claim("margin", 12.5))
    assert not audit.claims_match(claim("revenue", 12.5, "USD"),
                                  claim("revenue", 12.5, "%"))


def test_each_reference_claim_is_matched_at_most_once():
    """Repeating one true claim must not collect credit twice."""
    got = run([claim("revenue", 1.2e9), claim("revenue", 1.2e9)],
              grader=[claim("revenue", 1.2e9)])
    assert got.miner_keys.count(("g", 0)) == 1


def test_duplicates_cannot_push_recall_above_one():
    from alpharidge_ai.mechanism import scoring
    got = run([claim("revenue", 1.2e9)] * 5, grader=[claim("revenue", 1.2e9)])
    score = scoring.article_score(got.miner_keys, got.grader_keys, got.valid, {})
    assert score.recall <= 1.0
    assert score.precision <= 1.0


# ---- tier 3: the checked model call ------------------------------------------------

def test_the_residual_goes_out_in_one_batched_call():
    seen = []

    def adjudicator(text, claims):
        seen.append(claims)
        return [{"i": c["i"], "supported": False, "evidence": ""} for c in claims]

    got = run([claim("a", 8.8e9), claim("b", 7.7e9), claim("c", 6.6e9)],
              adjudicator=adjudicator)
    assert got.llm_calls == 1
    assert len(seen) == 1
    assert [c["i"] for c in seen[0]] == [0, 1, 2]


def test_a_grader_cannot_validate_by_asserting():
    """Evidence has to be in the article, on the same alignment the floor uses."""
    def liar(text, claims):
        return [{"i": c["i"], "supported": True,
                 "evidence": "the company reported eight point eight billion"}
                for c in claims]

    got = run([claim("revenue", 8.8e9)], adjudicator=liar)
    assert got.valid == set()


def test_evidence_that_is_in_the_article_is_accepted():
    def honest(text, claims):
        return [{"i": c["i"], "supported": True,
                 "evidence": "margins held steady near 40%"} for c in claims]

    got = run([claim("margin", 41.0, "%")], adjudicator=honest)
    assert got.valid == {("m", 0)}


def test_an_unsupported_verdict_is_respected():
    def honest(text, claims):
        return [{"i": c["i"], "supported": False,
                 "evidence": "margins held steady near 40%"} for c in claims]
    assert run([claim("margin", 41.0, "%")], adjudicator=honest).valid == set()


def test_no_adjudicator_leaves_the_residual_unsupported():
    got = run([claim("revenue", 8.8e9)])
    assert got.valid == set() and got.llm_calls == 0


def test_a_failing_adjudicator_does_not_break_the_article():
    def broken(text, claims):
        raise RuntimeError("model unavailable")
    got = run([claim("revenue", 8.8e9)], adjudicator=broken)
    assert got.valid == set()
    assert got.miner_keys == [("m", 0)]


# ---- reply parsing -----------------------------------------------------------------

def test_parses_a_json_string_reply():
    got = audit.parse_adjudication('[{"i": 1, "supported": true, "evidence": "x"}]', [1])
    assert got[1]["supported"] is True


def test_parses_a_wrapped_reply():
    got = audit.parse_adjudication({"claims": [{"i": 0, "supported": True}]}, [0])
    assert got[0]["supported"] is True


def test_unparseable_replies_yield_nothing():
    for bad in ("not json", None, 42, [{"no_index": 1}]):
        assert audit.parse_adjudication(bad, [0, 1]) == {}


def test_indexes_outside_the_request_are_ignored():
    got = audit.parse_adjudication([{"i": 99, "supported": True, "evidence": "x"}], [0])
    assert got == {}


# ---- quotes ------------------------------------------------------------------------

def test_quote_keys_come_from_the_matched_span():
    text = 'The CEO said "we expect margins to recover" on Monday.'
    grader_quote = types.SimpleNamespace(text="we expect margins to recover")
    keys = audit.grader_quote_keys(text, [grader_quote])
    assert len(keys) == 1
    start, end = next(iter(keys))
    assert text[start:end] == "we expect margins to recover"


def test_a_quote_not_in_the_article_yields_no_key():
    assert audit.grader_quote_keys(ARTICLE, [types.SimpleNamespace(
        text="we will double the dividend")]) == set()


def test_miner_and_grader_key_the_same_sentence_identically():
    text = 'The CEO said "we expect margins to recover" on Monday.'
    grader = audit.grader_quote_keys(text, [types.SimpleNamespace(
        text="we expect margins to recover")])
    result = floor.evaluate(types.SimpleNamespace(
        numeric_claims=[], assets=[],
        quotes=[types.SimpleNamespace(text="expect margins to recover",
                                      start_offset=None, end_offset=None)]), text)
    assert set(audit.quote_keys(result.aligned_quotes)) & grader
