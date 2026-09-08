"""Deterministic floor: normalisation, number grounding, quote alignment, proof of read."""

import types

import pytest

from alpharidge_ai.oracle import floor


def claim(value, unit="none", metric="revenue"):
    return types.SimpleNamespace(metric_name=metric, value=value, unit=unit)


def quote(text, start=None, end=None):
    return types.SimpleNamespace(text=text, start_offset=start, end_offset=end)


def asset(symbol, evidence):
    return types.SimpleNamespace(symbol=symbol, evidence_span=evidence)


def intel(claims=(), quotes=(), assets=()):
    return types.SimpleNamespace(numeric_claims=list(claims), quotes=list(quotes),
                                 assets=list(assets))


# ---- normalisation ----------------------------------------------------------------

def test_normalisation_folds_quotes_dashes_and_whitespace():
    n = floor.normalize('The  CEO said “we’re up” — a lot.')
    assert n.text == 'the ceo said "we\'re up" - a lot.'


def test_offsets_round_trip_to_the_original_text():
    original = "Alpha   BETA gamma"
    n = floor.normalize(original)
    i = n.text.index("beta")
    start, end = n.to_original(i, i + 4)
    assert original[start:end] == "BETA"


def test_empty_text_is_handled():
    assert floor.normalize(None).text == ""
    assert floor.normalize("   ").text == ""


# ---- number parsing ---------------------------------------------------------------

@pytest.mark.parametrize("text,value,unit", [
    ("revenue of $1.2 billion this year", 1.2e9, "currency:USD"),
    ("shares fell 12.5% today", 12.5, "pct"),
    ("a headcount of 1,203 people", 1203.0, "count"),
    ("profit of 450 million usd", 450e6, "currency:USD"),
    ("guidance cut to 3.4bn", 3.4e9, "count"),
])
def test_parses_magnitude_and_unit(text, value, unit):
    got = floor.parse_numbers(floor.normalize(text).text)
    assert any(n.value == pytest.approx(value) and n.unit == unit for n in got), got


def test_grouped_digits_are_one_number():
    got = floor.parse_numbers(floor.normalize("revenue was 1,203,400,000 dollars").text)
    assert any(n.value == pytest.approx(1_203_400_000.0) for n in got)


# ---- grounding --------------------------------------------------------------------

def test_exact_value_is_grounded():
    numbers = floor.parse_numbers(floor.normalize("revenue of $1.2 billion").text)
    assert floor.ground_claim(claim(1.2e9, "USD"), numbers)


def test_rounded_claim_grounds_against_the_full_number():
    numbers = floor.parse_numbers(floor.normalize("revenue was $1,203,400,000").text)
    assert floor.ground_claim(claim(1.2e9, "USD"), numbers)


def test_a_number_not_in_the_text_is_ungrounded():
    numbers = floor.parse_numbers(floor.normalize("revenue of $1.2 billion").text)
    assert not floor.ground_claim(claim(9.9e9, "USD"), numbers)


def test_tolerance_boundary():
    numbers = floor.parse_numbers(floor.normalize("it rose 100 points").text)
    assert floor.ground_claim(claim(100.4, "count"), numbers)     # within 0.5%
    assert not floor.ground_claim(claim(107.0, "count"), numbers)


def test_unit_class_must_match():
    numbers = floor.parse_numbers(floor.normalize("shares fell 12.5%").text)
    assert floor.ground_claim(claim(12.5, "%"), numbers)
    assert not floor.ground_claim(claim(12.5, "USD"), numbers)


def test_percentage_claim_needs_a_percent_token():
    numbers = floor.parse_numbers(floor.normalize("headcount of 12.5 thousand").text)
    assert not floor.ground_claim(claim(12.5, "%"), numbers)


# ---- quotes -----------------------------------------------------------------------

ARTICLE = ('The chief executive said "we expect margins to recover in the second half" '
           'during the call, and the analyst disagreed.')


def test_exact_quote_aligns():
    art = floor.normalize(ARTICLE)
    hit = floor.align_quote(art, "we expect margins to recover in the second half", None, None)
    assert hit is not None and hit.ratio == 1.0
    assert ARTICLE[hit.start:hit.end] == "we expect margins to recover in the second half"


def test_near_miss_quote_aligns_above_the_ratio():
    art = floor.normalize(ARTICLE)
    hit = floor.align_quote(art, "we expect margin to recover in the second half", None, None)
    assert hit is not None and hit.ratio >= floor.QUOTE_MATCH_RATIO


def test_invented_quote_does_not_align():
    art = floor.normalize(ARTICLE)
    assert floor.align_quote(art, "we will double the dividend next quarter", None, None) is None


def test_quote_key_is_the_maximal_match_not_the_submitted_offsets():
    """Two submissions of the same sentence key to the same span."""
    art = floor.normalize(ARTICLE)
    text = "we expect margins to recover in the second half"
    a = floor.align_quote(art, text, None, None)
    b = floor.align_quote(art, text, 5, 200)
    assert (a.start, a.end) == (b.start, b.end)


def test_overlong_quote_is_rejected():
    art = floor.normalize(ARTICLE)
    assert floor.align_quote(art, "x" * (floor.QUOTE_MAX_CHARS + 1), None, None) is None


def test_sliding_windows_of_one_sentence_collapse_to_one_quote():
    """Overlapping restatements must not mint extra scoreable entries."""
    res = floor.evaluate(
        intel(quotes=[quote("we expect margins to recover in the second half"),
                      quote("expect margins to recover in the second"),
                      quote("margins to recover in the second half")]),
        ARTICLE)
    assert len(res.aligned_quotes) == 1
    assert len(res.rejected_quotes) == 2


# ---- evidence spans ---------------------------------------------------------------

def test_bare_ticker_is_not_evidence():
    art = floor.normalize("NVDA rose sharply after the report was published.")
    assert not floor.span_supported(art, "NVDA", "NVDA")


def test_span_with_context_is_evidence():
    art = floor.normalize("NVDA rose sharply after the report was published.")
    assert floor.span_supported(art, "NVDA rose sharply after", "NVDA")


def test_span_absent_from_the_article_fails():
    art = floor.normalize("NVDA rose sharply after the report.")
    assert not floor.span_supported(art, "NVDA collapsed overnight", "NVDA")


def test_span_failure_is_scoped_to_the_asset():
    res = floor.evaluate(
        intel(assets=[asset("NVDA", "NVDA"), asset("AMD", "AMD gained ground too")]),
        "NVDA rose sharply. AMD gained ground too.")
    assert res.floor_pass
    assert res.span_failures == {0}


# ---- proof of read ----------------------------------------------------------------

def test_content_hash_mismatch_fails_the_article():
    res = floor.evaluate(intel(), "the real article text", claimed_hash="0" * 64)
    assert not res.floor_pass and res.reason == "content_hash_mismatch"


def test_matching_hash_passes():
    text = "the real article text"
    res = floor.evaluate(intel(), text, claimed_hash=floor.content_hash(text))
    assert res.floor_pass


def test_text_stats_outside_tolerance_fail_the_article():
    text = "one two three four five six seven eight nine ten " * 20
    stats = floor.text_stats(text)
    stats["words"] = int(stats["words"] * 1.5)
    res = floor.evaluate(intel(), text, claimed_stats=stats)
    assert not res.floor_pass and res.reason == "text_stats_mismatch"


def test_text_stats_within_tolerance_pass():
    text = "one two three four five six seven eight nine ten " * 20
    res = floor.evaluate(intel(), text, claimed_stats=floor.text_stats(text))
    assert res.floor_pass


# ---- end to end -------------------------------------------------------------------

def test_evaluate_splits_grounded_from_ungrounded():
    text = 'Revenue rose to $1.2 billion, up 12.5%, the CEO said "growth is broadening".'
    res = floor.evaluate(
        intel(claims=[claim(1.2e9, "USD"), claim(12.5, "%"), claim(88.0, "USD")],
              quotes=[quote("growth is broadening")]),
        text)
    assert res.floor_pass
    assert res.grounded == {0, 1}
    assert res.ungrounded == {2}
    assert len(res.aligned_quotes) == 1


def test_claim_cap_bounds_the_scoreable_set():
    text = "the figure was 5 in the report"
    res = floor.evaluate(intel(claims=[claim(5.0, "count")] * 100), text, claim_cap=10)
    assert len(res.grounded) + len(res.ungrounded) == 10


def test_empty_article_fails():
    assert not floor.evaluate(intel(), "").floor_pass


def test_is_deterministic():
    text = 'Revenue rose to $1.2 billion, the CEO said "growth is broadening".'
    sub = intel(claims=[claim(1.2e9, "USD")], quotes=[quote("growth is broadening")])
    first = floor.evaluate(sub, text)
    for _ in range(5):
        again = floor.evaluate(sub, text)
        assert again.grounded == first.grounded
        assert again.aligned_quotes == first.aligned_quotes
