"""Keyed selection and grader rotation."""

import types

import pytest

from alpharidge_ai.oracle import selector as sel

KEY_A = b"validator-a-secret-key"
KEY_B = b"validator-b-secret-key"

NUMBERS = "Revenue rose to $1.2 billion in the quarter, up 12 percent."
PLAIN = "The company announced a change to its leadership team this week."
QUOTED = 'The chief executive said "we are pleased with the outcome" on Monday.'

MODELS = [types.SimpleNamespace(id="model-a", weight=0.7),
          types.SimpleNamespace(id="model-b", weight=0.3)]

TIERS = [sel.TIER_NUMBER_BEARING, sel.TIER_QUOTE_BEARING]


def _select(selector, article_id, text, pool_rate=0.9, keeper_rate=0.03):
    return selector.select(article_id, text, pool_tiers=TIERS,
                           keyed_rate_pool=pool_rate, keyed_rate_keeper=keeper_rate,
                           grader_models=MODELS)


# ---- pool membership --------------------------------------------------------------

def test_number_bearing_articles_are_in_the_pool():
    assert sel.in_pool(NUMBERS, TIERS)


def test_quote_bearing_articles_are_in_the_pool():
    assert sel.in_pool(QUOTED, TIERS)


def test_plain_articles_are_outside_the_pool():
    assert not sel.in_pool(PLAIN, TIERS)


def test_empty_article_is_outside_the_pool():
    assert not sel.in_pool("", TIERS)
    assert not sel.in_pool(None, TIERS)


# ---- keying -----------------------------------------------------------------------

def test_selection_is_reproducible_for_the_same_validator():
    s = sel.Selector(KEY_A)
    first = [_select(s, i, NUMBERS).graded for i in range(200)]
    second = [_select(s, i, NUMBERS).graded for i in range(200)]
    assert first == second


def test_two_validators_draw_different_articles():
    a, b = sel.Selector(KEY_A), sel.Selector(KEY_B)
    picks_a = {i for i in range(500) if _select(a, i, NUMBERS, pool_rate=0.5).graded}
    picks_b = {i for i in range(500) if _select(b, i, NUMBERS, pool_rate=0.5).graded}
    assert picks_a != picks_b


def test_the_overlap_between_validators_is_large_at_a_high_rate():
    """Cross-validator auditing needs the intersection to stay substantial."""
    a, b = sel.Selector(KEY_A), sel.Selector(KEY_B)
    ids = range(3000)
    picks_a = {i for i in ids if _select(a, i, NUMBERS).graded}
    picks_b = {i for i in ids if _select(b, i, NUMBERS).graded}
    overlap = len(picks_a & picks_b) / len(picks_a | picks_b)
    assert overlap > 0.75


def test_keyed_rates_are_honoured():
    s = sel.Selector(KEY_A)
    ids = range(5000)
    pool_rate = sum(_select(s, i, NUMBERS).graded for i in ids) / 5000
    keeper_rate = sum(_select(s, i, PLAIN).graded for i in ids) / 5000
    assert pool_rate == pytest.approx(0.90, abs=0.02)
    assert keeper_rate == pytest.approx(0.03, abs=0.01)


def test_selection_splits_the_two_paths():
    s = sel.Selector(KEY_A)
    pooled = [_select(s, i, NUMBERS) for i in range(300)]
    keepers = [_select(s, i, PLAIN) for i in range(300)]
    assert all(x.slice == sel.POOL for x in pooled if x.graded)
    assert all(x.slice == sel.KEEPER for x in keepers if x.graded)


def test_a_zero_rate_grades_nothing():
    s = sel.Selector(KEY_A)
    assert not any(_select(s, i, PLAIN, keeper_rate=0.0).graded for i in range(500))


def test_selection_ignores_everything_but_the_article_id():
    """Only the article id feeds the draw."""
    s = sel.Selector(KEY_A)
    base = _select(s, 4242, NUMBERS)
    for text in (NUMBERS, NUMBERS + " Extra trailing sentence about the quarter."):
        assert _select(s, 4242, text).graded == base.graded


# ---- grader rotation --------------------------------------------------------------

def test_grader_draw_is_reproducible():
    s = sel.Selector(KEY_A)
    assert s.draw_grader(99, MODELS) == s.draw_grader(99, MODELS)


def test_grader_draw_respects_weights():
    s = sel.Selector(KEY_A)
    picks = [s.draw_grader(i, MODELS) for i in range(4000)]
    share_a = picks.count("model-a") / len(picks)
    assert share_a == pytest.approx(0.7, abs=0.03)


def test_grader_draw_is_independent_of_the_selection_draw():
    s = sel.Selector(KEY_A)
    graded = [i for i in range(2000) if _select(s, i, NUMBERS, pool_rate=0.5).graded]
    picks = [s.draw_grader(i, MODELS) for i in graded]
    assert len(set(picks)) > 1


def test_grader_draw_with_no_models_is_empty():
    assert sel.Selector(KEY_A).draw_grader(1, []) == ""


def test_selected_articles_carry_a_grader():
    s = sel.Selector(KEY_A)
    picked = [_select(s, i, NUMBERS) for i in range(50)]
    assert all(p.grader_model for p in picked if p.graded)


# ---- key handling -----------------------------------------------------------------

def test_key_is_not_exposed_by_repr():
    s = sel.Selector(b"super-secret")
    assert "secret" not in repr(s)


def test_empty_key_is_refused():
    with pytest.raises(ValueError):
        sel.Selector(b"")
