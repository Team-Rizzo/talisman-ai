"""Per-article volume credit: the floor sweep, and the flag that gates on it."""

import types

import pytest

import alpharidge_ai.config as config
from alpharidge_ai.analyzer import scoring
from alpharidge_ai.models import article_intelligence as ai
from tests.test_cheat_probe import make_reference_intel

TITLE = "Quarterly results"
TEXT = ('Revenue rose to $1.2 billion in the quarter, up 12.5% year on year. '
        'The company said margins held steady near 40%.')


def _payload(claims=(), content_hash=None):
    payload = make_reference_intel(1).model_dump(mode="json")
    payload["numeric_claims"] = list(claims)
    payload["event_fingerprint"]["content_hash"] = (
        content_hash if content_hash is not None
        else ai.ArticleIntelligence.compute_content_hash(TITLE, TEXT))
    return payload


def article(article_id, blob, content=TEXT, title=TITLE):
    return types.SimpleNamespace(
        id=article_id, content=content, title=title,
        analysis=types.SimpleNamespace(analysis_data=blob))


def claim(value, unit="USD"):
    return {"metric_name": "revenue", "value": value, "unit": unit, "confidence": 0.9}


# ---- the sweep --------------------------------------------------------------------

def test_the_sweep_covers_every_article_not_just_a_sample():
    batch = [article(i, _payload([claim(1.2e9)])) for i in range(1, 33)]
    results = scoring._floor_sweep(batch)
    assert len(results) == 32
    assert all(results.values())


def test_an_unloadable_analysis_fails_the_floor():
    batch = [article(1, {"not": "an analysis"})]
    assert scoring._floor_sweep(batch) == {1: False}


def test_a_missing_analysis_fails_the_floor():
    batch = [article(1, None)]
    assert scoring._floor_sweep(batch) == {1: False}


def test_a_bad_proof_of_read_fails_the_article():
    batch = [article(1, _payload([claim(1.2e9)], content_hash="0" * 64))]
    assert scoring._floor_sweep(batch) == {1: False}


def test_an_article_with_no_reference_text_is_left_out():
    """The validator's own gap must not be charged to the sender."""
    batch = [article(1, _payload([claim(1.2e9)]), content=None)]
    assert scoring._floor_sweep(batch) == {}


def test_the_sweep_prefers_the_validators_own_copy_of_the_text():
    submitted = article(1, _payload([claim(1.2e9)]), content="unrelated filler text")
    reference = types.SimpleNamespace(id=1, content=TEXT, title=TITLE)
    results = scoring._floor_sweep([submitted], {"1": reference})
    assert results == {1: True}


def test_the_sweep_never_raises_on_a_malformed_article():
    batch = [types.SimpleNamespace(id="not-an-int", content=TEXT, title=TITLE,
                                   analysis=None)]
    assert scoring._floor_sweep(batch) == {}


# ---- the flag ---------------------------------------------------------------------

def test_gating_is_a_profile_field_and_defaults_on():
    """Unlike the other switches this withholds pay for work that failed a check, so it
    does not wait for the flip. Turning it off is possible but must be deliberate."""
    from alpharidge_ai.mechanism import profile as mp
    from tests.test_profile_client import valid
    assert mp.parse(valid()).settlement.floor_gating is True

    raw = valid()
    raw["settlement"]["floor_gating"] = False
    assert mp.parse(raw).settlement.floor_gating is False


def test_gating_has_no_local_switch():
    """One source, so two validators cannot credit volume differently."""
    assert not hasattr(config, "FLOOR_GATING_ENABLED")
    assert "FLOOR_GATING_ENABLED" not in config._REMOTE_CONFIG_KEYS


# ---- the credit decision ----------------------------------------------------------

def _credited(floor_results, gating, article_ids):
    """Mirror of the credit condition in the dispatch path."""
    return [aid for aid in article_ids
            if not (gating and floor_results.get(aid) is False)]


def test_while_gating_is_off_every_article_in_a_passing_batch_is_credited():
    results = {1: True, 2: False, 3: True}
    assert _credited(results, False, [1, 2, 3]) == [1, 2, 3]


def test_with_gating_on_only_articles_that_clear_the_floor_are_credited():
    results = {1: True, 2: False, 3: True}
    assert _credited(results, True, [1, 2, 3]) == [1, 3]


def test_an_article_the_sweep_skipped_is_still_credited():
    """Absence of a floor result is not a failure."""
    assert _credited({1: True}, True, [1, 2]) == [1, 2]
