"""Overlap audit: comparing two validators on the articles both graded."""

import pytest

from alpharidge_ai.consensus import overlap_audit as oa
from alpharidge_ai.validator.reputation_store import ReputationStore


def scores(n, value=0.8, start=0):
    return {start + i: value for i in range(n)}


# ---- flattening -------------------------------------------------------------------

def test_flatten_collapses_targets_to_articles():
    assert oa.flatten({"miner-a": [(1, 0.9, 1.0)],
                       "miner-b": [(2, 0.4, 1.0)]}) == {1: 0.9, 2: 0.4}


def test_flatten_keeps_the_lower_score_for_a_repeated_article():
    assert oa.flatten({"m": [(1, 0.9, 1.0), (1, 0.3, 1.0)]}) == {1: 0.3}


def test_flatten_skips_malformed_rows():
    assert oa.flatten({"m": [(1, 0.9, 1.0), ("bad",), None]}) == {1: 0.9}


def test_flatten_handles_nothing():
    assert oa.flatten({}) == {} and oa.flatten(None) == {}


# ---- assessment -------------------------------------------------------------------

def test_agreeing_validators_are_not_flagged():
    verdict = oa.assess(scores(100, 0.80), scores(100, 0.81))
    assert verdict.comparable and not verdict.flagged
    assert verdict.median_delta == pytest.approx(0.01)


def test_a_diverging_sender_is_flagged():
    verdict = oa.assess(scores(100, 0.20), scores(100, 0.90))
    assert verdict.comparable and verdict.flagged
    assert "score_divergence" in verdict.reason


def test_a_small_intersection_reaches_no_verdict():
    verdict = oa.assess(scores(5, 0.1), scores(5, 0.9))
    assert not verdict.comparable and not verdict.flagged
    assert "overlap_too_small" in verdict.reason


def test_only_shared_articles_are_compared():
    local = scores(60, 0.8, start=0)
    remote = scores(60, 0.8, start=30)      # 30 shared, 30 each side unshared
    remote.update({i: 0.0 for i in range(90, 120)})
    verdict = oa.assess(local, remote, min_overlap=20)
    assert verdict.overlap == 30
    assert not verdict.flagged


def test_a_few_hard_articles_do_not_convict():
    """The median resists a minority of genuine disagreements."""
    local = scores(100, 0.80)
    remote = dict(local)
    for i in range(20):
        remote[i] = 0.0
    assert not oa.assess(local, remote).flagged


def test_a_wholesale_difference_does_convict():
    local = scores(100, 0.80)
    remote = {i: 0.0 for i in range(100)}
    assert oa.assess(local, remote).flagged


def test_the_threshold_is_configurable():
    local, remote = scores(100, 0.80), scores(100, 0.60)
    assert not oa.assess(local, remote, tau=0.30).flagged
    assert oa.assess(local, remote, tau=0.10).flagged


def test_no_overlap_at_all_is_not_a_flag():
    verdict = oa.assess(scores(50, 0.8, start=0), scores(50, 0.8, start=500))
    assert not verdict.comparable and not verdict.flagged


# ---- the store read -----------------------------------------------------------------

def test_sender_observations_are_readable_per_epoch(tmp_path):
    store = ReputationStore(path=tmp_path / "rep.json")
    store.record_local(7, "self", "target", article_id=1, graded=0.9, weight=1.0)
    store.ingest("peer", 7, {"target": [(1, 0.2, 1.0)]}, seq=7)

    mine = oa.flatten(store.sender_observations(7, "self"))
    theirs = oa.flatten(store.sender_observations(7, "peer"))
    assert mine == {1: 0.9} and theirs == {1: 0.2}
    assert store.senders(7) == ["peer", "self"]


def test_an_unknown_sender_reads_empty(tmp_path):
    store = ReputationStore(path=tmp_path / "rep.json")
    assert store.sender_observations(7, "nobody") == {}
    assert store.senders(7) == []


def test_a_real_pair_of_validators_agrees(tmp_path):
    """Two validators scoring the same articles the same way must not flag."""
    store = ReputationStore(path=tmp_path / "rep.json")
    for aid in range(50):
        store.record_local(3, "self", "target", article_id=aid,
                           graded=0.7 + (aid % 5) / 100, weight=1.0)
    store.ingest("peer", 3, {"target": [(aid, 0.7 + (aid % 5) / 100, 1.0)
                                        for aid in range(50)]}, seq=3)

    verdict = oa.assess(oa.flatten(store.sender_observations(3, "self")),
                        oa.flatten(store.sender_observations(3, "peer")))
    assert verdict.comparable and not verdict.flagged
