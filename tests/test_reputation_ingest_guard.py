"""Ingest bounds on peer reputation broadcasts: sequence guard and volume caps."""

import pytest

from alpharidge_ai.validator import reputation_store as rs
from alpharidge_ai.validator.reputation_store import ReputationStore


@pytest.fixture
def store(tmp_path):
    return ReputationStore(path=tmp_path / "rep.json")


def _obs(n, start=0):
    return [(start + i, 1.0, 1.0) for i in range(n)]


def test_accepts_a_normal_broadcast(store):
    ok, reason = store.ingest("peer", 10, {"target": _obs(3)}, seq=10)
    assert ok and "accepted" in reason
    assert len(store.obs[10]["peer"]["target"]) == 3


def test_replayed_seq_is_rejected(store):
    assert store.ingest("peer", 10, {"t": _obs(2)}, seq=10)[0]
    ok, reason = store.ingest("peer", 10, {"t": _obs(2, start=99)}, seq=10)
    assert not ok and "duplicate_or_old_seq" in reason
    assert len(store.obs[10]["peer"]["t"]) == 2


def test_far_future_seq_cannot_poison_the_watermark(store):
    ok, reason = store.ingest("peer", 10, {"t": _obs(1)}, seq=750000)
    assert not ok and "seq_epoch_skew" in reason
    # the rejected seq must not become the watermark, or every later broadcast dies
    assert store.ingest("peer", 11, {"t": _obs(1)}, seq=11)[0]


def test_observations_per_target_are_capped(store):
    store.ingest("peer", 5, {"t": _obs(rs.MAX_OBS_PER_TARGET + 50)}, seq=5)
    assert len(store.obs[5]["peer"]["t"]) == rs.MAX_OBS_PER_TARGET


def test_target_count_is_capped(store):
    targets = {f"t{i}": _obs(1) for i in range(rs.MAX_TARGETS_PER_SENDER + 20)}
    store.ingest("peer", 5, targets, seq=5)
    assert len(store.obs[5]["peer"]) == rs.MAX_TARGETS_PER_SENDER


def test_finalized_epoch_is_closed(store):
    store.record_local(5, "self", "t", article_id=1, graded=1.0, weight=1.0)
    store.finalize(5)
    ok, reason = store.ingest("peer", 5, {"t": _obs(1)}, seq=5)
    assert not ok and "finalized" in reason


def test_seq_watermark_survives_a_reload(store, tmp_path):
    store.ingest("peer", 10, {"t": _obs(1)}, seq=10)
    store.save()
    reloaded = ReputationStore(path=tmp_path / "rep.json")
    reloaded.load()
    assert not reloaded.ingest("peer", 10, {"t": _obs(1)}, seq=10)[0]


def test_seq_is_optional(store):
    """Legacy callers that pass no seq still work."""
    assert store.ingest("peer", 10, {"t": _obs(1)})[0]
