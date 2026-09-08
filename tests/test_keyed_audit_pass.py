"""Keyed selection decides what is watched; the random sample still decides acceptance."""

import types

import pytest

import alpharidge_ai.config as config
from alpharidge_ai.analyzer import scoring
from alpharidge_ai.mechanism import profile as mp
from alpharidge_ai.oracle.runner import Auditor, Observation
from tests.test_profile_client import valid


def test_the_cap_is_served_and_bounded():
    assert config.AUDIT_MAX_PER_BATCH >= 1
    assert "AUDIT_MAX_PER_BATCH" in config._REMOTE_CONFIG_KEYS


def test_the_cap_is_not_a_consensus_key():
    """It bounds local spend and latency, not what anyone is paid."""
    assert "AUDIT_MAX_PER_BATCH" not in config._CONSENSUS_KEYS


# ---- selects() ---------------------------------------------------------------------

NUMBERS = "Revenue rose to $1.2 billion in the quarter, up 12 percent."


def _auditor(pool_rate=1.0, keeper_rate=0.0, key=b"k"):
    raw = valid()
    raw["oracle"]["keyed_rate_pool"] = pool_rate
    raw["oracle"]["keyed_rate_keeper"] = keeper_rate
    profile = mp.parse(raw)
    return Auditor(key, lambda block: profile)


def test_selects_says_yes_on_a_watched_article():
    assert _auditor().selects(1, NUMBERS, block=0)


def test_selects_says_no_when_nothing_is_watched():
    assert not _auditor(pool_rate=0.0).selects(1, NUMBERS, block=0)


def test_selects_needs_a_profile():
    assert not Auditor(b"k", lambda block: None).selects(1, NUMBERS, block=0)


def test_selects_never_raises():
    def explode(block):
        raise RuntimeError("no profile")
    assert not Auditor(b"k", explode).selects(1, NUMBERS, block=0)


def test_selects_agrees_with_the_audit_decision():
    """A no from selects must mean the audit would also decline, or work is wasted."""
    auditor = _auditor(pool_rate=0.5)
    from alpharidge_ai.oracle import floor
    miner = types.SimpleNamespace(numeric_claims=[], quotes=[], assets=[],
                                  schema_version="1.2.0")
    result = floor.evaluate(miner, NUMBERS)
    for article_id in range(60):
        if not auditor.selects(article_id, NUMBERS, block=0):
            assert auditor.audit(article_id, NUMBERS, miner, miner, result, 0) is None


# ---- the batch pass ----------------------------------------------------------------

class Recorder:
    """An auditor that watches everything and records what it was asked to audit."""

    def __init__(self):
        self.audited = []

    def selects(self, article_id, text, block):
        return True

    def audit(self, article_id, text, miner_intel, grader_intel, result, block):
        self.audited.append(article_id)
        return Observation(article_id=article_id, score=0.5, weight=1.0, path="pool")


class Analyzer:
    def __init__(self):
        self.calls = 0

    def analyze(self, **kwargs):
        self.calls += 1
        return types.SimpleNamespace(numeric_claims=[], quotes=[])


def _article(article_id, blob, content):
    return types.SimpleNamespace(
        id=article_id, content=content, title="t", url="u", source="s",
        published="2026-09-01", summary="", raw_html=None,
        analysis=types.SimpleNamespace(analysis_data=blob))


def test_the_keyed_pass_is_capped(monkeypatch):
    from tests.test_floor_gating import _payload, TEXT, TITLE

    monkeypatch.setattr(scoring, "_cfg_get",
                        lambda k, d=None: 2 if k == "AUDIT_MAX_PER_BATCH" else d)
    recorder, analyzer = Recorder(), Analyzer()
    blob = _payload([{"metric_name": "revenue", "value": 1.2e9, "unit": "USD",
                      "confidence": 0.9}])
    batch = [_article(i, blob, TEXT) for i in range(1, 11)]
    for a in batch:
        a.title = TITLE

    monkeypatch.setattr(scoring, "validate_article_intelligence",
                        lambda m, v: (True, 1.0, {}))
    monkeypatch.setattr(scoring, "_summary_agreement", lambda m, v: 1.0)

    _, result = scoring.validate_miner_article_intelligence_batch(
        batch, analyzer, sample_size=1, auditor=recorder, block=0)

    # The cap bounds EXTRA reference analyses. The acceptance sample's analysis is
    # already paid for, so its article is audited on top of the cap, not inside it.
    assert analyzer.calls <= 1 + 2
    assert len(recorder.audited) <= 1 + 2
    assert len(result["audit_observations"]) <= 1 + 2


def test_no_auditor_means_no_extra_analyses(monkeypatch):
    from tests.test_floor_gating import _payload, TEXT, TITLE

    analyzer = Analyzer()
    blob = _payload([{"metric_name": "revenue", "value": 1.2e9, "unit": "USD",
                      "confidence": 0.9}])
    batch = [_article(i, blob, TEXT) for i in range(1, 11)]
    for a in batch:
        a.title = TITLE

    monkeypatch.setattr(scoring, "validate_article_intelligence",
                        lambda m, v: (True, 1.0, {}))
    monkeypatch.setattr(scoring, "_summary_agreement", lambda m, v: 1.0)

    scoring.validate_miner_article_intelligence_batch(
        batch, analyzer, sample_size=1, auditor=None, block=0)

    assert analyzer.calls == 1   # the acceptance sample, and nothing more
