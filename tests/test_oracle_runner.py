"""The audit loop end to end, with a stand-in grader instead of model calls."""

import types

import pytest

from alpharidge_ai.mechanism import profile as mp
from alpharidge_ai.models import article_intelligence as ai
from alpharidge_ai.oracle import floor, runner, selector
from tests.test_profile_client import valid

KEY = b"validator-audit-key"

ARTICLE = ('Revenue rose to $1.2 billion in the quarter, up 12.5% year on year. '
           'The chief executive said "margins should recover in the second half" '
           'on the call.')
PLAIN = ('The company announced a change to its leadership team this week, and said '
         'the transition would proceed over the coming months without disruption.')


def profile(**oracle_overrides):
    raw = valid()
    raw["oracle"].update(oracle_overrides)
    return mp.parse(raw)


def claim(value, unit="USD", metric="revenue", confidence=0.9):
    return types.SimpleNamespace(metric_name=metric, value=value, unit=unit,
                                 context="", confidence=confidence)


def quote(text, confidence=0.9, start=None, end=None):
    return types.SimpleNamespace(text=text, confidence=confidence,
                                 start_offset=start, end_offset=end)


def intel(claims=(), quotes=(), schema="1.2.0", **judgment):
    base = dict(numeric_claims=list(claims), quotes=list(quotes), assets=[],
                entities=[], schema_version=schema)
    base.update(judgment)
    return types.SimpleNamespace(**base)


class Grader:
    """Answers as instructed and records what it was asked."""

    def __init__(self, supported=False, evidence="", judgment=None):
        self.supported = supported
        self.evidence = evidence
        self.judgment = judgment or {}
        self.adjudications = []
        self.judgements = []

    def adjudicate(self, text, claims, model):
        self.adjudications.append((model, claims))
        return [{"i": c["i"], "supported": self.supported,
                 "evidence": self.evidence} for c in claims]

    def judge(self, text, model):
        self.judgements.append(model)
        return dict(self.judgment)


def run(miner, grader_intel, text=ARTICLE, article_id=1, prof=None, grader=None,
        block=0):
    result = floor.evaluate(miner, text)
    return runner.audit_article(
        article_id, text, miner, grader_intel, result,
        audit_selector=selector.Selector(KEY), profile=prof or profile(),
        grader=grader, block=block)


def _always_pool():
    return profile(keyed_rate_pool=1.0, keyed_rate_keeper=0.0)


def _always_keeper():
    return profile(keyed_rate_pool=0.0, keyed_rate_keeper=1.0)


# ---- selection --------------------------------------------------------------------

def test_an_unselected_article_yields_nothing():
    prof = profile(keyed_rate_pool=0.0, keyed_rate_keeper=0.0)
    assert run(intel([claim(1.2e9)]), intel([claim(1.2e9)]), prof=prof) is None


def test_a_selected_article_yields_an_observation():
    obs = run(intel([claim(1.2e9)]), intel([claim(1.2e9)]), prof=_always_pool())
    assert obs is not None and obs.path == selector.POOL
    assert 0.0 <= obs.score <= 1.0 and obs.weight == 1.0


def test_no_profile_means_no_audit():
    miner = intel([claim(1.2e9)])
    result = floor.evaluate(miner, ARTICLE)
    assert runner.audit_article(1, ARTICLE, miner, miner, result,
                                audit_selector=selector.Selector(KEY),
                                profile=None) is None


def test_an_article_that_fails_the_floor_is_not_audited():
    miner = intel([claim(1.2e9)])
    failed = floor.FloorResult(False, "content_hash_mismatch")
    assert runner.audit_article(1, ARTICLE, miner, miner, failed,
                                audit_selector=selector.Selector(KEY),
                                profile=_always_pool()) is None


# ---- the claim path ---------------------------------------------------------------

def test_matching_the_reference_scores_well():
    obs = run(intel([claim(1.2e9)], [quote("margins should recover in the second half")]),
              intel([claim(1.2e9)], [quote("margins should recover in the second half")]),
              prof=_always_pool())
    assert obs.score > 0.6


def test_missing_what_the_reference_found_scores_worse():
    grader_intel = intel([claim(1.2e9), claim(12.5, "%", "growth")])
    full = run(intel([claim(1.2e9), claim(12.5, "%", "growth")]), grader_intel,
               prof=_always_pool())
    partial = run(intel([claim(1.2e9)]), grader_intel, prof=_always_pool())
    assert partial.score < full.score


def test_padding_with_junk_scores_worse():
    grader_intel = intel([claim(1.2e9)])
    honest = run(intel([claim(1.2e9)]), grader_intel, prof=_always_pool())
    padded = run(intel([claim(1.2e9)] + [claim(9.9e9 + i, metric=f"m{i}")
                                         for i in range(6)]),
                 grader_intel, prof=_always_pool())
    assert padded.score < honest.score


def test_confident_junk_scores_worse_than_hedged_junk():
    grader_intel = intel([claim(1.2e9)])
    bold = run(intel([claim(1.2e9), claim(9.9e9, metric="x", confidence=0.99)]),
               grader_intel, prof=_always_pool())
    hedged = run(intel([claim(1.2e9), claim(9.9e9, metric="x", confidence=0.05)]),
                 grader_intel, prof=_always_pool())
    assert hedged.score > bold.score


def test_an_article_with_no_gold_yields_nothing_not_a_zero():
    assert run(intel([claim(1.2e9)]), intel([]), prof=_always_pool()) is None


# ---- the residual -----------------------------------------------------------------

def test_grounded_claims_never_reach_the_grader():
    grader = Grader()
    run(intel([claim(1.2e9)]), intel([claim(1.2e9)]), prof=_always_pool(),
        grader=grader)
    assert grader.adjudications == []


def test_the_residual_is_batched_into_one_call():
    grader = Grader()
    miner = intel([claim(1.2e9)] + [claim(8.0e9 + i, metric=f"m{i}") for i in range(4)])
    run(miner, intel([claim(1.2e9)]), prof=_always_pool(), grader=grader)
    assert len(grader.adjudications) == 1
    assert len(grader.adjudications[0][1]) == 4


def test_grader_evidence_must_be_in_the_article():
    invented = Grader(supported=True, evidence="the company reported nine billion")
    real = Grader(supported=True, evidence="revenue rose to $1.2 billion")
    # The reference claim is included so recall is non-zero and precision is what moves.
    miner = intel([claim(1.2e9), claim(9.0e9, metric="other")])
    lying = run(miner, intel([claim(1.2e9)]), prof=_always_pool(), grader=invented)
    honest = run(miner, intel([claim(1.2e9)]), prof=_always_pool(), grader=real)
    assert honest.score > lying.score


def test_the_drawn_model_is_the_one_called():
    grader = Grader()
    run(intel([claim(9.9e9, metric="x")]), intel([claim(1.2e9)]),
        prof=_always_pool(), grader=grader)
    model, _ = grader.adjudications[0]
    assert model in {m.id for m in _always_pool().oracle.grader_models}


def test_no_grader_leaves_the_residual_unsupported():
    obs = run(intel([claim(9.9e9, metric="x"), claim(1.2e9)]), intel([claim(1.2e9)]),
              prof=_always_pool(), grader=None)
    assert obs is not None and obs.score < 1.0


# ---- the keeper path --------------------------------------------------------------

KEEPER_FIELDS = {"overall_sentiment": "bullish", "impact_potential": "high",
                 "urgency": "breaking", "content_type": "news"}


def test_the_keeper_path_scores_judgment_agreement():
    grader = Grader(judgment=dict(KEEPER_FIELDS))
    obs = run(intel(**KEEPER_FIELDS), intel(), text=PLAIN, prof=_always_keeper(),
              grader=grader)
    assert obs is not None and obs.path == selector.KEEPER
    assert obs.score == pytest.approx(1.0)
    assert obs.weight == pytest.approx(_always_keeper().oracle.keeper_weight)


def test_keeper_disagreement_scores_low():
    grader = Grader(judgment={"overall_sentiment": "bearish",
                              "impact_potential": "low",
                              "urgency": "evergreen", "content_type": "opinion"})
    obs = run(intel(**KEEPER_FIELDS), intel(), text=PLAIN, prof=_always_keeper(),
              grader=grader)
    assert obs.score == pytest.approx(0.0)


def test_the_keeper_path_carries_less_weight_than_a_claim_audit():
    prof = _always_keeper()
    assert prof.oracle.keeper_weight < 1.0


def test_the_keeper_path_needs_a_grader():
    assert run(intel(**KEEPER_FIELDS), intel(), text=PLAIN, prof=_always_keeper(),
               grader=None) is None


# ---- schema grace -----------------------------------------------------------------

def test_an_older_submission_is_scored_without_the_confidence_term():
    prof = profile(keyed_rate_pool=1.0, keyed_rate_keeper=0.0,
                   schema_cutover_block=1_000)
    older = intel([claim(1.2e9, confidence=None)], schema="1.1.0")
    obs = run(older, intel([claim(1.2e9)]), prof=prof, block=10)
    assert obs is not None and obs.score > 0.0


def test_an_older_submission_is_refused_after_the_cutover():
    prof = profile(keyed_rate_pool=1.0, keyed_rate_keeper=0.0,
                   schema_cutover_block=1_000)
    older = intel([claim(1.2e9, confidence=None)], schema="1.1.0")
    assert run(older, intel([claim(1.2e9)]), prof=prof, block=1_000) is None


def test_an_unknown_schema_is_refused():
    odd = intel([claim(1.2e9)], schema="0.9.0")
    assert run(odd, intel([claim(1.2e9)]), prof=_always_pool()) is None


# ---- determinism ------------------------------------------------------------------

def test_the_same_inputs_give_the_same_observation():
    miner, grader_intel = intel([claim(1.2e9)]), intel([claim(1.2e9)])
    first = run(miner, grader_intel, prof=_always_pool())
    for _ in range(4):
        again = run(miner, grader_intel, prof=_always_pool())
        assert again.score == first.score
