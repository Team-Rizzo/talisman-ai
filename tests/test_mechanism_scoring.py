"""Article scoring: exogenous recall, proper confidence rule, keeper agreement."""

import pytest

from alpharidge_ai.mechanism import scoring


GOLD = {"a", "b", "c", "d"}


def score(miner, gold=GOLD, valid=(), conf=None, **kw):
    return scoring.article_score(miner, gold, valid, conf, **kw)


def flat(keys, p):
    return {k: p for k in keys}


# ---- shape ------------------------------------------------------------------------

def test_a_perfect_confident_submission_scores_at_the_top():
    s = score(list(GOLD), conf=flat(GOLD, 0.99))
    assert s.recall == 1.0 and s.precision == 1.0 and s.f1 == 1.0
    assert s.normalized == pytest.approx(1.0, abs=0.01)


def test_no_gold_yields_no_observation():
    assert scoring.article_score(["a"], set()) is None


def test_missing_everything_scores_zero():
    s = score([])
    assert s.f1 == 0.0 and s.normalized == 0.0


def test_scores_stay_inside_the_observation_range():
    for miner in ([], ["a"], list(GOLD), list(GOLD) + ["x", "y"]):
        for p in (0.0, 0.5, 1.0):
            s = score(miner, conf=flat(miner, p))
            assert 0.0 <= s.normalized <= 1.0


# ---- recall is exogenous ----------------------------------------------------------

def test_recall_denominator_is_the_graders_set():
    """Submitting less cannot raise recall."""
    assert score(["a", "b"]).recall == 0.5
    assert score(["a"]).recall == 0.25


def test_padding_with_junk_costs_precision():
    honest = score(["a", "b", "c", "d"], conf=flat(GOLD, 0.9))
    padded = score(["a", "b", "c", "d", "x", "y", "z"],
                   conf=flat(list(GOLD) + ["x", "y", "z"], 0.9))
    assert padded.recall == honest.recall
    assert padded.precision < honest.precision
    assert padded.normalized < honest.normalized


def test_a_claim_the_grader_separately_accepts_is_not_punished():
    """Finding something outside the grader's own set is allowed if it is adjudicated."""
    unchecked = score(["a", "b", "extra"])
    accepted = score(["a", "b", "extra"], valid={"extra"})
    assert accepted.precision > unchecked.precision


# ---- the confidence term ----------------------------------------------------------

def test_confident_and_wrong_is_worse_than_uncertain_and_wrong():
    keys = ["a", "b", "wrong"]
    bold = score(keys, conf={"a": 0.9, "b": 0.9, "wrong": 0.99})
    humble = score(keys, conf={"a": 0.9, "b": 0.9, "wrong": 0.10})
    assert humble.normalized > bold.normalized


def test_honest_uncertainty_beats_indiscriminate_confidence():
    """A dump that cannot tell its good claims from its bad ones loses."""
    keys = list(GOLD) + ["x", "y", "z", "w"]
    discriminating = score(keys, conf={**flat(GOLD, 0.95),
                                       **flat(["x", "y", "z", "w"], 0.10)})
    indiscriminate = score(keys, conf=flat(keys, 0.95))
    assert discriminating.normalized > indiscriminate.normalized


def test_certainty_of_one_on_a_wrong_claim_is_finite():
    s = score(["a", "wrong"], conf={"a": 1.0, "wrong": 1.0})
    assert s.confidence >= 0.0
    assert s.normalized == s.normalized  # not NaN


def test_certainty_of_zero_is_finite():
    s = score(["a"], conf={"a": 0.0})
    assert 0.0 <= s.normalized <= 1.0


def test_missing_confidence_is_treated_as_neutral():
    assert score(["a", "b"], conf={}).confidence == pytest.approx(0.5)


def test_confidence_cannot_rescue_a_bad_extraction():
    assert score([], conf={}).normalized == 0.0


# ---- schema grace -----------------------------------------------------------------

def test_grace_holds_the_confidence_term_neutral():
    graced = score(list(GOLD), score_confidence=False)
    assert graced.raw == pytest.approx(graced.f1)
    assert graced.confidence == 0.0


def test_grace_neither_rewards_nor_punishes_relative_to_neutral():
    """A submission with no confidence field scores as one that guessed 0.5 everywhere."""
    graced = score(list(GOLD), score_confidence=False)
    neutral = score(list(GOLD), conf=flat(GOLD, 0.5))
    assert graced.normalized == pytest.approx(neutral.normalized, abs=1e-9)


def test_a_calibrated_submission_beats_the_graced_baseline():
    graced = score(list(GOLD), score_confidence=False)
    calibrated = score(list(GOLD), conf=flat(GOLD, 0.97))
    assert calibrated.normalized > graced.normalized


# ---- cap --------------------------------------------------------------------------

def test_claim_cap_bounds_what_is_scored():
    s = score([f"k{i}" for i in range(500)], claim_cap=25)
    assert s.scored_claims == 25


# ---- keeper -----------------------------------------------------------------------

def test_keeper_full_agreement():
    fields = {"overall_sentiment": "bullish", "impact_potential": "high",
              "urgency": "breaking", "content_type": "news",
              "assets": ["NVDA"], "entities": ["Nvidia"]}
    assert scoring.keeper_agreement(fields, fields) == pytest.approx(1.0)


def test_keeper_total_disagreement():
    miner = {"overall_sentiment": "bullish", "impact_potential": "high",
             "urgency": "breaking", "content_type": "news",
             "assets": ["NVDA"], "entities": ["Nvidia"]}
    grader = {"overall_sentiment": "bearish", "impact_potential": "low",
              "urgency": "evergreen", "content_type": "opinion",
              "assets": ["AMD"], "entities": ["AMD Inc"]}
    assert scoring.keeper_agreement(miner, grader) == pytest.approx(0.0)


def test_keeper_partial_agreement_is_between():
    miner = {"overall_sentiment": "bullish", "impact_potential": "high",
             "assets": ["NVDA", "AMD"]}
    grader = {"overall_sentiment": "bullish", "impact_potential": "low",
              "assets": ["NVDA"]}
    got = scoring.keeper_agreement(miner, grader)
    assert 0.0 < got < 1.0


def test_keeper_needs_both_sides():
    assert scoring.keeper_agreement({}, {"urgency": "flash"}) is None
    assert scoring.keeper_agreement({"urgency": "flash"}, {}) is None


def test_keeper_stays_in_the_observation_range():
    miner = {"overall_sentiment": "bullish", "assets": ["A", "B"], "entities": []}
    grader = {"overall_sentiment": "bearish", "assets": ["B"], "entities": ["X"]}
    assert 0.0 <= scoring.keeper_agreement(miner, grader) <= 1.0


def test_a_dump_scores_far_below_an_honest_extraction():
    """Shape check against the measured corpus: ~2.5 real claims, ~9.2 reported by a
    dump that reports everything it sees."""
    gold = {f"g{i}" for i in range(3)}
    honest = scoring.article_score(sorted(gold), gold, (), flat(gold, 0.9))
    junk = [f"j{i}" for i in range(6)]
    dump_keys = sorted(gold) + junk
    dump = scoring.article_score(dump_keys, gold, (), flat(dump_keys, 0.9))
    assert dump.normalized / honest.normalized < 0.35
