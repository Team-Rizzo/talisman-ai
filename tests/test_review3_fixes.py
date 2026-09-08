"""Unit-scaled claims, CJK compound sums, dispatch limits, and year handling."""

import types

import pytest

from alpharidge_ai.analyzer.article_intelligence_analyzer import (
    ArticleIntelligenceAnalyzer as A)
from alpharidge_ai.oracle import floor
from alpharidge_ai.utils.cooldown import MinerCooldownTracker


def kind(text, value, unit):
    return floor.ground_kind(types.SimpleNamespace(value=value, unit=unit),
                             floor.parse_numbers(floor.normalize(text).text))


def values(text):
    return [round(n.value, 3) for n in floor.parse_numbers(floor.normalize(text).text)]


# ---- 1. a scale named in the unit makes the claim ambiguous ------------------------

def test_a_unit_scale_never_grants_automatic_validity():
    """value=1 with unit "million" asserts a million; a bare 1 is not that."""
    assert kind("There was about 1 person.", 1.0, "million") == "inferred"


def test_the_scaled_reading_is_also_only_inferred():
    assert kind("Omzet van 25,7 miljoen euro.", 25.7, "million euros") == "inferred"


def test_an_unscaled_unit_still_grounds_literally():
    assert kind("Revenue was $1.2 billion.", 1.2e9, "USD") == "literal"
    assert kind("Exactly 1203 units shipped.", 1203.0, "count") == "literal"


def test_an_ambiguous_claim_can_still_be_upheld_by_a_grader():
    from alpharidge_ai.oracle import audit
    text = "Omzet van 25,7 miljoen euro."
    claims = [types.SimpleNamespace(metric_name="revenue", value=25.7,
                                    unit="million euros", context="")]
    res = floor.evaluate(types.SimpleNamespace(numeric_claims=claims, quotes=[],
                                               assets=[], entities=[]), text)
    assert res.grounded == set() and res.inferred == {0}

    def honest(article, batch):
        return [{"i": c["i"], "supported": True, "evidence": "omzet van 25,7 miljoen"}
                for c in batch]

    assert ("m", 0) in audit.adjudicate(claims, [], res.grounded, text, honest).valid


# ---- 2. the compound needs a component that carried a scale mark -------------------

def test_a_nearby_scale_mark_does_not_authorise_a_sum():
    """The guard checked whether a mark existed nearby, not whether either number
    actually carried one."""
    assert 120.0 not in values("100 and 20 억")


def test_a_real_cjk_compound_still_sums():
    assert 1.2e12 in values("거래액 1조 2천억 원")


def test_the_compound_remains_inferred():
    assert kind("거래액 1조 2천억 원", 1.2e12, "KRW") == "inferred"


def test_the_scaled_flag_marks_only_scale_bearing_numbers():
    parsed = floor.parse_numbers(floor.normalize("3억 and 20 people").text)
    assert any(n.scaled for n in parsed)
    assert any(not n.scaled for n in parsed)


# ---- 5. the reservation agrees with the selection ----------------------------------

def test_the_reservation_uses_the_same_ration_the_selector_did():
    """Selection allowed several batches while the reservation refused them."""
    tracker = MinerCooldownTracker(adaptive=True)
    tracker._window = {"hk": 1.0}
    tracker.set_ration_source(lambda hk: tracker._bs_max() * 3.0)

    assert sum(tracker.try_acquire("hk") for _ in range(3)) == 3


def test_without_a_ration_the_window_still_governs(monkeypatch):
    tracker = MinerCooldownTracker(adaptive=True)
    tracker._window = {"hk": 1.0}
    tracker.set_ration_source(lambda hk: None)
    monkeypatch.setattr(tracker, "_adaptive_active", lambda: True)
    assert tracker.try_acquire("hk")
    assert not tracker.try_acquire("hk")


def test_an_installed_but_inactive_ration_does_not_widen_the_limit():
    """The source exists from startup; only a ration in force may raise the cap."""
    tracker = MinerCooldownTracker(adaptive=True)
    tracker._window = {"hk": 1.0}
    tracker.set_ration_source(lambda hk: None)
    from alpharidge_ai.utils.cooldown import MAX_INFLIGHT_PER_MINER
    granted = sum(tracker.try_acquire("hk") for _ in range(MAX_INFLIGHT_PER_MINER + 3))
    assert granted == MAX_INFLIGHT_PER_MINER


# ---- 9. a year is a year in either form --------------------------------------------

def kept(raw):
    return [(c.metric_name, c.value) for c in A._build_numeric_claims(A, raw)]


@pytest.mark.parametrize("value", [2026, 2026.0, "2026", " 2026 "])
def test_a_year_is_dropped_in_either_form(value):
    assert kept([{"metric_name": "period", "value": value, "unit": "year",
                  "confidence": 0.9}]) == []


@pytest.mark.parametrize("value", [3, 3.0, "3"])
def test_a_duration_survives_in_either_form(value):
    assert kept([{"metric_name": "outage", "value": value, "unit": "hours",
                  "confidence": 0.9}]) != []
