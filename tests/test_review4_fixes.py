"""Claim matching compares magnitude, and the dispatch limit has one definition."""

import types

import pytest

from alpharidge_ai.oracle import audit, floor
from alpharidge_ai.utils import dispatch
from alpharidge_ai.utils.cooldown import MAX_INFLIGHT_PER_MINER, MinerCooldownTracker


def claim(value, unit, metric="revenue"):
    return types.SimpleNamespace(metric_name=metric, value=value, unit=unit, context="")


# ---- 1. the reference route compares magnitude, not just unit class ----------------

def test_a_scaled_unit_cannot_match_an_unscaled_reference():
    """"1 million" and "1" share the unit class `count`."""
    assert not audit.claims_match(claim(1.0, "million"), claim(1.0, "count"))


def test_the_same_holds_for_currency():
    assert not audit.claims_match(claim(1.0, "million USD"), claim(1.0, "USD"))


def test_an_identical_claim_still_matches():
    assert audit.claims_match(claim(1.2e9, "USD"), claim(1.2e9, "USD"))


def test_equivalent_claims_written_differently_still_match():
    """One million and 1,000,000 are the same assertion."""
    assert audit.claims_match(claim(1.0, "million"), claim(1e6, "count"))
    assert audit.claims_match(claim(2.5, "billion USD"), claim(2.5e9, "USD"))


def test_a_scaled_claim_does_not_reach_valid_through_the_reference_route():
    reference = [claim(1.0, "count")]
    submitted = [claim(1.0, "million")]
    decided = audit.adjudicate(submitted, reference, set(), "About 1 person.", None)
    assert decided.valid == set()
    assert decided.miner_keys == [("m", 0)]      # not granted the reference key
    assert decided.residual == [0]               # sent for adjudication instead


def test_an_honest_claim_still_takes_the_reference_key():
    reference = [claim(1.2e9, "USD")]
    submitted = [claim(1.2e9, "USD")]
    decided = audit.adjudicate(submitted, reference, set(), "Revenue $1.2 billion.",
                               None)
    assert decided.miner_keys == [("g", 0)]


# ---- 3. one definition of the limit, read by both sides ----------------------------

@pytest.mark.parametrize("ration,adaptive", [
    (None, False), (None, True), (12.0, False), (12.0, True),
])
def test_selection_and_reservation_agree_in_every_combination(ration, adaptive,
                                                              monkeypatch):
    tracker = MinerCooldownTracker(adaptive=adaptive)
    tracker._window = {"hk": 2.0}
    tracker.set_ration_source(lambda hk: ration)
    monkeypatch.setattr(tracker, "_adaptive_active", lambda: adaptive)

    assert dispatch._slot_limit(tracker, "hk") == tracker.inflight_limit("hk")


def test_the_non_adaptive_default_is_the_static_cap():
    """Selection offered one batch while reservation allowed four."""
    tracker = MinerCooldownTracker(adaptive=False)
    tracker.set_ration_source(lambda hk: None)
    assert dispatch._slot_limit(tracker, "hk") == MAX_INFLIGHT_PER_MINER


def test_a_ration_still_wins_over_both():
    tracker = MinerCooldownTracker(adaptive=True)
    tracker._window = {"hk": 1.0}
    tracker.set_ration_source(lambda hk: tracker._bs_max() * 3.0)
    assert dispatch._slot_limit(tracker, "hk") == 3
    assert tracker.inflight_limit("hk") == 3


def test_a_tracker_without_the_limit_helper_still_works():
    class Window:
        def window(self, hotkey):
            return 5
    assert dispatch._slot_limit(Window(), "hk") == 5
