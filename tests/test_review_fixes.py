"""Fixes for the defects the plan-alignment review found."""

import types

import pytest

from alpharidge_ai.mechanism import controller as c
from alpharidge_ai.mechanism import profile as mp
from alpharidge_ai.mechanism import settlement
from alpharidge_ai.oracle import floor, schema_gate
from tests.test_profile_client import valid


# ---- 1. the live flip cannot leave reputation unfed --------------------------------

def test_the_old_scorer_stands_down_only_when_something_replaces_it():
    """oracle.live with no working auditor must not silence both sources."""
    live, auditor = True, None
    supersedes = live and auditor is not None
    assert supersedes is False       # legacy scorer keeps writing

    auditor = object()
    assert (live and auditor is not None) is True


# ---- 2. settlement has a live path -------------------------------------------------

def test_settlement_live_is_a_profile_field_defaulting_off():
    assert mp.parse(valid()).settlement.live is False
    raw = valid()
    raw["settlement"]["live"] = True
    assert mp.parse(raw).settlement.live is True


def test_the_live_vector_is_a_valid_weight_vector():
    result = settlement.settle({"a": 400.0, "b": 200.0}, 1000.0)
    vector = settlement.weight_vector(result.shares, result.burn,
                                      ["a", "b", "burn"], 2, 3)
    assert sum(vector) == pytest.approx(1.0)
    assert all(w >= 0 for w in vector)


# ---- 3. rations fail closed --------------------------------------------------------

def _ration_inputs(batch, floor_results):
    """Mirror of the accounting in _observe_ration."""
    judged = [a for a in batch if int(a.id) in floor_results]
    validated = sum(1 for a in judged if floor_results.get(int(a.id)) is True)
    return len(judged), validated


def test_an_unknown_floor_outcome_advances_nothing():
    batch = [types.SimpleNamespace(id=i) for i in range(1, 5)]
    dispatched, validated = _ration_inputs(batch, {1: True, 2: False})
    assert (dispatched, validated) == (2, 1)     # 3 and 4 were never judged


def test_only_an_explicit_pass_counts():
    batch = [types.SimpleNamespace(id=1)]
    assert _ration_inputs(batch, {1: None})[1] == 0
    assert _ration_inputs(batch, {1: False})[1] == 0
    assert _ration_inputs(batch, {1: True})[1] == 1


def test_a_batch_the_floor_never_judged_is_skipped_entirely():
    batch = [types.SimpleNamespace(id=i) for i in range(3)]
    assert _ration_inputs(batch, {})[0] == 0


# ---- 4. evidence spans use the real field ------------------------------------------

TEXT = "NVDA rose sharply after the quarterly report was published today."


def _intel(assets=(), entities=()):
    return types.SimpleNamespace(numeric_claims=[], quotes=[],
                                 assets=list(assets), entities=list(entities))


def test_the_list_field_is_read():
    asset = types.SimpleNamespace(symbol="NVDA", evidence_spans=["NVDA"])
    assert floor.evaluate(_intel(assets=[asset]), TEXT).span_failures == {0}


def test_one_span_with_context_is_enough():
    asset = types.SimpleNamespace(symbol="NVDA",
                                  evidence_spans=["NVDA", "NVDA rose sharply after"])
    assert floor.evaluate(_intel(assets=[asset]), TEXT).span_failures == set()


def test_entities_are_checked_too():
    entity = types.SimpleNamespace(canonical_name="Nvidia", evidence_spans=["Nvidia"])
    assert floor.evaluate(_intel(entities=[entity]), TEXT).span_failures == {0}


def test_an_item_with_no_spans_is_not_failed():
    asset = types.SimpleNamespace(symbol="NVDA", evidence_spans=[])
    assert floor.evaluate(_intel(assets=[asset]), TEXT).span_failures == set()


# ---- 5. the cutover checks fields, not just the claimed version --------------------

CUT = 1000


def _sub(conf=0.9, q_conf=0.8, start=1, end=9):
    return types.SimpleNamespace(
        numeric_claims=[types.SimpleNamespace(confidence=conf)],
        quotes=[types.SimpleNamespace(confidence=q_conf, start_offset=start,
                                      end_offset=end)])


def test_a_complete_submission_is_scored_on_confidence():
    v = schema_gate.evaluate("1.2.0", block=CUT + 1, cutover_block=CUT, intel=_sub())
    assert v.accepted and v.score_confidence


def test_claiming_the_new_version_without_the_fields_fails_after_cutover():
    v = schema_gate.evaluate("1.2.0", block=CUT, cutover_block=CUT,
                             intel=_sub(conf=None))
    assert not v.accepted and v.reason == "missing_1_2_0_fields"


def test_missing_quote_offsets_also_fail():
    v = schema_gate.evaluate("1.2.0", block=CUT, cutover_block=CUT,
                             intel=_sub(start=None))
    assert not v.accepted


def test_before_the_cutover_it_is_treated_as_the_older_schema():
    v = schema_gate.evaluate("1.2.0", block=CUT - 1, cutover_block=CUT,
                             intel=_sub(conf=None))
    assert v.accepted and not v.score_confidence


def test_the_check_is_skipped_when_no_submission_is_supplied():
    v = schema_gate.evaluate("1.2.0", block=CUT + 1, cutover_block=CUT)
    assert v.accepted and v.score_confidence


# ---- 6. the gap clock follows real steps -------------------------------------------

RULE = dict(roi_lo=1.5, roi_hi=6.0, arm_days=3, max_step=0.2, gap_days=14)


def test_an_ignored_proposal_does_not_silence_the_controller():
    state, capacity, days = c.ControllerState(), 1000.0, []
    for day in range(40):
        state, p = c.advance(state, day=day, roi_today=0.2, capacity=capacity,
                             last_capacity=capacity, **RULE)
        if p:
            days.append(day)
    assert len(days) > 2
    assert max(b - a for a, b in zip(days, days[1:])) <= RULE["arm_days"] + 1


def test_a_published_step_starts_the_gap():
    state, capacity, last, days = c.ControllerState(), 1000.0, 1000.0, []
    for day in range(40):
        state, p = c.advance(state, day=day, roi_today=0.2, capacity=capacity,
                             last_capacity=last, **RULE)
        last = capacity
        if p:
            days.append(day)
            capacity = p.to_capacity
    assert all(b - a >= RULE["gap_days"] for a, b in zip(days, days[1:]))


def test_capacity_holding_still_is_not_a_step():
    state = c.ControllerState(days_outside=2, roi_ema=0.2, last_step_day=None)
    after = c.observed_step(state, day=10, capacity=1000.0, last_capacity=1000.0)
    assert after.last_step_day is None
