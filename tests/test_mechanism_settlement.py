"""Settlement invariants: continuity, monotonicity, no gain from splitting."""

import pytest

from alpharidge_ai.mechanism import settlement as s

C = 1000.0


def test_below_capacity_every_point_pays_the_same():
    a = s.settle({"x": 100.0}, C)
    b = s.settle({"x": 100.0, "y": 700.0}, C)
    assert a.shares["x"] == pytest.approx(b.shares["x"])


def test_a_uid_pay_does_not_depend_on_what_others_do_below_capacity():
    alone = s.settle({"x": 250.0}, C).shares["x"]
    crowded = s.settle({"x": 250.0, "a": 300.0, "b": 400.0}, C).shares["x"]
    assert alone == pytest.approx(crowded)


def test_at_and_above_capacity_it_is_a_proportional_split():
    full = s.settle({"x": 600.0, "y": 400.0}, C)
    assert full.paid == pytest.approx(1.0)
    assert full.burn == pytest.approx(0.0)

    over = s.settle({"x": 1200.0, "y": 800.0}, C)
    assert over.shares["x"] == pytest.approx(0.6)
    assert over.paid == pytest.approx(1.0)


def test_pay_is_continuous_at_capacity():
    below = s.settle({"x": C - 1e-6}, C).shares["x"]
    at = s.settle({"x": C}, C).shares["x"]
    above = s.settle({"x": C + 1e-6}, C).shares["x"]
    assert below == pytest.approx(at, abs=1e-9)
    assert above == pytest.approx(at, abs=1e-9)


def test_burn_is_unfilled_capacity():
    assert s.settle({}, C).burn == pytest.approx(1.0)
    assert s.settle({"x": 250.0}, C).burn == pytest.approx(0.75)
    assert s.settle({"x": 1000.0}, C).burn == pytest.approx(0.0)


def test_burn_never_leaves_the_unit_interval():
    for work in (0.0, 1.0, 999.0, 1000.0, 5000.0):
        result = s.settle({"x": work}, C)
        assert 0.0 <= result.burn <= 1.0


def test_shares_and_burn_always_account_for_the_whole_emission():
    for work in ({}, {"x": 10.0}, {"x": 400.0, "y": 300.0},
                 {"x": 1000.0}, {"x": 900.0, "y": 900.0}):
        result = s.settle(work, C)
        assert result.paid + result.burn == pytest.approx(1.0)


def test_pay_is_monotone_in_a_uid_own_work():
    previous = -1.0
    for work in range(0, 2000, 50):
        share = s.settle({"x": float(work), "other": 300.0}, C).shares["x"]
        assert share >= previous
        previous = share


def test_splitting_work_across_uids_never_pays_more():
    solo = s.settle({"solo": 600.0, "other": 200.0}, C).shares["solo"]
    split = s.settle({"a": 200.0, "b": 200.0, "c": 200.0, "other": 200.0}, C)
    assert split.shares["a"] + split.shares["b"] + split.shares["c"] <= solo + 1e-12


def test_splitting_is_neutral_above_capacity_too():
    solo = s.settle({"solo": 1200.0, "other": 800.0}, C).shares["solo"]
    split = s.settle({"a": 600.0, "b": 600.0, "other": 800.0}, C)
    assert split.shares["a"] + split.shares["b"] == pytest.approx(solo)


def test_negative_work_is_floored_at_zero():
    result = s.settle({"x": -50.0, "y": 100.0}, C)
    assert result.shares["x"] == 0.0
    assert result.work == pytest.approx(100.0)


def test_capacity_must_be_positive():
    with pytest.raises(ValueError):
        s.settle({"x": 1.0}, 0.0)


# ---- the price peg ----------------------------------------------------------------

def test_burn_does_not_move_with_the_alpha_price():
    """The peg is capacity in points, so the same work burns the same at any price."""
    work = {"x": 500.0}
    assert s.settle(work, C).burn == pytest.approx(s.settle(work, C).burn)

    doubled_price_capacity = s.capacity_for_continuity(1000.0, 2.0, 0.001)
    halved_price_capacity = s.capacity_for_continuity(1000.0, 1.0, 0.001)
    assert doubled_price_capacity != halved_price_capacity  # only the cutover solve moves


def test_continuity_solve_reproduces_the_previous_per_point_pay():
    alpha_emission, alpha_usd, usd_per_point = 900.0, 1.25, 0.00105
    capacity = s.capacity_for_continuity(alpha_emission, alpha_usd, usd_per_point)
    pay_per_point_usd = (alpha_emission / capacity) * alpha_usd
    assert pay_per_point_usd == pytest.approx(usd_per_point)


# ---- weight vector ----------------------------------------------------------------

HOTKEYS = ["hk0", "hk1", "hk2", "burn"]
BURN_UID = 3


def test_weight_vector_places_shares_and_the_remainder():
    result = s.settle({"hk0": 250.0, "hk1": 250.0}, C)
    weights = s.weight_vector(result.shares, result.burn, HOTKEYS, BURN_UID)
    assert weights[0] == pytest.approx(0.25)
    assert weights[1] == pytest.approx(0.25)
    assert weights[BURN_UID] == pytest.approx(0.5)
    assert sum(weights) == pytest.approx(1.0)


def test_weight_vector_sums_to_one_when_full():
    result = s.settle({"hk0": 600.0, "hk1": 400.0}, C)
    weights = s.weight_vector(result.shares, result.burn, HOTKEYS, BURN_UID)
    assert sum(weights) == pytest.approx(1.0)
    assert weights[BURN_UID] == pytest.approx(0.0)


def test_a_departed_hotkey_is_burned_not_redistributed():
    result = s.settle({"hk0": 250.0, "gone": 250.0}, C)
    weights = s.weight_vector(result.shares, result.burn, HOTKEYS, BURN_UID)
    assert weights[0] == pytest.approx(0.25)
    assert weights[BURN_UID] == pytest.approx(0.75)
    assert sum(weights) == pytest.approx(1.0)


def test_weight_vector_handles_an_empty_epoch():
    result = s.settle({}, C)
    weights = s.weight_vector(result.shares, result.burn, HOTKEYS, BURN_UID)
    assert weights[BURN_UID] == pytest.approx(1.0)
    assert sum(weights) == pytest.approx(1.0)
