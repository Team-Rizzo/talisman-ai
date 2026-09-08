"""Capacity controller: band, hysteresis, bounded steps, and the minimum gap."""

import pytest

from alpharidge_ai.mechanism import controller as c

BAND = dict(roi_lo=1.5, roi_hi=6.0)
RULE = dict(arm_days=3, max_step=0.20, gap_days=14, **BAND)
C0 = 26_697.0


def drive(roi_series, capacity=C0, start_day=0, rule=None, publish=True):
    """Run a series of daily returns and collect the proposals.

    `publish=True` models an operator who acts on every proposal, so capacity moves and
    the gap applies. `publish=False` models one who ignores them: capacity holds and the
    controller keeps raising the breach.
    """
    rule = rule or RULE
    state = c.ControllerState()
    proposals, last = [], capacity
    for i, value in enumerate(roi_series):
        state, proposal = c.advance(state, day=start_day + i, roi_today=value,
                                    capacity=capacity, last_capacity=last, **rule)
        last = capacity
        if proposal:
            proposals.append(proposal)
            if publish:
                capacity = proposal.to_capacity
    return state, proposals, capacity


# ---- the measurement --------------------------------------------------------------

def test_roi_is_pay_over_cost():
    pay = c.pay_per_point(900.0, 1000.0, 1.0)
    assert pay == pytest.approx(0.9)
    assert c.roi(900.0, 1000.0, 1.0, 0.3) == pytest.approx(pay / 0.3)


def test_lower_capacity_pays_more_per_point():
    assert c.pay_per_point(900.0, 500.0, 1.0) > c.pay_per_point(900.0, 1000.0, 1.0)


def test_capacity_and_cost_must_be_positive():
    with pytest.raises(ValueError):
        c.pay_per_point(900.0, 0.0, 1.0)
    with pytest.raises(ValueError):
        c.roi(900.0, 1000.0, 1.0, 0.0)


# ---- the band ---------------------------------------------------------------------

def test_inside_the_band_nothing_happens():
    _, proposals, capacity = drive([3.6] * 180)
    assert proposals == []
    assert capacity == C0


def test_a_brief_excursion_does_not_arm_a_step():
    _, proposals, _ = drive([3.6] * 30 + [0.5, 0.5] + [3.6] * 30)
    assert proposals == []


def test_a_sustained_low_return_lowers_capacity():
    _, proposals, capacity = drive([3.6] * 10 + [0.2] * 60)
    assert proposals
    first = proposals[0]
    assert first.direction == c.DOWN
    assert first.to_capacity == pytest.approx(C0 * 0.8)
    assert capacity < C0


def test_a_sustained_high_return_raises_capacity():
    _, proposals, capacity = drive([3.6] * 10 + [20.0] * 60)
    assert proposals[0].direction == c.UP
    assert proposals[0].to_capacity == pytest.approx(C0 * 1.2)
    assert capacity > C0


def test_lowering_capacity_raises_the_return():
    """The step has to close the gap it was armed by."""
    before = c.roi(900.0, C0, 1.0, 0.000292)
    after = c.roi(900.0, C0 * 0.8, 1.0, 0.000292)
    assert after > before


# ---- hysteresis and the gap -------------------------------------------------------

def test_arming_needs_consecutive_days_not_a_total():
    """A day back inside the band clears the count."""
    state = c.observe(c.ControllerState(roi_ema=0.2, days_outside=2), 0.2, **BAND)
    assert state.days_outside == 3

    state = c.observe(c.ControllerState(roi_ema=1.4, days_outside=2), 3.0, **BAND)
    assert BAND["roi_lo"] <= state.roi_ema <= BAND["roi_hi"]
    assert state.days_outside == 0


def test_the_band_applies_to_the_average_not_to_one_day():
    """Daily returns outside the band are fine while the average sits inside it."""
    _, proposals, _ = drive([0.5, 8.0] * 90)
    assert proposals == []


def test_steps_respect_the_minimum_gap():
    _, proposals, _ = drive([0.2] * 180)
    days = [p.day for p in proposals]
    assert all(b - a >= RULE["gap_days"] for a, b in zip(days, days[1:]))


def test_a_step_waits_the_full_gap_before_the_next_one():
    _, within_gap, _ = drive([0.2] * 15)
    assert len(within_gap) == 1

    _, past_gap, _ = drive([0.2] * 20)
    assert len(past_gap) == 2
    assert past_gap[1].day - past_gap[0].day >= RULE["gap_days"]


def test_an_ignored_proposal_is_raised_again():
    """The gap is between real steps. A breach nobody acted on stays visible."""
    _, ignored, capacity = drive([0.2] * 40, publish=False)
    assert capacity == C0                       # nothing moved
    assert len(ignored) > 2
    gaps = [b.day - a.day for a, b in zip(ignored, ignored[1:])]
    assert max(gaps) <= RULE["arm_days"] + 1


def test_steps_are_bounded_by_the_cap():
    _, proposals, _ = drive([0.01] * 180)
    for p in proposals:
        change = abs(p.to_capacity - p.from_capacity) / p.from_capacity
        assert change <= RULE["max_step"] + 1e-12


# ---- the average ------------------------------------------------------------------

def test_the_average_is_a_seven_day_ema():
    assert c.ROI_EMA_DAYS == 7
    assert c.ROI_EMA_ALPHA == pytest.approx(0.25)


def test_the_first_day_seeds_the_average():
    state = c.observe(c.ControllerState(), 3.6, **BAND)
    assert state.roi_ema == pytest.approx(3.6)


def test_the_average_absorbs_a_single_ordinary_spike():
    state = c.ControllerState()
    for _ in range(20):
        state = c.observe(state, 3.6, **BAND)
    spiked = c.observe(state, 7.2, **BAND)
    assert spiked.roi_ema < BAND["roi_hi"]
    assert spiked.days_outside == 0


# ---- governance -------------------------------------------------------------------

def test_a_proposal_does_not_move_capacity_by_itself():
    state = c.ControllerState()
    for i in range(10):
        state, proposal = c.advance(state, day=i, roi_today=0.2, capacity=C0, **RULE)
        if proposal:
            assert proposal.from_capacity == C0  # the caller publishes, or does not


def test_state_round_trips_through_a_dict():
    state, _, _ = drive([0.2] * 20)
    assert c.ControllerState.from_dict(state.to_dict()) == state


# ---- shape against the backtest ---------------------------------------------------

def test_step_count_over_half_a_year_stays_low():
    """A crawling peg, not a weekly negotiation."""
    _, steps, _ = drive([0.2] * 180)
    assert len(steps) <= 13


def test_no_whipsaw_on_a_smooth_regime_change():
    """A one-way move must not produce reversals."""
    _, proposals, _ = drive([3.6] * 20 + [0.3] * 160)
    directions = [p.direction for p in proposals]
    assert all(d == c.DOWN for d in directions)


# ---- closed loop, the way it actually runs -----------------------------------------

def test_a_price_move_is_tracked_with_few_steps_and_no_reversals():
    """Capacity feeds back into the return, so a step closes the gap it was armed by."""
    alpha_emission, cost = 900.0, 0.000292
    capacity = C0
    alpha_usd = capacity * cost * 3.6 / alpha_emission   # start the band mid-range

    state = c.ControllerState()
    proposals, in_band = [], 0
    for day in range(180):
        alpha_usd *= 1.0125 if day < 90 else 0.995      # a bull run, then a drift back
        today = c.roi(alpha_emission, capacity, alpha_usd, cost)
        state, proposal = c.advance(state, day=day, roi_today=today,
                                    capacity=capacity, **RULE)
        if proposal:
            proposals.append(proposal)
            capacity = proposal.to_capacity
        if BAND["roi_lo"] <= state.roi_ema <= BAND["roi_hi"]:
            in_band += 1

    assert len(proposals) <= 8
    assert in_band / 180 > 0.60
    reversals = sum(1 for a, b in zip(proposals, proposals[1:])
                    if b.direction == -a.direction and b.day - a.day <= 28)
    assert reversals == 0
