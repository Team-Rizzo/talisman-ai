"""The capacity controller as it runs: daily aggregation, persistence, reporting."""

import json
import types

import pytest

from alpharidge_ai.mechanism import controller
from alpharidge_ai.mechanism import profile as mp
from alpharidge_ai.validator import capacity_controller as cc
from tests.test_profile_client import valid

EPOCHS_PER_DAY = mp.EPOCHS_PER_DAY


@pytest.fixture
def profile():
    return mp.parse(valid())


@pytest.fixture
def ctrl(tmp_path):
    return cc.CapacityController(path=tmp_path / "controller.json")


def drive(ctrl, profile, roi_by_day, monkeypatch, start_day=0, publish=True):
    """Run whole days at a fixed return, collecting any proposals.

    `publish=True` models an operator republishing capacity after each proposal, which
    is what makes a proposal a step.
    """
    proposals = []
    for offset, value in enumerate(roi_by_day):
        monkeypatch.setattr(ctrl, "measure", lambda p, v=value: v)
        for e in range(EPOCHS_PER_DAY):
            epoch = (start_day + offset) * EPOCHS_PER_DAY + e
            got = ctrl.observe(epoch, profile)
            if got:
                proposals.append(got)
                if publish:
                    profile = _with_capacity(profile, got.to_capacity)
    return proposals


def _with_capacity(profile, capacity):
    """A profile identical but for capacity, as a republish would produce."""
    raw = dict(profile.raw)
    raw["settlement"] = {**raw["settlement"], "C": float(capacity)}
    return mp.parse(raw)


# ---- measurement ------------------------------------------------------------------

def test_the_return_is_pay_over_cost(ctrl, profile, monkeypatch):
    monkeypatch.setattr(cc.burn, "get_miner_alpha_per_block", lambda: 9.0)
    monkeypatch.setattr(cc.burn, "alpha_usd", lambda: 1.0)
    monkeypatch.setattr(cc.config, "BLOCK_LENGTH", 100, raising=False)

    expected = controller.roi(9.0 * 100, profile.settlement.C, 1.0,
                              profile.controller.cost_per_point)
    assert ctrl.measure(profile) == pytest.approx(expected)


def test_capacity_is_compared_against_one_epoch_of_emission(ctrl, profile, monkeypatch):
    """Capacity is points per epoch, so per-block emission would be off by the epoch."""
    monkeypatch.setattr(cc.burn, "get_miner_alpha_per_block", lambda: 9.0)
    monkeypatch.setattr(cc.burn, "alpha_usd", lambda: 1.0)
    monkeypatch.setattr(cc.config, "BLOCK_LENGTH", 100, raising=False)
    per_block = controller.roi(9.0, profile.settlement.C, 1.0,
                               profile.controller.cost_per_point)
    assert ctrl.measure(profile) == pytest.approx(per_block * 100)


def test_an_unreadable_chain_yields_no_measurement(ctrl, profile, monkeypatch):
    def boom():
        raise RuntimeError("subtensor down")
    monkeypatch.setattr(cc.burn, "get_miner_alpha_per_block", boom)
    assert ctrl.measure(profile) is None


def test_a_failed_measurement_does_not_break_the_day(ctrl, profile, monkeypatch):
    monkeypatch.setattr(ctrl, "measure", lambda p: None)
    for e in range(EPOCHS_PER_DAY * 2):
        assert ctrl.observe(e, profile) is None


# ---- daily aggregation ------------------------------------------------------------

def test_epochs_are_averaged_into_one_day(ctrl, profile, monkeypatch):
    values = iter([2.0] * (EPOCHS_PER_DAY // 2) + [4.0] * (EPOCHS_PER_DAY // 2))
    monkeypatch.setattr(ctrl, "measure", lambda p: next(values, 3.0))
    for e in range(EPOCHS_PER_DAY):
        ctrl.observe(e, profile)
    ctrl.observe(EPOCHS_PER_DAY, profile)          # rolls the day
    assert ctrl.state.roi_ema == pytest.approx(3.0)


def test_no_step_while_the_return_sits_in_band(ctrl, profile, monkeypatch):
    assert drive(ctrl, profile, [3.6] * 40, monkeypatch) == []


def test_a_sustained_low_return_arms_a_downward_step(ctrl, profile, monkeypatch):
    proposals = drive(ctrl, profile, [0.2] * 20, monkeypatch)
    assert proposals and proposals[0].direction == controller.DOWN
    assert proposals[0].to_capacity < profile.settlement.C


def test_a_brief_excursion_arms_nothing(ctrl, profile, monkeypatch):
    assert drive(ctrl, profile, [3.6] * 10 + [0.2, 0.2] + [3.6] * 10,
                 monkeypatch) == []


def test_steps_respect_the_gap(ctrl, profile, monkeypatch):
    days = [p.day for p in drive(ctrl, profile, [0.2] * 60, monkeypatch)]
    assert all(b - a >= profile.controller.gap_days for a, b in zip(days, days[1:]))


def test_an_unpublished_proposal_does_not_start_the_gap(ctrl, profile, monkeypatch):
    """Capacity never moved, so the controller keeps flagging the breach."""
    ignored = drive(ctrl, profile, [0.2] * 30, monkeypatch, publish=False)
    assert len(ignored) > 1


def test_without_a_profile_nothing_runs(ctrl):
    assert ctrl.observe(1, None) is None
    assert ctrl.samples == []


# ---- persistence ------------------------------------------------------------------

def test_the_arming_clock_survives_a_restart(ctrl, profile, monkeypatch, tmp_path):
    drive(ctrl, profile, [0.2] * 3, monkeypatch)
    assert ctrl.state.days_outside > 0

    restarted = cc.CapacityController(path=tmp_path / "controller.json")
    restarted.load()
    assert restarted.state.days_outside == ctrl.state.days_outside
    assert restarted.state.roi_ema == pytest.approx(ctrl.state.roi_ema)


def test_a_corrupt_state_file_does_not_crash_startup(tmp_path):
    path = tmp_path / "controller.json"
    path.write_text("{not json")
    ctrl = cc.CapacityController(path=path)
    ctrl.load()
    assert ctrl.state.roi_ema is None


def test_a_restart_mid_day_does_not_double_count(ctrl, profile, monkeypatch, tmp_path):
    monkeypatch.setattr(ctrl, "measure", lambda p: 3.6)
    for e in range(10):
        ctrl.observe(e, profile)
    ctrl.save()

    restarted = cc.CapacityController(path=tmp_path / "controller.json")
    restarted.load()
    assert restarted.day == ctrl.day


# ---- governance -------------------------------------------------------------------

def test_a_proposal_does_not_change_capacity(ctrl, profile, monkeypatch):
    before = profile.settlement.C
    proposals = drive(ctrl, profile, [0.2] * 20, monkeypatch)
    assert proposals
    assert profile.settlement.C == before


def test_the_reported_magnitude_matches_the_step_cap(ctrl, profile, monkeypatch):
    proposal = drive(ctrl, profile, [0.2] * 20, monkeypatch)[0]
    magnitude = abs(proposal.to_capacity / proposal.from_capacity - 1.0)
    assert magnitude == pytest.approx(profile.controller.max_step)
