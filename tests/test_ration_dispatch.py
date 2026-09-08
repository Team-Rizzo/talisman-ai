"""Dispatch on earned rations: off by default, and a clean swap when on."""

import pytest

import alpharidge_ai.config as config
from alpharidge_ai.utils import dispatch
from alpharidge_ai.utils.cooldown import MinerCooldownTracker


@pytest.fixture
def tracker():
    return MinerCooldownTracker(adaptive=True)


def test_dispatch_is_a_profile_field_and_defaults_off():
    from alpharidge_ai.mechanism import profile as mp
    from tests.test_profile_client import valid
    assert mp.parse(valid()).rations.dispatch is False

    raw = valid()
    raw["rations"]["dispatch"] = True
    assert mp.parse(raw).rations.dispatch is True


def test_dispatch_has_no_local_switch():
    assert not hasattr(config, "RATION_DISPATCH_ENABLED")
    assert "RATION_DISPATCH_ENABLED" not in config._REMOTE_CONFIG_KEYS


# ---- the adapter ------------------------------------------------------------------

def test_without_a_ration_source_nothing_changes(tracker):
    baseline = tracker.batch_size("hk")
    tracker.set_ration_source(lambda hk: None)
    assert tracker.batch_size("hk") == baseline


def test_a_ration_replaces_the_adaptive_batch_size(tracker):
    tracker.set_ration_source(lambda hk: 7.0)
    assert tracker.batch_size("hk") == min(7, tracker._bs_max())


def test_the_batch_stays_within_the_maximum(tracker):
    tracker.set_ration_source(lambda hk: 10_000.0)
    assert tracker.batch_size("hk") == tracker._bs_max()


def test_a_tiny_ration_still_leases_something(tracker):
    tracker.set_ration_source(lambda hk: 0.2)
    assert tracker.batch_size("hk") >= 1


def test_a_failing_ration_source_falls_back(tracker):
    baseline = tracker.batch_size("hk")

    def broken(hk):
        raise RuntimeError("no plan")
    tracker.set_ration_source(broken)
    assert tracker.batch_size("hk") == baseline


# ---- batches per epoch ------------------------------------------------------------

def test_one_batch_when_the_ration_fits_in_one(tracker):
    tracker.set_ration_source(lambda hk: float(tracker._bs_max()))
    assert tracker.batches_per_epoch("hk") == 1


def test_a_large_ration_buys_more_batches_not_a_bigger_one(tracker):
    cap = tracker._bs_max()
    tracker.set_ration_source(lambda hk: cap * 2.5)
    assert tracker.batch_size("hk") == cap
    assert tracker.batches_per_epoch("hk") == 3


def test_without_rations_a_uid_gets_one_batch(tracker):
    assert tracker.batches_per_epoch("hk") == 1


# ---- the slot limit ---------------------------------------------------------------

class Window:
    """Stands in for the adaptive window when rations are not in play."""

    def __init__(self, window):
        self._window = window

    def window(self, hotkey):
        return self._window


def test_the_slot_limit_follows_the_window_without_rations():
    assert dispatch._slot_limit(Window(4), "hk") == 4


def test_the_slot_limit_follows_the_ration_when_set(tracker):
    cap = tracker._bs_max()
    tracker._window = {"hk": 1.0}
    tracker.set_ration_source(lambda hk: cap * 3.0)
    assert dispatch._slot_limit(tracker, "hk") == 3


def test_the_slot_limit_is_at_least_one(tracker):
    tracker.set_ration_source(lambda hk: 0.0)
    assert dispatch._slot_limit(tracker, "hk") >= 1
