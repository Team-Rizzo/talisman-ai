"""Capacity is per epoch; rewards span the weight window. They must be compared on the
same span, or burn becomes a function of how long the window is rather than of work."""

import types

import pytest

import alpharidge_ai.config as config
from alpharidge_ai.mechanism import profile as mp
from alpharidge_ai.mechanism import settlement
from alpharidge_ai.validator import validation_client as vc
from tests.test_profile_client import valid


class Metagraph:
    def __init__(self, hotkeys):
        self.hotkeys = list(hotkeys)
        self.n = len(hotkeys)


def _client(profile, hotkeys=("a", "b", "burn")):
    validator = types.SimpleNamespace(
        metagraph=Metagraph(hotkeys), block=1_000_000,
        _mechanism_profile=types.SimpleNamespace(resolve=lambda b: profile))
    client = object.__new__(vc.ValidationClient)
    client._validator = validator
    return client


def reward(hotkey, points):
    return types.SimpleNamespace(hotkey=hotkey, reward=points)


@pytest.fixture
def live_profile():
    raw = valid()
    raw["settlement"]["live"] = True
    raw["settlement"]["C"] = 1000.0
    return mp.parse(raw)


def test_burn_does_not_change_with_window_length(live_profile, monkeypatch):
    """The same work per epoch must burn the same whatever K is."""
    monkeypatch.setattr(config, "BURN_UID", 2, raising=False)
    monkeypatch.setattr(config, "BLOCK_LENGTH", 100, raising=False)
    client = _client(live_profile)

    one = client._weights_for([reward("a", 500)], 100, "")          # 1 epoch
    five = client._weights_for([reward("a", 2500)], 500, "")        # 5 epochs, same rate

    assert one[2] == pytest.approx(five[2], abs=1e-9)   # identical burn


def test_a_longer_window_does_not_manufacture_burn(live_profile, monkeypatch):
    monkeypatch.setattr(config, "BURN_UID", 2, raising=False)
    monkeypatch.setattr(config, "BLOCK_LENGTH", 100, raising=False)
    client = _client(live_profile)

    full = client._weights_for([reward("a", 1000)], 100, "")        # exactly at capacity
    full_long = client._weights_for([reward("a", 10_000)], 1000, "")

    assert full[2] == pytest.approx(0.0, abs=1e-9)
    assert full_long[2] == pytest.approx(0.0, abs=1e-9)


def test_the_vector_still_sums_to_one(live_profile, monkeypatch):
    monkeypatch.setattr(config, "BURN_UID", 2, raising=False)
    monkeypatch.setattr(config, "BLOCK_LENGTH", 100, raising=False)
    weights = _client(live_profile)._weights_for(
        [reward("a", 300), reward("b", 200)], 500, "")
    assert sum(weights) == pytest.approx(1.0)


def test_the_legacy_path_is_used_when_the_switch_is_off(monkeypatch):
    monkeypatch.setattr(config, "BURN_UID", 2, raising=False)
    profile = mp.parse(valid())          # settlement.live defaults False
    called = {}
    monkeypatch.setattr(vc, "calculate_weights",
                        lambda *a, **k: called.setdefault("legacy", True) or [0.0, 0, 1])
    _client(profile)._weights_for([reward("a", 100)], 100, "")
    assert called.get("legacy")
