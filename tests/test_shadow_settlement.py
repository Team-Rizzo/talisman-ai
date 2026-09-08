"""The settlement shadow: computed and logged, never applied."""

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


def _client(profile, hotkeys, block=1_000_000):
    validator = types.SimpleNamespace(
        metagraph=Metagraph(hotkeys),
        block=block,
        _mechanism_profile=types.SimpleNamespace(resolve=lambda b: profile),
    )
    client = object.__new__(vc.ValidationClient)
    client._validator = validator
    return client


def reward(hotkey, points):
    return types.SimpleNamespace(hotkey=hotkey, reward=points)


@pytest.fixture
def profile():
    return mp.parse(valid())


def test_shadow_runs_without_touching_the_live_weights(profile, monkeypatch):
    monkeypatch.setattr(config, "BURN_UID", 2, raising=False)
    live = [0.4, 0.3, 0.3]
    before = list(live)
    client = _client(profile, ["a", "b", "burn"])

    client._shadow_settlement([reward("a", 100), reward("b", 50)], live, 7)
    assert live == before


def test_shadow_is_silent_without_a_profile(monkeypatch):
    monkeypatch.setattr(config, "BURN_UID", 2, raising=False)
    client = _client(None, ["a", "b", "burn"])
    client._shadow_settlement([reward("a", 100)], [0.5, 0.0, 0.5], 7)


def test_shadow_never_raises_on_a_broken_reward_list(profile, monkeypatch):
    monkeypatch.setattr(config, "BURN_UID", 2, raising=False)
    client = _client(profile, ["a", "b", "burn"])
    client._shadow_settlement([types.SimpleNamespace(hotkey="a")], [0.5, 0.0, 0.5], 7)


def test_the_shadow_vector_is_a_valid_weight_vector(profile, monkeypatch):
    monkeypatch.setattr(config, "BURN_UID", 2, raising=False)
    work = {"a": 100.0, "b": 50.0}
    result = settlement.settle(work, profile.settlement.C)
    vector = settlement.weight_vector(result.shares, result.burn,
                                      ["a", "b", "burn"], 2, 3)
    assert sum(vector) == pytest.approx(1.0)
    assert all(w >= 0 for w in vector)
