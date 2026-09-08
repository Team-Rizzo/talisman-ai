"""Ration store: persistence, and the profile gate on every economic value."""

import json
import types

import pytest

from alpharidge_ai.market.ration_store import RationStore
from alpharidge_ai.mechanism import profile as mp
from tests.test_profile_client import valid


@pytest.fixture
def profile():
    return mp.parse(valid())


@pytest.fixture
def store(tmp_path):
    return RationStore(path=tmp_path / "ration.json")


def test_without_a_profile_nothing_is_recorded(store):
    store.observe("hk", epoch=1, validated=10, dispatched=10, profile=None)
    assert store.book.states == {}


def test_without_a_profile_no_plan_is_produced(store):
    assert store.plan(["hk"], epoch=1, supply=100.0, profile=None) == {}


def test_a_profile_drives_the_observation(store, profile):
    store.observe("hk", epoch=1, validated=10, dispatched=10, profile=profile)
    assert store.book.states["hk"].ema > 0


def test_growth_keys_on_validated_work(store, profile):
    for epoch in range(50):
        store.observe("honest", epoch=epoch, validated=20, dispatched=20,
                      profile=profile)
        store.observe("padder", epoch=epoch, validated=0, dispatched=50,
                      profile=profile)
    assert store.book.states["padder"].ema == pytest.approx(0.0)
    assert store.book.states["honest"].ema > 0


def test_a_plan_never_exceeds_supply(store, profile):
    hotkeys = [f"m{i}" for i in range(30)]
    for epoch in range(20):
        for hk in hotkeys:
            store.observe(hk, epoch=epoch, validated=15, dispatched=15,
                          profile=profile)
    plan = store.plan(hotkeys, epoch=20, supply=200.0, profile=profile)
    assert sum(plan.values()) <= 200.0 + 1e-9
    assert len(plan) == len(hotkeys)


def test_state_survives_a_restart(store, profile, tmp_path):
    store.observe("hk", epoch=1, validated=10, dispatched=10, profile=profile)
    store.save()

    restored = RationStore(path=tmp_path / "ration.json")
    restored.load()
    assert restored.book.states["hk"].ema == pytest.approx(store.book.states["hk"].ema)


def test_a_corrupt_store_does_not_crash_startup(tmp_path):
    path = tmp_path / "ration.json"
    path.write_text("{not json")
    store = RationStore(path=path)
    store.load()
    assert store.book.states == {}


def test_departed_hotkeys_are_pruned(store, profile):
    store.observe("stays", epoch=1, validated=1, dispatched=1, profile=profile)
    store.observe("goes", epoch=1, validated=1, dispatched=1, profile=profile)
    store.prune(["stays"])
    assert set(store.book.states) == {"stays"}


def test_day_constants_reach_the_book_as_epoch_values(store, profile):
    """Conversion happens in the profile, never at the call site."""
    store.observe("hk", epoch=1, validated=72.0, dispatched=72.0, profile=profile)
    step = profile.rations.alpha_epoch
    assert store.book.states["hk"].ema == pytest.approx(72.0 * step)
    assert step < profile.rations.alpha_day
