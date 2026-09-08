"""The flip: audit observations replace the old scorer's, at one block."""

import pytest

from alpharidge_ai.mechanism import profile as mp
from tests.test_profile_client import valid


def _profile(**switches):
    raw = valid()
    for section, key, value in switches.get("set", []):
        raw[section][key] = value
    return mp.parse(raw)


def test_the_switches_that_change_economics_default_off():
    p = mp.parse(valid())
    assert p.oracle.live is False
    assert p.settlement.live is False
    assert p.rations.dispatch is False


def test_floor_gating_is_the_exception_and_defaults_on():
    """It only stops paying for work that failed a check, so it ships early and alone."""
    assert mp.parse(valid()).settlement.floor_gating is True


def test_the_switches_arrive_together_at_one_block():
    """A single activation carries the whole flip, so no half state exists."""
    before = mp.parse(valid())
    raw = valid()
    raw["version"] = 3
    raw["publish_block"] = 19_000
    raw["activation_block"] = 20_000
    raw["oracle"]["live"] = True
    raw["settlement"]["live"] = True
    raw["rations"]["dispatch"] = True

    resolver = mp.ProfileResolver(current=before, refresh_seconds=3600)
    assert resolver.offer(raw)[0]

    at = resolver.resolve(19_999)
    assert (at.oracle.live, at.settlement.live, at.rations.dispatch) == \
        (False, False, False)

    after = resolver.resolve(20_000)
    assert (after.oracle.live, after.settlement.live, after.rations.dispatch) == \
        (True, True, True)


def test_a_non_boolean_switch_is_refused():
    for section, key in (("oracle", "live"), ("settlement", "floor_gating"),
                         ("rations", "dispatch")):
        raw = valid()
        raw[section][key] = "yes"
        with pytest.raises(mp.ProfileError):
            mp.parse(raw)


def test_rolling_the_flip_back_is_a_forward_publish():
    raw = valid()
    raw["oracle"]["live"] = True
    live = mp.parse(raw)
    resolver = mp.ProfileResolver(current=live, refresh_seconds=3600)

    back = valid()
    back["version"] = 9
    back["publish_block"] = 30_000
    back["activation_block"] = 31_000
    assert resolver.offer(back)[0]
    assert resolver.resolve(31_000).oracle.live is False
