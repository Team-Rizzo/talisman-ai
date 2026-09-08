"""Fixes for the second review's findings."""

import types

import pytest

from alpharidge_ai.analyzer.article_intelligence_analyzer import (
    ArticleIntelligenceAnalyzer as A)
from alpharidge_ai.mechanism import profile as mp
from alpharidge_ai.oracle import runner
from alpharidge_ai.utils import dispatch
from alpharidge_ai.utils.cooldown import MinerCooldownTracker
from tests.test_profile_client import valid


# ---- the ration source is inert until its switch -----------------------------------

class Window:
    def __init__(self, window):
        self._w = window

    def window(self, hotkey):
        return self._w


def test_an_installed_but_inactive_ration_source_does_not_cap_dispatch():
    """Installed at startup, returns None until published. Keying on its mere presence
    capped every miner at one batch before the switch was ever set."""
    tracker = MinerCooldownTracker(adaptive=True)
    tracker._window = {"hk": 4.0}
    tracker.set_ration_source(lambda hk: None)          # installed, not in force
    assert dispatch._slot_limit(tracker, "hk") == 4     # the adaptive window, not 1


def test_an_active_ration_governs_the_slot_limit():
    tracker = MinerCooldownTracker(adaptive=True)
    tracker._window = {"hk": 1.0}
    tracker.set_ration_source(lambda hk: tracker._bs_max() * 3.0)
    assert dispatch._slot_limit(tracker, "hk") == 3


def test_a_tracker_with_no_ration_support_still_works():
    assert dispatch._slot_limit(Window(5), "hk") == 5


# ---- a fresh validator can take current and next together --------------------------

def _raw(version, publish, activation):
    raw = valid()
    raw["version"] = version
    raw["publish_block"] = publish
    raw["activation_block"] = activation
    return raw


def test_a_fresh_validator_adopts_the_active_profile():
    r = mp.ProfileResolver(refresh_seconds=3600)
    assert r.adopt(_raw(2, 1_000, 1_400), block=5_000)[0]
    assert r.resolve(5_000).version == 2


def test_the_active_profile_survives_a_staged_successor():
    """Routing both through offer() put the active one in the single next slot and then
    overwrote it, leaving nothing in force."""
    r = mp.ProfileResolver(refresh_seconds=3600)
    r.adopt(_raw(2, 1_000, 1_400), block=5_000)
    r.offer(_raw(3, 5_000, 9_000))

    assert r.resolve(5_000).version == 2      # still governed
    assert r.resolve(9_000).version == 3      # and switches on time


def test_a_future_profile_is_not_adopted_as_active():
    r = mp.ProfileResolver(refresh_seconds=3600)
    ok, reason = r.adopt(_raw(3, 5_000, 9_000), block=5_000)
    assert not ok and "not_yet_active" in reason


def test_adopting_an_older_version_is_refused():
    r = mp.ProfileResolver(refresh_seconds=3600)
    r.adopt(_raw(5, 1_000, 1_400), block=5_000)
    assert not r.adopt(_raw(4, 1_000, 1_400), block=5_000)[0]


# ---- unaligned quotes ---------------------------------------------------------------

TEXT = 'The CEO said "margins should recover" on Monday.'


def test_a_quote_that_cannot_be_located_is_not_emitted():
    """A 1.2.0 submission with null offsets fails the cutover gate for the whole
    article, so one unlocatable quote must not sink the rest."""
    quotes = A._build_quotes(A, [
        {"speaker": "CEO", "text": "margins should recover", "confidence": 0.9},
        {"speaker": "CFO", "text": "we will double the dividend", "confidence": 0.9},
    ], TEXT)
    assert len(quotes) == 1
    assert all(q.start_offset is not None for q in quotes)


def test_an_unaligned_quote_still_counts_against_precision():
    """Every submitted quote stays in the denominator."""
    miner = types.SimpleNamespace(quotes=[
        types.SimpleNamespace(text="margins should recover", confidence=0.9),
        types.SimpleNamespace(text="invented entirely", confidence=0.9)])
    grader = types.SimpleNamespace(quotes=[])
    keys, _, _ = runner._quote_keys(miner, grader, TEXT, {0: (14, 36)}, 40)
    assert len(keys) == 2                       # both submitted, both counted
    assert any(k[0] == "qx" for k in keys)      # the unaligned one is unsupported


# ---- durations are measurements ----------------------------------------------------

def kept(raw):
    return [(c.metric_name, c.value) for c in A._build_numeric_claims(A, raw)]


@pytest.mark.parametrize("metric,value,unit", [
    ("outage duration", 3.0, "hours"),
    ("delivery time", 2.0, "weeks"),
    ("contract length", 18.0, "months"),
])
def test_a_duration_is_kept(metric, value, unit):
    assert kept([{"metric_name": metric, "value": value, "unit": unit,
                  "confidence": 0.9}]) != []


def test_a_year_in_a_calendar_unit_is_still_dropped():
    assert kept([{"metric_name": "reporting period", "value": 2026, "unit": "year",
                  "confidence": 0.9}]) == []


def test_an_explicit_date_is_still_dropped():
    assert kept([{"metric_name": "published", "value": "2026-09-03", "unit": "date",
                  "confidence": 0.9}]) == []
