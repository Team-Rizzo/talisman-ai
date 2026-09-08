"""Where the emission curve comes from: profile when published, config otherwise."""

import pytest

import alpharidge_ai.config as config
from alpharidge_ai.mechanism import profile as mp
from alpharidge_ai.validator import emission_params as ep
from alpharidge_ai.validator import reputation
from tests.test_profile_client import valid


@pytest.fixture
def profile():
    return mp.parse(valid())


# ---- resolution --------------------------------------------------------------------

def test_without_a_profile_the_served_config_applies(monkeypatch):
    monkeypatch.setattr(config, "EMISSION_GAIN", 100.0, raising=False)
    monkeypatch.setattr(config, "EMISSION_MIDPOINT", 0.59, raising=False)
    params = ep.resolve(None)
    assert params.gain == 100.0 and params.midpoint == 0.59
    assert params.source == "config"


def test_a_published_profile_wins(monkeypatch, profile):
    monkeypatch.setattr(config, "EMISSION_GAIN", 100.0, raising=False)
    params = ep.resolve(profile)
    assert params.gain == 15.0
    assert params.midpoint == pytest.approx(0.766)
    assert params.n_min == 100
    assert "profile" in params.source


def test_the_arguments_match_what_the_curve_expects(profile):
    params = ep.resolve(profile)
    direct = reputation.emission(0.8, profile.emission.midpoint, profile.emission.gain,
                                 profile.emission.ceiling, profile.emission.bonus_start,
                                 profile.emission.bonus_full, n=500, n_min=100)
    through = reputation.emission(0.8, *params.as_args(), n=500, n_min=100)
    assert through == pytest.approx(direct)


def test_the_spread_widens_under_the_published_curve(profile):
    """The point of the flip: quality separates instead of everyone landing at 1.0."""
    served = ep.resolve(None)
    published = ep.resolve(profile)

    def spread(p):
        low = reputation.emission(0.67, *p.as_args(), n=500, n_min=p.n_min)
        high = reputation.emission(0.82, *p.as_args(), n=500, n_min=p.n_min)
        return high / max(low, 1e-9)

    assert spread(published) > spread(served)


def test_under_observed_hotkeys_stay_neutral_under_the_profile(profile):
    params = ep.resolve(profile)
    assert reputation.emission(0.30, *params.as_args(), n=1,
                               n_min=params.n_min) == 1.0


# ---- the median report -------------------------------------------------------------

def test_the_live_median_ignores_under_observed_hotkeys():
    snapshot = {"a": {"r": 0.70, "n": 500}, "b": {"r": 0.80, "n": 500},
                "c": {"r": 0.10, "n": 3}}
    assert ep.live_median(snapshot, 100) == pytest.approx(0.75)


def test_the_live_median_is_none_when_nobody_qualifies():
    assert ep.live_median({"a": {"r": 0.7, "n": 1}}, 100) is None
    assert ep.live_median({}, 100) is None
    assert ep.live_median(None, 100) is None


def test_the_median_is_reported_not_applied(profile):
    """Moving the midpoint is a publish, so validators move together."""
    snapshot = {f"h{i}": {"r": 0.90, "n": 500} for i in range(10)}
    median = ep.live_median(snapshot, 100)
    assert median == pytest.approx(0.90)
    assert ep.resolve(profile).midpoint == pytest.approx(0.766)
