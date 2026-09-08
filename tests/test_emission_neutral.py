"""Under-observed hotkeys take a neutral multiplier, not the cold-start prior.

At a steep gain the prior sits far below the midpoint, so gating an unobserved hotkey
pays it ~0 for having no history rather than for bad work.
"""

import pytest

from alpharidge_ai.validator import reputation as rep


GAIN = 15.0
MID = 0.766


def test_under_observed_is_neutral():
    assert rep.emission(0.58, MID, GAIN, n=0, n_min=100) == 1.0
    assert rep.emission(0.58, MID, GAIN, n=99, n_min=100) == 1.0


def test_at_threshold_the_curve_applies():
    scored = rep.emission(0.58, MID, GAIN, n=100, n_min=100)
    assert scored == pytest.approx(rep.emission(0.58, MID, GAIN))
    assert scored < 0.2


def test_prior_would_otherwise_cull():
    """Without the rule, a hotkey at the prior is paid near zero at gain 15."""
    assert rep.emission(0.58, MID, GAIN) < 0.15


def test_rule_is_opt_in():
    bare = rep.emission(0.58, MID, GAIN)
    assert rep.emission(0.58, MID, GAIN, n=None, n_min=100) == pytest.approx(bare)
    assert rep.emission(0.58, MID, GAIN, n=0, n_min=0) == pytest.approx(bare)


def test_neutral_does_not_mask_a_good_score():
    """A well-observed strong hotkey still earns above neutral."""
    good = rep.emission(0.85, MID, GAIN, 1.0, MID - 0.05, MID + 0.05, n=500, n_min=100)
    assert good > 1.0
