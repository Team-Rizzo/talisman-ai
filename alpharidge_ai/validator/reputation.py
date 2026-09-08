"""Reputation integration and emission gate.

reputation: substantive-weighted recency EMA of per-article graded scores, in [0,1].
emission:   steep logistic floor of reputation, multiplied by volume elsewhere.

Params are injected (defaults match the validated config); the caller reads served
config and passes them in.
"""
from __future__ import annotations
import math

# defaults — override from served config
ALPHA = 0.03           # EMA step; half-life ~ ln(2)/alpha samples
PRIOR = 0.5            # cold-start value (below the floor => new state earns ~0 until proven)
                       # fallback only — reputation_store reads REPUTATION_PRIOR from served config
MIDPOINT = 0.59       # logistic centre; calibrate on the live distribution before gating
GAIN = 100.0          # logistic steepness


def update(prev, score, weight, alpha=ALPHA):
    """Single-step EMA update. `weight` scales the step (substantive samples move more).
    prev/score in [0,1]; returns the new reputation."""
    a = min(1.0, alpha * max(weight, 0.0))
    return (1.0 - a) * prev + a * score


def replay(scored, alpha=ALPHA, prior=PRIOR):
    """Recompute reputation from an ordered sequence of (score, weight). For backfill /
    verification; production updates incrementally with update()."""
    r = prior
    for s, w in scored:
        r = update(r, s, w, alpha)
    return r


def gate(reputation, midpoint=MIDPOINT, gain=GAIN):
    """Emission multiplier in (0,1). Steep => ~0 below the floor, ~1 above; the caller
    multiplies this by volume, so a sub-floor value earns ~0 at any volume."""
    return 1.0 / (1.0 + math.exp(-gain * (reputation - midpoint)))


def emission(reputation, midpoint=MIDPOINT, gain=GAIN,
             bonus_ceiling=0.0, bonus_start=0.63, bonus_full=0.75,
             n=None, n_min=0):
    """Emission multiplier: gate() times a bonus ramp above 1.0 between bonus_start and
    bonus_full. bonus_ceiling == 0 reproduces gate() exactly.

    While a hotkey holds fewer than `n_min` observations the multiplier is neutral 1.0.
    Too few samples is not evidence of bad work, and at a steep gain the cold-start
    prior sits far below the midpoint, so scoring it would read absence as failure.
    Pass n=None (or n_min=0) to disable the rule.
    """
    if n is not None and n_min > 0 and int(n) < int(n_min):
        return 1.0
    floor = gate(reputation, midpoint, gain)
    denom = bonus_full - bonus_start
    if denom <= 1e-9:
        # bonus_full <= bonus_start: step at bonus_start (avoid divide-by-zero).
        ramp = 1.0 if reputation >= bonus_start else 0.0
    else:
        ramp = min(1.0, max(0.0, (reputation - bonus_start) / denom))
    return floor * (1.0 + bonus_ceiling * ramp)
