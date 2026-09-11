"""Profile field bounds must be able to express the live configuration.

If a bound is tighter than the range a deployment may need, the profile cannot
mirror a running configuration, so "publish a no-op" becomes impossible and every
first publish carries an unintended economic change. The `emission.gain` bound was
one such case: its upper limit sat below values the curve is expected to take.
"""
import json
from pathlib import Path

import pytest

from alpharidge_ai.mechanism import profile as mp

_EXAMPLE = Path(__file__).resolve().parent.parent / "profiles" / "profile.example.json"


def _valid_body():
    body = json.loads(_EXAMPLE.read_text())
    body.setdefault("publish_block", 0)
    body.setdefault("activation_block", 0)
    body["oracle"]["grader_models"] = [{"id": "test/model", "weight": 1.0}]
    return body


def test_the_example_body_parses():
    assert mp.parse(_valid_body()).version >= 1


def test_the_gain_bound_covers_the_served_value():
    body = _valid_body()
    body["emission"]["gain"] = 100.0
    assert mp.parse(body).emission.gain == 100.0


def test_the_gain_bound_still_rejects_nonsense():
    body = _valid_body()
    body["emission"]["gain"] = 500.0
    with pytest.raises(mp.ProfileError):
        mp.parse(body)


@pytest.mark.parametrize("field,served", [
    ("midpoint", 0.57), ("ceiling", 0.0), ("bonus_start", 0.63),
    ("bonus_full", 0.75), ("gain", 100.0),
])
def test_every_served_emission_value_is_expressible(field, served):
    """The whole emission block, not just the field that prompted this."""
    body = _valid_body()
    body["emission"].update({"midpoint": 0.57, "gain": 100.0, "ceiling": 0.0,
                             "bonus_start": 0.63, "bonus_full": 0.75, "n_min": 100})
    body["emission"][field] = served
    assert getattr(mp.parse(body).emission, field) == served
