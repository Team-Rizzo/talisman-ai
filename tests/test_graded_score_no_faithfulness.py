"""The graded score is the composite alone.

Reference-free faithfulness takes no reference, so its value is computable by the
submitter offline. While it was the binding term of a min(), a submitter could set its
own score on any article where it was the lower of the two.
"""

import inspect

from alpharidge_ai.validator import quality


def test_graded_score_takes_no_faithfulness_argument():
    assert "faith" not in inspect.signature(quality.graded_score).parameters


def test_graded_score_returns_the_composite(monkeypatch):
    monkeypatch.setattr(quality, "composite", lambda *a, **kw: 0.82)
    assert quality.graded_score({}, {}, {}, None, None) == 0.82


def test_graded_score_is_clamped(monkeypatch):
    monkeypatch.setattr(quality, "composite", lambda *a, **kw: -0.5)
    assert quality.graded_score({}, {}, {}, None, None) == 0.0
    monkeypatch.setattr(quality, "composite", lambda *a, **kw: 1.4)
    assert quality.graded_score({}, {}, {}, None, None) == 1.0


def test_faithfulness_remains_available_as_a_gate():
    assert hasattr(quality, "Faithfulness")
    from alpharidge_ai.validator.graded_scorer import GradedScorer
    assert hasattr(GradedScorer, "faithfulness")
