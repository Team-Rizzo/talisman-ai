"""The floor gate must be consulted on every lane, not just the legacy one.

Both floor reads previously sat inside the `else:` of the three-way branch on
`triage_active`, which the triage lanes do not reach, so the gate was consulted on
one lane only — and `_observe_ration`, in the same branch, ran on one lane only.

These tests come in two halves, because reaching the gate and honouring it are
separate failures.

**Reached** — a spy asserts the gate is consulted on all three lanes. Consultation
alone was the gap: a "gating withholds credit" test passes against a dead path, so
it cannot be the only evidence.

**Honoured** — a gated article is asserted to actually lose credit, per lane.
Consultation is necessary and not sufficient: a lane that consults the gate and pays
anyway satisfies every test in the first half.

Withholding is deliberately NOT uniform. The legacy lane pays nothing for a gated
article. The triage and verification lanes withhold the relevance component but still
pay the triage fee, which is assigned before the gate applies, so a gated article keeps
roughly 1-3%. The fee pays for the relevance decision, which the miner made whether or
not the analysis grounded. The assertions below encode that difference rather than
flattening it.
"""
import types

import pytest

from neurons.validator import Validator


class _Spy:
    """Stands in for the shared gate and records that it was consulted."""

    def __init__(self, blocked=()):
        self.blocked = {int(a) for a in blocked}
        self.consulted = []

    def __call__(self, aid):
        self.consulted.append(int(aid))
        return int(aid) in self.blocked


def _article(aid, content="x" * 600):
    return types.SimpleNamespace(id=aid, content=content, analysis=None,
                                 model_copy=lambda update=None: _article(aid, content))


def _triage_res(ids):
    return types.SimpleNamespace(
        relevant_ids=list(ids), borderline_valuable_ids=[], borderline_discard_ids=[],
        retire_candidate_ids=[], canary_ids=[])


def _validator(**over):
    v = types.SimpleNamespace()
    v._k_for = lambda aid: 1
    v._has_full_analysis = lambda a: True
    v._attribute_pay = lambda per, payout, record=True: per
    v._buffer_variants = lambda *a, **k: None
    v._triage_only_analysis = lambda a: None
    v._miner_reward = types.SimpleNamespace(add_reward=lambda hk, n: None,
                                            _get_current_epoch=lambda: 1)
    store = types.SimpleNamespace(
        update_article=lambda *a, **k: None, add_article=lambda *a, **k: None,
        set_processed=lambda *a, **k: None, reset_to_unprocessed=lambda *a, **k: None,
        mark_rewarded=lambda *a, **k: None, is_rewarded=lambda *a, **k: False)
    v._article_store = store
    v.__dict__.update(over)
    return v


def test_the_gate_is_consulted_on_a_triage_batch():
    spy = _Spy()
    v = _validator()
    batch = [_article(1), _article(2)]
    Validator._apply_triage_outcome(v, batch, "hk", _triage_res([1, 2]), set(),
                                    {}, full_push=True, gated_out=spy)
    assert spy.consulted, "triage lane never consulted the floor gate"


def test_the_gate_is_consulted_on_a_verification_batch():
    spy = _Spy()
    v = _validator()
    batch = [_article(3)]
    Validator._apply_verification_outcome(v, batch, "hk", _triage_res([3]), set(),
                                          None, None, 1, gated_out=spy)
    assert spy.consulted, "verification lane never consulted the floor gate"


def test_the_lanes_keep_their_own_pay_rules():
    """The gate is shared; the payout is not. Folding pay into the shared method
    would silently reprice the triage lanes."""
    import inspect
    src = inspect.getsource(Validator._credit_gate)
    assert "add_reward" not in src, "_credit_gate must not pay; lanes price differently"
    assert "_observe_ration" in src, "_credit_gate must observe the ration"


# ---------------------------------------------------------------------------
# Consultation is necessary, not sufficient. The tests above prove the gate is
# reached; these prove its answer is honoured. A lane that consults the gate and
# then pays anyway passes every test above.
# ---------------------------------------------------------------------------


def _paid(fn, *args, **kwargs):
    """Run one lane and return what it awarded."""
    out = []
    v = _validator()
    v._miner_reward = types.SimpleNamespace(add_reward=lambda hk, n: out.append(n),
                                            _get_current_epoch=lambda: 1)
    v._article_pay = {}
    v._canary_pool = types.SimpleNamespace(label_of=lambda aid: None)
    fn(v, *args, **kwargs)
    return sum(out)


def test_the_gate_is_consulted_on_a_legacy_batch():
    """The legacy lane works today, which is exactly when a path stops being
    tested and starts being assumed."""
    spy = _Spy()
    v = _validator()
    v._article_pay = {}
    v._canary_pool = types.SimpleNamespace(label_of=lambda aid: None)
    Validator._apply_legacy_outcome(v, [_article(11), _article(12)], "hk", gated_out=spy)
    assert spy.consulted, "legacy lane never consulted the floor gate"


def test_a_gated_article_withholds_on_the_legacy_lane():
    open_pay = _paid(Validator._apply_legacy_outcome, [_article(21)], "hk",
                     gated_out=_Spy())
    gated_pay = _paid(Validator._apply_legacy_outcome, [_article(21)], "hk",
                      gated_out=_Spy(blocked=[21]))
    assert open_pay > 0, "ungated legacy article should pay"
    assert gated_pay == 0, "gated legacy article still paid"


def test_a_gated_article_withholds_on_the_verification_lane():
    open_pay = _paid(Validator._apply_verification_outcome, [_article(31)], "hk",
                     _triage_res([31]), set(), None, None, 1, gated_out=_Spy())
    gated_pay = _paid(Validator._apply_verification_outcome, [_article(31)], "hk",
                      _triage_res([31]), set(), None, None, 1,
                      gated_out=_Spy(blocked=[31]))
    assert open_pay > gated_pay, "gating a verification article did not reduce its pay"


def test_a_gated_article_withholds_on_the_triage_lane():
    open_pay = _paid(Validator._apply_triage_outcome, [_article(41)], "hk",
                     _triage_res([41]), set(), {}, full_push=False, gated_out=_Spy())
    gated_pay = _paid(Validator._apply_triage_outcome, [_article(41)], "hk",
                      _triage_res([41]), set(), {}, full_push=False,
                      gated_out=_Spy(blocked=[41]))
    assert open_pay > gated_pay, "gating a triage article did not reduce its pay"


def _canary_res(ids):
    return types.SimpleNamespace(
        relevant_ids=list(ids), borderline_valuable_ids=[], borderline_discard_ids=[],
        retire_candidate_ids=[], canary_ids=list(ids))


def test_a_gated_canary_is_withheld_on_the_triage_lane():
    """Deliberate, not a side effect of making the gate reachable.

    The floor check previously existed only in the legacy lane, and that lane skips
    canaries outright — so an analysed canary positive was paid without ever clearing
    the floor. It now clears the floor like any other article.

    Narrow in practice: `blocked` carries only ids the floor explicitly judged False,
    so a canary the floor never swept is unaffected. And as everywhere on this lane the
    triage fee is still paid, because it is assigned before the gate applies.
    """
    open_pay = _paid(Validator._apply_triage_outcome, [_article(51)], "hk",
                     _canary_res([51]), set(), {}, full_push=False, gated_out=_Spy())
    gated_pay = _paid(Validator._apply_triage_outcome, [_article(51)], "hk",
                      _canary_res([51]), set(), {}, full_push=False,
                      gated_out=_Spy(blocked=[51]))
    assert open_pay > gated_pay, "a gated canary positive still earned the relevance component"
