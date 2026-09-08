"""Property tests for the floor, adjudication, scoring and ration modules.

These drive the real modules rather than a model of them, so the properties below hold
of the code that runs rather than of a description of it.
"""

import random
import types

import pytest

from alpharidge_ai.mechanism import rations, scoring, settlement
from alpharidge_ai.oracle import audit, floor

ARTICLE = ('Revenue rose to $1.2 billion in the quarter, up 12.5% year on year. '
           'The chief executive said "margins should recover in the second half" '
           'on Monday. Headcount reached 1,203 and the vote was 100 to 20. '
           'Ventas de 1.203.400 unidades. 매출은 3억 원.')


def claim(value, unit="count", metric="m", confidence=0.9):
    return types.SimpleNamespace(metric_name=metric, value=value, unit=unit,
                                 context="", confidence=confidence)


def intel(claims=()):
    return types.SimpleNamespace(numeric_claims=list(claims), quotes=[], assets=[],
                                 entities=[], schema_version="1.2.0")


def granted(submitted, reference=(), text=ARTICLE, adjudicator=None):
    """Everything the validator accepts without a grader saying so."""
    result = floor.evaluate(intel(submitted), text)
    decided = audit.adjudicate(submitted, list(reference), result.grounded, text,
                               adjudicator)
    return decided, result


# ---- values a submitter might construct from what is in the text -------------------

def test_no_route_grants_a_value_absent_from_the_article():
    attacks = [
        claim(120.0),                    # 100 + 20, adjacent numbers
        claim(1.0, "million"),           # unscaled unit against a bare 1
        claim(1.203),                    # alternate reading of 1,203
        claim(1e9),                      # from the word "billion"
        claim(3.0, "KRW"),               # 3억 read as 3
        claim(1220.0),                   # 1203 plus a neighbour
    ]
    decided, _ = granted(attacks)
    assert decided.valid == set(), f"granted with no grader: {decided.valid}"


def test_the_reference_route_grants_nothing_a_scaled_unit_asserts():
    # `1.2 billion USD` is deliberately absent: it equals the reference and SHOULD
    # match. The property concerns claims asserting a different magnitude.
    reference = [claim(1.0, "count", "headcount"), claim(1.2e9, "USD", "revenue")]
    attacks = [claim(1.0, "million", "headcount"),
               claim(1.2e9, "million USD", "revenue"),
               claim(1.2e9, "billion USD", "revenue")]
    decided, _ = granted(attacks, reference)
    assert decided.valid == set()
    assert all(k[0] == "m" for k in decided.miner_keys), decided.miner_keys


def test_honest_claims_survive_all_of_it():
    """The properties above must not have been bought by rejecting real work."""
    honest = [claim(1.2e9, "USD", "revenue"), claim(12.5, "%", "growth"),
              claim(1203.0, "count", "headcount"), claim(300000000.0, "KRW", "sales")]
    decided, result = granted(honest)
    assert len(decided.valid) >= 3, (decided.valid, result.inferred)


def test_an_equivalent_claim_written_differently_still_matches():
    decided, _ = granted([claim(1.0, "million", "units")], [claim(1e6, "count", "units")])
    assert decided.miner_keys == [("g", 0)]


# ---- adjudication ------------------------------------------------------------------

def test_a_grader_asserting_support_without_evidence_is_ignored():
    def liar(text, batch):
        return [{"i": c["i"], "supported": True, "evidence": "the company said so"}
                for c in batch]
    assert granted([claim(9.9e9, "USD")], adjudicator=liar)[0].valid == set()


def test_malformed_grader_output_never_grants():
    for reply in (None, "garbage", [], [{"i": 99, "supported": True, "evidence": "x"}],
                  [{"supported": True}], {"claims": "nope"}):
        decided, _ = granted([claim(9.9e9, "USD")], adjudicator=lambda t, b, r=reply: r)
        assert decided.valid == set(), reply


def test_a_grader_that_raises_grants_nothing():
    def broken(text, batch):
        raise RuntimeError("model down")
    assert granted([claim(9.9e9, "USD")], adjudicator=broken)[0].valid == set()


# ---- scoring -----------------------------------------------------------------------

def test_padding_never_beats_an_honest_submission():
    gold = {("g", 0), ("g", 1), ("g", 2)}
    honest = scoring.article_score(sorted(gold), gold, (), {k: 0.9 for k in gold})
    for extra in (1, 5, 20, 100):
        keys = sorted(gold) + [("m", i) for i in range(extra)]
        padded = scoring.article_score(keys, gold, (), {k: 0.9 for k in keys})
        assert padded.normalized <= honest.normalized, extra


def test_flat_confidence_never_beats_discrimination():
    gold = {("g", 0), ("g", 1)}
    keys = sorted(gold) + [("m", 0), ("m", 1)]
    honest = scoring.article_score(keys, gold, (),
                                   {("g", 0): 0.95, ("g", 1): 0.95,
                                    ("m", 0): 0.05, ("m", 1): 0.05})
    for p in (0.0, 0.01, 0.5, 0.99, 1.0):
        flat = scoring.article_score(keys, gold, (), {k: p for k in keys})
        assert flat.normalized <= honest.normalized + 1e-9, p


def test_a_score_never_leaves_the_observation_range():
    rng = random.Random(11)
    gold = {("g", i) for i in range(3)}
    for _ in range(300):
        keys = [("g", rng.randrange(3)) for _ in range(rng.randrange(0, 6))] + \
               [("m", i) for i in range(rng.randrange(0, 8))]
        s = scoring.article_score(keys, gold, (), {k: rng.random() for k in keys})
        if s is not None:
            assert 0.0 <= s.normalized <= 1.0
            assert 0.0 <= s.precision <= 1.0 and 0.0 <= s.recall <= 1.0


def test_repeating_one_true_claim_collects_credit_once():
    decided, _ = granted([claim(1.2e9, "USD", "revenue")] * 8,
                         [claim(1.2e9, "USD", "revenue")])
    assert decided.miner_keys.count(("g", 0)) == 1
    s = scoring.article_score(decided.miner_keys, decided.grader_keys, decided.valid, {})
    assert s.recall <= 1.0 and s.precision <= 1.0


# ---- rations ------------------------------------------------------------------------

EPOCHS = 72
ALPHA_E = 1.0 - (1.0 - 0.5) ** (1.0 / EPOCHS)
PROBE_E = 2.0 ** (1.0 / EPOCHS)
CAP_E, EXPLORE_E, GATE = 5000.0 / EPOCHS, 25.0 / EPOCHS, 0.97


def drive(validated, dispatched, epochs=300):
    state = None
    for e in range(epochs):
        state = rations.observe(state, epoch=e, validated=validated,
                                dispatched=dispatched, alpha_epoch=ALPHA_E)
    return state


def want(state):
    return rations.want(state, probe_epoch=PROBE_E, fill_gate=GATE, cap_epoch=CAP_E)


def test_filling_slots_with_failing_work_earns_no_ration():
    assert want(drive(0.0, 50.0)) == pytest.approx(0.0)


def test_partial_junk_earns_only_the_valid_part():
    assert drive(20.0, 50.0).ema == pytest.approx(drive(20.0, 20.0).ema)


def test_unfilled_slots_block_the_probe():
    assert not rations.is_saturated(drive(20.0, 50.0), GATE)


def test_bursty_delivery_does_not_beat_steady_delivery():
    steady = drive(30.0, 30.0)
    bursty = None
    for e in range(300):
        on = (e % 10 == 0)
        bursty = rations.observe(bursty, epoch=e, validated=300.0 if on else 0.0,
                                 dispatched=300.0 if on else 30.0, alpha_epoch=ALPHA_E)
    assert not rations.is_saturated(bursty, GATE)
    assert want(bursty) <= want(steady) * 1.5


@pytest.mark.parametrize("n", [2, 3, 5, 10, 40])
def test_splitting_capacity_across_uids_never_pays_more(n):
    solo = rations.allocate({"solo": 600.0}, {"solo": EXPLORE_E}, supply=1000.0,
                            explore_epoch=EXPLORE_E)["solo"]
    keys = [f"s{i}" for i in range(n)]
    split = rations.allocate({k: 600.0 / n for k in keys},
                             {k: EXPLORE_E for k in keys}, supply=1000.0,
                             explore_epoch=EXPLORE_E)
    assert sum(split.values()) <= solo + n * EXPLORE_E + 1e-9


def test_a_registration_wave_cannot_starve_an_incumbent():
    wants, floors = {"incumbent": 400.0}, {"incumbent": EXPLORE_E}
    for i in range(500):
        wants[f"n{i}"], floors[f"n{i}"] = 0.0, 200.0 / EPOCHS
    given = rations.allocate(wants, floors, supply=1000.0, explore_epoch=EXPLORE_E,
                             boost_tranche_max=0.05)
    assert given["incumbent"] > 0.5 * 400.0


def test_allocation_never_exceeds_supply_under_any_demand():
    rng = random.Random(7)
    for _ in range(200):
        keys = [f"m{i}" for i in range(rng.randrange(1, 40))]
        wants = {k: rng.choice([0.0, rng.random() * 5000]) for k in keys}
        floors = {k: rng.choice([0.0, EXPLORE_E, 200.0 / EPOCHS]) for k in keys}
        supply = rng.choice([0.0, 1.0, 500.0, 100000.0])
        given = rations.allocate(wants, floors, supply, explore_epoch=EXPLORE_E,
                                 boost_tranche_max=0.05)
        assert sum(given.values()) <= supply + 1e-6
        assert all(v >= 0 for v in given.values())


# ---- settlement --------------------------------------------------------------------

def test_pay_and_burn_always_account_for_the_whole_emission():
    rng = random.Random(3)
    for _ in range(500):
        work = {f"m{i}": rng.choice([0.0, rng.random() * 5000])
                for i in range(rng.randrange(0, 30))}
        result = settlement.settle(work, rng.choice([1.0, 1000.0, 26697.0]))
        assert 0.0 <= result.burn <= 1.0
        assert result.paid + result.burn == pytest.approx(1.0)


def test_no_uid_is_paid_more_by_splitting_its_work():
    rng = random.Random(5)
    for _ in range(200):
        total, capacity = rng.random() * 5000, rng.choice([100.0, 1000.0, 26697.0])
        others = {"other": rng.random() * 2000}
        solo = settlement.settle({"solo": total, **others}, capacity).shares["solo"]
        n = rng.randrange(2, 8)
        # The rest of the field has to be present on both sides, or the denominators
        # differ and the comparison says nothing.
        split = settlement.settle({**{f"s{i}": total / n for i in range(n)}, **others},
                                  capacity)
        assert sum(split.shares[f"s{i}"] for i in range(n)) <= solo + 1e-9


# ---- a search, rather than a list --------------------------------------------------
# The cases above are specific inputs. This generates values arithmetically derivable
# from the article that the article does not state, and asserts none is granted —
# the same question asked without a prior list.

CORPORA = [
    'Revenue rose to $1.2 billion, up 12.5%. Headcount reached 1,203.',
    'The vote was 100 to 20. Turnout was 45 percent of 3,000 members.',
    'Ventas de 1.203.400 unidades y 17,99 millones de euros este ano.',
    '매출은 3억 원, 거래액 1조 2천억 원을 기록했다.',
    'Six children and twenty-five adults were rescued over 3 hours.',
    'Rund 20 Millionen Barrel und 1.234,56 Euro pro Tonne.',
]


def _literal_values(text):
    """What the article states, per the parser's own literal reading."""
    return {round(n.value, 6) for n in floor.parse_numbers(floor.normalize(text).text)
            if n.literal}


def _derived_candidates(literals):
    """Values derivable from what the article states, that it does not state.

    Anything inside the documented ±0.5% tolerance is excluded: a claim of 1,200
    against a stated 1,203 is a rounded restatement, which §E18 grounds on purpose.
    That window is small, bounded, and specified — see the test below that pins it.
    """
    out = set()
    vals = sorted(literals)
    for a in vals:
        for b in vals:
            if a == b:
                continue
            out.update({a + b, a - b, a * b})
        out.update({a * 1e3, a * 1e6, a * 1e9, a / 1e3, a / 1e6,
                    a * 10, a / 10, a + 1, a - 1})

    def within_tolerance(v):
        return any(abs(v - l) / max(abs(l), 1e-9) <= floor.VALUE_REL_TOLERANCE
                   for l in literals)

    return {round(v, 6) for v in out
            if v == v and abs(v) != float("inf") and not within_tolerance(v)}


@pytest.mark.parametrize("text", CORPORA)
def test_no_derived_value_is_granted_without_a_grader(text):
    literals = _literal_values(text)
    candidates = _derived_candidates(literals)
    assert candidates, "the generator produced nothing to test"

    submitted = [claim(v) for v in sorted(candidates)[:400]]
    result = floor.evaluate(intel(submitted), text)
    decided = audit.adjudicate(submitted, [], result.grounded, text, None)

    leaked = sorted(submitted[i].value for kind, i in decided.valid if kind == "m")
    assert not leaked, f"granted without a grader in {text[:40]!r}: {leaked[:5]}"


@pytest.mark.parametrize("text", CORPORA)
def test_every_stated_value_is_still_grantable(text):
    """The search above must not pass because the floor grants nothing at all.

    Each value is claimed with the unit the text gives it — a count is not an amount of
    won, and refusing that mismatch is correct rather than a gap.
    """
    parsed = [n for n in floor.parse_numbers(floor.normalize(text).text) if n.literal]
    submitted = [claim(n.value, _claim_unit(n.unit)) for n in parsed]
    result = floor.evaluate(intel(submitted), text)
    decided = audit.adjudicate(submitted, [], result.grounded, text, None)
    assert len(decided.valid) >= len(parsed) * 0.5, (decided.valid, len(parsed))


def _claim_unit(parsed_unit):
    """The unit string a submitter would write for a parsed unit class."""
    if parsed_unit == "pct":
        return "%"
    if parsed_unit.startswith("currency:"):
        return parsed_unit.split(":", 1)[1]
    return "count"


@pytest.mark.parametrize("text", CORPORA)
def test_unit_variants_of_stated_values_are_not_granted(text):
    """The same number wearing a scale it does not have in the text."""
    submitted = [claim(v, u) for v in sorted(_literal_values(text))[:40]
                 for u in ("million", "billion", "thousand")]
    result = floor.evaluate(intel(submitted), text)
    decided = audit.adjudicate(submitted, [], result.grounded, text, None)
    assert decided.valid == set(), sorted(decided.valid)[:5]


def test_the_grounding_tolerance_is_bounded_and_deliberate():
    """A rounded restatement grounds; that is specified, and this pins how far it goes.

    §E18 sets ±0.5%. The search above excludes that window, so if the tolerance ever
    widens silently this is what says so.
    """
    assert floor.VALUE_REL_TOLERANCE == 0.005

    text = "Headcount reached 1,203 this year."
    numbers = floor.parse_numbers(floor.normalize(text).text)
    inside = 1203 * (1 + floor.VALUE_REL_TOLERANCE * 0.9)
    outside = 1203 * (1 + floor.VALUE_REL_TOLERANCE * 3)
    assert floor.ground_claim(claim(inside), numbers)
    assert not floor.ground_claim(claim(outside), numbers)


# ---- invariants ------------------------------------------------------------------
#
# The cases above are individual inputs. These state the general property over generated
# ones: what a claim asserts is what the article has to state.
#
# The scale table is written out again rather than imported: a check that reads the
# same table as the code cannot catch a wrong entry in it.

SCALES = {
    "": 1.0, "%": 1.0, "pct": 1.0, "percent": 1.0,
    "bps": 0.01, "bp": 0.01, "basis points": 0.01, "basis_points": 0.01,
    "thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12,
    "lakh": 1e5, "crore": 1e7,
    "barrels": 1.0, "users": 1.0, "vehicles": 1.0,
}

_SCALE_WORDS = ("", "thousand", "million", "billion", "crore", "lakh")
_VALUES = (5, 12.5, 50, 250, 3.2, 1000, 1.203)


def _asserts(value, unit):
    """What a claim says, in one scale, computed independently of the floor."""
    return value * SCALES[unit]


@pytest.mark.parametrize("seed", (11, 12, 13, 14))
def test_a_literal_grounding_means_the_article_states_that_figure(seed):
    """A literal grounding may only be reached when the figure the claim asserts is
    the figure the article states."""
    rng = random.Random(seed)
    for _ in range(250):
        stated = rng.choice(_VALUES)
        article_scale = rng.choice(_SCALE_WORDS)
        text = f"The company reported a figure of {stated} {article_scale} today."
        article_figure = stated * SCALES[article_scale]

        value = rng.choice(_VALUES)
        unit = rng.choice(list(SCALES))
        result = floor.evaluate(intel([claim(value, unit)]), text)
        if not result.grounded:
            continue

        asserted = _asserts(value, unit)
        assert abs(asserted - article_figure) <= abs(article_figure) * 0.01, (
            f"{value} {unit!r} asserts {asserted}, article states {article_figure}")


@pytest.mark.parametrize("seed", (21, 22, 23))
def test_two_claims_match_only_when_they_assert_the_same_figure(seed):
    """The same property, on the reference comparison."""
    rng = random.Random(seed)
    units = list(SCALES)
    for _ in range(400):
        lv, lu = rng.choice(_VALUES), rng.choice(units)
        rv, ru = rng.choice(_VALUES), rng.choice(units)
        if not audit.claims_match(claim(lv, lu), claim(rv, ru)):
            continue
        left, right = _asserts(lv, lu), _asserts(rv, ru)
        assert abs(left - right) <= abs(right) * 0.01, (
            f"{lv} {lu!r} ({left}) matched {rv} {ru!r} ({right})")


def test_a_claim_beyond_the_cap_earns_nothing():
    """The cap bounds what one submission can put in front of the validator; claims
    past it must not reach validity."""
    stated = [claim(v, "") for v in (5, 12.5, 50)]
    padding = [claim(5, "") for _ in range(60)]
    result = floor.evaluate(intel(padding + stated), ARTICLE, claim_cap=40)
    assert max(result.grounded | result.inferred | result.ungrounded) < 40

    decided, _ = granted(padding + stated)
    assert all(key[1] < 40 for key in decided.valid if key[0] == "m")


def test_the_cap_is_applied_before_scoring_not_after():
    scored = scoring.article_score([("m", i) for i in range(100)], {("m", 0)},
                                   claim_cap=40)
    assert scored.scored_claims == 40
