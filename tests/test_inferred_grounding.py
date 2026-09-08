"""A reading the parser inferred is not evidence the article states the figure.

Derived candidates — the second reading of an ambiguous grouping mark, and CJK compound
sums — keep honest claims from being denied. They are adjudicated, not granted.
"""

import types

import pytest

from alpharidge_ai.oracle import audit, floor


def kind(text, value, unit="count"):
    return floor.ground_kind(types.SimpleNamespace(value=value, unit=unit),
                             floor.parse_numbers(floor.normalize(text).text))


def claim(value, unit="count", metric="m"):
    return types.SimpleNamespace(metric_name=metric, value=value, unit=unit, context="")


# ---- no sum where no scale mark ----------------------------------------------------

def test_ordinary_adjacent_numbers_are_not_summed():
    """'100 and 20' contains no 120, in any reading."""
    assert 120.0 not in [n.value for n in
                         floor.parse_numbers(floor.normalize("100 and 20 people").text)]
    assert kind("100 and 20 people", 120.0) is None


def test_a_vote_tally_cannot_be_summed_into_a_claim():
    assert kind("The vote was 100 to 20.", 120.0) is None


def test_the_cjk_compound_still_matches_but_only_as_inferred():
    assert kind("거래액 1조 2천억 원", 1.2e12, "KRW") == "inferred"


def test_the_cjk_parts_remain_literal():
    assert kind("거래액 1조 2천억 원", 1.0e12, "KRW") == "literal"


# ---- ambiguous locale readings -----------------------------------------------------

def test_the_first_reading_is_literal():
    assert kind("Cost 1,203 euros.", 1203.0, "EUR") == "literal"


def test_the_alternate_reading_is_inferred():
    assert kind("Cost 1,203 euros.", 1.203, "EUR") == "inferred"


def test_a_decimal_comma_still_reads():
    assert kind("Le prix atteint 17,99 euros.", 17.99, "EUR") is not None


def test_an_invented_value_grounds_neither_way():
    assert kind("Revenue was $1.2 billion.", 9.9e9, "USD") is None


# ---- what the floor reports --------------------------------------------------------

def _intel(claims):
    return types.SimpleNamespace(numeric_claims=list(claims), quotes=[], assets=[],
                                 entities=[])


def test_the_floor_separates_literal_from_inferred():
    res = floor.evaluate(_intel([claim(1203.0, "EUR"), claim(1.203, "EUR"),
                                 claim(9999.0, "EUR")]), "Cost 1,203 euros.")
    assert res.grounded == {0}
    assert res.inferred == {1}
    assert res.ungrounded == {2}


# ---- the audit does not grant an inferred match ------------------------------------

def test_only_literal_matches_are_granted_without_adjudication():
    text = "Cost 1,203 euros."
    intel = _intel([claim(1203.0, "EUR"), claim(1.203, "EUR")])
    res = floor.evaluate(intel, text)
    decided = audit.adjudicate(intel.numeric_claims, [], res.grounded, text, None)

    assert ("m", 0) in decided.valid       # literal, granted
    assert ("m", 1) not in decided.valid   # inferred, must be adjudicated
    assert 1 in decided.residual


def test_an_inferred_claim_can_still_be_upheld_by_a_grader():
    text = "Cost 1,203 euros."
    intel = _intel([claim(1.203, "EUR")])
    res = floor.evaluate(intel, text)

    def honest(article, claims):
        return [{"i": c["i"], "supported": True, "evidence": "cost 1,203 euros"}
                for c in claims]

    decided = audit.adjudicate(intel.numeric_claims, [], res.grounded, text, honest)
    assert ("m", 0) in decided.valid
