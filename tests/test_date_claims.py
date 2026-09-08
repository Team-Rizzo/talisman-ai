"""Dates are not numeric claims.

The 2026-09-03 re-grounding found dates and times were 7.2% of extracted claims — the
single largest bucket after outright invention. No floor should ever ground one, and
they dilute precision for every miner that emits them. Dropped deterministically rather
than asked away in a prompt, which drifts.
"""

import pytest

from alpharidge_ai.analyzer.article_intelligence_analyzer import (
    ArticleIntelligenceAnalyzer as A)


def kept(raw):
    return [(c.metric_name, c.value) for c in A._build_numeric_claims(A, raw)]


def claim(metric="revenue", value=1.2e9, unit="USD"):
    return {"metric_name": metric, "value": value, "unit": unit, "confidence": 0.9}


# ---- what gets dropped -------------------------------------------------------------

@pytest.mark.parametrize("unit", ["date", "datetime", "timestamp"])
def test_a_unit_that_only_names_an_instant_is_dropped(unit):
    assert kept([claim(unit=unit)]) == []


@pytest.mark.parametrize("unit", ["year", "month", "day", "week", "quarter", "hours"])
def test_a_calendar_unit_is_a_date_only_when_the_value_is_a_year(unit):
    """"3 hours" is a duration; "2026 years" is a date wearing a unit."""
    assert kept([claim(value=2026, unit=unit)]) == []
    assert kept([claim(value=3.0, unit=unit)]) != []


@pytest.mark.parametrize("metric", [
    "publication date", "Datum", "report date", "Uhrzeit", "deadline", "Q3",
    "timestamp",
])
def test_a_date_metric_name_is_dropped(metric):
    assert kept([claim(metric=metric)]) == []


@pytest.mark.parametrize("value", ["31.08.26", "2026-09-03", "28.08.2026"])
def test_a_written_date_value_is_dropped(value):
    assert kept([claim(value=value)]) == []


@pytest.mark.parametrize("value", [28.08, 8.30, 1203.0, 12.5])
def test_a_bare_decimal_is_not_assumed_to_be_a_date(value):
    """A float is indistinguishable from a decimal; dropping it would cost real work."""
    assert kept([claim(value=value)]) != []


# ---- what survives -----------------------------------------------------------------

def test_a_real_measurement_survives():
    assert kept([claim()]) == [("revenue", 1.2e9)]


def test_a_plain_number_is_not_mistaken_for_a_date():
    assert kept([claim(metric="barrels", value=1203.0, unit="count")]) == \
        [("barrels", 1203.0)]


def test_a_percentage_survives():
    assert kept([claim(metric="growth", value=12.5, unit="%")]) == [("growth", 12.5)]


def test_a_metric_merely_containing_a_date_word_is_kept():
    """'year-on-year revenue' measures revenue, not a year."""
    assert kept([claim(metric="revenue year-on-year", value=1.2e9)]) != []


# ---- shape -------------------------------------------------------------------------

def test_dropped_claims_do_not_consume_the_cap():
    from alpharidge_ai.models.article_intelligence import MAX_NUMERIC_CLAIMS
    raw = ([claim(unit="date")] * 30
           + [claim(metric=f"m{i}", value=float(i)) for i in range(MAX_NUMERIC_CLAIMS)])
    assert len(kept(raw)) == MAX_NUMERIC_CLAIMS


def test_malformed_entries_are_skipped_not_crashed():
    assert kept(["a bare string", None, 42, claim()]) == [("revenue", 1.2e9)]


def test_the_grader_is_told_the_same_thing():
    from alpharidge_ai.oracle import grader
    assert "not numeric claims" in grader.ADJUDICATION_PROMPT
