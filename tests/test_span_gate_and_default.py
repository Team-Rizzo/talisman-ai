"""The evidence-span gate has a consequence, and floor gating is on unless disabled."""

import types

import pytest

from alpharidge_ai.mechanism import profile as mp
from alpharidge_ai.oracle import audit, floor, runner
from tests.test_profile_client import valid

TEXT = "NVDA rose sharply after the quarterly report was published today."


# Real models, not stand-ins. The first version of this file invented `symbol` and
# `canonical_name`, which do not exist on either model — so the code passed its tests
# and did nothing on production data.
from alpharidge_ai.models.article_intelligence import AssetSentiment, ExtractedEntity


def asset(ticker, spans):
    return AssetSentiment.model_construct(ticker=ticker, asset_name=ticker,
                                          evidence_spans=list(spans))


def entity(name, spans):
    return ExtractedEntity.model_construct(name=name, entity_type="organization",
                                           evidence_spans=list(spans))


def intel(assets=(), entities=()):
    return types.SimpleNamespace(numeric_claims=[], quotes=[], assets=list(assets),
                                 entities=list(entities))


# ---- the gate now bites ------------------------------------------------------------

def test_an_asset_without_usable_evidence_is_not_scored():
    """E18: a failed span fails that asset, not the article."""
    sub = intel([asset("NVDA", ["NVDA"]), asset("AMD", ["NVDA rose sharply after"])])
    result = floor.evaluate(sub, TEXT)
    assert result.unevidenced_assets == {"nvda"}
    assert runner._judgment_fields(sub, result)["assets"] == ["AMD"]


def test_the_article_itself_still_passes():
    sub = intel([asset("NVDA", ["NVDA"])])
    assert floor.evaluate(sub, TEXT).floor_pass


def test_the_entity_path_gates_once_spans_exist():
    """The code gates entities, but the shipped ExtractedEntity has no evidence_spans
    field, so nothing reaches it today. This uses a shape that carries them, to show
    the path works if the schema gains the field — see the schema note in DEVIATIONS.
    """
    with_spans = [types.SimpleNamespace(name="Nvidia", evidence_spans=["Nvidia"]),
                  types.SimpleNamespace(name="Report",
                                        evidence_spans=["quarterly report was published"])]
    sub = intel(entities=with_spans)
    result = floor.evaluate(sub, TEXT)
    assert "nvidia" in result.unevidenced_entities
    assert runner._judgment_fields(sub, result)["entities"] == ["report"]


def test_the_shipped_entity_model_cannot_express_a_failed_span():
    """Recorded rather than worked around: E18 asks for entity spans and the schema
    does not carry them. Adding the field is a schema decision, not a code fix."""
    assert "evidence_spans" not in ExtractedEntity.model_fields


def test_the_real_asset_model_is_read():
    """The production model carries ticker, not symbol."""
    assert "ticker" in AssetSentiment.model_fields
    assert "symbol" not in AssetSentiment.model_fields
    a = asset("NVDA", ["NVDA"])
    assert floor.asset_form(a) == "NVDA"


def test_the_real_entity_model_is_read():
    """The production model carries name, not canonical_name."""
    assert "name" in ExtractedEntity.model_fields
    assert "canonical_name" not in ExtractedEntity.model_fields
    assert floor.entity_form(entity("Nvidia", [])) == "Nvidia"


def test_production_assets_reach_keeper_scoring_at_all():
    """They were absent entirely: two of the six judgment fields scored nothing."""
    sub = intel([asset("NVDA", ["NVDA rose sharply after"])],
                [entity("Nvidia", ["NVDA rose sharply after"])])
    fields = runner._judgment_fields(sub, floor.evaluate(sub, TEXT))
    assert fields["assets"] == ["NVDA"]
    assert fields["entities"] == ["nvidia"]


def test_an_entity_without_the_span_field_is_not_failed():
    """ExtractedEntity has no evidence_spans in the shipped schema; absent is not
    failed, so such an entity is scored rather than dropped."""
    bare = ExtractedEntity.model_construct(name="Nvidia", entity_type="organization")
    sub = intel(entities=[bare])
    result = floor.evaluate(sub, TEXT)
    assert result.unevidenced_entities == set()
    assert runner._judgment_fields(sub, result)["entities"] == ["nvidia"]


def test_an_item_with_no_spans_at_all_is_left_alone():
    """Absent evidence is not failed evidence; only a span that does not hold counts."""
    sub = intel([asset("NVDA", [])])
    result = floor.evaluate(sub, TEXT)
    assert result.unevidenced_assets == set()
    assert runner._judgment_fields(sub, result)["assets"] == ["NVDA"]


def test_scoring_without_a_floor_result_is_unchanged():
    sub = intel([asset("NVDA", ["NVDA"])])
    assert runner._judgment_fields(sub, None)["assets"] == ["NVDA"]


def test_a_dropped_asset_lowers_keeper_agreement():
    from alpharidge_ai.mechanism import scoring
    sub = intel([asset("NVDA", ["NVDA"]), asset("AMD", ["NVDA rose sharply after"])])
    result = floor.evaluate(sub, TEXT)
    grader = {"assets": ["NVDA", "AMD"]}
    gated = scoring.keeper_agreement(runner._judgment_fields(sub, result), grader)
    ungated = scoring.keeper_agreement(runner._judgment_fields(sub, None), grader)
    assert gated < ungated


# ---- floor gating defaults on ------------------------------------------------------

def test_floor_gating_is_on_when_a_profile_does_not_mention_it():
    """It withholds pay for work that failed a check. There is no operating state where
    paying for that is wanted, so it does not wait for the flip."""
    assert mp.parse(valid()).settlement.floor_gating is True


def test_it_can_still_be_turned_off_explicitly():
    raw = valid()
    raw["settlement"]["floor_gating"] = False
    assert mp.parse(raw).settlement.floor_gating is False


def test_the_switches_that_change_economics_stay_off():
    p = mp.parse(valid())
    assert p.settlement.live is False
    assert p.oracle.live is False
    assert p.rations.dispatch is False


def test_it_is_still_rejected_when_it_is_not_a_boolean():
    raw = valid()
    raw["settlement"]["floor_gating"] = "yes"
    with pytest.raises(mp.ProfileError):
        mp.parse(raw)


def test_an_entity_with_a_ticker_is_named_by_its_name():
    """ExtractedEntity has an optional ticker; the grader answers with names, so the
    name is the identity. Naming it by ticker scored honest submissions at zero."""
    e = ExtractedEntity.model_construct(name="Nvidia", entity_type="organization",
                                        ticker="NVDA")
    assert floor.entity_form(e) == "Nvidia"
    sub = intel(entities=[e])
    assert runner._judgment_fields(sub, None)["entities"] == ["nvidia"]


def test_an_asset_is_named_by_its_ticker():
    a = AssetSentiment.model_construct(ticker="NVDA", asset_name="Nvidia Corp")
    assert floor.asset_form(a) == "NVDA"
    assert runner._judgment_fields(intel([a]), None)["assets"] == ["NVDA"]


def _claim(value, unit):
    return types.SimpleNamespace(value=value, unit=unit, metric_name="margin",
                                 context="")


def test_basis_points_do_not_ground_against_a_percentage():
    text = "Margin expanded 12.5% in the quarter."
    sub = types.SimpleNamespace(numeric_claims=[_claim(12.5, "bps")])
    result = floor.evaluate(sub, text)
    assert result.grounded == set()
    assert audit.adjudicate(sub.numeric_claims, [], result.grounded,
                            text, None).valid == set()


def test_basis_points_still_match_the_percentage_they_equal():
    assert audit.claims_match(_claim(50, "bps"), _claim(0.5, "%"))
    assert not audit.claims_match(_claim(50, "bps"), _claim(50, "%"))
    assert not audit.claims_match(_claim(12.5, "basis_points"), _claim(12.5, "pct"))


def test_every_spelling_of_a_basis_point_is_treated_alike():
    for unit in ("bps", "bp", "basis points", "basis_points", "BP"):
        assert floor._unit_class(unit) == "pct", unit
        assert floor.unit_magnitude(unit) == floor.BASIS_POINT, unit
        assert audit.claims_match(_claim(50, unit), _claim(0.5, "%")), unit
        assert not audit.claims_match(_claim(50, unit), _claim(50, "%")), unit


def test_a_blank_identity_field_does_not_become_a_blank_name():
    e = ExtractedEntity.model_construct(name="   ", entity_type="organization",
                                        ticker="NVDA")
    assert floor.entity_form(e) == "NVDA"
    assert runner._judgment_fields(intel(entities=[e]), None)["entities"] == ["nvda"]
    blank = ExtractedEntity.model_construct(name="  ", entity_type="organization")
    assert floor.entity_form(blank) is None
    assert runner._judgment_fields(intel(entities=[blank]), None)["entities"] == []
