"""After the cutover block, a submission on the old schema earns nothing.

§E5: the old schema fails the floor, not merely the audit.
"""

import types

import pytest

from alpharidge_ai.analyzer import scoring
from tests.test_floor_gating import _payload, TEXT, TITLE

CUTOVER = 1_000


def article(article_id, schema, *, confidence=0.9, offsets=True):
    blob = _payload([{"metric_name": "revenue", "value": 1.2e9, "unit": "USD",
                      **({"confidence": confidence} if confidence is not None else {})}])
    blob["schema_version"] = schema
    blob["quotes"] = [{"speaker": "CEO", "text": "margins should recover",
                       **({"confidence": 0.8, "start_offset": 1, "end_offset": 9}
                          if offsets else {})}]
    return types.SimpleNamespace(
        id=article_id, content=TEXT, title=TITLE,
        analysis=types.SimpleNamespace(analysis_data=blob))


def swept(batch, block):
    return scoring._floor_sweep(batch, block=block, schema_cutover_block=CUTOVER)


# ---- before the cutover ------------------------------------------------------------

def test_the_old_schema_still_earns_before_the_block():
    assert swept([article(1, "1.1.0", confidence=None, offsets=False)],
                 block=CUTOVER - 1) == {1: True}


def test_the_new_schema_earns_before_the_block():
    assert swept([article(1, "1.2.0")], block=CUTOVER - 1) == {1: True}


# ---- after the cutover -------------------------------------------------------------

def test_the_old_schema_earns_nothing_after_the_block():
    """Not merely unaudited — unpaid. Otherwise not upgrading is free."""
    assert swept([article(1, "1.1.0", confidence=None, offsets=False)],
                 block=CUTOVER) == {1: False}


def test_claiming_the_new_version_without_the_fields_also_earns_nothing():
    assert swept([article(1, "1.2.0", confidence=None)], block=CUTOVER) == {1: False}


def test_an_upgraded_miner_is_unaffected():
    assert swept([article(1, "1.2.0")], block=CUTOVER + 5_000) == {1: True}


# ---- the gate stays off until a cutover is published -------------------------------

def test_no_cutover_block_means_no_enforcement():
    """Publishing the switch is what starts it; a deploy alone changes nothing."""
    batch = [article(1, "1.1.0", confidence=None, offsets=False)]
    assert scoring._floor_sweep(batch, block=9_000_000,
                                schema_cutover_block=0) == {1: True}


def test_a_mixed_batch_is_judged_per_article():
    got = swept([article(1, "1.2.0"),
                 article(2, "1.1.0", confidence=None, offsets=False)], block=CUTOVER)
    assert got == {1: True, 2: False}
