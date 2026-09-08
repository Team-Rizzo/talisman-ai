"""Schema grace: what an older submission is scored on, and when that ends.

Miners run their own machines, so a schema change takes real calendar time to reach
everyone. Until the cutover block a submission on the previous schema is accepted and
scored on extraction alone, neither rewarded nor punished for a field it cannot send.
After that block the field is required and its absence fails the floor.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from alpharidge_ai.models.article_intelligence import (
    SCHEMA_VERSION, SCHEMA_VERSION_PREV)

ACCEPTED_VERSIONS = (SCHEMA_VERSION, SCHEMA_VERSION_PREV)


@dataclass(frozen=True)
class SchemaVerdict:
    accepted: bool
    score_confidence: bool
    reason: str = ""


def _carries_new_fields(intel) -> bool:
    """Whether a submission actually supplies what 1.2.0 added.

    The version string is the sender's own claim. After the cutover the fields have to
    be there, or claiming the new version would be a way to keep the neutral treatment
    the grace window exists to give people who genuinely cannot send them yet.
    """
    for claim in (getattr(intel, "numeric_claims", None) or []):
        if getattr(claim, "confidence", None) is None:
            return False
    for quote in (getattr(intel, "quotes", None) or []):
        if (getattr(quote, "confidence", None) is None
                or getattr(quote, "start_offset", None) is None
                or getattr(quote, "end_offset", None) is None):
            return False
    return True


def evaluate(schema_version: Optional[str], *, block: int, cutover_block: int,
             intel=None) -> SchemaVerdict:
    """Decide how to treat a submission at a given block.

    Pass `intel` to check the fields as well as the claimed version.
    """
    version = (schema_version or "").strip()
    past_cutover = int(block) >= int(cutover_block)

    if version == SCHEMA_VERSION:
        if intel is not None and not _carries_new_fields(intel):
            if past_cutover:
                return SchemaVerdict(False, False, "missing_1_2_0_fields")
            # Before the cutover, treat it as the older schema it really is.
            return SchemaVerdict(True, False, "grace_incomplete")
        return SchemaVerdict(True, True, "current")

    if version == SCHEMA_VERSION_PREV:
        if past_cutover:
            return SchemaVerdict(False, False, f"schema_{version}_after_cutover")
        return SchemaVerdict(True, False, "grace")

    return SchemaVerdict(False, False, f"unsupported_schema_{version or 'missing'}")


def confidences(items) -> dict:
    """Per-item stated confidence, keyed by position. Missing values are left out so
    the scorer applies its own neutral value rather than inventing one here."""
    out = {}
    for i, item in enumerate(items or ()):
        value = getattr(item, "confidence", None)
        if value is not None:
            out[i] = float(value)
    return out
