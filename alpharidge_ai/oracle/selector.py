"""Keyed audit selection.

Selection is decided by a per-validator secret and keys on the article id alone:
never on dispatch order, batch position, timestamp, or who holds the article.

Each validator holds its own key, so per-article agreement between validators is
audited on the articles both happened to draw.
"""
from __future__ import annotations

import hashlib
import hmac
import re
from dataclasses import dataclass
from typing import Optional, Sequence

from alpharidge_ai.oracle import floor

POOL = "pool"
KEEPER = "keeper"

TIER_NUMBER_BEARING = "number_bearing"
TIER_QUOTE_BEARING = "quote_bearing"

_QUOTE_MARK_RE = re.compile(r'"[^"]{12,}"')

_DENOM = float(1 << 64)


def _fraction(secret: bytes, article_id, domain: str) -> float:
    """A uniform value in [0,1) from the key and the article id."""
    msg = f"{domain}:{int(article_id)}".encode("utf-8")
    digest = hmac.new(secret, msg, hashlib.sha256).digest()
    return int.from_bytes(digest[:8], "big") / _DENOM


def article_tiers(article_text: Optional[str]) -> frozenset:
    """Deterministic content tiers. Public by construction; the key decides the rest."""
    normalized = floor.normalize(article_text).text
    if not normalized:
        return frozenset()
    tiers = set()
    if floor.parse_numbers(normalized):
        tiers.add(TIER_NUMBER_BEARING)
    if _QUOTE_MARK_RE.search(normalized):
        tiers.add(TIER_QUOTE_BEARING)
    return frozenset(tiers)


def in_pool(article_text: Optional[str], pool_tiers: Sequence[str]) -> bool:
    """The claim-bearing slice worth spending the audit budget on."""
    return bool(article_tiers(article_text) & set(pool_tiers or ()))


@dataclass(frozen=True)
class Selection:
    graded: bool
    slice: str = ""          # POOL or KEEPER
    grader_model: str = ""


class Selector:
    """Holds this validator's audit key. The key stays in memory and is never logged."""

    def __init__(self, secret: bytes):
        if not secret:
            raise ValueError("audit key must not be empty")
        self._secret = secret if isinstance(secret, bytes) else str(secret).encode("utf-8")

    def __repr__(self) -> str:
        return "<Selector key=redacted>"

    def fraction(self, article_id, domain: str = "select") -> float:
        return _fraction(self._secret, article_id, domain)

    def select(self, article_id, article_text: Optional[str], *,
               pool_tiers: Sequence[str], keyed_rate_pool: float,
               keyed_rate_keeper: float,
               grader_models: Sequence = ()) -> Selection:
        """Decide whether to grade this article, on which path, and with which grader.

        The claim-bearing pool is sampled heavily; a thin keyed slice of everything
        else is graded on judgment fields.
        """
        pooled = in_pool(article_text, pool_tiers)
        rate = float(keyed_rate_pool if pooled else keyed_rate_keeper)
        if rate <= 0.0 or self.fraction(article_id) >= rate:
            return Selection(False)
        return Selection(
            graded=True,
            slice=POOL if pooled else KEEPER,
            grader_model=self.draw_grader(article_id, grader_models),
        )

    def draw_grader(self, article_id, models: Sequence) -> str:
        """Pick a grader by weight, reproducibly."""
        entries = [(getattr(m, "id", None) or m.get("id"),
                    float(getattr(m, "weight", None) if hasattr(m, "weight") else m.get("weight", 0)))
                   for m in (models or [])]
        entries = [(mid, w) for mid, w in entries if mid and w > 0]
        if not entries:
            return ""
        total = sum(w for _, w in entries)
        target = self.fraction(article_id, "grader") * total
        cumulative = 0.0
        for mid, weight in entries:
            cumulative += weight
            if target < cumulative:
                return mid
        return entries[-1][0]
