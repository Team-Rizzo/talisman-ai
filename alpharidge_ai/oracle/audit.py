"""Adjudication: deciding which submitted claims hold, as cheaply as possible.

Most of this is not model work. A claim whose number is in the article is settled by
the floor, and a claim the validator's own reference run also found is settled by
comparison. Only what is left over goes to a model, in one batched call per article.

A grader cannot validate a claim by asserting it: the evidence it returns has to align
against the article itself, on the same alignment the floor uses.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from alpharidge_ai.oracle import floor

VALUE_MATCH_TOLERANCE = 0.005

_WS = re.compile(r"\s+")

# An adjudicator takes (article_text, residual claims) and returns
# [{"i": int, "supported": bool, "evidence": str}].
Adjudicator = Callable[[str, List[dict]], List[dict]]


def _metric(name) -> str:
    return _WS.sub(" ", str(name or "").strip().lower())


def _values_match(a: float, b: float) -> bool:
    return abs(a - b) / max(abs(b), 1e-9) <= VALUE_MATCH_TOLERANCE


def claims_match(left, right) -> bool:
    """Two claims describe the same fact: same metric, same unit class, same magnitude.

    Magnitude, not just class: "1 million" and "1" share the class `count`.
    """
    try:
        lv, rv = float(getattr(left, "value")), float(getattr(right, "value"))
    except (TypeError, ValueError):
        return False
    lu, ru = getattr(left, "unit", None), getattr(right, "unit", None)
    if floor._unit_class(lu) != floor._unit_class(ru):
        return False
    # Compare what each claim actually asserts, scale included.
    lv *= floor.unit_magnitude(lu)
    rv *= floor.unit_magnitude(ru)
    if _metric(getattr(left, "metric_name", None)) != \
            _metric(getattr(right, "metric_name", None)):
        return False
    return _values_match(lv, rv)


@dataclass
class Adjudication:
    miner_keys: List = field(default_factory=list)
    grader_keys: Set = field(default_factory=set)
    valid: Set = field(default_factory=set)
    residual: List[int] = field(default_factory=list)
    llm_calls: int = 0


def _residual_payload(claims, indexes: Sequence[int]) -> List[dict]:
    out = []
    for i in indexes:
        claim = claims[i]
        out.append({
            "i": i,
            "metric": str(getattr(claim, "metric_name", "") or ""),
            "value": float(getattr(claim, "value", 0.0) or 0.0),
            "unit": str(getattr(claim, "unit", "") or ""),
            "context": str(getattr(claim, "context", "") or "")[:200],
        })
    return out


def parse_adjudication(raw, expected: Sequence[int]) -> Dict[int, dict]:
    """Read a grader's reply. Anything unparseable leaves its claim unsupported."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            return {}
    if isinstance(raw, Mapping):
        raw = raw.get("claims") or raw.get("results") or []
    if not isinstance(raw, (list, tuple)):
        return {}

    allowed = set(expected)
    out: Dict[int, dict] = {}
    for row in raw:
        if not isinstance(row, Mapping):
            continue
        try:
            i = int(row.get("i"))
        except (TypeError, ValueError):
            continue
        if i not in allowed:
            continue
        out[i] = {"supported": bool(row.get("supported")),
                  "evidence": str(row.get("evidence") or "")}
    return out


def adjudicate(miner_claims: Sequence, grader_claims: Sequence,
               grounded: Set[int], article_text: str,
               adjudicator: Optional[Adjudicator] = None) -> Adjudication:
    """Settle each submitted claim, cheapest test first.

    1. The floor already found its number in the article.
    2. The validator's own reference run found the same claim.
    3. Whatever is left, in one batched call, with the evidence checked.
    """
    result = Adjudication()
    result.grader_keys = {("g", j) for j in range(len(grader_claims))}

    taken: Set[int] = set()
    residual: List[int] = []

    for i, claim in enumerate(miner_claims):
        match = None
        for j, gold in enumerate(grader_claims):
            if j in taken:
                continue
            if claims_match(claim, gold):
                match = j
                break

        if match is not None:
            taken.add(match)
            result.miner_keys.append(("g", match))
            continue

        key = ("m", i)
        result.miner_keys.append(key)
        if i in grounded:
            result.valid.add(key)
        else:
            # Includes claims that matched only an inferred reading; those are
            # adjudicated rather than granted.
            residual.append(i)

    if residual and adjudicator is not None:
        result.llm_calls = 1
        article = floor.normalize(article_text)
        try:
            replies = parse_adjudication(
                adjudicator(article_text, _residual_payload(miner_claims, residual)),
                residual)
        except Exception:
            replies = {}
        for i, reply in replies.items():
            if not reply["supported"]:
                continue
            if floor.align_quote(article, reply["evidence"], None, None) is None:
                continue
            result.valid.add(("m", i))

    result.residual = residual
    return result


# ---- quotes -----------------------------------------------------------------------

def quote_keys(aligned: Mapping[int, Tuple[int, int]]) -> List[Tuple[int, int]]:
    """A quote is keyed by the span the validator matched, not by what was submitted."""
    return [aligned[i] for i in sorted(aligned)]


def grader_quote_keys(article_text: str, grader_quotes: Sequence) -> Set[Tuple[int, int]]:
    article = floor.normalize(article_text)
    keys = set()
    for quote in grader_quotes or ():
        hit = floor.align_quote(article, getattr(quote, "text", None), None, None)
        if hit is not None:
            keys.add((hit.start, hit.end))
    return keys


RESIDUAL_PROMPT = (
    "You are checking numeric claims against a source article.\n"
    "For each claim decide whether the article supports it.\n"
    "Return JSON only: a list of {\"i\": int, \"supported\": bool, \"evidence\": string}.\n"
    "\"evidence\" must be a verbatim span copied from the article that shows the claim. "
    "If there is no such span, return supported=false and an empty evidence string.\n"
    "Do not infer, compute, or rely on outside knowledge.\n"
)
