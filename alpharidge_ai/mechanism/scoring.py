"""Per-article scoring: F1 against an independent claim set, times calibration.

Recall is measured against the grader's own claim set. Precision is measured against
that set plus whatever the grader separately accepted, so finding something the grader
missed is not punished.

Confidence is scored under a proper rule: a high number on a claim that does not hold
costs more than an honest low one. Knowing which of your own claims are solid is the
skill being paid for, so the exponent stays at one.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Set

# (0.5 + conf) spans [0.5, 1.5]; reputation observations are carried in [0, 1].
CONFIDENCE_OFFSET = 0.5
MAX_RAW_SCORE = 1.5
NEUTRAL_CONFIDENCE_FACTOR = 1.0

# Keeps a stated certainty of exactly 0 or 1 from producing an infinite term.
_P_CLAMP = 1e-6

JUDGMENT_ENUMS = ("overall_sentiment", "impact_potential", "urgency", "content_type")
JUDGMENT_SETS = ("assets", "entities")


@dataclass(frozen=True)
class ArticleScore:
    precision: float
    recall: float
    f1: float
    confidence: float
    raw: float
    normalized: float
    scored_claims: int

    @property
    def observation(self) -> float:
        return self.normalized


def _f1(precision: float, recall: float) -> float:
    if precision <= 0.0 or recall <= 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def calibration(miner_keys: Sequence, supported: Set,
                confidences: Mapping) -> float:
    """Geometric mean of the probability assigned to what turned out to be true."""
    if not miner_keys:
        return 0.0
    total = 0.0
    for key in miner_keys:
        p = confidences.get(key, CONFIDENCE_OFFSET)
        try:
            p = float(p)
        except (TypeError, ValueError):
            p = CONFIDENCE_OFFSET
        p = min(1.0 - _P_CLAMP, max(_P_CLAMP, p))
        total += math.log(p if key in supported else 1.0 - p)
    return math.exp(total / len(miner_keys))


def article_score(miner_keys: Sequence, grader_keys: Iterable,
                  adjudicated_valid: Iterable = (),
                  confidences: Optional[Mapping] = None,
                  *, claim_cap: int = 40,
                  score_confidence: bool = True) -> Optional[ArticleScore]:
    """Score one audited article, or None when there is nothing to score against.

    An article the grader found nothing in yields no observation rather than a zero:
    absence of gold is a property of the article, not evidence about the submission.

    `score_confidence=False` holds the calibration term neutral, for submissions on a
    schema that cannot carry per-claim confidence.
    """
    gold: Set = set(grader_keys)
    if not gold:
        return None

    submitted = list(miner_keys)[:claim_cap]
    supported: Set = gold | set(adjudicated_valid)
    confidences = confidences or {}

    hits = len({k for k in submitted if k in gold})
    recall = min(1.0, hits / len(gold))
    precision = (min(1.0, sum(1 for k in submitted if k in supported) / len(submitted))
                 if submitted else 0.0)
    f1 = _f1(precision, recall)

    conf = calibration(submitted, supported, confidences) if score_confidence else 0.0
    factor = (CONFIDENCE_OFFSET + conf) if score_confidence else NEUTRAL_CONFIDENCE_FACTOR

    raw = f1 * factor
    return ArticleScore(
        precision=precision,
        recall=recall,
        f1=f1,
        confidence=conf,
        raw=raw,
        normalized=min(1.0, max(0.0, raw / MAX_RAW_SCORE)),
        scored_claims=len(submitted),
    )


# ---- keeper path ------------------------------------------------------------------

def _jaccard(a: Iterable, b: Iterable) -> float:
    sa, sb = set(a or ()), set(b or ())
    if not sa and not sb:
        return 1.0
    union = sa | sb
    return len(sa & sb) / len(union) if union else 1.0


def keeper_agreement(miner_fields: Mapping, grader_fields: Mapping) -> Optional[float]:
    """Agreement on judgment fields, for articles that carry no claims to check.

    These fields have no single right answer, so this is agreement with a drawn grader,
    not correctness. It is not rescaled by that grader's self-agreement: the same
    ceiling applies to everyone, so it shifts the whole field rather than the ranking.
    """
    if not miner_fields or not grader_fields:
        return None

    scores = []
    for field in JUDGMENT_ENUMS:
        if field not in grader_fields:
            continue
        scores.append(1.0 if miner_fields.get(field) == grader_fields.get(field) else 0.0)
    for field in JUDGMENT_SETS:
        if field not in grader_fields:
            continue
        scores.append(_jaccard(miner_fields.get(field) or (), grader_fields.get(field) or ()))

    if not scores:
        return None
    return sum(scores) / len(scores)
