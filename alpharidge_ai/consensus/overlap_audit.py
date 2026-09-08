"""Cross-validator agreement on the articles two validators both happened to grade.

Each validator picks what it audits from its own secret, so the two no longer grade
identical sets and whole-vector comparison stops working. They still coincide on most
of the claim-bearing pool, and that intersection is enough to tell an honest scoring
difference from a validator reporting something other than what it measured.

This does not replace the per-sender Merkle check, which verifies a sender against its
own record. This checks a sender against ours.

Pure: no I/O.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Set

# Median absolute score difference above which a sender is worth reporting.
DEFAULT_TAU = 0.15

# Below this many shared articles the median is noise, so no verdict is reached.
DEFAULT_MIN_OVERLAP = 20


@dataclass(frozen=True)
class OverlapVerdict:
    comparable: bool
    flagged: bool
    overlap: int
    median_delta: Optional[float] = None
    reason: str = ""


def flatten(observations_by_target: Mapping) -> Dict[int, float]:
    """Collapse one sender's per-target observations to {article_id: score}.

    An article is leased to a single miner, so its id identifies the observation. Where
    a sender reported the same article twice, the lower score is kept, matching how
    pooled observations are already resolved.
    """
    out: Dict[int, float] = {}
    for rows in (observations_by_target or {}).values():
        for row in rows or ():
            try:
                aid, score = int(row[0]), float(row[1])
            except (TypeError, ValueError, IndexError):
                continue
            if aid not in out or score < out[aid]:
                out[aid] = score
    return out


def overlap(local: Mapping[int, float], remote: Mapping[int, float]) -> Set[int]:
    return set(local or {}) & set(remote or {})


def assess(local: Mapping[int, float], remote: Mapping[int, float], *,
           tau: float = DEFAULT_TAU,
           min_overlap: int = DEFAULT_MIN_OVERLAP) -> OverlapVerdict:
    """Compare two validators on the articles both graded.

    The median is deliberate: a handful of genuinely hard articles should not convict a
    sender, while a sender reporting something unrelated to what it measured moves the
    whole distribution.
    """
    shared = overlap(local, remote)
    if len(shared) < int(min_overlap):
        return OverlapVerdict(False, False, len(shared),
                              reason=f"overlap_too_small({len(shared)}<{min_overlap})")

    deltas = [abs(float(local[a]) - float(remote[a])) for a in sorted(shared)]
    median = statistics.median(deltas)
    flagged = median > float(tau)
    return OverlapVerdict(
        comparable=True,
        flagged=flagged,
        overlap=len(shared),
        median_delta=median,
        reason=(f"score_divergence(median={median:.4f}, tau={tau})" if flagged
                else f"agreed(median={median:.4f})"),
    )
