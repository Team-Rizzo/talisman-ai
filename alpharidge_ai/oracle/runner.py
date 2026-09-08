"""The audit loop: from an article to one reputation observation, or to nothing.

Selection decides whether an article is watched at all. Claim-bearing articles are
scored on what was extracted and how well the submitter knew its own claims; the rest
are scored on judgment agreement against a drawn model. An article the validator found
nothing in yields no observation rather than a zero, because absence of gold says
something about the article, not about the submission.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import bittensor as bt

from alpharidge_ai.mechanism import scoring
from alpharidge_ai.oracle import audit, floor, schema_gate, selector

JUDGMENT_FIELDS = ("overall_sentiment", "impact_potential", "urgency", "content_type")


@dataclass(frozen=True)
class Observation:
    article_id: int
    score: float
    weight: float
    path: str
    grader_model: str = ""
    detail: str = ""


def _enum(value) -> Optional[str]:
    if value is None:
        return None
    return str(getattr(value, "value", value)).lower()


def _judgment_fields(intel, floor_result=None) -> Dict:
    """The fields a keeper article is scored on.

    Items whose evidence span did not hold are left out: an asset a submission cannot
    point to in the text is not something it found.
    """
    dropped_assets = getattr(floor_result, "unevidenced_assets", set()) or set()
    dropped_entities = getattr(floor_result, "unevidenced_entities", set()) or set()

    def kept(items, dropped, form, upper):
        out = set()
        for item in items or ():
            name = form(item)
            if not name or name.strip().lower() in dropped:
                continue
            out.add(name.strip().upper() if upper else name.strip().lower())
        return sorted(out)

    fields = {f: _enum(getattr(intel, f, None)) for f in JUDGMENT_FIELDS}
    fields["assets"] = kept(getattr(intel, "assets", None), dropped_assets,
                            floor.asset_form, True)
    fields["entities"] = kept(getattr(intel, "entities", None), dropped_entities,
                              floor.entity_form, False)
    return fields


def _normalise_reply(reply: Dict) -> Dict:
    out = {f: _enum(reply.get(f)) for f in JUDGMENT_FIELDS if reply.get(f) is not None}
    if reply.get("assets") is not None:
        out["assets"] = sorted({str(a).upper() for a in reply["assets"]})
    if reply.get("entities") is not None:
        out["entities"] = sorted({str(e).lower() for e in reply["entities"]})
    return out


def _claim_keys(miner_intel, grader_intel, grounded, article_text, adjudicator,
                claim_cap: int):
    """Adjudicate claims, and key quotes by the span the validator matched."""
    miner_claims = list(getattr(miner_intel, "numeric_claims", None) or [])[:claim_cap]
    grader_claims = list(getattr(grader_intel, "numeric_claims", None) or [])[:claim_cap]

    decided = audit.adjudicate(miner_claims, grader_claims, grounded, article_text,
                               adjudicator)

    miner_keys = list(decided.miner_keys)
    grader_keys = set(decided.grader_keys)
    valid = set(decided.valid)
    confidences = {}
    for i, claim in enumerate(miner_claims):
        value = getattr(claim, "confidence", None)
        if value is not None and i < len(miner_keys):
            confidences[miner_keys[i]] = float(value)

    return miner_keys, grader_keys, valid, confidences, decided


def _quote_keys(miner_intel, grader_intel, article_text, aligned, claim_cap: int):
    """Quote keys, on both sides, from the spans this validator matched."""
    miner_quotes = list(getattr(miner_intel, "quotes", None) or [])[:claim_cap]
    keys, confidences = [], {}
    for i in range(len(miner_quotes)):
        # A quote that did not align is still something the submission asserted, so it
        # stays in the denominator as an unsupported key.
        key = ("q", aligned[i]) if i in aligned else ("qx", i)
        keys.append(key)
        value = getattr(miner_quotes[i], "confidence", None)
        if value is not None:
            confidences[key] = float(value)

    grader_spans = audit.grader_quote_keys(
        article_text, list(getattr(grader_intel, "quotes", None) or [])[:claim_cap])
    return keys, {("q", s) for s in grader_spans}, confidences


def audit_article(article_id: int, article_text: str, miner_intel, grader_intel,
                  floor_result: floor.FloorResult, *,
                  audit_selector: selector.Selector, profile, grader=None,
                  block: int = 0) -> Optional[Observation]:
    """Audit one article. Returns an observation, or None when it is not watched."""
    if profile is None or not floor_result.floor_pass:
        return None

    oracle = profile.oracle
    choice = audit_selector.select(
        article_id, article_text,
        pool_tiers=oracle.pool_tiers,
        keyed_rate_pool=oracle.keyed_rate_pool,
        keyed_rate_keeper=oracle.keyed_rate_keeper,
        grader_models=oracle.grader_models)
    if not choice.graded:
        return None

    verdict = schema_gate.evaluate(getattr(miner_intel, "schema_version", None),
                                  block=block,
                                  cutover_block=oracle.schema_cutover_block,
                                  intel=miner_intel)
    if not verdict.accepted:
        return None

    if choice.slice == selector.KEEPER:
        return _keeper(article_id, article_text, miner_intel, choice, grader,
                       oracle.keeper_weight, floor_result)

    adjudicator = None
    if grader is not None and choice.grader_model:
        def adjudicator(text, claims, _model=choice.grader_model):
            return grader.adjudicate(text, claims, _model)

    # Only literally grounded claims are granted; an inferred match joins the residual
    # and is adjudicated like anything else.
    miner_keys, grader_keys, valid, confidences, decided = _claim_keys(
        miner_intel, grader_intel, floor_result.grounded, article_text,
        adjudicator, oracle.claim_cap)

    q_miner, q_grader, q_conf = _quote_keys(
        miner_intel, grader_intel, article_text, floor_result.aligned_quotes,
        oracle.claim_cap)
    miner_keys += q_miner
    grader_keys |= q_grader
    confidences.update(q_conf)

    score = scoring.article_score(miner_keys, grader_keys, valid, confidences,
                                  claim_cap=oracle.claim_cap,
                                  score_confidence=verdict.score_confidence)
    if score is None:
        return None

    return Observation(
        article_id=int(article_id), score=score.observation, weight=1.0,
        path=selector.POOL, grader_model=choice.grader_model,
        detail=(f"p={score.precision:.2f} r={score.recall:.2f} "
                f"conf={score.confidence:.2f} residual={len(decided.residual)}"))


def _keeper(article_id, article_text, miner_intel, choice, grader,
            keeper_weight: float, floor_result=None) -> Optional[Observation]:
    """Judgment agreement, for articles with no claims to check.

    Not rescaled by how well models agree with themselves: the same ceiling applies to
    every submission, so it shifts the whole field rather than the ranking, and the
    emission midpoint tracks the field.
    """
    if grader is None or not choice.grader_model:
        return None
    reply = _normalise_reply(grader.judge(article_text, choice.grader_model) or {})
    agreement = scoring.keeper_agreement(
        _judgment_fields(miner_intel, floor_result), reply)
    if agreement is None:
        return None
    return Observation(article_id=int(article_id), score=float(agreement),
                       weight=float(keeper_weight), path=selector.KEEPER,
                       grader_model=choice.grader_model,
                       detail=f"agreement={agreement:.2f}")


class Auditor:
    """Carries what an audit needs: this validator's key, the profile, the grader.

    Shadow: observations are returned to the caller to log, never written to the
    reputation store. Anything written there propagates fleet-wide within an epoch and
    cannot be taken back, so it waits for the flip rather than a first unwatched run.
    """

    def __init__(self, key: bytes, profile_reader, grader=None):
        self._selector = selector.Selector(key)
        self._profile_reader = profile_reader
        self._grader = grader

    def selects(self, article_id, article_text, block: int) -> bool:
        """Whether this article is watched, without doing the audit.

        Lets the caller decide which articles are worth a reference analysis before
        paying for one.
        """
        try:
            profile = self._profile_reader(block)
            if profile is None:
                return False
            oracle = profile.oracle
            return self._selector.select(
                article_id, article_text,
                pool_tiers=oracle.pool_tiers,
                keyed_rate_pool=oracle.keyed_rate_pool,
                keyed_rate_keeper=oracle.keyed_rate_keeper,
                grader_models=oracle.grader_models).graded
        except Exception:
            return False

    def audit(self, article_id, article_text, miner_intel, grader_intel,
              floor_result, block: int) -> Optional[Observation]:
        try:
            profile = self._profile_reader(block)
            if profile is None:
                return None
            return audit_article(article_id, article_text, miner_intel, grader_intel,
                                 floor_result, audit_selector=self._selector,
                                 profile=profile, grader=self._grader, block=block)
        except Exception as e:
            bt.logging.debug(f"[AUDIT] article {article_id} failed: {e}")
            return None
