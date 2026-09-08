"""Per-article quality score: continuous graded = min(composite, faithfulness).

Operates on ArticleIntelligence.model_dump(mode="json") dicts. Reference composite
compares a submission to the validator's own analysis; faithfulness is reference-free
(depends only on the submission and the article). Both in [0,1]; the min is a one-sided
veto so neither can inflate the other.
"""
from __future__ import annotations
import math
import re
import numpy as np

# ---- ordinal enum ladders ----
_SENTIMENT = ["very_bullish", "bullish", "slightly_bullish", "neutral",
              "slightly_bearish", "bearish", "very_bearish"]
_IMPACT = ["critical", "high", "medium", "low", "negligible"]
_TECH = ["exceptional", "high", "medium", "low", "none"]
_URGENCY = ["flash", "breaking", "developing", "same_day", "evergreen"]
_FACTUAL = ["confirmed", "attributed", "speculative", "conditional", "rumor"]
_LADDERS = {"overall_sentiment": _SENTIMENT, "impact_potential": _IMPACT,
            "technical_quality": _TECH, "urgency": _URGENCY, "factual_confidence": _FACTUAL}
_SCORED_ENUMS = [
    "overall_sentiment", "sentiment_direction", "impact_potential", "technical_quality",
    "urgency", "factual_confidence", "temporal_focus", "market_analysis_type",
    "positioning_signal", "target_audience", "forward_event_type", "staleness_flag",
    "credibility_flag", "content_type", "primary_geo",
]

# composite component weights (renormalized over present components)
_W = {"grounding": 0.45, "enums": 0.30, "llm_extract": 0.15, "semantic": 0.10}


def _ev(x):
    return (x.get("value") if isinstance(x, dict) and "value" in x else str(x)).lower()


def _cos(a, b):
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return None  # both empty -> excluded, not credited
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _ordinal(pred, gold, ladder):
    if pred == gold:
        return 1.0
    if ladder is None or pred not in ladder or gold not in ladder:
        return 1.0 if pred == gold else 0.0
    return max(0.0, 1.0 - abs(ladder.index(pred) - ladder.index(gold)) / (len(ladder) - 1))


def _cs(g, analytic_only=False):
    cs = g.get("chart_summary") or {}
    parts = ([cs.get("one_liner"), cs.get("context_paragraph")] if analytic_only
             else [cs.get("headline"), cs.get("one_liner"), cs.get("context_paragraph")])
    return " ".join(p for p in parts if p).strip()


def _tickers(g):
    return {(a.get("ticker") or "").lower() for a in (g.get("assets") or []) if a.get("ticker")}


def _numeric(g):
    return {str(n.get("value")) for n in (g.get("numeric_claims") or []) if isinstance(n, dict)}


def _econ(g):
    return {str(d.get("event_type")) for d in (g.get("economic_data") or []) if isinstance(d, dict)}


def _quotes(g):
    return {(q.get("text") or "")[:60].lower() for q in (g.get("quotes") or []) if q.get("text")}


# ---- composite (reference comparison) ----
def _grounding(sub, ref, article, emb):
    s = _cs(sub, analytic_only=True)
    if not s:
        return 0.0
    r = _cs(ref, analytic_only=True)
    if not r:
        return None
    return max(0.0, emb.sim(s, r))


def _enums(sub, ref, rarity):
    num = den = 0.0
    for f in _SCORED_ENUMS:
        if f not in ref or f not in sub:
            continue
        gold, pred = _ev(ref[f]), _ev(sub[f])
        w = rarity.get(f, {}).get(gold, 1.0)
        num += _ordinal(pred, gold, _LADDERS.get(f)) * w
        den += w
    return num / den if den else 0.0


def _llm_extract(sub, ref):
    vals = [v for v in (_jaccard(_numeric(sub), _numeric(ref)),
                        _jaccard(_econ(sub), _econ(ref)),
                        _jaccard(_quotes(sub), _quotes(ref))) if v is not None]
    return sum(vals) / len(vals) if vals else None


def _narr_text(g):
    cs = g.get("chart_summary") or {}
    return f"{cs.get('headline') or ''} {', '.join(g.get('narrative_keywords') or [])}".strip()


def _semantic(sub, ref, emb):
    mt, vt = _narr_text(sub), _narr_text(ref)
    ne = max(0.0, emb.sim(mt, vt)) if (mt and vt) else None
    sf = _jaccard(
        {x.lower() for x in ((ref.get("event_fingerprint") or {}).get("semantic_fingerprint") or [])},
        {x.lower() for x in ((sub.get("event_fingerprint") or {}).get("semantic_fingerprint") or [])})
    parts = [p for p in (ne, sf) if p is not None]
    return sum(parts) / len(parts) if parts else None


def composite(sub, ref, article, emb, rarity):
    g = _grounding(sub, ref, article, emb)
    present = {"enums": _enums(sub, ref, rarity)}
    if g is not None:
        present["grounding"] = g
    x = _llm_extract(sub, ref)
    if x is not None:
        present["llm_extract"] = x
    s = _semantic(sub, ref, emb)
    if s is not None:
        present["semantic"] = s
    wsum = sum(_W[k] for k in present)
    return sum(present[k] * _W[k] for k in present) / wsum


# ---- faithfulness (reference-free) ----
_SENT_POL = {"very_bullish": 1, "bullish": 1, "slightly_bullish": 1, "neutral": 0, "mixed": 0,
             "slightly_bearish": -1, "bearish": -1, "very_bearish": -1, "positive": 1, "negative": -1}
_FINBERT_POL = {"positive": 1, "neutral": 0, "negative": -1}
_WORD = re.compile(r"[^\w]+", re.UNICODE)


def _tok(t):
    return [x for x in _WORD.split((t or "").lower()) if x]


def _tri(toks):
    return {tuple(toks[i:i + 3]) for i in range(len(toks) - 2)}


class Faithfulness:
    """Reference-free: groundedness x abstractiveness x consistency (weighted geo-mean).
    `emb` exposes .sim(a, b). Sentiment classifier is lazy/offline; consistency degrades
    to neutral if unavailable."""

    def __init__(self, emb, use_sentiment=True):
        self.emb = emb
        self._clf = None
        self._clf_failed = not use_sentiment

    def _sent(self, text):
        if self._clf_failed:
            return None
        if self._clf is None:
            try:
                import os
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
                from transformers import pipeline
                self._clf = pipeline("sentiment-analysis", model="ProsusAI/finbert",
                                     truncation=True, max_length=256)
            except Exception:
                self._clf_failed = True
                return None
        try:
            return _FINBERT_POL.get(self._clf(text[:1000])[0]["label"].lower(), 0)
        except Exception:
            return None

    def _grounded(self, synth, article):
        body = (article.get("content") or article.get("title") or "").strip()
        if not synth or not body:
            return 0.0
        return max(0.0, self.emb.sim(synth, body))

    def _abstractive(self, synth, article):
        s_tri = _tri(_tok(synth))
        if not s_tri:
            return 0.0
        b_tri = _tri(_tok(article.get("content")))
        if not b_tri:
            return 1.0
        copy = len(s_tri & b_tri) / len(s_tri)
        return max(0.0, 1.0 - max(0.0, (copy - 0.35) / 0.65))

    def _consistent(self, synth, sub):
        if len(_tok(synth)) < 8:
            return 0.7
        declared = _SENT_POL.get(_ev(sub.get("overall_sentiment", "neutral")))
        if declared is None:
            declared = _SENT_POL.get(_ev(sub.get("sentiment_direction", "neutral")), 0)
        pol = self._sent(synth)
        if pol is None:
            return 0.7
        if declared == pol:
            return 1.0
        return 0.7 if 0 in (declared, pol) else 0.5

    def score(self, sub, article):
        synth = _cs(sub, analytic_only=True)
        g = self._grounded(synth, article)
        a = self._abstractive(synth, article)
        c = self._consistent(synth, sub)
        gw, aw, cw = 0.45, 0.40, 0.15
        return math.exp(gw * math.log(max(g, 1e-6)) + aw * math.log(max(a, 1e-6))
                        + cw * math.log(max(c, 1e-6)))


def graded_score(sub, ref, article, emb, rarity):
    """Continuous per-article score in [0,1]. `sub`/`ref` are model_dump dicts,
    `article` has title/content.

    Reference-free faithfulness is not a term here; it stays a liveness gate, scored
    separately via GradedScorer.faithfulness.
    """
    return min(1.0, max(0.0, composite(sub, ref, article, emb, rarity)))
