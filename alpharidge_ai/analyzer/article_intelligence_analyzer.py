"""
ArticleIntelligenceAnalyzer — Three-stage pipeline for full article analysis.

Stage 1: Deterministic + NER (~1s)
  - text_stats, keyword assets, spaCy+GLiNER+Flair+ReFinED NER, FinBERT sentiment
  - Override dict + Wikidata QID resolution
Stage 2: LLM Call 1 "Extract & Classify" (~8-12s)
  - Full article text + NER hints → classification enums, economic data, quotes, event fingerprint
Stage 3: LLM Call 2 "Reason & Summarize" (~3-5s)
  - Structured fact sheet (NOT raw article) → chart summaries + narrative keywords only.
  - Per-asset sentiment (FinBERT aspect) and contagion (dependency-graph) are computed
    deterministically off-LLM in assembly — they are no longer asked of the LLM.

Total: ~12-18s per article via OpenRouter (default model: deepseek/deepseek-v4-flash).
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import traceback
from datetime import datetime, timezone
from typing import Dict, List, Optional

import bittensor as bt
from openai import OpenAI

from alpharidge_ai.models.article_intelligence import (
    ArticleContentType, ArticleIntelligence, AssetClass, AssetSentiment,
    ChartSummary, ContagionLink, ContagionMechanism, CredibilityFlag,
    DisambiguationMethod, EconomicDataPoint, EconomicEventType,
    EntityRole, EntityType, EventFingerprint, EventType, ExtractedEntity,
    FactualConfidence, ForwardEventType, GeoImpactZone, ImpactPotential,
    InferredImpact, MNPIRiskFlag, MarketAnalysisType, MarketSession,
    NumericClaim, NumericUnit, PositioningSignalType, QuoteExtraction,
    MAX_NUMERIC_CLAIMS, MAX_QUOTES,
    SCHEMA_VERSION, Sentiment, SentimentDirection, SourceAttributionType,
    SourceCategory, SourceMetadata, StalenessFlag, TargetAudience,
    TechnicalQuality, TemporalFocus, TextStatistics, TopicSignature, Urgency,
)
from alpharidge_ai.analyzer.text_stats import compute_text_stats
from alpharidge_ai.analyzer.ner_fusion import NERFusionEngine
from alpharidge_ai.analyzer.llm_cache import LLMCache
from alpharidge_ai.analyzer.aspect_sentiment import AspectSentimentScorer, score_assets
from alpharidge_ai.analyzer.horizon import reconcile_direction_with_horizons
from alpharidge_ai.oracle import floor as oracle_floor

try:
    from alpharidge_ai import config
except ImportError:
    config = None

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MAX_CONTENT_CHARS = 3000


def count_entity_mentions(text: str, forms: List[str]) -> tuple:
    """Deterministic (count, first_offset) for an entity given its surface forms.

    Counts non-overlapping occurrences of ANY form (word-boundary, case-insensitive)
    by merging match intervals — so "Apple" inside "Apple Inc." is counted once, not
    twice. Floors at 1: the NER detected the entity even if no form matches verbatim
    (e.g. it was resolved from a coreference). Pure → miner/validator agree.
    """
    text_l = (text or "").lower()
    intervals = []
    for f in forms or []:
        f = (f or "").strip().lower()
        if len(f) < 3:  # skip tickers / 1-2 char forms — too noisy to count in prose
            continue
        for m in re.finditer(rf"\b{re.escape(f)}\b", text_l):
            intervals.append((m.start(), m.end()))
    if not intervals:
        return 1, None
    intervals.sort()
    merged = 0
    cur_end = -1
    first = intervals[0][0]
    for s, e in intervals:
        if s >= cur_end:       # disjoint from the previous kept interval
            merged += 1
            cur_end = e
        elif e > cur_end:      # overlapping (e.g. Apple ⊂ Apple Inc.) — extend, no new count
            cur_end = e
    return max(1, merged), first


def sanitize_forward_event_date(raw, published_iso: Optional[str]):
    """Drop fabricated/junk forward-event dates.

    A *forward* event cannot predate publication, yet the LLM routinely emits stale
    past dates (e.g. "2025-09-25" on a 2026 article). Returns the cleaned string, or
    None for null-ish junk and any date whose year/full-date is before publication.
    """
    if not raw or not isinstance(raw, str):
        return None
    s = raw.strip()
    if s.lower() in ("", "null", "none", "n/a", "na", "tbd", "unknown", "-"):
        return None
    ym = re.search(r"(\d{4})(?:-(\d{2})(?:-(\d{2}))?)?", s)
    if not ym:
        return s  # non-dated phrasing ("next quarter") — keep as-is
    pub_year = pub_full = None
    if published_iso:
        pm = re.search(r"(\d{4})-(\d{2})-(\d{2})", published_iso)
        if pm:
            pub_year = int(pm.group(1))
            pub_full = pm.group(0)
    year = int(ym.group(1))
    if pub_year is not None:
        if year < pub_year:
            return None
        # same year with a full date that's strictly before publication -> stale
        if year == pub_year and ym.group(2) and ym.group(3) and pub_full:
            if ym.group(0) < pub_full:
                return None
    return s


# GICS / exchange sector strings -> the coarse 9-bucket topic taxonomy (sectors.json).
_GICS_SECTOR_MAP = {
    "information technology": (7, "TECH"),
    "technology": (7, "TECH"),
    "communication services": (7, "TECH"),
    "communications": (7, "TECH"),
    "health care": (8, "SCIENCE"),
    "healthcare": (8, "SCIENCE"),
    "energy": (5, "COMMODITIES"),
    "materials": (5, "COMMODITIES"),
    "basic materials": (5, "COMMODITIES"),
    "financials": (4, "EQUITIES"),
    "financial services": (4, "EQUITIES"),
    "finance": (4, "EQUITIES"),
    "consumer discretionary": (4, "EQUITIES"),
    "consumer staples": (4, "EQUITIES"),
    "consumer cyclical": (4, "EQUITIES"),
    "consumer defensive": (4, "EQUITIES"),
    "industrials": (4, "EQUITIES"),
    "utilities": (4, "EQUITIES"),
    "real estate": (4, "EQUITIES"),
}


_ANALYST_DESK_RE = re.compile(
    r"\b(Research Division|Securities|Capital Markets|Equity Research|Research Analyst|"
    r"Research,? LLC|Global Markets)\b", re.I)
_TRANSCRIPT_MARKERS = ("research division", "conference call participants",
                       "earnings call transcript", "prepared remarks")


# Footer/disclosure boilerplate markers (Motley Fool, Seeking Alpha, generic).
# These start the "stocks mentioned" + position-disclosure tail, which lists tickers
# that are NOT the article's subject and inject noise assets. High-precision phrases
# (not the bare word "disclosure") so real prose isn't truncated.
_DISCLOSURE_RE = re.compile(
    r"(?i)("
    r"\bDisclosure\s*:|"
    r"The Motley Fool (has|owns|recommends|disclosure)|"
    r"\bhas (positions? in|no position)\b|"
    r"\bI\s*/\s*we have (a beneficial|no\b)|"
    r"\bowns shares of\b|"
    r"Past performance is no guarantee|"
    r"\bhas a (beneficial )?(long|short) position\b|"
    r"expresses (my|his|her|their) own opinions"
    r")")


def strip_disclosure_tail(text: str) -> str:
    """Truncate a disclosure/footer tail so its "stocks mentioned" ticker list does
    not become noise assets. Only cuts at a marker in the back portion (>=40% in),
    so an early use of a disclosure-ish phrase in real content is left intact."""
    if not text:
        return text
    floor = len(text) * 0.4
    for m in _DISCLOSURE_RE.finditer(text):
        if m.start() >= floor:
            return text[:m.start()].rstrip()
    return text


def _looks_like_transcript(text: str) -> bool:
    t = (text or "").lower()
    return any(mk in t for mk in _TRANSCRIPT_MARKERS)


def strip_analyst_roster_assets(assets: list, text: str) -> list:
    """Drop sell-side banks that appear only as the analyst roster in an earnings
    transcript (``Bryan Keane - Citigroup Inc., Research Division``). They are
    participants, not the article's subject. Only fires on transcript-shaped text,
    never drops the primary subject, and keeps any bank that also appears in a
    non-attribution (subject) context. Pure → deterministic across the boundary.
    """
    if not _looks_like_transcript(text):
        return assets
    text_l = (text or "").lower()
    kept = []
    for a in assets:
        if getattr(a, "is_primary_subject", False):
            kept.append(a)
            continue
        forms = [f for f in ([getattr(a, "asset_name", "")] + list(getattr(a, "evidence_spans", []) or []))
                 if f and len(f) >= 3]
        positions = []
        for f in forms:
            for m in re.finditer(rf"\b{re.escape(f.lower())}\b", text_l):
                positions.append((m.start(), m.end()))
        if not positions:
            kept.append(a)  # can't verify -> keep (conservative)
            continue
        # An occurrence is an analyst attribution if a desk suffix follows closely.
        all_attrib = all(_ANALYST_DESK_RE.search(text[e:e + 40]) for _, e in positions)
        if not all_attrib:
            kept.append(a)
    return kept


def map_gics_to_sector(gics: Optional[str]) -> tuple:
    """Map a GICS/exchange sector label to (sector_id, symbol) in the 9-bucket
    taxonomy. Unknown/empty -> (9, "OTHER"). Pure → deterministic across miner/validator."""
    key = (gics or "").strip().lower()
    return _GICS_SECTOR_MAP.get(key, (9, "OTHER"))


def _safe_enum(enum_cls, value, default=None):
    if value is None:
        return default
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(str(value).lower().strip())
    except (ValueError, KeyError):
        try:
            return enum_cls(str(value).strip())
        except (ValueError, KeyError):
            return default


def _load_json(filename: str):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


# Map dependency_graph relationship strings -> ContagionMechanism enum values.
# Deterministic, used by the off-LLM graph contagion builder.
_RELATION_MECHANISM_MAP = {
    "supply_chain": "supply_chain",
    "competitor": "competitive",
    "regulatory_spillover": "regulatory_spillover",
    "capital_flow": "capital_flow",
    "ecosystem_dependency": "protocol_dependency",
    "protocol_dependency": "protocol_dependency",
    "collateral": "collateral",
    "macro_sensitivity": "macro_sensitivity",
    "correlation": "correlation",
    "narrative": "narrative",
    "liquid_staking_derivative": "protocol_dependency",
    "ecosystem_token": "protocol_dependency",
}


def _relation_to_mechanism(label: str) -> str:
    return _RELATION_MECHANISM_MAP.get(str(label).lower().strip(), "correlation")


# FinBERT 3-class label -> our coarse sentiment direction (lowercase, matches enum values).
_FINBERT_DIRECTION = {"positive": "bullish", "negative": "bearish", "neutral": "neutral"}


def _article_level_direction(sentence_sentiments: list) -> str:
    """Deterministic article-level sentiment direction from FinBERT sentence votes.

    Majority of bullish vs bearish sentence labels; ties or no sentences -> neutral.
    This is the consensus-safe fallback for per-asset `direction` when FinABSA has
    no asset-specific evidence: it is computed off-LLM and identically by miner and
    validator (unlike the LLM's article sentiment, whose two independent temp-0
    calls are not bit-identical and would break the Tier-2b determinism gate).
    """
    pos = sum(1 for s in (sentence_sentiments or []) if s.get("sentiment") == "bullish")
    neg = sum(1 for s in (sentence_sentiments or []) if s.get("sentiment") == "bearish")
    if pos > neg:
        return "bullish"
    if neg > pos:
        return "bearish"
    return "neutral"


# ============================================================================
# LLM Tool Definitions — Two calls only
# ============================================================================

EXTRACT_CLASSIFY_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_and_classify",
        "description": "Extract structured data and classify the article across all dimensions",
        "parameters": {
            "type": "object",
            "properties": {
                "content_type": {"type": "string", "enum": [e.value for e in ArticleContentType]},
                "sentiment": {"type": "string", "enum": [e.value for e in Sentiment]},
                "sentiment_score": {"type": "number", "description": "-1.0 (very bearish) to 1.0 (very bullish)"},
                "sentiment_direction": {"type": "string", "enum": [e.value for e in SentimentDirection]},
                "market_analysis_type": {"type": "string", "enum": [e.value for e in MarketAnalysisType]},
                "impact_potential": {"type": "string", "enum": [e.value for e in ImpactPotential]},
                "technical_quality": {"type": "string", "enum": [e.value for e in TechnicalQuality]},
                "urgency": {"type": "string", "enum": [e.value for e in Urgency]},
                "temporal_focus": {"type": "string", "enum": [e.value for e in TemporalFocus]},
                "factual_confidence": {"type": "string", "enum": [e.value for e in FactualConfidence]},
                "source_attribution": {"type": "string", "enum": [e.value for e in SourceAttributionType]},
                "positioning_signal": {"type": "string", "enum": [e.value for e in PositioningSignalType]},
                "primary_geo": {"type": "string", "enum": [e.value for e in GeoImpactZone]},
                "target_audience": {"type": "string", "enum": [e.value for e in TargetAudience]},
                "credibility_flag": {"type": "string", "enum": [e.value for e in CredibilityFlag]},
                "economic_data": {"type": "array", "items": {"type": "object", "properties": {
                    "event_type": {"type": "string"}, "event_name": {"type": "string"},
                    "actual_value": {"type": "number"}, "expected_value": {"type": "number"},
                    "previous_value": {"type": "number"}, "unit": {"type": "string"}, "period": {"type": "string"},
                    "reporting_country": {"type": "string"},
                }, "required": ["event_name"]}},
                "quotes": {"type": "array", "items": {"type": "object", "properties": {
                    "speaker": {"type": "string"}, "speaker_title": {"type": "string"},
                    "text": {"type": "string"}, "sentiment": {"type": "string"},
                    "is_market_moving": {"type": "boolean"},
                    "confidence": {"type": "number", "description":
                        "0-1: how certain this quote is verbatim from the article."},
                }, "required": ["speaker", "text", "confidence"]}},
                "numeric_claims": {"type": "array", "items": {"type": "object", "properties": {
                    "metric_name": {"type": "string"}, "value": {"type": "number"},
                    "unit": {"type": "string"}, "context": {"type": "string"},
                    "is_percentage_change": {"type": "boolean"},
                    "confidence": {"type": "number", "description":
                        "0-1: how certain this claim is stated in the article. Be "
                        "honest; a high number on a claim that is not there costs "
                        "more than a low one."},
                }, "required": ["metric_name", "value", "unit", "confidence"],
                }, "description":
                    "Measured quantities only. NOT dates, times, article timestamps, "
                    "years, quarters, page or channel numbers."},
                "event_type": {"type": "string", "enum": [e.value for e in EventType]},
                "event_title": {"type": "string"},
                "event_date": {"type": "string", "description": "YYYY-MM-DD or null"},
                "semantic_fingerprint": {"type": "array", "items": {"type": "string"}},
                "staleness_flag": {"type": "string", "enum": [e.value for e in StalenessFlag]},
                "event_timestamp": {"type": "string"},
                "forward_event_type": {"type": "string", "enum": [e.value for e in ForwardEventType]},
                "forward_event_date": {"type": "string"},
                "forward_event_description": {"type": "string"},
                "additional_tickers": {"type": "array", "items": {"type": "string"},
                    "description": "Asset tickers the NER may have missed"},
            },
            "required": ["content_type", "sentiment", "sentiment_score", "sentiment_direction",
                         "market_analysis_type", "impact_potential", "technical_quality",
                         "urgency", "temporal_focus", "factual_confidence", "positioning_signal",
                         "primary_geo", "target_audience", "credibility_flag",
                         "event_type", "event_title", "semantic_fingerprint",
                         "staleness_flag", "forward_event_type"],
        },
    },
}

REASON_SUMMARIZE_TOOL = {
    "type": "function",
    "function": {
        "name": "reason_and_summarize",
        "description": "Generate chart-ready summaries and narrative keywords from extracted facts",
        "parameters": {
            "type": "object",
            "properties": {
                "headline": {"type": "string", "description": "Max 120 chars"},
                "one_liner": {"type": "string", "description": "Max 280 chars"},
                "context_paragraph": {"type": "string", "description": "2-3 sentences, max 1000 chars"},
                "what_changed": {"type": "string", "description": "Regime shift description or null"},
            },
            "required": ["headline", "one_liner", "context_paragraph"],
        },
    },
}


class ArticleIntelligenceAnalyzer:
    """Three-stage pipeline: NER fusion → LLM extract/classify → LLM reason/summarize."""

    def __init__(self, model: str = None, api_key: str = None, llm_base: str = None,
                 enable_refined: bool = True, enable_flair: bool = True):
        from .determinism import configure_determinism
        configure_determinism()  # reproducible neural output before any model loads
        if config:
            self.model = model or config.MODEL
            self.api_key = api_key or config.API_KEY
            self.llm_base = llm_base or config.LLM_BASE
        else:
            self.model = model
            self.api_key = api_key
            self.llm_base = llm_base

        if not self.api_key:
            raise ValueError("API_KEY is required")

        self.client = OpenAI(base_url=self.llm_base, api_key=self.api_key)

        cache_ttl = float(getattr(config, "LLM_CACHE_TTL", 300)) if config else 300.0
        cache_size = int(getattr(config, "LLM_CACHE_MAX_SIZE", 1024)) if config else 1024
        self._cache = LLMCache(max_size=cache_size, ttl_seconds=cache_ttl)

        self.ner_engine = NERFusionEngine(
            enable_refined=enable_refined,
            enable_flair=enable_flair,
        )
        # Per-asset direction: target-aware FinABSA on CPU for bit-exact output
        # across the miner/validator consensus boundary (lazy model load).
        self._aspect_scorer = AspectSentimentScorer(device="cpu")
        self.source_profiles = _load_json("source_profiles.json").get("profiles", {})
        self.dependency_graph = _load_json("dependency_graph.json").get("dependencies", {})
        self._init_narrative_index()

        bt.logging.info(f"[ARTICLE_INTEL] Ready: model={self.model} endpoint={self.llm_base}")

    # Cosine threshold for narrative-slug selection (calibrated on the GLM gold
    # calibration split: tau=0.30 maximizes mean Jaccard at ~0.371, ~70% of the
    # GLM-vs-Opus ceiling). Overridable via config.NARRATIVE_TAU.
    def _init_narrative_index(self):
        """Precompute one embedding centroid per narrative (name + keywords).

        Narrative slugs are then selected deterministically by cosine similarity
        against the article embedding — no LLM, abstaining when nothing is close.
        """
        import numpy as _np
        self._narr_slugs = []
        self._narr_cent = None
        self._narr_crypto_only = []  # per-narrative: True when sector_ids subset of {crypto}
        self._crypto_terms = ()
        self._crypto_re = None
        self._narr_tau = float(getattr(config, "NARRATIVE_TAU", 0.30)) if config else 0.30
        _CRYPTO_SECTOR_ID = 1
        try:
            # Crypto vocabulary (sector 1) used to gate crypto-only narratives so a
            # rates/yield article can't be tagged bitcoin-halving-cycle / defi-revival.
            import re as _re
            sectors = _load_json("sectors.json")
            if isinstance(sectors, list):
                for s in sectors:
                    if s.get("id") == _CRYPTO_SECTOR_ID:
                        terms = [t.lower() for t in s.get("unique_identifiers", [])]
                        terms += [c.lstrip("$").lower() for c in s.get("cashtags", [])]
                        self._crypto_terms = tuple(sorted({t for t in terms if t}, key=len, reverse=True))
            # Word-boundary match: short cashtag stems ("sol","eth","ada","dot")
            # are substrings of ordinary words, so substring matching mis-fires.
            self._crypto_re = (
                _re.compile(r"\b(?:" + "|".join(_re.escape(t) for t in self._crypto_terms) + r")\b", _re.I)
                if self._crypto_terms else None)
            narr = _load_json("narratives.json")
            if isinstance(narr, list) and narr and self.ner_engine._embedder is not None:
                self._narr_slugs = [n["slug"] for n in narr]
                self._narr_crypto_only = [
                    bool(n.get("sector_ids")) and set(n["sector_ids"]) <= {_CRYPTO_SECTOR_ID}
                    for n in narr
                ]
                texts = [n["name"] + ": " + ", ".join(n.get("keywords") or []) for n in narr]
                self._narr_cent = _np.asarray(
                    self.ner_engine._embedder.encode(texts, normalize_embeddings=True))
        except Exception as ex:
            bt.logging.warning(f"[ARTICLE_INTEL] narrative index unavailable: {ex}")

    def _select_narratives(self, title: str, one_liner: str, ctx: str) -> List[str]:
        """Return up to 3 taxonomy slugs whose centroid clears tau; [] if none.

        Crypto-only narratives are suppressed unless the article actually mentions
        a crypto signal — embedding similarity alone confuses dividend/treasury
        'yield' with DeFi 'yield', so this gate enforces topical presence.
        """
        if self._narr_cent is None or not self._narr_slugs:
            return []
        import numpy as _np
        text = f"{title}. {one_liner} {ctx}"[:2000]
        vec = self.ner_engine.encode_text(text)
        if not vec:
            return []
        has_crypto = bool(self._crypto_re.search(text)) if self._crypto_re else False
        sims = self._narr_cent @ _np.asarray(vec)
        order = sorted(range(len(sims)), key=lambda i: -float(sims[i]))
        out = []
        for i in order:
            if float(sims[i]) < self._narr_tau:
                break
            if self._narr_crypto_only[i] and not has_crypto:
                continue
            out.append(self._narr_slugs[i])
            if len(out) == 3:
                break
        return out

    def analyze(
        self, article_id: int, url: str, title: str, source: str,
        published: Optional[str] = None, summary: Optional[str] = None,
        content: Optional[str] = None, miner_hotkey: Optional[str] = None,
        raw_html: Optional[str] = None,
    ) -> Optional[ArticleIntelligence]:
        start_ms = int(time.time() * 1000)
        body = content or summary or ""
        # Strip the disclosure/footer tail for the EXTRACTION path only (NER, assets,
        # LLM, sectors) so a "stocks mentioned" footer doesn't inject noise tickers.
        # text_stats/content_hash below still use the full `body` (Tier-2 unchanged).
        body_truncated = strip_disclosure_tail(body[:MAX_CONTENT_CHARS])
        article_text = f"{title}\n\n{summary or ''}\n\n{body_truncated}".strip()

        try:
            # ── STAGE 1: Deterministic + NER (~1s) ──
            # text_stats/content_hash stay on the plain `content` (Tier-2 gated
            # fields must not move). Only the NER/entity path consumes raw_html,
            # which lets the content extractor run trafilatura on real DOM.
            text_stats = compute_text_stats(title, body)
            ner_result = self.ner_engine.extract_and_resolve(
                title, body_truncated, raw_html=raw_html)
            # Real detected language (was hardcoded "en"). Drives both the Tier-1
            # detected_language gate and text_stats.language deterministically.
            text_stats.language = ner_result.detected_language
            content_hash = ArticleIntelligence.compute_content_hash(title, body)
            source_meta = self._build_source_metadata(source)
            market_sess = self._compute_market_session(published)

            sector_matches = self.ner_engine._asset_extractor.extract_sectors(title, body_truncated)
            primary_sector = sector_matches[0] if sector_matches else {"id": 9, "symbol": "OTHER"}

            # Build NER hints for LLM Call 1
            ner_hints = self._format_ner_hints(ner_result)
            all_tickers = list({e.ticker for e in ner_result.resolved_assets if e.ticker})

            # ── STAGE 2: LLM Call 1 — Extract & Classify (~8-12s) ──
            call1_prompt = (
                f"Analyze this financial news article. Pre-detected entities are provided as hints — "
                f"confirm, correct, or supplement them.\n\n"
                f"Article:\n\"\"\"{article_text}\"\"\"\n\n"
                f"Pre-detected (NER):\n{ner_hints}"
            )
            call1 = self._llm_call(call1_prompt, EXTRACT_CLASSIFY_TOOL, "extract_and_classify")

            # Merge additional tickers from LLM
            for t in call1.get("additional_tickers", []):
                if t and t.upper() not in {x.upper() for x in all_tickers}:
                    all_tickers.append(t.upper())

            # ── STAGE 3: LLM Call 2 — Reason & Summarize (~5-8s) ──
            fact_sheet = self._build_fact_sheet(title, source, published, primary_sector,
                                                call1, ner_result, all_tickers)
            call2_prompt = (
                f"Based on these extracted facts, write chart-ready summaries: a headline, "
                f"a one-liner, and a context paragraph.\n\n"
                f"{fact_sheet}"
            )
            call2 = self._llm_call(call2_prompt, REASON_SUMMARIZE_TOOL, "reason_and_summarize")

            # ── ASSEMBLY ──
            # Contagion + per-asset sentiment are computed off-LLM from the DETERMINISTIC
            # NER-resolved tickers (NOT all_tickers, which includes non-deterministic
            # LLM-suggested additional_tickers) so they match across the consensus boundary.
            ner_tickers = [e.ticker for e in ner_result.resolved_assets if e.ticker]
            # Per-asset direction fallback MUST be deterministic — it feeds the
            # validator's Tier-2b asset-sentiment determinism gate. The LLM's
            # article sentiment varies across the miner/validator calls, so use the
            # off-LLM FinBERT sentence-majority instead.
            overall_dir = _article_level_direction(ner_result.sentence_sentiments)
            asset_sentiments = score_assets(
                [e for e in ner_result.resolved_assets if e.ticker],
                ner_result, self._aspect_scorer, fallback_direction=overall_dir)
            # Per-asset DIRECTION and HORIZONS both stay deterministic (FinABSA +
            # off-LLM temporal projector) so they survive the validator's outlook
            # determinism gate; reconcile direction against the horizons only.
            self._reconcile_asset_directions(asset_sentiments)
            assets = self._build_assets(ner_result, asset_sentiments)
            # Drop sell-side banks that are only the analyst roster of an earnings
            # transcript (participants, not the article's subject).
            assets = strip_analyst_roster_assets(assets, article_text)
            entities = self._build_entities_from_ner(ner_result, article_text)
            inferred = self._compute_inferred_impacts(all_tickers)

            # Sector from the primary-subject asset's GICS metadata. The keyword
            # sector classifier often returns OTHER for a single-stock article
            # ("Micron: ..." -> OTHER); the covered asset knows its sector, so use
            # it to fill gics_sector/industry and to replace an OTHER primary sector.
            gics_sector, gics_industry = self._primary_asset_gics(assets)
            if gics_sector and primary_sector.get("id") == 9:
                sid, sym = map_gics_to_sector(gics_sector)
                if sym != "OTHER":
                    primary_sector = {"id": sid, "symbol": sym}
            elapsed_ms = int(time.time() * 1000) - start_ms

            # ── STAGE 4: Embeddings (~45ms) ──
            headline = (call2.get("headline") or title[:120])[:120]
            one_liner = (call2.get("one_liner") or title[:280])[:280]
            ctx_para = (call2.get("context_paragraph") or "")[:1000]
            # Narrative slugs are selected deterministically from the taxonomy by
            # embedding similarity (with abstention), NOT by the LLM — this kills
            # the generic-crypto-slug hallucination and removes the field from Call 2.
            narr_kws = self._select_narratives(headline, one_liner, ctx_para)

            title_emb = self.ner_engine.encode_text(title)
            body_emb = self.ner_engine.encode_text(f"{one_liner} {ctx_para}")
            narr_emb = self.ner_engine.encode_text(f"{headline} {', '.join(narr_kws)}")

            result = ArticleIntelligence(
                schema_version=SCHEMA_VERSION,
                article_id=article_id, url=url, title=title,
                published_at=published or "",
                analyzed_at=datetime.now(timezone.utc).isoformat(),
                miner_hotkey=miner_hotkey, analysis_model=self.model,
                analysis_latency_ms=elapsed_ms,
                source=source_meta,
                content_type=_safe_enum(ArticleContentType, call1.get("content_type"), ArticleContentType.OTHER),
                market_analysis_type=_safe_enum(MarketAnalysisType, call1.get("market_analysis_type"), MarketAnalysisType.NONE),
                impact_potential=_safe_enum(ImpactPotential, call1.get("impact_potential"), ImpactPotential.LOW),
                technical_quality=_safe_enum(TechnicalQuality, call1.get("technical_quality"), TechnicalQuality.NONE),
                urgency=_safe_enum(Urgency, call1.get("urgency"), Urgency.SAME_DAY),
                temporal_focus=_safe_enum(TemporalFocus, call1.get("temporal_focus"), TemporalFocus.CURRENT),
                overall_sentiment=_safe_enum(Sentiment, call1.get("sentiment"), Sentiment.NEUTRAL),
                overall_sentiment_score=max(-1.0, min(1.0, float(call1.get("sentiment_score", 0.0) or 0.0))),
                sentiment_direction=_safe_enum(SentimentDirection, call1.get("sentiment_direction"), SentimentDirection.NEUTRAL),
                assets=assets,
                entities=entities,
                economic_data=self._build_economic_data(call1.get("economic_data", [])),
                numeric_claims=self._build_numeric_claims(call1.get("numeric_claims", [])),
                quotes=self._build_quotes(call1.get("quotes", []), content or ""),
                contagion_links=self._build_contagion_from_graph(ner_tickers),
                chart_summary=ChartSummary(
                    headline=headline,
                    one_liner=one_liner,
                    context_paragraph=ctx_para,
                    what_changed=(call2.get("what_changed") or "")[:200] or None,
                ),
                event_fingerprint=EventFingerprint(
                    event_type=_safe_enum(EventType, call1.get("event_type"), EventType.OTHER),
                    event_title=(call1.get("event_title") or title[:200])[:200],
                    event_date=call1.get("event_date"),
                    content_hash=content_hash,
                    semantic_fingerprint=sorted(call1.get("semantic_fingerprint") or [(title.split() or ["article"])[0].lower()])[:10],
                ),
                narrative_keywords=narr_kws,
                title_embedding=title_emb,
                body_embedding=body_emb,
                narrative_embedding=narr_emb,
                topic_signature=TopicSignature(
                    primary_sector_id=primary_sector["id"],
                    primary_sector_symbol=primary_sector["symbol"],
                    secondary_sector_ids=[s["id"] for s in sector_matches[1:4]],
                    gics_sector=gics_sector or None,
                    gics_industry=gics_industry or None,
                ),
                text_stats=text_stats,
                factual_confidence=_safe_enum(FactualConfidence, call1.get("factual_confidence"), FactualConfidence.SPECULATIVE),
                source_attribution_type=_safe_enum(SourceAttributionType, call1.get("source_attribution"), SourceAttributionType.NONE),
                positioning_signal=_safe_enum(PositioningSignalType, call1.get("positioning_signal"), PositioningSignalType.NONE),
                target_audience=_safe_enum(TargetAudience, call1.get("target_audience"), TargetAudience.GENERAL),
                credibility_flag=_safe_enum(CredibilityFlag, call1.get("credibility_flag"), CredibilityFlag.UNVERIFIED),
                primary_geo=_safe_enum(GeoImpactZone, call1.get("primary_geo"), GeoImpactZone.GLOBAL),
                market_session=market_sess,
                detected_language=text_stats.language,
                staleness_flag=_safe_enum(StalenessFlag, call1.get("staleness_flag"), StalenessFlag.UNKNOWN),
                event_timestamp=call1.get("event_timestamp"),
                forward_event_type=_safe_enum(ForwardEventType, call1.get("forward_event_type"), ForwardEventType.NONE),
                forward_event_date_approximate=sanitize_forward_event_date(
                    call1.get("forward_event_date"), published),
                forward_event_description=call1.get("forward_event_description"),
                inferred_impacts=inferred if inferred else None,
            )
            bt.logging.info(f"[ARTICLE_INTEL] Done article {article_id} in {elapsed_ms}ms: "
                           f"{len(assets)} assets, {len(entities)} entities")
            return result

        except Exception as e:
            bt.logging.error(f"[ARTICLE_INTEL] Failed article {article_id}: {e}")
            bt.logging.error(f"[ARTICLE_INTEL] {traceback.format_exc()}")
            return None

    # ========================================================================
    # LLM
    # ========================================================================

    def _llm_call(self, prompt: str, tool: dict, tool_name: str) -> dict:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                tools=[tool],
                tool_choice={"type": "function", "function": {"name": tool_name}},
                temperature=0,
                max_tokens=4000,
            )
            tc = response.choices[0].message.tool_calls
            if not tc:
                bt.logging.warning(f"[ARTICLE_INTEL] {tool_name}: no tool calls returned")
                return {}
            return json.loads(tc[0].function.arguments)
        except Exception as e:
            bt.logging.warning(f"[ARTICLE_INTEL] {tool_name} failed: {e}")
            return {}

    # ========================================================================
    # Fact Sheet Builder (compact input for LLM Call 2)
    # ========================================================================

    def _format_ner_hints(self, ner_result) -> str:
        lines = []
        tickers = [e.ticker for e in ner_result.resolved_assets if e.ticker]
        if tickers:
            lines.append(f"Assets: {', '.join(tickers)}")
        entities = [(e.canonical_name, e.entity_type, e.role or "") for e in ner_result.resolved_entities
                     if e.entity_type in ("person", "regulatory_body", "organization", "economic_indicator")]
        for name, etype, role in entities[:10]:
            role_str = f"/{role}" if role else ""
            lines.append(f"  {name} ({etype}{role_str})")
        money = [m["text"] for m in ner_result.money_values[:5]]
        if money:
            lines.append(f"Money: {', '.join(money)}")
        pcts = [p["text"] for p in ner_result.percentages[:5]]
        if pcts:
            lines.append(f"Percentages: {', '.join(pcts)}")
        sents = [(s["sentiment"], s["text"][:60]) for s in ner_result.sentence_sentiments[:3]]
        if sents:
            lines.append("FinBERT sentiment hints:")
            for direction, txt in sents:
                lines.append(f"  {direction}: {txt}")
        return "\n".join(lines)

    def _build_fact_sheet(self, title, source, published, sector, call1, ner_result, tickers) -> str:
        lines = [
            f"Title: {title}",
            f"Source: {source} | Published: {published or 'unknown'} | Geo: {call1.get('primary_geo', 'global')}",
            f"Event: [{call1.get('event_type', 'other')}] {call1.get('event_title', 'unknown')}",
            f"Classification: {call1.get('content_type', 'other')} | {call1.get('sentiment', 'neutral')} "
            f"({call1.get('sentiment_score', 0)}) | {call1.get('impact_potential', 'low')} impact",
            f"Sector: {sector.get('symbol', 'OTHER')}",
            f"Detected Assets: {', '.join(tickers)}",
        ]
        entities = [(e.canonical_name, e.entity_type) for e in ner_result.resolved_entities
                     if e.entity_type in ("person", "regulatory_body", "organization")]
        if entities:
            lines.append("Entities: " + ", ".join(f"{n} ({t})" for n, t in entities[:8]))
        econ = call1.get("economic_data", [])
        if econ:
            parts = []
            for d in econ[:5]:
                actual = f"{d.get('actual_value', '?')}" if d.get("actual_value") is not None else "?"
                parts.append(f"{d.get('event_name', '?')}: {actual} {d.get('unit', '')}")
            lines.append("Economic Data: " + " | ".join(parts))
        quotes = call1.get("quotes", [])
        if quotes:
            for q in quotes[:2]:
                lines.append(f"Quote: {q.get('speaker', '?')}: \"{q.get('text', '')[:100]}\"")
        sents = ner_result.sentence_sentiments[:3]
        if sents:
            lines.append("FinBERT hints: " + ", ".join(f"{s['sentiment']}({s['score']:.2f})" for s in sents))
        return "\n".join(lines)

    # ========================================================================
    # Assembly Helpers
    # ========================================================================

    def _reconcile_asset_directions(self, sentiments_raw: list) -> None:
        """Reconcile each asset's deterministic FinABSA `direction` against its
        deterministic horizon outlooks.

        Horizons come ONLY from the off-LLM temporal projector (`project_horizons`,
        already in the dict from `score_assets`). They are NEVER overridden by the
        LLM: the validator hard-gates per-asset outlook determinism (scoring.py
        Tier-2b, tol=0.9) and two separate miner/validator LLM calls are not
        bit-identical even at temperature 0 — an LLM override drops outlook
        determinism and gets the whole batch rejected. When the FinABSA direction
        contradicts ALL horizons, the horizons are the stronger (multi-bucket)
        signal, so reconcile the direction toward them; the horizons themselves
        stay untouched.
        """
        for s in sentiments_raw:
            s["direction"] = reconcile_direction_with_horizons(
                s.get("direction") or "neutral",
                s.get("short_term") or "neutral",
                s.get("medium_term") or "neutral",
                s.get("long_term") or "neutral")

    def _build_assets(self, ner_result, sentiments_raw: list) -> List[AssetSentiment]:
        sentiment_by_ticker = {}
        for s in sentiments_raw:
            t = s.get("ticker", "").upper()
            if t:
                sentiment_by_ticker[t] = s

        assets = []
        for e in ner_result.resolved_assets:
            if not e.ticker:
                continue
            s = sentiment_by_ticker.get(e.ticker.upper(), {})
            try:
                assets.append(AssetSentiment(
                    ticker=e.ticker,
                    asset_name=e.canonical_name,
                    asset_class=_safe_enum(AssetClass, e.asset_class, AssetClass.UNKNOWN),
                    coingecko_id=None,
                    yahoo_ticker=None,
                    direction=_safe_enum(Sentiment, s.get("direction"), Sentiment.NEUTRAL),
                    magnitude=max(0.0, min(1.0, float(s.get("magnitude", 0.5) or 0.5))),
                    confidence=max(0.0, min(1.0, float(s.get("confidence", 0.5) or 0.5))),
                    short_term_outlook=_safe_enum(Sentiment, s.get("short_term"), Sentiment.NEUTRAL),
                    medium_term_outlook=_safe_enum(Sentiment, s.get("medium_term"), Sentiment.NEUTRAL),
                    long_term_outlook=_safe_enum(Sentiment, s.get("long_term"), Sentiment.NEUTRAL),
                    causal_driver=(s.get("causal_driver") or "No specific driver identified")[:500],
                    relevance_score=min(1.0, e.confidence),
                    is_primary_subject=getattr(e, "is_primary_subject", False),
                    resolved_via=getattr(e, "source", "keyword") or "keyword",
                    evidence_spans=[e.text],
                ))
            except Exception as ex:
                bt.logging.debug(f"[ARTICLE_INTEL] Skip asset {e.ticker}: {ex}")
        return assets

    def _primary_asset_gics(self, assets) -> tuple:
        """(gics_sector, gics_industry) for the primary-subject asset, from the
        static universe metadata. Falls back to the highest-relevance asset."""
        if not assets:
            return None, None
        if not hasattr(self, "_ticker_gics"):
            self._ticker_gics = {}
            for a in self.ner_engine._asset_extractor.assets.values():
                tk = a.get("symbol") or a.get("ticker")
                if tk:
                    self._ticker_gics[tk] = (a.get("sector") or None, a.get("industry") or None)
        # Only the PRIMARY subject's sector describes the article. A merely-mentioned
        # asset (e.g. BTC in a Fed-policy piece) must not set gics_sector.
        primary = next((a for a in assets if getattr(a, "is_primary_subject", False)), None)
        if primary is None:
            return None, None
        return self._ticker_gics.get(primary.ticker, (None, None))

    def _build_entities_from_ner(self, ner_result, text: str = "") -> List[ExtractedEntity]:
        # Deduplicate by (canonical name, type): the same entity resolved from
        # several spans (e.g. "Medtronic" x4) must collapse to one record — which
        # is also how the eval `entities` metric keys (name.lower()). On collision
        # keep the highest-confidence representative and merge in a more specific
        # role / a sentiment signal so we never lose information to dedup.
        best = {}  # (name_lower, etype) -> (confidence, ExtractedEntity)
        forms_by_key = {}  # key -> set of surface forms (for mention counting)
        for e in ner_result.resolved_entities:
            try:
                etype = _safe_enum(EntityType, e.entity_type, EntityType.ORGANIZATION)
                if etype is None:
                    etype = EntityType.ORGANIZATION
                role = _safe_enum(EntityRole, e.role, EntityRole.MENTIONED)
                ent = ExtractedEntity(
                    name=e.canonical_name,
                    entity_type=etype,
                    role=role,
                    ticker=e.ticker,
                    sentiment_toward=_safe_enum(Sentiment, e.sentiment_toward, None),
                )
                key = (e.canonical_name.lower(), etype)
                forms = forms_by_key.setdefault(key, set())
                forms.add(e.canonical_name)
                for sf in (getattr(e, "surface_forms", None) or []):
                    forms.add(sf)
                conf = float(getattr(e, "confidence", 0.0) or 0.0)
                if key not in best:
                    best[key] = (conf, ent)
                else:
                    prev_conf, prev = best[key]
                    if prev.role == EntityRole.MENTIONED and role != EntityRole.MENTIONED:
                        prev.role = role
                    if prev.sentiment_toward is None and ent.sentiment_toward is not None:
                        prev.sentiment_toward = ent.sentiment_toward
                    if conf > prev_conf:
                        # keep the higher-confidence record, preserving any merged role/sentiment
                        ent.role = ent.role if ent.role != EntityRole.MENTIONED else prev.role
                        if ent.sentiment_toward is None:
                            ent.sentiment_toward = prev.sentiment_toward
                        best[key] = (conf, ent)
            except Exception:
                pass
        # Real per-entity mention_count / first_mention_offset from the article text
        # (was hardcoded to the model default of 1). Deterministic.
        for key, (_, ent) in best.items():
            cnt, off = count_entity_mentions(text, sorted(forms_by_key.get(key, set())))
            ent.mention_count = cnt
            if off is not None:
                ent.first_mention_offset = off
        return [v[1] for v in best.values()][:15]

    def _build_economic_data(self, raw: list) -> List[EconomicDataPoint]:
        points = []
        for d in (raw or [])[:5]:
            try:
                actual = d.get("actual_value")
                expected = d.get("expected_value")
                previous = d.get("previous_value")
                points.append(EconomicDataPoint(
                    event_type=_safe_enum(EconomicEventType, d.get("event_type"), EconomicEventType.OTHER),
                    event_name=d.get("event_name", "Unknown"),
                    actual_value=actual, expected_value=expected, previous_value=previous,
                    delta_vs_expected=(actual - expected) if actual is not None and expected is not None else None,
                    delta_vs_previous=(actual - previous) if actual is not None and previous is not None else None,
                    unit=_safe_enum(NumericUnit, d.get("unit"), NumericUnit.OTHER),
                    period=d.get("period"), reporting_country=d.get("reporting_country"),
                ))
            except Exception:
                pass
        return points

    # A claim that is really a date. Dates were 7.2% of extracted claims and no floor
    # should ground one. Deliberately conservative: dropping an honest measurement
    # costs a miner credit, while a date that slips through only costs it precision,
    # so every rule here needs an explicit date signal rather than a resemblance.
    # Units that only ever name a point in time. Durations ("3 hours of outage") are
    # measurements and are kept; a date needs a date-shaped value or metric to go with
    # a calendar unit.
    _DATE_UNITS = {"date", "datetime", "timestamp", "yyyy", "dd/mm/yyyy"}
    _CALENDAR_UNITS = {"year", "years", "month", "months", "day", "days",
                       "week", "weeks", "quarter", "quarters", "time",
                       "hour", "hours", "minute", "minutes"}
    _DATE_METRICS = {"date", "datum", "fecha", "data", "time", "hora", "uhrzeit",
                     "year", "jahr", "ano", "month", "day", "timestamp", "deadline",
                     "quarter"}
    _DATE_METRIC_TAIL = ("date", "datum", "fecha", "timestamp", "time")
    _QUARTER = re.compile(r"^q[1-4](\s|$)", re.I)
    # A written date needs two separators (28.08.2026) or ISO order (2026-09-03).
    _DATE_STRING = re.compile(
        r"^\s*(?:\d{1,4}[./-]\d{1,2}[./-]\d{1,4}|\d{4}-\d{2}-\d{2})\s*$")

    @classmethod
    def _is_date_claim(cls, raw: dict) -> bool:
        unit = str(raw.get("unit") or "").strip().lower()
        if unit in cls._DATE_UNITS:
            return True

        metric = str(raw.get("metric_name") or "").strip().lower()
        # A calendar unit carrying anything but a year is a duration, whatever the
        # metric is called: "delivery time: 2 weeks" measures a span, not an instant.
        # The unit settles it, so this comes before the metric-name rules.
        if unit in cls._CALENDAR_UNITS:
            # Models send years as numbers and as strings; pydantic coerces the
            # string later, so checking only the numeric form lets half of them past.
            try:
                number = float(str(raw.get("value")).strip())
            except (TypeError, ValueError):
                return False
            return 1900 <= number <= 2100 and number == int(number)
        words = re.findall(r"[a-z\u00c0-\u024f]+", metric)
        if cls._QUARTER.match(metric):
            return True
        if words and (words[-1] in cls._DATE_METRIC_TAIL
                      or (len(words) <= 2 and set(words) <= cls._DATE_METRICS)):
            return True

        # Only a value still written as a date; a float is indistinguishable from a
        # decimal and is left alone.
        value = raw.get("value")
        return isinstance(value, str) and bool(cls._DATE_STRING.match(value))

    @staticmethod
    def _confidence(raw, default: float = 0.5) -> float:
        """A stated confidence in [0, 1]. Missing or unusable becomes the neutral value
        rather than an optimistic one."""
        try:
            return min(1.0, max(0.0, float(raw)))
        except (TypeError, ValueError):
            return default

    def _build_numeric_claims(self, raw: list) -> List[NumericClaim]:
        claims = []
        for c in (raw or []):
            if len(claims) >= MAX_NUMERIC_CLAIMS:
                break
            if not isinstance(c, dict) or self._is_date_claim(c):
                continue
            try:
                claims.append(NumericClaim(
                    metric_name=c["metric_name"], value=c["value"],
                    unit=c.get("unit", "other"),
                    context=(c.get("context") or "")[:200],
                    is_percentage_change=c.get("is_percentage_change", False),
                    confidence=self._confidence(c.get("confidence")),
                ))
            except Exception:
                pass
        return claims

    def _build_quotes(self, raw: list, content: str = "") -> List[QuoteExtraction]:
        """Quotes, with their span located in the served text.

        Offsets are found here rather than asked of the model: the same alignment the
        validator uses, run over the same text, so both sides agree on where a quote
        sits instead of trusting a model to count characters.
        """
        article = oracle_floor.normalize(content)
        quotes = []
        for q in (raw or [])[:MAX_QUOTES]:
            try:
                text = q["text"][:1000]
                if not article.text:
                    continue
                hit = oracle_floor.align_quote(article, text, None, None)
                if hit is None:
                    # A quote we cannot locate cannot carry offsets, and a 1.2.0
                    # submission missing them fails the cutover gate for the whole
                    # article. Dropping the one quote costs a little recall; keeping it
                    # would cost the article.
                    continue
                start, end = hit.start, hit.end
                quotes.append(QuoteExtraction(
                    speaker=q["speaker"], speaker_title=q.get("speaker_title"),
                    text=text,
                    sentiment=_safe_enum(Sentiment, q.get("sentiment"), Sentiment.NEUTRAL),
                    is_market_moving=q.get("is_market_moving", False),
                    confidence=self._confidence(q.get("confidence")),
                    start_offset=start, end_offset=end,
                ))
            except Exception:
                pass
        return quotes

    def _build_contagion_from_graph(self, tickers: List[str]) -> List[ContagionLink]:
        """Deterministic contagion links from the curated dependency_graph.

        For each detected ticker, emit one ContagionLink per known dependent edge.
        Direction is NEUTRAL and strength/confidence a fixed 0.7 prior (the graph
        encodes structure, not event-specific magnitude). Replaces the former
        LLM-reasoned contagion — both miner and validator compute this identically,
        so it is deterministic across the consensus boundary.
        """
        links = []
        seen = set()
        for t in tickers:
            for dep in self.dependency_graph.get(t, {}).get("dependents", []):
                target = dep.get("ticker")
                if not target:
                    continue
                key = (t, target)
                if key in seen:
                    continue
                seen.add(key)
                try:
                    links.append(ContagionLink(
                        source_ticker=t,
                        target_ticker=target,
                        target_asset_class=_safe_enum(AssetClass, dep.get("asset_class"), AssetClass.UNKNOWN),
                        mechanism=_safe_enum(ContagionMechanism,
                                             _relation_to_mechanism(dep.get("relationship", "")),
                                             ContagionMechanism.CORRELATION),
                        direction=SentimentDirection.NEUTRAL,
                        strength=0.7,
                        confidence=0.7,
                        lag_hours=None,
                        reasoning=f"dependency_graph: {t} -> {target} ({dep.get('relationship', 'correlation')})"[:300],
                    ))
                except Exception:
                    pass
                if len(links) >= 8:
                    return links
        return links

    def _build_source_metadata(self, source: str) -> SourceMetadata:
        key = source.lower().replace(" ", "_").replace("-", "_")
        profile = self.source_profiles.get(key, {})
        return SourceMetadata(
            source_id=key, source_name=profile.get("name", source),
            credibility_score=profile.get("credibility", 0.5),
            source_category=_safe_enum(SourceCategory, profile.get("category"), SourceCategory.UNKNOWN),
        )

    def _compute_market_session(self, published: Optional[str]) -> MarketSession:
        if not published:
            return MarketSession.REGULAR_HOURS
        try:
            # Normalize to UTC so weekday()/hour are representation-independent. The hour
            # buckets below are UTC-based (US market open 09:30 ET == 13:30 UTC), so a
            # tz-aware input in another offset would otherwise misclassify the session and
            # flip WEEKEND/AFTER_HOURS across the Fri/Sat boundary for the same instant.
            dt = datetime.fromisoformat(published.replace("Z", "+00:00")).astimezone(timezone.utc)
            if dt.weekday() >= 5:
                return MarketSession.WEEKEND
            h = dt.hour
            if 13 <= h < 14:
                return MarketSession.PRE_MARKET
            if 14 <= h < 21:
                return MarketSession.REGULAR_HOURS
            return MarketSession.AFTER_HOURS
        except (ValueError, TypeError):
            return MarketSession.REGULAR_HOURS

    def _compute_inferred_impacts(self, tickers: List[str]) -> List[InferredImpact]:
        impacts = []
        seen = set(t.upper() for t in tickers)
        for ticker in tickers:
            deps = self.dependency_graph.get(ticker, {}).get("dependents", [])
            for dep in deps:
                if dep["ticker"].upper() not in seen:
                    seen.add(dep["ticker"].upper())
                    impacts.append(InferredImpact(
                        ticker=dep["ticker"],
                        asset_class=_safe_enum(AssetClass, dep.get("asset_class"), AssetClass.UNKNOWN),
                        relationship=dep.get("relationship", "unknown"),
                        confidence=0.7,
                    ))
        return impacts[:20]
