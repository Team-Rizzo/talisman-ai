"""
ArticleIntelligence: Complete per-article output schema for SN45 news intelligence pipeline.

Miners produce one ArticleIntelligence object per article. Validators re-run
the analysis and compare across 4 validation tiers. The API stores the full
object and performs cross-article aggregation (event clustering, narrative
matching, sentiment momentum, etc.).

Schema version is included for forward compatibility during network upgrades.
"""

from __future__ import annotations

import hashlib
import re
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# 1.1.0 adds the additive triage keys (`triage`, `proof_of_read`,
# `triage_schema_version`) carried alongside the analysis payload. Not gated by
# any validation tier, so miners and validators may run mixed versions.
# 1.2.0 adds per-claim and per-quote confidence, quote character offsets, and caps on
# every extraction list. The new fields are optional at the model layer so submissions
# on 1.1.0 still parse during the grace window; presence is enforced at the scoring
# layer from the cutover block onward.
SCHEMA_VERSION = "1.2.0"
SCHEMA_VERSION_PREV = "1.1.0"

# Caps on extraction lists. A submission is truncated to these, so reporting more
# cannot buy a larger scoreable set.
MAX_ASSETS = 40
MAX_ENTITIES = 60
MAX_ECONOMIC_DATA = 40
MAX_NUMERIC_CLAIMS = 40
MAX_QUOTES = 20
MAX_CONTAGION_LINKS = 20


# ============================================================================
# ENUMS — Article-Level Classification
# ============================================================================


class ArticleContentType(str, Enum):
    BREAKING_NEWS = "breaking_news"
    ANALYSIS = "analysis"
    OPINION = "opinion"
    EARNINGS = "earnings"
    REGULATORY = "regulatory"
    RESEARCH = "research"
    PRESS_RELEASE = "press_release"
    INTERVIEW = "interview"
    MARKET_RECAP = "market_recap"
    DATA_RELEASE = "data_release"
    FORECAST = "forecast"
    INVESTIGATIVE = "investigative"
    TUTORIAL = "tutorial"
    LISTICLE = "listicle"
    SPONSORED = "sponsored"
    OTHER = "other"


class MarketAnalysisType(str, Enum):
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    MACRO = "macro"
    POLITICAL = "political"
    SOCIAL = "social"
    QUANTITATIVE = "quantitative"
    ON_CHAIN = "on_chain"
    MIXED = "mixed"
    NONE = "none"


class ImpactPotential(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NEGLIGIBLE = "negligible"


class TechnicalQuality(str, Enum):
    EXCEPTIONAL = "exceptional"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


class Urgency(str, Enum):
    FLASH = "flash"
    BREAKING = "breaking"
    DEVELOPING = "developing"
    SAME_DAY = "same_day"
    EVERGREEN = "evergreen"


class TemporalFocus(str, Enum):
    RETROSPECTIVE = "retrospective"
    CURRENT = "current"
    FORWARD_LOOKING = "forward_looking"
    MIXED = "mixed"


# ============================================================================
# ENUMS — Sentiment
# ============================================================================


class Sentiment(str, Enum):
    VERY_BULLISH = "very_bullish"
    BULLISH = "bullish"
    SLIGHTLY_BULLISH = "slightly_bullish"
    NEUTRAL = "neutral"
    SLIGHTLY_BEARISH = "slightly_bearish"
    BEARISH = "bearish"
    VERY_BEARISH = "very_bearish"


class SentimentDirection(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    MIXED = "mixed"


# ============================================================================
# ENUMS — Asset Classification
# ============================================================================


class AssetClass(str, Enum):
    CRYPTO = "crypto"
    EQUITY = "equity"
    FOREX = "forex"
    COMMODITY = "commodity"
    INDEX = "index"
    FIXED_INCOME = "fixed_income"
    DERIVATIVE = "derivative"
    UNKNOWN = "unknown"


# ============================================================================
# ENUMS — Entity Extraction
# ============================================================================


class EntityType(str, Enum):
    PERSON = "person"
    ORGANIZATION = "organization"
    PRODUCT = "product"
    PROTOCOL = "protocol"
    EXCHANGE = "exchange"
    REGULATORY_BODY = "regulatory_body"
    GOVERNMENT = "government"
    LOCATION = "location"
    EVENT = "event"
    METRIC = "metric"


class EntityRole(str, Enum):
    SUBJECT = "subject"
    OBJECT = "object"
    SOURCE = "source"
    COMPETITOR = "competitor"
    REGULATOR = "regulator"
    BENEFICIARY = "beneficiary"
    VICTIM = "victim"
    ANALYST = "analyst"
    MENTIONED = "mentioned"


# ============================================================================
# ENUMS — Economic Events
# ============================================================================


class EconomicEventType(str, Enum):
    FOMC_DECISION = "fomc_decision"
    FOMC_MINUTES = "fomc_minutes"
    ECB_DECISION = "ecb_decision"
    BOJ_DECISION = "boj_decision"
    BOE_DECISION = "boe_decision"
    CENTRAL_BANK_SPEECH = "central_bank_speech"
    CPI = "cpi"
    CORE_CPI = "core_cpi"
    PPI = "ppi"
    PCE = "pce"
    CORE_PCE = "core_pce"
    NFP = "nfp"
    UNEMPLOYMENT_RATE = "unemployment_rate"
    JOBLESS_CLAIMS = "jobless_claims"
    ADP_EMPLOYMENT = "adp_employment"
    GDP = "gdp"
    GDP_REVISION = "gdp_revision"
    RETAIL_SALES = "retail_sales"
    CONSUMER_CONFIDENCE = "consumer_confidence"
    HOUSING_STARTS = "housing_starts"
    EXISTING_HOME_SALES = "existing_home_sales"
    ISM_MANUFACTURING = "ism_manufacturing"
    ISM_SERVICES = "ism_services"
    DURABLE_GOODS = "durable_goods"
    INDUSTRIAL_PRODUCTION = "industrial_production"
    TRADE_BALANCE = "trade_balance"
    EARNINGS_REPORT = "earnings_report"
    TOKEN_UNLOCK = "token_unlock"
    PROTOCOL_UPGRADE = "protocol_upgrade"
    HALVING = "halving"
    OTHER = "other"


class NumericUnit(str, Enum):
    PERCENT = "percent"
    PERCENT_CHANGE = "percent_change"
    BASIS_POINTS = "basis_points"
    USD_BILLIONS = "usd_billions"
    USD_MILLIONS = "usd_millions"
    USD_TRILLIONS = "usd_trillions"
    THOUSANDS = "thousands"
    MILLIONS = "millions"
    RATIO = "ratio"
    INDEX_POINTS = "index_points"
    USD_PER_UNIT = "usd_per_unit"
    COUNT = "count"
    OTHER = "other"


# ============================================================================
# ENUMS — Event Fingerprinting
# ============================================================================


class EventType(str, Enum):
    MONETARY_POLICY = "monetary_policy"
    EARNINGS = "earnings"
    REGULATORY = "regulatory"
    MARKET_MOVE = "market_move"
    PRODUCT_LAUNCH = "product_launch"
    PARTNERSHIP = "partnership"
    SECURITY_INCIDENT = "security_incident"
    LEGAL = "legal"
    ECONOMIC_DATA = "economic_data"
    GEOPOLITICAL = "geopolitical"
    PERSONNEL = "personnel"
    FUNDING = "funding"
    SUPPLY_CHAIN = "supply_chain"
    BANKRUPTCY = "bankruptcy"
    IPO_LISTING = "ipo_listing"
    TOKEN_EVENT = "token_event"
    NATURAL_DISASTER = "natural_disaster"
    OTHER = "other"


# ============================================================================
# ENUMS — Contagion
# ============================================================================


class ContagionMechanism(str, Enum):
    CORRELATION = "correlation"
    SUPPLY_CHAIN = "supply_chain"
    REGULATORY_SPILLOVER = "regulatory_spillover"
    CAPITAL_FLOW = "capital_flow"
    NARRATIVE = "narrative"
    COLLATERAL = "collateral"
    PROTOCOL_DEPENDENCY = "protocol_dependency"
    COMPETITIVE = "competitive"
    MACRO_SENSITIVITY = "macro_sensitivity"


# ============================================================================
# ENUMS — Additional Signals
# ============================================================================


class FactualConfidence(str, Enum):
    CONFIRMED = "confirmed"
    ATTRIBUTED = "attributed"
    SPECULATIVE = "speculative"
    CONDITIONAL = "conditional"
    RUMOR = "rumor"


class PositioningSignalType(str, Enum):
    INSTITUTIONAL_BUYING = "institutional_buying"
    INSTITUTIONAL_SELLING = "institutional_selling"
    SHORT_INTEREST_HIGH = "short_interest_high"
    SHORT_INTEREST_LOW = "short_interest_low"
    ETF_INFLOW = "etf_inflow"
    ETF_OUTFLOW = "etf_outflow"
    WHALE_ACCUMULATION = "whale_accumulation"
    WHALE_DISTRIBUTION = "whale_distribution"
    RETAIL_INFLOW = "retail_inflow"
    RETAIL_OUTFLOW = "retail_outflow"
    OPTIONS_SKEW_BULLISH = "options_skew_bullish"
    OPTIONS_SKEW_BEARISH = "options_skew_bearish"
    NONE = "none"


class TargetAudience(str, Enum):
    INSTITUTIONAL = "institutional"
    PROFESSIONAL = "professional"
    RETAIL = "retail"
    DEVELOPER = "developer"
    REGULATORY = "regulatory"
    GENERAL = "general"


class ForwardEventType(str, Enum):
    FOMC_MEETING = "fomc_meeting"
    EARNINGS_RELEASE = "earnings_release"
    TOKEN_UNLOCK = "token_unlock"
    PROTOCOL_UPGRADE = "protocol_upgrade"
    REGULATORY_DEADLINE = "regulatory_deadline"
    ETF_DECISION = "etf_decision"
    HALVING = "halving"
    ELECTION = "election"
    OTHER_SCHEDULED = "other_scheduled"
    NONE = "none"


class GeoImpactZone(str, Enum):
    US = "us"
    EU = "eu"
    UK = "uk"
    CHINA = "china"
    JAPAN = "japan"
    SOUTH_KOREA = "south_korea"
    ASIA_EX_CHINA = "asia_ex_china"
    LATAM = "latam"
    MENA = "mena"
    AFRICA = "africa"
    GLOBAL = "global"


class CredibilityFlag(str, Enum):
    VERIFIED_SOURCE = "verified_source"
    KNOWN_SATIRE = "known_satire"
    SUSPECTED_SPAM = "suspected_spam"
    SUSPECTED_AI_GENERATED = "suspected_ai_generated"
    LOW_CREDIBILITY = "low_credibility"
    UNVERIFIED = "unverified"


class StalenessFlag(str, Enum):
    FRESH = "fresh"
    RECYCLED_CONTENT = "recycled_content"
    EVERGREEN = "evergreen"
    UNKNOWN = "unknown"


class ArticleStatus(str, Enum):
    LIVE = "live"
    CORRECTED = "corrected"
    RETRACTED = "retracted"
    UPDATED = "updated"


class SourceCategory(str, Enum):
    WIRE_SERVICE = "wire_service"
    BROADSHEET = "broadsheet"
    TRADE_PRESS = "trade_press"
    CRYPTO_NATIVE = "crypto_native"
    BLOG = "blog"
    AGGREGATOR = "aggregator"
    GOVERNMENT = "government"
    ACADEMIC = "academic"
    UNKNOWN = "unknown"


class MarketSession(str, Enum):
    PRE_MARKET = "pre_market"
    REGULAR_HOURS = "regular_hours"
    AFTER_HOURS = "after_hours"
    WEEKEND = "weekend"


class MNPIRiskFlag(str, Enum):
    NONE = "none"
    POSSIBLE_LEAK = "possible_leak"
    UNVERIFIED_INSIDER = "unverified_insider"


class DisambiguationMethod(str, Enum):
    CASHTAG = "cashtag"
    KEYWORD_HIGH = "keyword_high"
    KEYWORD_CONTEXTUAL = "keyword_contextual"
    LLM_FALLBACK = "llm_fallback"
    NONE = "none"


class SourceAttributionType(str, Enum):
    NAMED_OFFICIAL = "named_official"
    NAMED_ANALYST = "named_analyst"
    ANONYMOUS = "anonymous"
    NONE = "none"


# ============================================================================
# SUB-OBJECTS — Nested Pydantic models
# ============================================================================


class AssetSentiment(BaseModel):
    """Per-asset sentiment decomposition."""

    ticker: str
    asset_name: str
    asset_class: AssetClass
    coingecko_id: Optional[str] = None
    yahoo_ticker: Optional[str] = None

    direction: Sentiment
    magnitude: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)

    short_term_outlook: Sentiment
    medium_term_outlook: Sentiment
    long_term_outlook: Sentiment

    causal_driver: str = Field(..., max_length=500)

    relevance_score: float = Field(..., ge=0.0, le=1.0)
    is_primary_subject: bool = False
    # How the ticker was resolved: "keyword" is the pure-gazetteer string matcher
    # (bit-identical cross-host); "override" is a deterministic dict lookup but fires
    # only on a neural-NER-surfaced span; "refined"/model are neural. The validator's
    # asset-presence gate counts ONLY "keyword" toward its floor (see scoring.py).
    # Defaults to "keyword" for backward-compat deserialization.
    resolved_via: str = "keyword"

    evidence_spans: List[str] = Field(default_factory=list)

    class Config:
        populate_by_name = True


class ExtractedEntity(BaseModel):
    """A named entity extracted from the article."""

    name: str
    entity_type: EntityType
    role: EntityRole = EntityRole.MENTIONED
    ticker: Optional[str] = None
    mention_count: int = Field(1, ge=1)
    first_mention_offset: Optional[int] = Field(None, ge=0)
    sentiment_toward: Optional[Sentiment] = None

    class Config:
        populate_by_name = True


class EconomicDataPoint(BaseModel):
    """A quantitative economic data point extracted from the article."""

    event_type: EconomicEventType
    event_name: str
    actual_value: Optional[float] = None
    expected_value: Optional[float] = None
    previous_value: Optional[float] = None
    delta_vs_expected: Optional[float] = None
    delta_vs_previous: Optional[float] = None
    unit: NumericUnit = NumericUnit.OTHER
    period: Optional[str] = None
    reporting_country: Optional[str] = None
    reporting_body: Optional[str] = None

    class Config:
        populate_by_name = True


class NumericClaim(BaseModel):
    """A specific numeric claim extracted from the article."""

    metric_name: str
    value: float
    unit: str
    context: Optional[str] = Field(None, max_length=200)
    is_percentage_change: bool = False
    comparison_period: Optional[str] = None
    # Stated probability that this claim holds. Required from 1.2.0; None only during
    # the grace window, where it is scored at a neutral constant.
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)

    class Config:
        populate_by_name = True


class QuoteExtraction(BaseModel):
    """A notable direct quote extracted from the article."""

    speaker: str
    speaker_title: Optional[str] = None
    text: str = Field(..., max_length=1000)
    sentiment: Sentiment = Sentiment.NEUTRAL
    is_market_moving: bool = False
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    # Character offsets of the quote in the served article text. The validator keys a
    # quote by the span it finds itself, so these say where to look, not what counts.
    start_offset: Optional[int] = Field(None, ge=0)
    end_offset: Optional[int] = Field(None, ge=0)

    class Config:
        populate_by_name = True


class ContagionLink(BaseModel):
    """A predicted cross-market impact chain link."""

    source_ticker: str
    target_ticker: str
    target_asset_class: AssetClass
    mechanism: ContagionMechanism
    direction: SentimentDirection
    strength: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    lag_hours: Optional[float] = Field(None, ge=0.0)
    reasoning: str = Field(..., max_length=300)

    class Config:
        populate_by_name = True


class ChartSummary(BaseModel):
    """Context-rich summary for chart overlay annotations."""

    headline: str = Field(..., max_length=120)
    one_liner: str = Field(..., max_length=280)
    context_paragraph: str = Field(..., max_length=1000)
    what_changed: Optional[str] = Field(None, max_length=200)

    class Config:
        populate_by_name = True


class EventFingerprint(BaseModel):
    """Fields enabling downstream event dedup and clustering."""

    event_type: EventType
    event_title: str = Field(..., max_length=200)
    event_date: Optional[str] = None
    content_hash: str
    semantic_fingerprint: List[str] = Field(default_factory=list, max_length=10)
    calendar_event_id: Optional[str] = None

    class Config:
        populate_by_name = True


class NarrativeTag(BaseModel):
    """Links this article to a tracked market narrative."""

    narrative_id: str
    narrative_name: str
    relevance: float = Field(..., ge=0.0, le=1.0)
    stance: SentimentDirection

    class Config:
        populate_by_name = True


class TopicSignature(BaseModel):
    """Topic/keyword representation for clustering and search."""

    keywords: Dict[str, float] = Field(default_factory=dict)
    primary_sector_id: int
    primary_sector_symbol: str
    secondary_sector_ids: List[int] = Field(default_factory=list)
    topics: List[str] = Field(default_factory=list, min_length=0, max_length=5)
    gics_sector: Optional[str] = None
    gics_industry: Optional[str] = None

    class Config:
        populate_by_name = True


class TextStatistics(BaseModel):
    """Computed text features for ML pipelines. Deterministic (no LLM)."""

    # Length
    char_count: int = Field(..., ge=0)
    word_count: int = Field(..., ge=0)
    sentence_count: int = Field(..., ge=0)
    paragraph_count: int = Field(..., ge=0)

    # Readability
    avg_sentence_length: float = Field(..., ge=0.0)
    avg_word_length: float = Field(..., ge=0.0)
    flesch_reading_ease: Optional[float] = None
    flesch_kincaid_grade: Optional[float] = None

    # Density
    numeric_density: float = Field(..., ge=0.0, le=1.0)
    quote_density: float = Field(..., ge=0.0, le=1.0)
    named_entity_density: float = Field(0.0, ge=0.0, le=1.0)
    ticker_mention_count: int = Field(0, ge=0)
    unique_ticker_count: int = Field(0, ge=0)
    link_count: int = Field(0, ge=0)
    image_count: int = Field(0, ge=0)

    # Structure
    has_table: bool = False
    has_chart_image: bool = False
    has_code_block: bool = False
    subheading_count: int = Field(0, ge=0)
    title_word_count: int = Field(0, ge=0)
    title_has_number: bool = False
    title_has_question: bool = False
    title_sentiment: Optional[Sentiment] = None

    # Language features
    hedging_score: float = Field(0.0, ge=0.0, le=1.0)
    certainty_score: float = Field(0.0, ge=0.0, le=1.0)
    clickbait_score: float = Field(0.0, ge=0.0, le=1.0)
    language: str = "en"

    class Config:
        populate_by_name = True


class SourceMetadata(BaseModel):
    """Metadata about the article source."""

    source_id: str
    source_name: str
    credibility_score: float = Field(..., ge=0.0, le=1.0)
    source_category: SourceCategory = SourceCategory.UNKNOWN
    is_original_reporting: bool = True
    author_name: Optional[str] = None
    author_is_known_analyst: bool = False

    class Config:
        populate_by_name = True


class InferredImpact(BaseModel):
    """An asset not mentioned but inferred to be affected via dependency graph."""

    ticker: str
    asset_class: AssetClass
    relationship: str
    confidence: float = Field(..., ge=0.0, le=1.0)

    class Config:
        populate_by_name = True


# ============================================================================
# TOP-LEVEL SCHEMA — ArticleIntelligence
# ============================================================================


class ArticleIntelligence(BaseModel):
    """
    Complete per-article intelligence output from a miner.

    28 feature groups covering: classification, sentiment, multi-asset extraction,
    entity extraction, economic data, contagion chains, chart summaries, event
    fingerprinting, narrative tagging, embeddings, text statistics, and more.
    """

    # === Metadata ===
    schema_version: str = SCHEMA_VERSION
    article_id: int
    url: str
    title: str
    published_at: str
    analyzed_at: str
    miner_hotkey: Optional[str] = None
    analysis_model: Optional[str] = None
    analysis_latency_ms: Optional[int] = Field(None, ge=0)

    # === Source ===
    source: SourceMetadata

    # === Article-Level Classification ===
    content_type: ArticleContentType
    market_analysis_type: MarketAnalysisType = MarketAnalysisType.NONE
    impact_potential: ImpactPotential
    technical_quality: TechnicalQuality
    urgency: Urgency = Urgency.SAME_DAY
    temporal_focus: TemporalFocus = TemporalFocus.CURRENT

    # === Overall Sentiment ===
    overall_sentiment: Sentiment
    overall_sentiment_score: float = Field(..., ge=-1.0, le=1.0)
    sentiment_direction: SentimentDirection

    # === Per-Asset Sentiment ===
    assets: List[AssetSentiment] = Field(default_factory=list, max_length=MAX_ASSETS)

    # === Named Entities ===
    entities: List[ExtractedEntity] = Field(default_factory=list, max_length=MAX_ENTITIES)

    # === Economic Data Points ===
    economic_data: List[EconomicDataPoint] = Field(default_factory=list, max_length=MAX_ECONOMIC_DATA)

    # === Numeric Claims ===
    numeric_claims: List[NumericClaim] = Field(default_factory=list, max_length=MAX_NUMERIC_CLAIMS)

    # === Quotes ===
    quotes: List[QuoteExtraction] = Field(default_factory=list, max_length=MAX_QUOTES)

    # === Cross-Market Contagion ===
    contagion_links: List[ContagionLink] = Field(default_factory=list, max_length=MAX_CONTAGION_LINKS)

    # === Chart Summary ===
    chart_summary: ChartSummary

    # === Event Fingerprint ===
    event_fingerprint: EventFingerprint

    # === Narrative Keywords ===
    narrative_keywords: List[str] = Field(default_factory=list, max_length=3)

    # === Topic Signature ===
    topic_signature: TopicSignature

    # === Embeddings ===
    title_embedding: Optional[List[float]] = None
    body_embedding: Optional[List[float]] = None
    narrative_embedding: Optional[List[float]] = None

    # === Text Statistics ===
    text_stats: TextStatistics

    # === Factual Confidence ===
    factual_confidence: FactualConfidence
    source_attribution_type: SourceAttributionType = SourceAttributionType.NONE

    # === Positioning Signals ===
    positioning_signal: PositioningSignalType = PositioningSignalType.NONE
    positioning_actor: Optional[str] = None
    positioning_magnitude: Optional[str] = None

    # === Target Audience ===
    target_audience: TargetAudience = TargetAudience.GENERAL

    # === Credibility / Manipulation ===
    credibility_flag: CredibilityFlag = CredibilityFlag.UNVERIFIED
    is_sponsored: bool = False
    manipulation_signals: Optional[List[str]] = None

    # === Disambiguation ===
    disambiguation_method: DisambiguationMethod = DisambiguationMethod.NONE
    disambiguation_confidence: float = Field(1.0, ge=0.0, le=1.0)
    ambiguous_entities: Optional[List[str]] = None

    # === Staleness ===
    event_timestamp: Optional[str] = None
    staleness_flag: StalenessFlag = StalenessFlag.UNKNOWN
    publication_event_gap_hours: Optional[float] = None

    # === Forward Calendar References ===
    forward_event_type: ForwardEventType = ForwardEventType.NONE
    forward_event_date_approximate: Optional[str] = None
    forward_event_description: Optional[str] = None

    # === Article Status ===
    article_status: ArticleStatus = ArticleStatus.LIVE

    # === Geographic Impact ===
    primary_geo: GeoImpactZone = GeoImpactZone.GLOBAL
    secondary_geos: List[GeoImpactZone] = Field(default_factory=list)

    # === Market Session ===
    market_session: MarketSession = MarketSession.REGULAR_HOURS

    # === Language ===
    detected_language: str = "en"

    # === MNPI / Compliance ===
    mnpi_risk_flag: MNPIRiskFlag = MNPIRiskFlag.NONE
    mnpi_detection_signals: Optional[List[str]] = None

    # === Inferred Impacts ===
    inferred_impacts: Optional[List[InferredImpact]] = None

    class Config:
        populate_by_name = True

    def to_canonical_string(self) -> str:
        """Tier 1 exact-match canonical string for validator verification."""
        parts = [
            str(self.topic_signature.primary_sector_id),
            self.content_type.value,
            self.overall_sentiment.value,
            self.technical_quality.value,
            self.market_analysis_type.value,
            self.impact_potential.value,
            self.factual_confidence.value,
            self.positioning_signal.value,
            self.urgency.value,
            self.temporal_focus.value,
            self.sentiment_direction.value,
            self.primary_geo.value,
            self.target_audience.value,
            self.forward_event_type.value,
            self.staleness_flag.value,
            self.credibility_flag.value,
            self.detected_language,
            self.market_session.value,
            self.event_fingerprint.event_type.value,
            self.event_fingerprint.event_date or "none",
        ]
        # Per-asset sentiment for primary assets (sorted by ticker for determinism)
        primary_assets = sorted(
            [a for a in self.assets if a.is_primary_subject],
            key=lambda a: a.ticker,
        )
        for asset in primary_assets:
            parts.extend([
                asset.ticker,
                asset.direction.value,
                asset.short_term_outlook.value,
                asset.medium_term_outlook.value,
                asset.long_term_outlook.value,
            ])
        return "|".join(parts)

    @staticmethod
    def compute_content_hash(title: str, content: str) -> str:
        """Deterministic content hash for exact dedup of syndicated articles."""
        normalized = re.sub(r"\s+", " ", (title + " " + content[:500]).lower().strip())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
