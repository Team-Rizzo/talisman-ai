"""
Deterministic X Post Classification for Crypto Asset Relevance

Uses atomic tool calls for each classification dimension to achieve deterministic
LLM evaluation. Validators can verify miner classifications via exact matching
of canonical strings.

Key Features:
- Atomic decisions: One tool call per classification dimension
- Keyword-based asset identification (cashtags, names, aliases)
- Explicit abstain logic (asset_id=0 for ties/unknown)
- Evidence extraction (exact spans for auditability)
"""

from openai import OpenAI
import json
import re
import threading
from typing import Dict, List, Optional
from dataclasses import dataclass
import time
from datetime import datetime
import bittensor as bt
from .classifications import ContentType, Sentiment, TechnicalQuality, MarketAnalysis, ImpactPotential
from .llm_cache import LLMCache

# Import centralized config
try:
    from talisman_ai import config
except ImportError:
    config = None


@dataclass
class PostClassification:
    """Canonical classification result"""
    asset_id: int
    asset_symbol: str
    content_type: ContentType
    sentiment: Sentiment
    technical_quality: TechnicalQuality
    market_analysis: MarketAnalysis
    impact_potential: ImpactPotential
    relevance_confidence: str  # "high", "medium", "low"
    evidence_spans: List[str]  # Exact substrings that triggered the decision
    
    def to_canonical_string(self) -> str:
        """Deterministic string for exact matching by validators"""
        sorted_evidence = "|".join(sorted([s.lower() for s in self.evidence_spans]))
        return f"{self.asset_id}|{self.content_type.value}|{self.sentiment.value}|{self.technical_quality.value}|{self.market_analysis.value}|{self.impact_potential.value}|{self.relevance_confidence}|{sorted_evidence}"
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization or database storage"""
        return {
            "asset_id": self.asset_id,
            "asset_symbol": self.asset_symbol,
            "content_type": self.content_type.value,
            "sentiment": self.sentiment.value,
            "technical_quality": self.technical_quality.value,
            "market_analysis": self.market_analysis.value,
            "impact_potential": self.impact_potential.value,
            "relevance_confidence": self.relevance_confidence,
            "evidence_spans": self.evidence_spans,
        }


# Atomic tool definitions - one per classification dimension
ASSET_ID_TOOL = {
    "type": "function",
    "function": {
        "name": "identify_asset",
        "description": "Identify which crypto asset this post is about",
        "parameters": {
            "type": "object",
            "properties": {
                "asset_id": {"type": "integer", "description": "Asset ID (0 if none/unclear)"},
                "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                "evidence_spans": {"type": "array", "items": {"type": "string"}, "description": "Exact text spans that identify this asset"},
            },
            "required": ["asset_id", "confidence", "evidence_spans"]
        }
    }
}

CONTENT_TYPE_TOOL = {
    "type": "function",
    "function": {
        "name": "classify_content_type",
        "description": "Classify the type of content",
        "parameters": {
            "type": "object",
            "properties": {
                "content_type": {
                    "type": "string",
                    "enum": [ct.value for ct in ContentType],
                    "description": "Primary content type"
                }
            },
            "required": ["content_type"]
        }
    }
}

SENTIMENT_TOOL = {
    "type": "function",
    "function": {
        "name": "classify_sentiment",
        "description": "Classify market sentiment",
        "parameters": {
            "type": "object",
            "properties": {
                "sentiment": {
                    "type": "string",
                    "enum": [s.value for s in Sentiment],
                    "description": "Market sentiment"
                }
            },
            "required": ["sentiment"]
        }
    }
}

TECHNICAL_QUALITY_TOOL = {
    "type": "function",
    "function": {
        "name": "assess_technical_quality",
        "description": "Assess technical content quality",
        "parameters": {
            "type": "object",
            "properties": {
                "quality": {
                    "type": "string",
                    "enum": [tq.value for tq in TechnicalQuality],
                    "description": "Technical quality level"
                }
            },
            "required": ["quality"]
        }
    }
}

MARKET_ANALYSIS_TOOL = {
    "type": "function",
    "function": {
        "name": "classify_market_analysis",
        "description": "Classify market analysis type",
        "parameters": {
            "type": "object",
            "properties": {
                "analysis_type": {
                    "type": "string",
                    "enum": [ma.value for ma in MarketAnalysis],
                    "description": "Type of market analysis"
                }
            },
            "required": ["analysis_type"]
        }
    }
}

IMPACT_TOOL = {
    "type": "function",
    "function": {
        "name": "assess_impact",
        "description": "Assess potential impact",
        "parameters": {
            "type": "object",
            "properties": {
                "impact": {
                    "type": "string",
                    "enum": [ip.value for ip in ImpactPotential],
                    "description": "Expected impact level"
                }
            },
            "required": ["impact"]
        }
    }
}


class AssetRelevanceAnalyzer:
    """
    Deterministic X post classifier using atomic tool calls.
    
    Each classification dimension is decided independently via its own tool call,
    eliminating compound decision variance.
    """
    
    def __init__(self, model: str = None, api_key: str = None, llm_base: str = None, assets: List[Dict] = None):
        """Initialize analyzer with asset registry and LLM config"""
        self.asset_registry = {}
        
        if config:
            self.model = model or config.MODEL
            self.api_key = api_key or config.API_KEY
            self.llm_base = llm_base or config.LLM_BASE
        else:
            self.model = model
            self.api_key = api_key
            self.llm_base = llm_base
        
        if not self.api_key:
            raise ValueError("API_KEY environment variable is required")
        
        self.client = OpenAI(base_url=self.llm_base, api_key=self.api_key)

        cache_ttl = float(getattr(config, "LLM_CACHE_TTL", 300)) if config else 300.0
        cache_size = int(getattr(config, "LLM_CACHE_MAX_SIZE", 1024)) if config else 1024
        self._cache = LLMCache(max_size=cache_size, ttl_seconds=cache_ttl)
        self._tl = threading.local()
        bt.logging.info(f"[ANALYZER] LLM cache enabled: max_size={cache_size}, ttl={cache_ttl}s")

        if assets:
            self.assets = {a["id"]: a for a in assets}
            for a in assets:
                self.asset_registry[a["id"]] = a
        else:
            self.assets = {}
        
        self.assets[0] = {
            "id": 0,
            "symbol": "NONE",
            "name": "NONE_OF_THE_ABOVE",
            "description": "Content not specific to a listed crypto asset"
        }
        
        bt.logging.info(f"[ANALYZER] Initialized with model: {self.model}")
        if assets:
            bt.logging.info(f"[ANALYZER] Registered {len(self.assets)-1} assets (+1 NONE)")
    
    def register_asset(self, asset_data: dict):
        """Register a crypto asset"""
        asset_id = asset_data['id']
        self.asset_registry[asset_id] = asset_data
        self.assets[asset_id] = asset_data
        bt.logging.debug(f"[ANALYZER] Registered asset {asset_id}: {asset_data.get('symbol')}")
    
    def classify_keyword_based(self, text: str) -> Dict:
        """
        Keyword-based asset classification using cashtags and identifier matching.
        
        Matching priority:
        1. Cashtag match ($BTC, $ETH, etc.) — highest confidence
        2. Exact word-boundary match on unique_identifiers
        3. No match → asset_id 0
        
        Args:
            text: Post text to classify
            
        Returns:
            Dict with:
            - is_relevant: Whether post matches a tracked crypto asset
            - confidence: 'high' or 'medium'
            - asset_scores: {asset_id: score}
            - matched_assets: [(asset_id, symbol, score, evidence), ...] sorted by score
        """
        text_lower = text.lower()
        
        matches = []
        for aid, data in self.assets.items():
            if aid == 0:
                continue
            
            symbol = data.get('symbol', '')
            evidence = []
            
            # Cashtag match (highest priority) — e.g. $BTC, $ETH
            for tag in data.get('cashtags', []):
                if tag.lower() in text_lower:
                    evidence.append(tag)
            
            if evidence:
                matches.append((aid, symbol, 1.0, evidence))
                continue
            
            # Case-sensitive identifiers (e.g., "SOL" must be uppercase)
            for identifier in data.get('case_sensitive_identifiers', []):
                if len(identifier) < 3:
                    continue
                if re.search(rf'\b{re.escape(identifier)}\b', text):
                    evidence.append(identifier)
            
            # Case-insensitive word-boundary match on unique_identifiers
            for identifier in data.get('unique_identifiers', []):
                id_lower = identifier.lower()
                if len(id_lower) < 3:
                    continue
                if re.search(rf'\b{re.escape(id_lower)}\b', text_lower):
                    evidence.append(identifier)
            
            if evidence:
                confidence_score = 0.9 if len(evidence) > 1 else 0.8
                matches.append((aid, symbol, confidence_score, evidence))
        
        matches.sort(key=lambda x: x[2], reverse=True)
        
        if not matches:
            return {'is_relevant': False, 'confidence': None, 'asset_scores': {}, 'matched_assets': []}
        
        return {
            'is_relevant': True,
            'confidence': 'high' if matches[0][2] >= 0.9 else 'medium',
            'asset_scores': {m[0]: m[2] for m in matches},
            'matched_assets': matches
        }
    
    def _build_asset_context(self) -> str:
        """Build context string for asset identification (used in LLM prompts if needed)"""
        contexts = []
        for aid in sorted(self.assets.keys()):
            if aid == 0:
                continue
            a = self.assets[aid]
            ctx = f"{a.get('symbol', '?')} (id={aid}, {a.get('name', 'Unknown')}): {a.get('description', '')}"
            ids = a.get('unique_identifiers', [])
            if ids:
                ctx += f" | IDs: {', '.join(ids)}"
            contexts.append(ctx)
        return '\n'.join(contexts)
    
    def classify_post(self, text: str) -> Optional[PostClassification]:
        """
        Classify using atomic tool calls for each dimension.
        Results are cached by text content (LLM calls use temperature=0).
        
        Args:
            text: X post text to classify
            
        Returns:
            PostClassification if successful, None if parsing fails
        """
        cached = self._cache.get(text)
        if cached is not None:
            return cached

        try:
            self._tl.had_llm_error = False

            asset_result = self._identify_asset(text)
            
            content_type = self._classify_content_type(text)
            sentiment = self._classify_sentiment(text)
            technical_quality = self._assess_technical_quality(text)
            market_analysis = self._classify_market_analysis(text)
            impact = self._assess_impact(text)
            
            result = PostClassification(
                asset_id=asset_result['id'],
                asset_symbol=asset_result['symbol'],
                content_type=ContentType(content_type),
                sentiment=Sentiment(sentiment),
                technical_quality=TechnicalQuality(technical_quality),
                market_analysis=MarketAnalysis(market_analysis),
                impact_potential=ImpactPotential(impact),
                relevance_confidence=asset_result['confidence'],
                evidence_spans=asset_result['evidence'],
            )
            if not getattr(self._tl, 'had_llm_error', False):
                self._cache.put(text, result)
            return result
            
        except Exception as e:
            bt.logging.error(f"[ANALYZER] Classification error: {e}")
            return None
    
    def _identify_asset(self, text: str) -> dict:
        """Identify crypto asset using keyword-based matching (no LLM)."""
        result = self.classify_keyword_based(text)
        
        if not result['is_relevant']:
            return {'id': 0, 'symbol': "NONE", 'confidence': "low", 'evidence': []}
        
        if result['matched_assets']:
            top = result['matched_assets'][0]  # (aid, symbol, score, evidence)
            confidence = 'high' if top[2] >= 0.9 else 'medium' if top[2] >= 0.8 else 'low'
            return {
                'id': top[0],
                'symbol': top[1],
                'confidence': confidence,
                'evidence': top[3],
            }
        
        return {'id': 0, 'symbol': "NONE", 'confidence': "low", 'evidence': []}
    
    def _classify_content_type(self, text: str) -> str:
        """Atomic decision: Content type"""
        prompt = f"""Classify content type of: "{text}"

Pick the MOST SPECIFIC category that applies:
- announcement: product launches, releases, updates
- partnership: collaborations, integrations, joint ventures
- technical_insight: technical analysis, architecture, code discussions
- milestone: achievements, metrics, progress updates
- tutorial: how-to guides, educational content
- security: audits, vulnerabilities, exploits, security updates
- governance: voting, proposals, DAO decisions
- market_discussion: price talk, trading, speculation
- hiring: job postings, recruitment
- meme: jokes, entertainment, humor
- hype: excitement, enthusiasm, promotional content
- opinion: personal views, analysis, commentary
- community: general chatter, engagement, discussions
- fud: fear, uncertainty, doubt, negative speculation
- other: doesn't fit any category above"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                tools=[CONTENT_TYPE_TOOL],
                tool_choice={"type": "function", "function": {"name": "classify_content_type"}},
                temperature=0,
                max_tokens=50
            )
            args = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
            return args.get("content_type", "other")
        except Exception as e:
            bt.logging.warning(f"[ANALYZER] _classify_content_type failed: {e}")
            self._tl.had_llm_error = True
            return "other"
    
    def _classify_sentiment(self, text: str) -> str:
        """Atomic decision: Sentiment"""
        prompt = f"""Classify sentiment of: "{text}"

Choose the sentiment that best matches the tone:
- very_bullish: 🚀, moon, ATH, pump, explosive growth, massive gains
- bullish: positive outlook, optimistic, growth potential, upward trend
- neutral: factual reporting, balanced, no strong opinion, informational
- bearish: concerns raised, negative outlook, downward trend, issues mentioned
- very_bearish: crash, failure, exploit, major problem, severe concerns"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                tools=[SENTIMENT_TOOL],
                tool_choice={"type": "function", "function": {"name": "classify_sentiment"}},
                temperature=0,
                max_tokens=50
            )
            args = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
            return args.get("sentiment", "neutral")
        except Exception as e:
            bt.logging.warning(f"[ANALYZER] _classify_sentiment failed: {e}")
            self._tl.had_llm_error = True
            return "neutral"
    
    def _assess_technical_quality(self, text: str) -> str:
        """Atomic decision: Technical quality"""
        prompt = f"""Assess technical quality of: "{text}"

- high: ≥2 specifics (APIs, versions, metrics)
- medium: 1 specific detail
- low: claims without specifics
- none: no technical content"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                tools=[TECHNICAL_QUALITY_TOOL],
                tool_choice={"type": "function", "function": {"name": "assess_technical_quality"}},
                temperature=0,
                max_tokens=50
            )
            args = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
            return args.get("quality", "none")
        except Exception as e:
            bt.logging.warning(f"[ANALYZER] _assess_technical_quality failed: {e}")
            self._tl.had_llm_error = True
            return "none"
    
    def _classify_market_analysis(self, text: str) -> str:
        """Atomic decision: Market analysis type"""
        prompt = f"""Classify market analysis type in: "{text}"

- technical: indicators, patterns
- economic: fundamentals, costs
- political: regulatory, governance
- social: narrative, virality
- other: none or different"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                tools=[MARKET_ANALYSIS_TOOL],
                tool_choice={"type": "function", "function": {"name": "classify_market_analysis"}},
                temperature=0,
                max_tokens=50
            )
            args = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
            return args.get("analysis_type", "other")
        except Exception as e:
            bt.logging.warning(f"[ANALYZER] _classify_market_analysis failed: {e}")
            self._tl.had_llm_error = True
            return "other"
    
    def _assess_impact(self, text: str) -> str:
        """Atomic decision: Impact potential"""
        prompt = f"""Assess impact potential of: "{text}"

- HIGH: major releases, critical issues
- MEDIUM: notable updates, partnerships
- LOW: minor information
- NONE: chatter, no impact"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                tools=[IMPACT_TOOL],
                tool_choice={"type": "function", "function": {"name": "assess_impact"}},
                temperature=0,
                max_tokens=50
            )
            args = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
            return args.get("impact", "NONE")
        except Exception as e:
            bt.logging.warning(f"[ANALYZER] _assess_impact failed: {e}")
            self._tl.had_llm_error = True
            return "NONE"
    
    def analyze_post_complete(self, text: str) -> dict:
        """
        Analyze post and return rich classification data.
        """
        start_time = time.time()
        bt.logging.info(f"[ANALYZER] Starting analysis for post (length: {len(text)} chars)")
        
        classification = self.classify_post(text)
        
        if classification is None:
            bt.logging.warning(f"[ANALYZER] Classification failed")
            return {
                "classification": None,
                "asset_relevance": {},
                "timestamp": datetime.now().isoformat()
            }
        
        asset_relevance = {}
        if classification.asset_id != 0:
            symbol = classification.asset_symbol
            asset_relevance[symbol] = {
                "asset_id": classification.asset_id,
                "asset_symbol": symbol,
                "relevance": 1.0,
                "relevance_confidence": classification.relevance_confidence,
                "content_type": classification.content_type.value,
                "sentiment": classification.sentiment.value,
                "technical_quality": classification.technical_quality.value,
                "market_analysis": classification.market_analysis.value,
                "impact_potential": classification.impact_potential.value,
                "evidence_spans": classification.evidence_spans,
            }
        
        total_time = time.time() - start_time
        bt.logging.info(f"[ANALYZER] Analysis completed in {total_time:.2f}s")
        
        sentiment_to_float = {
            "very_bullish": 1.0,
            "bullish": 0.5,
            "neutral": 0.0,
            "bearish": -0.5,
            "very_bearish": -1.0
        }
        sentiment_enum = classification.sentiment.value if classification else "neutral"
        sentiment_float = sentiment_to_float.get(sentiment_enum, 0.0)
        
        return {
            "classification": classification,
            "asset_relevance": asset_relevance,
            "sentiment": sentiment_float,
            "sentiment_enum": sentiment_enum,
            "timestamp": datetime.now().isoformat()
        }
    
    def _parse_classification(self, args: dict) -> Optional[PostClassification]:
        """Parse and validate function call arguments"""
        try:
            asset_id = int(args["asset_id"])
            if asset_id not in self.assets:
                bt.logging.warning(f"[ANALYZER] Unknown asset_id: {asset_id}")
                return None
            
            return PostClassification(
                asset_id=asset_id,
                asset_symbol=self.assets[asset_id].get("symbol", "NONE"),
                content_type=ContentType(args["content_type"]),
                sentiment=Sentiment(args["sentiment"]),
                technical_quality=TechnicalQuality(args["technical_quality"]),
                market_analysis=MarketAnalysis(args["market_analysis"]),
                impact_potential=ImpactPotential(args["impact_potential"]),
                relevance_confidence=args["relevance_confidence"],
                evidence_spans=args.get("evidence_spans", []),
            )
        except (ValueError, KeyError) as e:
            bt.logging.error(f"[ANALYZER] Parse error: {e}")
            return None


# Backward-compatible alias for existing imports
SubnetRelevanceAnalyzer = AssetRelevanceAnalyzer
