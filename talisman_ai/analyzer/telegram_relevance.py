"""
Telegram Message Group Classification for Crypto Asset Relevance

Analyzes groups of Telegram messages to determine asset relevance.
Uses keyword-based matching (cashtags, asset names, aliases).

Key Features:
- Group analysis: Process multiple messages as a conversation
- Keyword-based asset detection (cashtags, names, identifiers)
- Evidence aggregation across message groups
"""

import os
import json
import re
import logging
import threading
from openai import OpenAI
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from dotenv import load_dotenv

from talisman_ai.analyzer.classifications import ContentType, Sentiment, TechnicalQuality, MarketAnalysis, ImpactPotential
from talisman_ai.analyzer.llm_cache import LLMCache

# Load environment variables
load_dotenv()

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class TelegramMessage:
    """Represents a single Telegram message"""
    message_id: str
    username: str
    content: str
    timestamp: Optional[datetime] = None
    reply_to: Optional[str] = None  # message_id of parent message


@dataclass
class MessageGroupClassification:
    """Classification result for a group of Telegram messages"""
    asset_id: int
    asset_symbol: str
    content_type: ContentType
    sentiment: Sentiment
    technical_quality: TechnicalQuality
    market_analysis: MarketAnalysis
    impact_potential: ImpactPotential
    relevance_confidence: str  # "high", "medium", "low"
    evidence_spans: List[str]  # Exact substrings that triggered the decision
    message_count: int  # Number of messages in the group
    contributing_messages: List[str] = field(default_factory=list)
    
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
            "message_count": self.message_count,
            "contributing_messages": self.contributing_messages,
        }


# Atomic tool definitions - one per classification dimension
ASSET_ID_TOOL = {
    "type": "function",
    "function": {
        "name": "identify_asset",
        "description": "Identify which crypto asset this message group is about",
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


class TelegramRelevanceAnalyzer:
    """
    Telegram message group classifier using atomic tool calls.
    
    Analyzes groups of messages to identify crypto asset relevance and
    classify content type, sentiment, and other dimensions.
    """
    
    def __init__(self, model: str = None, api_key: str = None, llm_base: str = None, assets: List[Dict] = None):
        """Initialize analyzer with asset registry and LLM config"""
        self.asset_registry = {}
        
        self.model = model or os.getenv("MODEL", "gpt-4o-mini")
        self.api_key = api_key or os.getenv("API_KEY")
        self.llm_base = llm_base or os.getenv("LLM_BASE", "https://api.openai.com/v1")
        
        if not self.api_key:
            raise ValueError("API_KEY environment variable is required")
        
        self.client = OpenAI(base_url=self.llm_base, api_key=self.api_key)

        cache_ttl = float(os.getenv("LLM_CACHE_TTL", "300"))
        cache_size = int(os.getenv("LLM_CACHE_MAX_SIZE", "1024"))
        self._cache = LLMCache(max_size=cache_size, ttl_seconds=cache_ttl)
        self._tl = threading.local()
        logger.info(f"[TELEGRAM_ANALYZER] LLM cache enabled: max_size={cache_size}, ttl={cache_ttl}s")

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
        
        logger.info(f"[TELEGRAM_ANALYZER] Initialized with model: {self.model}")
        if assets:
            logger.info(f"[TELEGRAM_ANALYZER] Registered {len(self.assets)-1} assets (+1 NONE)")
    
    def register_asset(self, asset_data: dict):
        """Register a crypto asset"""
        asset_id = asset_data['id']
        self.asset_registry[asset_id] = asset_data
        self.assets[asset_id] = asset_data
        logger.debug(f"[TELEGRAM_ANALYZER] Registered asset {asset_id}: {asset_data.get('symbol')}")
    
    def identify_asset_from_text(self, text: str) -> Dict:
        """
        Identify crypto asset from text using keyword matching.
        
        Priority:
        1. Cashtag match ($BTC, $ETH, etc.)
        2. Word-boundary match on unique_identifiers
        
        Args:
            text: Combined message text
            
        Returns:
            Dict with id, symbol, confidence, evidence
        """
        text_lower = text.lower()
        
        matches = []
        for aid, data in self.assets.items():
            if aid == 0:
                continue
            
            symbol = data.get('symbol', '')
            evidence = []
            
            for tag in data.get('cashtags', []):
                if tag.lower() in text_lower:
                    evidence.append(tag)
            
            if evidence:
                matches.append((aid, symbol, 1.0, evidence))
                continue
            
            for identifier in data.get('case_sensitive_identifiers', []):
                if len(identifier) < 3:
                    continue
                if re.search(rf'\b{re.escape(identifier)}\b', text):
                    evidence.append(identifier)
            
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
        
        if matches:
            top = matches[0]
            confidence = 'high' if top[2] >= 0.9 else 'medium'
            return {
                'id': top[0],
                'symbol': top[1],
                'confidence': confidence,
                'evidence': top[3],
            }
        
        return {'id': 0, 'symbol': "NONE", 'confidence': "low", 'evidence': []}
    
    def _normalize_messages(self, messages: List) -> List[TelegramMessage]:
        """Convert dict messages to TelegramMessage objects if needed"""
        normalized = []
        for msg in messages:
            if isinstance(msg, TelegramMessage):
                normalized.append(msg)
            elif isinstance(msg, dict):
                normalized.append(TelegramMessage(
                    message_id=str(msg.get('message_id', msg.get('id', ''))),
                    username=msg.get('username', msg.get('from', 'unknown')),
                    content=msg.get('content', msg.get('text', '')),
                    timestamp=msg.get('timestamp'),
                    reply_to=msg.get('reply_to')
                ))
            else:
                raise ValueError(f"Message must be TelegramMessage or dict, got {type(msg)}")
        return normalized
    
    def _combine_messages(self, messages: List[TelegramMessage]) -> str:
        """Combine messages into a single text block for analysis"""
        lines = []
        for msg in messages:
            lines.append(f"{msg.username}: {msg.content}")
        return "\n".join(lines)
    
    def _combine_messages_simple(self, messages: List[TelegramMessage]) -> str:
        """Combine message contents only (no usernames)"""
        return " ".join([msg.content for msg in messages])
    
    def classify_message_group(self, messages: List, asset_id: Optional[int] = None) -> Optional[MessageGroupClassification]:
        """
        Classify a group of Telegram messages.
        
        Args:
            messages: List of TelegramMessage objects or dicts to analyze as a group.
            asset_id: Optional asset ID. If provided, uses this asset instead of detecting from text.
            
        Returns:
            MessageGroupClassification if successful, None if parsing fails
        """
        if not messages:
            return None
        
        try:
            normalized_messages = self._normalize_messages(messages)
            combined_text = self._combine_messages(normalized_messages)
            simple_text = self._combine_messages_simple(normalized_messages)

            cache_key = f"{simple_text}||asset={asset_id}"
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

            self._tl.had_llm_error = False

            if asset_id is not None:
                if asset_id in self.assets:
                    asset_symbol = self.assets[asset_id].get('symbol', 'NONE')
                else:
                    asset_symbol = f'ASSET_{asset_id}'
                
                asset_result = {
                    'id': asset_id,
                    'symbol': asset_symbol,
                    'confidence': 'high',
                    'evidence': [asset_symbol],
                }
            else:
                asset_result = self.identify_asset_from_text(simple_text)
            
            content_type = self._classify_content_type(combined_text)
            sentiment = self._classify_sentiment(combined_text)
            technical_quality = self._assess_technical_quality(combined_text)
            market_analysis = self._classify_market_analysis(combined_text)
            impact = self._assess_impact(combined_text)
            
            contributing = []
            for msg in normalized_messages:
                msg_result = self.identify_asset_from_text(msg.content)
                if msg_result['id'] != 0:
                    contributing.append(msg.message_id)
            
            result = MessageGroupClassification(
                asset_id=asset_result['id'],
                asset_symbol=asset_result['symbol'],
                content_type=ContentType(content_type),
                sentiment=Sentiment(sentiment),
                technical_quality=TechnicalQuality(technical_quality),
                market_analysis=MarketAnalysis(market_analysis),
                impact_potential=ImpactPotential(impact),
                relevance_confidence=asset_result['confidence'],
                evidence_spans=asset_result['evidence'],
                message_count=len(normalized_messages),
                contributing_messages=contributing
            )
            if not getattr(self._tl, 'had_llm_error', False):
                self._cache.put(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"[TELEGRAM_ANALYZER] Classification error: {e}")
            return None
    
    def classify_messages_from_dicts(self, messages: List[Dict], asset_id: Optional[int] = None) -> Optional[MessageGroupClassification]:
        """
        Classify messages from dictionary format.
        
        Args:
            messages: List of dicts with keys: message_id, username, content, timestamp (optional), reply_to (optional)
            asset_id: Optional asset ID. If provided, uses this asset instead of detecting from text.
            
        Returns:
            MessageGroupClassification if successful, None if parsing fails
        """
        telegram_messages = []
        for msg in messages:
            telegram_messages.append(TelegramMessage(
                message_id=str(msg.get('message_id', msg.get('id', ''))),
                username=msg.get('username', msg.get('from', 'unknown')),
                content=msg.get('content', msg.get('text', '')),
                timestamp=msg.get('timestamp'),
                reply_to=msg.get('reply_to')
            ))
        return self.classify_message_group(telegram_messages, asset_id=asset_id)
    
    def classify_last_message_sentiment(self, messages: List) -> Optional[Sentiment]:
        """
        Classify the sentiment of the last message in context of the conversation.
        
        Args:
            messages: List of TelegramMessage objects or dicts in the conversation
            
        Returns:
            Sentiment if successful, None if parsing fails
        """
        if not messages:
            return None
        
        # Normalize messages (convert dicts to TelegramMessage if needed)
        normalized_messages = self._normalize_messages(messages)
        combined_text = self._combine_messages(normalized_messages)
        prompt = f"""
        Given the following Telegram conversation:
        <conversation>
        {combined_text}
        </conversation>
        Classify the sentiment of the last message using the following categories:
        <sentiment_categories>
        - very_bullish: rocket emoji, moon, ATH, pump, explosive growth, massive gains
        - bullish: positive outlook, optimistic, growth potential, upward trend
        - neutral: factual reporting, balanced, no strong opinion, informational
        - bearish: concerns raised, negative outlook, downward trend, issues mentioned
        - very_bearish: crash, failure, exploit, major problem, severe concerns
        </sentiment_categories>
        """
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
            return Sentiment(args.get("sentiment", "neutral"))
        except Exception as e:
            logger.warning(f"[TELEGRAM_ANALYZER] classify_last_message_sentiment failed: {e}")
            self._tl.had_llm_error = True
            return None
    
    def _classify_content_type(self, text: str) -> str:
        """Atomic decision: Content type"""
        prompt = f"""Classify content type of this Telegram conversation: "{text}"

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
            logger.warning(f"[TELEGRAM_ANALYZER] _classify_content_type failed: {e}")
            self._tl.had_llm_error = True
            return "other"
    
    def _classify_sentiment(self, text: str) -> str:
        """Atomic decision: Sentiment"""
        prompt = f"""Classify sentiment of this Telegram conversation: "{text}"

Choose the sentiment that best matches the overall tone:
- very_bullish: rocket emoji, moon, ATH, pump, explosive growth, massive gains
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
            logger.warning(f"[TELEGRAM_ANALYZER] _classify_sentiment failed: {e}")
            self._tl.had_llm_error = True
            return "neutral"
    
    def _assess_technical_quality(self, text: str) -> str:
        """Atomic decision: Technical quality"""
        prompt = f"""Assess technical quality of this Telegram conversation: "{text}"

- high: 2+ specifics (APIs, versions, metrics)
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
            logger.warning(f"[TELEGRAM_ANALYZER] _assess_technical_quality failed: {e}")
            self._tl.had_llm_error = True
            return "none"
    
    def _classify_market_analysis(self, text: str) -> str:
        """Atomic decision: Market analysis type"""
        prompt = f"""Classify market analysis type in this Telegram conversation: "{text}"

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
            logger.warning(f"[TELEGRAM_ANALYZER] _classify_market_analysis failed: {e}")
            self._tl.had_llm_error = True
            return "other"
    
    def _assess_impact(self, text: str) -> str:
        """Atomic decision: Impact potential"""
        prompt = f"""Assess impact potential of this Telegram conversation: "{text}"

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
            logger.warning(f"[TELEGRAM_ANALYZER] _assess_impact failed: {e}")
            self._tl.had_llm_error = True
            return "NONE"
    
    def analyze_message_group_complete(self, messages: List, asset_id: Optional[int] = None) -> dict:
        """
        Analyze message group and return rich classification data.
        
        Args:
            messages: List of TelegramMessage objects or dicts.
            asset_id: Optional asset ID. If provided, uses this asset instead of detecting from text.
            
        Returns:
            Dict with classification, asset_relevance, sentiment info
        """
        import time
        start_time = time.time()
        logger.info(f"[TELEGRAM_ANALYZER] Starting analysis for {len(messages)} messages" + (f" (asset_id={asset_id})" if asset_id else ""))
        
        classification = self.classify_message_group(messages, asset_id=asset_id)
        
        if classification is None:
            logger.warning(f"[TELEGRAM_ANALYZER] Classification failed")
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
                "message_count": classification.message_count,
                "contributing_messages": classification.contributing_messages,
            }
        
        total_time = time.time() - start_time
        logger.info(f"[TELEGRAM_ANALYZER] Analysis completed in {total_time:.2f}s")
        
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
            "message_count": len(messages),
            "timestamp": datetime.now().isoformat()
        }
    
    def analyze_messages_from_dicts_complete(self, messages: List[Dict], asset_id: Optional[int] = None) -> dict:
        """
        Convenience method to analyze messages from dictionary format.
        
        Args:
            messages: List of message dicts
            asset_id: Optional asset ID. If provided, uses this asset instead of detecting from text.
            
        Returns:
            Dict with classification results
        """
        telegram_messages = []
        for msg in messages:
            telegram_messages.append(TelegramMessage(
                message_id=str(msg.get('message_id', msg.get('id', ''))),
                username=msg.get('username', msg.get('from', 'unknown')),
                content=msg.get('content', msg.get('text', '')),
                timestamp=msg.get('timestamp'),
                reply_to=msg.get('reply_to')
            ))
        return self.analyze_message_group_complete(telegram_messages, asset_id=asset_id)

