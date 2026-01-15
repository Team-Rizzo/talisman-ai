"""
Telegram Message Group Classification for BitTensor Subnet Relevance

Analyzes groups of Telegram messages to determine subnet relevance.
Identifies subnets from patterns like: SN45, 45, SN 45, subnet 45

Key Features:
- Group analysis: Process multiple messages as a conversation
- Flexible subnet detection: Various pattern formats supported
- Evidence aggregation across message groups
"""

import os
import json
import re
import logging
from openai import OpenAI
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from dotenv import load_dotenv

from talisman_ai.analyzer.classifications import ContentType, Sentiment, TechnicalQuality, MarketAnalysis, ImpactPotential

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
    subnet_id: int
    subnet_name: str
    content_type: ContentType
    sentiment: Sentiment
    technical_quality: TechnicalQuality
    market_analysis: MarketAnalysis
    impact_potential: ImpactPotential
    relevance_confidence: str  # "high", "medium", "low"
    evidence_spans: List[str]  # Exact substrings that triggered the decision
    anchors_detected: List[str]  # BitTensor anchor words found
    message_count: int  # Number of messages in the group
    contributing_messages: List[str] = field(default_factory=list)  # message_ids that contributed to classification
    
    def to_canonical_string(self) -> str:
        """Deterministic string for exact matching by validators"""
        sorted_evidence = "|".join(sorted([s.lower() for s in self.evidence_spans]))
        sorted_anchors = "|".join(sorted([s.lower() for s in self.anchors_detected]))
        return f"{self.subnet_id}|{self.content_type.value}|{self.sentiment.value}|{self.technical_quality.value}|{self.market_analysis.value}|{self.impact_potential.value}|{self.relevance_confidence}|{sorted_evidence}|{sorted_anchors}"
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization or database storage"""
        return {
            "subnet_id": self.subnet_id,
            "subnet_name": self.subnet_name,
            "content_type": self.content_type.value,
            "sentiment": self.sentiment.value,
            "technical_quality": self.technical_quality.value,
            "market_analysis": self.market_analysis.value,
            "impact_potential": self.impact_potential.value,
            "relevance_confidence": self.relevance_confidence,
            "evidence_spans": self.evidence_spans,
            "anchors_detected": self.anchors_detected,
            "message_count": self.message_count,
            "contributing_messages": self.contributing_messages,
        }
    
    def get_tokens_dict(self) -> dict:
        """Get subnet tokens dict for grader compatibility"""
        if self.subnet_id == 0:
            return {}
        return {self.subnet_name: 1.0}


# Atomic tool definitions - one per classification dimension
SUBNET_ID_TOOL = {
    "type": "function",
    "function": {
        "name": "identify_subnet",
        "description": "Identify which subnet this message group is about",
        "parameters": {
            "type": "object",
            "properties": {
                "subnet_id": {"type": "integer", "description": "Subnet ID (0 if none/unclear)"},
                "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                "evidence_spans": {"type": "array", "items": {"type": "string"}, "description": "Exact text spans that identify this subnet"},
                "anchors_detected": {"type": "array", "items": {"type": "string"}, "description": "BitTensor anchor words found"}
            },
            "required": ["subnet_id", "confidence", "evidence_spans", "anchors_detected"]
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


# Subnet identification patterns
# Matches: SN45, SN 45, sn45, sn 45, subnet 45, subnet45, Subnet 45, 45 (standalone number)
SUBNET_PATTERNS = [
    (r'\bSN\s*(\d{1,3})\b', 'sn_format'),           # SN45, SN 45
    (r'\bsn\s*(\d{1,3})\b', 'sn_format'),           # sn45, sn 45
    (r'\bsubnet\s*(\d{1,3})\b', 'subnet_format'),   # subnet 45, subnet45
    (r'\bSubnet\s*(\d{1,3})\b', 'subnet_format'),   # Subnet 45
]


class TelegramRelevanceAnalyzer:
    """
    Telegram message group classifier using atomic tool calls
    
    Analyzes groups of messages to identify subnet relevance and
    classify content type, sentiment, and other dimensions.
    """
    
    def __init__(self, model: str = None, api_key: str = None, llm_base: str = None, subnets: List[Dict] = None):
        """Initialize analyzer with subnet registry and LLM config"""
        self.subnet_registry = {}
        
        # Load from environment variables
        self.model = model or os.getenv("MODEL", "gpt-4o-mini")
        self.api_key = api_key or os.getenv("API_KEY")
        self.llm_base = llm_base or os.getenv("LLM_BASE", "https://api.openai.com/v1")
        
        if not self.api_key:
            raise ValueError("API_KEY environment variable is required")
        
        self.client = OpenAI(base_url=self.llm_base, api_key=self.api_key)
        
        # Initialize subnets
        if subnets:
            self.subnets = {s["id"]: s for s in subnets}
            for s in subnets:
                self.subnet_registry[s["id"]] = s
        else:
            self.subnets = {}
        
        # Add NONE subnet
        self.subnets[0] = {
            "id": 0,
            "name": "NONE_OF_THE_ABOVE",
            "description": "General BitTensor content not specific to a listed subnet"
        }
        
        logger.info(f"[TELEGRAM_ANALYZER] Initialized with model: {self.model}")
        if subnets:
            logger.info(f"[TELEGRAM_ANALYZER] Registered {len(self.subnets)-1} subnets (+1 NONE)")
    
    def register_subnet(self, subnet_data: dict):
        """Register a subnet"""
        subnet_id = subnet_data['id']
        self.subnet_registry[subnet_id] = subnet_data
        self.subnets[subnet_id] = subnet_data
        logger.debug(f"[TELEGRAM_ANALYZER] Registered subnet {subnet_id}: {subnet_data.get('name')}")
    
    def extract_subnet_mentions(self, text: str) -> List[Tuple[int, str, str]]:
        """
        Extract subnet mentions from text using various patterns.
        
        Args:
            text: Text to search for subnet mentions
            
        Returns:
            List of (subnet_id, matched_text, pattern_type) tuples
        """
        mentions = []
        
        for pattern, pattern_type in SUBNET_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                subnet_id = int(match.group(1))
                matched_text = match.group(0)
                mentions.append((subnet_id, matched_text, pattern_type))
        
        return mentions
    
    def identify_subnet_from_text(self, text: str) -> Dict:
        """
        Identify subnet from text using pattern matching.
        
        Priority:
        1. Explicit SN pattern (SN45, SN 45)
        2. Subnet pattern (subnet 45)
        3. Subnet name/alias matching
        
        Args:
            text: Combined message text
            
        Returns:
            Dict with id, name, confidence, evidence, anchors
        """
        text_lower = text.lower()
        
        # Check for BitTensor anchors
        has_anchor = bool(re.search(r'\bsn\s*\d+\b|\bsubnet\b|\bbittensor\b|\btao\b|\$tao\b|\bopentensor\b', text_lower))
        
        # Extract explicit subnet mentions
        mentions = self.extract_subnet_mentions(text)
        
        if mentions:
            # Count mentions per subnet
            subnet_counts = {}
            for subnet_id, matched_text, pattern_type in mentions:
                if subnet_id not in subnet_counts:
                    subnet_counts[subnet_id] = {'count': 0, 'evidence': [], 'patterns': set()}
                subnet_counts[subnet_id]['count'] += 1
                subnet_counts[subnet_id]['evidence'].append(matched_text)
                subnet_counts[subnet_id]['patterns'].add(pattern_type)
            
            # Pick the most mentioned subnet
            top_subnet = max(subnet_counts.items(), key=lambda x: x[1]['count'])
            subnet_id = top_subnet[0]
            subnet_info = top_subnet[1]
            
            # Check if subnet is registered
            if subnet_id in self.subnets:
                subnet_name = self.subnets[subnet_id].get('name', f'Subnet {subnet_id}')
            else:
                subnet_name = f'Subnet {subnet_id}'
            
            # Determine confidence
            if subnet_info['count'] >= 3:
                confidence = 'high'
            elif subnet_info['count'] >= 2 or 'sn_format' in subnet_info['patterns']:
                confidence = 'high'
            else:
                confidence = 'medium'
            
            return {
                'id': subnet_id,
                'name': subnet_name,
                'confidence': confidence,
                'evidence': list(set(subnet_info['evidence'])),
                'anchors': ['bittensor'] if has_anchor else []
            }
        
        # Fall back to keyword-based matching if no explicit mentions
        return self._keyword_based_subnet_match(text, has_anchor)
    
    def _keyword_based_subnet_match(self, text: str, has_anchor: bool) -> Dict:
        """
        Fallback keyword-based matching using subnet names and identifiers.
        """
        text_lower = text.lower()
        
        # Extract words for matching (4+ chars, exclude ecosystem terms)
        ecosystem = re.compile(r'^(bittensor|opentensor|tao|subnets?|sn\d+)$')
        words = {w for w in re.findall(r'\b\w{4,}\b', text_lower) if not ecosystem.match(w)}
        
        # Find subnet matches
        matches = []
        for sid, data in self.subnets.items():
            if sid == 0:
                continue
            
            best = (0.0, [])
            for name in [data.get('name', '')] + data.get('unique_identifiers', []):
                if not name or len(name) < 4 or ecosystem.match(name.lower()):
                    continue
                name_lower = name.lower()
                
                # Exact match
                if name_lower in words:
                    best = (0.9, [name])
                    break
                
                # Fuzzy match (edit distance >= 0.85)
                for word in words:
                    sim = SequenceMatcher(None, word, name_lower).ratio()
                    if sim >= 0.85 and sim > best[0]:
                        best = (sim * 0.9, [f'{word}≈{name}'])
            
            if best[0] >= 0.75:
                matches.append((sid, data['name'], best[0], best[1]))
        
        # Sort by score descending
        matches.sort(key=lambda x: x[2], reverse=True)
        
        if not has_anchor and not matches:
            return {'id': 0, 'name': "NONE_OF_THE_ABOVE", 'confidence': "low", 'evidence': [], 'anchors': []}
        
        if matches:
            top = matches[0]
            confidence = 'high' if top[2] >= 0.9 else 'medium' if top[2] >= 0.8 else 'low'
            return {
                'id': top[0],
                'name': top[1],
                'confidence': confidence,
                'evidence': top[3],
                'anchors': []
            }
        
        return {'id': 0, 'name': "NONE_OF_THE_ABOVE", 'confidence': "low", 'evidence': [], 'anchors': []}
    
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
    
    def classify_message_group(self, messages: List, subnet_id: Optional[int] = None) -> Optional[MessageGroupClassification]:
        """
        Classify a group of Telegram messages.
        
        Args:
            messages: List of TelegramMessage objects or dicts to analyze as a group.
                     Dicts should have: message_id, username, content (or text), timestamp (optional), reply_to (optional)
            subnet_id: Optional subnet ID (e.g., 45, 30). If provided, uses this subnet instead of detecting from text.
            
        Returns:
            MessageGroupClassification if successful, None if parsing fails
        """
        if not messages:
            return None
        
        try:
            # Normalize messages (convert dicts to TelegramMessage if needed)
            normalized_messages = self._normalize_messages(messages)
            
            # Combine messages for analysis
            combined_text = self._combine_messages(normalized_messages)
            simple_text = self._combine_messages_simple(normalized_messages)
            
            # Step 1: Identify subnet - use provided subnet_id or detect from text
            if subnet_id is not None:
                # Use provided subnet ID
                if subnet_id in self.subnets:
                    subnet_name = self.subnets[subnet_id].get('name', f'Subnet {subnet_id}')
                else:
                    subnet_name = f'Subnet {subnet_id}'
                
                # Extract evidence spans from text
                mentions = self.extract_subnet_mentions(simple_text)
                evidence = [m[1] for m in mentions if m[0] == subnet_id]
                if not evidence:
                    # If no explicit mention found, use any subnet mentions as evidence
                    evidence = [m[1] for m in mentions] if mentions else []
                
                # Check for BitTensor anchors
                has_anchor = bool(re.search(r'\bsn\s*\d+\b|\bsubnet\b|\bbittensor\b|\btao\b|\$tao\b|\bopentensor\b', simple_text.lower()))
                
                subnet_result = {
                    'id': subnet_id,
                    'name': subnet_name,
                    'confidence': 'high' if evidence or has_anchor else 'medium',
                    'evidence': evidence if evidence else [f'SN{subnet_id}'],
                    'anchors': ['bittensor'] if has_anchor else []
                }
            else:
                # Detect subnet from text
                subnet_result = self.identify_subnet_from_text(simple_text)
            
            # Step 2-6: Classify other dimensions atomically
            content_type = self._classify_content_type(combined_text)
            sentiment = self._classify_sentiment(combined_text)
            technical_quality = self._assess_technical_quality(combined_text)
            market_analysis = self._classify_market_analysis(combined_text)
            impact = self._assess_impact(combined_text)
            
            # Identify contributing messages (those with subnet mentions)
            contributing = []
            for msg in normalized_messages:
                mentions = self.extract_subnet_mentions(msg.content)
                if mentions:
                    contributing.append(msg.message_id)
            
            # Build final classification
            return MessageGroupClassification(
                subnet_id=subnet_result['id'],
                subnet_name=subnet_result['name'],
                content_type=ContentType(content_type),
                sentiment=Sentiment(sentiment),
                technical_quality=TechnicalQuality(technical_quality),
                market_analysis=MarketAnalysis(market_analysis),
                impact_potential=ImpactPotential(impact),
                relevance_confidence=subnet_result['confidence'],
                evidence_spans=subnet_result['evidence'],
                anchors_detected=subnet_result['anchors'],
                message_count=len(normalized_messages),
                contributing_messages=contributing
            )
            
        except Exception as e:
            logger.error(f"[TELEGRAM_ANALYZER] Classification error: {e}")
            return None
    
    def classify_messages_from_dicts(self, messages: List[Dict], subnet_id: Optional[int] = None) -> Optional[MessageGroupClassification]:
        """
        Classify messages from dictionary format.
        
        Args:
            messages: List of dicts with keys: message_id, username, content, timestamp (optional), reply_to (optional)
            subnet_id: Optional subnet ID (e.g., 45, 30). If provided, uses this subnet instead of detecting from text.
            
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
        return self.classify_message_group(telegram_messages, subnet_id=subnet_id)
    
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
        except:
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
        except:
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
        except:
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
        except:
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
        except:
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
        except:
            return "NONE"
    
    def analyze_message_group_complete(self, messages: List, subnet_id: Optional[int] = None) -> dict:
        """
        Analyze message group and return rich classification data.
        
        Args:
            messages: List of TelegramMessage objects or dicts. Dicts should have: message_id, username, content (or text)
            subnet_id: Optional subnet ID (e.g., 45, 30). If provided, uses this subnet instead of detecting from text.
            
        Returns:
            Dict with classification, subnet_relevance, sentiment info
        """
        import time
        start_time = time.time()
        logger.info(f"[TELEGRAM_ANALYZER] Starting analysis for {len(messages)} messages" + (f" (subnet_id={subnet_id})" if subnet_id else ""))
        
        # Run atomic classification
        classification = self.classify_message_group(messages, subnet_id=subnet_id)
        
        if classification is None:
            logger.warning(f"[TELEGRAM_ANALYZER] Classification failed")
            return {
                "classification": None,
                "subnet_relevance": {},
                "timestamp": datetime.now().isoformat()
            }
        
        # Build subnet_relevance dict
        subnet_relevance = {}
        if classification.subnet_id != 0:
            subnet_name = classification.subnet_name
            subnet_relevance[subnet_name] = {
                "subnet_id": classification.subnet_id,
                "subnet_name": subnet_name,
                "relevance": 1.0,
                "relevance_confidence": classification.relevance_confidence,
                "content_type": classification.content_type.value,
                "sentiment": classification.sentiment.value,
                "technical_quality": classification.technical_quality.value,
                "market_analysis": classification.market_analysis.value,
                "impact_potential": classification.impact_potential.value,
                "evidence_spans": classification.evidence_spans,
                "anchors_detected": classification.anchors_detected,
                "message_count": classification.message_count,
                "contributing_messages": classification.contributing_messages,
            }
        
        total_time = time.time() - start_time
        logger.info(f"[TELEGRAM_ANALYZER] Analysis completed in {total_time:.2f}s")
        
        # Sentiment mapping for backward compatibility
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
            "subnet_relevance": subnet_relevance,
            "sentiment": sentiment_float,
            "sentiment_enum": sentiment_enum,
            "message_count": len(messages),
            "timestamp": datetime.now().isoformat()
        }
    
    def analyze_messages_from_dicts_complete(self, messages: List[Dict], subnet_id: Optional[int] = None) -> dict:
        """
        Convenience method to analyze messages from dictionary format.
        
        Args:
            messages: List of message dicts
            subnet_id: Optional subnet ID (e.g., 45, 30). If provided, uses this subnet instead of detecting from text.
            
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
        return self.analyze_message_group_complete(telegram_messages, subnet_id=subnet_id)

