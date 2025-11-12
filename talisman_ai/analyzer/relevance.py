"""
Deterministic X Post Classification for BitTensor Subnet Relevance

Uses atomic tool calls for each classification dimension to achieve deterministic
LLM evaluation. Validators can verify miner classifications via exact matching
of canonical strings.

Key Features:
- Atomic decisions: One tool call per classification dimension
- Hierarchical trigger rules (SN mention > alias > name+anchor > NONE)
- Explicit abstain logic (subnet_id=0 for ties/unknown)
- Evidence extraction (exact spans + anchors for auditability)
"""

from openai import OpenAI
import json
from typing import Dict, List, Optional
from dataclasses import dataclass
import time
from datetime import datetime
import bittensor as bt

from .classifications import ContentType, Sentiment, TechnicalQuality, MarketAnalysis, ImpactPotential

# Import centralized config
try:
    from talisman_ai import config
except ImportError:
    config = None


@dataclass
class PostClassification:
    """Canonical classification result"""
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
        "description": "Identify which subnet this post is about",
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


class SubnetRelevanceAnalyzer:
    """
    Deterministic X post classifier using atomic tool calls
    
    Each classification dimension is decided independently via its own tool call,
    eliminating compound decision variance.
    """
    
    def __init__(self, model: str = None, api_key: str = None, llm_base: str = None, subnets: List[Dict] = None):
        """Initialize analyzer with subnet registry and LLM config"""
        self.subnet_registry = {}
        
        # Use provided values or fall back to centralized config
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
        
        bt.logging.info(f"[ANALYZER] Initialized with model: {self.model}")
        if subnets:
            bt.logging.info(f"[ANALYZER] Registered {len(self.subnets)-1} subnets (+1 NONE)")
    
    def register_subnet(self, subnet_data: dict):
        """Register a subnet (backward compatibility)"""
        subnet_id = subnet_data['id']
        self.subnet_registry[subnet_id] = subnet_data
        self.subnets[subnet_id] = subnet_data
        bt.logging.debug(f"[ANALYZER] Registered subnet {subnet_id}: {subnet_data.get('name')}")
    
    def _build_subnet_context(self) -> str:
        """Build rich semantic context for subnet identification"""
        contexts = []
        for sid in sorted(self.subnets.keys()):
            if sid == 0:
                continue
            s = self.subnets[sid]
            
            # Build comprehensive description
            ctx = f"SN{sid} ({s.get('name', 'Unknown')}): "
            ctx += s.get('description', '')[:150]
            
            # Add identifiers
            ids = s.get('unique_identifiers', [])
            if ids:
                ctx += f" | IDs: {', '.join(ids[:3])}"
            
            contexts.append(ctx)
            
            if len(contexts) >= 30:  # Limit context size
                break
        
        return '\n'.join(contexts)
    
    def classify_post(self, text: str) -> Optional[PostClassification]:
        """
        Classify using atomic tool calls for each dimension
        
        Args:
            text: X post text to classify
            
        Returns:
            PostClassification if successful, None if parsing fails
        """
        try:
            # Step 1: Identify subnet (most critical decision)
            subnet_result = self._identify_subnet(text)
            
            # Step 2-6: Classify other dimensions atomically
            content_type = self._classify_content_type(text)
            sentiment = self._classify_sentiment(text)
            technical_quality = self._assess_technical_quality(text)
            market_analysis = self._classify_market_analysis(text)
            impact = self._assess_impact(text)
            
            # Build final classification
            return PostClassification(
                subnet_id=subnet_result['id'],
                subnet_name=subnet_result['name'],
                content_type=ContentType(content_type),
                sentiment=Sentiment(sentiment),
                technical_quality=TechnicalQuality(technical_quality),
                market_analysis=MarketAnalysis(market_analysis),
                impact_potential=ImpactPotential(impact),
                relevance_confidence=subnet_result['confidence'],
                evidence_spans=subnet_result['evidence'],
                anchors_detected=subnet_result['anchors']
            )
            
        except Exception as e:
            bt.logging.error(f"[ANALYZER] Classification error: {e}")
            return None
    
    def _identify_subnet(self, text: str) -> dict:
        """Atomic decision: Which subnet?"""
        context = self._build_subnet_context()
        
        prompt = f"""Identify which BitTensor subnet this post is about.

SUBNET DATABASE:
{context}

POST: "{text}"

RULES:
1. "SN45" or "Subnet 45" → subnet_id=45
2. "@omron_ai" or similar handles → match to corresponding subnet
3. Project names near BitTensor words → match to subnet
4. General BitTensor content → subnet_id=0
5. Ambiguous or unclear → subnet_id=0"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                tools=[SUBNET_ID_TOOL],
                tool_choice={"type": "function", "function": {"name": "identify_subnet"}},
                temperature=0,
                max_tokens=200
            )
            
            args = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
            subnet_id = args.get("subnet_id", 0)
            
            return {
                'id': subnet_id,
                'name': self.subnets.get(subnet_id, {}).get("name", "NONE_OF_THE_ABOVE"),
                'confidence': args.get("confidence", "low"),
                'evidence': args.get("evidence_spans", []),
                'anchors': args.get("anchors_detected", [])
            }
        except:
            return {'id': 0, 'name': "NONE_OF_THE_ABOVE", 'confidence': "low", 'evidence': [], 'anchors': []}
    
    def _classify_content_type(self, text: str) -> str:
        """Atomic decision: Content type"""
        prompt = f"""Classify content type of: "{text}"

Pick the MOST SPECIFIC category:
- announcement: launches, releases
- partnership: collaborations
- technical_insight: technical analysis
- milestone: achievements
- security: audits, vulnerabilities
- governance: voting, proposals
- market_discussion: price talk
- community: general chat
- other: doesn't fit"""

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
        prompt = f"""Classify sentiment of: "{text}"

- very_bullish: 🚀, moon, explosive
- bullish: positive, optimistic
- neutral: factual, balanced
- bearish: concerns, negative
- very_bearish: crash, failure"""

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
        except:
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
        except:
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
        except:
            return "NONE"
    
    def analyze_tweet_complete(self, text: str) -> dict:
        """
        Analyze post and return rich classification data
        
        Maintains backward compatibility with existing interface.
        """
        start_time = time.time()
        bt.logging.info(f"[ANALYZER] Starting analysis for post (length: {len(text)} chars)")
        
        # Run atomic classification
        classification = self.classify_post(text)
        
        if classification is None:
            bt.logging.warning(f"[ANALYZER] Classification failed")
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
            }
        
        total_time = time.time() - start_time
        bt.logging.info(f"[ANALYZER] Analysis completed in {total_time:.2f}s")
        
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
            "timestamp": datetime.now().isoformat()
        }
    
    def _parse_classification(self, args: dict) -> Optional[PostClassification]:
        """Parse and validate function call arguments (backward compatibility)"""
        try:
            subnet_id = int(args["subnet_id"])
            if subnet_id not in self.subnets:
                bt.logging.warning(f"[ANALYZER] Unknown subnet_id: {subnet_id}")
                return None
            
            return PostClassification(
                subnet_id=subnet_id,
                subnet_name=self.subnets[subnet_id]["name"],
                content_type=ContentType(args["content_type"]),
                sentiment=Sentiment(args["sentiment"]),
                technical_quality=TechnicalQuality(args["technical_quality"]),
                market_analysis=MarketAnalysis(args["market_analysis"]),
                impact_potential=ImpactPotential(args["impact_potential"]),
                relevance_confidence=args["relevance_confidence"],
                evidence_spans=args.get("evidence_spans", []),
                anchors_detected=args.get("anchors_detected", [])
            )
        except (ValueError, KeyError) as e:
            bt.logging.error(f"[ANALYZER] Parse error: {e}")
            return None
