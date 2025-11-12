"""
Deterministic X Post Classification for BitTensor Subnet Relevance

Uses structured function calling with categorical enums to achieve deterministic
LLM evaluation. Validators can verify miner classifications via exact matching
of canonical strings.

Key Features:
- Hierarchical trigger rules (SN mention > alias > name+anchor > NONE)
- Explicit abstain logic (subnet_id=0 for ties/unknown)
- Evidence extraction (exact spans + anchors for auditability)
- Fixed rubrics for each enum (not vibes-based assessment)
"""

from openai import OpenAI
import json
from typing import Dict, List, Optional
from dataclasses import dataclass
import time
from datetime import datetime
import bittensor as bt

from .classifications import ContentType, Sentiment, TechnicalQuality, MarketAnalysis, ImpactPotential

# Import centralized config (loads .miner_env and .vali_env)
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
        # Sort evidence for determinism
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
        """
        Get subnet tokens dict for grader compatibility.
        
        Returns dict mapping subnet_name -> relevance score:
        - relevance=1.0 if this classification matched a subnet
        - relevance=0.0 if no match (subnet_id=0)
        """
        if self.subnet_id == 0:
            return {}  # No subnet matched
        return {self.subnet_name: 1.0}  # Binary: matched or not


# Tool schema for function calling
CLASSIFICATION_TOOL = {
    "type": "function",
    "function": {
        "name": "classify_bittensor_post",
        "description": "Classify a BitTensor subnet-related X post across multiple dimensions",
        "parameters": {
            "type": "object",
            "properties": {
                "subnet_id": {
                    "type": "integer",
                    "description": "Primary subnet this tweet is about (0 for NONE/UNKNOWN)"
                },
                "content_type": {
                    "type": "string",
                    "enum": [ct.value for ct in ContentType],
                    "description": "Primary type of content in the tweet"
                },
                "sentiment": {
                    "type": "string",
                    "enum": [s.value for s in Sentiment],
                    "description": "Overall market sentiment tone"
                },
                "technical_quality": {
                    "type": "string",
                    "enum": [tq.value for tq in TechnicalQuality],
                    "description": "Quality/clarity of technical information"
                },
                "market_analysis": {
                    "type": "string",
                    "enum": [ma.value for ma in MarketAnalysis],
                    "description": "Type of market analysis if applicable"
                },
                "impact_potential": {
                    "type": "string",
                    "enum": [ip.value for ip in ImpactPotential],
                    "description": "Expected community/ecosystem impact"
                },
                "relevance_confidence": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                    "description": "Confidence in subnet match"
                },
                "evidence_spans": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Exact substrings from post that triggered the subnet match"
                },
                "anchors_detected": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "BitTensor anchor words found in post"
                }
            },
            "required": ["subnet_id", "content_type", "sentiment", "technical_quality", 
                        "market_analysis", "impact_potential", "relevance_confidence", 
                        "evidence_spans", "anchors_detected"]
        }
    }
}


class SubnetRelevanceAnalyzer:
    """
    Deterministic X post classifier using structured function calling with categorical enums
    
    Backward compatible with original SubnetRelevanceAnalyzer while adding new deterministic
    classification capabilities.
    """
    
    def __init__(self, model: str = None, api_key: str = None, llm_base: str = None, subnets: List[Dict] = None):
        """
        Initialize analyzer with subnet registry and LLM config
        
        Args:
            model: LLM model identifier (optional, uses config.MODEL if not provided)
            api_key: API key for LLM service (optional, uses config.API_KEY if not provided)
            llm_base: Base URL for LLM service (optional, uses config.LLM_BASE if not provided)
            subnets: List of subnet dictionaries (optional, can be added via register_subnet)
        """
        self.subnet_registry = {}  # For backward compatibility with register_subnet()
        
        # Use provided values or fall back to centralized config
        if config:
            self.model = model or config.MODEL
            self.api_key = api_key or config.API_KEY
            self.llm_base = llm_base or config.LLM_BASE
        else:
            self.model = model
            self.api_key = api_key
            self.llm_base = llm_base
        
        # Validate API_KEY is set
        if not self.api_key:
            raise ValueError(
                f"API_KEY environment variable is required.\n"
                f"Please set it in talisman_ai_subnet/.miner_env or .vali_env file."
            )
        
        self.client = OpenAI(base_url=self.llm_base, api_key=self.api_key)
        
        # Initialize subnets dict for new classification system
        if subnets:
            self.subnets = {s["id"]: s for s in subnets}
            for s in subnets:
                self.subnet_registry[s["id"]] = s
        else:
            self.subnets = {}
        
        # Add NONE subnet for abstain logic
        self.subnets[0] = {
            "id": 0, 
            "name": "NONE_OF_THE_ABOVE", 
            "description": "General BitTensor content not specific to a listed subnet"
        }
        
        bt.logging.info(f"[ANALYZER] Initialized with model: {self.model}")
        if subnets:
            bt.logging.info(f"[ANALYZER] Registered {len(self.subnets)-1} subnets (+1 NONE)")
    
    def register_subnet(self, subnet_data: dict):
        """
        Register a subnet with its metadata (backward compatibility method)
        
        Args:
            subnet_data: Dict with subnet info including 'id', 'name', etc.
        """
        subnet_id = subnet_data['id']
        self.subnet_registry[subnet_id] = subnet_data
        self.subnets[subnet_id] = subnet_data
        bt.logging.debug(f"[ANALYZER] Registered subnet {subnet_id}: {subnet_data.get('name')}")
    
    def _build_structured_subnet_rows(self, candidate_ids: set = None) -> str:
        """
        Build structured subnet rows for exact-match lookup
        
        Args:
            candidate_ids: Optional set of subnet IDs to include. If None, includes all subnets.
        """
        rows = []
        subnet_ids = candidate_ids if candidate_ids is not None else set(self.subnets.keys())
        
        for sid in sorted(subnet_ids):
            if sid == 0:
                continue  # Skip NONE for the main list
            if sid not in self.subnets:
                continue
            s = self.subnets[sid]
            
            # Extract structured fields
            aliases = s.get("unique_identifiers", [])
            if s.get("name"):
                aliases.append(s["name"])
            
            # Get keywords from description
            desc_words = s.get("description", "").split()[:10]
            keywords = [w.strip(".,;:") for w in desc_words if len(w) > 4][:5]
            
            row = f"ID: {sid:2d} | Name: {s.get('name', 'Unknown'):20s} | Aliases: {aliases[:4]} | Keywords: {keywords[:4]}"
            rows.append(row)
        
        return "\n".join(rows)
    
    def classify_post(self, text: str) -> Optional[PostClassification]:
        """
        Classify an X post using structured function calling with hierarchical rules
        
        Args:
            text: X post text to classify
            
        Returns:
            PostClassification if successful, None if parsing fails
        """
        
        # PRE-FILTER: Deterministically narrow down to candidate subnets (no LLM)
        # This reduces prompt size from ~28K to ~5-8K tokens (70%+ savings)
        text_lower = text.lower()
        candidate_subnet_ids = set()
        
        for subnet_id, subnet_data in self.subnets.items():
            if subnet_id == 0:  # Always include NONE option
                candidate_subnet_ids.add(subnet_id)
                continue
            
            # Check 1: Direct SN pattern match (SN13, SN20, etc)
            if f"sn{subnet_id}" in text_lower or f"sn {subnet_id}" in text_lower or f"subnet {subnet_id}" in text_lower:
                candidate_subnet_ids.add(subnet_id)
                continue
            
            # Check 2: Unique identifier match
            for identifier in subnet_data.get('unique_identifiers', []):
                if identifier.lower() in text_lower:
                    candidate_subnet_ids.add(subnet_id)
                    break
            
            # Check 3: Subnet name appears near BitTensor anchor words
            subnet_name = subnet_data.get('name', '').lower()
            if subnet_name and len(subnet_name) > 3:
                if subnet_name in text_lower:
                    # Check if there's a BitTensor anchor nearby
                    anchors = ['bittensor', 'subnet', 'validator', 'tao', 'emissions', 'miner']
                    if any(anchor in text_lower for anchor in anchors):
                        candidate_subnet_ids.add(subnet_id)
        
        # If no candidates found, only include NONE (subnet_id=0)
        if len(candidate_subnet_ids) == 1 and 0 in candidate_subnet_ids:
            bt.logging.debug(f"[ANALYZER] Pre-filter: no candidates found, using NONE only")
        else:
            bt.logging.debug(f"[ANALYZER] Pre-filter: {len(self.subnets)} subnets → {len(candidate_subnet_ids)} candidates")
        
        # Build subnet rows ONLY for candidates (massive token savings)
        subnet_rows = self._build_structured_subnet_rows(candidate_subnet_ids)
        
        system_prompt = """You are a deterministic BitTensor post classifier.
You MUST respond exactly once by calling classify_bittensor_post.
Never write free text. Output only the function call.

DECISION RULES (apply in order; earlier rules override later ones):

[Subnet Selection Triggers - HIERARCHICAL]
1) If post contains exact SN pattern (SN<number>) present in AVAILABLE_SUBNETS → choose that subnet_id
2) Else if post contains exact alias/handle/org from AVAILABLE_SUBNETS → choose that subnet_id
3) Else if post contains subnet NAME within 50 chars of anchor word (bittensor, subnet, validator, TAO, emissions, miner) → choose that subnet_id
4) Else → subnet_id=0 (NONE/UNKNOWN - do not guess)

TIES: If two or more subnets satisfy the highest-priority trigger → subnet_id=0

HOMONYM RULE: If token matches non-BitTensor meaning (e.g. Omron PLC) and post lacks BitTensor anchors → subnet_id=0

[Enum Rules]
- sentiment: very_bullish (🚀, moon, ATH, pump), bullish (positive price/growth), neutral (default), bearish (concerns, dump), very_bearish (crash, exploit, failure)
- technical_quality: high (≥2 specifics: APIs, versions, repos, endpoints, metrics), medium (1 specific), low (claims without specifics), none (no technical content)
- content_type: prefer most specific (announcement > milestone > partnership > technical_insight > tutorial > security > governance > market_discussion > community/hype/opinion > other)
- market_analysis: technical (indicators, order flow), economic (fundamentals, costs), political (regulatory/governance), social (narrative/virality), other
- impact_potential: HIGH (major release/mainnet/security incident/governance passed), MEDIUM (notable update/launch), LOW (minor info), NONE (chatty/irrelevant)

[Confidence]
- high: direct SN/alias/handle mention
- medium: name+anchor rule (trigger 3)
- low: weak hints (prefer subnet_id=0)

[Evidence]
- evidence_spans: exact substrings that triggered match (e.g. "SN13", "@omron_subnet", "x402 API")
- anchors_detected: BitTensor anchor words found (e.g. "bittensor", "subnet", "validator", "TAO")"""

        # Few-shot examples
        few_shot_examples = """
EXAMPLE A — Direct SN mention → high confidence:
Post: "SN13 just launched x402 API for social data."
Call: {"subnet_id": 13, "content_type": "announcement", "sentiment": "neutral", "technical_quality": "medium", "market_analysis": "other", "impact_potential": "MEDIUM", "relevance_confidence": "high", "evidence_spans": ["SN13", "x402 API"], "anchors_detected": ["SN13"]}

EXAMPLE B — Alias/handle match:
Post: "Omron's zkML fingerprinting proves inference authenticity."
Call: {"subnet_id": 2, "content_type": "technical_insight", "sentiment": "neutral", "technical_quality": "high", "market_analysis": "other", "impact_potential": "MEDIUM", "relevance_confidence": "high", "evidence_spans": ["Omron", "zkML"], "anchors_detected": []}

EXAMPLE C — General BitTensor, no specific subnet → NONE:
Post: "Bittensor emissions debate rages on; validators should vote."
Call: {"subnet_id": 0, "content_type": "governance", "sentiment": "neutral", "technical_quality": "none", "market_analysis": "political", "impact_potential": "LOW", "relevance_confidence": "low", "evidence_spans": [], "anchors_detected": ["Bittensor", "emissions", "validators"]}
"""

        user_prompt = f"""Classify the post below using the rules. Use only subnets from the list. If no rule fires, pick subnet_id=0.

POST:
"{text}"

AVAILABLE_SUBNETS (exact-match fields only):
{subnet_rows}

{few_shot_examples}

Now classify the provided post. Return exact substrings that triggered your decision."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                tools=[CLASSIFICATION_TOOL],
                tool_choice={"type": "function", "function": {"name": "classify_bittensor_post"}},
                temperature=0,  # Deterministic
                max_tokens=300
            )
            
            # Extract function call
            tool_call = response.choices[0].message.tool_calls[0]
            args = json.loads(tool_call.function.arguments)
            
            # Parse and validate
            return self._parse_classification(args)
            
        except Exception as e:
            bt.logging.error(f"[ANALYZER] Classification error: {e}")
            return None
    
    def analyze_tweet_complete(self, text: str) -> dict:
        """
        Analyze post and return rich classification data (NOT lossy conversions)
        
        Returns the full PostClassification object with all categorical enums intact.
        This preserves data quality for database storage and downstream processing.
        
        Args:
            text: Post text to analyze
            
        Returns:
            Dict with:
                - classification: Full PostClassification object (or None if failed)
                - subnet_relevance: Dict mapping subnet_name -> classification dict
                - timestamp: ISO timestamp
        """
        start_time = time.time()
        bt.logging.info(f"[ANALYZER] Starting analysis for post (length: {len(text)} chars)")
        
        # Run classification
        classification = self.classify_post(text)
        
        if classification is None:
            bt.logging.warning(f"[ANALYZER] Classification failed")
            return {
                "classification": None,
                "subnet_relevance": {},
                "timestamp": datetime.now().isoformat()
            }
        
        # Build subnet_relevance dict with FULL classification data (no lossy conversions)
        subnet_relevance = {}
        if classification.subnet_id != 0:
            subnet_name = classification.subnet_name
            subnet_relevance[subnet_name] = {
                "subnet_id": classification.subnet_id,
                "subnet_name": subnet_name,
                "relevance": 1.0,  # Binary: matched (1.0) or not matched (0.0) - grader compatibility
                "relevance_confidence": classification.relevance_confidence,  # Keep as "high"/"medium"/"low"
                "content_type": classification.content_type.value,
                "sentiment": classification.sentiment.value,  # Keep as enum string
                "technical_quality": classification.technical_quality.value,
                "market_analysis": classification.market_analysis.value,
                "impact_potential": classification.impact_potential.value,
                "evidence_spans": classification.evidence_spans,
                "anchors_detected": classification.anchors_detected,
            }
        
        total_time = time.time() - start_time
        bt.logging.info(f"[ANALYZER] Analysis completed in {total_time:.2f}s")
        
        # Map sentiment enum to float for grader compatibility (while keeping enum available)
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
            "classification": classification,  # Full PostClassification object
            "subnet_relevance": subnet_relevance,  # Rich dict ready for DB storage
            "sentiment": sentiment_float,  # Float for grader compatibility
            "sentiment_enum": sentiment_enum,  # Enum string for new code
            "timestamp": datetime.now().isoformat()
        }
    
    def _parse_classification(self, args: dict) -> Optional[PostClassification]:
        """Parse and validate function call arguments"""
        try:
            subnet_id = int(args["subnet_id"])
            if subnet_id not in self.subnets:
                bt.logging.warning(f"[ANALYZER] Unknown subnet_id: {subnet_id}")
                return None
            
            # Parse evidence spans
            evidence_spans = args.get("evidence_spans", [])
            if not isinstance(evidence_spans, list):
                evidence_spans = []
            
            anchors_detected = args.get("anchors_detected", [])
            if not isinstance(anchors_detected, list):
                anchors_detected = []
            
            return PostClassification(
                subnet_id=subnet_id,
                subnet_name=self.subnets[subnet_id]["name"],
                content_type=ContentType(args["content_type"]),
                sentiment=Sentiment(args["sentiment"]),
                technical_quality=TechnicalQuality(args["technical_quality"]),
                market_analysis=MarketAnalysis(args["market_analysis"]),
                impact_potential=ImpactPotential(args["impact_potential"]),
                relevance_confidence=args["relevance_confidence"],
                evidence_spans=evidence_spans,
                anchors_detected=anchors_detected
            )
        except (ValueError, KeyError) as e:
            bt.logging.error(f"[ANALYZER] Parse error: {e}")
            return None

