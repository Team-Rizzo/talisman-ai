"""
X Post Scoring and Validator Batch Verification

Provides functions to:
1. Score post components (value, recency)
2. Validate miner batches via sampling and exact canonical string matching

Validator Flow:
- Miner submits batch of N posts with classifications
- Validator samples M posts (e.g., 10-20 from 100)
- Validator runs classification on sampled posts
- Validator compares canonical strings for exact match
- If all match → accept batch, else → reject batch
"""

from datetime import datetime, timezone
from typing import Dict, List, Tuple
import random
import bittensor as bt

from talisman_ai.utils.api_models import TweetWithAuthor
from .relevance import SubnetRelevanceAnalyzer, PostClassification


# ===== Normalization Caps =====
CAPS = {
    "likes": 5_000,
    "retweets": 1_000,
    "quotes": 300,
    "replies": 600,
    "followers": 200_000,
    "account_age_days": 7 * 365,
}


def _clamp01(x: float) -> float:
    """Clamp a float value to the range [0.0, 1.0]"""
    return max(0.0, min(1.0, float(x)))


def _norm(value: float, cap: float) -> float:
    """
    Normalize a value to [0.0, 1.0] using linear scaling with a hard cap
    
    Args:
        value: The raw value to normalize
        cap: The cap threshold - values at or above this threshold yield 1.0
        
    Returns:
        Normalized value in [0.0, 1.0]
    """
    return _clamp01(value / cap)


def recency_score(post_date_iso: str, horizon_hours: float = 24.0) -> float:
    """
    Compute recency score based on post age using linear time decay
    
    Args:
        post_date_iso: ISO format date string (e.g., "2024-01-01T12:00:00Z")
        horizon_hours: Time window in hours (default: 24.0)
        
    Returns:
        Recency score in [0.0, 1.0]
    """
    dt = datetime.fromisoformat(post_date_iso.replace("Z", "+00:00")).astimezone(timezone.utc)
    age_hours = (datetime.now(timezone.utc) - dt).total_seconds() / 3600.0
    return _clamp01(1.0 - age_hours / horizon_hours)


def value_score(
    post_info: TweetWithAuthor,
    caps: Dict = CAPS,
) -> float:
    """
    Compute value score based on engagement metrics and author credibility
    
    The value score is an equal-weight average of five normalized components:
    1-4. Engagement signals (likes, retweets, quotes, replies)
    5. Author credibility (follower count)
    
    Args:
        post_info: TweetWithAuthor object
        caps: Dictionary of cap values for normalization (defaults to CAPS)
        
    Returns:
        Value score in [0.0, 1.0]
    """
    # Get followers count from author if available
    followers = post_info.author.followers_count if post_info.author else 0
    comps = [
        _norm(post_info.like_count or 0, caps["likes"]),
        _norm(post_info.retweet_count or 0, caps["retweets"]),
        _norm(post_info.quote_count or 0, caps["quotes"]),
        _norm(post_info.reply_count or 0, caps["replies"]),
        _norm(followers or 0, caps["followers"]),
        # _norm(post_info.author.account_age_days or 0, caps["account_age_days"]),  # Excluded for now
    ]
    return sum(comps) / len(comps)


# ===== Validator Batch Verification =====

def validate_miner_batch(
    miner_batch: List[TweetWithAuthor],
    analyzer: SubnetRelevanceAnalyzer,
    sample_size: int = 1,
    seed: int = None
) -> Tuple[bool, Dict]:
    """
    Validate a miner's batch by sampling posts and checking classifications.
    
    All fields require exact match:
    subnet_id, sentiment, content_type, technical_quality, market_analysis, impact_potential
    
    Args:
        miner_batch: List of TweetWithAuthor objects
        analyzer: SubnetRelevanceAnalyzer instance
        sample_size: Number of posts to sample (default: 1)
        seed: Random seed for reproducible sampling
        
    Returns:
        Tuple of (is_valid, result_dict)
    """
    if seed is not None:
        random.seed(seed)
    
    sample_size = min(sample_size, len(miner_batch))
    sampled_posts = random.sample(miner_batch, sample_size)
    
    bt.logging.info(f"[Validator] Sampling {sample_size} post(s) from batch of {len(miner_batch)}")
    
    matches = 0
    discrepancies = []
    
    for i, post_data in enumerate(sampled_posts):
        post_text = post_data.text
        miner_analysis = post_data.analysis
        
        if miner_analysis is None:
            bt.logging.warning(f"[Validator] No miner classification for post {i+1}")
            discrepancies.append({
                "post_index": i,
                "reason": "missing_miner_classification",
                "post_preview": post_text[:100] if post_text else ""
            })
            continue
        
        validator_result = analyzer.classify_post(post_text)
        if validator_result is None:
            bt.logging.warning(f"[Validator] Failed to classify post {i+1}")
            discrepancies.append({
                "post_index": i,
                "reason": "validator_classification_failed",
                "post_preview": post_text[:100] if post_text else ""
            })
            continue
        
        # Helper to safely lowercase for case-insensitive comparison
        def _lower(val):
            return val.lower() if isinstance(val, str) else val
        
        # Extract miner fields
        m_subnet = miner_analysis.subnet_id
        m_sent = miner_analysis.sentiment
        m_content = miner_analysis.content_type
        m_tech = miner_analysis.technical_quality
        m_market = miner_analysis.market_analysis
        m_impact = miner_analysis.impact_potential
        
        # Extract validator fields
        v_subnet = validator_result.subnet_id
        v_sent = validator_result.sentiment.value if validator_result.sentiment else None
        v_content = validator_result.content_type.value if validator_result.content_type else None
        v_tech = validator_result.technical_quality.value if validator_result.technical_quality else None
        v_market = validator_result.market_analysis.value if validator_result.market_analysis else None
        v_impact = validator_result.impact_potential.value if validator_result.impact_potential else None
        
        # Check matches (case-insensitive for string fields)
        subnet_ok = m_subnet == v_subnet
        sentiment_ok = _lower(m_sent) == _lower(v_sent)
        content_ok = _lower(m_content) == _lower(v_content)
        tech_ok = _lower(m_tech) == _lower(v_tech)
        market_ok = _lower(m_market) == _lower(v_market)
        impact_ok = _lower(m_impact) == _lower(v_impact)
        
        all_ok = subnet_ok and sentiment_ok and content_ok and tech_ok and market_ok and impact_ok
        
        if all_ok:
            matches += 1
            bt.logging.debug(f"[Validator] Post {i+1}: MATCH")
        else:
            # Build list of failed fields for detailed logging
            failed_fields = []
            if not subnet_ok:
                failed_fields.append(f"subnet_id (miner={m_subnet} vs validator={v_subnet})")
            if not sentiment_ok:
                failed_fields.append(f"sentiment (miner={m_sent} vs validator={v_sent})")
            if not content_ok:
                failed_fields.append(f"content_type (miner={m_content} vs validator={v_content})")
            if not tech_ok:
                failed_fields.append(f"technical_quality (miner={m_tech} vs validator={v_tech})")
            if not market_ok:
                failed_fields.append(f"market_analysis (miner={m_market} vs validator={v_market})")
            if not impact_ok:
                failed_fields.append(f"impact_potential (miner={m_impact} vs validator={v_impact})")
            
            # Log detailed mismatch info
            bt.logging.warning(f"[Validator] Post {i+1}: MISMATCH - Failed fields: {', '.join(failed_fields)}")
            bt.logging.warning(f"[Validator] Post {i+1} text preview: {post_text[:200] if post_text else '(empty)'}")
            bt.logging.warning(f"[Validator] Post {i+1} Miner classification: subnet_id={m_subnet}, sentiment={m_sent}, content_type={m_content}, tech={m_tech}, market={m_market}, impact={m_impact}")
            bt.logging.warning(f"[Validator] Post {i+1} Validator classification: subnet_id={v_subnet}, sentiment={v_sent}, content_type={v_content}, tech={v_tech}, market={v_market}, impact={v_impact}")
            
            discrepancies.append({
                "post_index": i,
                "reason": "classification_mismatch",
                "miner": {
                    "subnet_id": m_subnet, "sentiment": m_sent, "content_type": m_content,
                    "technical_quality": m_tech, "market_analysis": m_market, "impact_potential": m_impact
                },
                "validator": {
                    "subnet_id": v_subnet, "sentiment": v_sent, "content_type": v_content,
                    "technical_quality": v_tech, "market_analysis": v_market, "impact_potential": v_impact
                },
                "field_results": {
                    "subnet_id": subnet_ok, "sentiment": sentiment_ok, "content_type": content_ok,
                    "technical_quality": tech_ok, "market_analysis": market_ok, "impact_potential": impact_ok
                },
                "post_preview": post_text[:100] if post_text else ""
            })
    
    is_valid = matches == sample_size and len(discrepancies) == 0
    
    result = {
        "is_valid": is_valid,
        "matches": matches,
        "total_sampled": sample_size,
        "discrepancies": discrepancies,
        "match_rate": matches / sample_size if sample_size > 0 else 0.0
    }
    
    if is_valid:
        bt.logging.success(f"[Validator] Batch ACCEPTED: {matches}/{sample_size} matches")
    else:
        bt.logging.warning(f"[Validator] Batch REJECTED: {matches}/{sample_size} matches, {len(discrepancies)} discrepancies")
    
    return is_valid, result


def _build_canonical_from_dict(classification: PostClassification) -> str:
    """
    Build canonical string from miner's classification dictionary
    
    This must match the exact format from PostClassification.to_canonical_string()
    
    Args:
        classification: Dict with keys matching PostClassification fields
        
    Returns:
        Canonical string for exact matching
    """
    # Extract fields
    subnet_id = int(classification.subnet_id)
    content_type = classification.content_type
    sentiment = classification.sentiment
    technical_quality = classification.technical_quality
    market_analysis = classification.market_analysis
    impact_potential = classification.impact_potential
    relevance_confidence = classification.relevance_confidence
    evidence_spans = classification.evidence_spans
    anchors_detected = classification.anchors_detected
    
    # Sort evidence for determinism (same as PostClassification)
    sorted_evidence = "|".join(sorted([s.lower() for s in evidence_spans]))
    sorted_anchors = "|".join(sorted([s.lower() for s in anchors_detected]))
    
    return f"{subnet_id}|{content_type}|{sentiment}|{technical_quality}|{market_analysis}|{impact_potential}|{relevance_confidence}|{sorted_evidence}|{sorted_anchors}"


# ===== Scoring Weights =====
# Default weights for production scoring (used by score_post_entry)
# These weights prioritize relevance over value, with recency as a minor factor
RELEVANCE_WEIGHT = 0.50  # 50% weight on subnet relevance
VALUE_WEIGHT = 0.40      # 40% weight on signal value/quality
RECENCY_WEIGHT = 0.10    # 10% weight on recency


def compute_post_score(
    classification: PostClassification,
    post_info: TweetWithAuthor,
    weights: Dict = None
) -> float:
    """
    Compute final post score combining classification + engagement + recency
    
    Args:
        classification: PostClassification result
        post_info: TweetWithAuthor object
        weights: Optional custom weights dict
        
    Returns:
        Final score in [0.0, 1.0]
    """
    # Check if post is older than 10 days - if so, return 0
    if not post_info.created_at:
        return 0.0
    post_date_str = post_info.created_at if isinstance(post_info.created_at, str) else post_info.created_at.isoformat()
    dt = datetime.fromisoformat(post_date_str.replace("Z", "+00:00")).astimezone(timezone.utc)
    age_days = (datetime.now(timezone.utc) - dt).total_seconds() / (3600.0 * 24.0)
    if age_days > 10:
        return 0.0
    
    if weights is None:
        weights = {
            "relevance": RELEVANCE_WEIGHT,
            "value": VALUE_WEIGHT,
            "recency": RECENCY_WEIGHT
        }
    
    # Relevance: binary (1.0 if subnet_id != 0, else 0.0)
    relevance = 1.0 if classification.subnet_id != 0 else 0.0
    
    # Value: engagement + author credibility
    val = value_score(
        post_info=post_info,
    )
    
    # Recency
    rec = recency_score(post_info.created_at.isoformat())
    
    # Combine
    final = weights["relevance"] * relevance + weights["value"] * val + weights["recency"] * rec
    
    return _clamp01(final)


# ===== Backward Compatibility Functions for Legacy Code =====

def get_tokens_from_analysis(analysis_result: PostClassification) -> Dict[str, float]:
    """
    Extract tokens dict from analysis result for grader compatibility.
    
    Returns dict mapping subnet_name -> 1.0 (binary: matched or not).
    This maintains API compatibility while using the new classification system.
    
    Args:
        analysis_result: Result from analyzer.analyze_post_complete()
        
    Returns:
        Dict mapping subnet_name to 1.0 if matched, empty dict if no match
    """
    classification = analysis_result
    if classification is None:
        return {}
    return classification.to_tokens_dict()


def top_k_relevance_from_analyzer(text: str, analyzer, k: int = 5, analysis_result: PostClassification = None) -> Tuple[float, List[Tuple[str, PostClassification]]]:
    """
    Get classification from analyzer and return subnet relevance data.
    
    Args:
        text: The post text to analyze (only used if analysis_result is None)
        analyzer: SubnetRelevanceAnalyzer instance configured with subnet registry
        k: Number of top subnets to consider (default: 5)
        analysis_result: Optional pre-computed analysis result dict. If provided,
                        this will be used instead of calling analyze_post_complete again.
        
    Returns:
        Tuple of:
            - relevance: Binary relevance (1.0 if matched a subnet, 0.0 otherwise)
            - top: List of (subnet_name, classification_dict) tuples with full data
    """
    if analysis_result is None:
        out = analyzer.analyze_post_complete(text)
    else:
        out = analysis_result
    
    # Get classification object
    classification = out
    if classification is None or classification.subnet_id == 0:
        return 0.0, []  # No match = 0.0 relevance
    
    # Return full classification data (not lossy floats)
    subnet_name = classification.subnet_name
    classification_data = classification.to_dict()
    
    return 1.0, [(subnet_name, classification_data)]  # Binary: matched or not


def score_post_entry(entry: TweetWithAuthor, analyzer, k: int = 5, analysis_result: PostClassification = None) -> Dict:
    """
    Score a single post entry with rich classification data preserved.
    
    Returns both the final score AND the full classification object, so downstream
    consumers can use the rich categorical data for database storage, analytics, etc.
    
    Args:
        entry: TweetWithAuthor object
        analyzer: SubnetRelevanceAnalyzer instance
        k: Kept for API compatibility (not used anymore - we return 1 subnet or none)
        analysis_result: Optional pre-computed analysis result dict
        
    Returns:
        Dictionary containing:
            - url: Original post URL/identifier
            - classification: Full PostClassification object (or None)
            - subnet_data: Full classification dict for the matched subnet
            - relevance: Binary relevance (1.0 if matched, 0.0 if not)
            - value: Value score based on engagement [0.0, 1.0]
            - recency: Recency score based on post age [0.0, 1.0]
            - score: Final weighted score [0.0, 1.0]
    """
    info = entry

    # Check if post is older than 10 days - if so, return 0 score
    if not info.created_at:
        return {
            "url": entry.url,
            "classification": None,
            "subnet_data": None,
            "relevance": 0.0,
            "value": 0.0,
            "recency": 0.0,
            "score": 0.0
        }
    post_date_str = info.created_at if isinstance(info.created_at, str) else info.created_at.isoformat()
    dt = datetime.fromisoformat(post_date_str.replace("Z", "+00:00")).astimezone(timezone.utc)
    age_days = (datetime.now(timezone.utc) - dt).total_seconds() / (3600.0 * 24.0)
    if age_days > 10:
        return {
            "url": entry.url,
            "classification": None,
            "subnet_data": None,
            "relevance": 0.0,
            "value": 0.0,
            "recency": 0.0,
            "score": 0.0
        }

    # Get classification with full rich data
    rel, subnet_data = top_k_relevance_from_analyzer(info.text, analyzer, k=k, analysis_result=analysis_result)
    
    # Get the full classification object if available
    if analysis_result is None:
        analysis_result = analyzer.analyze_post_complete(info.text)
    classification = analysis_result
    
    # Compute engagement scores
    val = value_score(
        post_info=info,
    )
    rec = recency_score(info.created_at.isoformat())

    # Combine components using default weights
    final = RELEVANCE_WEIGHT * rel + VALUE_WEIGHT * val + RECENCY_WEIGHT * rec

    return {
        "url": entry.url,
        "classification": classification,  # Full PostClassification object
        "subnet_data": subnet_data[0] if subnet_data else None,  # Full classification dict
        "relevance": rel,  # Binary: 1.0 or 0.0
        "value": val,
        "recency": rec,
        "score": max(0.0, min(1.0, float(final)))
    }

