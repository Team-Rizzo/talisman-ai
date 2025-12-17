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

from talisman_ai.utils.api_models import TweetWithUser
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
    post_info: TweetWithUser,
    caps: Dict = CAPS,
) -> float:
    """
    Compute value score based on engagement metrics and author credibility
    
    The value score is an equal-weight average of five normalized components:
    1-4. Engagement signals (likes, retweets, quotes, replies)
    5. Author credibility (follower count)
    
    Args:
        post_info: TweetWithUser object
        caps: Dictionary of cap values for normalization (defaults to CAPS)
        
    Returns:
        Value score in [0.0, 1.0]
    """
    comps = [
        _norm(post_info.like_count or 0, caps["likes"]),
        _norm(post_info.retweet_count or 0, caps["retweets"]),
        _norm(post_info.quote_count or 0, caps["quotes"]),
        _norm(post_info.reply_count or 0, caps["replies"]),
        _norm(post_info.user.followers or 0, caps["followers"]),
        # _norm(post_info.user.account_age_days or 0, caps["account_age_days"]),  # Excluded for now
    ]
    return sum(comps) / len(comps)


# ===== Validator Batch Verification =====

def validate_miner_batch(
    miner_batch: List[TweetWithUser],
    analyzer: SubnetRelevanceAnalyzer,
    sample_size: int = 10,
    seed: int = None
) -> Tuple[bool, Dict]:
    """
    Validate a miner's batch by sampling posts and checking for exact classification matches
    
    Validator Logic:
    1. Sample N posts from miner's batch
    2. Run LLM classification on each sampled post
    3. Compare miner's canonical string vs validator's canonical string
    4. If all match exactly → accept batch
    5. If any deviate → reject batch
    
    Args:
        miner_batch: List of TweetWithUser objects with keys:
            - text: The post content
            - user: User object
        analyzer: SubnetRelevanceAnalyzer instance
        sample_size: Number of posts to sample for validation (default: 10)
        seed: Random seed for reproducible sampling (optional)
        
    Returns:
        Tuple of (is_valid, result_dict):
            - is_valid: True if all sampled posts match exactly, False otherwise
            - result_dict: Contains 'matches', 'total_sampled', 'discrepancies'
    """
    
    # Sample posts
    if seed is not None:
        random.seed(seed)
    
    sample_size = min(sample_size, len(miner_batch))
    sampled_posts = random.sample(miner_batch, sample_size)
    
    bt.logging.info(f"[Validator] Sampling {sample_size} posts from batch of {len(miner_batch)}")
    
    matches = 0
    discrepancies = []
    
    for i, post_data in enumerate[TweetWithUser](sampled_posts):
        post_text = post_data.text
        miner_classification = post_data.get("miner_classification", {})
        
        # Validator runs classification
        validator_result = analyzer.classify_post(post_text)
        
        if validator_result is None:
            bt.logging.warning(f"[Validator] Failed to classify sampled post {i+1}")
            discrepancies.append({
                "post_index": i,
                "reason": "validator_classification_failed",
                "post_preview": post_text[:100]
            })
            continue
        
        # Build miner's canonical string from their claimed classification
        try:
            miner_canonical = _build_canonical_from_dict(miner_classification)
        except Exception as e:
            bt.logging.warning(f"[Validator] Invalid miner classification format: {e}")
            discrepancies.append({
                "post_index": i,
                "reason": "invalid_miner_format",
                "error": str(e)
            })
            continue
        
        # Get validator's canonical string
        validator_canonical = validator_result.to_canonical_string()
        
        # Exact match check
        if miner_canonical == validator_canonical:
            matches += 1
            bt.logging.debug(f"[Validator] Post {i+1}: MATCH")
        else:
            bt.logging.warning(f"[Validator] Post {i+1}: MISMATCH")
            bt.logging.debug(f"  Miner:     {miner_canonical}")
            bt.logging.debug(f"  Validator: {validator_canonical}")
            discrepancies.append({
                "post_index": i,
                "reason": "canonical_mismatch",
                "miner_canonical": miner_canonical,
                "validator_canonical": validator_canonical,
                "post_preview": post_text[:100]
            })
    
    is_valid = (matches == sample_size) and (len(discrepancies) == 0)
    
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
    post_info: TweetWithUser,
    weights: Dict = None
) -> float:
    """
    Compute final post score combining classification + engagement + recency
    
    Args:
        classification: PostClassification result
        post_info: TweetWithUser object
        weights: Optional custom weights dict
        
    Returns:
        Final score in [0.0, 1.0]
    """
    # Check if post is older than 10 days - if so, return 0
    post_date_str = post_info.created_at.isoformat()
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


def score_post_entry(entry: TweetWithUser, analyzer, k: int = 5, analysis_result: PostClassification = None) -> Dict:
    """
    Score a single post entry with rich classification data preserved.
    
    Returns both the final score AND the full classification object, so downstream
    consumers can use the rich categorical data for database storage, analytics, etc.
    
    Args:
        entry: TweetWithUser object
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
    post_date_str = info.created_at.isoformat()
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

