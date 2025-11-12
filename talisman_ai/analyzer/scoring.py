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

from relevance import SubnetRelevanceAnalyzer, PostClassification


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
    like_count: int,
    retweet_count: int,
    quote_count: int,
    reply_count: int,
    author_followers: int,
    account_age_days: int,
    caps: Dict = CAPS,
) -> float:
    """
    Compute value score based on engagement metrics and author credibility
    
    The value score is an equal-weight average of six normalized components:
    1-4. Engagement signals (likes, retweets, quotes, replies)
    5-6. Author credibility (follower count, account age)
    
    Args:
        like_count: Number of likes on the tweet
        retweet_count: Number of retweets
        quote_count: Number of quote tweets
        reply_count: Number of replies
        author_followers: Number of followers the author has
        account_age_days: Age of the author's account in days
        caps: Dictionary of cap values for normalization (defaults to CAPS)
        
    Returns:
        Value score in [0.0, 1.0]
    """
    comps = [
        _norm(like_count or 0, caps["likes"]),
        _norm(retweet_count or 0, caps["retweets"]),
        _norm(quote_count or 0, caps["quotes"]),
        _norm(reply_count or 0, caps["replies"]),
        _norm(author_followers or 0, caps["followers"]),
        _norm(account_age_days or 0, caps["account_age_days"]),
    ]
    return sum(comps) / len(comps)


# ===== Validator Batch Verification =====

def validate_miner_batch(
    miner_batch: List[Dict],
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
        miner_batch: List of post dicts with keys:
            - post_text: The post content
            - miner_classification: Dict with miner's claimed classification
                - Must contain all fields to build canonical string
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
    
    for i, post_data in enumerate(sampled_posts):
        post_text = post_data.get("post_text", "")
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


def _build_canonical_from_dict(classification: Dict) -> str:
    """
    Build canonical string from miner's classification dictionary
    
    This must match the exact format from PostClassification.to_canonical_string()
    
    Args:
        classification: Dict with keys matching PostClassification fields
        
    Returns:
        Canonical string for exact matching
    """
    # Extract fields
    subnet_id = int(classification["subnet_id"])
    content_type = classification["content_type"]
    sentiment = classification["sentiment"]
    technical_quality = classification["technical_quality"]
    market_analysis = classification["market_analysis"]
    impact_potential = classification["impact_potential"]
    relevance_confidence = classification["relevance_confidence"]
    evidence_spans = classification.get("evidence_spans", [])
    anchors_detected = classification.get("anchors_detected", [])
    
    # Sort evidence for determinism (same as PostClassification)
    sorted_evidence = "|".join(sorted([s.lower() for s in evidence_spans]))
    sorted_anchors = "|".join(sorted([s.lower() for s in anchors_detected]))
    
    return f"{subnet_id}|{content_type}|{sentiment}|{technical_quality}|{market_analysis}|{impact_potential}|{relevance_confidence}|{sorted_evidence}|{sorted_anchors}"


# ===== Scoring Weights =====
# Default weights for production scoring (used by score_tweet_entry)
# These weights prioritize relevance over value, with recency as a minor factor
RELEVANCE_WEIGHT = 0.50  # 50% weight on subnet relevance
VALUE_WEIGHT = 0.40      # 40% weight on signal value/quality
RECENCY_WEIGHT = 0.10    # 10% weight on recency


def compute_post_score(
    classification: PostClassification,
    post_info: Dict,
    weights: Dict = None
) -> float:
    """
    Compute final post score combining classification + engagement + recency
    
    Args:
        classification: PostClassification result
        post_info: Dict with engagement metrics and post_date
        weights: Optional custom weights dict
        
    Returns:
        Final score in [0.0, 1.0]
    """
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
        like_count=post_info.get("like_count", 0) or 0,
        retweet_count=post_info.get("retweet_count", 0) or 0,
        quote_count=post_info.get("quote_count", 0) or 0,
        reply_count=post_info.get("reply_count", 0) or 0,
        author_followers=post_info.get("author_followers", 0) or 0,
        account_age_days=post_info.get("account_age_days", 0) or 0,
    )
    
    # Recency
    rec = recency_score(post_info.get("post_date", datetime.now(timezone.utc).isoformat()))
    
    # Combine
    final = weights["relevance"] * relevance + weights["value"] * val + weights["recency"] * rec
    
    return _clamp01(final)


# ===== Backward Compatibility Functions for Legacy Code =====

def top_k_relevance_from_analyzer(text: str, analyzer, k: int = 5, analysis_result: Dict = None) -> Tuple[float, List[Tuple[str, float]]]:
    """
    Compute subnet relevance scores using the analyzer and return top-k results.
    
    This function uses the SubnetRelevanceAnalyzer to determine how relevant a post
    is to different BitTensor subnets, then returns the mean relevance score of the
    top-k most relevant subnets.
    
    Args:
        text: The post text to analyze (only used if analysis_result is None)
        analyzer: SubnetRelevanceAnalyzer instance configured with subnet registry
        k: Number of top subnets to consider when computing the mean (default: 5)
        analysis_result: Optional pre-computed analysis result dict. If provided,
                        this will be used instead of calling analyze_tweet_complete again.
        
    Returns:
        Tuple of:
            - mean_top: Mean relevance score of top-k subnets [0.0, 1.0]
            - top: List of (subnet_name, relevance_score) tuples for top-k subnets,
                   sorted by relevance (highest first)
    """
    if analysis_result is None:
        out = analyzer.analyze_tweet_complete(text)
    else:
        out = analysis_result
    items = [(name, data.get("relevance", 0.0)) for name, data in out.get("subnet_relevance", {}).items()]
    items.sort(key=lambda x: x[1], reverse=True)
    top = items[:k]
    mean_top = float(sum(s for _, s in top) / len(top)) if top else 0.0
    return mean_top, top


def score_tweet_entry(entry: Dict, analyzer, k: int = 5, analysis_result: Dict = None) -> Dict:
    """
    Score a single tweet/post entry using LLM-based subnet relevance analysis.
    
    This is the primary scoring function for production use. It computes three
    component scores (relevance, value, recency) and combines them into a final
    score using the default weights defined above.
    
    The relevance component uses an LLM-based analyzer to determine how relevant
    the post is to BitTensor subnets, providing more accurate results than
    simple keyword matching.
    
    Args:
        entry: Dictionary containing tweet entry with keys:
            - url: Tweet URL or identifier
            - tweet_info: Dictionary with tweet metadata containing:
                - tweet_text: The tweet text content
                - tweet_date: ISO format date string
                - like_count: Number of likes (optional, defaults to 0)
                - retweet_count: Number of retweets (optional, defaults to 0)
                - quote_count: Number of quote tweets (optional, defaults to 0)
                - reply_count: Number of replies (optional, defaults to 0)
                - author_followers: Author's follower count (optional, defaults to 0)
                - account_age_days: Author's account age in days (optional, defaults to 0)
        analyzer: SubnetRelevanceAnalyzer instance configured with subnet registry
        k: Number of top subnets to consider when computing relevance (default: 5)
        analysis_result: Optional pre-computed analysis result dict. If provided,
                        this will be used instead of calling analyze_tweet_complete again.
        
    Returns:
        Dictionary containing:
            - url: Original tweet URL/identifier
            - top_subnets: List of (subnet_name, relevance_score) tuples for top-k subnets
            - relevance: Mean relevance score of top-k subnets [0.0, 1.0]
            - value: Value score based on engagement and author credibility [0.0, 1.0]
            - recency: Recency score based on tweet age [0.0, 1.0]
            - score: Final weighted score combining all components [0.0, 1.0]
    """
    info = entry["tweet_info"]

    # Compute component scores
    rel_mean, rel_top = top_k_relevance_from_analyzer(info["tweet_text"], analyzer, k=k, analysis_result=analysis_result)
    val = value_score(
        like_count=info.get("like_count", 0) or 0,
        retweet_count=info.get("retweet_count", 0) or 0,
        quote_count=info.get("quote_count", 0) or 0,
        reply_count=info.get("reply_count", 0) or 0,
        author_followers=info.get("author_followers", 0) or 0,
        account_age_days=info.get("account_age_days", 0) or 0,
    )
    rec = recency_score(info["tweet_date"])

    # Combine components using default weights
    final = RELEVANCE_WEIGHT * rel_mean + VALUE_WEIGHT * val + RECENCY_WEIGHT * rec

    return {
        "url": entry.get("url", ""),
        "top_subnets": rel_top,
        "relevance": rel_mean,
        "value": val,
        "recency": rec,
        "score": max(0.0, min(1.0, float(final)))
    }

