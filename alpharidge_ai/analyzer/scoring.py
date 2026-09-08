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
import os
import random
import threading
import bittensor as bt
import numpy as np

from alpharidge_ai.utils.api_models import TweetWithAuthor, TelegramMessageForScoring
from .relevance import AssetRelevanceAnalyzer, PostClassification
from .telegram_relevance import TelegramRelevanceAnalyzer, MessageGroupClassification
from .news_relevance import NewsRelevanceAnalyzer, ArticleClassification
from alpharidge_ai.utils.api_models import NewsArticleForScoring
from alpharidge_ai.oracle import floor as oracle_floor
from alpharidge_ai.oracle import schema_gate
from alpharidge_ai.models.article_intelligence import ArticleIntelligence


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
    analyzer: AssetRelevanceAnalyzer,
    sample_size: int = 1,
    seed: int = None
) -> Tuple[bool, Dict]:
    """
    Validate a miner's batch by sampling posts and checking classifications.
    
    All fields require exact match:
    asset_id, sentiment, content_type, technical_quality, market_analysis, impact_potential
    
    Miners that haven't updated (missing asset_id) get rejected with an update message.
    
    Args:
        miner_batch: List of TweetWithAuthor objects
        analyzer: AssetRelevanceAnalyzer instance
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
        
        # Grace period: miner hasn't updated to asset_id yet
        if not hasattr(miner_analysis, 'asset_id') or miner_analysis.asset_id is None:
            bt.logging.warning(
                f"[Validator] Post {i+1}: Miner returned subnet_id instead of asset_id. "
                f"Miner needs to update to the latest alpharidge-ai code."
            )
            discrepancies.append({
                "post_index": i,
                "reason": "miner_needs_update",
                "message": "Miner is using outdated code. Pull latest alpharidge-ai and restart.",
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
        
        def _lower(val):
            return val.lower() if isinstance(val, str) else val
        
        m_asset = miner_analysis.asset_id
        m_sent = miner_analysis.sentiment
        m_content = miner_analysis.content_type
        m_tech = miner_analysis.technical_quality
        m_market = miner_analysis.market_analysis
        m_impact = miner_analysis.impact_potential
        
        v_asset = validator_result.asset_id
        v_sent = validator_result.sentiment.value if validator_result.sentiment else None
        v_content = validator_result.content_type.value if validator_result.content_type else None
        v_tech = validator_result.technical_quality.value if validator_result.technical_quality else None
        v_market = validator_result.market_analysis.value if validator_result.market_analysis else None
        v_impact = validator_result.impact_potential.value if validator_result.impact_potential else None
        
        asset_ok = m_asset == v_asset
        sentiment_ok = _lower(m_sent) == _lower(v_sent)
        content_ok = _lower(m_content) == _lower(v_content)
        tech_ok = _lower(m_tech) == _lower(v_tech)
        market_ok = _lower(m_market) == _lower(v_market)
        impact_ok = _lower(m_impact) == _lower(v_impact)
        
        all_ok = asset_ok and sentiment_ok and content_ok and tech_ok and market_ok and impact_ok
        
        if all_ok:
            matches += 1
            bt.logging.debug(f"[Validator] Post {i+1}: MATCH")
        else:
            failed_fields = []
            if not asset_ok:
                failed_fields.append(f"asset_id (miner={m_asset} vs validator={v_asset})")
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
            
            bt.logging.warning(f"[Validator] Post {i+1}: MISMATCH - Failed fields: {', '.join(failed_fields)}")
            bt.logging.warning(f"[Validator] Post {i+1} text preview: {post_text[:200] if post_text else '(empty)'}")
            bt.logging.warning(f"[Validator] Post {i+1} Miner: asset_id={m_asset}, sentiment={m_sent}, content_type={m_content}, tech={m_tech}, market={m_market}, impact={m_impact}")
            bt.logging.warning(f"[Validator] Post {i+1} Validator: asset_id={v_asset}, sentiment={v_sent}, content_type={v_content}, tech={v_tech}, market={v_market}, impact={v_impact}")
            
            discrepancies.append({
                "post_index": i,
                "reason": "classification_mismatch",
                "miner": {
                    "asset_id": m_asset, "sentiment": m_sent, "content_type": m_content,
                    "technical_quality": m_tech, "market_analysis": m_market, "impact_potential": m_impact
                },
                "validator": {
                    "asset_id": v_asset, "sentiment": v_sent, "content_type": v_content,
                    "technical_quality": v_tech, "market_analysis": v_market, "impact_potential": v_impact
                },
                "field_results": {
                    "asset_id": asset_ok, "sentiment": sentiment_ok, "content_type": content_ok,
                    "technical_quality": tech_ok, "market_analysis": market_ok, "impact_potential": impact_ok
                },
                "post_preview": post_text[:100] if post_text else ""
            })
    
    # Stamp the sampled item's resource id onto each discrepancy so penalties can be
    # attributed per-item for the dashboard (display-only). Keyed by the existing
    # post_index; inert if the index is somehow out of range.
    for _d in discrepancies:
        _pi = _d.get("post_index")
        if isinstance(_pi, int) and 0 <= _pi < len(sampled_posts):
            _d["resource_id"] = getattr(sampled_posts[_pi], "id", None)

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


def validate_miner_telegram_batch(
    miner_batch: List[TelegramMessageForScoring],
    analyzer: TelegramRelevanceAnalyzer,
    sample_size: int = 1,
    seed: int = None
) -> Tuple[bool, Dict]:
    """
    Validate a miner's telegram message batch by sampling messages and checking classifications.
    
    All fields require exact match:
    asset_id, sentiment, content_type, technical_quality, market_analysis, impact_potential
    
    Args:
        miner_batch: List of TelegramMessageForScoring objects
        analyzer: TelegramRelevanceAnalyzer instance
        sample_size: Number of messages to sample (default: 1)
        seed: Random seed for reproducible sampling
        
    Returns:
        Tuple of (is_valid, result_dict)
    """
    if seed is not None:
        random.seed(seed)
    
    sample_size = min(sample_size, len(miner_batch))
    sampled_messages = random.sample(miner_batch, sample_size)
    
    bt.logging.info(f"[Validator] Sampling {sample_size} message(s) from telegram batch of {len(miner_batch)}")
    
    matches = 0
    discrepancies = []
    
    for i, msg_data in enumerate(sampled_messages):
        msg_content = msg_data.content
        miner_analysis = msg_data.analysis
        
        if miner_analysis is None:
            bt.logging.warning(f"[Validator] No miner classification for telegram message {i+1}")
            discrepancies.append({
                "message_index": i,
                "reason": "missing_miner_classification",
                "message_preview": msg_content[:100] if msg_content else ""
            })
            continue
        
        if not hasattr(miner_analysis, 'asset_id') or miner_analysis.asset_id is None:
            bt.logging.warning(
                f"[Validator] Telegram message {i+1}: Miner using outdated code (no asset_id). "
                f"Miner needs to pull latest alpharidge-ai and restart."
            )
            discrepancies.append({
                "message_index": i,
                "reason": "miner_needs_update",
                "message": "Miner is using outdated code. Pull latest alpharidge-ai and restart.",
                "message_preview": msg_content[:100] if msg_content else ""
            })
            continue
        
        messages_for_analysis = [{
            'message_id': msg_data.id,
            'username': msg_data.sender_username or msg_data.sender_name,
            'content': msg_data.content,
        }]
        
        if msg_data.context_messages:
            for ctx in msg_data.context_messages:
                messages_for_analysis.insert(0, {
                    'message_id': ctx.id,
                    'username': ctx.sender_username or ctx.sender_name,
                    'content': ctx.content,
                })
        
        inherited_asset_id = msg_data.inherited_asset_id
        
        validator_result = analyzer.classify_message_group(messages_for_analysis, asset_id=inherited_asset_id)
        if validator_result is None:
            bt.logging.warning(f"[Validator] Failed to classify telegram message {i+1}")
            discrepancies.append({
                "message_index": i,
                "reason": "validator_classification_failed",
                "message_preview": msg_content[:100] if msg_content else ""
            })
            continue
        
        def _lower(val):
            return val.lower() if isinstance(val, str) else val
        
        m_asset = miner_analysis.asset_id
        m_sent = miner_analysis.sentiment
        m_content = miner_analysis.content_type
        m_tech = miner_analysis.technical_quality
        m_market = miner_analysis.market_analysis
        m_impact = miner_analysis.impact_potential
        
        v_asset = validator_result.asset_id
        v_sent = validator_result.sentiment.value if validator_result.sentiment else None
        v_content = validator_result.content_type.value if validator_result.content_type else None
        v_tech = validator_result.technical_quality.value if validator_result.technical_quality else None
        v_market = validator_result.market_analysis.value if validator_result.market_analysis else None
        v_impact = validator_result.impact_potential.value if validator_result.impact_potential else None
        
        asset_ok = m_asset == v_asset
        sentiment_ok = _lower(m_sent) == _lower(v_sent)
        content_ok = _lower(m_content) == _lower(v_content)
        tech_ok = _lower(m_tech) == _lower(v_tech)
        market_ok = _lower(m_market) == _lower(v_market)
        impact_ok = _lower(m_impact) == _lower(v_impact)
        
        all_ok = asset_ok and sentiment_ok and content_ok and tech_ok and market_ok and impact_ok
        
        if all_ok:
            matches += 1
            bt.logging.debug(f"[Validator] Telegram message {i+1}: MATCH")
        else:
            failed_fields = []
            if not asset_ok:
                failed_fields.append(f"asset_id (miner={m_asset} vs validator={v_asset})")
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
            
            bt.logging.warning(f"[Validator] Telegram message {i+1}: MISMATCH - Failed fields: {', '.join(failed_fields)}")
            bt.logging.warning(f"[Validator] Telegram message {i+1} text preview: {msg_content[:200] if msg_content else '(empty)'}")
            bt.logging.warning(f"[Validator] Telegram message {i+1} Miner: asset_id={m_asset}, sentiment={m_sent}, content_type={m_content}")
            bt.logging.warning(f"[Validator] Telegram message {i+1} Validator: asset_id={v_asset}, sentiment={v_sent}, content_type={v_content}")
            
            discrepancies.append({
                "message_index": i,
                "reason": "classification_mismatch",
                "miner": {
                    "asset_id": m_asset, "sentiment": m_sent, "content_type": m_content,
                    "technical_quality": m_tech, "market_analysis": m_market, "impact_potential": m_impact
                },
                "validator": {
                    "asset_id": v_asset, "sentiment": v_sent, "content_type": v_content,
                    "technical_quality": v_tech, "market_analysis": v_market, "impact_potential": v_impact
                },
                "field_results": {
                    "asset_id": asset_ok, "sentiment": sentiment_ok, "content_type": content_ok,
                    "technical_quality": tech_ok, "market_analysis": market_ok, "impact_potential": impact_ok
                },
                "message_preview": msg_content[:100] if msg_content else ""
            })
    
    # Stamp the sampled item's resource id onto each discrepancy (display-only
    # attribution). Keyed by the existing message_index.
    for _d in discrepancies:
        _mi = _d.get("message_index")
        if isinstance(_mi, int) and 0 <= _mi < len(sampled_messages):
            _d["resource_id"] = getattr(sampled_messages[_mi], "id", None)

    is_valid = matches == sample_size and len(discrepancies) == 0

    result = {
        "is_valid": is_valid,
        "matches": matches,
        "total_sampled": sample_size,
        "discrepancies": discrepancies,
        "match_rate": matches / sample_size if sample_size > 0 else 0.0
    }

    if is_valid:
        bt.logging.success(f"[Validator] Telegram batch ACCEPTED: {matches}/{sample_size} matches")
    else:
        bt.logging.warning(f"[Validator] Telegram batch REJECTED: {matches}/{sample_size} matches, {len(discrepancies)} discrepancies")

    return is_valid, result


def _build_canonical_from_dict(classification: PostClassification) -> str:
    """
    Build canonical string from classification object.
    
    This must match the exact format from PostClassification.to_canonical_string()
    """
    asset_id = int(classification.asset_id)
    content_type = classification.content_type
    sentiment = classification.sentiment
    technical_quality = classification.technical_quality
    market_analysis = classification.market_analysis
    impact_potential = classification.impact_potential
    relevance_confidence = classification.relevance_confidence
    evidence_spans = classification.evidence_spans
    
    sorted_evidence = "|".join(sorted([s.lower() for s in evidence_spans]))
    
    return f"{asset_id}|{content_type}|{sentiment}|{technical_quality}|{market_analysis}|{impact_potential}|{relevance_confidence}|{sorted_evidence}"


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
    
    relevance = 1.0 if classification.asset_id != 0 else 0.0
    
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
    
    Returns dict mapping asset_symbol -> 1.0 (binary: matched or not).
    """
    if analysis_result is None or analysis_result.asset_id == 0:
        return {}
    return {analysis_result.asset_symbol: 1.0}


def top_k_relevance_from_analyzer(text: str, analyzer, k: int = 5, analysis_result: PostClassification = None) -> Tuple[float, List[Tuple[str, PostClassification]]]:
    """
    Get classification from analyzer and return asset relevance data.
    
    Args:
        text: The post text to analyze (only used if analysis_result is None)
        analyzer: AssetRelevanceAnalyzer instance
        k: Kept for API compatibility
        analysis_result: Optional pre-computed PostClassification.
        
    Returns:
        Tuple of:
            - relevance: Binary relevance (1.0 if matched an asset, 0.0 otherwise)
            - top: List of (asset_symbol, classification_dict) tuples
    """
    if analysis_result is None:
        out = analyzer.analyze_post_complete(text)
    else:
        out = analysis_result
    
    classification = out
    if classification is None or classification.asset_id == 0:
        return 0.0, []
    
    asset_symbol = classification.asset_symbol
    classification_data = classification.to_dict()
    
    return 1.0, [(asset_symbol, classification_data)]


def score_post_entry(entry: TweetWithAuthor, analyzer, k: int = 5, analysis_result: PostClassification = None) -> Dict:
    """
    Score a single post entry with rich classification data preserved.
    
    Args:
        entry: TweetWithAuthor object
        analyzer: AssetRelevanceAnalyzer instance
        k: Kept for API compatibility
        analysis_result: Optional pre-computed PostClassification
        
    Returns:
        Dictionary containing:
            - url: Original post URL/identifier
            - classification: Full PostClassification object (or None)
            - asset_data: Full classification dict for the matched asset
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
            "asset_data": None,
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
            "asset_data": None,
            "relevance": 0.0,
            "value": 0.0,
            "recency": 0.0,
            "score": 0.0
        }

    rel, asset_data = top_k_relevance_from_analyzer(info.text, analyzer, k=k, analysis_result=analysis_result)
    
    if analysis_result is None:
        analysis_result = analyzer.analyze_post_complete(info.text)
    classification = analysis_result
    
    val = value_score(post_info=info)
    rec = recency_score(info.created_at.isoformat())

    final = RELEVANCE_WEIGHT * rel + VALUE_WEIGHT * val + RECENCY_WEIGHT * rec

    return {
        "url": entry.url,
        "classification": classification,
        "asset_data": asset_data[0] if asset_data else None,
        "relevance": rel,
        "value": val,
        "recency": rec,
        "score": max(0.0, min(1.0, float(final)))
    }


# ===== News Article Scoring =====

SOURCE_CREDIBILITY = {
    # Tier 1 — Wire services & papers of record
    "reuters": 1.0, "ap_news": 1.0, "bbc": 1.0, "financial_times": 1.0,
    "wsj": 1.0, "bloomberg": 1.0,
    # Tier 2 — Major broadsheets & established outlets
    "economist": 0.9, "nytimes": 0.9, "washington_post": 0.9,
    # Tier 3 — Respected broadcast/print with editorial depth
    "cnbc": 0.8, "guardian": 0.8, "politico": 0.8, "npr": 0.8,
    "forbes": 0.8, "barrons": 0.8, "abc_news": 0.8, "cbs_news": 0.8,
    "nbc_news": 0.8, "cnn_finance": 0.8, "la_times": 0.8,
    "usa_today": 0.8, "chicago_tribune": 0.8, "the_atlantic": 0.8,
    # Tier 4 — Finance/market-focused & quality tech
    "techcrunch": 0.7, "ars_technica": 0.7, "wired": 0.7,
    "marketwatch": 0.7, "investopedia": 0.7, "nasdaq": 0.7,
    "seeking_alpha": 0.7, "yahoo_finance": 0.7, "sp_global": 0.7,
    "investing_com": 0.7, "business_insider": 0.7, "the_hill": 0.7,
    "propublica": 0.7, "mit_tech_review": 0.7, "vox": 0.7,
    # Tier 5 — Smaller finance/niche outlets
    "motley_fool": 0.6, "benzinga": 0.6, "thestreet": 0.6, "zacks": 0.6,
    "zero_hedge": 0.6, "engadget": 0.6, "gizmodo": 0.6,
    # Tier 6 — Government/institutional sources (high trust, low volume)
    "federal_reserve": 0.9, "sec": 0.9, "treasury": 0.9,
    "brookings": 0.8, "nasa": 0.8, "cdc_newsroom": 0.8,
    # Tier 7 — International outlets
    "al_jazeera": 0.7, "france24": 0.7, "deutsche_welle": 0.7,
    "scmp": 0.7, "nhk_world": 0.7, "japan_times": 0.6, "the_hindu": 0.6,
    "le_monde": 0.7, "der_spiegel": 0.7, "kqed": 0.6,
    # Tier 8 — Science/academic
    "nature_news": 0.8, "scientific_american": 0.7, "new_scientist": 0.7,
    "ieee_spectrum": 0.7, "science_daily": 0.6, "live_science": 0.6,
    "space_com": 0.6,
    # Tier 9 — Culture/niche (lower market relevance)
    "rolling_stone": 0.5, "pitchfork": 0.4, "variety": 0.5,
    "hollywood_reporter": 0.5, "artforum": 0.4, "scotusblog": 0.6,
    "smithsonian": 0.6, "inside_climate": 0.6,
    # Tier 10 — State media (lower editorial independence)
    "rt_news": 0.4,
}


def article_value_score(article: NewsArticleForScoring) -> float:
    """
    Compute value score for a news article based on source credibility and content availability.

    Since articles don't have engagement metrics (likes/retweets), value is derived from:
    1. Source credibility (60% weight)
    2. Content availability (40% weight)

    Args:
        article: NewsArticleForScoring object

    Returns:
        Value score in [0.0, 1.0]
    """
    source_cred = SOURCE_CREDIBILITY.get(article.source, 0.5)
    content_score = 1.0 if article.content else 0.5
    return 0.6 * source_cred + 0.4 * content_score


def compute_article_score(
    classification: ArticleClassification,
    article: NewsArticleForScoring,
    weights: Dict = None
) -> float:
    """
    Compute final article score combining classification + source credibility + recency

    Args:
        classification: ArticleClassification result
        article: NewsArticleForScoring object
        weights: Optional custom weights dict

    Returns:
        Final score in [0.0, 1.0]
    """
    if not article.published:
        return 0.0

    published_str = article.published if isinstance(article.published, str) else article.published.isoformat()
    dt = datetime.fromisoformat(published_str.replace("Z", "+00:00")).astimezone(timezone.utc)
    age_days = (datetime.now(timezone.utc) - dt).total_seconds() / (3600.0 * 24.0)
    if age_days > 10:
        return 0.0

    if weights is None:
        weights = {
            "relevance": 0.50,
            "value": 0.40,
            "recency": 0.10,
        }

    # sector_id 9 = Other (irrelevant)
    relevance = 1.0 if classification.sector_id != 9 else 0.0

    val = article_value_score(article)

    rec = recency_score(article.published)

    final = weights["relevance"] * relevance + weights["value"] * val + weights["recency"] * rec

    return _clamp01(final)


def validate_miner_article_batch(
    miner_batch: List[NewsArticleForScoring],
    analyzer: NewsRelevanceAnalyzer,
    sample_size: int = 1,
    seed: int = None
) -> Tuple[bool, Dict]:
    """
    Validate a miner's news article batch by sampling articles and checking classifications.

    All fields require exact match:
    sector_id, sentiment, content_type, technical_quality, market_analysis, impact_potential

    Args:
        miner_batch: List of NewsArticleForScoring objects
        analyzer: NewsRelevanceAnalyzer instance
        sample_size: Number of articles to sample (default: 1)
        seed: Random seed for reproducible sampling

    Returns:
        Tuple of (is_valid, result_dict)
    """
    if seed is not None:
        random.seed(seed)

    sample_size = min(sample_size, len(miner_batch))
    sampled_articles = random.sample(miner_batch, sample_size)

    bt.logging.info(f"[Validator] Sampling {sample_size} article(s) from batch of {len(miner_batch)}")

    matches = 0
    discrepancies = []

    for i, article in enumerate(sampled_articles):
        article_preview = article.title[:100]
        miner_analysis = article.analysis

        if miner_analysis is None:
            bt.logging.warning(f"[Validator] No miner classification for article {i+1}")
            discrepancies.append({
                "article_index": i,
                "reason": "missing_miner_classification",
                "article_preview": article_preview
            })
            continue

        # Grace period: miner hasn't updated to sector_id yet
        if not hasattr(miner_analysis, 'sector_id') or miner_analysis.sector_id is None:
            bt.logging.warning(
                f"[Validator] Article {i+1}: Miner using outdated code (no sector_id). "
                f"Miner needs to pull latest alpharidge-ai and restart."
            )
            discrepancies.append({
                "article_index": i,
                "reason": "miner_needs_update",
                "message": "Miner is using outdated code. Pull latest alpharidge-ai and restart.",
                "article_preview": article_preview
            })
            continue

        validator_result = analyzer.classify_article(article.title, article.summary, article.content)
        if validator_result is None:
            bt.logging.warning(f"[Validator] Failed to classify article {i+1}")
            discrepancies.append({
                "article_index": i,
                "reason": "validator_classification_failed",
                "article_preview": article_preview
            })
            continue

        def _lower(val):
            return val.lower() if isinstance(val, str) else val

        m_sector = miner_analysis.sector_id
        m_sent = miner_analysis.sentiment
        m_content = miner_analysis.content_type
        m_tech = miner_analysis.technical_quality
        m_market = miner_analysis.market_analysis
        m_impact = miner_analysis.impact_potential

        v_sector = validator_result.sector_id
        v_sent = validator_result.sentiment.value if validator_result.sentiment else None
        v_content = validator_result.content_type.value if validator_result.content_type else None
        v_tech = validator_result.technical_quality.value if validator_result.technical_quality else None
        v_market = validator_result.market_analysis.value if validator_result.market_analysis else None
        v_impact = validator_result.impact_potential.value if validator_result.impact_potential else None

        sector_ok = m_sector == v_sector
        sentiment_ok = _lower(m_sent) == _lower(v_sent)
        content_ok = _lower(m_content) == _lower(v_content)
        tech_ok = _lower(m_tech) == _lower(v_tech)
        market_ok = _lower(m_market) == _lower(v_market)
        impact_ok = _lower(m_impact) == _lower(v_impact)

        all_ok = sector_ok and sentiment_ok and content_ok and tech_ok and market_ok and impact_ok

        if all_ok:
            matches += 1
            bt.logging.debug(f"[Validator] Article {i+1}: MATCH")
        else:
            failed_fields = []
            if not sector_ok:
                failed_fields.append(f"sector_id (miner={m_sector} vs validator={v_sector})")
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

            bt.logging.warning(f"[Validator] Article {i+1}: MISMATCH - Failed fields: {', '.join(failed_fields)}")
            bt.logging.warning(f"[Validator] Article {i+1} title preview: {article_preview}")
            bt.logging.warning(f"[Validator] Article {i+1} Miner: sector_id={m_sector}, sentiment={m_sent}, content_type={m_content}, tech={m_tech}, market={m_market}, impact={m_impact}")
            bt.logging.warning(f"[Validator] Article {i+1} Validator: sector_id={v_sector}, sentiment={v_sent}, content_type={v_content}, tech={v_tech}, market={v_market}, impact={v_impact}")

            discrepancies.append({
                "article_index": i,
                "reason": "classification_mismatch",
                "miner": {
                    "sector_id": m_sector, "sentiment": m_sent, "content_type": m_content,
                    "technical_quality": m_tech, "market_analysis": m_market, "impact_potential": m_impact
                },
                "validator": {
                    "sector_id": v_sector, "sentiment": v_sent, "content_type": v_content,
                    "technical_quality": v_tech, "market_analysis": v_market, "impact_potential": v_impact
                },
                "field_results": {
                    "sector_id": sector_ok, "sentiment": sentiment_ok, "content_type": content_ok,
                    "technical_quality": tech_ok, "market_analysis": market_ok, "impact_potential": impact_ok
                },
                "article_preview": article_preview
            })

    # Stamp the sampled item's resource id onto each discrepancy (display-only
    # attribution). Keyed by the existing article_index.
    for _d in discrepancies:
        _ai = _d.get("article_index")
        if isinstance(_ai, int) and 0 <= _ai < len(sampled_articles):
            _d["resource_id"] = getattr(sampled_articles[_ai], "id", None)

    is_valid = matches == sample_size and len(discrepancies) == 0

    result = {
        "is_valid": is_valid,
        "matches": matches,
        "total_sampled": sample_size,
        "discrepancies": discrepancies,
        "match_rate": matches / sample_size if sample_size > 0 else 0.0
    }

    if is_valid:
        bt.logging.success(f"[Validator] Article batch ACCEPTED: {matches}/{sample_size} matches")
    else:
        bt.logging.warning(f"[Validator] Article batch REJECTED: {matches}/{sample_size} matches, {len(discrepancies)} discrepancies")

    return is_valid, result


# ============================================================================
# V2: ArticleIntelligence 4-Tier Validation
# ============================================================================


def _jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


def _levenshtein_ratio(s1: str, s2: str) -> float:
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    len1, len2 = len(s1), len(s2)
    if len1 > len2:
        s1, s2 = s2, s1
        len1, len2 = len2, len1
    prev = list(range(len1 + 1))
    for j in range(1, len2 + 1):
        curr = [j] + [0] * len1
        for i in range(1, len1 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            curr[i] = min(curr[i - 1] + 1, prev[i] + 1, prev[i - 1] + cost)
        prev = curr
    dist = prev[len1]
    max_len = max(len1, len2)
    return 1.0 - (dist / max_len) if max_len > 0 else 1.0


def _normalize_text(text: str) -> str:
    t = text.lower().strip()
    t = " ".join(t.split())
    for article in (" a ", " an ", " the "):
        t = t.replace(article, " ")
    return " ".join(t.split())


# ---- Tolerant scoring for subjective LLM-derived enums ----
# These fields come from the LLM and are NOT deterministic across two independent
# temp=0 calls (miner vs validator), so they are scored with partial credit inside
# the Tier-3 composite rather than hard-gated. Ordinal enums use distance-based
# partial credit; nominal enums use exact (1.0/0.0) within the averaged group.
_SENTIMENT_LADDER = ["very_bullish", "bullish", "slightly_bullish", "neutral",
                     "slightly_bearish", "bearish", "very_bearish"]
_SENTIMENT_DIRECTION_LADDER = ["positive", "neutral", "negative"]  # "mixed" -> nominal
_IMPACT_LADDER = ["critical", "high", "medium", "low", "negligible"]
_TECH_QUALITY_LADDER = ["exceptional", "high", "medium", "low", "none"]
_URGENCY_LADDER = ["flash", "breaking", "developing", "same_day", "evergreen"]
_FACTUAL_CONF_LADDER = ["confirmed", "attributed", "speculative", "conditional", "rumor"]


def _ev(x) -> str:
    """Enum value as lowercase string, robust to enum or raw-string inputs."""
    return (x.value if hasattr(x, "value") else str(x)).lower()


# Tier-3 composite floor a miner's analysis must clear. This is a QUALITY floor,
# NOT the anti-cheat line (the deterministic Tier-1/2/2.5 gates are). Calibrated on
# an n~920 honest deepseek-vs-deepseek sweep (scripts/sweep_analyze.py): honest
# composites have P1~0.61 / median ~0.865, and at the old 0.75 ~8% of honest
# single-article samples were rejected on pure LLM free-text variance
# (chart_summary/event_fingerprint differ between two independent temp-0 calls).
# 0.70 cuts honest false-reject to ~4% while a skip-LLM miner still fails ~70% of
# samples, so consistent failers are still sorted out. Env-overridable for tuning.
TIER3_THRESHOLD = float(os.getenv("TIER3_THRESHOLD", "0.70"))


def _ordinal_score(pred: str, gold: str, ladder: List[str]) -> float:
    """1.0 on exact match; otherwise 1 - normalized ladder distance. 0.0 if off-ladder."""
    if pred == gold:
        return 1.0
    if pred not in ladder or gold not in ladder:
        return 0.0
    dist = abs(ladder.index(pred) - ladder.index(gold))
    return max(0.0, 1.0 - dist / (len(ladder) - 1))


# ── Tolerant per-asset sentiment determinism ───────────────────────────────────
# Miner and validator run the same off-LLM sentiment models, but cross-hardware
# neural inference is NOT bit-identical, so the per-asset direction/outlook can
# differ by a class at a decision boundary (e.g. bullish vs slightly_bullish).
# We compare on the ordinal sentiment ladder with adjacent-class tolerance and
# average across the batch's common assets.
# (_SENTIMENT_LADDER is defined once above with the other tolerant-scoring ladders.)
_SENTIMENT_IDX = {v: i for i, v in enumerate(_SENTIMENT_LADDER)}

# Mean per-asset agreement threshold. Env-overridable for field tuning.
DET_AGREEMENT_THRESHOLD = float(os.getenv("DET_AGREEMENT_THRESHOLD", "0.80"))


def _sentiment_agreement(a: str, b: str) -> float:
    """Agreement in [0,1] between two 7-class sentiment labels on the ordinal
    ladder. Adjacent classes (distance <=1) are inherent cross-hardware jitter and
    count as full agreement; credit then decays linearly to 0.0 at a full sign
    reversal (very_bullish vs very_bearish). Off-ladder labels fall back to exact
    match."""
    ia, ib = _SENTIMENT_IDX.get(a), _SENTIMENT_IDX.get(b)
    if ia is None or ib is None:
        return 1.0 if a == b else 0.0
    dist = abs(ia - ib)
    if dist <= 1:
        return 1.0
    # distance 2..6 -> credit 1.0..0.0 (denominator = max distance minus the
    # one free adjacent step)
    return max(0.0, 1.0 - (dist - 1) / (len(_SENTIMENT_LADDER) - 2))


# Minimum validator-resolved asset count above which a miner submitting ZERO
# assets is treated as skipping the Tier-2b gate (anti-cheat). Env-overridable so
# testnet can raise it if cross-hardware NER ever erases a lone borderline asset.
ASSET_PRESENCE_FLOOR = int(os.getenv("ASSET_PRESENCE_FLOOR", "1"))

# Asset-resolution source that is bit-identical across hardware: ONLY the gazetteer
# keyword extractor (asset_extractor.extract_assets — pure string matching, no
# neural dependency). Everything else is excluded from the presence floor:
# "refined"/model are neural, and "override" — though a deterministic dict lookup —
# only fires on a candidate span the neural NER first surfaced (_resolve_entity runs
# on grounded candidates), so its trigger inherits the same cross-host variance.
# Counting any of them would let a validator-only neural/override asset hard-fail an
# honest miner whose NER didn't surface that span.
_DETERMINISTIC_ASSET_SOURCES = {"keyword"}


def asset_presence_ok(m_assets: dict, v_assets: dict, floor: int = None) -> bool:
    """False (-> hard fail) iff the validator resolved >= `floor` *deterministic*
    assets but the miner submitted none.

    The Tier-2b agreement gate is skipped when there are no common assets, so a
    miner that sends zero assets bypasses it for only the Tier-3 partial-credit
    cost — the same omission free-pass §2.3 closed for embeddings. Asymmetric: a
    validator that itself resolved nothing never penalizes the miner.

    Only the deterministic (gazetteer/override) subset of the validator's assets
    counts toward the floor: those are bit-identical on both sides, so a total
    miner-absence against them is a genuine skip signal. Neural-NER-resolved assets
    are excluded because they can legitimately differ across hardware, and failing
    an honest miner over a lone divergent neural asset would be worse than the
    bypass we're closing.
    """
    if floor is None:
        floor = ASSET_PRESENCE_FLOOR
    det_v = sum(1 for a in v_assets.values()
                if getattr(a, "resolved_via", "keyword") in _DETERMINISTIC_ASSET_SOURCES)
    return not (det_v >= floor and len(m_assets) == 0)


def asset_sentiment_agreement(m_assets: dict, v_assets: dict) -> Tuple[float, int]:
    """Mean per-asset sentiment agreement over tickers common to both sides.

    Each asset's score averages agreement on direction + the three horizon
    outlooks. Returns (agreement, n_common); n_common == 0 -> (1.0, 0) since there
    is nothing to disagree on.
    """
    common = set(m_assets) & set(v_assets)
    if not common:
        return 1.0, 0
    per_asset = []
    for t in common:
        ma, va = m_assets[t], v_assets[t]
        fields = [
            (_ev(ma.direction), _ev(va.direction)),
            (_ev(ma.short_term_outlook), _ev(va.short_term_outlook)),
            (_ev(ma.medium_term_outlook), _ev(va.medium_term_outlook)),
            (_ev(ma.long_term_outlook), _ev(va.long_term_outlook)),
        ]
        per_asset.append(sum(_sentiment_agreement(x, y) for x, y in fields) / len(fields))
    return sum(per_asset) / len(per_asset), len(common)


def validate_article_intelligence(
    miner_intel: ArticleIntelligence,
    validator_intel: ArticleIntelligence,
) -> Tuple[bool, float, Dict]:
    """4-tier validation of ArticleIntelligence objects.

    Returns (is_valid, composite_score, details_dict).
    """
    m, v = miner_intel, validator_intel
    details = {"tier1": {}, "tier2": {}, "tier3": {}}

    # ---- Tier 1: DETERMINISTIC fields only (exact match, HARD GATE) ----
    # Empirically (deepseek-vs-deepseek), NO LLM-generated enum survives an exact-match
    # gate across two independent temp=0 calls — even "objective" ones like content_type
    # and primary_geo vary. So only fields computed deterministically by both sides
    # (date-derived session, text-derived language, gazetteer-derived sector) are gated;
    # all LLM enums are scored with partial credit in Tier 3.
    # detected_language is gated on the en/non-en BUCKET, not the exact ISO code.
    # langdetect is probabilistic and short text (RSS titles) can disagree on the
    # specific non-English code across hosts even at a pinned version; the only
    # thing the code actually drives is the English vs multilingual NER route, so
    # that bucket is the consensus-relevant decision.
    def _lang_bucket(x) -> str:
        return "en" if str(x).lower().strip() == "en" else "non-en"

    # market_session is intentionally NOT a Tier-1 hard-gate field: it is trivially derived
    # from the timestamp (zero independent quality signal and absent from the Tier-3
    # composite), and a timezone-representation mismatch at the Fri/Sat boundary was
    # systematically failing honest batches on exact-match. The field's own derivation is
    # normalized in _compute_market_session; it stays ungated here as low-value.
    tier1_fields = [
        ("detected_language", _lang_bucket(m.detected_language), _lang_bucket(v.detected_language)),
        ("primary_sector_id", str(m.topic_signature.primary_sector_id), str(v.topic_signature.primary_sector_id)),
    ]

    tier1_pass = True
    for field_name, m_val, v_val in tier1_fields:
        match = (m_val == v_val)
        details["tier1"][field_name] = {"match": match, "miner": m_val, "validator": v_val}
        if not match:
            tier1_pass = False
            bt.logging.warning(f"[V2_VALIDATE] Tier 1 FAIL: {field_name} miner={m_val} validator={v_val}")

    if not tier1_pass:
        return False, 0.0, details

    # ---- Tier 2: deterministic content + off-LLM ML fields (HARD GATE) ----
    tier2_checks = [
        ("content_hash", m.event_fingerprint.content_hash, v.event_fingerprint.content_hash),
        ("word_count", m.text_stats.word_count, v.text_stats.word_count),
        ("sentence_count", m.text_stats.sentence_count, v.text_stats.sentence_count),
        ("char_count", m.text_stats.char_count, v.text_stats.char_count),
        ("ticker_mention_count", m.text_stats.ticker_mention_count, v.text_stats.ticker_mention_count),
    ]

    tier2_pass = True
    for field_name, m_val, v_val in tier2_checks:
        match = (m_val == v_val)
        details["tier2"][field_name] = {"match": match, "miner": m_val, "validator": v_val}
        if not match:
            tier2_pass = False
            bt.logging.warning(f"[V2_VALIDATE] Tier 2 FAIL: {field_name} miner={m_val} validator={v_val}")

    if not tier2_pass:
        return False, 0.0, details

    # Tier 2b: off-LLM ML fields. Miner and validator run the same sentiment
    # models, but cross-hardware neural inference is NOT bit-identical, so per-asset
    # direction/outlook can differ by a class at a decision boundary. Exact-match
    # is therefore unachievable AND, at the 1-6 assets an article resolves, a rate
    # threshold collapses to exact-match (one of two assets disagreeing = 0.50).
    # Instead require the MEAN ordinal agreement across the batch's common assets to
    # clear a threshold, tolerating adjacent-class jitter.
    m_assets = {a.ticker: a for a in m.assets}
    v_assets = {a.ticker: a for a in v.assets}
    # Anti-cheat presence check (asymmetric, mirrors the Tier-2.5 embedding check):
    # if the validator resolved assets but the miner submitted none, the miner is
    # skipping the agreement gate below (which is a no-op with no common assets).
    if not asset_presence_ok(m_assets, v_assets):
        details["tier2"]["asset_presence"] = {"status": "fail", "reason": "miner_missing_assets",
                                               "validator_assets": len(v_assets), "miner_assets": 0}
        bt.logging.warning(f"[V2_VALIDATE] Tier 2 FAIL: miner submitted 0 assets but validator "
                           f"resolved {len(v_assets)} (asset-gate bypass)")
        return False, 0.0, details
    agreement, n_common = asset_sentiment_agreement(m_assets, v_assets)
    if n_common:
        details["tier2"]["asset_sentiment_determinism"] = {
            "agreement": round(agreement, 4), "common_assets": n_common,
            "threshold": DET_AGREEMENT_THRESHOLD}
        if agreement < DET_AGREEMENT_THRESHOLD:
            bt.logging.warning(f"[V2_VALIDATE] Tier 2 FAIL: asset sentiment agreement "
                               f"{agreement:.3f} < {DET_AGREEMENT_THRESHOLD} over {n_common} asset(s)")
            return False, 0.0, details

    DETERMINISM_TOL = 0.9
    m_cont = {(l.source_ticker, l.target_ticker) for l in m.contagion_links}
    v_cont = {(l.source_ticker, l.target_ticker) for l in v.contagion_links}
    cont_jac = _jaccard(m_cont, v_cont)
    details["tier2"]["contagion_determinism"] = {"jaccard": round(cont_jac, 4)}
    if cont_jac < DETERMINISM_TOL:
        bt.logging.warning(f"[V2_VALIDATE] Tier 2 FAIL: contagion determinism jaccard={cont_jac:.2f}")
        return False, 0.0, details

    # ---- Tier 2.5: Embedding verification ----
    # Format checks (dim / non-zero / normalized) are anti-cheat HARD GATES for both
    # embeddings. Cosine similarity is a HARD GATE only for title_embedding (its input,
    # the title, is deterministic so it must match). narrative_embedding is built from
    # LLM-generated headline+keywords, which vary call-to-call, so its cosine is SCORED
    # in Tier 3 rather than gated — gating it zeroed out honest miners.
    details["tier2_5"] = {}
    EMBEDDING_DIM = 384

    def _emb_ok(emb) -> bool:
        return bool(emb) and len(emb) == EMBEDDING_DIM

    # Required-embedding presence (anti-cheat). If the validator produced an
    # embedding (the normal case) but the miner did not, the miner is trying to
    # skip the title cosine gate and bank free narrative_embedding credit — HARD
    # FAIL. Asymmetric on purpose: when the validator itself lacks an embedding
    # (our embedder is down), we don't penalize the miner.
    for _name, _m_emb, _v_emb in (
        ("title_embedding", m.title_embedding, v.title_embedding),
        ("narrative_embedding", m.narrative_embedding, v.narrative_embedding),
    ):
        if _emb_ok(_v_emb) and not _emb_ok(_m_emb):
            details["tier2_5"][_name] = {"status": "fail", "reason": "miner_missing"}
            bt.logging.warning(f"[V2_VALIDATE] Tier 2.5 FAIL: {_name} absent/malformed on miner")
            return False, 0.0, details

    def _embedding_cosine(name, m_emb, v_emb):
        """Return cosine sim, or None if absent. Hard-fail (raise) on bad format."""
        if not (m_emb and v_emb and len(m_emb) == EMBEDDING_DIM and len(v_emb) == EMBEDDING_DIM):
            return None
        m_norm = float(np.linalg.norm(m_emb))
        v_norm = float(np.linalg.norm(v_emb))
        if m_norm < 0.01 or v_norm < 0.01:
            details["tier2_5"][name] = {"status": "fail", "reason": "zero_vector"}
            raise ValueError("zero_vector")
        if abs(m_norm - 1.0) > 0.05 or abs(v_norm - 1.0) > 0.05:
            details["tier2_5"][name] = {"status": "fail", "reason": "not_normalized",
                                        "m_norm": round(m_norm, 4), "v_norm": round(v_norm, 4)}
            raise ValueError("not_normalized")
        sim = float(np.dot(m_emb, v_emb))
        details["tier2_5"][name] = {"sim": round(sim, 4)}
        return sim

    try:
        title_sim_emb = _embedding_cosine("title_embedding", m.title_embedding, v.title_embedding)
        narr_sim_emb = _embedding_cosine("narrative_embedding", m.narrative_embedding, v.narrative_embedding)
    except ValueError as e:
        bt.logging.warning(f"[V2_VALIDATE] Tier 2.5 FAIL: embedding format {e}")
        return False, 0.0, details

    # title_embedding cosine IS gated (deterministic input). title_sim_emb is None
    # only when the VALIDATOR lacks the embedding (miner-missing already hard-failed
    # in the presence check above), so a None here is our outage -> skip, don't penalize.
    if title_sim_emb is not None and title_sim_emb < 0.90:
        details["tier2_5"]["title_embedding"]["threshold"] = 0.90
        bt.logging.warning(f"[V2_VALIDATE] Tier 2.5 FAIL: title_embedding cosine={title_sim_emb:.4f} < 0.90")
        return False, 0.0, details

    # ---- Tier 3: tolerant scoring (subjective LLM enums + text/extraction) ----
    tier3_scores = {}

    # (a) Subjective classification enums — partial credit. Ordinal enums use
    # ladder distance; nominal enums use exact match. Averaged into one component
    # so no single borderline disagreement can fail an otherwise-honest miner.
    enum_scores = []
    enum_detail = {}

    def _score_enum(name, ladder=None):
        mv, vv = _ev(getattr(m, name)), _ev(getattr(v, name))
        s = _ordinal_score(mv, vv, ladder) if ladder else (1.0 if mv == vv else 0.0)
        enum_scores.append(s)
        enum_detail[name] = {"score": round(s, 4), "miner": mv, "validator": vv}

    _score_enum("overall_sentiment", _SENTIMENT_LADDER)
    _score_enum("sentiment_direction", _SENTIMENT_DIRECTION_LADDER)
    _score_enum("impact_potential", _IMPACT_LADDER)
    _score_enum("technical_quality", _TECH_QUALITY_LADDER)
    _score_enum("urgency", _URGENCY_LADDER)
    _score_enum("factual_confidence", _FACTUAL_CONF_LADDER)
    _score_enum("temporal_focus")
    _score_enum("market_analysis_type")
    _score_enum("positioning_signal")
    _score_enum("target_audience")
    _score_enum("forward_event_type")
    _score_enum("staleness_flag")
    _score_enum("credibility_flag")
    _score_enum("content_type")      # LLM, but objective-ish -> nominal partial credit
    _score_enum("primary_geo")       # LLM, varies across calls -> nominal partial credit

    def _score_pair(name, mv, vv):
        s = 1.0 if mv == vv else 0.0
        enum_scores.append(s)
        enum_detail[name] = {"score": s, "miner": mv, "validator": vv}

    # Nested / LLM-extracted nominal fields.
    _score_pair("event_type", _ev(m.event_fingerprint.event_type), _ev(v.event_fingerprint.event_type))
    _score_pair("event_date", m.event_fingerprint.event_date or "none", v.event_fingerprint.event_date or "none")
    tier3_scores["classification_enums"] = sum(enum_scores) / len(enum_scores)
    details["tier3_enums"] = enum_detail

    # (b) Extraction + free-text components (kept from prior contract).
    m_tickers = {a.ticker for a in m.assets}
    v_tickers = {a.ticker for a in v.assets}
    tier3_scores["asset_extraction"] = _jaccard(m_tickers, v_tickers)

    headline_sim = _levenshtein_ratio(
        _normalize_text(m.chart_summary.headline), _normalize_text(v.chart_summary.headline))
    oneliner_sim = _levenshtein_ratio(
        _normalize_text(m.chart_summary.one_liner), _normalize_text(v.chart_summary.one_liner))
    paragraph_sim = _levenshtein_ratio(
        _normalize_text(m.chart_summary.context_paragraph), _normalize_text(v.chart_summary.context_paragraph))
    tier3_scores["chart_summary"] = 0.4 * headline_sim + 0.3 * oneliner_sim + 0.3 * paragraph_sim

    m_entities = {e.name.lower() for e in m.entities}
    v_entities = {e.name.lower() for e in v.entities}
    tier3_scores["entities"] = _jaccard(m_entities, v_entities)

    if m.economic_data or v.economic_data:
        m_data = {(d.event_type.value, round(d.actual_value or 0, 1)) for d in m.economic_data}
        v_data = {(d.event_type.value, round(d.actual_value or 0, 1)) for d in v.economic_data}
        tier3_scores["economic_data"] = _jaccard(m_data, v_data)
    else:
        tier3_scores["economic_data"] = 1.0

    title_sim = _levenshtein_ratio(
        _normalize_text(m.event_fingerprint.event_title), _normalize_text(v.event_fingerprint.event_title))
    fp_sim = _jaccard(set(m.event_fingerprint.semantic_fingerprint), set(v.event_fingerprint.semantic_fingerprint))
    tier3_scores["event_fingerprint"] = 0.5 * title_sim + 0.5 * fp_sim

    m_narr = {kw.lower() for kw in m.narrative_keywords}
    v_narr = {kw.lower() for kw in v.narrative_keywords}
    tier3_scores["narrative_keywords"] = _jaccard(m_narr, v_narr)

    # narrative_embedding cosine: scored (not gated) since its LLM-text input varies.
    # Map [0,1] cosine straight through; clamp negatives to 0. Miner-missing already
    # hard-failed in the Tier-2.5 presence check, so a None here means the VALIDATOR
    # lacked the embedding (our outage) -> neutral 1.0 rather than penalize the miner.
    tier3_scores["narrative_embedding"] = max(0.0, narr_sim_emb) if narr_sim_emb is not None else 1.0

    # Asset sentiment + contagion are NOT scored here — they are deterministic and
    # already hard-gated in Tier 2b. Weights re-normalized to sum to 1.0.
    weights = {
        "classification_enums": 0.42, "asset_extraction": 0.15, "chart_summary": 0.10,
        "entities": 0.08, "economic_data": 0.07, "event_fingerprint": 0.08,
        "narrative_keywords": 0.05, "narrative_embedding": 0.05,
    }
    composite = sum(tier3_scores[k] * weights[k] for k in weights)
    details["tier3"] = {k: {"score": round(tier3_scores[k], 4), "weight": weights[k]} for k in weights}
    details["tier3"]["composite"] = round(composite, 4)

    # Quality floor. Read live from config (served subnet-wide / OVERRIDE-able) so it
    # can be recalibrated without a code change AND stays identical across validators;
    # falls back to the module default (recalibrated 0.75 -> 0.70 on an n~920 honest
    # sweep) if config is unavailable.
    try:
        from alpharidge_ai import config as _cfg
        _threshold = float(getattr(_cfg, "TIER3_THRESHOLD", TIER3_THRESHOLD))
    except Exception:
        _threshold = TIER3_THRESHOLD
    is_valid = composite >= _threshold
    if is_valid:
        bt.logging.success(f"[V2_VALIDATE] Article ACCEPTED: composite={composite:.4f}")
    else:
        bt.logging.warning(f"[V2_VALIDATE] Article REJECTED: composite={composite:.4f} < {_threshold}")

    return is_valid, composite, details


def _cfg_get(name: str, default):
    """Read a remote-config-served value live (falls back to the module default)."""
    try:
        from alpharidge_ai import config as _cfg
        return getattr(_cfg, name, default)
    except Exception:
        return default


def _same_source_article(a, b) -> bool:
    """True if two batch items are the same underlying article (duplicate input,
    not a clone). Matches on URL, falling back to the analysis content_hash."""
    ua, ub = getattr(a, "url", None), getattr(b, "url", None)
    if ua and ub and ua == ub:
        return True

    def _content_hash(x):
        ad = getattr(getattr(x, "analysis", None), "analysis_data", None)
        if isinstance(ad, dict):
            fp = ad.get("event_fingerprint")
            if isinstance(fp, dict):
                return fp.get("content_hash")
        return None

    ha, hb = _content_hash(a), _content_hash(b)
    return bool(ha and hb and ha == hb)


def _validator_title_embedding(analyzer, article):
    """The validator's own normalized title embedding for a batch item, via the local
    embedder (cheap, no LLM). Returns None if unavailable — caller then falls back to
    the conservative absolute clone rule for that pair."""
    try:
        title = getattr(article, "title", None)
        engine = getattr(analyzer, "ner_engine", None)
        if not title or engine is None or not hasattr(engine, "encode_text"):
            return None
        emb = engine.encode_text(title)
        if not emb:
            return None
        v = np.array(emb, dtype=np.float32)
        norm = float(np.linalg.norm(v))
        return v / norm if norm > 0 else None
    except Exception:
        return None


# Failure classes for an article batch. Capacity = the miner hasn't finished the work
# we dispatched (a burst out-pacing its analyzer) — not cheating. Validator-side = OUR
# analyzer/classifier failed. Everything else is a genuine integrity failure.
_CAPACITY_REASONS = {
    "missing_analysis",               # V2: sampled article carried no analysis object
    "no_v2_analysis_data",            # V2: sampled article carried no analysis_data dict
    "missing_miner_classification",   # V1: nothing submitted yet
    "miner_needs_update",             # V1: stale miner build (version, not cheating)
}
_VALIDATOR_SIDE_REASONS = {
    "validator_analysis_failed",      # V2: our analyzer.analyze returned None
    "validator_classification_failed",  # V1: our classifier failed
}


def classify_article_batch_failure(discrepancies) -> str:
    """Bucket a failed batch's discrepancies into 'integrity' | 'capacity' |
    'validator_side'. ANY genuine wrong/cloned discrepancy makes the whole batch
    integrity (a real cheat is never excused by an un-analyzed sibling article).
    'invalid_analysis_data: <exc>' is prefix-matched to capacity (incomplete/unparseable
    write — earns no credit, so there's no exploit in not penalizing it)."""
    reasons = [d.get("reason", "") for d in (discrepancies or [])]

    def _capacity(r):
        return r in _CAPACITY_REASONS or r.startswith("invalid_analysis_data")

    def _validator_side(r):
        return r in _VALIDATOR_SIDE_REASONS

    if any(not _capacity(r) and not _validator_side(r) for r in reasons):
        return "integrity"
    if reasons and all(_validator_side(r) for r in reasons):
        return "validator_side"
    return "capacity"


def _titles_match(record, reference) -> bool:
    if reference is None:
        return True
    blob = getattr(getattr(record, "analysis", None), "analysis_data", None)
    a = _normalize_text((blob.get("title") if isinstance(blob, dict) else "") or "")
    b = _normalize_text(getattr(reference, "title", "") or "")
    return a != "" and a == b


_ml_embedder = None
_ml_embedder_lock = threading.Lock()


def _get_ml_embedder():
    global _ml_embedder
    if _ml_embedder is None:
        with _ml_embedder_lock:
            if _ml_embedder is None:
                from sentence_transformers import SentenceTransformer
                _ml_embedder = SentenceTransformer(
                    "paraphrase-multilingual-MiniLM-L12-v2", device="cpu")
    return _ml_embedder


def _summary_text(intel):
    cs = getattr(intel, "chart_summary", None)
    if cs is None:
        return ""
    h = getattr(cs, "headline", "") or ""
    o = getattr(cs, "one_liner", "") or ""
    return (str(h) + " " + str(o)).strip()


def _summary_agreement(record_intel, reference_intel):
    a = _summary_text(record_intel)
    b = _summary_text(reference_intel)
    if len(a) < 3 or len(b) < 3:
        return None
    try:
        emb = _get_ml_embedder().encode([a, b], normalize_embeddings=True)
        return float(np.dot(emb[0], emb[1]))
    except Exception as e:
        bt.logging.warning(f"[V2_VALIDATE] summary comparison unavailable: {e}")
        return None


# Sectors that carry plausible market transmission under rubric R2 even when no
# asset resolves and no economic release is extracted — tariffs, sanctions and
# supply shocks classify here. An article in one of these is never treated as
# clearly irrelevant, so honest miners keeping it are not charged for it and it
# never becomes a negative canary.
from alpharidge_ai.triage import MARKET_ADJACENT_SECTORS as _MARKET_ADJACENT_SECTORS


def _reference_clearly_irrelevant(intel) -> bool:
    """True only when the validator's own reference analysis says the article is
    unambiguously outside the rubric.

    Deliberately one-sided: this drives false-positive charges and negative
    canary selection, both of which punish miners, so anything defensible —
    any asset, any economic datum, any market-adjacent sector — returns False.
    """
    if getattr(intel, "assets", None) or getattr(intel, "economic_data", None):
        return False
    sector = getattr(getattr(intel, "topic_signature", None),
                     "primary_sector_symbol", None)
    return sector not in _MARKET_ADJACENT_SECTORS


def _floor_sweep(miner_batch, reference_by_id=None, *, block: int = 0,
                 schema_cutover_block: int = 0) -> Dict[int, bool]:
    """Run the deterministic floor over every article in a batch.

    Returns {article_id: passed}. An article whose analysis will not even load fails;
    an article the validator has no reference text for is left out rather than failed,
    since that is the validator's own gap.
    """
    results: Dict[int, bool] = {}
    for article in miner_batch or []:
        try:
            aid = int(getattr(article, "id", 0))
        except (TypeError, ValueError):
            continue

        ref = (reference_by_id or {}).get(str(getattr(article, "id", "")))
        text = getattr(ref or article, "content", None)
        if not text:
            continue

        blob = getattr(getattr(article, "analysis", None), "analysis_data", None)
        if not blob or not isinstance(blob, dict):
            results[aid] = False
            continue
        try:
            intel = ArticleIntelligence(**blob)
        except Exception:
            results[aid] = False
            continue

        # After the cutover block a submission on the old schema earns nothing.
        if schema_cutover_block:
            verdict = schema_gate.evaluate(
                getattr(intel, "schema_version", None), block=int(block),
                cutover_block=int(schema_cutover_block), intel=intel)
            if not verdict.accepted:
                bt.logging.info(
                    f"[FLOOR] id={aid} rejected: {verdict.reason}")
                results[aid] = False
                continue

        title = getattr(ref or article, "title", None) or ""
        claimed_hash = getattr(getattr(intel, "event_fingerprint", None),
                               "content_hash", None)
        try:
            result = oracle_floor.evaluate(
                intel, text, claimed_hash=claimed_hash,
                expected_hash=ArticleIntelligence.compute_content_hash(title, text))
            results[aid] = bool(result.floor_pass)
        except Exception as e:
            bt.logging.debug(f"[FLOOR] evaluation failed id={aid}: {e}")
    return results


def validate_miner_article_intelligence_batch(
    miner_batch: List[NewsArticleForScoring],
    analyzer,
    sample_size: int = 1,
    seed: int = None,
    graded_scorer=None,
    reference_by_id=None,
    miner_hotkey=None,
    auditor=None,
    block=0,
    schema_cutover_block=0,
) -> Tuple[bool, Dict]:
    """Validate a miner's article batch using V2 4-tier validation.

    Falls back to V1 if analysis_data is missing. When `graded_scorer` is provided, a
    continuous per-sampled-article observation is also collected (result["observations"]);
    this is independent of the pass/fail result and has no effect on it.
    """
    if reference_by_id is not None:
        mismatches = []
        for i, a in enumerate(miner_batch):
            ref = reference_by_id.get(str(getattr(a, "id", "")))
            if ref is None:
                continue
            if not _titles_match(a, ref):
                blob = getattr(getattr(a, "analysis", None), "analysis_data", None)
                claimed = (blob.get("title") if isinstance(blob, dict) else None) or ""
                mismatches.append({
                    "article_index": i,
                    "resource_id": str(getattr(a, "id", "")),
                    "reason": "article_content_mismatch",
                    "article_preview": (getattr(ref, "title", "") or "")[:100],
                    "miner": {"title": str(claimed)[:80]},
                    "validator": {"title": (getattr(ref, "title", "") or "")[:80]},
                })
        if mismatches:
            return False, {"discrepancies": mismatches, "match_rate": 0.0, "observations": []}

    if seed is not None:
        random.seed(seed)

    sample_size = min(sample_size, len(miner_batch))
    sampled = random.sample(miner_batch, sample_size)

    bt.logging.info(f"[V2_VALIDATE] Sampling {sample_size} article(s) from batch of {len(miner_batch)}")

    # Per-article floor over the WHOLE batch, not just the sample. Cheap and
    # deterministic, so volume can be credited per article that clears it rather
    # than on one verdict for the batch.
    floor_results = _floor_sweep(miner_batch, reference_by_id, block=block,
                                 schema_cutover_block=schema_cutover_block)

    matches = 0
    total_composite = 0.0
    discrepancies = []
    observations = []  # (article_id, graded, weight) when graded_scorer is set
    audit_observations = []  # keyed-audit results, shadow only
    faithfulness_scores = []  # reference-free faithfulness per sampled article
    reference_irrelevant = []  # (article_id, bool) — True when our own reference
    # says the article is clearly outside the rubric; feeds triage FP events
    # and negative-canary selection

    for i, article in enumerate(sampled):
        miner_analysis = article.analysis
        if miner_analysis is None:
            discrepancies.append({"article_index": i, "reason": "missing_analysis"})
            continue

        analysis_data = getattr(miner_analysis, "analysis_data", None)
        if not analysis_data or not isinstance(analysis_data, dict):
            discrepancies.append({"article_index": i, "reason": "no_v2_analysis_data"})
            continue

        try:
            miner_intel = ArticleIntelligence(**analysis_data)
        except Exception as e:
            discrepancies.append({"article_index": i, "reason": f"invalid_analysis_data: {e}"})
            continue

        ref = (reference_by_id or {}).get(str(getattr(article, "id", "")))
        src = ref or article
        validator_intel = analyzer.analyze(
            article_id=article.id,
            url=src.url,
            title=src.title,
            source=src.source,
            published=src.published,
            summary=src.summary,
            content=src.content,
            raw_html=getattr(src, "raw_html", None),
        )
        if validator_intel is None:
            discrepancies.append({"article_index": i, "reason": "validator_analysis_failed"})
            continue

        try:
            reference_irrelevant.append(
                (int(article.id), _reference_clearly_irrelevant(validator_intel)))
        except Exception:
            pass

        if ref is not None:
            agreement = _summary_agreement(miner_intel, validator_intel)
            if agreement is not None:
                bt.logging.info(f"[V2_VALIDATE] summary_agreement={agreement:.3f} id={getattr(article, 'id', '')} hk={miner_hotkey}")
                if agreement < float(_cfg_get("SUMMARY_AGREEMENT_FLOOR", 0.4)):
                    discrepancies.append({
                        "article_index": i,
                        "resource_id": str(getattr(article, "id", "")),
                        "reason": "article_content_mismatch",
                        "article_preview": (getattr(src, "title", "") or "")[:100],
                        "summary_agreement": agreement,   # display-only: how close vs the 0.40 floor
                    })
                    continue

        is_valid, composite, details = validate_article_intelligence(miner_intel, validator_intel)
        if is_valid:
            matches += 1
            total_composite += composite
        else:
            discrepancies.append({
                "article_index": i, "reason": "validation_failed",
                "resource_id": str(getattr(article, "id", "")),
                "composite_score": composite, "details": details,
            })

        if auditor is not None:
            # Runs only on articles already analysed here, so it adds no reference
            # analysis of its own. Full keyed coverage needs selection to drive the
            # analysis, which is the change that carries the grader load.
            try:
                text = getattr(src, "content", None) or ""
                seen = _floor_sweep([article], reference_by_id).get(int(article.id))
                if seen:
                    result = oracle_floor.evaluate(miner_intel, text)
                    observed = auditor.audit(int(article.id), text, miner_intel,
                                             validator_intel, result, int(block))
                    if observed is not None:
                        audit_observations.append(observed)
            except Exception as e:
                bt.logging.debug(f"[AUDIT] failed on {getattr(article, 'id', '?')}: {e}")

        if graded_scorer is not None:
            try:
                g, w = graded_scorer.score(miner_intel, validator_intel, article)
                observations.append((int(article.id), float(g), float(w)))
            except Exception as e:
                bt.logging.warning(f"[REPUTATION] graded scoring failed: {e}")
            # Reference-free faithfulness vs the reference article (src). Skip when the
            # reference has too little content to score it reliably.
            src_content = (getattr(src, "content", None) or "").strip()
            if len(src_content) >= int(_cfg_get("FAITHFULNESS_MIN_CONTENT_CHARS", 200)):
                try:
                    faithfulness_scores.append(
                        float(graded_scorer.faithfulness(miner_intel, src)))
                except Exception as e:
                    bt.logging.warning(f"[FAITHFULNESS] scoring failed: {e}")
            else:
                bt.logging.debug(
                    f"[FAITHFULNESS] skipped id={getattr(src, 'id', '?')} — content too short")

    # The random sample above decides batch acceptance; quality auditing is selected
    # by key. Capped per batch so one batch cannot run unbounded analyses.
    if auditor is not None:
        already = {int(getattr(a, "id", 0)) for a in sampled}
        cap = int(_cfg_get("AUDIT_MAX_PER_BATCH", 4))
        picked = 0
        for article in miner_batch:
            if picked >= cap:
                bt.logging.debug(
                    f"[AUDIT] per-batch cap {cap} reached; "
                    f"{len(miner_batch) - len(already)} article(s) left unwatched")
                break
            try:
                aid = int(getattr(article, "id", 0))
            except (TypeError, ValueError):
                continue
            if aid in already:
                continue

            ref = (reference_by_id or {}).get(str(getattr(article, "id", "")))
            src = ref or article
            text = getattr(src, "content", None) or ""
            if not text or not auditor.selects(aid, text, int(block)):
                continue

            blob = getattr(getattr(article, "analysis", None), "analysis_data", None)
            if not blob or not isinstance(blob, dict):
                continue
            try:
                miner_intel = ArticleIntelligence(**blob)
            except Exception:
                continue
            if not _floor_sweep([article], reference_by_id, block=block,
                                schema_cutover_block=schema_cutover_block).get(aid):
                continue

            picked += 1
            try:
                validator_intel = analyzer.analyze(
                    article_id=article.id, url=src.url, title=src.title,
                    source=src.source, published=src.published, summary=src.summary,
                    content=src.content, raw_html=getattr(src, "raw_html", None))
                if validator_intel is None:
                    continue
                result = oracle_floor.evaluate(miner_intel, text)
                observed = auditor.audit(aid, text, miner_intel, validator_intel,
                                         result, int(block))
                if observed is not None:
                    audit_observations.append(observed)
            except Exception as e:
                bt.logging.debug(f"[AUDIT] keyed pass failed on {aid}: {e}")

    # Cross-article adversarial detection: cloned embeddings.
    # Legacy rule (default): flag any within-batch title-embedding pair with cosine
    # above the threshold. When CLONE_DIFFERENTIAL_ENABLED is served on, a candidate
    # pair is only a clone if the miner's similarity exceeds the validator's own
    # re-embedding of the same titles by >= margin — honest syndicated clusters (which
    # the validator also sees as similar) no longer false-reject.
    EMBEDDING_DIM = 384
    clone_threshold = float(_cfg_get("CLONE_COSINE_THRESHOLD", 0.99))
    clone_differential = bool(_cfg_get("CLONE_DIFFERENTIAL_ENABLED", False))
    clone_margin = float(_cfg_get("CLONE_DIVERGENCE_MARGIN", 0.05))

    indexed = []  # (batch_index, article, normalized_miner_embedding)
    for idx, article in enumerate(miner_batch):
        ad = getattr(getattr(article, "analysis", None), "analysis_data", None)
        if ad and isinstance(ad, dict):
            te = ad.get("title_embedding")
            if te and isinstance(te, list) and len(te) == EMBEDDING_DIM:
                indexed.append((idx, article, np.array(te, dtype=np.float32)))

    if len(indexed) >= 3:
        for a in range(len(indexed)):
            for b in range(a + 1, len(indexed)):
                miner_cos = float(indexed[a][2] @ indexed[b][2])
                if miner_cos <= clone_threshold:
                    continue
                if clone_differential:
                    ai, bi = indexed[a][0], indexed[b][0]
                    # A duplicate of the SAME underlying article isn't cloning.
                    if _same_source_article(indexed[a][1], indexed[b][1]):
                        bt.logging.info(f"[V2_VALIDATE] clone candidate suppressed: articles {ai},{bi} "
                                        f"miner_cos={miner_cos:.4f} (same source article)")
                        continue
                    # Corroborate against the validator's own embeddings of the titles.
                    v_a = _validator_title_embedding(analyzer, indexed[a][1])
                    v_b = _validator_title_embedding(analyzer, indexed[b][1])
                    if v_a is not None and v_b is not None:
                        validator_cos = float(np.dot(v_a, v_b))
                        if (miner_cos - validator_cos) < clone_margin:
                            # validator agrees they're similar -> honest syndicate, not a clone
                            bt.logging.info(f"[V2_VALIDATE] clone candidate suppressed: articles {ai},{bi} "
                                            f"miner_cos={miner_cos:.4f} validator_cos={validator_cos:.4f} "
                                            f"delta={miner_cos - validator_cos:.4f} (validator corroborates)")
                            continue
                bt.logging.warning(f"[V2_VALIDATE] Adversarial: articles {indexed[a][0]} and "
                                   f"{indexed[b][0]} have cloned embeddings (miner_cos={miner_cos:.4f})")
                discrepancies.append({
                    "reason": "cloned_embeddings",
                    "articles": [indexed[a][0], indexed[b][0]],
                    "cosine": miner_cos,
                })

    batch_valid = matches == sample_size and len(discrepancies) == 0
    avg_composite = total_composite / max(matches, 1)

    result = {
        "is_valid": batch_valid, "matches": matches, "total_sampled": sample_size,
        "avg_composite_score": round(avg_composite, 4), "discrepancies": discrepancies,
        "observations": observations,
        "faithfulness_scores": faithfulness_scores,
        "reference_irrelevant": reference_irrelevant,
        "floor_results": floor_results,
        "audit_observations": audit_observations,
    }

    if batch_valid:
        bt.logging.success(f"[V2_VALIDATE] Batch ACCEPTED: {matches}/{sample_size}, avg={avg_composite:.4f}")
    else:
        bt.logging.warning(f"[V2_VALIDATE] Batch REJECTED: {matches}/{sample_size}")

    return batch_valid, result

