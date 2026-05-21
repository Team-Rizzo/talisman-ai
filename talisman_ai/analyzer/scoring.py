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

from talisman_ai.utils.api_models import TweetWithAuthor, TelegramMessageForScoring
from .relevance import AssetRelevanceAnalyzer, PostClassification
from .telegram_relevance import TelegramRelevanceAnalyzer, MessageGroupClassification
from .news_relevance import NewsRelevanceAnalyzer, ArticleClassification
from talisman_ai.utils.api_models import NewsArticleForScoring


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
                f"Miner needs to update to the latest talisman-ai code."
            )
            discrepancies.append({
                "post_index": i,
                "reason": "miner_needs_update",
                "message": "Miner is using outdated code. Pull latest talisman-ai and restart.",
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
                f"Miner needs to pull latest talisman-ai and restart."
            )
            discrepancies.append({
                "message_index": i,
                "reason": "miner_needs_update",
                "message": "Miner is using outdated code. Pull latest talisman-ai and restart.",
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
                f"Miner needs to pull latest talisman-ai and restart."
            )
            discrepancies.append({
                "article_index": i,
                "reason": "miner_needs_update",
                "message": "Miner is using outdated code. Pull latest talisman-ai and restart.",
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

