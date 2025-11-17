
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timezone
import math

import bittensor as bt
import tweepy
from dateutil.parser import isoparse
from talisman_ai.analyzer import setup_analyzer
from talisman_ai.analyzer.scoring import score_post_entry
from talisman_ai import config
from talisman_ai.utils.normalization import norm_text
from talisman_ai.validator import x_api_client, sn13_api_client

# =============================================================================
# Miner-Facing Overview
# =============================================================================
# This file validates a batch of your posts and then scores you.
#
# Philosophy (plain language):
# - Deterministic: everyone is checked the same way, in the same order.
# - Simple: each post is validated step-by-step; we stop at the first failure.
# - Public-friendly: tolerances and checks are explicit and documented below.
# - Ground truth: live X (Twitter) API + validator's analyzer. The X API is REQUIRED.
#
# What you must submit per post:
#   post_id, content, author, date (unix seconds), likes, retweets, replies,
#   followers, tokens (dict[str->float]), sentiment (float), score (float).
#
# Validation order (for each post, stop on first error):
#   1) X API check (post exists and data matches)
#      - content text matches exactly after normalization
#      - author username matches exactly (lowercase)
#      - timestamp must match exactly (Unix seconds)
#      - likes/retweets/replies/followers are NOT overstated beyond tolerance
#        (you may understate; you may NOT overstate beyond 10% or +1, whichever is larger)
#   2) Content analysis check (validator analyzes your text)
#      - tokens must match validator's tokens within ±0.05 absolute
#      - sentiment must match within ±0.05 absolute
#   3) Score check
#      - validator computes its own score using the same algorithm
#      - your score may be <= validator_score + 0.05; larger -> error
#
# Scoring:
#   - The validator validates sampled posts to determine VALID/INVALID
#   - If VALID, the validator uses avg_score_all_posts (API-calculated average of ALL posts)
#   - This grader only validates posts; it does not calculate final scores
#
# Return shape:
#   - If a post fails: (CONSENSUS_INVALID, { "error": {...}, "final_score": 0.0 })
#     Error includes code, message, failing post_index and post_id, plus details.
#   - If all pass: (CONSENSUS_VALID, { n_posts, tolerances, analyzer })
# Keep reading comments inline to see exactly how each check works.
# =============================================================================

# === Constants (publicly documented tolerances) ===
CONSENSUS_VALID, CONSENSUS_INVALID = 1, 0
TOKEN_TOLERANCE = 0.05           # tokens in [0,1] must be within ±0.05 of validator
SENTIMENT_TOLERANCE = 0.05       # sentiment in [-1,1] must be within ±0.05 of validator
SCORE_TOLERANCE = 0.05           # your per-post score may not exceed validator by > 0.05
POST_METRIC_TOLERANCE = 0.1      # 10% relative (with a floor of 1) for overstatement checks

# === Error helper: standardizes INVALID responses ===
def _err(code: str, message: str, post_id=None, details=None, post_index: Optional[int] = None):
    e = {"code": code, "message": message, "post_id": post_id, "details": details or {}}
    if post_index is not None:
        e["post_index"] = post_index
    return CONSENSUS_INVALID, {"error": e, "final_score": 0.0}

# === Analyzer/X client creation ===
def make_analyzer():
    """Create the validator's analyzer. If this fails, grading cannot proceed."""
    try:
        return setup_analyzer()
    except Exception as e:
        bt.logging.error(f"[GRADER] Analyzer init failed: {e}")
        return None

def make_x_client():
    """
    Create the appropriate API client based on X_API_SOURCE config.
    Returns a client object with a fetch_post() method.
    """
    api_source = getattr(config, "X_API_SOURCE", "sn13_api")
    
    if api_source == "sn13_api":
        return sn13_api_client.create_client()
    else:
        return x_api_client.create_client()

# === Metric tolerance and inflation rules ===
def metric_tol(live: int) -> int:
    """Tolerance for likes/retweets/replies/followers overstatement: max(1, ceil(10% of live))."""
    return 1 if live == 0 else max(1, math.ceil(live * POST_METRIC_TOLERANCE))

def metric_inflated(miner: int, live: int) -> bool:
    """True if your value > live + tolerance. Understatement is allowed."""
    return miner > live + metric_tol(live)

# === Utilities ===
def iso_from_unix(ts: int) -> str:
    """Convert unix seconds -> ISO-8601 (UTC) for the scoring function."""
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat()

def analyze_text(text: str, analyzer) -> Tuple[Dict[str, float], float]:
    """Run validator analysis on your text -> (tokens, sentiment)."""
    if analyzer is None:
        return {}, 0.0
    try:
        out = analyzer.analyze_post_complete(text)
    except Exception as e:
        bt.logging.error(f"[GRADER] Analyzer error: {e}")
        return {}, 0.0
    tokens_raw = (out.get("subnet_relevance") or {})
    # Keys are normalized here so later comparisons are deterministic.
    tokens = {str(k).strip().lower(): float(v.get("relevance", 0.0)) for k, v in tokens_raw.items()}
    sentiment = float(out.get("sentiment", 0.0))
    return tokens, sentiment

def normalize_keys(d: Dict) -> Dict[str, float]:
    """Normalize dict keys: string, stripped, lowercase; values -> float."""
    return {str(k).strip().lower(): float(v) for k, v in (d or {}).items()}

def select_tokens(miner_raw: Dict, ref_raw: Dict, k: int = 128, eps: float = 0.05) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Token selection policy used for comparison:
      - Remove tiny values (< eps) on both sides (treated as noise).
      - Always keep ALL validator tokens.
      - Add additional miner tokens (largest magnitude first) until we reach cap k.
    This prevents you from dropping validator-relevant tokens via truncation.
    """
    mt = {k: v for k, v in normalize_keys(miner_raw).items() if abs(v) >= eps}
    rt = {k: v for k, v in normalize_keys(ref_raw).items() if abs(v) >= eps}
    keep = set(rt.keys())
    extras = sorted((set(mt) - keep), key=lambda x: (-abs(mt[x]), x))
    for x in extras:
        if len(keep) >= k:
            break
        keep.add(x)
    return {k: mt.get(k, 0.0) for k in keep}, rt

def tokens_match_within(miner: Dict[str, float], ref: Dict[str, float], abs_tol: float, eps: float = 0.05) -> Tuple[bool, Dict]:
    """
    Compare tokens with absolute tolerance. Pairs below eps on BOTH sides are ignored.
    Returns (ok, diffs). diffs shows where you exceed tolerance.
    """
    diffs = {}
    for k in (set(miner) | set(ref)):
        a, b = float(miner.get(k, 0.0)), float(ref.get(k, 0.0))
        if a < eps and b < eps:  # ignore noise
            continue
        if abs(a - b) > abs_tol:
            diffs[k] = {"miner": a, "validator": b, "allowed": abs_tol, "diff": abs(a - b)}
    return (len(diffs) == 0, diffs)

def compute_validator_score(content: str, date_iso: str, likes: int, retweets: int, replies: int,
                            account_age_days: int, followers_from_x: int, analyzer, analysis_result: Dict = None) -> float:
    """
    Validator score uses the same open scoring routine miners should use.
    Inputs:
      - text, date
      - public metrics (likes, retweets, replies)
      - author metrics (followers, account_age_days)
      - analysis_result: Optional pre-computed analysis to avoid duplicate LLM calls
    """
    entry = {
        "url": "post",
        "post_info": {
            "post_text": content,
            "post_date": date_iso,
            "like_count": int(likes or 0),
            "retweet_count": int(retweets or 0),
            "quote_count": 0,
            "reply_count": int(replies or 0),
            "author_followers": int(followers_from_x or 0),
            "account_age_days": int(account_age_days or 0),
        },
    }
    try:
        # Pass analysis_result to avoid re-analyzing the same text
        scored = score_post_entry(entry, analyzer, k=5, analysis_result=analysis_result)
        return float(scored.get("score", 0.0))
    except Exception as e:
        # We surface this upstream as an INVALID result with code=score_compute_error.
        raise RuntimeError(f"score_compute_error: {e}")


# === Core post-level validation against X ===
def validate_with_x(post: Dict, x_client) -> Tuple[Optional[Dict], Dict]:
    """
    Returns (error_dict, live_metrics) where:
      - error_dict is None if the post passes all X-related checks
      - live_metrics contains metrics/timestamps/author info used for scoring
    """
    # 0) post_id must exist
    post_id = post.get("post_id")
    if not post_id:
        return ({"code": "missing_post_id", "message": "post_id is required", "post_id": None, "details": {}}, {})

    # 1) Fetch from X or SN13 API
    try:
        post_record = x_client.fetch_post(post_id)
    except Exception as e:
        return ({"code": "x_api_error", "message": f"API error: {e}", "post_id": post_id, "details": {}}, {})
    if post_record is None:
        return ({"code": "x_api_no_response", "message": "API gave no response after retries", "post_id": post_id, "details": {}}, {})

    # 2) Text must match exactly after normalization (NFC, whitespace normalized)
    miner_text = (post.get("content") or "")
    live_text = post_record.text or ""
    if norm_text(miner_text) != norm_text(live_text):
        return ({"code": "text_mismatch", "message": "content does not match live post text (after normalization)",
                 "post_id": post_id, "details": {"miner": miner_text[:100], "live": live_text[:100], "preview_len": 100}}, {})

    # 3) Author must match (lowercase usernames)
    miner_author = (post.get("author") or "").strip().lower()
    live_author = (post_record.author.username if post_record.author else "").strip().lower()
    if miner_author != live_author:
        return ({"code": "author_mismatch", "message": "author does not match", "post_id": post_id,
                 "details": {"miner": post.get("author", ""), "live": post_record.author.username if post_record.author else ""}}, {})

    # 4) Timestamp must match exactly (Unix seconds)
    miner_ts = post.get("date") or post.get("timestamp")
    if miner_ts is None:
        bt.logging.error(f"[GRADER] Unexpected: miner timestamp None after API validation (post_id={post_id})")
        return ({"code": "timestamp_missing", "message": "timestamp is missing (API validation should have caught this)", "post_id": post_id, "details": {}}, {})
    miner_ts = int(miner_ts)
    if not post_record.created_at:
        return ({"code": "missing_created_at", "message": "live post missing created_at from X API", "post_id": post_id, "details": {}}, {})
    live_ts = int(post_record.created_at.timestamp())
    if miner_ts != live_ts:
        return ({"code": "timestamp_mismatch", "message": "timestamp must match exactly",
                 "post_id": post_id, "details": {"miner": miner_ts, "live": live_ts, "diff_seconds": abs(live_ts - miner_ts)}}, {})

    # 5) Engagement/author metrics may NOT be overstated beyond tolerance
    live_likes = post_record.public_metrics.like_count
    live_rts = post_record.public_metrics.retweet_count
    live_replies = post_record.public_metrics.reply_count
    m_likes = int(post.get("likes") or 0)
    m_rts = int(post.get("retweets") or 0)
    m_replies = int((post.get("replies") if post.get("replies") is not None else post.get("responses")) or 0)

    if metric_inflated(m_likes, live_likes):
        return ({"code": "metric_inflation_likes", "message": "likes overstated beyond tolerance",
                 "post_id": post_id, "details": {"miner": m_likes, "live": live_likes, "tolerance": metric_tol(live_likes)}}, {})
    if metric_inflated(m_rts, live_rts):
        return ({"code": "metric_inflation_retweets", "message": "retweets overstated beyond tolerance",
                 "post_id": post_id, "details": {"miner": m_rts, "live": live_rts, "tolerance": metric_tol(live_rts)}}, {})
    if metric_inflated(m_replies, live_replies):
        return ({"code": "metric_inflation_replies", "message": "replies overstated beyond tolerance",
                 "post_id": post_id, "details": {"miner": m_replies, "live": live_replies, "tolerance": metric_tol(live_replies)}}, {})

    followers = post_record.author.followers_count if post_record.author else 0
    m_followers = int(post.get("followers") or 0)
    if metric_inflated(m_followers, followers):
        return ({"code": "metric_inflation_followers", "message": "followers overstated beyond tolerance",
                 "post_id": post_id, "details": {"miner": m_followers, "live": followers, "tolerance": metric_tol(followers)}}, {})

    # 6) Account age is excluded from grading since SN13 API doesn't provide author created_at
    # Always set to 0 for consistency across X API and SN13 API
    account_age_days = 0

    # Success: pass back live data for the scoring stage
    return (None, {
        "followers": followers,
        "likes": live_likes,
        "retweets": live_rts,
        "replies": live_replies,
        "created_at": live_ts,
        "account_age_days": account_age_days,
    })

# === Batch-level grading entry point ===
def grade_hotkey(posts: List[Dict], analyzer=None, x_client=None) -> Tuple[int, Dict]:
    """
    Grade a list of posts using LLM validation only (tokens and sentiment).
    X API validation is now done on the API side before posts are sent to validators.
    
    Stops on first failure; otherwise returns VALID.
    """
    # Basic sanity
    if not posts:
        return _err("no_posts", "no posts submitted")
    try:
        analyzer = analyzer or make_analyzer()
        if analyzer is None:
            return _err("analyzer_unavailable", "Analyzer not initialized")
    except Exception as e:
        return _err("analyzer_unavailable", str(e))

    for i, post in enumerate(posts):
        post_id = post.get("post_id")
        if not post_id:
            return _err("missing_post_id", "post_id is required", None, {}, i)

        # --- LLM validation: content analysis (tokens and sentiment) ---
        # Note: X API validation is now done on the API side, so we only validate
        # the LLM analysis results (tokens/relevance and sentiment)
        content = post.get("content") or ""
        if not content:
            return _err("empty_content", "post content is empty", post_id, {}, i)

        miner_tokens_raw = post.get("tokens") or {}
        miner_sent = float(post.get("sentiment") or 0.0)
        
        # Get analysis result from LLM
        try:
            analysis_result = analyzer.analyze_post_complete(content)
        except Exception as e:
            bt.logging.error(f"[GRADER] Analyzer error: {e}")
            return _err("analyzer_error", f"Analyzer failed: {e}", post_id, {}, i)
        
        # Extract tokens and sentiment from analysis result
        ref_tokens_raw = (analysis_result.get("subnet_relevance") or {})
        # Keys are normalized here so later comparisons are deterministic.
        ref_tokens_raw_normalized = {str(k).strip().lower(): float(v.get("relevance", 0.0)) for k, v in ref_tokens_raw.items()}
        ref_sent = float(analysis_result.get("sentiment", 0.0))
        
        # Validate tokens/relevance match
        miner_tokens, ref_tokens = select_tokens(miner_tokens_raw, ref_tokens_raw_normalized, k=128, eps=0.05)
        matches, token_diffs = tokens_match_within(miner_tokens, ref_tokens, TOKEN_TOLERANCE)
        if not matches:
            # We include top mismatches to help you debug
            top = dict(sorted(token_diffs.items(), key=lambda kv: kv[1]["diff"], reverse=True)[:5])
            return _err("tokens_mismatch", "subnet relevance differs beyond tolerance", post_id,
                        {"mismatches": top, "total_mismatches": len(token_diffs)}, i)

        # Validate sentiment match
        if abs(miner_sent - ref_sent) > SENTIMENT_TOLERANCE:
            return _err("sentiment_mismatch", "sentiment differs beyond tolerance", post_id,
                        {"miner": miner_sent, "validator": ref_sent, "allowed": SENTIMENT_TOLERANCE, "diff": abs(miner_sent - ref_sent)}, i)

        # If we get here, this post passed all LLM validation checks
        # Note: Score validation is removed since it requires X API metrics,
        # which are now validated on the API side

    # All posts passed LLM validation
    # Note: We don't calculate final_score here because the validator uses
    # avg_score_all_posts (API-calculated average of ALL posts) instead.
    n = len(posts)

    analyzer_version = "unknown"
    if analyzer and hasattr(analyzer, "model"):
        analyzer_version = str(analyzer.model)

    return CONSENSUS_VALID, {
        "n_posts": n,
        "tolerances": {
            "token": TOKEN_TOLERANCE,
            "sentiment": SENTIMENT_TOLERANCE,
        },
        "analyzer": {"version": analyzer_version},
    }
