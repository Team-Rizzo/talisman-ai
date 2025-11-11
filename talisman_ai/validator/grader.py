
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timezone
import time, hashlib, math

import bittensor as bt
import tweepy
from dateutil.parser import isoparse
from talisman_ai.analyzer import setup_analyzer
from talisman_ai.analyzer.scoring import score_tweet_entry
from talisman_ai import config
from talisman_ai.utils.normalization import norm_text

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
    """Create an X API client (REQUIRED). Uses talisman_ai.config.X_BEARER_TOKEN."""
    token = getattr(config, "X_BEARER_TOKEN", None)
    if not token or token == "null":
        raise ValueError("[GRADER] X_BEARER_TOKEN not set - X API is required for validation")
    try:
        return tweepy.Client(bearer_token=token)
    except Exception as e:
        raise RuntimeError(f"[GRADER] Failed to initialize X API client: {e}") from e

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
        out = analyzer.analyze_tweet_complete(text)
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
        "tweet_info": {
            "tweet_text": content,
            "tweet_date": date_iso,
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
        scored = score_tweet_entry(entry, analyzer, k=5, analysis_result=analysis_result)
        return float(scored.get("score", 0.0))
    except Exception as e:
        # We surface this upstream as an INVALID result with code=score_compute_error.
        raise RuntimeError(f"score_compute_error: {e}")

# === X API fetch with deterministic retries ===
def _get_tweet_with_retry(x_client: tweepy.Client, post_id: str, attempts: int = 3):
    """
    Fetch post with retries for rate limits/server hiccups. Backoff includes
    deterministic jitter seeded by post_id so replays behave the same.
    """
    for j in range(attempts):
        try:
            return x_client.get_tweet(
                id=str(post_id),
                expansions=["author_id"],
                tweet_fields=["created_at", "public_metrics", "text"],
                user_fields=["username", "name", "created_at", "public_metrics"],
            )
        except tweepy.TooManyRequests:
            if j == attempts - 1: raise
        except tweepy.TweepyException as e:
            if j == attempts - 1: raise
            # Only retry transient errors
            if not any(s in str(e).lower() for s in ["500", "502", "503", "504", "timeout", "connection"]):
                raise
        # deterministic jitter
        jitter_seed = int(hashlib.md5(str(post_id).encode()).hexdigest()[:8], 16) % 21
        time.sleep(0.5 * (j + 1) + (jitter_seed / 100.0))
    return None

# === Core post-level validation against X ===
def validate_with_x(post: Dict, x_client: tweepy.Client) -> Tuple[Optional[Dict], Dict]:
    """
    Returns (error_dict, live_metrics) where:
      - error_dict is None if the post passes all X-related checks
      - live_metrics contains metrics/timestamps/author info used for scoring
    """
    # 0) post_id must exist
    post_id = post.get("post_id")
    if not post_id:
        return ({"code": "missing_post_id", "message": "post_id is required", "post_id": None, "details": {}}, {})

    # 1) Fetch from X
    try:
        resp = _get_tweet_with_retry(x_client, post_id)
    except Exception as e:
        return ({"code": "x_api_error", "message": f"X API error: {e}", "post_id": post_id, "details": {}}, {})
    if resp is None:
        return ({"code": "x_api_no_response", "message": "X API gave no response after retries", "post_id": post_id, "details": {}}, {})
    if not getattr(resp, "data", None):
        return ({"code": "post_not_found", "message": "Post not found or inaccessible", "post_id": post_id, "details": {}}, {})

    post_data = resp.data
    users = {u.id: u for u in (getattr(resp, "includes", {}) or {}).get("users", [])}
    author = users.get(post_data.author_id)

    # 2) Text must match exactly after normalization (NFC, whitespace normalized)
    miner_text = (post.get("content") or "")
    live_text = post_data.text or ""
    if norm_text(miner_text) != norm_text(live_text):
        return ({"code": "text_mismatch", "message": "content does not match live post text (after normalization)",
                 "post_id": post_id, "details": {"miner": miner_text[:100], "live": live_text[:100], "preview_len": 100}}, {})

    # 3) Author must match (lowercase usernames)
    miner_author = (post.get("author") or "").strip().lower()
    live_author = (author.username if author else "").strip().lower()
    if miner_author != live_author:
        return ({"code": "author_mismatch", "message": "author does not match", "post_id": post_id,
                 "details": {"miner": post.get("author", ""), "live": author.username if author else ""}}, {})

    # 4) Timestamp must match exactly (Unix seconds)
    miner_ts = post.get("date") or post.get("timestamp")
    if miner_ts is None:
        bt.logging.error(f"[GRADER] Unexpected: miner timestamp None after API validation (post_id={post_id})")
        return ({"code": "timestamp_missing", "message": "timestamp is missing (API validation should have caught this)", "post_id": post_id, "details": {}}, {})
    miner_ts = int(miner_ts)
    if not getattr(post_data, "created_at", None):
        return ({"code": "missing_created_at", "message": "live post missing created_at from X API", "post_id": post_id, "details": {}}, {})
    live_ts = int(post_data.created_at.timestamp())
    if miner_ts != live_ts:
        return ({"code": "timestamp_mismatch", "message": "timestamp must match exactly",
                 "post_id": post_id, "details": {"miner": miner_ts, "live": live_ts, "diff_seconds": abs(live_ts - miner_ts)}}, {})

    # 5) Engagement/author metrics may NOT be overstated beyond tolerance
    pm = getattr(post_data, "public_metrics", None) or {}
    live_likes = int(pm.get("like_count", 0) or 0)
    live_rts = int(pm.get("retweet_count", 0) or 0)
    live_replies = int(pm.get("reply_count", 0) or 0)
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

    followers = 0
    if author and getattr(author, "public_metrics", None):
        followers = int(author.public_metrics.get("followers_count", 0) or 0)
    m_followers = int(post.get("followers") or 0)
    if metric_inflated(m_followers, followers):
        return ({"code": "metric_inflation_followers", "message": "followers overstated beyond tolerance",
                 "post_id": post_id, "details": {"miner": m_followers, "live": followers, "tolerance": metric_tol(followers)}}, {})

    # 6) Compute author account age (days) from X (used later for scoring; we do NOT trust miner value)
    account_age_days = 0
    if author and getattr(author, "created_at", None):
        try:
            now = datetime.now(timezone.utc)
            account_created = author.created_at
            if isinstance(account_created, str):
                account_created = isoparse(account_created)
            if account_created.tzinfo is None:
                account_created = account_created.replace(tzinfo=timezone.utc)
            account_age_days = max(0, (now - account_created).days)
        except Exception as e:
            bt.logging.warning(f"[GRADER] Failed to compute account age: {e}, using 0")
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
    """Grade a list of posts. Stops on first failure; otherwise returns final score.">
    """
    # Basic sanity
    if not posts:
        return _err("no_posts", "no posts submitted")
    try:
        analyzer = analyzer or make_analyzer()
        if analyzer is None:
            return _err("analyzer_unavailable", "Analyzer not initialized")
        x_client = x_client or make_x_client()
    except Exception as e:
        return _err("x_api_unavailable", str(e))

    for i, post in enumerate(posts):
        post_id = post.get("post_id")
        if not post_id:
            return _err("missing_post_id", "post_id is required", None, {}, i)

        # --- Stage 1: live X validation ---
        err, live = validate_with_x(post, x_client)
        if err is not None:
            err.update({"post_index": i})
            return CONSENSUS_INVALID, {"error": err, "final_score": 0.0}

        # --- Stage 2: content analysis validation ---
        content = post.get("content") or ""
        if not content:
            return _err("empty_content", "post content is empty", post_id, {}, i)

        miner_tokens_raw = post.get("tokens") or {}
        miner_sent = float(post.get("sentiment") or 0.0)
        
        # Get analysis result once - we'll reuse it for scoring to avoid duplicate LLM calls
        try:
            analysis_result = analyzer.analyze_tweet_complete(content)
        except Exception as e:
            bt.logging.error(f"[GRADER] Analyzer error: {e}")
            return _err("analyzer_error", f"Analyzer failed: {e}", post_id, {}, i)
        
        # Extract tokens and sentiment from analysis result
        ref_tokens_raw = (analysis_result.get("subnet_relevance") or {})
        # Keys are normalized here so later comparisons are deterministic.
        ref_tokens_raw_normalized = {str(k).strip().lower(): float(v.get("relevance", 0.0)) for k, v in ref_tokens_raw.items()}
        ref_sent = float(analysis_result.get("sentiment", 0.0))
        
        miner_tokens, ref_tokens = select_tokens(miner_tokens_raw, ref_tokens_raw_normalized, k=128, eps=0.05)
        matches, token_diffs = tokens_match_within(miner_tokens, ref_tokens, TOKEN_TOLERANCE)
        if not matches:
            # We include top mismatches to help you debug
            top = dict(sorted(token_diffs.items(), key=lambda kv: kv[1]["diff"], reverse=True)[:5])
            return _err("tokens_mismatch", "subnet relevance differs beyond tolerance", post_id,
                        {"mismatches": top, "total_mismatches": len(token_diffs)}, i)

        if abs(miner_sent - ref_sent) > SENTIMENT_TOLERANCE:
            return _err("sentiment_mismatch", "sentiment differs beyond tolerance", post_id,
                        {"miner": miner_sent, "validator": ref_sent, "allowed": SENTIMENT_TOLERANCE, "diff": abs(miner_sent - ref_sent)}, i)

        # --- Stage 3: score cross-check ---
        followers_x = live.get("followers", 0)
        likes_x, rts_x, replies_x = live.get("likes", 0), live.get("retweets", 0), live.get("replies", 0)
        account_age_days = live.get("account_age_days", 0)
        miner_score = float(post.get("score") or 0.0)
        date_iso = iso_from_unix(live["created_at"])
        try:
            # Pass the already-computed analysis_result to avoid re-analyzing
            v_score = compute_validator_score(content, date_iso, likes_x, rts_x, replies_x,
                                              account_age_days, followers_x, analyzer, analysis_result)
        except RuntimeError as e:
            return _err("score_compute_error", str(e), post_id, {}, i)

        if miner_score > v_score + SCORE_TOLERANCE:
            return _err("score_inflation", "miner score exceeds validator tolerance", post_id,
                        {"miner_score": miner_score, "validator_score": v_score, "allowed_over": SCORE_TOLERANCE}, i)

        # If we get here, this post passed all validation checks

    # All posts passed validation
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
            "score": SCORE_TOLERANCE,
            "metrics_rel": POST_METRIC_TOLERANCE,
        },
        "analyzer": {"version": analyzer_version},
    }
