
from __future__ import annotations
from typing import Dict, List, Tuple, Optional

import bittensor as bt
from talisman_ai.analyzer import setup_analyzer
from talisman_ai.utils.normalization import norm_text

# =============================================================================
# Post Validation System
# =============================================================================
# Validates posts using LLM-based analysis (tokens and sentiment).
#
# Validation checks (stops on first failure):
#   1. Content analysis: tokens and sentiment must match validator's analysis
#      - Token tolerance: ±0.05 absolute
#      - Sentiment tolerance: ±0.05 absolute
#
# Note: X API validation (post existence, text/author/timestamp matching,
# metric inflation) is done on the API side before posts reach validators.
#
# Scoring: Validator determines VALID/INVALID. If VALID, uses avg_score_all_posts
# from API (average of ALL posts). This grader only validates, doesn't calculate scores.
#
# Returns:
#   - VALID: (CONSENSUS_VALID, { n_posts, tolerances, analyzer })
#   - INVALID: (CONSENSUS_INVALID, { "error": {...}, "final_score": 0.0 })
# =============================================================================

CONSENSUS_VALID, CONSENSUS_INVALID = 1, 0
TOKEN_TOLERANCE = 0.05
SENTIMENT_TOLERANCE = 0.05

def _err(code: str, message: str, post_id=None, details=None, post_index: Optional[int] = None):
    """Create standardized error response."""
    e = {"code": code, "message": message, "post_id": post_id, "details": details or {}}
    if post_index is not None:
        e["post_index"] = post_index
    return CONSENSUS_INVALID, {"error": e, "final_score": 0.0}

def make_analyzer():
    """Create validator analyzer instance."""
    try:
        return setup_analyzer()
    except Exception as e:
        bt.logging.error(f"[GRADER] Analyzer init failed: {e}")
        return None

def analyze_text(text: str, analyzer) -> Tuple[Dict[str, float], float]:
    """Run validator analysis on text, return (tokens, sentiment)."""
    if analyzer is None:
        return {}, 0.0
    try:
        out = analyzer.analyze_post_complete(text)
    except Exception as e:
        bt.logging.error(f"[GRADER] Analyzer error: {e}")
        return {}, 0.0
    tokens_raw = (out.get("subnet_relevance") or {})
    tokens = {str(k).strip().lower(): float(v.get("relevance", 0.0)) for k, v in tokens_raw.items()}
    sentiment = float(out.get("sentiment", 0.0))
    return tokens, sentiment

def normalize_keys(d: Dict) -> Dict[str, float]:
    """Normalize dict keys: lowercase, stripped; values -> float."""
    return {str(k).strip().lower(): float(v) for k, v in (d or {}).items()}

def select_tokens(miner_raw: Dict, ref_raw: Dict, k: int = 128, eps: float = 0.05) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Select tokens for comparison.
    
    Only compares tokens found by BOTH miner and validator, plus validator-only tokens
    (which represent missed detections). This prevents miners from padding submissions
    with fake low-value tokens that would pass validation against 0.0.
    """
    mt = {k: v for k, v in normalize_keys(miner_raw).items() if abs(v) >= eps}
    rt = {k: v for k, v in normalize_keys(ref_raw).items() if abs(v) >= eps}
    
    # Only compare tokens that both parties found relevant (common tokens)
    # Plus validator-only tokens (miner missed these - should be penalized)
    common_tokens = set(mt.keys()) & set(rt.keys())
    validator_only = set(rt.keys()) - set(mt.keys())
    
    # Combine: common + validator-only (up to k tokens)
    keep = common_tokens | validator_only
    if len(keep) > k:
        # If too many, prioritize by validator's relevance scores
        keep = set(sorted(keep, key=lambda x: -abs(rt.get(x, 0.0)))[:k])
    
    return {k: mt.get(k, 0.0) for k in keep}, {k: rt.get(k, 0.0) for k in keep}

def tokens_match_within(miner: Dict[str, float], ref: Dict[str, float], abs_tol: float, eps: float = 0.05) -> Tuple[bool, Dict]:
    """Compare tokens with absolute tolerance. Returns (match, diffs dict)."""
    diffs = {}
    for k in (set(miner) | set(ref)):
        a, b = float(miner.get(k, 0.0)), float(ref.get(k, 0.0))
        if a < eps and b < eps:  # ignore noise
            continue
        if abs(a - b) > abs_tol:
            diffs[k] = {"miner": a, "validator": b, "allowed": abs_tol, "diff": abs(a - b)}
    return (len(diffs) == 0, diffs)

def grade_hotkey(posts: List[Dict], analyzer=None) -> Tuple[int, Dict]:
    """
    Grade posts using LLM validation (tokens and sentiment).
    
    X API validation is done server-side. Stops on first failure.
    
    Returns:
        (CONSENSUS_VALID or CONSENSUS_INVALID, result_dict)
    """
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

        content = post.get("content") or ""
        if not content:
            return _err("empty_content", "post content is empty", post_id, {}, i)
        
        content = norm_text(content)
        if not content:
            return _err("empty_content", "post content is empty after normalization", post_id, {}, i)

        miner_tokens_raw = post.get("tokens") or {}
        miner_sent = float(post.get("sentiment") or 0.0)
        
        try:
            analysis_result = analyzer.analyze_post_complete(content)
        except Exception as e:
            bt.logging.error(f"[GRADER] Analyzer error: {e}")
            return _err("analyzer_error", f"Analyzer failed: {e}", post_id, {}, i)
        
        ref_tokens_raw = (analysis_result.get("subnet_relevance") or {})
        ref_tokens_raw_normalized = {str(k).strip().lower(): float(v.get("relevance", 0.0)) for k, v in ref_tokens_raw.items()}
        ref_sent = float(analysis_result.get("sentiment", 0.0))
        
        miner_tokens, ref_tokens = select_tokens(miner_tokens_raw, ref_tokens_raw_normalized, k=128, eps=0.05)
        matches, token_diffs = tokens_match_within(miner_tokens, ref_tokens, TOKEN_TOLERANCE)
        if not matches:
            top = dict(sorted(token_diffs.items(), key=lambda kv: kv[1]["diff"], reverse=True)[:5])
            return _err("tokens_mismatch", "subnet relevance differs beyond tolerance", post_id,
                        {"mismatches": top, "total_mismatches": len(token_diffs)}, i)

        if abs(miner_sent - ref_sent) > SENTIMENT_TOLERANCE:
            return _err("sentiment_mismatch", "sentiment differs beyond tolerance", post_id,
                        {"miner": miner_sent, "validator": ref_sent, "allowed": SENTIMENT_TOLERANCE, "diff": abs(miner_sent - ref_sent)}, i)

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
