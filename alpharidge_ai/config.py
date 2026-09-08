"""
Centralized configuration loader for alpharidge_ai_subnet.
Loads environment variables from .miner_env and .vali_env files.

This module should be imported at the top of any module that needs configuration:
    from alpharidge_ai import config
    
Then access config values as:
    config.MODEL
    config.BLOCKS_PER_WINDOW
    etc.
"""

from pathlib import Path
import os

# Find the alpharidge_ai_subnet root directory
# This file is at alpharidge_ai_subnet/alpharidge_ai/config.py
_SUBNET_ROOT = Path(__file__).resolve().parent.parent

# Paths to environment files
_MINER_ENV_PATH = _SUBNET_ROOT / ".miner_env"
_VALI_ENV_PATH = _SUBNET_ROOT / ".vali_env"

# Load environment files
try:
    from dotenv import load_dotenv
    
    # Load miner env file (if it exists)
    if _MINER_ENV_PATH.exists():
        load_dotenv(str(_MINER_ENV_PATH), override=True)
        print(f"[CONFIG] Loaded {_MINER_ENV_PATH}")
    else:
        print(f"[CONFIG] Warning: {_MINER_ENV_PATH} not found")
    
    # Load validator env file (if it exists)
    # Note: validator vars will override miner vars if both exist
    if _VALI_ENV_PATH.exists():
        load_dotenv(str(_VALI_ENV_PATH), override=True)
        print(f"[CONFIG] Loaded {_VALI_ENV_PATH}")
    else:
        print(f"[CONFIG] Warning: {_VALI_ENV_PATH} not found")
        
except ImportError:
    print("[CONFIG] Warning: python-dotenv not installed, using system environment variables only")


# ============================================================================
# Shared Configuration (available to both miners and validators)
# ============================================================================

# LLM Analysis
MODEL = os.getenv("MODEL", "null")
API_KEY = os.getenv("API_KEY", "null")
LLM_BASE = os.getenv("LLM_BASE", "null")

# X/Twitter API Configuration
X_BEARER_TOKEN = os.getenv("X_BEARER_TOKEN", "null")
X_API_BASE = os.getenv("X_API_BASE", "null")

# SN13/Macro API Configuration
SN13_API_KEY = os.getenv("SN13_API_KEY", "null")
SN13_API_URL = os.getenv("SN13_API_URL", "https://constellation.api.cloud.macrocosmos.ai/sn13.v1.Sn13Service/OnDemandData")

# API Source Selection (for validators)
# Set to "x_api" or "sn13_api" to choose which API to use for validation
X_API_SOURCE = os.getenv("X_API_SOURCE", "x_api")


# ============================================================================
# Miner-Specific Configuration
# ============================================================================

# V3 miners process TweetBatch requests from validators - no scraping/submission config needed


# ============================================================================
# Validator-Specific Configuration
# ============================================================================

# Miner API configuration
MINER_API_URL = os.getenv("MINER_API_URL", "null")
BATCH_HTTP_TIMEOUT = float(os.getenv("BATCH_HTTP_TIMEOUT", "30.0"))
VOTE_ENDPOINT = os.getenv("VOTE_ENDPOINT", "null")
# Backward compatibility: support both old and new names
VALIDATION_POLL_SECONDS = int(os.getenv("VALIDATION_POLL_SECONDS", os.getenv("BATCH_POLL_SECONDS", "10")))
SCORES_BLOCK_INTERVAL = int(os.getenv("SCORES_BLOCK_INTERVAL", "100"))

MINER_BATCH_SIZE = int(os.getenv("MINER_BATCH_SIZE", "3"))
# How many tweets/messages to fetch from the API per poll cycle.
# Fetched items are split into MINER_BATCH_SIZE chunks and dispatched to different miners.
VALIDATION_FETCH_LIMIT = int(os.getenv("VALIDATION_FETCH_LIMIT", "24"))
# Rolling buffer cap for the local article store. Prune keeps this many most-recent
# articles (older ones age out) so validators always dispatch fresh news without a
# stale backlog accumulating. Remote-config tunable so it can be adjusted subnet-wide.
ARTICLE_STORE_MAX_ARTICLES = int(os.getenv("ARTICLE_STORE_MAX_ARTICLES", "2000"))
BLOCK_LENGTH = int(os.getenv("BLOCK_LENGTH", "100"))
START_BLOCK = int(os.getenv("START_BLOCK", "0"))

# Validator -> miner dispatch behavior (push-based mining).
# The validator should only "dispatch" work; miners will push results back asynchronously.
MINER_SEND_TIMEOUT = float(os.getenv("MINER_SEND_TIMEOUT", "6.0"))
VALIDATOR_MINER_QUERY_CONCURRENCY = int(os.getenv("VALIDATOR_MINER_QUERY_CONCURRENCY", "8"))
VALIDATOR_MAX_PENDING_MINER_TASKS = int(os.getenv("VALIDATOR_MAX_PENDING_MINER_TASKS", "256"))

# Validation thread pool: controls how many concurrent LLM-based validations run.
# Lower values reduce LLM API pressure at the cost of slower validation throughput.
VALIDATION_MAX_WORKERS = int(os.getenv("VALIDATION_MAX_WORKERS", "8"))
# Torch intra-op threads per validation worker. Torch defaults to ncores (48 here), so
# VALIDATION_MAX_WORKERS=16 puts 16*48=768 threads on 96 cores (8x oversubscription).
# 0 = leave torch's default.
TORCH_NUM_THREADS = int(os.getenv("TORCH_NUM_THREADS", "0"))

# LLM result cache: avoids redundant API calls for identical post text.
LLM_CACHE_TTL = float(os.getenv("LLM_CACHE_TTL", "300"))
LLM_CACHE_MAX_SIZE = int(os.getenv("LLM_CACHE_MAX_SIZE", "1024"))

# Tweet store configuration
TWEET_STORE_LOCATION = os.getenv("TWEET_STORE_LOCATION", str(_SUBNET_ROOT / ".tweet_store.json"))
TWEET_MAX_PROCESS_TIME = float(os.getenv("TWEET_MAX_PROCESS_TIME", "300.0"))  # 5 minutes default

# Tweet scoring master switch. Default OFF (article-first): when false the validator does NOT
# fetch tweets or apply tweet timeout penalties, so article-only miners aren't zeroed for tweet
# timeouts. Remote-config toggleable via /config/subnet (key ENABLE_TWEET_SCORING).
ENABLE_TWEET_SCORING = os.getenv("ENABLE_TWEET_SCORING", "false").lower() == "true"

# Telegram store configuration
TELEGRAM_STORE_LOCATION = os.getenv("TELEGRAM_STORE_LOCATION", str(_SUBNET_ROOT / ".telegram_store.json"))

# Message max process time (shared for tweets and telegram, with fallback chain for backward compatibility)
MESSAGE_MAX_PROCESS_TIME = float(os.getenv("MESSAGE_MAX_PROCESS_TIME", os.getenv("TWEET_MAX_PROCESS_TIME", "300.0")))

# Penalty and reward store configuration
PENALTY_STORE_LOCATION = os.getenv("PENALTY_STORE_LOCATION", str(_SUBNET_ROOT / ".penalty_store.json"))
REWARD_STORE_LOCATION = os.getenv("REWARD_STORE_LOCATION", str(_SUBNET_ROOT / ".reward_store.json"))

USD_PRICE_PER_POINT = float(os.getenv("USD_PRICE_PER_POINT", "0.040"))
FINNEY_RPC = os.getenv("FINNEY_RPC", "wss://entrypoint-finney.opentensor.ai:443")

EPOCH_LENGTH = int(os.getenv("EPOCH_LENGTH", "100"))

BURN_UID = int(os.getenv("BURN_UID", "189"))

# Validator↔validator broadcast state (rewards and penalties)
BROADCAST_STATE_LOCATION = os.getenv("BROADCAST_STATE_LOCATION", str(_SUBNET_ROOT / ".broadcast_state.json"))
PENALTY_BROADCAST_STATE_LOCATION = os.getenv("PENALTY_BROADCAST_STATE_LOCATION", str(_SUBNET_ROOT / ".penalty_broadcast_state.json"))
VALIDATOR_BROADCAST_MAX_TARGETS = int(os.getenv("VALIDATOR_BROADCAST_MAX_TARGETS", "32"))

# Validator allowlist selection
VALIDATOR_STAKE_THRESHOLD = float(os.getenv("VALIDATOR_STAKE_THRESHOLD", "0"))
VALIDATOR_CACHE_SECONDS = float(os.getenv("VALIDATOR_CACHE_SECONDS", "120"))
ALLOW_MANUAL_VALIDATOR_HOTKEYS = os.getenv("ALLOW_MANUAL_VALIDATOR_HOTKEYS", "false").lower() == "true"
MANUAL_VALIDATOR_HOTKEYS = [hk.strip() for hk in os.getenv("MANUAL_VALIDATOR_HOTKEYS", "").split(",") if hk.strip()]

# Verifiable validator points: pinned API attestation pubkey (sr25519 ss58) and
# the fraction of received broadcasts to deep-verify against raw /verdicts.
# The pubkey default is the live SN45 attestation key (public by design — safe to
# ship). Enforcement defaults ON so an updated validator drops fabricated/unsigned
# broadcasts out of the box (override either via .vali_env if needed).
API_ATTESTATION_PUBKEY = os.getenv("API_ATTESTATION_PUBKEY", "5DqYRNaJ9FJ2cuJTFxbU5HDLeeotpgj6Zgkrkw4RgS6SA4nf")
DEEP_VERIFY_SAMPLE_RATE = float(os.getenv("DEEP_VERIFY_SAMPLE_RATE", "0.1"))
ENFORCE_SIGNED_ATTESTATIONS = os.getenv("ENFORCE_SIGNED_ATTESTATIONS", "true").lower() == "true"


# ============================================================================
# Remote Config (fetched from API)
# ============================================================================

import time
import threading
import requests


def _log_info(msg: str) -> None:
    try:
        import bittensor as bt
        bt.logging.info(msg)
    except Exception:
        print(msg)


def _log_warning(msg: str) -> None:
    try:
        import bittensor as bt
        bt.logging.warning(msg)
    except Exception:
        print(msg)

def _log_error(msg: str) -> None:
    try:
        import bittensor as bt
        bt.logging.error(msg)
    except Exception:
        print(msg)

MIN_PERCENT_PER_POINT = float(os.getenv("MIN_PERCENT_PER_POINT", "0.003"))

# ---------------------------------------------------------------------------
# Weight-window aggregation
# ---------------------------------------------------------------------------
# K is how many BLOCK_LENGTH epochs the weight calculation sums over. K=1 is the
# prior behaviour. _PREV holds the value in force before _ACTIVE_BLOCK so that K
# is a pure function of block height and every validator switches together
# rather than whenever it last polled the API.
WEIGHT_WINDOW_EPOCHS_MAX = 24


def _clamp_window(value, default: int, minimum: int = 1) -> int:
    """Bound K at ingestion so no consumer has to re-test it.

    Store retention is sized from the raw values (it must cover K before the
    activation block), so an out-of-range K would grow the store even though
    weight_window_epochs() would refuse to use it.
    """
    try:
        k = int(value)
    except (TypeError, ValueError):
        _log_warning(f"[WEIGHT_WINDOW] K={value!r} is not an integer; using {default}")
        return default
    if not minimum <= k <= WEIGHT_WINDOW_EPOCHS_MAX:
        _log_warning(f"[WEIGHT_WINDOW] K={k} outside {minimum}..{WEIGHT_WINDOW_EPOCHS_MAX}; using {default}")
        return default
    return k


WEIGHT_WINDOW_EPOCHS = _clamp_window(os.getenv("WEIGHT_WINDOW_EPOCHS", "1"), 1)
WEIGHT_WINDOW_EPOCHS_PREV = _clamp_window(os.getenv("WEIGHT_WINDOW_EPOCHS_PREV", "1"), 1)
WEIGHT_WINDOW_ACTIVE_BLOCK = int(os.getenv("WEIGHT_WINDOW_ACTIVE_BLOCK", "0"))
# Local only, never served: shadow mode logs a second vector without setting it.
# 0 is off. Feeds retention sizing, so it is clamped like the others.
WEIGHT_WINDOW_SHADOW_EPOCHS = _clamp_window(os.getenv("WEIGHT_WINDOW_SHADOW_EPOCHS", "0"), 0, minimum=0)


if BLOCK_LENGTH != EPOCH_LENGTH:
    # The reward store buckets points by BLOCK_LENGTH, but the burn calculation
    # historically divided by the EPOCH_LENGTH constant. The window divisor is
    # derived from BLOCK_LENGTH, so where the two disagree the calculated percent
    # shifts by their ratio and landing at K=1 is not a no-op.
    _log_warning(
        f"[WEIGHT_WINDOW] BLOCK_LENGTH={BLOCK_LENGTH} != EPOCH_LENGTH={EPOCH_LENGTH}; "
        f"weight percents shift by {EPOCH_LENGTH / BLOCK_LENGTH:.3g}x versus the previous release"
    )


def _cast_window_epochs(v) -> int:
    """Remote-config cast for the window keys.

    Raises on an out-of-range value; refresh_remote_config's except-clause then
    keeps the previous value rather than applying a bad one.
    """
    k = int(v)
    if not 1 <= k <= WEIGHT_WINDOW_EPOCHS_MAX:
        _log_error(f"[REMOTE_CONFIG] rejected weight window K={k}; keeping previous")
        raise ValueError(k)
    return k


def weight_window_epochs(block: int) -> int:
    """K in force at a given block. Read this, never the raw keys, at a call site."""
    if WEIGHT_WINDOW_EPOCHS != WEIGHT_WINDOW_EPOCHS_PREV and WEIGHT_WINDOW_ACTIVE_BLOCK <= 0:
        # A transition is configured with no block to make it at. Every validator
        # would switch whenever it happened to poll, spread over the refresh
        # interval — the divergence the activation block exists to prevent. Hold
        # the previous value; the served config is incomplete, not the schedule.
        _log_error(
            f"[WEIGHT_WINDOW] K={WEIGHT_WINDOW_EPOCHS} differs from previous "
            f"{WEIGHT_WINDOW_EPOCHS_PREV} but WEIGHT_WINDOW_ACTIVE_BLOCK is "
            f"{WEIGHT_WINDOW_ACTIVE_BLOCK}; holding {WEIGHT_WINDOW_EPOCHS_PREV}. "
            f"Set the activation block on the API and re-serve."
        )
        return WEIGHT_WINDOW_EPOCHS_PREV
    if int(block) >= WEIGHT_WINDOW_ACTIVE_BLOCK:
        return WEIGHT_WINDOW_EPOCHS
    return WEIGHT_WINDOW_EPOCHS_PREV


def weight_window_retention() -> int:
    """Epochs the reward/penalty stores must keep.

    Sized from the raw keys, not weight_window_epochs(): the epochs a window
    needs after the activation block were recorded before it. The floor keeps
    retention at or above its historical value for any configuration.
    """
    k_max = max(WEIGHT_WINDOW_EPOCHS, WEIGHT_WINDOW_EPOCHS_PREV, WEIGHT_WINDOW_SHADOW_EPOCHS)
    return max(10, k_max + 5)


BLACKLISTED_MINER_HOTKEYS: set = set()

def _as_bool(v) -> bool:
    """Parse a remote-config / env value as bool (bool("false") is truthy, so we can't use bool())."""
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("true", "1", "yes", "on")


# Adaptive dispatch (RFC 2026-06-28). All behaviour is gated behind
# ADAPTIVE_DISPATCH_ENABLED; with the flag off these values are never read, so
# defaults reproduce today's behaviour exactly. Served by the API at
# /config/subnet and overridable per-validator via OVERRIDE_<key>.
ADAPTIVE_DISPATCH_ENABLED = _as_bool(os.getenv("ADAPTIVE_DISPATCH_ENABLED", "false"))
DISPATCH_WINDOW_MIN = int(os.getenv("DISPATCH_WINDOW_MIN", "1"))
DISPATCH_WINDOW_CAP_PCT = float(os.getenv("DISPATCH_WINDOW_CAP_PCT", "0.15"))
# Budget DISPATCH_WINDOW_CAP_PCT is a percentage of. Was hardwired to
# VALIDATOR_MINER_QUERY_CONCURRENCY, conflating the send semaphore (an IO limit) with how much
# work one miner may hold (which drives validation load). 0 = track the semaphore (pre-2026-07-17).
DISPATCH_WINDOW_BUDGET = float(os.getenv("DISPATCH_WINDOW_BUDGET", "0"))


def dispatch_window_budget() -> float:
    """Send budget the per-miner window cap derives from. Reads globals per call so
    remote-config / OVERRIDE_ apply without a restart; 0 falls back to the semaphore."""
    return float(DISPATCH_WINDOW_BUDGET or VALIDATOR_MINER_QUERY_CONCURRENCY)
DISPATCH_WINDOW_GROW = float(os.getenv("DISPATCH_WINDOW_GROW", "1.0"))
DISPATCH_WINDOW_SHRINK = float(os.getenv("DISPATCH_WINDOW_SHRINK", "0.5"))
DISPATCH_LATE_FRACTION = float(os.getenv("DISPATCH_LATE_FRACTION", "0.6"))
# Must match the IsAlive liveness ping timeout (12s in get_alive_uids): the roster is
# built from that ping, so a shorter ack window systematically fails alive-but-slow
# miners. (TODO: share one constant with get_alive_uids so they can't drift.)
DISPATCH_ACK_TIMEOUT_S = float(os.getenv("DISPATCH_ACK_TIMEOUT_S", "12.0"))
# How long a dispatched article may sit in PROCESSING before it is reclaimed. Both readers
# (article_store.get_timeouts, cooldown._late_threshold_s) used a hardcoded 900 fallback and it
# was never declared here. Doubly-loaded: also scales the late-growth threshold (0.6 * TTL),
# inert only while the window cap floors to 1.
SCORING_LEASE_TTL_SECONDS = float(os.getenv("SCORING_LEASE_TTL_SECONDS", "900"))
DISPATCH_CHRONIC_TIMEOUT_N = int(os.getenv("DISPATCH_CHRONIC_TIMEOUT_N", "5"))
LIVENESS_TTL_S = int(os.getenv("LIVENESS_TTL_S", "120"))
LIVENESS_SWEEP_INTERVAL_S = int(os.getenv("LIVENESS_SWEEP_INTERVAL_S", "60"))
# Penalty-split sub-flag: when adaptive dispatch is on, this gates the cross-validator
# timeout→broadcast reclassification SEPARATELY, so it can be enabled/rolled back
# independently of the dispatch behaviour. Default true (timeouts are capacity signals;
# only chronic non-response penalizes/broadcasts). Set false to run adaptive dispatch
# with the legacy per-timeout penalty/broadcast intact (window still shrinks either way).
ADAPTIVE_PENALTY_SPLIT_ENABLED = _as_bool(os.getenv("ADAPTIVE_PENALTY_SPLIT_ENABLED", "true"))

# Missing/incomplete-analysis penalty split. A batch returned without finished
# analysis (the miner hasn't caught up with a dispatched burst) is a CAPACITY signal,
# not cheating — like a timeout, not a wrong answer. With this on, such failures back
# off the dispatch window (no integrity penalty, no emission-gate hit); wrong/cloned
# analysis still penalizes. Default false = legacy (penalize everything) so the deploy
# is a no-op until piloted. Gated separately from the timeout split for isolated rollback.
ADAPTIVE_MISSING_ANALYSIS_SPLIT_ENABLED = _as_bool(os.getenv("ADAPTIVE_MISSING_ANALYSIS_SPLIT_ENABLED", "false"))

# Faithfulness cooldown (2026-07-09).
DISPATCH_COOLDOWN_SHADOW_MODE = _as_bool(os.getenv("DISPATCH_COOLDOWN_SHADOW_MODE", "true"))
DISPATCH_COOLDOWN_FAITHFULNESS_FLOOR = float(os.getenv("DISPATCH_COOLDOWN_FAITHFULNESS_FLOOR", "0.5"))
DISPATCH_CONSEC_INVALID_N = int(os.getenv("DISPATCH_CONSEC_INVALID_N", "10"))
DISPATCH_INVALID_COOLDOWN_FIRST_S = int(os.getenv("DISPATCH_INVALID_COOLDOWN_FIRST_S", "60"))
DISPATCH_INVALID_COOLDOWN_MAX_S = int(os.getenv("DISPATCH_INVALID_COOLDOWN_MAX_S", "600"))
DISPATCH_FAILSTREAK_SHADOW_MODE = _as_bool(os.getenv("DISPATCH_FAILSTREAK_SHADOW_MODE", "true"))
DISPATCH_CONSEC_FAIL_N = int(os.getenv("DISPATCH_CONSEC_FAIL_N", "10"))

# Per-miner batch size, clamped to [MIN, MAX]. Bounds default to MINER_BATCH_SIZE (off until raised).
ADAPTIVE_BATCH_SIZE_ENABLED = _as_bool(os.getenv("ADAPTIVE_BATCH_SIZE_ENABLED", "false"))
MINER_BATCH_SIZE_MAX     = int(os.getenv("MINER_BATCH_SIZE_MAX", str(MINER_BATCH_SIZE)))
MINER_BATCH_SIZE_MIN     = int(os.getenv("MINER_BATCH_SIZE_MIN", str(MINER_BATCH_SIZE)))
BATCH_SIZE_GROW_STEP     = int(os.getenv("BATCH_SIZE_GROW_STEP", "2"))
BATCH_SIZE_SHRINK_FACTOR = float(os.getenv("BATCH_SIZE_SHRINK_FACTOR", "0.75"))

# Validation quality floor (composite >= TIER3_THRESHOLD). Served centrally so every
# validator uses the same threshold — divergent thresholds would score the same article
# differently. Not gated by the dispatch flag (separate scoring track). Default 0.70.
TIER3_THRESHOLD = float(os.getenv("TIER3_THRESHOLD", "0.70"))

# Cross-article cloned-embedding gate. The within-batch title-embedding similarity
# above which a pair is a clone candidate (default 0.99 = current behavior). When
# CLONE_DIFFERENTIAL_ENABLED is off, any candidate is flagged (the legacy absolute
# rule); when on, a candidate is only a clone if the miner's similarity exceeds the
# validator's own re-embedding of the same titles by >= CLONE_DIVERGENCE_MARGIN —
# so honest syndicated clusters (which the validator also sees as similar) pass.
# Defaults reproduce current behavior, so deploying this is a no-op until served.
CLONE_COSINE_THRESHOLD     = float(os.getenv("CLONE_COSINE_THRESHOLD", "0.99"))
CLONE_DIFFERENTIAL_ENABLED = _as_bool(os.getenv("CLONE_DIFFERENTIAL_ENABLED", "false"))
CLONE_DIVERGENCE_MARGIN    = float(os.getenv("CLONE_DIVERGENCE_MARGIN", "0.05"))

# Reputation-scoring track. Served so every validator matches (consensus-critical).
# SCORING_ENABLED computes + observes only (no effect on weights); GATING_ENABLED lets
# reputation drive emission. Both default off => deploy is a no-op until served on.
REPUTATION_SCORING_ENABLED  = _as_bool(os.getenv("REPUTATION_SCORING_ENABLED", "false"))
REPUTATION_GATING_ENABLED   = _as_bool(os.getenv("REPUTATION_GATING_ENABLED", "false"))
REPUTATION_EMA_ALPHA        = float(os.getenv("REPUTATION_EMA_ALPHA", "0.03"))
REPUTATION_PRIOR            = float(os.getenv("REPUTATION_PRIOR", "0.5"))
EMISSION_MIDPOINT           = float(os.getenv("EMISSION_MIDPOINT", "0.59"))
EMISSION_GAIN               = float(os.getenv("EMISSION_GAIN", "100.0"))
# Emission bonus: multiplier ramps above 1.0 for reputation between START and FULL. CEILING=0 = off.
EMISSION_BONUS_CEILING      = float(os.getenv("EMISSION_BONUS_CEILING", "0.0"))
EMISSION_BONUS_START        = float(os.getenv("EMISSION_BONUS_START", "0.63"))
EMISSION_BONUS_FULL         = float(os.getenv("EMISSION_BONUS_FULL", "0.75"))
# Observations required before reputation may move a hotkey's multiplier. Below it the
# multiplier is neutral 1.0 (see reputation.emission).
EMISSION_N_MIN              = int(os.getenv("EMISSION_N_MIN", "100"))
# Run the keyed audit and log what it would observe. Off by default: it spends on
# grader calls and writes nothing, so it is opt-in per validator.
AUDIT_SHADOW_ENABLED        = _as_bool(os.getenv("AUDIT_SHADOW_ENABLED", "false"))
# Reference analyses the audit may run per batch beyond the acceptance sample. Bounds
# both the grader spend and how long one batch can take to validate.
AUDIT_MAX_PER_BATCH         = int(os.getenv("AUDIT_MAX_PER_BATCH", "4"))
VALIDATION_SAMPLE_SIZE      = int(os.getenv("VALIDATION_SAMPLE_SIZE", "1"))
SUMMARY_AGREEMENT_FLOOR     = float(os.getenv("SUMMARY_AGREEMENT_FLOOR", "0.4"))
SAMPLING_SUBSTANTIVE_WEIGHT = float(os.getenv("SAMPLING_SUBSTANTIVE_WEIGHT", "2.0"))

# Cutover switch for mandatory triage. While false, a batch with no triage
# data grades the legacy way; once served true, it fails integrity checks.
# The flag and the grace path it controls are removed after the migration.
TRIAGE_ENFORCED             = _as_bool(os.getenv("TRIAGE_ENFORCED", "false"))

# run the release. Triage events feed the existing reputation observation


_REMOTE_CONFIG_KEYS = {
    "USD_PRICE_PER_POINT":    (float, "USD_PRICE_PER_POINT"),
    "MINER_BATCH_SIZE":       (int,   "MINER_BATCH_SIZE"),
    "VALIDATION_FETCH_LIMIT": (int,   "VALIDATION_FETCH_LIMIT"),
    "MIN_PERCENT_PER_POINT":  (float, "MIN_PERCENT_PER_POINT"),
    "ENABLE_TWEET_SCORING":   (_as_bool, "ENABLE_TWEET_SCORING"),
    "ARTICLE_STORE_MAX_ARTICLES": (int, "ARTICLE_STORE_MAX_ARTICLES"),
    # Adaptive dispatch (RFC 2026-06-28)
    "ADAPTIVE_DISPATCH_ENABLED":  (_as_bool, "ADAPTIVE_DISPATCH_ENABLED"),
    "DISPATCH_WINDOW_MIN":        (int,   "DISPATCH_WINDOW_MIN"),
    "DISPATCH_WINDOW_CAP_PCT":    (float, "DISPATCH_WINDOW_CAP_PCT"),
    "DISPATCH_WINDOW_BUDGET":     (float, "DISPATCH_WINDOW_BUDGET"),
    "DISPATCH_WINDOW_GROW":       (float, "DISPATCH_WINDOW_GROW"),
    "DISPATCH_WINDOW_SHRINK":     (float, "DISPATCH_WINDOW_SHRINK"),
    "DISPATCH_LATE_FRACTION":     (float, "DISPATCH_LATE_FRACTION"),
    "DISPATCH_ACK_TIMEOUT_S":     (float, "DISPATCH_ACK_TIMEOUT_S"),
    "SCORING_LEASE_TTL_SECONDS":  (float, "SCORING_LEASE_TTL_SECONDS"),
    "DISPATCH_CHRONIC_TIMEOUT_N": (int,   "DISPATCH_CHRONIC_TIMEOUT_N"),
    "LIVENESS_TTL_S":             (int,   "LIVENESS_TTL_S"),
    "LIVENESS_SWEEP_INTERVAL_S":  (int,   "LIVENESS_SWEEP_INTERVAL_S"),
    "ADAPTIVE_PENALTY_SPLIT_ENABLED": (_as_bool, "ADAPTIVE_PENALTY_SPLIT_ENABLED"),
    "ADAPTIVE_MISSING_ANALYSIS_SPLIT_ENABLED": (_as_bool, "ADAPTIVE_MISSING_ANALYSIS_SPLIT_ENABLED"),
    "DISPATCH_COOLDOWN_SHADOW_MODE":       (_as_bool, "DISPATCH_COOLDOWN_SHADOW_MODE"),
    "DISPATCH_COOLDOWN_FAITHFULNESS_FLOOR": (float, "DISPATCH_COOLDOWN_FAITHFULNESS_FLOOR"),
    "DISPATCH_CONSEC_INVALID_N":           (int, "DISPATCH_CONSEC_INVALID_N"),
    "DISPATCH_INVALID_COOLDOWN_FIRST_S":   (int, "DISPATCH_INVALID_COOLDOWN_FIRST_S"),
    "DISPATCH_INVALID_COOLDOWN_MAX_S":     (int, "DISPATCH_INVALID_COOLDOWN_MAX_S"),
    "DISPATCH_FAILSTREAK_SHADOW_MODE":     (_as_bool, "DISPATCH_FAILSTREAK_SHADOW_MODE"),
    "DISPATCH_CONSEC_FAIL_N":              (int, "DISPATCH_CONSEC_FAIL_N"),
    # Adaptive per-miner batch size (2026-07-08) — local dispatch behavior, served so the
    # bounds are consistent fleet-wide (the per-miner state stays local).
    "ADAPTIVE_BATCH_SIZE_ENABLED": (_as_bool, "ADAPTIVE_BATCH_SIZE_ENABLED"),
    "MINER_BATCH_SIZE_MAX":       (int,   "MINER_BATCH_SIZE_MAX"),
    "MINER_BATCH_SIZE_MIN":       (int,   "MINER_BATCH_SIZE_MIN"),
    "BATCH_SIZE_GROW_STEP":       (int,   "BATCH_SIZE_GROW_STEP"),
    "BATCH_SIZE_SHRINK_FACTOR":   (float, "BATCH_SIZE_SHRINK_FACTOR"),
    # Scoring track (not gated by the dispatch flag): served so all validators match.
    "TIER3_THRESHOLD":            (float, "TIER3_THRESHOLD"),
    "CLONE_COSINE_THRESHOLD":     (float, "CLONE_COSINE_THRESHOLD"),
    "CLONE_DIFFERENTIAL_ENABLED": (_as_bool, "CLONE_DIFFERENTIAL_ENABLED"),
    "CLONE_DIVERGENCE_MARGIN":    (float, "CLONE_DIVERGENCE_MARGIN"),
    # Reputation-scoring track (served so all validators match).
    "REPUTATION_SCORING_ENABLED": (_as_bool, "REPUTATION_SCORING_ENABLED"),
    "REPUTATION_GATING_ENABLED":  (_as_bool, "REPUTATION_GATING_ENABLED"),
    "REPUTATION_EMA_ALPHA":       (float, "REPUTATION_EMA_ALPHA"),
    "REPUTATION_PRIOR":           (float, "REPUTATION_PRIOR"),
    "EMISSION_MIDPOINT":          (float, "EMISSION_MIDPOINT"),
    "EMISSION_GAIN":              (float, "EMISSION_GAIN"),
    "EMISSION_BONUS_CEILING":     (float, "EMISSION_BONUS_CEILING"),
    "EMISSION_BONUS_START":       (float, "EMISSION_BONUS_START"),
    "EMISSION_BONUS_FULL":        (float, "EMISSION_BONUS_FULL"),
    "EMISSION_N_MIN":             (int,   "EMISSION_N_MIN"),
    "AUDIT_SHADOW_ENABLED":       (_as_bool, "AUDIT_SHADOW_ENABLED"),
    "AUDIT_MAX_PER_BATCH":        (int,   "AUDIT_MAX_PER_BATCH"),
    "VALIDATION_SAMPLE_SIZE":     (int,   "VALIDATION_SAMPLE_SIZE"),
    "SAMPLING_SUBSTANTIVE_WEIGHT":(float, "SAMPLING_SUBSTANTIVE_WEIGHT"),
    "SUMMARY_AGREEMENT_FLOOR":    (float, "SUMMARY_AGREEMENT_FLOOR"),
    "TRIAGE_ENFORCED":            (_as_bool, "TRIAGE_ENFORCED"),
    # Weight window. Must also be present in the API's SUBNET_CONFIG or the
    # served value never reaches the elif below and the rollout does nothing.
    "WEIGHT_WINDOW_EPOCHS":       (_cast_window_epochs, "WEIGHT_WINDOW_EPOCHS"),
    "WEIGHT_WINDOW_EPOCHS_PREV":  (_cast_window_epochs, "WEIGHT_WINDOW_EPOCHS_PREV"),
    "WEIGHT_WINDOW_ACTIVE_BLOCK": (int,   "WEIGHT_WINDOW_ACTIVE_BLOCK"),
}

# Keys that must be identical on every validator. A local OVERRIDE_ on one of
# these is a weight difference that no config push can correct, so it is
# ignored and reported rather than applied.
_CONSENSUS_KEYS = {
    "WEIGHT_WINDOW_EPOCHS",
    "WEIGHT_WINDOW_EPOCHS_PREV",
    "WEIGHT_WINDOW_ACTIVE_BLOCK",
    # Scoring and emission: these decide points and the multiplier applied to them, so
    # a local override is a weight difference no config push can correct.
    "REPUTATION_SCORING_ENABLED",
    "REPUTATION_GATING_ENABLED",
    "REPUTATION_EMA_ALPHA",
    "REPUTATION_PRIOR",
    "EMISSION_MIDPOINT",
    "EMISSION_GAIN",
    "EMISSION_BONUS_CEILING",
    "EMISSION_BONUS_START",
    "EMISSION_BONUS_FULL",
    "EMISSION_N_MIN",
    "SAMPLING_SUBSTANTIVE_WEIGHT",
}

REMOTE_CONFIG_REFRESH_SECONDS = int(os.getenv("REMOTE_CONFIG_REFRESH_SECONDS", "3600"))
_remote_config_last_fetch: float = 0.0
_remote_config_lock = threading.Lock()
_wallet_ref = None
_applied_reset_ids: set = set()


def set_wallet(wallet) -> None:
    global _wallet_ref
    _wallet_ref = wallet


def _build_auth_headers() -> dict:
    if _wallet_ref is None:
        return {}
    try:
        from alpharidge_ai import __version__
        timestamp = time.time()
        message = f"alpharidge-ai-auth:{int(timestamp)}"
        signature = _wallet_ref.hotkey.sign(message).hex()
        return {
            "X-Auth-SS58Address": _wallet_ref.hotkey.ss58_address,
            "X-Auth-Signature": signature,
            "X-Auth-Message": message,
            "X-Auth-Timestamp": str(timestamp),
            "X-Validator-Version": __version__,
        }
    except Exception:
        return {}


def refresh_remote_config(force: bool = False) -> dict:
    """
    Fetch recommended config from the API.

    Values from the API are applied unless a local OVERRIDE_<key> env var exists.
    Returns the raw API response dict (empty on failure).
    """
    global _remote_config_last_fetch, BLACKLISTED_MINER_HOTKEYS

    now = time.time()
    if not force and (now - _remote_config_last_fetch) < REMOTE_CONFIG_REFRESH_SECONDS:
        return {}

    with _remote_config_lock:
        if not force and (now - _remote_config_last_fetch) < REMOTE_CONFIG_REFRESH_SECONDS:
            return {}

        api_url = MINER_API_URL
        if not api_url or api_url == "null":
            return {}

        try:
            headers = _build_auth_headers()
            resp = requests.get(f"{api_url}/config/subnet", headers=headers, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            _log_warning(f"[REMOTE_CONFIG] Failed to fetch config: {e}")
            return {}

        _remote_config_last_fetch = time.time()

        cfg = data.get("config", {})
        for key, (cast, attr) in _REMOTE_CONFIG_KEYS.items():
            override_val = os.getenv(f"OVERRIDE_{key}")
            if override_val is not None and key in _CONSENSUS_KEYS:
                _log_error(
                    f"[REMOTE_CONFIG] OVERRIDE_{key} is set and IGNORED: "
                    f"consensus key, the served value applies. Unset it."
                )
                override_val = None
            if override_val is not None:
                try:
                    globals()[attr] = cast(override_val)
                    _log_info(f"[REMOTE_CONFIG] {key} = {globals()[attr]} (local OVERRIDE)")
                except (ValueError, TypeError):
                    pass
            elif key in cfg:
                try:
                    globals()[attr] = cast(cfg[key])
                    _log_info(f"[REMOTE_CONFIG] {key} = {globals()[attr]} (from API)")
                except (ValueError, TypeError):
                    pass

        # Blacklisted hotkeys
        api_blacklist = set(data.get("blacklisted_hotkeys", []))
        local_override = os.getenv("OVERRIDE_BLACKLISTED_HOTKEYS")
        if local_override is not None:
            BLACKLISTED_MINER_HOTKEYS = set(hk.strip() for hk in local_override.split(",") if hk.strip())
            _log_info(f"[REMOTE_CONFIG] BLACKLISTED_MINER_HOTKEYS = {len(BLACKLISTED_MINER_HOTKEYS)} hotkeys (local OVERRIDE)")
        else:
            BLACKLISTED_MINER_HOTKEYS = api_blacklist
            if api_blacklist:
                _log_info(f"[REMOTE_CONFIG] BLACKLISTED_MINER_HOTKEYS = {len(api_blacklist)} hotkeys (from API)")

        # Version check
        min_ver = data.get("min_validator_version", "0.0.0")
        try:
            from alpharidge_ai import __version__
            c = tuple(int(x) for x in __version__.split("."))
            m = tuple(int(x) for x in min_ver.split("."))
            if c < m:
                _log_warning(
                    f"[REMOTE_CONFIG] Validator version {__version__} is below minimum "
                    f"{min_ver} — the API will not distribute tweets until you update. "
                    f"Run 'git pull && pm2 restart' to fix."
                )
        except Exception:
            pass

        # Reset signals
        _handle_reset_signals(data)

        return data


def _handle_reset_signals(data: dict) -> None:
    """Process one-shot reset directives from the API."""
    global _applied_reset_ids

    reset_epoch = data.get("reset_broadcasts_before_epoch", -1)
    purge_hotkeys = data.get("purge_broadcast_hotkeys", [])
    reset_scores_id = data.get("reset_scores_id", "")

    reset_id = f"epoch:{reset_epoch}|purge:{','.join(sorted(purge_hotkeys))}|scores:{reset_scores_id}"
    if reset_id in _applied_reset_ids:
        return
    if reset_epoch < 0 and not purge_hotkeys and not reset_scores_id:
        return

    _applied_reset_ids.add(reset_id)
    _log_info(f"[REMOTE_CONFIG] Reset signal received: reset_epoch={reset_epoch}, purge_hotkeys={len(purge_hotkeys)}, reset_scores_id={reset_scores_id}")

    # The actual reset is performed by the validation_client which has
    # access to the broadcast stores. We store the directives here.
    globals()["_pending_reset_epoch"] = reset_epoch
    globals()["_pending_purge_hotkeys"] = list(purge_hotkeys)
    if reset_scores_id:
        globals()["_pending_reset_scores"] = True


def get_pending_resets() -> tuple:
    """
    Return and clear pending reset directives.
    Returns (reset_epoch: int, purge_hotkeys: list[str], reset_scores: bool).
    """
    epoch = globals().pop("_pending_reset_epoch", -1)
    hotkeys = globals().pop("_pending_purge_hotkeys", [])
    reset_scores = globals().pop("_pending_reset_scores", False)
    return epoch, hotkeys, reset_scores