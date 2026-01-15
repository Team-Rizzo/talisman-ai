"""
Centralized configuration loader for talisman_ai_subnet.
Loads environment variables from .miner_env and .vali_env files.

This module should be imported at the top of any module that needs configuration:
    from talisman_ai import config
    
Then access config values as:
    config.MODEL
    config.BLOCKS_PER_WINDOW
    etc.
"""

from pathlib import Path
import os

# Find the talisman_ai_subnet root directory
# This file is at talisman_ai_subnet/talisman_ai/config.py
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
BLOCK_LENGTH = int(os.getenv("BLOCK_LENGTH", "100"))
START_BLOCK = int(os.getenv("START_BLOCK", "0"))

# Validator -> miner dispatch behavior (push-based mining).
# The validator should only "dispatch" work; miners will push results back asynchronously.
MINER_SEND_TIMEOUT = float(os.getenv("MINER_SEND_TIMEOUT", "6.0"))
VALIDATOR_MINER_QUERY_CONCURRENCY = int(os.getenv("VALIDATOR_MINER_QUERY_CONCURRENCY", "8"))
VALIDATOR_MAX_PENDING_MINER_TASKS = int(os.getenv("VALIDATOR_MAX_PENDING_MINER_TASKS", "256"))

# Tweet store configuration
TWEET_STORE_LOCATION = os.getenv("TWEET_STORE_LOCATION", str(_SUBNET_ROOT / ".tweet_store.json"))
TWEET_MAX_PROCESS_TIME = float(os.getenv("TWEET_MAX_PROCESS_TIME", "300.0"))  # 5 minutes default

# Telegram store configuration
TELEGRAM_STORE_LOCATION = os.getenv("TELEGRAM_STORE_LOCATION", str(_SUBNET_ROOT / ".telegram_store.json"))

# Message max process time (shared for tweets and telegram, with fallback chain for backward compatibility)
MESSAGE_MAX_PROCESS_TIME = float(os.getenv("MESSAGE_MAX_PROCESS_TIME", os.getenv("TWEET_MAX_PROCESS_TIME", "300.0")))

# Penalty and reward store configuration
PENALTY_STORE_LOCATION = os.getenv("PENALTY_STORE_LOCATION", str(_SUBNET_ROOT / ".penalty_store.json"))
REWARD_STORE_LOCATION = os.getenv("REWARD_STORE_LOCATION", str(_SUBNET_ROOT / ".reward_store.json"))

USD_PRICE_PER_POINT = float(os.getenv("USD_PRICE_PER_POINT", "0.020"))
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