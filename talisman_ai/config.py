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
# Normalize empty strings to "null" (dotenv may set empty values as "")
def _getenv_or_null(key: str, default: str = "null") -> str:
    """Get env var, treating empty strings as missing (returns default)."""
    value = os.getenv(key, default)
    return default if (value == "" or value.strip() == "") else value

MODEL = _getenv_or_null("MODEL", "null")
API_KEY = _getenv_or_null("API_KEY", "null")
LLM_BASE = _getenv_or_null("LLM_BASE", "null")

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

# API v2 rate limit configuration
# IMPORTANT: These values MUST match the API server's configuration.
# Changing these without updating the API server will cause rate limit mismatches.
# Maximum submissions per block window (default: 5, matches API v2 default)
# Must match MAX_SUBMISSION_RATE in api/database.py if using API v2
MAX_SUBMISSIONS_PER_WINDOW = int(os.getenv("MAX_SUBMISSIONS_PER_WINDOW", os.getenv("MAX_SUBMISSION_RATE", "5")))
# Blocks per window (default: 100 blocks, ~20 minutes at 12s per block)
# Must match BLOCKS_PER_WINDOW in api/database.py
BLOCKS_PER_WINDOW = int(os.getenv("BLOCKS_PER_WINDOW", "100"))

# Miner polling configuration
# Approximate block time (seconds per block)
POLL_INTERVAL_SECONDS = float(os.getenv("POLL_INTERVAL_SECONDS", "12.0"))
# Check status endpoint every N poll iterations (~36 seconds with default values)
STATUS_CHECK_INTERVAL = int(os.getenv("STATUS_CHECK_INTERVAL", "3"))

# API retry configuration
MAX_SUBMIT_ATTEMPTS = int(os.getenv("MAX_SUBMIT_ATTEMPTS", "3"))
SUBMIT_BACKOFF_BASE_SECONDS = float(os.getenv("SUBMIT_BACKOFF_BASE_SECONDS", "3"))

# Post scraper configuration
# Comma-separated list of keywords to search for
SCRAPER_KEYWORDS = os.getenv("SCRAPER_KEYWORDS", "sn45,talismanai,sn13,sn64").split(",")


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

