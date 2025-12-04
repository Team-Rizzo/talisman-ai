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
# Note: We don't print file paths to avoid leaking credential locations in logs
try:
    from dotenv import load_dotenv
    
    _config_loaded = False
    
    # Load miner env file (if it exists)
    if _MINER_ENV_PATH.exists():
        load_dotenv(str(_MINER_ENV_PATH), override=True)
        _config_loaded = True
    
    # Load validator env file (if it exists)
    # Note: validator vars will override miner vars if both exist
    if _VALI_ENV_PATH.exists():
        load_dotenv(str(_VALI_ENV_PATH), override=True)
        _config_loaded = True
    
    # Only log at debug level, don't expose paths
    if not _config_loaded:
        import logging
        logging.debug("[CONFIG] No env files found, using system environment variables")
        
except ImportError:
    pass  # Silently use system environment variables


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

# Scraping configuration
# DEPRECATED: These are no longer used with block-based scraping approach.
# The miner now uses BLOCKS_PER_WINDOW and MAX_SUBMISSIONS_PER_WINDOW instead.
# Keeping these for backward compatibility but they have no effect.
SCRAPE_INTERVAL_SECONDS = int(os.getenv("SCRAPE_INTERVAL_SECONDS", "300"))  # Unused - block-based now
MAX_RESULTS = int(os.getenv("MAX_RESULTS", "10"))  # Unused - count parameter used instead
POSTS_PER_SCRAPE = int(os.getenv("POSTS_PER_SCRAPE", "1"))  # Unused - posts_per_window used instead
POSTS_TO_SUBMIT = int(os.getenv("POSTS_TO_SUBMIT", "1"))  # Unused - all scraped posts are submitted

# API v2 rate limit configuration
# Maximum submissions per block window (default: 5, matches API v2 default)
# Must match MAX_SUBMISSION_RATE in api_v2 if using API v2
MAX_SUBMISSIONS_PER_WINDOW = int(os.getenv("MAX_SUBMISSIONS_PER_WINDOW", os.getenv("MAX_SUBMISSION_RATE", "5")))
# Blocks per window (default: 100 blocks, ~20 minutes at 12s per block)
BLOCKS_PER_WINDOW = int(os.getenv("BLOCKS_PER_WINDOW", "100"))


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

