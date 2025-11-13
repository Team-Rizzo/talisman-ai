"""
Centralized configuration loader for talisman_ai_subnet.
Loads environment variables from .miner_env and .vali_env files.

This module should be imported at the top of any module that needs configuration:
    from talisman_ai import config
    
Then access config values as:
    config.MODEL
    config.MAX_POSTS
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

# Maximum number of posts to process before stopping
# Set to 0 or a very large number for unlimited processing
# Default: 1000 if not specified
MAX_POSTS = int(os.getenv("MAX_POSTS", "1000"))

# Scraping configuration
# Interval between scrape cycles in seconds (default: 300 = 5 minutes)
SCRAPE_INTERVAL_SECONDS = int(os.getenv("SCRAPE_INTERVAL_SECONDS", "300"))
# Number of posts to scrape per cycle (default: 1)
POSTS_PER_SCRAPE = int(os.getenv("POSTS_PER_SCRAPE", "1"))
# Number of posts to submit per cycle (default: 1)
# Set this lower than POSTS_PER_SCRAPE to analyze more posts but submit fewer
POSTS_TO_SUBMIT = int(os.getenv("POSTS_TO_SUBMIT", "1"))


# ============================================================================
# Validator-Specific Configuration
# ============================================================================

# Miner API configuration
MINER_API_URL = os.getenv("MINER_API_URL", "null")
BATCH_HTTP_TIMEOUT = float(os.getenv("BATCH_HTTP_TIMEOUT", "30.0"))
VOTE_ENDPOINT = os.getenv("VOTE_ENDPOINT", "null")
BATCH_POLL_SECONDS = int(os.getenv("BATCH_POLL_SECONDS", "10"))

