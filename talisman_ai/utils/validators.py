"""
Validator hotkey selection utilities.

We use this to decide which peers we broadcast validator↔validator synapses to.

Design goals:
- Prefer validator-permit hotkeys with stake >= threshold
- Cache results briefly to avoid frequent metagraph sync
- Allow an optional manual allowlist (useful for testing)
"""

from __future__ import annotations

import threading
import time
from typing import Dict, List, Optional, Sequence

import bittensor as bt

from talisman_ai import config

_CACHE_LOCK = threading.Lock()
_CACHE_TS: float = 0.0
_CACHED_VALIDATOR_HOTKEYS: List[str] = []
_CACHED_VALIDATOR_DATA: List[Dict[str, str]] = []


def _defaults():
    stake_threshold = float(getattr(config, "VALIDATOR_STAKE_THRESHOLD", 0.0))
    cache_seconds = float(getattr(config, "VALIDATOR_CACHE_SECONDS", 120.0))
    allow_manual = bool(getattr(config, "ALLOW_MANUAL_VALIDATOR_HOTKEYS", False))
    manual_hotkeys = list(getattr(config, "MANUAL_VALIDATOR_HOTKEYS", [])) or []
    return stake_threshold, cache_seconds, allow_manual, manual_hotkeys


def get_validator_data(
    *,
    metagraph: Optional["bt.metagraph.Metagraph"] = None,
    subtensor: Optional["bt.subtensor"] = None,
    netuid: Optional[int] = None,
    stake_threshold: Optional[float] = None,
    cache_seconds: Optional[float] = None,
) -> List[Dict[str, str]]:
    """
    Get list of validator data (name and hotkey pairs), cached.

    Filters validators by:
    - validator_permit == True
    - stake >= stake_threshold
    """
    global _CACHE_TS, _CACHED_VALIDATOR_HOTKEYS, _CACHED_VALIDATOR_DATA

    default_stake_threshold, default_cache_seconds, _, _ = _defaults()
    stake_threshold = default_stake_threshold if stake_threshold is None else float(stake_threshold)
    cache_seconds = default_cache_seconds if cache_seconds is None else float(cache_seconds)

    with _CACHE_LOCK:
        now = time.time()
        if (not _CACHED_VALIDATOR_DATA) or (now - _CACHE_TS >= cache_seconds):
            try:
                if metagraph is None:
                    if subtensor is None:
                        subtensor = bt.subtensor(config=getattr(config, "bt_config", None)) if hasattr(config, "bt_config") else bt.subtensor()
                    if netuid is None:
                        raise ValueError("netuid must be provided if metagraph is not provided")
                    metagraph = subtensor.metagraph(netuid)
                    # Lite sync is fine; we don't need full recency for allowlisting.
                    metagraph.sync(subtensor=subtensor, lite=True)

                validator_hotkeys = [
                    hk
                    for uid, hk in enumerate(metagraph.hotkeys)
                    if bool(metagraph.validator_permit[uid]) and float(metagraph.S[uid]) >= stake_threshold
                ]

                _CACHED_VALIDATOR_HOTKEYS = validator_hotkeys
                _CACHED_VALIDATOR_DATA = [
                    {"name": f"Validator {i + 1}", "hotkey": hk}
                    for i, hk in enumerate(validator_hotkeys)
                ]
                _CACHE_TS = now
            except Exception:
                # If cache is empty, return empty. Otherwise serve stale cache.
                if not _CACHED_VALIDATOR_DATA:
                    return []

        return list(_CACHED_VALIDATOR_DATA)


def get_validator_hotkeys(
    *,
    metagraph: Optional["bt.metagraph.Metagraph"] = None,
    subtensor: Optional["bt.subtensor"] = None,
    netuid: Optional[int] = None,
    stake_threshold: Optional[float] = None,
    cache_seconds: Optional[float] = None,
    allow_manual_hotkeys: Optional[bool] = None,
    manual_hotkeys: Optional[Sequence[str]] = None,
) -> List[str]:
    """
    Get list of whitelisted validator hotkeys.

    Includes metagraph validators (permit + stake threshold) and optionally merges
    a manual allowlist.
    """
    global _CACHED_VALIDATOR_HOTKEYS

    default_stake_threshold, default_cache_seconds, default_allow_manual, default_manual_hotkeys = _defaults()
    stake_threshold = default_stake_threshold if stake_threshold is None else float(stake_threshold)
    cache_seconds = default_cache_seconds if cache_seconds is None else float(cache_seconds)
    allow_manual_hotkeys = default_allow_manual if allow_manual_hotkeys is None else bool(allow_manual_hotkeys)
    manual_hotkeys = default_manual_hotkeys if manual_hotkeys is None else list(manual_hotkeys)

    # Ensure cache is populated.
    get_validator_data(
        metagraph=metagraph,
        subtensor=subtensor,
        netuid=netuid,
        stake_threshold=stake_threshold,
        cache_seconds=cache_seconds,
    )

    metagraph_validators = list(_CACHED_VALIDATOR_HOTKEYS)
    if allow_manual_hotkeys and manual_hotkeys:
        return list(set(metagraph_validators + list(manual_hotkeys)))
    return metagraph_validators


