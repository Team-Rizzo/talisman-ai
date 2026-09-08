import requests
import time
import logging
import bittensor as bt
from bittensor.core.subtensor import Subtensor
from alpharidge_ai import config
from substrateinterface import SubstrateInterface
from typing import Any, Optional
import numpy as np
from alpharidge_ai.models.reward import Reward

logger = logging.getLogger(__name__)

NETUID = 45
CACHE_TTL_SECONDS = 300  # 5 minutes
_storage_cache: dict[tuple, tuple[Any, float]] = {}  # {cache_key: (value, timestamp)}
BLOCKS_PER_DAY = 7200
ALPHA_DECIMALS = 9
ALPHA_UNIT = 10 ** ALPHA_DECIMALS

# Local TAO price cache (for fallback)
_tao_price_cache: dict[str, Any] = {"price": None, "timestamp": 0.0}
TAO_PRICE_CACHE_TTL = 300  # 5 minutes local cache

# Shared Subtensor instance to avoid repeated connections
_subtensor_instance: Optional[Subtensor] = None

def _get_subtensor() -> Subtensor:
    """Get or create a shared Subtensor instance."""
    global _subtensor_instance
    if _subtensor_instance is None:
        _subtensor_instance = Subtensor()
        bt.logging.debug("[burn] Created shared Subtensor instance")
    return _subtensor_instance

def _reset_subtensor():
    """Reset the shared Subtensor instance (e.g., on connection errors)."""
    global _subtensor_instance
    if _subtensor_instance is not None:
        try:
            # Close the WebSocket connection before discarding.
            if hasattr(_subtensor_instance, 'substrate') and _subtensor_instance.substrate:
                _subtensor_instance.substrate.close()
        except Exception:
            pass
    _subtensor_instance = None
    bt.logging.debug("[burn] Reset shared Subtensor instance")


def tao_price() -> float:
    """
    Fetch TAO/USD price from the coordination API (cached by TaoStats).
    Returns cached value if API is unavailable.
    On cold start (no cache), retries up to 3 times with backoff.
    """
    now = time.time()
    
    # Return cached value if fresh
    if _tao_price_cache["price"] is not None:
        if now - _tao_price_cache["timestamp"] < TAO_PRICE_CACHE_TTL:
            return _tao_price_cache["price"]
    
    # Fetch from coordination API
    api_url = config.MINER_API_URL
    if not api_url or api_url == "null":
        if _tao_price_cache["price"] is not None:
            logger.warning("MINER_API_URL not configured, using stale TAO price")
            return _tao_price_cache["price"]
        raise RuntimeError("MINER_API_URL not configured and no cached price available")
    
    # Determine retry behavior: more retries on cold start
    is_cold_start = _tao_price_cache["price"] is None
    max_attempts = 3 if is_cold_start else 1
    retry_delays = [2, 5, 10]
    
    last_error = None
    for attempt in range(max_attempts):
        try:
            response = requests.get(
                f"{api_url.rstrip('/')}/price/tao-usd",
                timeout=5.0,
            )
            response.raise_for_status()
            data = response.json()
            price = float(data["price_usd"])
            _tao_price_cache["price"] = price
            _tao_price_cache["timestamp"] = time.time()
            logger.debug(f"TAO price from API: ${price:.2f}")
            return price
        except Exception as e:
            last_error = e
            if attempt < max_attempts - 1:
                delay = retry_delays[attempt]
                logger.warning(f"Failed to fetch TAO price (attempt {attempt + 1}/{max_attempts}): {e}. Retrying in {delay}s...")
                time.sleep(delay)
    
    # All attempts failed
    logger.warning(f"Failed to fetch TAO price from API: {last_error}")
    if _tao_price_cache["price"] is not None:
        logger.warning(f"Using stale TAO price: ${_tao_price_cache['price']:.2f}")
        return _tao_price_cache["price"]
    raise RuntimeError(f"Failed to fetch TAO price after {max_attempts} attempts: {last_error}")

def get_alpha_per_point():
    try:
        sub = _get_subtensor()
        price = sub.get_subnet_price(netuid=45)
        alpha_price = price.tao*tao_price()
        return alpha_price / config.USD_PRICE_PER_POINT
    except Exception as e:
        bt.logging.warning(f"[get_alpha_per_point] Error: {e}, resetting subtensor")
        _reset_subtensor()
        raise 

def alpha_usd() -> float:
    """One alpha in USD, from the same on-chain source the weight path reads."""
    sub = _get_subtensor()
    return sub.get_subnet_price(netuid=45).tao * tao_price()


def get_storage_value(module: str, storage_fn: str, params=None) -> Any:
    cache_key = (module, storage_fn, tuple(params) if params else ())
    now = time.time()
    
    # Check cache
    if cache_key in _storage_cache:
        cached_value, cached_time = _storage_cache[cache_key]
        if now - cached_time < CACHE_TTL_SECONDS:
            return cached_value
    
    try:
        # Use shared Subtensor instance to avoid repeated connections
        sub = _get_subtensor()
        result = sub.substrate.query(module, storage_fn, params or [])
        # bittensor returns BittensorScaleType with .value attribute
        if hasattr(result, 'value'):
            if isinstance(result.value, int):
                value = result.value
            elif isinstance(result.value, str):
                value = int(result.value)
            elif result.value is not None:
                value = int(result.value)
            else:
                value = 0
        elif isinstance(result, int):
            value = result
        elif hasattr(result, "__getitem__"):
            try:
                value = int(result[0])
            except Exception:
                value = int(result)
        else:
            value = int(result)
        
        _storage_cache[cache_key] = (value, now)
        return value
    except Exception as e:
        bt.logging.warning(f"get_storage_value failed for {module}.{storage_fn}: {e}")
        _reset_subtensor()  # Reset on error so next call creates fresh connection
        return 0

def get_pending_server_emission(netuid: int) -> int:
    return get_storage_value('SubtensorModule', 'PendingServerEmission', [netuid])

def get_blocks_since_last_step(netuid: int) -> int:
    return get_storage_value('SubtensorModule', 'BlocksSinceLastStep', [netuid])

def get_subnet_tempo(netuid: int) -> int:
    tempo = get_storage_value('SubtensorModule', 'Tempo', [netuid])
    return tempo + 1 if tempo is not None else 360

def get_subnet_alpha_out_emission(netuid: int) -> int:
    return get_storage_value('SubtensorModule', 'SubnetAlphaOutEmission', [netuid])

def get_miner_alpha_per_block() -> float:
    # SubnetAlphaOutEmission is RAO/block (1e9 RAO = 1 alpha). Miner share = (1 - 18% owner) * 50%.
    # Returns alpha/block; caller multiplies by EPOCH_LENGTH (blocks) for the per-epoch total.
    return (get_subnet_alpha_out_emission(45) * (1 - .18) * 0.5) / 10**9

def get_percent_needed_to_equal_points(points: int, window_blocks: int) -> float:
    """
    Returns the percentage of the total miner alpha needed to equal the given points
    over window_blocks blocks.

    window_blocks is required: the points passed in span the whole window, so the
    divisor must span it too. A default here would silently under-divide.
    """
    alpha_per_point = get_alpha_per_point()
    if alpha_per_point == 0:
        return 0

    # Convert points to equivalent alpha needed
    alpha_needed = points / alpha_per_point

    total_alpha_over_window = get_miner_alpha_per_block() * window_blocks
    if total_alpha_over_window == 0:
        return 0

    return (alpha_needed / total_alpha_over_window) * 100


def calculate_weights(rewards: list[Reward], metagraph, window_blocks: int, *, context: str = "") -> np.ndarray:
    """
    Calculates the weights for a given list of points and hotkeys.

    window_blocks is the block span the points cover; it is required so that an
    un-updated caller fails loudly instead of under-dividing.
    context is free-form text from the caller for the summary log line. It is
    keyword-only so it can never be transposed with window_blocks.

    Returns: np.ndarray of shape (metagraph.n,)
    """
    if window_blocks <= 0:
        raise ValueError(f"[calculate_weights] window_blocks must be positive, got {window_blocks}")

    MIN_PERCENT_PER_POINT = config.MIN_PERCENT_PER_POINT

    weights = np.zeros(metagraph.n if hasattr(metagraph, 'n') else len(metagraph), dtype=np.float64)

    percent_needed_for_each_hotkey = {}
    total_percent_needed = 0.0
    total_min_percent = 0.0
    min_percent_hits = 0
    for reward in rewards:
        try:
            percent_needed = get_percent_needed_to_equal_points(reward.reward, window_blocks)
        except Exception as e:
            bt.logging.warning(f"[calculate_weights] Failed to get percent for reward={reward.reward}: {e}")
            percent_needed = 0

        # Apply minimum floor: each point gets at least MIN_PERCENT_PER_POINT.
        # The floor is a rate, expressed per BLOCK_LENGTH blocks, so it stays level
        # as the window grows — the points scale with the window and the divisor
        # cancels that back out. Without this the floor would overtake the
        # calculated percent at a large enough window and pin the burn.
        points_per_epoch = reward.reward * config.BLOCK_LENGTH / window_blocks
        min_percent_for_reward = points_per_epoch * MIN_PERCENT_PER_POINT
        total_min_percent += min_percent_for_reward
        if percent_needed < min_percent_for_reward:
            min_percent_hits += 1
            bt.logging.debug(f"[calculate_weights] Scaling up percent for {reward.reward} points: {percent_needed:.6f}% -> {min_percent_for_reward:.6f}%")
            percent_needed = min_percent_for_reward

        percent_needed_for_each_hotkey[reward.hotkey] = percent_needed
        total_percent_needed += percent_needed

    # max_possible_burn: what burn would be if every miner sat exactly on the floor.
    max_possible_burn = 1 - (min(total_min_percent, 100) / 100)
    bt.logging.info(
        f"[calculate_weights] window_blocks={window_blocks} {context}"
        f"total_percent_needed={total_percent_needed:.4f}% rewards_count={len(rewards)} "
        f"min_percent_hits={min_percent_hits} max_possible_burn={max_possible_burn:.4f}"
    )

    if total_percent_needed > 0:
        # If over 100%, scale all values down so the total sums to 100% or less
        scale = 1.0
        if total_percent_needed > 100:
            scale = 100.0 / total_percent_needed if total_percent_needed != 0 else 0.0
        for reward in rewards:
            if reward.hotkey in metagraph.hotkeys:
                uid = metagraph.hotkeys.index(reward.hotkey)
                percent = percent_needed_for_each_hotkey.get(reward.hotkey, 0)
                weights[uid] = (percent * scale / 100.0) if scale != 0 else 0.0  # convert percent to fraction
    else:
        # If total_percent_needed is 0, avoid dividing by zero in denominator
        for reward in rewards:
            if reward.hotkey in metagraph.hotkeys:
                uid = metagraph.hotkeys.index(reward.hotkey)
                percent = percent_needed_for_each_hotkey.get(reward.hotkey, 0)
                weights[uid] = (percent / 100.0) if 100.0 != 0 else 0.0

    weights[config.BURN_UID] = (1 - (min(total_percent_needed, 100) / 100))
    
    # Log non-zero weights for debugging
    non_zero = [(i, w) for i, w in enumerate(weights) if w > 0]
    bt.logging.info(f"[calculate_weights] Non-zero weights: {non_zero[:10]}... (burn_uid={config.BURN_UID}, burn_weight={weights[config.BURN_UID]:.4f})")

    return weights