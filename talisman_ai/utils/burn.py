import requests
import time
import logging
from bittensor.core.subtensor import Subtensor
from talisman_ai import config
from substrateinterface import SubstrateInterface
from typing import Any, Optional
import numpy as np
from talisman_ai.models.reward import Reward

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
    sub = Subtensor()
    price = sub.get_subnet_price(netuid=45)
    alpha_price = price.tao*tao_price()
    return alpha_price / config.USD_PRICE_PER_POINT 

def get_storage_value(module: str, storage_fn: str, params=None) -> Any:
    cache_key = (module, storage_fn, tuple(params) if params else ())
    now = time.time()
    
    # Check cache
    if cache_key in _storage_cache:
        cached_value, cached_time = _storage_cache[cache_key]
        if now - cached_time < CACHE_TTL_SECONDS:
            return cached_value
    
    try:
        substrate = SubstrateInterface(config.FINNEY_RPC)
        result = substrate.query(module, storage_fn, params or [])
        # substrateinterface returns an object with .value attribute or directly as an integer.
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
    except Exception:
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
    return (get_subnet_alpha_out_emission(45) * (1 - .18) * 0.5) * 7200 / 2 / 10**9

def get_percent_needed_to_equal_points(points: int) -> float:
    """
    Returns the percentage of the total miner alpha needed to equal the given points over the epoch length.
    """
    alpha_per_point = get_alpha_per_point()
    if alpha_per_point == 0:
        return 0
    
    # Convert points to equivalent alpha needed
    alpha_needed = points / alpha_per_point
    
    total_alpha_over_epoch = get_miner_alpha_per_block() * config.EPOCH_LENGTH
    if total_alpha_over_epoch == 0:
        return 0
    
    return (alpha_needed / total_alpha_over_epoch) * 100


def calculate_weights(rewards: list[Reward], metagraph) -> np.ndarray:
    """
    Calculates the weights for a given list of points and hotkeys.
    Returns: np.ndarray of shape (metagraph.n,)
    """
    weights = np.zeros(metagraph.n if hasattr(metagraph, 'n') else len(metagraph), dtype=np.float64)

    percent_needed_for_each_hotkey = {}
    total_percent_needed = 0.0
    for reward in rewards:
        percent_needed = get_percent_needed_to_equal_points(reward.reward)
        percent_needed_for_each_hotkey[reward.hotkey] = percent_needed
        total_percent_needed += percent_needed

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

    return weights