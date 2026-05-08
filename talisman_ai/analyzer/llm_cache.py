"""
Thread-safe in-memory TTL cache for LLM classification results.

LLM calls with temperature=0 are deterministic — identical input text always
produces the same classification. Caching avoids redundant API calls when the
same post/message is validated multiple times within a short window (e.g.
multiple miners submitting the same tweet).
"""

import threading
import time
import bittensor as bt


class LLMCache:
    """Thread-safe in-memory cache with TTL expiration and FIFO eviction."""

    def __init__(self, max_size: int = 1024, ttl_seconds: float = 300.0):
        self._cache: dict = {}
        self._timestamps: dict = {}
        self._lock = threading.Lock()
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._hits = 0
        self._misses = 0

    def get(self, key: str):
        with self._lock:
            if key in self._cache:
                age = time.time() - self._timestamps[key]
                if age < self._ttl:
                    self._hits += 1
                    return self._cache[key]
                del self._cache[key]
                del self._timestamps[key]
            self._misses += 1
            return None

    def put(self, key: str, value):
        with self._lock:
            if len(self._cache) >= self._max_size and key not in self._cache:
                oldest_key = min(self._timestamps, key=self._timestamps.get)
                del self._cache[oldest_key]
                del self._timestamps[oldest_key]
            self._cache[key] = value
            self._timestamps[key] = time.time()

    def log_stats(self, label: str = "LLM_CACHE"):
        with self._lock:
            total = self._hits + self._misses
            rate = (self._hits / total * 100) if total > 0 else 0.0
            bt.logging.info(
                f"[{label}] size={len(self._cache)} "
                f"hits={self._hits} misses={self._misses} "
                f"hit_rate={rate:.1f}%"
            )
