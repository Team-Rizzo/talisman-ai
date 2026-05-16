import time
import bittensor as bt
from typing import Dict, Set, Tuple


BACKOFF_SCHEDULE = [30, 60, 120, 300, 600]  # seconds
CONSECUTIVE_FAILURES_BEFORE_COOLDOWN = 10
MAX_INFLIGHT_PER_MINER = 4


class MinerCooldownTracker:
    """
    Tracks miner dispatch failures with exponential backoff, and limits
    concurrent in-flight dispatches per miner to avoid overwhelming healthy ones.
    """

    def __init__(self):
        # {hotkey: (consecutive_fails, cooldown_level, cooldown_until)}
        self._state: Dict[str, Tuple[int, int, float]] = {}
        self._inflight: Dict[str, int] = {}

    # ---- In-flight tracking ----

    def try_acquire(self, hotkey: str) -> bool:
        """Returns True if the miner has capacity for another dispatch."""
        count = self._inflight.get(hotkey, 0)
        if count >= MAX_INFLIGHT_PER_MINER:
            return False
        self._inflight[hotkey] = count + 1
        return True

    def release(self, hotkey: str) -> None:
        count = self._inflight.get(hotkey, 0)
        if count > 1:
            self._inflight[hotkey] = count - 1
        elif hotkey in self._inflight:
            del self._inflight[hotkey]

    # ---- Cooldown tracking ----

    def record_failure(self, hotkey: str) -> None:
        consec, level, _ = self._state.get(hotkey, (0, 0, 0.0))
        consec += 1

        if consec < CONSECUTIVE_FAILURES_BEFORE_COOLDOWN:
            self._state[hotkey] = (consec, level, 0.0)
            return

        level = min(level + 1, len(BACKOFF_SCHEDULE))
        backoff_idx = min(level - 1, len(BACKOFF_SCHEDULE) - 1)
        cooldown_secs = BACKOFF_SCHEDULE[backoff_idx]
        self._state[hotkey] = (consec, level, time.time() + cooldown_secs)
        bt.logging.info(
            f"[COOLDOWN] Miner {hotkey[:12]}.. {consec} consecutive failures, "
            f"cooldown for {cooldown_secs}s (level {level})"
        )

    def record_success(self, hotkey: str) -> None:
        if hotkey in self._state:
            _, level, _ = self._state[hotkey]
            if level > 0:
                bt.logging.info(f"[COOLDOWN] Miner {hotkey[:12]}.. recovered, clearing cooldown")
            del self._state[hotkey]

    def is_on_cooldown(self, hotkey: str) -> bool:
        entry = self._state.get(hotkey)
        if entry is None:
            return False
        _, _, cooldown_until = entry
        return cooldown_until > 0 and time.time() < cooldown_until

    def get_cooled_down_hotkeys(self) -> Set[str]:
        now = time.time()
        return {hk for hk, (_, _, until) in self._state.items() if until > 0 and now < until}

    def prune(self, active_hotkeys: Set[str]) -> None:
        stale = [hk for hk in self._state if hk not in active_hotkeys]
        for hk in stale:
            del self._state[hk]
        stale_inflight = [hk for hk in self._inflight if hk not in active_hotkeys]
        for hk in stale_inflight:
            del self._inflight[hk]
        if stale:
            bt.logging.debug(f"[COOLDOWN] Pruned {len(stale)} stale hotkey(s)")

    def stats(self) -> Tuple[int, int]:
        """Returns (total_tracked, currently_on_cooldown)."""
        now = time.time()
        on_cooldown = sum(1 for _, (_, _, until) in self._state.items() if until > 0 and now < until)
        return len(self._state), on_cooldown
