import math
import time
import bittensor as bt
from collections import deque
from typing import Dict, Set, Tuple


BACKOFF_SCHEDULE = [30, 60, 120, 300, 600]  # seconds
CONSECUTIVE_FAILURES_BEFORE_COOLDOWN = 10
MAX_INFLIGHT_PER_MINER = 4
LATENCY_WINDOW = 20  # rolling per-miner batch round-trip samples for the median telemetry
EVENT_BUFFER_MAX = 2000  # bound the display-only event buffer if the API is unreachable


def _cfg(name, default):
    """Read an adaptive-dispatch knob from config lazily — avoids an import cycle
    and lets remote-config updates take effect without a restart."""
    try:
        from alpharidge_ai import config
        return getattr(config, name, default)
    except Exception:
        return default


class MinerCooldownTracker:
    """
    Tracks miner dispatch failures with exponential backoff, and limits
    concurrent in-flight dispatches per miner to avoid overwhelming healthy ones.

    Adaptive dispatch (RFC 2026-06-28): when constructed with ``adaptive=True`` AND
    ``config.ADAPTIVE_DISPATCH_ENABLED`` is on, the static per-miner in-flight cap is
    replaced by a dynamic congestion *window* that grows on clean, on-time completion
    and shrinks on invalid / timeout. Only the article tracker is adaptive; tweet and
    telegram trackers stay static. With the flag off (or adaptive=False) behaviour is
    identical to before.
    """

    def __init__(self, adaptive: bool = False):
        # {hotkey: (consecutive_fails, cooldown_level, cooldown_until)}
        self._state: Dict[str, Tuple[int, int, float]] = {}
        self._inflight: Dict[str, int] = {}

        # ---- Adaptive dispatch state (RFC 2026-06-28) ----
        self._adaptive = adaptive
        self._window: Dict[str, float] = {}       # per-miner congestion window
        self._consec_to: Dict[str, int] = {}      # consecutive lease timeouts (non-response)
        self._covered_ep: Dict[str, int] = {}     # last epoch given a coverage batch
        self._cap: float = None                   # per-tick anti-monopoly cap; None => from config

        # ---- Faithfulness cooldown (2026-07-09) ----
        # Counters kept separate from _state so the success path doesn't clear them.
        self._consec_inv: Dict[str, int] = {}     # consecutive low-faithfulness batches
        self._inv_level: Dict[str, int] = {}      # cooldown escalation level
        self._inv_until: Dict[str, float] = {}    # cooldown expiry (separate from _state)
        self._last_faith: Dict[str, float] = {}   # last observed min-faithfulness (telemetry)
        self._consec_fail: Dict[str, int] = {}    # consecutive validation failures
        self._latency: Dict[str, deque] = {}      # rolling batch round-trip times (telemetry)
        self._events: list = []                   # display-only dispatch/cooldown events (drained on flush)

        self._batch_size: Dict[str, float] = {}

    # ---- Adaptive knobs (read live so remote-config updates apply) ----

    def _window_min(self) -> float:
        return float(_cfg("DISPATCH_WINDOW_MIN", 1))

    def _grow(self) -> float:
        return float(_cfg("DISPATCH_WINDOW_GROW", 1.0))

    def _shrink(self) -> float:
        return float(_cfg("DISPATCH_WINDOW_SHRINK", 0.5))

    def _chronic_n(self) -> int:
        return int(_cfg("DISPATCH_CHRONIC_TIMEOUT_N", 5))

    def _late_threshold_s(self) -> float:
        # A valid push-back slower than this fraction of the lease freezes growth.
        return float(_cfg("DISPATCH_LATE_FRACTION", 0.6)) * float(_cfg("SCORING_LEASE_TTL_SECONDS", 900))

    def _adaptive_active(self) -> bool:
        return self._adaptive and bool(_cfg("ADAPTIVE_DISPATCH_ENABLED", False))

    def _bs_active(self) -> bool:
        return self._adaptive and bool(_cfg("ADAPTIVE_BATCH_SIZE_ENABLED", False))

    def _bs_base(self) -> int:
        """Served batch-size baseline (the size every validator gives by default)."""
        return max(1, int(_cfg("MINER_BATCH_SIZE", 12)))

    def _bs_min(self) -> int:
        return max(1, int(_cfg("MINER_BATCH_SIZE_MIN", self._bs_base())))

    def _bs_max(self) -> int:
        return max(self._bs_min(), int(_cfg("MINER_BATCH_SIZE_MAX", self._bs_base())))

    def _get_window(self, hotkey: str) -> float:
        return self._window.get(hotkey, self._window_min())

    def _effective_cap(self) -> float:
        if self._cap is not None:
            return self._cap
        budget = float(_cfg("DISPATCH_WINDOW_BUDGET", 0) or _cfg("VALIDATOR_MINER_QUERY_CONCURRENCY", 8))
        return max(self._window_min(), float(_cfg("DISPATCH_WINDOW_CAP_PCT", 0.15)) * budget)

    def set_cap(self, cap: float) -> None:
        """Allocator sets the anti-monopoly cap each tick: cap_pct * total in-flight budget."""
        self._cap = max(self._window_min(), float(cap))

    # ---- In-flight tracking ----

    def inflight_limit(self, hotkey: str) -> int:
        """How many dispatches this miner may hold at once.

        One definition, read by both the selector and the reservation. Keeping two
        copies is how they came to disagree.
        """
        if self.ration_for(hotkey) is not None:
            return max(1, int(self.batches_per_epoch(hotkey)))
        if self._adaptive_active():
            return max(1, int(math.floor(self._get_window(hotkey))))
        return MAX_INFLIGHT_PER_MINER

    def try_acquire(self, hotkey: str) -> bool:
        """Returns True if the miner has capacity for another dispatch.

        Under an earned ration the limit is the ration, matching what dispatch
        selection already allowed. Otherwise the adaptive window governs, or the static
        cap. Selection and reservation must agree, or work is offered and then refused.
        """
        count = self._inflight.get(hotkey, 0)
        limit = self.inflight_limit(hotkey)
        if count >= limit:
            return False
        self._inflight[hotkey] = count + 1
        return True

    def release(self, hotkey: str) -> None:
        count = self._inflight.get(hotkey, 0)
        if count > 1:
            self._inflight[hotkey] = count - 1
        elif hotkey in self._inflight:
            del self._inflight[hotkey]

    def inflight(self, hotkey: str) -> int:
        return self._inflight.get(hotkey, 0)

    def window(self, hotkey: str) -> float:
        return self._get_window(hotkey)

    def window_values(self) -> list:
        """Current per-miner window sizes (for pilot metrics)."""
        return list(self._window.values())

    def set_ration_source(self, source) -> None:
        """Install a callable returning a UID's earned ration, in articles per epoch.

        When set, it replaces the adaptive batch size: dispatch is then leased on what
        a UID has actually delivered rather than on how many submissions were accepted.
        """
        self._ration_source = source

    def ration_for(self, hotkey: str):
        source = getattr(self, "_ration_source", None)
        if source is None:
            return None
        try:
            value = source(hotkey)
        except Exception:
            return None
        return None if value is None else float(value)

    def batch_size(self, hotkey: str) -> int:
        """Per-miner batch size, clamped to [MIN, MAX]; baseline when disabled."""
        ration = self.ration_for(hotkey)
        if ration is not None:
            return int(max(1, min(self._bs_max(), int(ration))))
        if not self._bs_active():
            return self._bs_base()
        bs = self._batch_size.get(hotkey, float(self._bs_base()))
        return int(max(self._bs_min(), min(self._bs_max(), int(bs))))

    def batches_per_epoch(self, hotkey: str) -> int:
        """How many batches a UID is eligible for this epoch.

        A ration above one batch buys more batches rather than a larger one, so the
        batch stays a size miners already handle.
        """
        ration = self.ration_for(hotkey)
        if ration is None:
            return 1
        return max(1, int(math.ceil(ration / max(1, self._bs_max()))))

    def record_batch_valid(self, hotkey: str, latency_s: float) -> None:
        """Grow on an on-time valid return; hold on valid-but-slow. No-op when disabled."""
        if not self._bs_active():
            return
        if latency_s is not None and latency_s <= self._late_threshold_s():
            before = self.batch_size(hotkey)
            cur = self._batch_size.get(hotkey, float(self._bs_base()))
            step = int(_cfg("BATCH_SIZE_GROW_STEP", 2))
            self._batch_size[hotkey] = min(float(self._bs_max()), cur + step)
            after = self.batch_size(hotkey)
            if after != before:
                self._emit_event(hotkey, "batch_size_changed", reason="grew on on-time valid",
                                 detail={"from": before, "to": after})

    def record_batch_shrink(self, hotkey: str) -> None:
        """Shrink toward MIN. No-op when disabled."""
        if not self._bs_active():
            return
        before = self.batch_size(hotkey)
        cur = self._batch_size.get(hotkey, float(self._bs_base()))
        factor = float(_cfg("BATCH_SIZE_SHRINK_FACTOR", 0.75))
        self._batch_size[hotkey] = max(float(self._bs_min()), cur * factor)
        after = self.batch_size(hotkey)
        if after != before:
            self._emit_event(hotkey, "batch_size_changed", reason="shrank on invalid/park",
                             detail={"from": before, "to": after})

    def _emit_event(self, hotkey: str, event_type: str, **fields) -> None:
        """Append a display-only dispatch/cooldown event (drained on the dispatch flush).
        occurred_at is stamped here so buffered events keep accurate timing. Never raises."""
        try:
            ev = {"miner_hotkey": hotkey, "event_type": event_type, "occurred_at": time.time()}
            ev.update({k: v for k, v in fields.items() if v is not None})
            self._events.append(ev)
            if len(self._events) > EVENT_BUFFER_MAX:
                del self._events[0:len(self._events) - EVENT_BUFFER_MAX]
        except Exception:
            pass

    def drain_events(self) -> list:
        """Return and clear the buffered events (called by the validator's dispatch flush)."""
        evs = self._events
        self._events = []
        return evs

    def record_latency(self, hotkey: str, latency_s: float) -> None:
        """Record a batch round-trip time for telemetry (rolling window). Display-only;
        fires on every push-back regardless of validity/adaptive flags for a true median."""
        if latency_s is None:
            return
        dq = self._latency.get(hotkey)
        if dq is None:
            dq = deque(maxlen=LATENCY_WINDOW)
            self._latency[hotkey] = dq
        dq.append(float(latency_s))

    def median_latency(self, hotkey: str):
        """Median recent batch round-trip time (seconds), or None if unseen."""
        dq = self._latency.get(hotkey)
        if not dq:
            return None
        vals = sorted(dq)
        n = len(vals)
        mid = n // 2
        return round(vals[mid] if n % 2 else (vals[mid - 1] + vals[mid]) / 2.0, 2)

    def snapshot(self) -> Dict[str, dict]:
        """Per-hotkey adaptive state for every miner we hold any state for (display-only,
        for the dashboard diagnostics flush)."""
        now = time.time()
        hotkeys = (set(self._window) | set(self._inflight) | set(self._consec_to)
                   | set(self._covered_ep) | set(self._state) | set(self._consec_inv)
                   | set(self._inv_until) | set(self._last_faith) | set(self._batch_size)
                   | set(self._consec_fail) | set(self._latency))
        out = {}
        for hk in hotkeys:
            # Effective cooldown = later of the timeout-cooldown (_state) and the faith-park (_inv_until).
            until = max(self._state.get(hk, (0, 0, 0.0))[2], self._inv_until.get(hk, 0.0))
            out[hk] = {
                "window": round(self._get_window(hk), 3),
                "inflight": self._inflight.get(hk, 0),
                "consec_to": self._consec_to.get(hk, 0),
                "consec_inv": self._consec_inv.get(hk, 0),
                "consec_fail": self._consec_fail.get(hk, 0),
                "inv_level": self._inv_level.get(hk, 0),
                "last_faith": round(self._last_faith[hk], 3) if hk in self._last_faith else None,
                "batch_size": self.batch_size(hk),
                "median_latency_s": self.median_latency(hk),
                "covered_epoch": self._covered_ep.get(hk, -1),
                "on_cooldown": bool(until > 0 and now < until),
                "cooldown_remaining_s": int(until - now) if until > now else 0,
            }
        return out

    # ---- Adaptive window updates (RFC 2026-06-28) ----
    #
    # These update ONLY the window + the chronic-timeout counter. They do NOT touch
    # in-flight: under adaptive dispatch in-flight is reconciled from the article
    # store's PROCESSING set each cycle (reconcile_inflight), which is leak-proof by
    # construction — a missed or duplicated completion event cannot strand the
    # counter, and the per-article store can't desync the per-batch window. (The
    # static path still uses release() at the ack.)

    def record_timely_valid(self, hotkey: str, latency_s: float) -> None:
        """Valid push-back: reset the chronic counter and grow the window — but only
        if the completion was comfortably on-time. A valid-but-slow return means the
        miner is at capacity, so we freeze (hold) the window rather than grow. This is
        what finds capacity *without* ramping into a timeout (objective 8)."""
        self._consec_to[hotkey] = 0
        if latency_s is not None and latency_s <= self._late_threshold_s():
            w = self._get_window(hotkey)
            self._window[hotkey] = min(self._effective_cap(), w + self._grow() / max(w, 1e-9))

    def record_invalid(self, hotkey: str) -> None:
        """Returned-but-invalid: shrink the window. Resets the chronic counter — the
        miner *responded* (it is alive); bad quality is the integrity gate's job, not
        the non-response counter's."""
        self._consec_to[hotkey] = 0
        self._window[hotkey] = max(self._window_min(), self._get_window(hotkey) * self._shrink())

    # ---- Faithfulness cooldown (2026-07-09) ----

    def record_faithfulness(self, hotkey: str, min_faith) -> None:
        """Update the cooldown from the batch's min faithfulness. ``min_faith`` None is a
        no-op; shadow mode only logs."""
        if min_faith is None:
            return
        min_faith = float(min_faith)
        self._last_faith[hotkey] = min_faith
        floor = float(_cfg("DISPATCH_COOLDOWN_FAITHFULNESS_FLOOR", 0.5))
        threshold = int(_cfg("DISPATCH_CONSEC_INVALID_N", CONSECUTIVE_FAILURES_BEFORE_COOLDOWN))
        if min_faith >= floor:
            # Grounded batch -> recovered: clear streak, level, and any active park (self-heal).
            if self._consec_inv.get(hotkey) or self._inv_level.get(hotkey) or hotkey in self._inv_until:
                was_parked = hotkey in self._inv_until
                self._consec_inv[hotkey] = 0
                self._inv_level[hotkey] = 0
                self._inv_until.pop(hotkey, None)
                if was_parked:
                    self._emit_event(hotkey, "unparked", streak="faith",
                                     reason=f"recovered, faith={min_faith:.3f}")
            return
        n = self._consec_inv.get(hotkey, 0) + 1
        self._consec_inv[hotkey] = n
        bt.logging.info(
            f"[FAITHFULNESS] {hotkey[:12]}.. faith={min_faith:.3f} < floor={floor:.2f} "
            f"streak={n}/{threshold}"
        )
        if n >= threshold:
            shadow = bool(_cfg("DISPATCH_COOLDOWN_SHADOW_MODE", True))
            self._trip_stub_cooldown(hotkey, f"{n} consec faith<floor, last={min_faith:.3f}",
                                     shadow, tag="COOLDOWN")
            self._consec_inv[hotkey] = 0

    # Consecutive validation-fail park (2026-07-09). Its own shadow flag + N so it rolls out
    # independently of the faithfulness cooldown (which is already enforcing). Shares the park
    # machinery (_inv_until / _inv_level).

    def record_validation_fail(self, hotkey: str, reason: str = "") -> None:
        """Advance the consecutive validation-fail counter; park at DISPATCH_CONSEC_FAIL_N.
        Reset only on a genuine validation pass (record_validation_pass), never on the
        success/ack path. Callers must exclude validator-side failures."""
        n = self._consec_fail.get(hotkey, 0) + 1
        self._consec_fail[hotkey] = n
        threshold = int(_cfg("DISPATCH_CONSEC_FAIL_N", 10))
        bt.logging.info(
            f"[FAILSTREAK] {hotkey[:12]}.. fail (reason={reason}) streak={n}/{threshold}")
        if n >= threshold:
            shadow = bool(_cfg("DISPATCH_FAILSTREAK_SHADOW_MODE", True))
            self._trip_stub_cooldown(hotkey, f"{n} consec fails, last={reason}",
                                     shadow, tag="FAILSTREAK")
            self._consec_fail[hotkey] = 0

    def record_validation_pass(self, hotkey: str) -> None:
        """A genuine validation pass clears the fail streak (not the success/ack path)."""
        if self._consec_fail.get(hotkey):
            self._consec_fail[hotkey] = 0

    def _trip_stub_cooldown(self, hotkey: str, detail: str, shadow: bool,
                            tag: str = "COOLDOWN") -> None:
        """Apply the cooldown. Expiry goes in _inv_until (not _state) so the success path
        can't clear it; _inv_level holds the level across re-probes. Shadow mode only logs.
        The caller resets its own counter after this returns."""
        self.record_batch_shrink(hotkey)
        lvl = min(self._inv_level.get(hotkey, 0) + 1, 2)
        self._inv_level[hotkey] = lvl
        first_s = float(_cfg("DISPATCH_INVALID_COOLDOWN_FIRST_S", 60))
        max_s = float(_cfg("DISPATCH_INVALID_COOLDOWN_MAX_S", 600))
        cooldown_secs = first_s if lvl == 1 else max_s
        streak = "faith" if tag == "COOLDOWN" else ("failstreak" if tag == "FAILSTREAK" else "timeout")
        park_detail = {"cooldown_s": int(cooldown_secs), "level": lvl,
                       "last_faith": self._last_faith.get(hotkey)}
        if shadow:
            bt.logging.warning(
                f"[{tag}-SHADOW] WOULD park {hotkey[:12]}.. ({detail}) "
                f"for {int(cooldown_secs)}s (level {lvl}) — shadow, NOT enforced")
            # Emit even in shadow so a miner can see the streak "would have parked" them.
            self._emit_event(hotkey, "parked", streak=streak, shadow=True,
                             reason=detail, detail=park_detail)
            return
        self._inv_until[hotkey] = time.time() + cooldown_secs
        bt.logging.warning(
            f"[{tag}] parked {hotkey[:12]}.. ({detail}) for {int(cooldown_secs)}s (level {lvl})")
        self._emit_event(hotkey, "parked", streak=streak, shadow=False,
                         reason=detail, detail=park_detail)

    def record_timeout(self, hotkey: str) -> bool:
        """A reclaim cycle in which this miner had ≥1 lease timeout: shrink the window
        and advance the consecutive-timeout counter. Call once per hotkey per reclaim
        cycle (not per article). Returns True when chronic (>= DISPATCH_CHRONIC_TIMEOUT_N
        consecutive) — the caller then applies the integrity penalty/broadcast. On
        chronic it also drops the miner into the exponential-backoff cooldown and resets
        the counter (clean slate for a fair re-probe after cooldown)."""
        self._window[hotkey] = max(self._window_min(), self._get_window(hotkey) * self._shrink())
        n = self._consec_to.get(hotkey, 0) + 1
        if n >= self._chronic_n():
            self._consec_to[hotkey] = 0
            self.escalate_to_cooldown(hotkey)
            return True
        self._consec_to[hotkey] = n
        return False

    # ---- Coverage tracking (used by the allocator, PR 4) ----

    def covered_epoch(self, hotkey: str) -> int:
        return self._covered_ep.get(hotkey, -1)

    def mark_covered(self, hotkey: str, epoch: int) -> None:
        self._covered_ep[hotkey] = int(epoch)

    # ---- Reconciliation (anti-leak; RFC Component 2) ----

    def reconcile_inflight(self, counts: Dict[str, int]) -> None:
        """Rebuild in-flight counts from an authoritative source (articles still in
        PROCESSING status). Because release moved off the synchronous ack path, a lost
        push-back + a missed reclaim would otherwise leak the counter upward and
        silently shrink a miner's window to zero. Call this periodically."""
        self._inflight = {hk: int(c) for hk, c in counts.items() if c and c > 0}

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

    def escalate_to_cooldown(self, hotkey: str) -> None:
        """Force the next exponential-backoff cooldown step immediately. Used by
        chronic-timeout escalation (a separate, faster signal than record_failure's
        consecutive-dispatch-failure count); repeat escalations lengthen the cooldown."""
        consec, level, _ = self._state.get(hotkey, (0, 0, 0.0))
        level = min(level + 1, len(BACKOFF_SCHEDULE))
        backoff_idx = min(level - 1, len(BACKOFF_SCHEDULE) - 1)
        cooldown_secs = BACKOFF_SCHEDULE[backoff_idx]
        self._state[hotkey] = (consec, level, time.time() + cooldown_secs)
        bt.logging.info(
            f"[COOLDOWN] Miner {hotkey[:12]}.. chronic timeout escalation, "
            f"cooldown for {cooldown_secs}s (level {level})"
        )

    def record_success(self, hotkey: str) -> None:
        if hotkey in self._state:
            _, level, _ = self._state[hotkey]
            if level > 0:
                bt.logging.info(f"[COOLDOWN] Miner {hotkey[:12]}.. recovered, clearing cooldown")
            del self._state[hotkey]

    def is_on_cooldown(self, hotkey: str) -> bool:
        now = time.time()
        _, _, cooldown_until = self._state.get(hotkey, (0, 0, 0.0))
        return (cooldown_until > 0 and now < cooldown_until) or now < self._inv_until.get(hotkey, 0.0)

    def get_cooled_down_hotkeys(self) -> Set[str]:
        now = time.time()
        cooled = {hk for hk, (_, _, until) in self._state.items() if until > 0 and now < until}
        cooled |= {hk for hk, until in self._inv_until.items() if now < until}
        return cooled

    def prune(self, active_hotkeys: Set[str]) -> None:
        stale = [hk for hk in self._state if hk not in active_hotkeys]
        for hk in stale:
            del self._state[hk]
        stale_inflight = [hk for hk in self._inflight if hk not in active_hotkeys]
        for hk in stale_inflight:
            del self._inflight[hk]
        for d in (self._window, self._consec_to, self._covered_ep,
                  self._consec_inv, self._inv_level, self._inv_until, self._last_faith,
                  self._batch_size, self._consec_fail, self._latency):
            for hk in [h for h in d if h not in active_hotkeys]:
                del d[hk]
        if stale:
            bt.logging.debug(f"[COOLDOWN] Pruned {len(stale)} stale hotkey(s)")

    def stats(self) -> Tuple[int, int]:
        """Returns (total_tracked, currently_on_cooldown)."""
        now = time.time()
        on_cooldown = sum(1 for _, (_, _, until) in self._state.items() if until > 0 and now < until)
        return len(self._state), on_cooldown
