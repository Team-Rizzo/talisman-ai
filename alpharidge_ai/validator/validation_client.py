# alpharidge_ai/validator/validation_client.py
import asyncio
from typing import Any, Dict, List, Callable, Optional
import httpx
import bittensor as bt
import numpy as np
import time
import random
from datetime import datetime, timezone

from alpharidge_ai import config
from alpharidge_ai.utils.api_client import AlpharidgeAPIClient
from alpharidge_ai.utils.api_models import TweetWithAuthor, TelegramMessageForScoring, NewsArticleForScoring
from alpharidge_ai.utils.burn import calculate_weights
from alpharidge_ai.models.reward import Reward
from alpharidge_ai.protocol import ValidatorRewards
from alpharidge_ai.protocol import ValidatorPenalties
from alpharidge_ai.protocol import ValidatorReputationObs
from alpharidge_ai.protocol import Score
from alpharidge_ai.validator import reputation
from alpharidge_ai.validator import emission_params
from alpharidge_ai.utils.validators import get_validator_hotkeys
from alpharidge_ai.validator import deep_verify
from alpharidge_ai.consensus import overlap_audit
from alpharidge_ai.mechanism import settlement

class ValidationClient:
    """
    """

    def __init__(
        self,
        validator,
        api_url: Optional[str] = None,
        poll_seconds: Optional[int] = None,
        http_timeout: Optional[float] = None,
        scores_block_interval: Optional[int] = None,
        wallet: Optional[bt.Wallet] = None,
    ):
        """
        Initialize the ValidationClient.
        
        Args:
            api_url: Base URL for the miner API. Defaults to MINER_API_URL env var
            poll_seconds: Seconds between poll attempts. Defaults to VALIDATION_POLL_SECONDS env var or 10 seconds
            http_timeout: HTTP request timeout in seconds. Defaults to BATCH_HTTP_TIMEOUT env var or 30.0 seconds
            scores_block_interval: Blocks between score fetches. Defaults to SCORES_BLOCK_INTERVAL env var or 100
            wallet: Optional Bittensor wallet for authentication
        """
        self.wallet = wallet
        self.api_client = AlpharidgeAPIClient(
            base_url=api_url or config.MINER_API_URL,
            wallet=self.wallet,
            timeout=http_timeout or config.BATCH_HTTP_TIMEOUT,
            max_retries=3,
            retry_delay=1.0,
        )
        self._running: bool = False
        self._validator = validator
        # Avoid spamming broadcasts: publish at most once per chain epoch.
        self._last_publish_epoch: Optional[int] = None
        # Avoid spamming API submissions: submit at most once per target epoch.
        self._last_api_submit_epoch: Optional[int] = None
        # Avoid spamming score sends: send at most once per target epoch.
        self._last_score_epoch: Optional[int] = None
        # Recompute weights/scores at most once per target epoch (deterministic).
        self._last_weight_epoch: Optional[int] = None
        # Run deep-verify at most once per target epoch to avoid spamming post_report.
        self._last_deep_verify_epoch: Optional[int] = None
        # Loop cadence. Keep defaults aligned with config.
        self.poll_seconds = poll_seconds if poll_seconds is not None else config.VALIDATION_POLL_SECONDS
        self.scores_block_interval = scores_block_interval if scores_block_interval is not None else config.SCORES_BLOCK_INTERVAL
        # Exponential backoff state for API errors
        self._consecutive_errors = 0
        self._max_backoff_seconds = 300  # Max 5 minutes between retries

    async def _run_capacity_controller(self, epoch: int) -> None:
        """Measure the return, and report a step when the rule arms.

        Reporting is informational by design: capacity moves when an operator publishes
        a profile carrying the new value, so there is no path from here to anyone's pay.
        """
        ctrl = getattr(self._validator, "_capacity_controller", None)
        if ctrl is None:
            return
        profile = self._validator._mechanism_profile.resolve(int(self._validator.block))
        if profile is None:
            return

        proposal = ctrl.observe(epoch, profile)
        if proposal is None:
            return
        try:
            await self.api_client.post_controller_proposal(
                epoch=epoch, roi_ema=proposal.roi_ema, direction=proposal.direction,
                magnitude=abs(proposal.to_capacity / proposal.from_capacity - 1.0))
        except Exception as e:
            bt.logging.warning(f"[CONTROLLER] Could not report the step: {e}")

    def _weights_for(self, rewards, window_blocks: int, context: str):
        """The weight vector to set, from whichever settlement is in force.

        Under the capacity formula a point pays E/C until the subnet's work reaches
        capacity, and the remainder burns. The legacy path prices points in USD and
        carries a floor and a rescale above 100%; both are gone here.
        """
        profile = self._validator._mechanism_profile.resolve(int(self._validator.block))
        if profile is None or not profile.settlement.live:
            return calculate_weights(rewards, self._validator.metagraph,
                                     window_blocks, context=context)

        # Capacity is points per epoch; the rewards span the whole weight window. The
        # legacy path divides by the window too. Comparing a window total against one
        # epoch of capacity would make burn a function of window length rather than of
        # work, so capacity is scaled to the same span.
        epochs = max(1.0, float(window_blocks) / float(config.BLOCK_LENGTH))
        work = {r.hotkey: float(r.reward) for r in (rewards or [])}
        result = settlement.settle(work, profile.settlement.C * epochs)
        hotkeys = list(self._validator.metagraph.hotkeys)
        size = int(getattr(self._validator.metagraph, "n", len(hotkeys)))
        vector = settlement.weight_vector(result.shares, result.burn, hotkeys,
                                          int(config.BURN_UID), size)
        bt.logging.info(
            f"[SETTLEMENT] live W={result.work:.0f} C={result.capacity:.0f} "
            f"({epochs:.0f} epochs) burn={result.burn:.4f} paid={result.paid:.4f} "
            f"{context}")
        return np.array(vector, dtype=np.float64)

    def _shadow_settlement(self, rewards, live_weights, epoch: int,
                           window_blocks: int = None) -> None:
        """Compute what the capacity formula would pay, and log the difference.

        Log-only. Weights on chain still come from the live path; this exists so the
        two can be compared over a full window before either is switched.
        """
        try:
            profile = self._validator._mechanism_profile.resolve(int(self._validator.block))
            if profile is None or profile.settlement.live:
                return

            # The same populated span the live path used. The configured window can be
            # wider than the epochs actually present, and scaling by it would report a
            # burn that the live calculation never saw.
            epochs = (max(1.0, float(window_blocks) / float(config.BLOCK_LENGTH))
                      if window_blocks
                      else float(max(1, config.weight_window_epochs(self._validator.block))))
            work = {r.hotkey: float(r.reward) for r in (rewards or [])}
            result = settlement.settle(work, profile.settlement.C * epochs)

            hotkeys = list(self._validator.metagraph.hotkeys)
            shadow = settlement.weight_vector(result.shares, result.burn, hotkeys,
                                              int(config.BURN_UID), len(live_weights))
            live_burn = float(live_weights[int(config.BURN_UID)]) if len(live_weights) > int(config.BURN_UID) else 0.0
            deltas = [abs(float(a) - float(b)) for a, b in zip(live_weights, shadow)]

            bt.logging.info(
                f"[SETTLEMENT] epoch={epoch} W={result.work:.0f} C={result.capacity:.0f} "
                f"burn={result.burn:.4f} live_burn={live_burn:.4f} "
                f"paid={result.paid:.4f} max_weight_delta={max(deltas or [0.0]):.6f} (shadow)")
        except Exception as e:
            bt.logging.debug(f"[SETTLEMENT] shadow failed: {e}")

    def _overlap_audit(self, epoch: int) -> None:
        """Check each peer's scores against ours on the articles we both graded.

        Keyed selection means two validators no longer grade identical sets, so
        agreement is measured on the intersection.
        """
        store = getattr(self._validator, "_reputation_store", None)
        if store is None:
            return
        self_hk = str(self._validator.wallet.hotkey.ss58_address)
        mine = overlap_audit.flatten(store.sender_observations(epoch, self_hk))
        if not mine:
            return

        for sender in store.senders(epoch):
            if sender == self_hk:
                continue
            theirs = overlap_audit.flatten(store.sender_observations(epoch, sender))
            verdict = overlap_audit.assess(mine, theirs)
            if not verdict.comparable:
                bt.logging.debug(
                    f"[OVERLAP] {sender[:12]}.. epoch={epoch} {verdict.reason}")
            elif verdict.flagged:
                bt.logging.warning(
                    f"[OVERLAP] {sender[:12]}.. epoch={epoch} overlap={verdict.overlap} "
                    f"{verdict.reason} (log-only)")
            else:
                bt.logging.info(
                    f"[OVERLAP] {sender[:12]}.. epoch={epoch} overlap={verdict.overlap} "
                    f"{verdict.reason}")

    # ---- Display-only penalty attribution (decoupled from consensus) ----

    def _timeout_penalty_row(self, miner_hotkey, resource_type, resource_id):
        """Build a display-only timeout attribution row. No diff — there's no miner
        analysis to compare on a timeout. Never raises (it runs inside the poll loop)."""
        try:
            epoch = int(self._validator._miner_reward._get_current_epoch())
        except Exception:
            epoch = -1
        return {
            "miner_hotkey": miner_hotkey,
            "epoch": epoch,
            "resource_type": resource_type,
            "resource_id": str(resource_id),
            "cause": "timeout",
            "failed_fields": None,
            "miner_values": None,
            "validator_values": None,
            "post_preview": None,
        }

    async def _flush_penalty_detail(self):
        """Best-effort flush of buffered display-only attribution to the diagnostics
        endpoint. DECOUPLED from consensus: failures are swallowed and the snapshot is
        dropped (the buffer is bounded), so this can never stall or break the loop or
        affect scoring/weights. Snapshot-and-clear is atomic (no await between them)."""
        try:
            buf = self._validator._penalty_detail_buffer
            if not buf:
                return
            snapshot = buf[:]
            buf.clear()
            try:
                await self.api_client.submit_penalty_detail(snapshot)
                bt.logging.debug(f"[PENALTY_DETAIL] flushed {len(snapshot)} row(s)")
            except Exception as e:
                bt.logging.debug(f"[PENALTY_DETAIL] flush failed, dropped {len(snapshot)} row(s): {e}")
        except Exception as e:
            bt.logging.debug(f"[PENALTY_DETAIL] flush block error (ignored): {e}")

    async def _flush_variants(self):
        """Best-effort flush of buffered verifier analyses to /articles/variants.
        Snapshot-and-clear is atomic; failures drop the snapshot (bounded buffer)."""
        try:
            buf = getattr(self._validator, "_variant_buffer", None)
            if not buf:
                return
            snapshot = buf[:500]
            del buf[:500]
            try:
                resp = await self.api_client.submit_analysis_variants(snapshot)
                if not getattr(resp, "success", False):
                    buf[:0] = snapshot   # API refused: retry next cycle
                else:
                    bt.logging.debug(f"[VARIANTS] flushed {len(snapshot)}")
            except Exception as e:
                bt.logging.debug(f"[VARIANTS] flush failed, dropped {len(snapshot)}: {e}")
        except Exception as e:
            bt.logging.debug(f"[VARIANTS] flush block error (ignored): {e}")

    async def _flush_dispatch_status(self):
        """Best-effort push of the per-miner adaptive-dispatch status snapshot to the
        diagnostics endpoint (display-only, decoupled from consensus). Only while
        adaptive dispatch is on, and throttled — status changes slowly, no need to send
        every poll. Failures are swallowed; this can never stall the loop or touch
        scoring/weights."""
        try:
            if not getattr(config, "ADAPTIVE_DISPATCH_ENABLED", False):
                return
            now = time.time()
            if now - getattr(self, "_last_dispatch_status_flush", 0.0) < 30:
                return
            self._last_dispatch_status_flush = now
            rows = self._validator._build_dispatch_status()
            if not rows:
                return
            try:
                await self.api_client.submit_dispatch_status(rows)
                bt.logging.debug(f"[DISPATCH_STATUS] flushed {len(rows)} miner row(s)")
            except Exception as e:
                bt.logging.debug(f"[DISPATCH_STATUS] flush failed, dropped {len(rows)} row(s): {e}")
        except Exception as e:
            bt.logging.debug(f"[DISPATCH_STATUS] flush block error (ignored): {e}")

    async def _flush_miner_events(self):
        """Best-effort push of buffered per-miner dispatch/cooldown events (park/unpark,
        batch-size change) to the diagnostics endpoint (display-only, decoupled from
        consensus). Drained from the cooldown tracker; each event carries its own
        occurred_at so the flush cadence doesn't affect its timing. Failures are swallowed;
        this can never stall the loop or touch scoring/weights."""
        try:
            events = self._validator._article_cooldown.drain_events()
            if not events:
                return
            # Stamp occurred_at (float epoch) as an ISO8601 UTC string for the API model.
            out = []
            for ev in events:
                e = dict(ev)
                ts = e.get("occurred_at")
                if isinstance(ts, (int, float)):
                    e["occurred_at"] = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
                out.append(e)
            try:
                await self.api_client.submit_miner_events(out)
                bt.logging.debug(f"[MINER_EVENT] flushed {len(out)} event(s)")
            except Exception as e:
                bt.logging.debug(f"[MINER_EVENT] flush failed, dropped {len(out)} event(s): {e}")
        except Exception as e:
            bt.logging.debug(f"[MINER_EVENT] flush block error (ignored): {e}")

    def _epoch_rewards(self, epoch: int) -> Dict[str, int]:
        """Reward points for one epoch, hotkey -> points. Absent epoch -> {}."""
        try:
            return self._validator._miner_reward.get_rewards(int(epoch)) or {}
        except (KeyError, IndexError):
            return {}
        except Exception as e:
            bt.logging.warning(f"[ValidationClient] Failed to read rewards for epoch={epoch}: {e}")
            return {}

    def _epoch_penalties(self, epoch: int) -> Dict[str, int]:
        """Penalty counts for one epoch, hotkey -> count. Absent epoch -> {}."""
        try:
            return self._validator._miner_penalty.get_penalties(int(epoch)) or {}
        except (KeyError, IndexError):
            return {}
        except Exception as e:
            bt.logging.warning(f"[ValidationClient] Failed to read penalties for epoch={epoch}: {e}")
            return {}

    def _aggregate_window(self, target_epoch: int, window_start: int, window_end: int, quiet: bool = False):
        """Pool local and broadcast points over the inclusive epoch window.

        Returns (rewards, present_epochs, uid_rekey_skipped). present_epochs is the
        reward store's count only — the broadcast stores are legitimately sparse and
        must never shrink the window the caller divides by.

        quiet demotes the per-UID lines to debug, for the shadow pass.
        """
        info = bt.logging.debug if quiet else bt.logging.info
        combined_uid_rewards: Dict[int, int] = {}

        # Local rewards to uid->points
        present_epochs = 0
        try:
            local_rewards_map, present_epochs = self._validator._miner_reward.get_rewards_range(
                window_start, window_end
            )
            bt.logging.debug(f"[ValidationClient] Retrieved local_rewards_map: {local_rewards_map}")
            for hk, pts in local_rewards_map.items():
                if hk in self._validator.metagraph.hotkeys:
                    uid = self._validator.metagraph.hotkeys.index(hk)
                    combined_uid_rewards[uid] = combined_uid_rewards.get(uid, 0) + int(pts)
        except Exception as e:
            bt.logging.debug(f"[ValidationClient] [REWARDS] Failed to get local rewards: {e}")

        info(f"[REWARDS] Local rewards: {combined_uid_rewards}")

        # Broadcasted rewards (filter blacklisted hotkeys)
        broadcast_uid_rewards: Dict[int, int] = {}
        uid_rekey_skipped = 0
        try:
            broadcast_uid_rewards, uid_rekey_skipped = self._validator._reward_broadcasts.aggregate_range(
                window_start, window_end, self._validator.metagraph
            )
            bt.logging.debug(f"[ValidationClient] Aggregated broadcast_uid_rewards: {broadcast_uid_rewards}")
            for uid, pts in broadcast_uid_rewards.items():
                try:
                    hk = self._validator.metagraph.hotkeys[int(uid)]
                    if hk in config.BLACKLISTED_MINER_HOTKEYS:
                        info(f"[REWARDS] Ignoring broadcast rewards for blacklisted UID={uid} hotkey={hk[:12]}..")
                        continue
                except Exception:
                    pass
                combined_uid_rewards[uid] = combined_uid_rewards.get(uid, 0) + int(pts)
        except Exception as e:
            bt.logging.debug(f"[ValidationClient] [BROADCAST] Failed to aggregate rewards: {e}")

        info(f"[REWARDS] Broadcasted rewards: {broadcast_uid_rewards}")

        # Pooled penalty MAGNITUDE per UID: sum of penalty counts from this validator
        # (local) + every broadcasting validator. This mirrors how combined_uid_rewards
        # pools points, so points and penalties are compared on the same pooled basis.
        # Replaces the prior binary gate ("2+ validators recorded ANY penalty -> zero"),
        # which zeroed every miner that took even one routine penalty -> 100% burn.
        uid_penalty_totals: Dict[int, int] = {}

        try:
            local_penalties, _ = self._validator._miner_penalty.get_penalties_range(
                window_start, window_end
            )
            bt.logging.debug(f"[ValidationClient] Retrieved local_penalties: {local_penalties}")
            for hk, cnt in local_penalties.items():
                if cnt > 0 and hk in self._validator.metagraph.hotkeys:
                    uid = self._validator.metagraph.hotkeys.index(hk)
                    uid_penalty_totals[uid] = uid_penalty_totals.get(uid, 0) + int(cnt)
        except Exception as e:
            bt.logging.debug(f"[ValidationClient] [PENALTIES] Failed to get local penalties: {e}")

        try:
            # Sum broadcast penalty COUNTS (magnitude) across senders — not the number
            # of validators. aggregate_range() returns uid -> summed penalty count.
            broadcast_penalty_totals, _ = self._validator._penalty_broadcasts.aggregate_range(
                window_start, window_end, self._validator.metagraph
            )
            bt.logging.debug(f"[ValidationClient] Broadcast penalty totals: {broadcast_penalty_totals}")
            for uid, cnt in broadcast_penalty_totals.items():
                uid_penalty_totals[int(uid)] = uid_penalty_totals.get(int(uid), 0) + int(cnt)
        except Exception as e:
            bt.logging.debug(f"[ValidationClient] [PENALTY_BROADCAST] Failed to aggregate penalties: {e}")

        # Build rewards list: keep the reward (weight is later scaled ∝ points) only when
        # pooled points exceed pooled penalties; otherwise zero (net-negative miner ->
        # its share goes to burn). This is the documented "zero only if penalties >= rewards".
        #
        # The reputation gate is applied once, to the window total, never per epoch:
        # gating each epoch and then summing gives a different answer, and every
        # validator has to pick the same one.
        rewards = []
        rep_gating = getattr(config, "REPUTATION_GATING_ENABLED", False)
        params = emission_params.resolve(
            self._validator._mechanism_profile.resolve(int(self._validator.block))
            if hasattr(self._validator, "_mechanism_profile") else None)
        bt.logging.debug(f"[ValidationClient] Building rewards list, penalty_totals: {uid_penalty_totals}")
        for uid, pts in combined_uid_rewards.items():
            try:
                hk = self._validator.metagraph.hotkeys[int(uid)]
            except Exception:
                bt.logging.debug(f"[ValidationClient] Could not resolve hotkey for UID={uid}, skipping.")
                continue
            if rep_gating:
                r = self._validator._reputation_store.reputation(hk)
                n = self._validator._reputation_store.samples(hk)
                n_min = params.n_min
                g = reputation.emission(r, *params.as_args(), n=n, n_min=n_min)
                val = int(round(g * int(pts)))
                held = " neutral(under-observed)" if n_min > 0 and n < n_min else ""
                info(f"[REWARDS] UID={uid} hk={hk[:12]}.. gated={val} "
                     f"(rep={r:.3f} n={n} mult={g:.3f} vol={pts}){held}")
                rewards.append(Reward(hotkey=hk, reward=val, epoch=target_epoch))
                continue
            pen = uid_penalty_totals.get(uid, 0)
            if pts > pen:
                info(f"[REWARDS] Applying reward for UID={uid} hotkey={hk[:12]}... (points={pts} > penalties={pen})")
                rewards.append(Reward(hotkey=hk, reward=int(pts), epoch=target_epoch))
            else:
                info(f"[PENALTIES] Zeroing reward for UID={uid} hotkey={hk[:12]}... (points={pts} <= penalties={pen})")
                rewards.append(Reward(hotkey=hk, reward=0, epoch=target_epoch))

        return rewards, present_epochs, uid_rekey_skipped

    @staticmethod
    def _window_blocks(present_epochs: int, window_k: int, window_start: int, window_end: int):
        """Blocks to divide the window's points by, or None to keep the previous vector.

        Always the span actually present, never the full window. Dividing a partial
        point total by the full span understates every miner in proportion and
        drives burn up; dividing by the present span keeps the rate correct and
        costs only a smaller sample.
        """
        if present_epochs == 0:
            # Recomputing from nothing would send a near-empty vector. Keep the
            # previous one; the window is rebuilt at the next epoch anyway.
            bt.logging.error(
                f"[WEIGHT_WINDOW] No epochs present in [{window_start}..{window_end}]; "
                f"keeping the previous weight vector"
            )
            return None
        if present_epochs < window_k:
            bt.logging.warning(
                f"[WEIGHT_WINDOW] Only {present_epochs}/{window_k} epochs present in "
                f"[{window_start}..{window_end}]; dividing by the present span"
            )
        return present_epochs * config.BLOCK_LENGTH

    def _log_shadow_window(self, target_epoch: int) -> None:
        """Log the vector a wider window would produce, without setting it.

        Diagnostic only: it never touches self.scores and never raises into the
        live path. Off unless WEIGHT_WINDOW_SHADOW_EPOCHS is set.
        """
        shadow_k = getattr(config, "WEIGHT_WINDOW_SHADOW_EPOCHS", 0)
        if shadow_k <= 0:
            return
        try:
            start = target_epoch - shadow_k + 1
            rewards, present, rekeyed = self._aggregate_window(target_epoch, start, target_epoch, quiet=True)
            if present == 0:
                bt.logging.warning(f"[SHADOW] No epochs present in [{start}..{target_epoch}]; skipped")
                return
            context = (
                f"epochs=[{start}..{target_epoch}] present={present}/{shadow_k} "
                f"uid_rekey_skipped={rekeyed} "
            )
            calculate_weights(
                rewards, self._validator.metagraph, present * config.BLOCK_LENGTH,
                context="[SHADOW] " + context,
            )
        except Exception as e:
            bt.logging.warning(f"[SHADOW] Shadow window failed (live path unaffected): {e}")

    async def run(
        self,
        on_tweets: Callable[[List[TweetWithAuthor]], Any],
        on_telegram_messages: Callable[[List[TelegramMessageForScoring]], Any] = None,
        on_articles: Callable[[List[NewsArticleForScoring]], Any] = None,
    ):
        """
        Main validation loop.
        
        Args:
            on_tweets: Callback for batch of tweets (async or sync)
            on_telegram_messages: Callback for batch of telegram messages (async or sync)
        """
        self._running = True
        bt.logging.info("[ValidationClient.run] Entering main validation loop (poll_interval=%ss, scores_interval=%s blocks)", self.poll_seconds, self.scores_block_interval)

        # Bootstrap remote config on startup
        config.set_wallet(self.wallet)
        try:
            config.refresh_remote_config(force=True)
            bt.logging.info("[ValidationClient.run] Remote config loaded on startup")
        except Exception as e:
            bt.logging.warning(f"[ValidationClient.run] Remote config fetch failed on startup: {e}")

        while self._running:
            try:
                bt.logging.debug("[ValidationClient.run] Top of poll loop")
                try:
                    if not config.ENABLE_TWEET_SCORING:
                        # Tweet scoring disabled (article-first mode): skip the tweet fetch AND the
                        # tweet timeout penalties entirely, so article-only miners aren't zeroed for
                        # tweet timeouts. Toggle back on via remote config ENABLE_TWEET_SCORING.
                        unscored_tweets = []
                    else:
                        bt.logging.debug("[ValidationClient.run] Checking for timed-out tweets")
                        timed_out_tweets = self._validator._tweet_store.get_timeouts()
                        for tweet in timed_out_tweets:
                            bt.logging.debug(f"[ValidationClient.run] Resetting timed out tweet {tweet.tweet.id} to unprocessed")
                            self._validator._tweet_store.reset_to_unprocessed(tweet.tweet.id)
                            if tweet.hotkey:
                                bt.logging.info(f"[ValidationClient.run] Adding penalty to hotkey {tweet.hotkey} for tweet id {tweet.tweet.id}")
                                self._validator._miner_penalty.add_penalty(tweet.hotkey, 1)
                                # V3: display-only timeout attribution (decoupled from consensus).
                                self._validator._buffer_penalty_detail([self._timeout_penalty_row(
                                    tweet.hotkey, "tweet", tweet.tweet.id)])
                        bt.logging.debug("[ValidationClient.run] Fetching unscored tweets from api and local store")
                        unscored_tweets = (await self.api_client.get_unscored_tweets(limit=config.VALIDATION_FETCH_LIMIT)) + [tweet.tweet for tweet in self._validator._tweet_store.get_unprocessed_tweets()]
                    # Reset error counter on success
                    self._consecutive_errors = 0
                except Exception as e:
                    self._consecutive_errors += 1
                    # Exponential backoff: 2^errors seconds, capped at max_backoff
                    backoff = min(2 ** self._consecutive_errors, self._max_backoff_seconds)
                    bt.logging.warning(
                        f"[ValidationClient.run] Failed to fetch unscored tweets: {e} "
                        f"(attempt {self._consecutive_errors}, backing off {backoff}s)"
                    )
                    await asyncio.sleep(backoff)
                    continue

                if unscored_tweets:
                    bt.logging.debug(f"[ValidationClient.run] Passing {len(unscored_tweets)} unscored tweets to on_tweets() callback")
                    maybe_coro = on_tweets(unscored_tweets)
                    if asyncio.iscoroutine(maybe_coro):
                        bt.logging.debug("[ValidationClient.run] Awaiting on_tweets coroutine")
                        await maybe_coro

                # Fetch unscored telegram messages
                if on_telegram_messages is not None:
                    try:
                        bt.logging.debug("[ValidationClient.run] Checking for timed-out telegram messages")
                        timed_out_messages = self._validator._telegram_store.get_timeouts()
                        for msg_item in timed_out_messages:
                            bt.logging.debug(f"[ValidationClient.run] Resetting timed out telegram message {msg_item.message.id} to unprocessed")
                            self._validator._telegram_store.reset_to_unprocessed(msg_item.message.id)
                            if msg_item.hotkey:
                                bt.logging.info(f"[ValidationClient.run] Adding penalty to hotkey {msg_item.hotkey} for telegram message id {msg_item.message.id}")
                                self._validator._miner_penalty.add_penalty(msg_item.hotkey, 1)
                                # V3: display-only timeout attribution (decoupled from consensus).
                                self._validator._buffer_penalty_detail([self._timeout_penalty_row(
                                    msg_item.hotkey, "telegram", msg_item.message.id)])
                        bt.logging.debug("[ValidationClient.run] Fetching unscored telegram messages from api and local store")
                        unscored_telegram_messages = (await self.api_client.get_unscored_telegram_messages(limit=config.VALIDATION_FETCH_LIMIT)) + [item.message for item in self._validator._telegram_store.get_unprocessed_messages()]
                        
                        if unscored_telegram_messages:
                            bt.logging.debug(f"[ValidationClient.run] Passing {len(unscored_telegram_messages)} unscored telegram messages to on_telegram_messages() callback")
                            maybe_coro = on_telegram_messages(unscored_telegram_messages)
                            if asyncio.iscoroutine(maybe_coro):
                                bt.logging.debug("[ValidationClient.run] Awaiting on_telegram_messages coroutine")
                                await maybe_coro
                    except Exception as e:
                        bt.logging.warning(f"[ValidationClient.run] Failed to fetch/process telegram messages: {e}")

                # Fetch unscored news articles
                if on_articles is not None:
                    try:
                        bt.logging.debug("[ValidationClient.run] Checking for timed-out articles")
                        timed_out_articles = self._validator._article_store.get_timeouts()
                        adaptive_dispatch = getattr(config, "ADAPTIVE_DISPATCH_ENABLED", False)
                        # Penalty split (the consensus-affecting broadcast change) is gated
                        # SEPARATELY so it can be enabled / rolled back on its own.
                        penalty_split = adaptive_dispatch and getattr(config, "ADAPTIVE_PENALTY_SPLIT_ENABLED", True)
                        timed_out_hotkeys = set()
                        for article_item in timed_out_articles:
                            bt.logging.debug(f"[ValidationClient.run] Resetting timed out article {article_item.article.id} to unprocessed")
                            self._validator._article_store.reset_to_unprocessed(article_item.article.id)
                            if not article_item.hotkey:
                                continue
                            if adaptive_dispatch:
                                # Collect for the once-per-hotkey window update below.
                                timed_out_hotkeys.add(article_item.hotkey)
                            # Legacy broadcast behaviour: per-article penalty. Active when adaptive
                            # dispatch is off OR the penalty split is explicitly disabled.
                            if not penalty_split:
                                bt.logging.info(f"[ValidationClient.run] Adding penalty to hotkey {article_item.hotkey} for article id {article_item.article.id}")
                                self._validator._miner_penalty.add_penalty(article_item.hotkey, 1)
                                # V3: display-only timeout attribution (decoupled from consensus).
                                self._validator._buffer_penalty_detail([self._timeout_penalty_row(
                                    article_item.hotkey, "article", article_item.article.id)])
                        # Adaptive window dynamics: one capacity signal per miner per reclaim cycle.
                        # With the split ON (default), this is the ONLY path by which a timeout becomes
                        # an integrity penalty / enters the broadcast — and only on CHRONIC non-response,
                        # so ordinary timeouts never zero a miner or pollute the pooled total. With the
                        # split OFF, the window still shrinks here but the penalty was already applied
                        # per-article above (legacy broadcast preserved).
                        if adaptive_dispatch:
                            for hk in timed_out_hotkeys:
                                self._validator._adaptive_metrics.incr("timeout")
                                escalate = self._validator._article_cooldown.record_timeout(hk)
                                self._validator._article_cooldown.record_batch_shrink(hk)
                                if penalty_split and escalate:
                                    bt.logging.info(f"[ValidationClient.run] Chronic timeout escalation for {hk[:12]}.. — applying integrity penalty")
                                    self._validator._miner_penalty.add_penalty(hk, 1)
                                    self._validator._buffer_penalty_detail([self._timeout_penalty_row(
                                        hk, "article", "chronic-timeout")])
                        bt.logging.debug("[ValidationClient.run] Fetching unscored articles from api and local store")
                        unscored_articles = (await self.api_client.get_unscored_articles(limit=config.VALIDATION_FETCH_LIMIT)) + [item.article for item in self._validator._article_store.get_unprocessed_articles()]

                        if unscored_articles:
                            bt.logging.debug(f"[ValidationClient.run] Passing {len(unscored_articles)} unscored articles to on_articles() callback")
                            maybe_coro = on_articles(unscored_articles)
                            if asyncio.iscoroutine(maybe_coro):
                                bt.logging.debug("[ValidationClient.run] Awaiting on_articles coroutine")
                                await maybe_coro
                    except Exception as e:
                        bt.logging.warning(f"[ValidationClient.run] Failed to fetch/process articles: {e}")

                # Submit tweets processed locally but not yet submitted to the API.
                ready = self._validator._tweet_store.get_ready_to_submit()
                bt.logging.debug(f"[ValidationClient.run] Checking {len(ready) if ready else 0} tweets ready to submit to API")
                if ready:
                    for item in ready:
                        try:
                            bt.logging.debug(f"[ValidationClient.run] Submitting tweet {item.tweet.id} to API")
                            await self._validator._submit_tweet_batch([item.tweet])
                            self._validator._tweet_store.mark_submitted(item.tweet.id)
                            self._validator._tweet_store.delete_tweet(item.tweet.id)
                            bt.logging.info(f"[ValidationClient.run] Successfully submitted tweet {item.tweet.id} and removed it from store")
                        except Exception as e:
                            bt.logging.warning(f"[ValidationClient.run] Failed to submit completed tweet {item.tweet.id}: {e}")
                            continue

                # Submit telegram messages processed locally but not yet submitted to the API.
                telegram_ready = self._validator._telegram_store.get_ready_to_submit()
                bt.logging.debug(f"[ValidationClient.run] Checking {len(telegram_ready) if telegram_ready else 0} telegram messages ready to submit to API")
                if telegram_ready:
                    for item in telegram_ready:
                        try:
                            bt.logging.debug(f"[ValidationClient.run] Submitting telegram message {item.message.id} to API")
                            await self._validator._submit_telegram_batch([item.message])
                            self._validator._telegram_store.mark_submitted(item.message.id)
                            self._validator._telegram_store.delete_message(item.message.id)
                            bt.logging.info(f"[ValidationClient.run] Successfully submitted telegram message {item.message.id} and removed it from store")
                        except Exception as e:
                            bt.logging.warning(f"[ValidationClient.run] Failed to submit completed telegram message {item.message.id}: {e}")
                            continue

                # Submit articles processed locally but not yet submitted to the API.
                article_ready = self._validator._article_store.get_ready_to_submit()
                bt.logging.debug(f"[ValidationClient.run] Checking {len(article_ready) if article_ready else 0} articles ready to submit to API")
                if article_ready:
                    for item in article_ready:
                        try:
                            bt.logging.debug(f"[ValidationClient.run] Submitting article {item.article.id} to API")
                            await self._validator._submit_article_batch([item.article])
                            self._validator._article_store.mark_submitted(item.article.id)
                            self._validator._article_store.delete_article(item.article.id)
                            bt.logging.info(f"[ValidationClient.run] Successfully submitted article {item.article.id} and removed it from store")
                        except Exception as e:
                            bt.logging.warning(f"[ValidationClient.run] Failed to submit completed article {item.article.id}: {e}")
                            continue

                # Flush display-only penalty attribution (best-effort, decoupled from
                # consensus — never blocks scoring/weights; failures drop the snapshot).
                await self._flush_penalty_detail()
                await self._flush_variants()
                # Same channel, for the adaptive-dispatch status snapshot (throttled).
                await self._flush_dispatch_status()
                # And the per-miner dispatch/cooldown event log (park/unpark, batch-size).
                await self._flush_miner_events()

                # Persist local state.
                try:
                    bt.logging.debug("[ValidationClient.run] Saving tweet store to disk")
                    self._validator._tweet_store.save_to_file()
                except Exception as e:
                    bt.logging.debug(f"[ValidationClient.run] Failed to persist tweet store: {e}")
                try:
                    bt.logging.debug("[ValidationClient.run] Saving telegram store to disk")
                    self._validator._telegram_store.save_to_file()
                except Exception as e:
                    bt.logging.debug(f"[ValidationClient.run] Failed to persist telegram store: {e}")
                try:
                    bt.logging.debug("[ValidationClient.run] Saving article store to disk")
                    self._validator._article_store.save_to_file()
                except Exception as e:
                    bt.logging.debug(f"[ValidationClient.run] Failed to persist article store: {e}")

                # ---- Periodic mechanism profile refresh ----
                try:
                    profile_client = getattr(self._validator, "_mechanism_profile", None)
                    if profile_client is not None:
                        block = int(self._validator.block)
                        profile_client.refresh(block)
                        profile_client.resolve(block)
                except Exception as e:
                    bt.logging.debug(f"[PROFILE] refresh failed: {e}")

                # ---- Periodic remote config refresh ----
                try:
                    config.refresh_remote_config()
                    reset_epoch, purge_hotkeys, reset_scores = config.get_pending_resets()
                    if reset_epoch >= 0:
                        bt.logging.info(f"[REMOTE_CONFIG] Flushing broadcast rewards for epochs <= {reset_epoch}")
                        self._validator._reward_broadcasts.flush_before_epoch(reset_epoch)
                        self._validator._penalty_broadcasts.flush_before_epoch(reset_epoch)
                    if purge_hotkeys:
                        bt.logging.info(f"[REMOTE_CONFIG] Purging broadcast data for {len(purge_hotkeys)} hotkeys")
                        self._validator._reward_broadcasts.purge_hotkeys(purge_hotkeys)
                        self._validator._penalty_broadcasts.purge_hotkeys(purge_hotkeys)
                    if reset_scores:
                        bt.logging.info("[REMOTE_CONFIG] Resetting validator scores to zero (signal from API)")
                        self._validator.scores = np.zeros(self._validator.metagraph.n, dtype=np.float32)
                        self._validator.save_state()
                except Exception as e:
                    bt.logging.debug(f"[ValidationClient.run] Remote config refresh error: {e}")

                # ---- Broadcast rewards + penalties to other validators ----
                current_epoch = self._validator._miner_reward._get_current_epoch()
                publish_epoch = current_epoch - 1
                bt.logging.debug(f"[ValidationClient.run] current_epoch={current_epoch} publish_epoch={publish_epoch} last_publish_epoch={self._last_publish_epoch}")
                try:
                    if publish_epoch >= 0 and self._last_publish_epoch != int(publish_epoch):
                        bt.logging.info(f"[ValidationClient.run] Preparing to broadcast for epoch={publish_epoch}")
                        try:
                            hotkey_points = self._validator._miner_reward.get_rewards(epoch=publish_epoch)
                        except KeyError:
                            hotkey_points = {}
                            bt.logging.info((
                                f"[BROADCAST] No local rewards bucket for epoch={publish_epoch} yet "
                                f"(startup/restart). Publishing empty rewards snapshot."
                            ))
                        uid_points = {}
                        for hk, pts in hotkey_points.items():
                            if hk in self._validator.metagraph.hotkeys:
                                uid = self._validator.metagraph.hotkeys.index(hk)
                                uid_points[uid] = int(pts)
                        try:
                            hotkey_penalties = self._validator._miner_penalty.get_penalties(epoch=publish_epoch)
                        except KeyError:
                            hotkey_penalties = {}
                            bt.logging.info((
                                f"[BROADCAST] No local penalties bucket for epoch={publish_epoch} yet "
                                f"(startup/restart). Publishing empty penalties snapshot."
                            ))
                        uid_penalties = {}
                        for hk, cnt in hotkey_penalties.items():
                            if hk in self._validator.metagraph.hotkeys:
                                uid = self._validator.metagraph.hotkeys.index(hk)
                                uid_penalties[uid] = int(cnt)

                        bt.logging.debug(f"[ValidationClient.run] uid_points for broadcast: {uid_points}")
                        bt.logging.debug(f"[ValidationClient.run] uid_penalties for broadcast: {uid_penalties}")

                        attestation_obj = None
                        attestation_sig = None
                        try:
                            att = await self.api_client.get_attestation(epoch=int(publish_epoch))
                            attestation_obj = {
                                "validatorHotkey": att["validator_hotkey"],
                                "epoch": int(att["epoch"]),
                                "perMinerPoints": att["per_miner_points"],
                                "totalPoints": att["total_points"],
                                "merkleRoot": att["merkle_root"],
                            }
                            attestation_sig = att["signature"]
                        except Exception as e:
                            bt.logging.debug(f"[BROADCAST] No attestation for epoch={publish_epoch}: {e}")

                        rewards_syn = ValidatorRewards(
                            epoch=int(publish_epoch),
                            uid_points=uid_points,
                            sender_hotkey=str(self._validator.wallet.hotkey.ss58_address),
                            seq=int(publish_epoch),
                            attestation=attestation_obj,
                            attestation_sig=attestation_sig,
                        )
                        penalties_syn = ValidatorPenalties(
                            epoch=int(publish_epoch),
                            uid_penalties=uid_penalties,
                            sender_hotkey=str(self._validator.wallet.hotkey.ss58_address),
                            seq=int(publish_epoch),
                        )
                        bt.logging.info(
                            f"[BROADCAST] Preparing publish epoch={publish_epoch} "
                            f"uid_points={len(uid_points)} uid_penalties={len(uid_penalties)}"
                        )
                        whitelist = set(
                            get_validator_hotkeys(
                                metagraph=self._validator.metagraph,
                                netuid=self._validator.config.netuid,
                            )
                        )
                        bt.logging.debug(f"[ValidationClient.run] Built validator whitelist of {len(whitelist)} entries")
                        axons = []
                        for uid in range(int(self._validator.metagraph.n.item())):
                            if uid == int(self._validator.uid):
                                continue
                            try:
                                hk = self._validator.metagraph.hotkeys[uid]
                                if hk not in whitelist:
                                    continue
                                ax = self._validator.metagraph.axons[uid]
                                if not ax.is_serving:
                                    continue
                                axons.append(ax)
                            except Exception:
                                continue
                        bt.logging.info(f"[ValidationClient.run] Found {len(axons)} whitelisted validator axons for broadcast")
                        max_targets = int(getattr(config, "VALIDATOR_BROADCAST_MAX_TARGETS", 32))
                        if max_targets > 0 and len(axons) > max_targets:
                            # log before sampling
                            bt.logging.info(f"[ValidationClient.run] Reducing broadcast axons from {len(axons)} to {max_targets}")
                            axons = random.sample(axons, max_targets)
                        if axons:
                            bt.logging.info(
                                f"[BROADCAST] Publishing epoch={publish_epoch} to {len(axons)} validator axon(s) "
                                f"(whitelist={len(whitelist)}, max_targets={max_targets})"
                            )
                            bt.logging.debug(f"[ValidationClient.run] Broadcasting rewards_syn to axons")
                            await self._validator.dendrite.forward(
                                axons=axons,
                                synapse=rewards_syn,
                                timeout=12.0,
                                deserialize=True,
                            )
                            if uid_penalties:
                                bt.logging.debug(f"[ValidationClient.run] Broadcasting penalties_syn to axons")
                                await self._validator.dendrite.forward(
                                    axons=axons,
                                    synapse=penalties_syn,
                                    timeout=12.0,
                                    deserialize=True,
                                )
                            if getattr(config, "REPUTATION_SCORING_ENABLED", False):
                                try:
                                    self_hk = str(self._validator.wallet.hotkey.ss58_address)
                                    exp = self._validator._reputation_store.export(int(publish_epoch), self_hk)
                                    obs_payload = {t: [[float(a), float(g), float(w)] for (a, g, w) in lst]
                                                   for t, lst in exp.items()}
                                    if obs_payload:
                                        repobs_syn = ValidatorReputationObs(
                                            epoch=int(publish_epoch), observations=obs_payload,
                                            sender_hotkey=self_hk, seq=int(publish_epoch))
                                        await self._validator.dendrite.forward(
                                            axons=axons, synapse=repobs_syn, timeout=12.0, deserialize=True)
                                except Exception as e:
                                    bt.logging.debug(f"[REPUTATION_BROADCAST] send failed: {e}")
                        else:
                            bt.logging.info(
                                f"[BROADCAST] Skipping publish epoch={publish_epoch}: "
                                f"no target validator axons (whitelist={len(whitelist)})"
                            )
                        self._last_publish_epoch = int(publish_epoch)
                except Exception as e:
                    bt.logging.debug(f"[ValidationClient.run] [BROADCAST] Publish failed: {e}")

                # ---- Aggregate rewards/penalties and update scores ----
                target_epoch = current_epoch - 2
                bt.logging.debug(f"[ValidationClient.run] Calculating weights and scores for target_epoch={target_epoch}")

                # Apply the union of reputation observations for the settled epoch (both
                # validators aggregate the same set -> identical reputation), then push a
                # snapshot (once per epoch, display/monitoring only).
                if getattr(config, "REPUTATION_SCORING_ENABLED", False) and target_epoch >= 0:
                    try:
                        self._validator._reputation_store.finalize(
                            int(target_epoch), alpha=getattr(config, "REPUTATION_EMA_ALPHA", 0.03))
                        self._validator._reputation_store.save()
                    except Exception as e:
                        bt.logging.debug(f"[REPUTATION] finalize failed: {e}")
                    if int(target_epoch) != getattr(self, "_last_rep_snapshot_epoch", -1):
                        try:
                            p = emission_params.resolve(
                                self._validator._mechanism_profile.resolve(
                                    int(self._validator.block)))
                            n_min = p.n_min
                            snap = self._validator._reputation_store.snapshot()
                            median = emission_params.live_median(snap, n_min)
                            if median is not None:
                                bt.logging.info(
                                    f"[EMISSION] {p.source} midpoint={p.midpoint:.3f} "
                                    f"gain={p.gain:.1f} ceiling={p.ceiling:.2f} "
                                    f"live_median={median:.3f} "
                                    f"drift={median - p.midpoint:+.3f}")
                            # display-only, decoupled from consensus.
                            rows = [{"miner_hotkey": hk, "reputation": float(st.get("r", 0.5)),
                                     "samples": int(st.get("n", 0)),
                                     "gate": reputation.emission(
                                         float(st.get("r", 0.5)), *p.as_args(),
                                         n=int(st.get("n", 0)), n_min=n_min)}
                                    for hk, st in snap.items()]
                            if rows:
                                await self.api_client.post_reputation_snapshot(int(target_epoch), rows)
                            self._last_rep_snapshot_epoch = int(target_epoch)
                        except Exception as e:
                            bt.logging.debug(f"[REPUTATION] snapshot push failed: {e}")
                # Weight window: the K epochs ending at target_epoch. K is taken from
                # the block height so that every validator uses the same value at the
                # same block, whenever each one last polled the config.
                window_k = config.weight_window_epochs(self._validator.block)
                window_start = target_epoch - window_k + 1
                window_end = target_epoch

                # Recompute weights/scores once per target epoch so the snapshot the
                # base neuron loop commits on-chain is deterministic (not whatever
                # mid-aggregation state happens to be live every poll iteration).
                # The aggregation sits inside the guard because nothing downstream
                # reads it; running it every poll only repeated the work and the logs.
                if self._last_weight_epoch != target_epoch:
                    rewards, present_epochs, uid_rekey_skipped = self._aggregate_window(
                        target_epoch, window_start, window_end
                    )
                    bt.logging.debug(f"[ValidationClient.run] Calculating weights from rewards list (len={len(rewards)})")
                    window_blocks = self._window_blocks(
                        present_epochs, window_k, window_start, window_end
                    )
                    if window_blocks is not None:
                        context = (
                            f"epochs=[{window_start}..{window_end}] "
                            f"present={present_epochs}/{window_k} "
                            f"uid_rekey_skipped={uid_rekey_skipped} "
                        )
                        weights = self._weights_for(rewards, window_blocks, context)
                        bt.logging.debug(f"[ValidationClient.run] Updating scores with new weights")
                        self._validator.update_scores(weights, self._validator.metagraph.uids.tolist())
                        self._shadow_settlement(rewards, weights, int(target_epoch),
                                                window_blocks)
                        self._last_weight_epoch = target_epoch
                        self._log_shadow_window(target_epoch)

                # ---- Sampled deep-verify of received attestations (once per epoch) ----
                if self._last_deep_verify_epoch != target_epoch:
                    try:
                        senders = list((self._validator._reward_broadcasts
                                        .by_epoch_by_sender.get(int(target_epoch)) or {}).keys())
                        for sender in senders:
                            expected_root = self._validator._reward_broadcasts.get_merkle_root(
                                epoch=int(target_epoch), sender=sender)
                            if not expected_root:
                                continue
                            if not deep_verify.should_sample(config.DEEP_VERIFY_SAMPLE_RATE, random.random()):
                                continue
                            try:
                                resp = await self.api_client.get_verdicts(
                                    validator=sender, epoch=int(target_epoch))
                                leaves = [{
                                    "resource_type": l.get("resource_type"), "resource_id": l.get("resource_id"),
                                    "miner_hotkey": l.get("miner_hotkey"), "validator_verdict": l.get("validator_verdict"),
                                    "categorical_key": l.get("categorical_key"), "points_awarded": l.get("points_awarded"),
                                } for l in resp.get("verdicts", [])]
                                # NOTE: the attestation's merkleRoot was already API-signature-verified at
                                # ingest. A mismatch vs live /verdicts therefore signals API-side data drift
                                # rather than provable sender tampering; report drives 2+ consensus, and this
                                # runs once per epoch, bounding false-positive blast radius.
                                reason = deep_verify.merkle_mismatch(expected_root, leaves)
                                if reason:
                                    bt.logging.warning(
                                        f"[DEEP_VERIFY] {sender[:12]}.. epoch={target_epoch} {reason}; reporting")
                                    await self.api_client.post_report(
                                        accused_hotkey=sender, epoch=int(target_epoch),
                                        reason=reason, evidence={"expected_root": expected_root})
                            except Exception as e:
                                bt.logging.debug(f"[DEEP_VERIFY] fetch/recompute failed for {sender[:12]}..: {e}")
                    except Exception as e:
                        bt.logging.debug(f"[DEEP_VERIFY] pass failed: {e}")

                    # Overlap audit: compare a sender against us on the articles both
                    # of us graded.
                    try:
                        self._overlap_audit(int(target_epoch))
                    except Exception as e:
                        bt.logging.debug(f"[OVERLAP] pass failed: {e}")

                    try:
                        await self._run_capacity_controller(int(target_epoch))
                    except Exception as e:
                        bt.logging.debug(f"[CONTROLLER] pass failed: {e}")
                    finally:
                        self._last_deep_verify_epoch = target_epoch

                # Send each active miner their raw reward/penalty counts for this epoch (once per epoch)
                if target_epoch >= 0 and self._last_score_epoch != target_epoch:
                    try:
                        score_start_block = int(target_epoch) * int(config.BLOCK_LENGTH)
                        score_end_block = (int(target_epoch) + 1) * int(config.BLOCK_LENGTH) - 1
                        # Reports one epoch, so read the per-epoch stores.
                        epoch_rewards = self._epoch_rewards(target_epoch)
                        epoch_penalties = self._epoch_penalties(target_epoch)
                        active_hotkeys = set(epoch_rewards.keys()) | set(epoch_penalties.keys())

                        async def _send_score(hk: str) -> None:
                            try:
                                if hk not in self._validator.metagraph.hotkeys:
                                    return
                                uid = self._validator.metagraph.hotkeys.index(hk)
                                axon = self._validator.metagraph.axons[uid]
                                r = int(epoch_rewards.get(hk, 0))
                                p = int(epoch_penalties.get(hk, 0))
                                syn = Score(
                                    block_window_start=score_start_block,
                                    block_window_end=score_end_block,
                                    rewards=r,
                                    penalties=p,
                                    score=float(r - p),
                                    validator_hotkey=self._validator.wallet.hotkey.ss58_address,
                                )
                                await self._validator.dendrite.forward(
                                    axons=[axon],
                                    synapse=syn,
                                    timeout=12.0,
                                    deserialize=False,
                                )
                            except Exception:
                                pass

                        await asyncio.gather(*[_send_score(hk) for hk in active_hotkeys])
                        self._last_score_epoch = target_epoch
                        bt.logging.info(f"[SCORE] Sent epoch={target_epoch} scores to {len(active_hotkeys)} miner(s)")
                    except Exception as e:
                        bt.logging.warning(f"[ValidationClient.run] [SCORE] Failed to send scores: {e}")

                bt.logging.debug(f"[ValidationClient.run] Sleeping for {self.poll_seconds} seconds before next poll loop")
                
                self._validator._penalty_broadcasts.save()
                self._validator._reward_broadcasts.save()
                self._validator._tweet_store.save_to_file()
                self._validator._telegram_store.save_to_file()
                self._validator._article_store.save_to_file()
                self._validator._miner_reward.save()
                self._validator._miner_penalty.save()
                
                # submit rewards and penalties to the API
                try:
                    if target_epoch >= 0 and self._last_api_submit_epoch != int(target_epoch):
                        # Rewards: API expects {start_block, stop_block, hotkey, points}
                        start_block = int(target_epoch) * int(config.BLOCK_LENGTH)
                        stop_block = (int(target_epoch) + 1) * int(config.BLOCK_LENGTH) - 1
                        # One epoch per row, to match the block range stamped on it.
                        epoch_rewards_list, _, _ = self._aggregate_window(
                            target_epoch, target_epoch, target_epoch, quiet=True
                        )
                        rewards_payload = [
                            {
                                "start_block": start_block,
                                "stop_block": stop_block,
                                "hotkey": r.hotkey,
                                "points": float(getattr(r, "reward", 0)),
                            }
                            for r in epoch_rewards_list
                        ]

                        # Penalties: API expects {hotkey, reason}
                        epoch_penalties = self._epoch_penalties(target_epoch)
                        penalties_payload = [
                            {
                                "hotkey": hk,
                                "reason": f"epoch={int(target_epoch)} count={int(cnt)}",
                            }
                            for hk, cnt in epoch_penalties.items()
                            if int(cnt) > 0
                        ]

                        if rewards_payload:
                            await self.api_client.submit_rewards(rewards=rewards_payload)
                        if penalties_payload:
                            await self.api_client.submit_penalties(penalties=penalties_payload)
                        self._last_api_submit_epoch = int(target_epoch)
                        bt.logging.info(
                            f"[API_SUBMIT] epoch={int(target_epoch)} "
                            f"rewards={len(rewards_payload)} penalties={len(penalties_payload)}"
                        )
                except Exception as e:
                    bt.logging.warning(f"[ValidationClient.run] Failed to submit rewards and penalties: {e}")
                
                await asyncio.sleep(float(self.poll_seconds))
            except asyncio.CancelledError:
                bt.logging.info("[ValidationClient.run] Validation client cancelled")
                raise  # Re-raise to properly stop the task
            except Exception as e:
                bt.logging.warning(f"[ValidationClient.run] Loop iteration error (will retry): {type(e).__name__}: {e}")
                await asyncio.sleep(5.0)  # Brief pause before retry
                continue