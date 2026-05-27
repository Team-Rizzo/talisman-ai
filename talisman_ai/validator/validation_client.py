# talisman_ai/validator/validation_client.py
import asyncio
from typing import Any, Dict, List, Callable, Optional
import httpx
import bittensor as bt
import numpy as np
import time
import random

from talisman_ai import config
from talisman_ai.utils.api_client import TalismanAPIClient
from talisman_ai.utils.api_models import TweetWithAuthor, TelegramMessageForScoring, NewsArticleForScoring
from talisman_ai.utils.burn import calculate_weights
from talisman_ai.models.reward import Reward
from talisman_ai.protocol import ValidatorRewards
from talisman_ai.protocol import ValidatorPenalties
from talisman_ai.protocol import Score
from talisman_ai.utils.validators import get_validator_hotkeys

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
        self.api_client = TalismanAPIClient(
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
        # Loop cadence. Keep defaults aligned with config.
        self.poll_seconds = poll_seconds if poll_seconds is not None else config.VALIDATION_POLL_SECONDS
        self.scores_block_interval = scores_block_interval if scores_block_interval is not None else config.SCORES_BLOCK_INTERVAL
        # Exponential backoff state for API errors
        self._consecutive_errors = 0
        self._max_backoff_seconds = 300  # Max 5 minutes between retries

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
                    bt.logging.debug("[ValidationClient.run] Checking for timed-out tweets")
                    timed_out_tweets = self._validator._tweet_store.get_timeouts()
                    for tweet in timed_out_tweets:
                        bt.logging.debug(f"[ValidationClient.run] Resetting timed out tweet {tweet.tweet.id} to unprocessed")
                        self._validator._tweet_store.reset_to_unprocessed(tweet.tweet.id)
                        if tweet.hotkey:
                            bt.logging.info(f"[ValidationClient.run] Adding penalty to hotkey {tweet.hotkey} for tweet id {tweet.tweet.id}")
                            self._validator._miner_penalty.add_penalty(tweet.hotkey, 1)
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
                        for article_item in timed_out_articles:
                            bt.logging.debug(f"[ValidationClient.run] Resetting timed out article {article_item.article.id} to unprocessed")
                            self._validator._article_store.reset_to_unprocessed(article_item.article.id)
                            if article_item.hotkey:
                                bt.logging.info(f"[ValidationClient.run] Adding penalty to hotkey {article_item.hotkey} for article id {article_item.article.id}")
                                self._validator._miner_penalty.add_penalty(article_item.hotkey, 1)
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

                        rewards_syn = ValidatorRewards(
                            epoch=int(publish_epoch),
                            uid_points=uid_points,
                            sender_hotkey=str(self._validator.wallet.hotkey.ss58_address),
                            seq=int(publish_epoch),
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
                combined_uid_rewards: Dict[int, int] = {}

                # Local rewards to uid->points
                local_rewards_map: Dict[str, int] = {}
                try:
                    local_rewards_map = self._validator._miner_reward.get_rewards(epoch=target_epoch)
                    bt.logging.debug(f"[ValidationClient.run] Retrieved local_rewards_map: {local_rewards_map}")
                    for hk, pts in local_rewards_map.items():
                        if hk in self._validator.metagraph.hotkeys:
                            uid = self._validator.metagraph.hotkeys.index(hk)
                            combined_uid_rewards[uid] = combined_uid_rewards.get(uid, 0) + int(pts)
                except Exception as e:
                    bt.logging.debug(f"[ValidationClient.run] [REWARDS] Failed to get local rewards: {e}")

                bt.logging.info(f"[REWARDS] Local rewards: {combined_uid_rewards}")

                # Broadcasted rewards (filter blacklisted hotkeys)
                try:
                    broadcast_uid_rewards = self._validator._reward_broadcasts.aggregate_epoch(target_epoch)
                    bt.logging.debug(f"[ValidationClient.run] Aggregated broadcast_uid_rewards: {broadcast_uid_rewards}")
                    for uid, pts in broadcast_uid_rewards.items():
                        try:
                            hk = self._validator.metagraph.hotkeys[int(uid)]
                            if hk in config.BLACKLISTED_MINER_HOTKEYS:
                                bt.logging.info(f"[REWARDS] Ignoring broadcast rewards for blacklisted UID={uid} hotkey={hk[:12]}..")
                                continue
                        except Exception:
                            pass
                        combined_uid_rewards[uid] = combined_uid_rewards.get(uid, 0) + int(pts)
                except Exception as e:
                    bt.logging.debug(f"[ValidationClient.run] [BROADCAST] Failed to aggregate rewards: {e}")

                bt.logging.info(f"[REWARDS] Broadcasted rewards: {broadcast_uid_rewards}")

                # Penalized UIDs: only if 2+ validators (including local) penalized
                penalized_uids: set = set()
                local_penalties: Dict[str, int] = {}
                # Track validator counts per UID (local + broadcast)
                uid_validator_counts: Dict[int, int] = {}
                
                try:
                    local_penalties = self._validator._miner_penalty.get_penalties(epoch=target_epoch)
                    bt.logging.debug(f"[ValidationClient.run] Retrieved local_penalties: {local_penalties}")
                    # Count local validator penalties (counts as 1 validator per UID)
                    for hk, cnt in local_penalties.items():
                        if cnt > 0 and hk in self._validator.metagraph.hotkeys:
                            uid = self._validator.metagraph.hotkeys.index(hk)
                            uid_validator_counts[uid] = uid_validator_counts.get(uid, 0) + 1
                except Exception as e:
                    bt.logging.debug(f"[ValidationClient.run] [PENALTIES] Failed to get local penalties: {e}")

                # bt.logging.info(f"[PENALTIES] Local penalties: {local_penalties}")

                try:
                    # Get count of unique validators that penalized each UID from broadcasts
                    broadcast_validator_counts = self._validator._penalty_broadcasts.get_validator_penalty_counts(target_epoch)
                    bt.logging.debug(f"[ValidationClient.run] Broadcast validator penalty counts: {broadcast_validator_counts}")
                    # Add broadcast counts to total
                    for uid, count in broadcast_validator_counts.items():
                        uid_validator_counts[uid] = uid_validator_counts.get(uid, 0) + count
                except Exception as e:
                    bt.logging.debug(f"[ValidationClient.run] [PENALTY_BROADCAST] Failed to aggregate penalties: {e}")

                # Only penalize if 2+ validators (local + broadcast) penalized
                for uid, count in uid_validator_counts.items():
                    if count >= 2:
                        penalized_uids.add(uid)
                        bt.logging.info(f"[PENALTIES] UID={uid} penalized by {count} validators (local + broadcast), will zero rewards")

                # bt.logging.info(f"[PENALTY_BROADCAST] Broadcasted penalties: {broadcast_penalized}")

                # Build rewards list, zero only if penalties >= rewards
                rewards = []
                bt.logging.debug(f"[ValidationClient.run] Building rewards list, penalized_uids: {penalized_uids}")
                for uid, pts in combined_uid_rewards.items():
                    try:
                        hk = self._validator.metagraph.hotkeys[int(uid)]
                    except Exception:
                        bt.logging.debug(f"[ValidationClient.run] Could not resolve hotkey for UID={uid}, skipping.")
                        continue
                    penalty_count = uid_validator_counts.get(uid, 0)
                    # Apply reward if reward_count > 0 AND reward_count > penalty_count
                    if pts > 0 and pts > penalty_count:
                        bt.logging.info(f"[REWARDS] Applying reward for UID={uid} hotkey={hk[:12]}... (rewards={pts} > penalties={penalty_count})")
                        rewards.append(Reward(hotkey=hk, reward=int(pts), epoch=target_epoch))
                    else:
                        bt.logging.info(f"[PENALTIES] Zeroing reward for UID={uid} hotkey={hk[:12]}... (rewards={pts} <= penalties={penalty_count})")
                        rewards.append(Reward(hotkey=hk, reward=0, epoch=target_epoch))
                bt.logging.debug(f"[ValidationClient.run] Calculating weights from rewards list (len={len(rewards)})")
                weights = calculate_weights(rewards, self._validator.metagraph)
                # bt.logging.info(f"[REWARDS] Weights: {weights}")
                bt.logging.debug(f"[ValidationClient.run] Updating scores with new weights")
                self._validator.update_scores(weights, self._validator.metagraph.uids.tolist())

                # Send each active miner their raw reward/penalty counts for this epoch
                try:
                    score_start_block = int(target_epoch) * int(config.BLOCK_LENGTH)
                    score_end_block = (int(target_epoch) + 1) * int(config.BLOCK_LENGTH) - 1
                    active_hotkeys = set((local_rewards_map or {}).keys()) | set((local_penalties or {}).keys())

                    async def _send_score(hk: str) -> None:
                        try:
                            if hk not in self._validator.metagraph.hotkeys:
                                return
                            uid = self._validator.metagraph.hotkeys.index(hk)
                            axon = self._validator.metagraph.axons[uid]
                            r = int((local_rewards_map or {}).get(hk, 0))
                            p = int((local_penalties or {}).get(hk, 0))
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
                    bt.logging.info(f"[SCORE] Sent epoch={target_epoch} scores to {len(active_hotkeys)} miner(s)")
                except Exception as e:
                    bt.logging.debug(f"[ValidationClient.run] [SCORE] Failed to send scores: {e}")

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
                        rewards_payload = [
                            {
                                "start_block": start_block,
                                "stop_block": stop_block,
                                "hotkey": r.hotkey,
                                "points": float(getattr(r, "reward", 0)),
                            }
                            for r in rewards
                        ]

                        # Penalties: API expects {hotkey, reason}
                        penalties_payload = [
                            {
                                "hotkey": hk,
                                "reason": f"epoch={int(target_epoch)} count={int(cnt)}",
                            }
                            for hk, cnt in (local_penalties or {}).items()
                            if int(cnt) > 0
                        ]

                        if rewards_payload:
                            await self.api_client.submit_rewards(rewards=rewards_payload)
                        if penalties_payload:
                            await self.api_client.submit_penalties(penalties=penalties_payload)
                        self._last_api_submit_epoch = int(target_epoch)
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