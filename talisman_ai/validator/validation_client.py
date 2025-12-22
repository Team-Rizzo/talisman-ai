# talisman_ai/validator/validation_client.py
import asyncio
from typing import Any, Dict, List, Callable, Optional
import httpx
import bittensor as bt
import time
import random

from talisman_ai import config
from talisman_ai.utils.api_client import TalismanAPIClient
from talisman_ai.utils.api_models import TweetWithAuthor
from talisman_ai.utils.burn import calculate_weights
from talisman_ai.models.reward import Reward
from talisman_ai.protocol import ValidatorRewards
from talisman_ai.protocol import ValidatorPenalties
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
    ):
        """
        Main validation loop.
        
        Args:
            on_tweets: Callback for batch of tweets (async or sync)
        """
        self._running = True
        bt.logging.info("[ValidationClient.run] Entering main validation loop (poll_interval=%ss, scores_interval=%s blocks)", self.poll_seconds, self.scores_block_interval)

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
                    unscored_tweets = (await self.api_client.get_unscored_tweets(limit=config.MINER_BATCH_SIZE)) + [tweet.tweet for tweet in self._validator._tweet_store.get_unprocessed_tweets()]
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

                # Persist local state.
                try:
                    bt.logging.debug("[ValidationClient.run] Saving tweet store to disk")
                    self._validator._tweet_store.save_to_file()
                except Exception as e:
                    bt.logging.debug(f"[ValidationClient.run] Failed to persist tweet store: {e}")

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

                # Broadcasted rewards
                try:
                    broadcast_uid_rewards = self._validator._reward_broadcasts.aggregate_epoch(target_epoch)
                    bt.logging.debug(f"[ValidationClient.run] Aggregated broadcast_uid_rewards: {broadcast_uid_rewards}")
                    for uid, pts in broadcast_uid_rewards.items():
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

                # Build rewards list, zero for penalized
                rewards = []
                bt.logging.debug(f"[ValidationClient.run] Building rewards list, penalized_uids: {penalized_uids}")
                for uid, pts in combined_uid_rewards.items():
                    try:
                        hk = self._validator.metagraph.hotkeys[int(uid)]
                    except Exception:
                        bt.logging.debug(f"[ValidationClient.run] Could not resolve hotkey for UID={uid}, skipping.")
                        continue
                    if uid in penalized_uids:
                        bt.logging.info(f"[PENALTIES] Zeroing reward for penalized miner UID={uid} hotkey={hk[:12]}...")
                        rewards.append(Reward(hotkey=hk, reward=0, epoch=target_epoch))
                    else:
                        rewards.append(Reward(hotkey=hk, reward=int(pts), epoch=target_epoch))
                bt.logging.debug(f"[ValidationClient.run] Calculating weights from rewards list (len={len(rewards)})")
                weights = calculate_weights(rewards, self._validator.metagraph)
                # bt.logging.info(f"[REWARDS] Weights: {weights}")
                bt.logging.debug(f"[ValidationClient.run] Updating scores with new weights")
                self._validator.update_scores(weights, self._validator.metagraph.uids.tolist())
                bt.logging.debug(f"[ValidationClient.run] Sleeping for {self.poll_seconds} seconds before next poll loop")
                
                self._validator._penalty_broadcasts.save()
                self._validator._reward_broadcasts.save()
                self._validator._tweet_store.save_to_file()
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