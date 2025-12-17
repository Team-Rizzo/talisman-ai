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
from talisman_ai.utils.tweet_store import TweetStore
from talisman_ai.utils.reward import MinerReward
from talisman_ai.utils.penalty import MinerPenalty
from talisman_ai.utils.api_models import PenaltyCreate
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
        wallet: Optional[bt.wallet] = None,
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
        # Loop cadence. Keep defaults aligned with config.
        self.poll_seconds = poll_seconds if poll_seconds is not None else config.VALIDATION_POLL_SECONDS
        self.scores_block_interval = scores_block_interval if scores_block_interval is not None else config.SCORES_BLOCK_INTERVAL

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
        bt.logging.info(f"[VALIDATION] Starting validation client (poll_interval={self.poll_seconds}s, scores_interval={self.scores_block_interval} blocks)")

        try:
            while self._running:
                try:
                    # Check if we need to fetch scores (based on API's current window)
                    timed_out_tweets = self._validator._tweet_store.get_timeouts()
                    for tweet in timed_out_tweets:
                        self._validator._tweet_store.reset_to_unprocessed(tweet.tweet.id)
                        self._validator._miner_penalty.add_penalty(tweet.hotkey, "Timeout")
                        # Penalties intentionally ignored in the rewards-only commitments rollout.
                        # Also: submit_penalties is async; leaving this commented avoids un-awaited coroutine warnings.
                        # await self.api_client.submit_penalties([PenaltyCreate(
                        #     hotkey=tweet.hotkey,
                        #     reason="Timeout",
                        # )])
                    unscored_tweets = (await self.api_client.get_unscored_tweets(limit=config.MINER_BATCH_SIZE)) + [tweet.tweet for tweet in self._validator._tweet_store.get_unprocessed_tweets()]
                    
                except Exception as e:
                    bt.logging.warning(f"[VALIDATION] Failed to fetch unscored tweets: {e}")
                    continue

                if unscored_tweets:
                    maybe_coro = on_tweets(unscored_tweets)
                    if asyncio.iscoroutine(maybe_coro):
                        await maybe_coro
                
                processed_tweets = self._validator._tweet_store.get_processed_tweets()
                for tweet in processed_tweets:
                    self._validator._miner_reward.add_reward(tweet.hotkey, 1)
                    await self._validator._submit_tweet_batch([tweet.tweet])
                
                current_epoch = self._validator._miner_reward._get_current_epoch()
                try:
                    # Broadcast a single-epoch rewards snapshot to other validators (typically E-1).
                    # This replaces knowledge commitments (which are overwrite-only and tiny).
                    publish_epoch = current_epoch - 1
                    if publish_epoch >= 0:
                        # Convert hotkey->points to uid->points for compactness.
                        hotkey_points = self._validator._miner_reward.get_rewards(epoch=publish_epoch)
                        uid_points = {}
                        for hk, pts in hotkey_points.items():
                            if hk in self._validator.metagraph.hotkeys:
                                uid = self._validator.metagraph.hotkeys.index(hk)
                                uid_points[uid] = int(pts)
                        
                        # Also get penalties for broadcasting.
                        hotkey_penalties = self._validator._miner_penalty.get_penalties(epoch=publish_epoch)
                        uid_penalties = {}
                        for hk, cnt in hotkey_penalties.items():
                            if hk in self._validator.metagraph.hotkeys:
                                uid = self._validator.metagraph.hotkeys.index(hk)
                                uid_penalties[uid] = int(cnt)
                        
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
                        # Fan out to whitelisted validator hotkeys (permit + stake threshold),
                        # bounded by VALIDATOR_BROADCAST_MAX_TARGETS.
                        whitelist = set(
                            get_validator_hotkeys(
                                metagraph=self._validator.metagraph,
                                netuid=self._validator.config.netuid,
                            )
                        )
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
                        max_targets = int(getattr(config, "VALIDATOR_BROADCAST_MAX_TARGETS", 32))
                        if max_targets > 0 and len(axons) > max_targets:
                            axons = random.sample(axons, max_targets)
                        if axons:
                            # Broadcast both rewards and penalties.
                            await self._validator.dendrite.forward(
                                axons=axons,
                                synapse=rewards_syn,
                                timeout=12.0,
                                deserialize=True,
                            )
                            if uid_penalties:
                                await self._validator.dendrite.forward(
                                    axons=axons,
                                    synapse=penalties_syn,
                                    timeout=12.0,
                                    deserialize=True,
                                )
                except Exception as e:
                    bt.logging.debug(f"[BROADCAST] Publish failed: {e}")

                # Apply rewards for epoch E-2.
                target_epoch = current_epoch - 2

                # Combine local and broadcasted rewards (summing them together).
                combined_uid_rewards: Dict[int, int] = {}
                
                # Get local rewards (keyed by hotkey) and convert to uid->points.
                try:
                    local_rewards_map = self._validator._miner_reward.get_rewards(epoch=target_epoch)
                    for hk, pts in local_rewards_map.items():
                        if hk in self._validator.metagraph.hotkeys:
                            uid = self._validator.metagraph.hotkeys.index(hk)
                            combined_uid_rewards[uid] = combined_uid_rewards.get(uid, 0) + int(pts)
                except Exception as e:
                    bt.logging.debug(f"[REWARDS] Failed to get local rewards: {e}")
                
                # Get broadcasted rewards and combine.
                try:
                    broadcast_uid_rewards = self._validator._reward_broadcasts.aggregate_epoch(target_epoch)
                    for uid, pts in broadcast_uid_rewards.items():
                        combined_uid_rewards[uid] = combined_uid_rewards.get(uid, 0) + int(pts)
                except Exception as e:
                    bt.logging.debug(f"[BROADCAST] Failed to aggregate rewards: {e}")

                # Get penalized UIDs from both local and broadcasted penalties.
                penalized_uids: set = set()
                
                # Get local penalties.
                try:
                    local_penalties = self._validator._miner_penalty.get_penalties(epoch=target_epoch)
                    for hk, cnt in local_penalties.items():
                        if cnt > 0 and hk in self._validator.metagraph.hotkeys:
                            uid = self._validator.metagraph.hotkeys.index(hk)
                            penalized_uids.add(uid)
                except Exception as e:
                    bt.logging.debug(f"[PENALTIES] Failed to get local penalties: {e}")
                
                # Get broadcasted penalties.
                try:
                    broadcast_penalized = self._validator._penalty_broadcasts.get_penalized_uids(target_epoch)
                    penalized_uids.update(broadcast_penalized)
                except Exception as e:
                    bt.logging.debug(f"[PENALTY_BROADCAST] Failed to aggregate penalties: {e}")

                # Build rewards list, setting reward to 0 for penalized miners.
                rewards = []
                for uid, pts in combined_uid_rewards.items():
                    try:
                        hk = self._validator.metagraph.hotkeys[int(uid)]
                    except Exception:
                        continue
                    if uid in penalized_uids:
                        bt.logging.info(f"[PENALTIES] Zeroing reward for penalized miner UID={uid} hotkey={hk[:12]}...")
                        rewards.append(Reward(hotkey=hk, reward=0, epoch=target_epoch))
                    else:
                        rewards.append(Reward(hotkey=hk, reward=int(pts), epoch=target_epoch))
                weights = calculate_weights(rewards, self._validator.metagraph)
                self._validator.update_scores(weights, self._validator.metagraph.uids.tolist())
                await asyncio.sleep(float(self.poll_seconds))
        except Exception as e:
            bt.logging.error(f"[VALIDATION] Failed to run validation client: {e}", exc_info=True)
            