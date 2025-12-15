# talisman_ai/validator/validation_client.py
import asyncio
from typing import Any, Dict, List, Callable, Optional
import httpx
import bittensor as bt
import time

from talisman_ai import config
from talisman_ai.utils.api_client import TalismanAPIClient
from talisman_ai.utils.api_models import TweetWithUser
from talisman_ai.utils.tweet_store import TweetStore
from talisman_ai.utils.reward import MinerReward
from talisman_ai.utils.penalty import MinerPenalty
from talisman_ai.utils.api_models import PenaltyCreate
from talisman_ai.utils.burn import calculate_weights
from talisman_ai.models.reward import Reward

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

    async def run(
        self,
        on_tweets: Callable[[TweetWithUser], Any],
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
                        self.api_client.submit_penalties([PenaltyCreate(
                            hotkey=tweet.hotkey,
                            reason="Timeout",
                        )])
                    unscored_tweets = (await self.api_client.get_unscored_tweets(limit=config.MINER_BATCH_SIZE)) + [tweet.tweet for tweet in self._validator._tweet_store.get_unprocessed_tweets()]
                    
                except Exception as e:
                    bt.logging.warning(f"[VALIDATION] Failed to fetch unscored tweets: {e}")
                    continue

                if unscored_tweets:
                    for tweet in unscored_tweets:
                        maybe_coro = on_tweets(tweet)
                        if asyncio.iscoroutine(maybe_coro):
                            await maybe_coro
                
                processed_tweets = self._validator._tweet_store.get_processed_tweets()
                for tweet in processed_tweets:
                    self._validator._miner_reward.add_reward(tweet.hotkey, 1)
                    await self._validator._submit_tweet_batch([tweet.tweet])
                
                rewards = self._validator._miner_reward.get_rewards(epoch=self._validator._miner_reward._get_current_epoch() - 2)
                # TODO THIS SHOULD USE KNOWLEDGE COMMITMENTS
                # TODO this should check if theres penalties, if so, penalize that hotkey
                rewards = [Reward(hotkey=hotkey, reward=reward, epoch=self._validator._miner_reward._get_current_epoch() - 2) for hotkey, reward in rewards.items()]
                weights = calculate_weights(rewards, self._validator.metagraph)
                self._validator.update_scores(weights, self._validator.metagraph.uids.tolist())
        except Exception as e:
            bt.logging.error(f"[VALIDATION] Failed to run validation client: {e}", exc_info=True)
            