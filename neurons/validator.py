# neurons/validator.py
# The MIT License (MIT)
# Copyright © 2023 Team Rizzo

import asyncio
import time
from typing import List, Optional
import bittensor as bt
from talisman_ai.base.validator import BaseValidatorNeuron
from talisman_ai.validator.forward import forward
from talisman_ai.validator.validation_client import ValidationClient
from talisman_ai.analyzer import setup_analyzer
import talisman_ai.protocol
from talisman_ai import config
from talisman_ai.utils.api_models import TweetWithAuthor, CompletedTweetSubmission   
from talisman_ai.protocol import TweetBatch
from talisman_ai.utils.uids import get_random_uids
from talisman_ai.utils.tweet_store import TweetStore
from talisman_ai.utils.reward import MinerReward
from talisman_ai.utils.penalty import MinerPenalty
from talisman_ai.validator.reward_broadcast_store import RewardBroadcastStore
from talisman_ai.validator.penalty_broadcast_store import PenaltyBroadcastStore
from talisman_ai.protocol import ValidatorRewards
from talisman_ai.protocol import ValidatorPenalties
from talisman_ai.analyzer.scoring import validate_miner_batch
class Validator(BaseValidatorNeuron):
    """
    Validator neuron for SN45.

    Clean flow:
    - Poll coordination API for tweets to process
    - Batch tweets and query miners over Bittensor (TweetBatch synapse)
    - Validate miner batches and mark tweets completed back to the API
    - Accumulate epoch rewards/penalties, broadcast to other validators, and set on-chain weights
    """

    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)

        bt.logging.info("load_state()")
        self.load_state()

        # Initialize analyzer once (reused for all validations)
        bt.logging.info("[VALIDATION] Initializing analyzer...")
        self._analyzer = setup_analyzer()
        bt.logging.info("[VALIDATION] Analyzer initialized")

        # Initialize validation client
        self._validation_client = ValidationClient(validator=self, wallet=self.wallet)
        self._validation_task: Optional[asyncio.Task] = None
        self._tweet_store = TweetStore()
        self._miner_reward = MinerReward(config.BLOCK_LENGTH, self.block)
        self._miner_penalty = MinerPenalty(config.BLOCK_LENGTH, self.block)
        # Rewards broadcast store: holds validator↔validator reward messages for delayed application.
        self._reward_broadcasts = RewardBroadcastStore()
        self._reward_broadcasts.load()
        # Penalties broadcast store: holds validator↔validator penalty messages for delayed application.
        self._penalty_broadcasts = PenaltyBroadcastStore()
        self._penalty_broadcasts.load()
        
        self._tweet_store.load_from_file()
        self._miner_reward.load_from_file(block=self.block)
        self._miner_penalty.load_from_file(block=self.block)
        
    async def forward_tweets(self, synapse: talisman_ai.protocol.TweetBatch) -> talisman_ai.protocol.TweetBatch:
        """
        The synapse is a TweetBatch from the miner
        """
        # Note: in the normal workflow we query miners via dendrite and handle the response in
        # `_process_miner_batch()`. This axon handler is retained for compatibility/testing.
        # Since we didn't initiate this request, we pass tweet_batch as sent_batch (no size check).
        miner_hotkey = synapse.dendrite.hotkey if synapse.dendrite else None
        if not miner_hotkey:
            return synapse
        await self._handle_miner_batch_response(synapse.tweet_batch, miner_hotkey, synapse.tweet_batch)
        return synapse

    async def _handle_miner_batch_response(
        self,
        tweet_batch: List[TweetWithAuthor],
        miner_hotkey: str,
        sent_batch: List[TweetWithAuthor],
    ) -> bool:
        """
        Validate a miner's TweetBatch response and apply rewards/penalties exactly once per tweet.

        Args:
            tweet_batch: The batch returned by the miner.
            miner_hotkey: The miner's hotkey.
            sent_batch: The original batch sent to the miner (for size verification).

        Returns:
            True if batch accepted, False otherwise.
        """
        # Miner must return exactly what was sent (no cherry-picking).
        if len(tweet_batch) != len(sent_batch):
            bt.logging.warning(
                f"[VALIDATION] Batch size mismatch from miner {miner_hotkey[:12]}.. "
                f"sent {len(sent_batch)}, got {len(tweet_batch)}"
            )
            self._miner_penalty.add_penalty(miner_hotkey, 1)
            for tweet in sent_batch:
                try:
                    self._tweet_store.reset_to_unprocessed(tweet.id)
                except Exception:
                    pass
            return False

        # Validate by re-running analyzer on sampled posts.
        is_valid, _result = await asyncio.to_thread(
            validate_miner_batch, tweet_batch, self._analyzer, 1
        )
        if not is_valid:
            self._miner_penalty.add_penalty(miner_hotkey, 1)
            for tweet in tweet_batch:
                try:
                    self._tweet_store.reset_to_unprocessed(tweet.id)
                except Exception:
                    pass
            return False

        # Batch accepted: persist enriched tweets, mark processed, and reward once per tweet.
        for tweet in tweet_batch:
            # Ensure store has the enriched tweet for API submission.
            try:
                self._tweet_store.update_tweet(tweet.id, tweet)
            except Exception:
                # If missing, add it.
                self._tweet_store.add_tweet(tweet, tweet_id=tweet.id, hotkey=miner_hotkey, set_as_processing=False, overwrite=True)

            try:
                self._tweet_store.set_processed(tweet.id)
            except Exception:
                pass

            # Idempotent reward: only reward once per tweet_id.
            if not self._tweet_store.is_rewarded(tweet.id):
                self._miner_reward.add_reward(miner_hotkey, 1)
                try:
                    self._tweet_store.mark_rewarded(tweet.id)
                except Exception:
                    pass

        return True
        
    async def _on_tweets(self, tweets: List[TweetWithAuthor]):
        """
        Process multiple tweets in batch (sequentially).
        
        Args:
            tweets: List of tweets
        """
        if not tweets:
            return
        
        bt.logging.info(f"[VALIDATION] Processing {len(tweets)} tweets in batch")
        for tweet in tweets:
            # Preserve existing store entries (avoid losing processed/submitted/rewarded flags).
            self._tweet_store.add_tweet(tweet, set_as_processing=False, overwrite=False)
        miner_batches = []  
        for i in range(0, len(tweets), config.MINER_BATCH_SIZE):
            miner_batches.append(tweets[i:i + config.MINER_BATCH_SIZE])
        uids = get_random_uids(self.metagraph, self.dendrite, k=len(miner_batches), is_alive=True)

        for miner_batch, uid in zip(miner_batches, uids):
            await self._process_miner_batch(miner_batch, uid)
            
    async def _process_miner_batch( 
        self, 
        miner_batch: List[TweetWithAuthor],
        uid: int
    ) -> TweetBatch:
        """
        Process a miner batch.
        
        Args:
            miner_batch: List of tweets to send
            uid: Miner uid to query
        
        Returns:
            Miner response synapse, or None on failure
        """
        try:
            miner_hotkey = None
            try:
                miner_hotkey = self.metagraph.hotkeys[int(uid)]
            except Exception:
                miner_hotkey = None

            # Mark tweets as processing immediately (record attribution + start time).
            for tweet in miner_batch:
                # Ensure tweet exists in the store.
                self._tweet_store.add_tweet(tweet, tweet_id=tweet.id, hotkey=miner_hotkey, set_as_processing=False, overwrite=False)
                try:
                    self._tweet_store.set_processing(tweet.id, hotkey=miner_hotkey)
                except Exception:
                    pass

            tweet_batch = TweetBatch(
                tweet_batch=miner_batch
            )
            responses = await self.dendrite.forward(
                axons=[self.metagraph.axons[uid]],
                synapse=tweet_batch,
                timeout=12.0,
                deserialize=True
            )
            if not responses[0].dendrite.status_code == 200:
                bt.logging.error(f"[VALIDATION] Failed to process miner batch: {responses[0].dendrite.status_message}")
                # Requeue locally; do not reward; do not penalize on transport errors.
                for tweet in miner_batch:
                    try:
                        self._tweet_store.reset_to_unprocessed(tweet.id)
                    except Exception:
                        pass
                return None

            response_syn = responses[0]
            # Apply validation + reward/penalty based on miner response.
            if miner_hotkey is None and response_syn.dendrite and response_syn.dendrite.hotkey:
                miner_hotkey = response_syn.dendrite.hotkey
            if miner_hotkey is None:
                # Can't attribute; requeue.
                for tweet in miner_batch:
                    try:
                        self._tweet_store.reset_to_unprocessed(tweet.id)
                    except Exception:
                        pass
                return None

            await self._handle_miner_batch_response(response_syn.tweet_batch, miner_hotkey, miner_batch)
            return response_syn
        except Exception as e:
            bt.logging.error(f"[VALIDATION] Failed to process miner batch: {e}", exc_info=True)
            for tweet in miner_batch:
                try:
                    self._tweet_store.reset_to_unprocessed(tweet.id)
                except Exception:
                    pass
            return None
    
    async def _submit_tweet_batch(self, tweet_batch: List[TweetWithAuthor]):
        """Submit a tweet batch to the API"""
        completed_tweets = []
        for tweet in tweet_batch:
            # Get sentiment from analysis if available, otherwise default to neutral
            sentiment = tweet.analysis.sentiment if tweet.analysis and tweet.analysis.sentiment else "neutral"
            completed_tweets.append(CompletedTweetSubmission(
                tweet_id=tweet.id,
                sentiment=sentiment
            ))
        response = await self._validation_client.api_client.submit_completed_tweets(completed_tweets)
        return response

    async def forward(self):
        """
        Main validator forward loop.
        
        Starts the validation client on first invocation. The client runs independently
        in the background.
        """
        if self._validation_task is None:
            self._validation_task = asyncio.create_task(
                self._validation_client.run(
                    on_tweets=self._on_tweets,
                )
            )
            bt.logging.info("[VALIDATION] Started validation client")

        self.save_state()
        return await forward(self)

    async def forward_validator_rewards(self, synapse: ValidatorRewards) -> ValidatorRewards:
        """
        Receive reward broadcasts from other validators and cache locally.
        """
        try:
            accepted, reason = self._reward_broadcasts.ingest(
                sender_hotkey=synapse.sender_hotkey,
                epoch=synapse.epoch,
                seq=synapse.seq,
                uid_points=synapse.uid_points,
            )
            # Persist quickly so we can apply E-2 even after restart.
            self._reward_broadcasts.save()
            if accepted:
                bt.logging.info(
                    f"[BROADCAST] Ingested rewards from {synapse.sender_hotkey[:12]}.. "
                    f"epoch={synapse.epoch} uids={len(synapse.uid_points)}"
                )
            else:
                bt.logging.debug(
                    f"[BROADCAST] Ignored rewards from {synapse.sender_hotkey[:12]}.. "
                    f"epoch={synapse.epoch} reason={reason}"
                )
        except Exception as e:
            bt.logging.debug(f"[BROADCAST] Failed to ingest rewards: {e}")
        return synapse

    async def forward_validator_penalties(self, synapse: ValidatorPenalties) -> ValidatorPenalties:
        """
        Receive penalty broadcasts from other validators and cache locally.
        """
        try:
            accepted, reason = self._penalty_broadcasts.ingest(
                sender_hotkey=synapse.sender_hotkey,
                epoch=synapse.epoch,
                seq=synapse.seq,
                uid_penalties=synapse.uid_penalties,
            )
            # Persist quickly so we can apply E-2 even after restart.
            self._penalty_broadcasts.save()
            if accepted:
                bt.logging.info(
                    f"[PENALTY_BROADCAST] Ingested penalties from {synapse.sender_hotkey[:12]}.. "
                    f"epoch={synapse.epoch} uids={len(synapse.uid_penalties)}"
                )
            else:
                bt.logging.debug(
                    f"[PENALTY_BROADCAST] Ignored penalties from {synapse.sender_hotkey[:12]}.. "
                    f"epoch={synapse.epoch} reason={reason}"
                )
        except Exception as e:
            bt.logging.debug(f"[PENALTY_BROADCAST] Failed to ingest penalties: {e}")
        return synapse


# Entrypoint
if __name__ == "__main__":
    with Validator() as validator:
        while True:
            bt.logging.info(f"Validator running... {time.time()}")
            time.sleep(5)
