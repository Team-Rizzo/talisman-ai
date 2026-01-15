# neurons/validator.py
# The MIT License (MIT)
# Copyright © 2023 Team Rizzo

"""
Validator entrypoint.
"""

import asyncio
import concurrent.futures
import gc
import time
from typing import List, Optional, Set

import bittensor as bt
from talisman_ai.base.validator import BaseValidatorNeuron
from talisman_ai.validator.forward import forward
from talisman_ai.validator.validation_client import ValidationClient
from talisman_ai.analyzer import setup_analyzer
import talisman_ai.protocol
from talisman_ai import config
from talisman_ai.utils.api_models import TweetWithAuthor, CompletedTweetSubmission, TelegramMessageForScoring, CompletedTelegramMessageSubmission, TelegramMessageAnalysis   
from talisman_ai.protocol import TweetBatch, TelegramBatch
from talisman_ai.utils.uids import get_random_uids
from talisman_ai.utils.tweet_store import TweetStore
from talisman_ai.utils.telegram_store import TelegramStore
from talisman_ai.utils.reward import MinerReward
from talisman_ai.utils.penalty import MinerPenalty
from talisman_ai.validator.reward_broadcast_store import RewardBroadcastStore
from talisman_ai.validator.penalty_broadcast_store import PenaltyBroadcastStore
from talisman_ai.protocol import ValidatorRewards
from talisman_ai.protocol import ValidatorPenalties
from talisman_ai.analyzer.scoring import validate_miner_batch, validate_miner_telegram_batch
from talisman_ai.analyzer import setup_telegram_analyzer
class Validator(BaseValidatorNeuron):
    """
    Validator neuron for SN45.

    Clean flow:
    - Poll coordination API for tweets to process
    - Batch tweets and query miners over Bittensor (TweetBatch synapse)
    - Validate miner batches and mark tweets completed back to the API
    - Accumulate epoch rewards/penalties, broadcast to other validators, and set on-chain weights
    """

    def __init__(self, bt_config=None):
        # NOTE: this arg name must not shadow the imported `talisman_ai.config` module.
        super(Validator, self).__init__(config=bt_config)

        # Use a bounded thread pool for CPU-bound validation tasks.
        # Provides predictable resource usage regardless of system core count.
        self._validation_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=8,
            thread_name_prefix="validation_"
        )
        bt.logging.info("[INIT] Created validation executor with 8 workers")

        bt.logging.info("load_state()")
        self.load_state()

        # Initialize analyzer once (reused for all validations)
        bt.logging.info("[VALIDATION] Initializing analyzer...")
        self._analyzer = setup_analyzer()
        self._telegram_analyzer = setup_telegram_analyzer()
        bt.logging.info("[VALIDATION] Analyzer initialized")

        # Initialize validation client
        self._validation_client = ValidationClient(validator=self, wallet=self.wallet)
        self._validation_task: Optional[asyncio.Task] = None
        self._tweet_store = TweetStore()
        self._telegram_store = TelegramStore()
        # MinerReward / MinerPenalty expect a callable that returns the current block.
        # In Bittensor, `self.block` is an integer attribute (updated during sync), not a function.
        self._miner_reward = MinerReward(config.BLOCK_LENGTH, lambda: int(self.block))
        self._miner_penalty = MinerPenalty(config.BLOCK_LENGTH, lambda: int(self.block))
        # Rewards broadcast store: holds validator↔validator reward messages for delayed application.
        self._reward_broadcasts = RewardBroadcastStore()
        self._reward_broadcasts.load()
        # Penalties broadcast store: holds validator↔validator penalty messages for delayed application.
        self._penalty_broadcasts = PenaltyBroadcastStore()
        self._penalty_broadcasts.load()
        
        self._tweet_store.load_from_file()
        self._telegram_store.load_from_file()
        # Persisted stores expect a callable `block()`; pass a lambda (self.block is an int).
        self._miner_reward.load_from_file(block=lambda: int(self.block))
        self._miner_penalty.load_from_file(block=lambda: int(self.block))

        # Validator dispatches TweetBatch to miners (fire-and-forget).
        # Miners push analyzed TweetBatch back to this validator's axon when ready.
        self._miner_dispatch_semaphore = asyncio.Semaphore(
            max(1, int(getattr(config, "VALIDATOR_MINER_QUERY_CONCURRENCY", 8)))
        )
        self._pending_miner_tasks: Set[asyncio.Task] = set()
        self._max_pending_miner_tasks: int = int(
            getattr(config, "VALIDATOR_MAX_PENDING_MINER_TASKS", 256)
        )
        
    async def forward_tweets(self, synapse: talisman_ai.protocol.TweetBatch) -> talisman_ai.protocol.TweetBatch:
        """
        The synapse is a TweetBatch from the miner
        """
        # Push-based mining: this *is* where we validate miner results.
        miner_hotkey = synapse.dendrite.hotkey if synapse.dendrite else None
        if not miner_hotkey:
            return synapse

        bt.logging.info(f"[VALIDATION] Received TweetBatch with {len(synapse.tweet_batch)} tweet(s) from miner {miner_hotkey[:12]}..")

        # Build the original batch from the local store to prevent cherry-picking and
        # to avoid penalizing miners for the initial "ack" response (which has no analysis).
        sent_batch: List[TweetWithAuthor] = []
        for returned in synapse.tweet_batch:
            tid = str(getattr(returned, "id", ""))
            if not tid:
                continue
            try:
                # Only accept results for tweets we currently consider "processing" for that miner.
                # If we can't match (e.g. validator restarted, timeout already requeued), ignore quietly.
                if self._tweet_store.get_status(tid).value != "Processing":
                    return synapse
                if self._tweet_store.get_hotkey(tid) != miner_hotkey:
                    return synapse
                sent_batch.append(self._tweet_store.get_tweet(tid))
            except Exception:
                return synapse

        if not sent_batch:
            return synapse

        await self._handle_miner_batch_response(synapse.tweet_batch, miner_hotkey, sent_batch)
        return synapse

    async def forward_telegram_messages(self, synapse: talisman_ai.protocol.TelegramBatch) -> talisman_ai.protocol.TelegramBatch:
        """
        The synapse is a TelegramBatch from the miner
        """
        # Push-based mining: this *is* where we validate miner results.
        miner_hotkey = synapse.dendrite.hotkey if synapse.dendrite else None
        if not miner_hotkey:
            return synapse

        bt.logging.info(f"[VALIDATION] Received TelegramBatch with {len(synapse.message_batch)} message(s) from miner {miner_hotkey[:12]}..")

        # Build the original batch from the local store to prevent cherry-picking and
        # to avoid penalizing miners for the initial "ack" response (which has no analysis).
        sent_batch: List[TelegramMessageForScoring] = []
        for returned in synapse.message_batch:
            msg_id = str(getattr(returned, "id", ""))
            if not msg_id:
                continue
            try:
                # Only accept results for messages we currently consider "processing" for that miner.
                # If we can't match (e.g. validator restarted, timeout already requeued), ignore quietly.
                if self._telegram_store.get_status(msg_id).value != "Processing":
                    return synapse
                if self._telegram_store.get_hotkey(msg_id) != miner_hotkey:
                    return synapse
                sent_batch.append(self._telegram_store.get_message(msg_id))
            except Exception:
                return synapse

        if not sent_batch:
            return synapse

        await self._handle_telegram_miner_batch_response(synapse.message_batch, miner_hotkey, sent_batch)
        return synapse

    def _track_task(self, task: asyncio.Task) -> None:
        self._pending_miner_tasks.add(task)

        def _done(t: asyncio.Task) -> None:
            self._pending_miner_tasks.discard(t)
            try:
                exc = t.exception()
                if exc is not None:
                    bt.logging.debug(f"[VALIDATION] Miner dispatch task failed: {exc}")
            except asyncio.CancelledError:
                pass
            except Exception:
                pass

        task.add_done_callback(_done)

    async def _dispatch_miner_batch(self, miner_batch: List[TweetWithAuthor], uid: int) -> None:
        async with self._miner_dispatch_semaphore:
            await self._process_miner_batch(miner_batch, uid)

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
                f"[VALIDATION] Batch size mismatch from miner {miner_hotkey} "
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
        loop = asyncio.get_running_loop()
        is_valid, validation_result = await loop.run_in_executor(
            self._validation_executor,
            validate_miner_batch, tweet_batch, self._analyzer, 1
        )
        if not is_valid:
            # Log detailed rejection reason
            discrepancies = validation_result.get("discrepancies", [])
            match_rate = validation_result.get("match_rate", 0.0)
            bt.logging.warning(
                f"[VALIDATION] Batch validation FAILED for miner {miner_hotkey} "
                f"match_rate={match_rate:.1%}, discrepancies={len(discrepancies)}"
            )
            for disc in discrepancies:
                reason = disc.get("reason", "unknown")
                preview = disc.get("post_preview", "")
                if reason == "classification_mismatch":
                    field_results = disc.get("field_results", {})
                    failed_fields = [k for k, v in field_results.items() if not v]
                    miner_vals = disc.get("miner", {})
                    validator_vals = disc.get("validator", {})
                    # Log each failed field with miner vs validator values
                    field_comparisons = []
                    for f in failed_fields:
                        m = miner_vals.get(f, "?")
                        v = validator_vals.get(f, "?")
                        field_comparisons.append(f"{f}(m={m}|v={v})")
                    bt.logging.warning(
                        f"[VALIDATION] Mismatch for {miner_hotkey}: {', '.join(field_comparisons)} | preview={preview[:100]}"
                    )
                else:
                    bt.logging.warning(f"[VALIDATION] Rejection for {miner_hotkey}: reason={reason}, preview={preview[:100]}")
            
            self._miner_penalty.add_penalty(miner_hotkey, 1)
            for tweet in tweet_batch:
                try:
                    self._tweet_store.reset_to_unprocessed(tweet.id)
                except Exception:
                    pass
            return False

        bt.logging.info(f"[VALIDATION] Batch validation PASSED for miner {miner_hotkey}")
        # Batch accepted: persist analyzed tweets, mark processed, and reward once per tweet.
        for tweet in tweet_batch:
            # Ensure store has the analyzed tweet for API submission.
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

    async def _handle_telegram_miner_batch_response(
        self,
        message_batch: List[TelegramMessageForScoring],
        miner_hotkey: str,
        sent_batch: List[TelegramMessageForScoring],
    ) -> bool:
        """
        Validate a miner's TelegramBatch response and apply rewards/penalties exactly once per message.

        Args:
            message_batch: The batch returned by the miner.
            miner_hotkey: The miner's hotkey.
            sent_batch: The original batch sent to the miner (for size verification).

        Returns:
            True if batch accepted, False otherwise.
        """
        # Miner must return exactly what was sent (no cherry-picking).
        if len(message_batch) != len(sent_batch):
            bt.logging.warning(
                f"[VALIDATION] Telegram batch size mismatch from miner {miner_hotkey} "
                f"sent {len(sent_batch)}, got {len(message_batch)}"
            )
            self._miner_penalty.add_penalty(miner_hotkey, 1)
            for msg in sent_batch:
                try:
                    self._telegram_store.reset_to_unprocessed(msg.id)
                except Exception:
                    pass
            return False

        # Validate by re-running analyzer on sampled messages.
        loop = asyncio.get_running_loop()
        is_valid, validation_result = await loop.run_in_executor(
            self._validation_executor,
            validate_miner_telegram_batch, message_batch, self._telegram_analyzer, 1
        )
        if not is_valid:
            # Log detailed rejection reason
            discrepancies = validation_result.get("discrepancies", [])
            match_rate = validation_result.get("match_rate", 0.0)
            bt.logging.warning(
                f"[VALIDATION] Telegram batch validation FAILED for miner {miner_hotkey} "
                f"match_rate={match_rate:.1%}, discrepancies={len(discrepancies)}"
            )
            for disc in discrepancies:
                reason = disc.get("reason", "unknown")
                preview = disc.get("message_preview", "")
                if reason == "classification_mismatch":
                    field_results = disc.get("field_results", {})
                    failed_fields = [k for k, v in field_results.items() if not v]
                    miner_vals = disc.get("miner", {})
                    validator_vals = disc.get("validator", {})
                    # Log each failed field with miner vs validator values
                    field_comparisons = []
                    for f in failed_fields:
                        m = miner_vals.get(f, "?")
                        v = validator_vals.get(f, "?")
                        field_comparisons.append(f"{f}(m={m}|v={v})")
                    bt.logging.warning(
                        f"[VALIDATION] Telegram mismatch for {miner_hotkey}: {', '.join(field_comparisons)} | preview={preview[:100]}"
                    )
                else:
                    bt.logging.warning(f"[VALIDATION] Telegram rejection for {miner_hotkey}: reason={reason}, preview={preview[:100]}")
            
            self._miner_penalty.add_penalty(miner_hotkey, 1)
            for msg in message_batch:
                try:
                    self._telegram_store.reset_to_unprocessed(msg.id)
                except Exception:
                    pass
            return False

        bt.logging.info(f"[VALIDATION] Telegram batch validation PASSED for miner {miner_hotkey}")
        # Batch accepted: persist analyzed messages, mark processed, and reward once per message.
        for msg in message_batch:
            # Ensure store has the analyzed message for API submission.
            try:
                self._telegram_store.update_message(msg.id, msg)
            except Exception:
                # If missing, add it.
                self._telegram_store.add_message(msg, message_id=msg.id, hotkey=miner_hotkey, set_as_processing=False, overwrite=True)

            try:
                self._telegram_store.set_processed(msg.id)
            except Exception:
                pass

            # Idempotent reward: only reward once per message_id.
            if not self._telegram_store.is_rewarded(msg.id):
                self._miner_reward.add_reward(miner_hotkey, 1)
                try:
                    self._telegram_store.mark_rewarded(msg.id)
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
        # Select miners from the metagraph for each batch (exclude ourselves).
        # NOTE: get_random_uids() already filters to serving axons and applies vpermit limits.
        uids = list(get_random_uids(self, k=len(miner_batches), exclude=[int(self.uid)]))

        for miner_batch, uid in zip(miner_batches, uids):
            if len(self._pending_miner_tasks) >= self._max_pending_miner_tasks:
                bt.logging.warning(
                    f"[VALIDATION] Too many pending miner dispatch tasks ({len(self._pending_miner_tasks)}); "
                    f"skipping scheduling remaining batches this tick."
                )
                break
            task = asyncio.create_task(self._dispatch_miner_batch(miner_batch, int(uid)))
            self._track_task(task)

    async def _on_telegram_messages(self, messages: List[TelegramMessageForScoring]):
        """
        Process multiple telegram messages in batch.
        
        Args:
            messages: List of TelegramMessageForScoring
        """
        if not messages:
            return
        
        bt.logging.info(f"[VALIDATION] Processing {len(messages)} telegram messages in batch")
        for msg in messages:
            # Preserve existing store entries (avoid losing processed/submitted/rewarded flags).
            self._telegram_store.add_message(msg, set_as_processing=False, overwrite=False)
        miner_batches = []
        for i in range(0, len(messages), config.MINER_BATCH_SIZE):
            miner_batches.append(messages[i:i + config.MINER_BATCH_SIZE])
        # Select miners from the metagraph for each batch (exclude ourselves).
        uids = list(get_random_uids(self, k=len(miner_batches), exclude=[int(self.uid)]))

        for miner_batch, uid in zip(miner_batches, uids):
            if len(self._pending_miner_tasks) >= self._max_pending_miner_tasks:
                bt.logging.warning(
                    f"[VALIDATION] Too many pending miner dispatch tasks ({len(self._pending_miner_tasks)}); "
                    f"skipping scheduling remaining telegram batches this tick."
                )
                break
            task = asyncio.create_task(self._dispatch_telegram_miner_batch(miner_batch, int(uid)))
            self._track_task(task)

    async def _dispatch_telegram_miner_batch(self, miner_batch: List[TelegramMessageForScoring], uid: int) -> None:
        async with self._miner_dispatch_semaphore:
            await self._process_telegram_miner_batch(miner_batch, uid)
            
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
            Dispatch result synapse (ack), or None on failure.
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
            axon = self.metagraph.axons[uid]
            responses = await self.dendrite.forward(
                axons=[axon],
                synapse=tweet_batch,
                timeout=float(getattr(config, "MINER_SEND_TIMEOUT", 6.0)),
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

            # Miners are expected to ack immediately and push results back to our axon later.
            return responses[0]
        except Exception as e:
            bt.logging.error(f"[VALIDATION] Failed to process miner batch: {e}", exc_info=True)
            for tweet in miner_batch:
                try:
                    self._tweet_store.reset_to_unprocessed(tweet.id)
                except Exception:
                    pass
            return None

    async def _process_telegram_miner_batch( 
        self, 
        miner_batch: List[TelegramMessageForScoring],
        uid: int
    ) -> TelegramBatch:
        """
        Process a telegram miner batch.
        
        Args:
            miner_batch: List of telegram messages to send
            uid: Miner uid to query
        
        Returns:
            Dispatch result synapse (ack), or None on failure.
        """
        try:
            miner_hotkey = None
            try:
                miner_hotkey = self.metagraph.hotkeys[int(uid)]
            except Exception:
                miner_hotkey = None

            # Mark messages as processing immediately (record attribution + start time).
            for msg in miner_batch:
                # Ensure message exists in the store.
                self._telegram_store.add_message(msg, message_id=msg.id, hotkey=miner_hotkey, set_as_processing=False, overwrite=False)
                try:
                    self._telegram_store.set_processing(msg.id, hotkey=miner_hotkey)
                except Exception:
                    pass

            telegram_batch = TelegramBatch(
                message_batch=miner_batch
            )
            axon = self.metagraph.axons[uid]
            responses = await self.dendrite.forward(
                axons=[axon],
                synapse=telegram_batch,
                timeout=float(getattr(config, "MINER_SEND_TIMEOUT", 6.0)),
                deserialize=True
            )
            if not responses[0].dendrite.status_code == 200:
                bt.logging.error(f"[VALIDATION] Failed to process telegram miner batch: {responses[0].dendrite.status_message}")
                # Requeue locally; do not reward; do not penalize on transport errors.
                for msg in miner_batch:
                    try:
                        self._telegram_store.reset_to_unprocessed(msg.id)
                    except Exception:
                        pass
                return None

            # Miners are expected to ack immediately and push results back to our axon later.
            return responses[0]
        except Exception as e:
            bt.logging.error(f"[VALIDATION] Failed to process telegram miner batch: {e}", exc_info=True)
            for msg in miner_batch:
                try:
                    self._telegram_store.reset_to_unprocessed(msg.id)
                except Exception:
                    pass
            return None
    
    async def _submit_tweet_batch(self, tweet_batch: List[TweetWithAuthor]):
        """Submit a tweet batch to the API"""
        completed_tweets = []
        for tweet in tweet_batch:
            # Miner responses are expected to always include analysis.
            if tweet.analysis is None:
                bt.logging.warning(
                    f"[VALIDATION] Skipping tweet {tweet.id} in submission: missing miner analysis"
                )
                continue

            completed_tweets.append(
                CompletedTweetSubmission(
                    tweet_id=tweet.id,
                    sentiment=tweet.analysis.sentiment or "neutral",
                    subnet_id=tweet.analysis.subnet_id,
                    subnet_name=tweet.analysis.subnet_name,
                    content_type=tweet.analysis.content_type,
                    technical_quality=tweet.analysis.technical_quality,
                    market_analysis=tweet.analysis.market_analysis,
                    impact_potential=tweet.analysis.impact_potential,
                    relevance_confidence=getattr(tweet.analysis, "relevance_confidence", None),
                )
            )
        response = await self._validation_client.api_client.submit_completed_tweets(completed_tweets)
        return response

    async def _submit_telegram_batch(self, message_batch: List[TelegramMessageForScoring]):
        """Submit a telegram message batch to the API"""
        completed_messages = []
        for msg in message_batch:
            # Miner responses are expected to always include analysis.
            if msg.analysis is None:
                bt.logging.warning(
                    f"[VALIDATION] Skipping telegram message {msg.id} in submission: missing miner analysis"
                )
                continue

            completed_messages.append(
                CompletedTelegramMessageSubmission(
                    message_id=msg.id,
                    sentiment=msg.analysis.sentiment or "neutral",
                    subnet_id=msg.analysis.subnet_id,
                    subnet_name=msg.analysis.subnet_name,
                    content_type=msg.analysis.content_type,
                    technical_quality=msg.analysis.technical_quality,
                    market_analysis=msg.analysis.market_analysis,
                    impact_potential=msg.analysis.impact_potential,
                    relevance_confidence=getattr(msg.analysis, "relevance_confidence", None),
                )
            )
        response = await self._validation_client.api_client.submit_completed_telegram_messages(completed_messages)
        return response

    async def forward(self):
        """
        Main validator forward loop.
        
        Starts the validation client on first invocation. The client runs independently
        in the background.
        """
        # Start or restart validation client if crashed
        if self._validation_task is None or self._validation_task.done():
            if self._validation_task is not None and self._validation_task.done():
                # Log what killed it
                try:
                    exc = self._validation_task.exception()
                    if exc:
                        bt.logging.warning(f"[VALIDATION] Client crashed: {type(exc).__name__}: {exc}. Restarting...")
                except asyncio.CancelledError:
                    pass
            self._validation_task = asyncio.create_task(
                self._validation_client.run(
                    on_tweets=self._on_tweets,
                    on_telegram_messages=self._on_telegram_messages,
                )
            )
            bt.logging.info("[VALIDATION] Started validation client")

        self.save_state()
        
        # Periodically prune old data to prevent memory growth (every 100 steps)
        if self.step % 100 == 0:
            self._prune_stores()
        
        return await forward(self)
    
    def _prune_stores(self):
        """Prune old data from stores to maintain bounded memory usage."""
        try:
            # Prune tweet store: remove submitted tweets and old unprocessed ones
            self._tweet_store.prune_old_tweets(max_age_seconds=3600, max_tweets=1000)
            self._tweet_store.save_to_file()
            
            # Prune telegram store: remove submitted messages and old unprocessed ones
            self._telegram_store.prune_old_messages(max_age_seconds=3600, max_messages=1000)
            self._telegram_store.save_to_file()
            
            # Save reward/penalty stores (pruning happens in update_current_epoch)
            self._miner_reward.save_to_file()
            self._miner_penalty.save_to_file()
            
            # Explicit GC helps long-running processes reclaim memory promptly.
            collected = gc.collect()
            
            bt.logging.info(f"[PRUNE] Pruned stores at step {self.step}, GC collected {collected} objects")
        except Exception as e:
            bt.logging.warning(f"[PRUNE] Failed to prune stores: {e}")

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
