# neurons/validator.py
# The MIT License (MIT)
# Copyright © 2023 Team Rizzo

"""
Validator entrypoint.
"""

import asyncio
import concurrent.futures
import copy
import gc
import time
from typing import List, Optional, Set

import bittensor as bt
from talisman_ai.base.validator import BaseValidatorNeuron
from talisman_ai.validator.forward import forward
from talisman_ai.validator.validation_client import ValidationClient
from talisman_ai.analyzer import setup_analyzer
from talisman_ai.analyzer import setup_news_analyzer
import talisman_ai.protocol
from talisman_ai import config
from talisman_ai.utils.api_models import TweetWithAuthor, CompletedTweetSubmission, TelegramMessageForScoring, CompletedTelegramMessageSubmission, TelegramMessageAnalysis, NewsArticleForScoring, CompletedNewsArticleSubmission
from talisman_ai.protocol import TweetBatch, TelegramBatch, ArticleBatch
from talisman_ai.utils.uids import get_random_uids
from talisman_ai.utils.tweet_store import TweetStore
from talisman_ai.utils.telegram_store import TelegramStore
from talisman_ai.utils.article_store import ArticleStore
from talisman_ai.utils.reward import MinerReward
from talisman_ai.utils.penalty import MinerPenalty
from talisman_ai.validator.reward_broadcast_store import RewardBroadcastStore
from talisman_ai.validator.penalty_broadcast_store import PenaltyBroadcastStore
from talisman_ai.protocol import ValidatorRewards
from talisman_ai.protocol import ValidatorPenalties
from talisman_ai.analyzer.scoring import validate_miner_batch, validate_miner_telegram_batch, validate_miner_article_batch
from talisman_ai.analyzer import setup_telegram_analyzer
from talisman_ai.utils.cooldown import MinerCooldownTracker
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

        _vw = int(getattr(config, "VALIDATION_MAX_WORKERS", 2))
        self._validation_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=_vw,
            thread_name_prefix="validation_"
        )
        bt.logging.info(f"[INIT] Created validation executor with {_vw} workers")

        bt.logging.info("load_state()")
        self.load_state()

        # Initialize analyzer once (reused for all validations)
        bt.logging.info("[VALIDATION] Initializing analyzer...")
        self._analyzer = setup_analyzer()
        self._telegram_analyzer = setup_telegram_analyzer()
        self._news_analyzer = setup_news_analyzer()
        bt.logging.info("[VALIDATION] News analyzer initialized")
        bt.logging.info("[VALIDATION] Analyzer initialized")

        # Initialize validation client
        self._validation_client = ValidationClient(validator=self, wallet=self.wallet)
        self._validation_task: Optional[asyncio.Task] = None
        self._tweet_store = TweetStore()
        self._telegram_store = TelegramStore()
        self._article_store = ArticleStore()
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
        self._article_store.load_from_file()
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
        self._validating_tweet_ids: set = set()
        self._validating_message_ids: set = set()
        self._validating_article_ids: set = set()
        self._cooldown_tracker = MinerCooldownTracker()

    def resync_metagraph(self):
        super().resync_metagraph()
        if hasattr(self, "_cooldown_tracker"):
            self._cooldown_tracker.prune(set(self.metagraph.hotkeys))

    async def forward_tweets(self, synapse: talisman_ai.protocol.TweetBatch) -> talisman_ai.protocol.TweetBatch:
        """
        Axon handler for miner push-back of analyzed TweetBatch results.

        Validates store state synchronously (fast), then queues LLM validation
        as a background task so the axon returns immediately and the miner
        does not hit a 30s dendrite timeout.
        """
        miner_hotkey = synapse.dendrite.hotkey if synapse.dendrite else None
        if not miner_hotkey:
            return synapse

        bt.logging.info(f"[VALIDATION] Received TweetBatch with {len(synapse.tweet_batch)} tweet(s) from miner {miner_hotkey[:12]}..")

        sent_batch: List[TweetWithAuthor] = []
        for returned in synapse.tweet_batch:
            tid = str(getattr(returned, "id", ""))
            if not tid:
                continue
            if tid in self._validating_tweet_ids:
                bt.logging.info(
                    f"[VALIDATION] Dropping TweetBatch from {miner_hotkey[:12]}.. "
                    f"tweet {tid} already being validated (replay blocked)"
                )
                return synapse
            try:
                status = self._tweet_store.get_status(tid).value
                if status != "Processing":
                    bt.logging.info(
                        f"[VALIDATION] Dropping TweetBatch from {miner_hotkey[:12]}.. "
                        f"tweet {tid} status={status} (expected Processing)"
                    )
                    return synapse
                if self._tweet_store.get_hotkey(tid) != miner_hotkey:
                    bt.logging.info(
                        f"[VALIDATION] Dropping TweetBatch from {miner_hotkey[:12]}.. "
                        f"tweet {tid} hotkey mismatch"
                    )
                    return synapse
                sent_batch.append(self._tweet_store.get_tweet(tid))
            except Exception:
                return synapse

        if not sent_batch:
            return synapse

        # Lock these tweet IDs so replays are rejected while validation runs.
        batch_tids = {str(getattr(r, "id", "")) for r in synapse.tweet_batch if getattr(r, "id", "")}
        self._validating_tweet_ids.update(batch_tids)

        # Reset the timeout clock — the miner delivered results, we just need
        # time to grade them. Without this, slow LLM validation could trigger
        # a false timeout penalty even though results arrived on time.
        for returned in synapse.tweet_batch:
            tid = str(getattr(returned, "id", ""))
            if tid and tid in self._tweet_store._tweets:
                self._tweet_store._tweets[tid].start_time = time.time()

        # Queue validation as a background task so we return immediately.
        batch_copy = copy.deepcopy(synapse.tweet_batch)
        sent_batch_copy = copy.deepcopy(sent_batch)

        async def _validate_and_release():
            try:
                await self._handle_miner_batch_response(batch_copy, miner_hotkey, sent_batch_copy)
            finally:
                self._validating_tweet_ids -= batch_tids

        task = asyncio.create_task(_validate_and_release())
        self._track_task(task)
        return synapse

    async def forward_telegram_messages(self, synapse: talisman_ai.protocol.TelegramBatch) -> talisman_ai.protocol.TelegramBatch:
        """
        Axon handler for miner push-back of analyzed TelegramBatch results.

        Validates store state synchronously (fast), then queues LLM validation
        as a background task so the axon returns immediately.
        """
        miner_hotkey = synapse.dendrite.hotkey if synapse.dendrite else None
        if not miner_hotkey:
            return synapse

        bt.logging.info(f"[VALIDATION] Received TelegramBatch with {len(synapse.message_batch)} message(s) from miner {miner_hotkey[:12]}..")

        sent_batch: List[TelegramMessageForScoring] = []
        for returned in synapse.message_batch:
            msg_id = str(getattr(returned, "id", ""))
            if not msg_id:
                continue
            if msg_id in self._validating_message_ids:
                bt.logging.info(
                    f"[VALIDATION] Dropping TelegramBatch from {miner_hotkey[:12]}.. "
                    f"message {msg_id} already being validated (replay blocked)"
                )
                return synapse
            try:
                status = self._telegram_store.get_status(msg_id).value
                if status != "Processing":
                    bt.logging.info(
                        f"[VALIDATION] Dropping TelegramBatch from {miner_hotkey[:12]}.. "
                        f"message {msg_id} status={status} (expected Processing)"
                    )
                    return synapse
                if self._telegram_store.get_hotkey(msg_id) != miner_hotkey:
                    bt.logging.info(
                        f"[VALIDATION] Dropping TelegramBatch from {miner_hotkey[:12]}.. "
                        f"message {msg_id} hotkey mismatch"
                    )
                    return synapse
                sent_batch.append(self._telegram_store.get_message(msg_id))
            except Exception:
                return synapse

        if not sent_batch:
            return synapse

        # Lock these message IDs so replays are rejected while validation runs.
        batch_mids = {str(getattr(r, "id", "")) for r in synapse.message_batch if getattr(r, "id", "")}
        self._validating_message_ids.update(batch_mids)

        # Reset the timeout clock — the miner delivered results, we just need
        # time to grade them.
        for returned in synapse.message_batch:
            msg_id = str(getattr(returned, "id", ""))
            if msg_id and msg_id in self._telegram_store._messages:
                self._telegram_store._messages[msg_id].start_time = time.time()

        # Queue validation as a background task so we return immediately.
        batch_copy = copy.deepcopy(synapse.message_batch)
        sent_batch_copy = copy.deepcopy(sent_batch)

        async def _validate_and_release():
            try:
                await self._handle_telegram_miner_batch_response(batch_copy, miner_hotkey, sent_batch_copy)
            finally:
                self._validating_message_ids -= batch_mids

        task = asyncio.create_task(_validate_and_release())
        self._track_task(task)
        return synapse

    async def forward_articles(self, synapse: talisman_ai.protocol.ArticleBatch) -> talisman_ai.protocol.ArticleBatch:
        """
        Axon handler for miner push-back of analyzed ArticleBatch results.
        """
        miner_hotkey = synapse.dendrite.hotkey if synapse.dendrite else None
        if not miner_hotkey:
            return synapse

        bt.logging.info(f"[VALIDATION] Received ArticleBatch with {len(synapse.article_batch)} article(s) from miner {miner_hotkey[:12]}..")

        sent_batch: List[NewsArticleForScoring] = []
        for returned in synapse.article_batch:
            aid = str(getattr(returned, "id", ""))
            if not aid:
                continue
            if aid in self._validating_article_ids:
                bt.logging.info(
                    f"[VALIDATION] Dropping ArticleBatch from {miner_hotkey[:12]}.. "
                    f"article {aid} already being validated (replay blocked)"
                )
                return synapse
            try:
                status = self._article_store.get_status(aid).value
                if status != "Processing":
                    bt.logging.info(
                        f"[VALIDATION] Dropping ArticleBatch from {miner_hotkey[:12]}.. "
                        f"article {aid} status={status} (expected Processing)"
                    )
                    return synapse
                if self._article_store.get_hotkey(aid) != miner_hotkey:
                    bt.logging.info(
                        f"[VALIDATION] Dropping ArticleBatch from {miner_hotkey[:12]}.. "
                        f"article {aid} hotkey mismatch"
                    )
                    return synapse
                sent_batch.append(self._article_store.get_article(aid))
            except Exception:
                return synapse

        if not sent_batch:
            return synapse

        batch_aids = {str(getattr(r, "id", "")) for r in synapse.article_batch if getattr(r, "id", "")}
        self._validating_article_ids.update(batch_aids)

        for returned in synapse.article_batch:
            aid = str(getattr(returned, "id", ""))
            if aid and aid in self._article_store._articles:
                self._article_store._articles[aid].start_time = time.time()

        batch_copy = copy.deepcopy(synapse.article_batch)
        sent_batch_copy = copy.deepcopy(sent_batch)

        async def _validate_and_release():
            try:
                await self._handle_article_miner_batch_response(batch_copy, miner_hotkey, sent_batch_copy)
            finally:
                self._validating_article_ids -= batch_aids

        task = asyncio.create_task(_validate_and_release())
        self._track_task(task)
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
        hotkey = None
        try:
            hotkey = self.metagraph.hotkeys[int(uid)]
        except Exception:
            pass
        if hotkey and not self._cooldown_tracker.try_acquire(hotkey):
            for tweet in miner_batch:
                try:
                    self._tweet_store.reset_to_unprocessed(tweet.id)
                except Exception:
                    pass
            return
        try:
            async with self._miner_dispatch_semaphore:
                await self._process_miner_batch(miner_batch, uid)
        finally:
            if hotkey:
                self._cooldown_tracker.release(hotkey)

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

    async def _handle_article_miner_batch_response(
        self,
        article_batch: List[NewsArticleForScoring],
        miner_hotkey: str,
        sent_batch: List[NewsArticleForScoring],
    ) -> bool:
        if len(article_batch) != len(sent_batch):
            bt.logging.warning(
                f"[VALIDATION] Article batch size mismatch from miner {miner_hotkey} "
                f"sent {len(sent_batch)}, got {len(article_batch)}"
            )
            self._miner_penalty.add_penalty(miner_hotkey, 1)
            for article in sent_batch:
                try:
                    self._article_store.reset_to_unprocessed(article.id)
                except Exception:
                    pass
            return False

        loop = asyncio.get_running_loop()
        is_valid, validation_result = await loop.run_in_executor(
            self._validation_executor,
            validate_miner_article_batch, article_batch, self._news_analyzer, 1
        )
        if not is_valid:
            discrepancies = validation_result.get("discrepancies", [])
            match_rate = validation_result.get("match_rate", 0.0)
            bt.logging.warning(
                f"[VALIDATION] Article batch validation FAILED for miner {miner_hotkey} "
                f"match_rate={match_rate:.1%}, discrepancies={len(discrepancies)}"
            )
            for disc in discrepancies:
                reason = disc.get("reason", "unknown")
                preview = disc.get("article_preview", "")
                if reason == "classification_mismatch":
                    field_results = disc.get("field_results", {})
                    failed_fields = [k for k, v in field_results.items() if not v]
                    miner_vals = disc.get("miner", {})
                    validator_vals = disc.get("validator", {})
                    field_comparisons = []
                    for f in failed_fields:
                        m = miner_vals.get(f, "?")
                        v = validator_vals.get(f, "?")
                        field_comparisons.append(f"{f}(m={m}|v={v})")
                    bt.logging.warning(
                        f"[VALIDATION] Article mismatch for {miner_hotkey}: {', '.join(field_comparisons)} | preview={preview[:100]}"
                    )
                else:
                    bt.logging.warning(f"[VALIDATION] Article rejection for {miner_hotkey}: reason={reason}, preview={preview[:100]}")

            self._miner_penalty.add_penalty(miner_hotkey, 1)
            for article in article_batch:
                try:
                    self._article_store.reset_to_unprocessed(article.id)
                except Exception:
                    pass
            return False

        bt.logging.info(f"[VALIDATION] Article batch validation PASSED for miner {miner_hotkey}")
        for article in article_batch:
            try:
                self._article_store.update_article(article.id, article)
            except Exception:
                self._article_store.add_article(article, article_id=article.id, hotkey=miner_hotkey, set_as_processing=False, overwrite=True)

            try:
                self._article_store.set_processed(article.id)
            except Exception:
                pass

            if not self._article_store.is_rewarded(article.id):
                content_len = len(article.content or "") if article.content else 0
                if content_len >= 2000:
                    weight = 3
                elif content_len >= 500:
                    weight = 2
                else:
                    weight = 1
                self._miner_reward.add_reward(miner_hotkey, weight)
                try:
                    self._article_store.mark_rewarded(article.id)
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
        # Exclude ourselves and miners on cooldown from dispatch selection.
        cooled_hotkeys = self._cooldown_tracker.get_cooled_down_hotkeys()
        cooled_uids = [
            uid for uid in range(self.metagraph.n.item())
            if self.metagraph.hotkeys[uid] in cooled_hotkeys
        ]
        exclude = [int(self.uid)] + cooled_uids
        uids = list(get_random_uids(self, k=len(miner_batches), exclude=exclude))
        tracked, on_cd = self._cooldown_tracker.stats()
        if on_cd > 0:
            available = len(uids)
            bt.logging.debug(f"[COOLDOWN] {on_cd} miners on cooldown, {available} available for dispatch")

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
        cooled_hotkeys = self._cooldown_tracker.get_cooled_down_hotkeys()
        cooled_uids = [
            uid for uid in range(self.metagraph.n.item())
            if self.metagraph.hotkeys[uid] in cooled_hotkeys
        ]
        exclude = [int(self.uid)] + cooled_uids
        uids = list(get_random_uids(self, k=len(miner_batches), exclude=exclude))

        for miner_batch, uid in zip(miner_batches, uids):
            if len(self._pending_miner_tasks) >= self._max_pending_miner_tasks:
                bt.logging.warning(
                    f"[VALIDATION] Too many pending miner dispatch tasks ({len(self._pending_miner_tasks)}); "
                    f"skipping scheduling remaining telegram batches this tick."
                )
                break
            task = asyncio.create_task(self._dispatch_telegram_miner_batch(miner_batch, int(uid)))
            self._track_task(task)

    async def _on_articles(self, articles: List[NewsArticleForScoring]):
        if not articles:
            return

        bt.logging.info(f"[VALIDATION] Processing {len(articles)} articles in batch")
        for article in articles:
            self._article_store.add_article(article, set_as_processing=False, overwrite=False)
        miner_batches = []
        for i in range(0, len(articles), config.MINER_BATCH_SIZE):
            miner_batches.append(articles[i:i + config.MINER_BATCH_SIZE])
        cooled_hotkeys = self._cooldown_tracker.get_cooled_down_hotkeys()
        cooled_uids = [
            uid for uid in range(self.metagraph.n.item())
            if self.metagraph.hotkeys[uid] in cooled_hotkeys
        ]
        exclude = [int(self.uid)] + cooled_uids
        uids = list(get_random_uids(self, k=len(miner_batches), exclude=exclude))

        for miner_batch, uid in zip(miner_batches, uids):
            if len(self._pending_miner_tasks) >= self._max_pending_miner_tasks:
                bt.logging.warning(
                    f"[VALIDATION] Too many pending miner dispatch tasks ({len(self._pending_miner_tasks)}); "
                    f"skipping scheduling remaining article batches this tick."
                )
                break
            task = asyncio.create_task(self._dispatch_article_miner_batch(miner_batch, int(uid)))
            self._track_task(task)

    async def _dispatch_telegram_miner_batch(self, miner_batch: List[TelegramMessageForScoring], uid: int) -> None:
        hotkey = None
        try:
            hotkey = self.metagraph.hotkeys[int(uid)]
        except Exception:
            pass
        if hotkey and not self._cooldown_tracker.try_acquire(hotkey):
            for msg in miner_batch:
                try:
                    self._telegram_store.reset_to_unprocessed(msg.id)
                except Exception:
                    pass
            return
        try:
            async with self._miner_dispatch_semaphore:
                await self._process_telegram_miner_batch(miner_batch, uid)
        finally:
            if hotkey:
                self._cooldown_tracker.release(hotkey)

    async def _dispatch_article_miner_batch(self, miner_batch: List[NewsArticleForScoring], uid: int) -> None:
        hotkey = None
        try:
            hotkey = self.metagraph.hotkeys[int(uid)]
        except Exception:
            pass
        if hotkey and not self._cooldown_tracker.try_acquire(hotkey):
            for article in miner_batch:
                try:
                    self._article_store.reset_to_unprocessed(article.id)
                except Exception:
                    pass
            return
        try:
            async with self._miner_dispatch_semaphore:
                await self._process_article_miner_batch(miner_batch, uid)
        finally:
            if hotkey:
                self._cooldown_tracker.release(hotkey)

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

            if miner_hotkey and miner_hotkey in config.BLACKLISTED_MINER_HOTKEYS:
                bt.logging.info(f"[VALIDATION] Skipping blacklisted miner UID={uid} hotkey={miner_hotkey[:12]}..")
                return None

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
                if miner_hotkey:
                    self._cooldown_tracker.record_failure(miner_hotkey)
                for tweet in miner_batch:
                    try:
                        self._tweet_store.reset_to_unprocessed(tweet.id)
                    except Exception:
                        pass
                return None

            if miner_hotkey:
                self._cooldown_tracker.record_success(miner_hotkey)
            return responses[0]
        except Exception as e:
            bt.logging.error(f"[VALIDATION] Failed to process miner batch: {e}", exc_info=True)
            if miner_hotkey:
                self._cooldown_tracker.record_failure(miner_hotkey)
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

            if miner_hotkey and miner_hotkey in config.BLACKLISTED_MINER_HOTKEYS:
                bt.logging.info(f"[VALIDATION] Skipping blacklisted miner UID={uid} hotkey={miner_hotkey[:12]}.. (telegram)")
                return None

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
                if miner_hotkey:
                    self._cooldown_tracker.record_failure(miner_hotkey)
                for msg in miner_batch:
                    try:
                        self._telegram_store.reset_to_unprocessed(msg.id)
                    except Exception:
                        pass
                return None

            if miner_hotkey:
                self._cooldown_tracker.record_success(miner_hotkey)
            return responses[0]
        except Exception as e:
            bt.logging.error(f"[VALIDATION] Failed to process telegram miner batch: {e}", exc_info=True)
            if miner_hotkey:
                self._cooldown_tracker.record_failure(miner_hotkey)
            for msg in miner_batch:
                try:
                    self._telegram_store.reset_to_unprocessed(msg.id)
                except Exception:
                    pass
            return None

    async def _process_article_miner_batch(
        self,
        miner_batch: List[NewsArticleForScoring],
        uid: int
    ) -> ArticleBatch:
        try:
            miner_hotkey = None
            try:
                miner_hotkey = self.metagraph.hotkeys[int(uid)]
            except Exception:
                miner_hotkey = None

            if miner_hotkey and miner_hotkey in config.BLACKLISTED_MINER_HOTKEYS:
                bt.logging.info(f"[VALIDATION] Skipping blacklisted miner UID={uid} hotkey={miner_hotkey[:12]}.. (articles)")
                return None

            for article in miner_batch:
                self._article_store.add_article(article, article_id=article.id, hotkey=miner_hotkey, set_as_processing=False, overwrite=False)
                try:
                    self._article_store.set_processing(article.id, hotkey=miner_hotkey)
                except Exception:
                    pass

            article_batch = ArticleBatch(
                article_batch=miner_batch
            )
            axon = self.metagraph.axons[uid]
            responses = await self.dendrite.forward(
                axons=[axon],
                synapse=article_batch,
                timeout=float(getattr(config, "MINER_SEND_TIMEOUT", 6.0)),
                deserialize=True
            )
            if not responses[0].dendrite.status_code == 200:
                bt.logging.error(f"[VALIDATION] Failed to process article miner batch: {responses[0].dendrite.status_message}")
                if miner_hotkey:
                    self._cooldown_tracker.record_failure(miner_hotkey)
                for article in miner_batch:
                    try:
                        self._article_store.reset_to_unprocessed(article.id)
                    except Exception:
                        pass
                return None

            if miner_hotkey:
                self._cooldown_tracker.record_success(miner_hotkey)
            return responses[0]
        except Exception as e:
            bt.logging.error(f"[VALIDATION] Failed to process article miner batch: {e}", exc_info=True)
            if miner_hotkey:
                self._cooldown_tracker.record_failure(miner_hotkey)
            for article in miner_batch:
                try:
                    self._article_store.reset_to_unprocessed(article.id)
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
                    asset_id=tweet.analysis.asset_id,
                    asset_symbol=tweet.analysis.asset_symbol,
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
                    asset_id=msg.analysis.asset_id,
                    asset_symbol=msg.analysis.asset_symbol,
                    content_type=msg.analysis.content_type,
                    technical_quality=msg.analysis.technical_quality,
                    market_analysis=msg.analysis.market_analysis,
                    impact_potential=msg.analysis.impact_potential,
                    relevance_confidence=getattr(msg.analysis, "relevance_confidence", None),
                )
            )
        response = await self._validation_client.api_client.submit_completed_telegram_messages(completed_messages)
        return response

    async def _submit_article_batch(self, article_batch: List[NewsArticleForScoring]):
        """Submit an article batch to the API"""
        completed_articles = []
        for article in article_batch:
            if article.analysis is None:
                bt.logging.warning(
                    f"[VALIDATION] Skipping article {article.id} in submission: missing miner analysis"
                )
                continue

            completed_articles.append(
                CompletedNewsArticleSubmission(
                    article_id=article.id,
                    sentiment=article.analysis.sentiment or "neutral",
                    sector_id=article.analysis.sector_id,
                    sector_symbol=article.analysis.sector_symbol,
                    content_type=article.analysis.content_type,
                    technical_quality=article.analysis.technical_quality,
                    market_analysis=article.analysis.market_analysis,
                    impact_potential=article.analysis.impact_potential,
                    relevance_confidence=getattr(article.analysis, "relevance_confidence", None),
                )
            )
        response = await self._validation_client.api_client.submit_completed_articles(completed_articles)
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
                    on_articles=self._on_articles,
                )
            )
            bt.logging.info("[VALIDATION] Started validation client")

        self.save_state()
        
        # Periodically prune old data to prevent memory growth (every 100 steps)
        if self.step % 100 == 0:
            self._prune_stores()
            if hasattr(self._analyzer, '_cache'):
                self._analyzer._cache.log_stats("TWEET_LLM_CACHE")
            if hasattr(self._telegram_analyzer, '_cache'):
                self._telegram_analyzer._cache.log_stats("TELEGRAM_LLM_CACHE")
            if hasattr(self._news_analyzer, '_cache'):
                self._news_analyzer._cache.log_stats("NEWS_LLM_CACHE")
        
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

            # Prune article store: remove submitted articles and old unprocessed ones
            self._article_store.prune_old_articles(max_age_seconds=3600, max_articles=1000)
            self._article_store.save_to_file()

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
            authenticated_hotkey = synapse.dendrite.hotkey
            accepted, reason = self._reward_broadcasts.ingest(
                sender_hotkey=authenticated_hotkey,
                epoch=synapse.epoch,
                seq=synapse.seq,
                uid_points=synapse.uid_points,
            )
            # Persist quickly so we can apply E-2 even after restart.
            self._reward_broadcasts.save()
            if accepted:
                bt.logging.info(
                    f"[BROADCAST] Ingested rewards from {authenticated_hotkey[:12]}.. "
                    f"epoch={synapse.epoch} uids={len(synapse.uid_points)}"
                )
            else:
                bt.logging.debug(
                    f"[BROADCAST] Ignored rewards from {authenticated_hotkey[:12]}.. "
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
            authenticated_hotkey = synapse.dendrite.hotkey
            accepted, reason = self._penalty_broadcasts.ingest(
                sender_hotkey=authenticated_hotkey,
                epoch=synapse.epoch,
                seq=synapse.seq,
                uid_penalties=synapse.uid_penalties,
            )
            # Persist quickly so we can apply E-2 even after restart.
            self._penalty_broadcasts.save()
            if accepted:
                bt.logging.info(
                    f"[PENALTY_BROADCAST] Ingested penalties from {authenticated_hotkey[:12]}.. "
                    f"epoch={synapse.epoch} uids={len(synapse.uid_penalties)}"
                )
            else:
                bt.logging.debug(
                    f"[PENALTY_BROADCAST] Ignored penalties from {authenticated_hotkey[:12]}.. "
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
