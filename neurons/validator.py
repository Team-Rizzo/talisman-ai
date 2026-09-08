# neurons/validator.py
# The MIT License (MIT)
# Copyright © 2023 Team Rizzo

"""
Validator entrypoint.
"""

# Baked-in launch env (set BEFORE bittensor/torch import so operators don't have to set them):
#   BT_NO_PARSE_CLI_ARGS   — bittensor 10.4 ignores CLI args (--netuid/--wallet) without this
#   CUBLAS_WORKSPACE_CONFIG — deterministic cuBLAS for cross-host consensus parity (must precede CUDA init)
import os
os.environ.setdefault("BT_NO_PARSE_CLI_ARGS", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import asyncio
import concurrent.futures
import copy
import gc
import random
import time
from typing import List, Optional, Set

import bittensor as bt
from alpharidge_ai.base.validator import BaseValidatorNeuron
from alpharidge_ai.validator.forward import forward
from alpharidge_ai.validator.validation_client import ValidationClient
from alpharidge_ai.analyzer import setup_analyzer
from alpharidge_ai.analyzer import setup_news_analyzer
from alpharidge_ai.analyzer import setup_article_intelligence_analyzer
import alpharidge_ai.protocol
from alpharidge_ai import config
from alpharidge_ai.analyzer.aspect_sentiment import install_meta_init_guard

# Must precede every model load: accelerate's meta-device patch is process-wide, so
# concurrent loaders otherwise leave a model on meta and every analysis of that article
# fails, which the batch verdict charges to the miner as a rejection.
install_meta_init_guard()

# Bound torch intra-op threads before any model runs: each validation worker otherwise spawns
# torch's default (ncores) OpenMP threads, so the executor oversubscribes the box and validations
# thrash instead of scaling. See TORCH_NUM_THREADS in config.
if int(getattr(config, "TORCH_NUM_THREADS", 0) or 0) > 0:
    try:
        import torch
        torch.set_num_threads(int(config.TORCH_NUM_THREADS))
    except Exception as _e:  # torch absent / already fixed by the runtime
        pass
from alpharidge_ai.utils.api_models import TweetWithAuthor, CompletedTweetSubmission, TelegramMessageForScoring, CompletedTelegramMessageSubmission, TelegramMessageAnalysis, NewsArticleForScoring, CompletedNewsArticleSubmission
from alpharidge_ai.protocol import TweetBatch, TelegramBatch, ArticleBatch
from alpharidge_ai.utils.uids import get_random_uids, get_alive_uids
from alpharidge_ai.utils.liveness import LivenessRoster
from alpharidge_ai.utils.dispatch import coverage_depth_select
from alpharidge_ai.utils.dispatch_metrics import AdaptiveDispatchMetrics
from alpharidge_ai.utils.tweet_store import TweetStore
from alpharidge_ai.utils.telegram_store import TelegramStore
from alpharidge_ai.utils.article_store import ArticleStore
from alpharidge_ai.utils.reward import MinerReward
from alpharidge_ai.utils.penalty import MinerPenalty
from alpharidge_ai.validator.reward_broadcast_store import RewardBroadcastStore
from alpharidge_ai.validator.penalty_broadcast_store import PenaltyBroadcastStore
from alpharidge_ai.protocol import ValidatorRewards
from alpharidge_ai.protocol import ValidatorPenalties
from alpharidge_ai.protocol import ValidatorReputationObs
from alpharidge_ai.analyzer.scoring import validate_miner_batch, validate_miner_telegram_batch, validate_miner_article_batch, validate_miner_article_intelligence_batch, classify_article_batch_failure
from alpharidge_ai.validator.reputation_store import ReputationStore
from alpharidge_ai.validator.reputation import emission as _rep_emission
from alpharidge_ai.validator import emission_params
from alpharidge_ai.validator.profile_client import ProfileClient
from alpharidge_ai.market.ration_store import RationStore
from alpharidge_ai.validator.capacity_controller import CapacityController
from alpharidge_ai.triage import TRIAGE_SCHEMA_VERSION, gazetteer_assets
from alpharidge_ai.models.article_intelligence import SCHEMA_VERSION
from alpharidge_ai.utils.api_models import NewsArticleAnalysisBase
from alpharidge_ai.validator.triage_grader import (
    CanaryPool, TriageConfig, fp_soft_event, grade_batch)

# Triage design constants — one instance, no env var or served key behind it.
TRIAGE_CFG = TriageConfig()
from alpharidge_ai.analyzer import setup_telegram_analyzer
from alpharidge_ai.utils.cooldown import MinerCooldownTracker
from alpharidge_ai.validator.verdict_payload import build_verdict_fields, collect_verdict_meta  # T5: verdict payload
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
        # NOTE: this arg name must not shadow the imported `alpharidge_ai.config` module.
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

        try:
            self._article_intel_analyzer = setup_article_intelligence_analyzer()
            bt.logging.info("[VALIDATION] ArticleIntelligence analyzer initialized")
        except Exception as e:
            bt.logging.warning(f"[VALIDATION] ArticleIntelligence analyzer init failed, V1 only: {e}")
            self._article_intel_analyzer = None

        bt.logging.info("[VALIDATION] Analyzer initialized")

        # Initialize validation client
        self._validation_client = ValidationClient(validator=self, wallet=self.wallet)
        self._validation_task: Optional[asyncio.Task] = None
        self._liveness_task: Optional[asyncio.Task] = None
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

        self._reputation_store = ReputationStore()
        self._reputation_store.load()

        # Mechanism profile: every economic value resolves from here, by block height.
        # Nothing reads it yet; it is fetched and logged so activation can be observed
        # before anything depends on it.
        self._mechanism_profile = ProfileClient()
        self._mechanism_profile.load()

        # Earned rations. Local to this validator and never broadcast. Shadow only:
        # observed and logged beside today's batch size, which still drives dispatch.
        self._ration_store = RationStore()
        self._ration_store.load()

        # Capacity controller. Measures the return and reports a step when the rule
        # arms; capacity itself only ever moves by a published profile.
        self._capacity_controller = CapacityController()
        self._capacity_controller.load()
        self._graded_scorer = None  # lazy; built on first use when scoring is served on
        self._auditor = None        # lazy; built on first use when shadow auditing is on
        self._ration_plan = {}      # this tick's rations; empty leaves dispatch as-is
        # Transient per-item verdict metadata (resource_id -> {miner_signature, nonce,
        # validator_verdict, epoch}); populated during validation, drained at submission.
        self._verdict_meta = {}
        # Display-only penalty attribution buffer for the miner dashboard. DECOUPLED
        # from consensus: rows here are flushed best-effort to /diagnostics/penalty-detail
        # and never enter score_verdict / attestation / Merkle. Bounded so it can't grow
        # without limit if the API is unreachable (appends past the cap are dropped).
        self._penalty_detail_buffer = []
        self._penalty_detail_buffer_max = int(getattr(config, "PENALTY_DETAIL_BUFFER_MAX", 5000))
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
        self._tweet_cooldown = MinerCooldownTracker()
        self._telegram_cooldown = MinerCooldownTracker()
        # Article tracker is the only adaptive one (RFC 2026-06-28); tweet/telegram
        # stay static. Behaves identically to static until ADAPTIVE_DISPATCH_ENABLED.
        self._article_cooldown = MinerCooldownTracker(adaptive=True)
        self._article_cooldown.set_ration_source(self._ration_for)
        # Liveness roster (adaptive dispatch). Populated off the dispatch path;
        # only consulted for selection once ADAPTIVE_DISPATCH_ENABLED is on.
        self._liveness = LivenessRoster()
        # Per-cycle pilot metrics (adaptive dispatch).
        self._adaptive_metrics = AdaptiveDispatchMetrics()

        # Article triage: in-memory canary pool + cached article objects.
        self._canary_pool = CanaryPool(self._triage_cfg())
        self._canary_articles: dict = {}
        self._triage_extractor = None
        self._triage_auditor = None
        self._triage_stage = None
        # Overlap: pending verification assignments (hotkey -> {aid: (article, ts)})
        # and analyses awaiting submission to the variants endpoint.
        self._verification_pending: dict = {}
        self._article_k: dict = {}     # aid -> (assignment count, ts)
        self._article_pay: dict = {}   # aid -> points paid, reported on the verdict
        self._variant_buffer: list = []

    def resync_metagraph(self):
        super().resync_metagraph()
        # resync can fire during base __init__ (e.g. on a fast localnet) before the
        # cooldown trackers are created; skip pruning until they exist.
        if not hasattr(self, "_tweet_cooldown"):
            return
        active = set(self.metagraph.hotkeys)
        for tracker in (self._tweet_cooldown, self._telegram_cooldown, self._article_cooldown):
            tracker.prune(active)

    async def forward_tweets(self, synapse: alpharidge_ai.protocol.TweetBatch) -> alpharidge_ai.protocol.TweetBatch:
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
        sigs = dict(getattr(synapse, "miner_signatures", {}) or {})
        ncs = dict(getattr(synapse, "nonces", {}) or {})

        async def _validate_and_release():
            try:
                await self._handle_miner_batch_response(batch_copy, miner_hotkey, sent_batch_copy, sigs, ncs)
            finally:
                self._validating_tweet_ids -= batch_tids

        task = asyncio.create_task(_validate_and_release())
        self._track_task(task)
        return synapse

    async def forward_telegram_messages(self, synapse: alpharidge_ai.protocol.TelegramBatch) -> alpharidge_ai.protocol.TelegramBatch:
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
        sigs = dict(getattr(synapse, "miner_signatures", {}) or {})
        ncs = dict(getattr(synapse, "nonces", {}) or {})

        async def _validate_and_release():
            try:
                await self._handle_telegram_miner_batch_response(batch_copy, miner_hotkey, sent_batch_copy, sigs, ncs)
            finally:
                self._validating_message_ids -= batch_mids

        task = asyncio.create_task(_validate_and_release())
        self._track_task(task)
        return synapse

    async def forward_articles(self, synapse: alpharidge_ai.protocol.ArticleBatch) -> alpharidge_ai.protocol.ArticleBatch:
        """
        Axon handler for miner push-back of analyzed ArticleBatch results.
        """
        miner_hotkey = synapse.dendrite.hotkey if synapse.dendrite else None
        if not miner_hotkey:
            return synapse

        # A push-back proves the miner is reachable — record liveness off the hot
        # path. In-memory only; never gates anything until the allocator lands.
        self._liveness.mark_seen(miner_hotkey)

        bt.logging.info(f"[VALIDATION] Received ArticleBatch with {len(synapse.article_batch)} article(s) from miner {miner_hotkey[:12]}..")

        # Skip the offending article, don't bin the batch. Each guard below used to `return
        # synapse`, so ONE collision discarded up to 32 articles of honest work and left them
        # PROCESSING — holding that miner's only slot until the lease expired. Measured: 152/818
        # batches (19%) dropped this way. `accepted` stays 1:1 with `sent_batch` so the
        # size-mismatch penalty in _handle_article_miner_batch_response cannot misfire.
        sent_batch: List[NewsArticleForScoring] = []
        accepted: List = []
        v_sent: List[NewsArticleForScoring] = []
        v_accepted: List = []
        skipped = 0
        for returned in synapse.article_batch:
            aid = str(getattr(returned, "id", ""))
            if not aid:
                continue
            # A held lease outranks a verification assignment.
            is_primary = False
            try:
                is_primary = (self._article_store.get_status(aid).value == "Processing"
                              and self._article_store.get_hotkey(aid) == miner_hotkey)
            except Exception:
                is_primary = False
            if is_primary:
                if aid in self._validating_article_ids:
                    skipped += 1
                    continue
                sent_batch.append(self._article_store.get_article(aid))
                accepted.append(returned)
                continue
            v_copy = self._pop_verification(miner_hotkey, aid)
            if v_copy is not None:
                v_sent.append(v_copy)
                v_accepted.append(returned)
            else:
                skipped += 1

        if skipped:
            bt.logging.info(
                f"[VALIDATION] {miner_hotkey[:12]}.. skipped {skipped}/{len(synapse.article_batch)} "
                f"article(s), validating {len(sent_batch)}"
            )
        if v_accepted:
            vb, vs = copy.deepcopy(v_accepted), copy.deepcopy(v_sent)
            vtask = asyncio.create_task(self._handle_article_miner_batch_response(
                vb, miner_hotkey, vs, verification=True))
            self._track_task(vtask)
        if not sent_batch:
            return synapse

        batch_aids = {str(getattr(r, "id", "")) for r in accepted if getattr(r, "id", "")}
        self._validating_article_ids.update(batch_aids)

        # Adaptive dispatch: capture the dispatch→push-back round-trip latency NOW,
        # before the reset below repurposes start_time for the validation clock. This
        # is miner-capacity latency only (it excludes the validator's own analyzer
        # time), which is what the congestion window must measure (RFC Component 2).
        _now = time.time()
        _starts = [
            self._article_store._articles[aid].start_time
            for aid in batch_aids
            if aid in self._article_store._articles and self._article_store._articles[aid].start_time
        ]
        pushback_latency_s = (_now - min(_starts)) if _starts else None

        # Accepted only: a skipped article is owned by someone else (hotkey mismatch) or is
        # already being validated, so re-stamping its clock would extend the wrong lease.
        for returned in accepted:
            aid = str(getattr(returned, "id", ""))
            if aid and aid in self._article_store._articles:
                self._article_store._articles[aid].start_time = time.time()

        batch_copy = copy.deepcopy(accepted)
        sent_batch_copy = copy.deepcopy(sent_batch)
        sigs = dict(getattr(synapse, "miner_signatures", {}) or {})
        ncs = dict(getattr(synapse, "nonces", {}) or {})

        async def _validate_and_release():
            try:
                await self._handle_article_miner_batch_response(batch_copy, miner_hotkey, sent_batch_copy, sigs, ncs, latency_s=pushback_latency_s)
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
        if hotkey and not self._tweet_cooldown.try_acquire(hotkey):
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
                self._tweet_cooldown.release(hotkey)

    # ---- Display-only penalty attribution (decoupled from consensus) ----
    # The following two helpers feed self._penalty_detail_buffer, which is flushed
    # best-effort to /diagnostics/penalty-detail. None of this touches score_verdict,
    # attestation, Merkle, rewards, or penalty counts.

    # Map a discrepancy "reason" to the dashboard "cause". validator_classification_failed
    # is a validator-side failure (not the miner's fault), so it is not surfaced.
    _PENALTY_CAUSE_BY_REASON = {
        "classification_mismatch": "classification_mismatch",
        "missing_miner_classification": "missing_classification",
        "miner_needs_update": "needs_update",
        # July-8 hardening reasons (V2 article path). Without these the rows were dropped
        # here, silently emptying the buffer — the "0 penalty_detail rows since 07-07" bug.
        "article_content_mismatch": "article_content_mismatch",
        "validation_failed": "validation_failed",
        "cloned_embeddings": "cloned_embeddings",
    }

    def _buffer_penalty_detail(self, rows):
        """Append display-only attribution rows, bounded. Never raises."""
        try:
            if not rows:
                return
            buf = self._penalty_detail_buffer
            cap = self._penalty_detail_buffer_max
            for r in rows:
                if len(buf) >= cap:
                    # Drop oldest to stay bounded if the API has been unreachable.
                    del buf[0]
                buf.append(r)
        except Exception as e:
            bt.logging.debug(f"[PENALTY_DETAIL] buffer append skipped: {e}")

    def _penalty_rows_from_discrepancies(self, discrepancies, miner_hotkey, epoch, resource_type):
        """Build display-only penalty_detail rows from a rejected batch's discrepancies."""
        rows = []
        for disc in (discrepancies or []):
            try:
                cause = self._PENALTY_CAUSE_BY_REASON.get(disc.get("reason"))
                if cause is None:
                    continue  # skip validator-side / non-attributable reasons
                rid = disc.get("resource_id")
                if rid is None:
                    continue
                field_results = disc.get("field_results") or {}
                failed_fields = [k for k, ok in field_results.items() if not ok] or None
                preview = (disc.get("post_preview") or disc.get("message_preview")
                           or disc.get("article_preview"))
                # Numeric behind the rejection, when the check produced one: composite
                # (validation_failed, vs 0.65) or summary cosine (content mismatch, vs 0.40).
                score = disc.get("composite_score")
                if score is None:
                    score = disc.get("summary_agreement")
                rows.append({
                    "miner_hotkey": miner_hotkey,
                    "epoch": int(epoch),
                    "resource_type": resource_type,
                    "resource_id": str(rid),
                    "cause": cause,
                    "failed_fields": failed_fields,
                    "miner_values": disc.get("miner"),
                    "validator_values": disc.get("validator"),
                    "post_preview": preview,
                    "score": float(score) if score is not None else None,
                })
            except Exception as e:
                bt.logging.debug(f"[PENALTY_DETAIL] row build skipped: {e}")
                continue
        return rows

    async def _handle_miner_batch_response(
        self,
        tweet_batch: List[TweetWithAuthor],
        miner_hotkey: str,
        sent_batch: List[TweetWithAuthor],
        miner_signatures=None,
        nonces=None,
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
            
            current_epoch = self._miner_reward._get_current_epoch()
            # V6: do NOT write invalid entries into _verdict_meta — invalid items are
            # reset_to_unprocessed and never submitted, so those entries would leak.
            # V2: record display-only attribution instead (decoupled from consensus).
            self._buffer_penalty_detail(
                self._penalty_rows_from_discrepancies(discrepancies, miner_hotkey, current_epoch, "tweet"))
            self._miner_penalty.add_penalty(miner_hotkey, 1)
            for tweet in tweet_batch:
                try:
                    self._tweet_store.reset_to_unprocessed(tweet.id)
                except Exception:
                    pass
            return False

        bt.logging.info(f"[VALIDATION] Batch validation PASSED for miner {miner_hotkey}")
        self._tweet_cooldown.record_success(miner_hotkey)
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

        current_epoch = self._miner_reward._get_current_epoch()
        self._verdict_meta.update(
            collect_verdict_meta(tweet_batch, miner_signatures, nonces, "valid", current_epoch))
        return True

    async def _handle_telegram_miner_batch_response(
        self,
        message_batch: List[TelegramMessageForScoring],
        miner_hotkey: str,
        sent_batch: List[TelegramMessageForScoring],
        miner_signatures=None,
        nonces=None,
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
            
            current_epoch = self._miner_reward._get_current_epoch()
            # V6/V2: see tweet path — drop the leaking invalid _verdict_meta write and
            # record decoupled display-only attribution instead.
            self._buffer_penalty_detail(
                self._penalty_rows_from_discrepancies(discrepancies, miner_hotkey, current_epoch, "telegram"))
            self._miner_penalty.add_penalty(miner_hotkey, 1)
            for msg in message_batch:
                try:
                    self._telegram_store.reset_to_unprocessed(msg.id)
                except Exception:
                    pass
            return False

        bt.logging.info(f"[VALIDATION] Telegram batch validation PASSED for miner {miner_hotkey}")
        self._telegram_cooldown.record_success(miner_hotkey)
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

        current_epoch = self._miner_reward._get_current_epoch()
        self._verdict_meta.update(
            collect_verdict_meta(message_batch, miner_signatures, nonces, "valid", current_epoch))
        return True

    def _get_auditor(self):
        """The keyed auditor, built on first use.

        Built whenever the oracle is live, whatever the local shadow flag says. The
        flag only decides whether to run the audit *before* the oracle goes live; once
        it is live the audit is the reputation source, and letting a local flag suppress
        it would stop reputation moving at all.
        """
        if not (getattr(config, "AUDIT_SHADOW_ENABLED", False) or self._oracle_is_live()):
            return None
        if self._auditor is None:
            from alpharidge_ai.oracle import audit_key
            from alpharidge_ai.oracle.grader import Grader
            from alpharidge_ai.oracle.runner import Auditor
            self._auditor = Auditor(audit_key.load(),
                                    self._mechanism_profile.resolve,
                                    grader=Grader())
            bt.logging.info("[AUDIT] shadow auditing enabled")
        return self._auditor

    def _oracle_is_live(self):
        """Whether audit observations feed reputation, replacing the old scorer's."""
        profile = self._mechanism_profile.resolve(int(self.block))
        return bool(profile.oracle.live) if profile else False

    def _log_audit(self, miner_hotkey, observations, live=False):
        """Log what the audit saw, and record it once the oracle is live.

        Until then this only logs. An observation reaching the store propagates
        fleet-wide within an epoch and cannot be undone.
        """
        for obs in observations or []:
            bt.logging.info(
                f"[AUDIT] hk={miner_hotkey[:12]}.. id={obs.article_id} "
                f"path={obs.path} score={obs.score:.3f} w={obs.weight:.2f} "
                f"model={obs.grader_model} {obs.detail}"
                f"{'' if live else ' (shadow)'}")
        if live and observations:
            self._record_observations(
                miner_hotkey,
                [(o.article_id, float(o.score), float(o.weight)) for o in observations])

    def _ration_for(self, hotkey):
        """This UID's earned ration for the current epoch, or None to leave dispatch
        on the adaptive batch size."""
        profile = self._mechanism_profile.resolve(int(self.block))
        if profile is None or not profile.rations.dispatch:
            return None
        plan = self._ration_plan
        return plan.get(hotkey) if plan else None

    def _refresh_ration_plan(self, hotkeys, supply):
        """Split the articles available this tick across UIDs, once, before slicing."""
        try:
            profile = self._mechanism_profile.resolve(int(self.block))
            if profile is None or not profile.rations.dispatch:
                self._ration_plan = {}
                return
            epoch = int(self._miner_reward._get_current_epoch())
            self._ration_plan = self._ration_store.plan(
                list(hotkeys), epoch=epoch, supply=float(supply), profile=profile)
        except Exception as e:
            bt.logging.debug(f"[RATION] plan failed: {e}")
            self._ration_plan = {}

    def _observe_ration(self, miner_hotkey, article_batch, floor_results):
        """Fold a batch outcome into the ration state and log what it would lease.

        Growth keys on work that cleared the floor, never on what was submitted or on
        how many slots were held. Shadow only: nothing here changes dispatch yet.
        """
        try:
            profile = self._mechanism_profile.resolve(int(self.block))
            if profile is None:
                return
            # Only an explicit pass counts, and only articles the floor actually
            # judged are treated as dispatched. An unknown outcome advances nothing:
            # standing moves on validated work or it does not move.
            judged = [a for a in (article_batch or [])
                      if int(getattr(a, "id", -1)) in floor_results]
            dispatched = len(judged)
            if not dispatched:
                return
            validated = sum(1 for a in judged
                            if floor_results.get(int(a.id)) is True)
            epoch = int(self._miner_reward._get_current_epoch())
            self._ration_store.observe(miner_hotkey, epoch=epoch, validated=validated,
                                       dispatched=dispatched, profile=profile)

            state = self._ration_store.book.states.get(miner_hotkey)
            bt.logging.info(
                f"[RATION] hk={miner_hotkey[:12]}.. validated={validated}/{dispatched} "
                f"ema={state.ema:.2f} batch_size={self._article_cooldown.batch_size(miner_hotkey)} "
                f"(shadow)")
        except Exception as e:
            bt.logging.debug(f"[RATION] observe failed: {e}")

    def _get_graded_scorer(self):
        if self._graded_scorer is None:
            from alpharidge_ai.validator.graded_scorer import GradedScorer
            self._graded_scorer = GradedScorer()
        return self._graded_scorer

    # ------------------------------------------------------------------
    # Article triage (schema v3)

    def _triage_cfg(self) -> TriageConfig:
        return TRIAGE_CFG

    def _get_triage_auditor(self):
        """Lazy audit-LLM client. None when unavailable (no events raised)."""
        if self._triage_auditor is None and self._article_intel_analyzer is not None:
            try:
                from alpharidge_ai.validator.triage_audit import TriageAuditor
                self._triage_auditor = TriageAuditor(
                    self._article_intel_analyzer.client,
                    self._article_intel_analyzer.model,
                    min_confidence=TRIAGE_CFG.audit_min_confidence,
                )
            except Exception as e:
                bt.logging.warning(f"[TRIAGE] auditor unavailable: {e}")
        return self._triage_auditor

    def _llm_relevant_item(self, item: dict):
        auditor = self._get_triage_auditor()
        if auditor is None:
            return None
        return auditor.relevance_verdict(item.get("title") or "", item.get("body") or "")

    def _confirm_clearly_irrelevant(self, aid_flags, sent_by_id) -> set:
        """Keep only 'clearly irrelevant' verdicts the reference TriageStage
        agrees with. Blocking — run in the validation executor."""
        confirmed = set()
        for aid, flag in aid_flags:
            if not flag:
                continue
            aid = int(aid)
            art = sent_by_id.get(aid)
            if art is None:
                continue
            rec, _, _ = self._get_triage_stage().evaluate(
                art.title or "", art.content or "")
            if rec["label"] == "irrelevant":
                confirmed.add(aid)
        return confirmed

    def _get_triage_stage(self):
        if self._triage_stage is None:
            from alpharidge_ai.analyzer.triage_stage import TriageStage
            if self._triage_extractor is None:
                from alpharidge_ai.analyzer.asset_extractor import AssetExtractor
                self._triage_extractor = AssetExtractor()
            self._triage_stage = TriageStage(self._triage_extractor)
        return self._triage_stage

    def _register_verification(self, hotkey: str, articles) -> None:
        now = time.time()
        slot = self._verification_pending.setdefault(hotkey, {})
        for a in articles:
            slot[str(a.id)] = (a, now)

    def _pop_verification(self, hotkey: str, aid: str):
        entry = self._verification_pending.get(hotkey, {}).pop(aid, None)
        return entry[0] if entry else None

    def _prune_verification(self) -> None:
        cutoff = time.time() - TRIAGE_CFG.verification_ttl_s
        for hk in list(self._verification_pending):
            slot = {k: v for k, v in self._verification_pending[hk].items()
                    if v[1] >= cutoff}
            if slot:
                self._verification_pending[hk] = slot
            else:
                del self._verification_pending[hk]
        self._article_k = {a: v for a, v in self._article_k.items()
                           if v[1] >= cutoff}

    def _k_for(self, article_id) -> int:
        entry = self._article_k.get(str(article_id))
        if entry:
            return max(1, entry[0])
        return (TRIAGE_CFG.overlap_k
                if getattr(config, "TRIAGE_ENFORCED", False) else 1)

    def _attribute_pay(self, per_article: dict, payout: int, record: bool = True) -> dict:
        """Scale per-article pay so the verdicts sum to the integer payout."""
        if not per_article:
            return {}
        base = sum(per_article.values())
        scale = (payout / base) if base > 0 else 0.0
        out = {aid: round(pts * scale, 4) for aid, pts in per_article.items()}
        if record:
            if len(self._article_pay) > 50_000:
                for k in list(self._article_pay)[:10_000]:
                    self._article_pay.pop(k, None)
            for aid, pts in out.items():
                self._article_pay[str(aid)] = pts
        return out

    def _buffer_variants(self, miner_hotkey: str, articles, pay=None,
                         miner_signatures=None, nonces=None, epoch=None) -> None:
        pay = pay or {}
        sigs = miner_signatures or {}
        ncs = nonces or {}
        for a in articles:
            analysis = getattr(a, "analysis", None)
            data = getattr(analysis, "analysis_data", None)
            if not isinstance(data, dict) or len(str(data)) > 300_000:
                continue
            if len(self._variant_buffer) >= 5000:
                break
            entry = {
                "article_id": int(a.id),
                "miner_hotkey": miner_hotkey,
                "analysis_data": data,
            }
            rid = str(a.id)
            if sigs.get(rid) and ncs.get(rid) and epoch is not None:
                entry.update(build_verdict_fields(
                    miner_hotkey=miner_hotkey, miner_signature=sigs[rid],
                    nonce=ncs[rid], analysis=analysis, validator_verdict="valid",
                    points_awarded=float(pay.get(int(a.id), 0.0)), epoch=int(epoch)))
            self._variant_buffer.append(entry)

    def _stage_label_item(self, item: dict) -> str:
        rec, _, _ = self._get_triage_stage().evaluate(
            item.get("title") or "", item.get("body") or "")
        return rec["label"]

    def _det_relevant_item(self, item: dict) -> bool:
        """Deterministic R1 gazetteer check (shared with the miner path)."""
        if self._triage_extractor is None:
            from alpharidge_ai.analyzer.asset_extractor import AssetExtractor
            self._triage_extractor = AssetExtractor()
        return bool(gazetteer_assets(
            self._triage_extractor, item.get("title") or "", item.get("body") or ""))

    def _feed_pos_canaries(self, articles, budget: int = 8, target: int = 50):
        """Top up positive canaries from the gazetteer. Bounded per tick."""
        if self._canary_pool.size("pos") >= target:
            return
        for article in articles[:budget]:
            item = {"title": article.title, "body": article.content or ""}
            if self._det_relevant_item(item):
                self._canary_pool.add(int(article.id), "pos", deterministic=True)
                self._canary_articles[int(article.id)] = article

    def _refresh_canaries(self, articles):
        """Top up both canary pools and expire stale entries. Blocking —
        run off the event loop."""
        try:
            self._feed_pos_canaries(articles)
            self._mint_neg_canaries(articles)
            self._canary_pool.prune()
            live = self._canary_pool.ids()
            self._canary_articles = {
                k: v for k, v in self._canary_articles.items() if k in live}
        except Exception as e:
            bt.logging.warning(f"[TRIAGE] canary refresh failed: {e}")

    def _mint_neg_canaries(self, articles):
        """Mint negative canaries: the TriageStage and two audit-LLM passes
        must all concur. Bounded per tick."""
        budget, target = TRIAGE_CFG.neg_mint_budget, TRIAGE_CFG.neg_pool_target
        if self._canary_pool.size("neg") >= target:
            return
        auditor = self._get_triage_auditor()
        if auditor is None:
            return
        checked = 0
        for article in articles:
            if checked >= budget or self._canary_pool.size("neg") >= target:
                break
            aid = int(article.id)
            if aid in self._canary_pool.ids():
                continue
            rec, _, _ = self._get_triage_stage().evaluate(
                article.title or "", article.content or "")
            if rec["label"] != "irrelevant":
                continue
            checked += 1
            if (auditor.relevance_verdict(article.title, article.content or "",
                                          framing="strict") is False
                    and auditor.relevance_verdict(article.title, article.content or "",
                                                  framing="editorial") is False):
                self._canary_pool.add(aid, "neg", deterministic=False)
                self._canary_articles[aid] = article.model_copy(
                    update={"analysis": None})
                bt.logging.info(f"[TRIAGE] minted negative canary {aid}")

    def _inject_canaries(self, miner_batch, rng, charge: int = 1) -> dict:
        """Swap canaries into a dispatch batch (in place). Returns the injected
        {article_id: (kind, deterministic)} labels for later grading."""
        injected = {}
        batch_ids = {int(a.id) for a in miner_batch}
        for kind, rate in (("pos", TRIAGE_CFG.canary_pos_rate),
                           ("neg", TRIAGE_CFG.canary_neg_rate)):
            if len(miner_batch) < 3 or rng.random() >= rate:
                continue
            available = set(self._canary_articles) - batch_ids
            aid = self._canary_pool.draw(kind, rng, available=available, charge=charge)
            if aid is None:
                continue
            free = [i for i, a in enumerate(miner_batch) if int(a.id) not in injected]
            slot = rng.choice(free)
            miner_batch[slot] = self._canary_articles[aid]
            injected[aid] = self._canary_pool.label_of(aid)
            batch_ids.add(aid)
        return injected

    def _grade_triage(self, article_batch, sent_batch, miner_hotkey):
        """Grade a returned batch. Article text comes from our sent_batch
        copies; only analysis_data is taken from the response."""
        reference_by_id = {int(a.id): a for a in sent_batch}
        items = []
        for a in article_batch:
            ref = reference_by_id.get(int(a.id))
            if ref is None:
                continue   # unknown id; the size/identity guards handle it
            items.append({
                "article_id": int(a.id),
                "title": ref.title,
                "body": ref.content or "",
                "analysis_data": getattr(a.analysis, "analysis_data", None) if a.analysis else None,
            })
        canary_labels = {}
        for it in items:
            label = self._canary_pool.label_of(it["article_id"])
            if label is not None:
                canary_labels[it["article_id"]] = label
        res = grade_batch(
            items, canary_labels, self._det_relevant_item, self._llm_relevant_item,
            self._stage_label_item,
            random.Random(), self._triage_cfg(),
            enforced=bool(getattr(config, "TRIAGE_ENFORCED", False)))
        if res.events:
            bt.logging.info(
                f"[TRIAGE] hk={miner_hotkey} events="
                f"{[(e.kind, e.code, e.article_id) for e in res.events]}")
        return res

    def _record_triage_observations(self, miner_hotkey, triage_res, article_batch,
                                    graded_observations=(), allow_clean=True):
        """Merge triage and quality observations before recording (the store
        keeps one observation per (article_id, sender); worst score wins)."""
        clean_id = (int(article_batch[0].id)
                    if article_batch and allow_clean else None)
        merged = {}
        for aid, score, weight in (list(graded_observations)
                                   + triage_res.observations(self._triage_cfg(), clean_id)):
            aid = int(aid)
            prev = merged.get(aid)
            merged[aid] = ((min(prev[0], float(score)), max(prev[1], float(weight)))
                           if prev else (float(score), float(weight)))
        self._record_observations(
            miner_hotkey, [(aid, s, w) for aid, (s, w) in sorted(merged.items())])

    @staticmethod
    def _has_full_analysis(article) -> bool:
        """True when the response carries an analysis payload."""
        data = getattr(getattr(article, "analysis", None), "analysis_data", None)
        return isinstance(data, dict) and bool(data.get("event_fingerprint"))

    @staticmethod
    def _triage_only_analysis(article):
        """Minimal stored analysis for a retired article: the miner's triage
        record and proof-of-read only."""
        data = getattr(getattr(article, "analysis", None), "analysis_data", None)
        data = data if isinstance(data, dict) else {}
        return NewsArticleAnalysisBase(
            sentiment="neutral",
            analysisData={
                "schema_version": SCHEMA_VERSION,
                "triage_schema_version": TRIAGE_SCHEMA_VERSION,
                "triage": data.get("triage"),
                "proof_of_read": data.get("proof_of_read"),
            },
        )

    def _apply_triage_outcome(self, article_batch, miner_hotkey, triage_res, fp_ids,
                              sent_by_id=None, full_push=True):
        """Store + reward outcomes for a passing v3 batch. Relevant ->
        processed; uncontradicted irrelevant -> processed with a rebuilt
        triage-only analysis; borderline/contradicted -> back to the pool;
        canaries -> graded only, never stored."""
        relevant_ids = set(triage_res.relevant_ids)
        # Flagged-valuable borderline is kept and paid like relevant;
        # flagged-discard is stored triage-only and unpaid.
        keep_ids = relevant_ids | set(triage_res.borderline_valuable_ids)
        discard_ids = set(triage_res.borderline_discard_ids)
        retire_ids = set(triage_res.retire_candidate_ids) | discard_ids
        canary_ids = set(triage_res.canary_ids)
        # Each article's pot is split across the assignees recorded at
        # dispatch.
        total_pay = sum(TRIAGE_CFG.fee_points / self._k_for(a.id)
                        for a in article_batch)
        rel_mult = TRIAGE_CFG.rel_point_mult
        per_article: dict = {}

        for article in article_batch:
            aid = int(article.id)
            if aid in canary_ids:
                # Graded, never stored; analysed positives pay per miner.
                if (aid in relevant_ids and aid not in fp_ids
                        and self._has_full_analysis(article)):
                    content_len = len(article.content or "")
                    weight = 3 if content_len >= 2000 else (2 if content_len >= 500 else 1)
                    total_pay += rel_mult * weight / self._k_for(aid)
                try:
                    self._article_store.reset_to_unprocessed(article.id)
                except Exception:
                    pass
                continue
            if aid in keep_ids or aid in retire_ids:
                per_article[aid] = TRIAGE_CFG.fee_points / self._k_for(aid)
                stored = article
                if aid in retire_ids:
                    # Store our copy with a rebuilt triage-only analysis.
                    base = (sent_by_id or {}).get(aid) or article
                    stored = base.model_copy(
                        update={"analysis": self._triage_only_analysis(article)})
                try:
                    self._article_store.update_article(article.id, stored)
                except Exception:
                    self._article_store.add_article(
                        stored, article_id=article.id, hotkey=miner_hotkey,
                        set_as_processing=False, overwrite=True)
                try:
                    self._article_store.set_processed(article.id)
                except Exception:
                    pass
                if (aid in keep_ids and aid not in fp_ids
                        and not self._article_store.is_rewarded(article.id)):
                    content_len = len(article.content or "")
                    weight = 3 if content_len >= 2000 else (2 if content_len >= 500 else 1)
                    total_pay += rel_mult * weight / self._k_for(aid)
                    per_article[aid] += rel_mult * weight / self._k_for(aid)
                    try:
                        self._article_store.mark_rewarded(article.id)
                    except Exception:
                        pass
            else:
                # Borderline or contradicted-irrelevant: another miner gets it.
                try:
                    self._article_store.reset_to_unprocessed(article.id)
                except Exception:
                    pass
        # Fee floor applies only to a push covering the full lease.
        payout = int(round(total_pay))
        if full_push and article_batch:
            payout = max(1, payout)
        if payout > 0:
            self._miner_reward.add_reward(miner_hotkey, payout)
        self._attribute_pay(per_article, payout)

    def _apply_verification_outcome(self, article_batch, miner_hotkey, triage_res, fp_ids,
                                    miner_signatures=None, nonces=None, epoch=None):
        """Pay a verification response at the split rate and keep its analyses
        as variants. No store interaction — the primary owns the article."""
        keep_ids = set(triage_res.relevant_ids) | set(triage_res.borderline_valuable_ids)
        discard_ids = set(triage_res.borderline_discard_ids)
        canary_ids = set(triage_res.canary_ids)
        # No fee floor in this lane.
        total_pay = sum(TRIAGE_CFG.fee_points / self._k_for(a.id)
                        for a in article_batch)
        variants = []
        per_article: dict = {}
        for article in article_batch:
            aid = int(article.id)
            if aid in canary_ids or aid in fp_ids:
                continue
            k = self._k_for(aid)
            per_article[aid] = TRIAGE_CFG.fee_points / k
            content_len = len(article.content or "")
            weight = 3 if content_len >= 2000 else (2 if content_len >= 500 else 1)
            if aid in keep_ids and self._has_full_analysis(article):
                total_pay += TRIAGE_CFG.rel_point_mult * weight / k
                per_article[aid] += TRIAGE_CFG.rel_point_mult * weight / k
            variants.append(article)
        payout = int(round(total_pay))
        if payout > 0:
            self._miner_reward.add_reward(miner_hotkey, payout)
        self._buffer_variants(miner_hotkey, variants,
                              self._attribute_pay(per_article, payout, record=False),
                              miner_signatures, nonces, epoch)

    def _record_observations(self, target_hotkey, observations):
        if not observations:
            return
        try:
            epoch = self._miner_reward._get_current_epoch()
            self_hk = self.wallet.hotkey.ss58_address
            for aid, g, w in observations:
                self._reputation_store.record_local(
                    epoch, self_hk, target_hotkey, int(aid), float(g), float(w))
        except Exception as e:
            bt.logging.warning(f"[REPUTATION] record failed: {e}")

    async def _handle_article_miner_batch_response(
        self,
        article_batch: List[NewsArticleForScoring],
        miner_hotkey: str,
        sent_batch: List[NewsArticleForScoring],
        miner_signatures=None,
        nonces=None,
        latency_s: float = None,
        verification: bool = False,
    ) -> bool:
        # Verification copies are graded and paid but hold no lease: never
        # reset/store articles and never touch dispatch windows for them.
        adaptive = (getattr(config, "ADAPTIVE_DISPATCH_ENABLED", False)
                    and not verification)
        if latency_s is not None:
            bt.logging.info(f"[LATPROBE] hotkey={miner_hotkey} latency_s={latency_s:.2f} n={len(sent_batch)}")
            self._article_cooldown.record_latency(miner_hotkey, latency_s)  # display-only telemetry
        if len(article_batch) != len(sent_batch):
            bt.logging.warning(
                f"[VALIDATION] Article batch size mismatch from miner {miner_hotkey} "
                f"sent {len(sent_batch)}, got {len(article_batch)}"
            )
            self._miner_penalty.add_penalty(miner_hotkey, 1)
            if adaptive:
                self._article_cooldown.record_invalid(miner_hotkey)
                self._article_cooldown.record_validation_fail(miner_hotkey, "size_mismatch")
                self._adaptive_metrics.incr("invalid")
            self._article_cooldown.record_batch_shrink(miner_hotkey)
            if not verification:
                for article in sent_batch:
                    try:
                        self._article_store.reset_to_unprocessed(article.id)
                    except Exception:
                        pass
            return False

        loop = asyncio.get_running_loop()

        # A partial push (fewer articles than the lease was dispatched at)
        # is graded and paid per-article but earns no fee floor, no clean
        # observation, and no window growth.
        full_push = True
        if not verification:
            try:
                item = self._article_store._articles.get(str(article_batch[0].id))
                ds = getattr(item, "dispatch_batch_size", None)
                if ds:
                    full_push = len(article_batch) >= int(ds)
            except Exception:
                pass

        # Triage grading (schema v3). Always on.
        try:
            triage_res = await loop.run_in_executor(
                self._validation_executor, self._grade_triage,
                article_batch, sent_batch, miner_hotkey)
        except Exception as e:
            # Never strand a batch on a grading fault.
            bt.logging.warning(f"[TRIAGE] grading failed for {miner_hotkey}: {e}")
            triage_res = None
        # None = grading threw; grace = pre-triage batch before the cutover.
        # Both fall back to legacy grading and pay.
        triage_active = (triage_res is not None
                         and not getattr(triage_res, "grace", False))
        if verification and not triage_active:
            # A verification response is graded or dropped — never the
            # legacy path.
            return False
        if triage_active and triage_res.proof_failures:
            bt.logging.warning(
                f"[TRIAGE] proof-of-read FAILED for miner {miner_hotkey} "
                f"articles={triage_res.proof_failures}")
            self._record_triage_observations(miner_hotkey, triage_res, article_batch)
            self._miner_penalty.add_penalty(miner_hotkey, 1)
            if adaptive:
                self._article_cooldown.record_invalid(miner_hotkey)
                self._article_cooldown.record_validation_fail(miner_hotkey, "triage_proof")
                self._adaptive_metrics.incr("invalid")
            if not verification:
                self._article_cooldown.record_batch_shrink(miner_hotkey)
            if not verification:
                for article in sent_batch:
                    try:
                        self._article_store.reset_to_unprocessed(article.id)
                    except Exception:
                        pass
            return False

        # Deep validation covers claimed-relevant articles only.
        track_batch = article_batch
        if triage_active:
            deep_ids = set(triage_res.relevant_ids) | set(
                triage_res.borderline_valuable_ids)
            track_batch = [a for a in article_batch if int(a.id) in deep_ids]

        # Resolved once, before the branches: the triage path below reads it too, and
        # binding it inside one branch would leave it undefined on the others.
        _profile = self._mechanism_profile.resolve(int(self.block))
        oracle_live = self._oracle_is_live()
        # The old scorer stands down only when something is actually replacing it. A
        # published flip with no working auditor must not leave reputation unfed.
        try:
            audit_supersedes = oracle_live and self._get_auditor() is not None
        except Exception as e:
            # Building it can fail on a missing key or a bad dependency. The point of
            # the fallback is that reputation keeps moving when it does.
            bt.logging.error(f"[AUDIT] could not build the auditor: {e}")
            audit_supersedes = False
        if oracle_live and not audit_supersedes:
            bt.logging.error(
                "[AUDIT] oracle.live is published but no auditor could be built; "
                "keeping the existing scorer so reputation still moves")

        # Try V2 validation if miner submitted analysis_data
        has_v2 = any(
            getattr(a.analysis, "analysis_data", None)
            for a in track_batch if a.analysis
        )
        if triage_active and not track_batch:
            # Triage-only batch: nothing claimed relevant, nothing to deep-validate.
            is_valid, validation_result = True, {}
        elif has_v2 and self._article_intel_analyzer is not None:
            sample_size = int(getattr(config, "VALIDATION_SAMPLE_SIZE", 1))
            gscorer = self._get_graded_scorer() if config.REPUTATION_SCORING_ENABLED else None
            reference_by_id = {str(a.id): a for a in sent_batch}
            is_valid, validation_result = await loop.run_in_executor(
                self._validation_executor,
                validate_miner_article_intelligence_batch,
                track_batch, self._article_intel_analyzer, sample_size, None, gscorer,
                reference_by_id, miner_hotkey,
                (self._auditor if audit_supersedes or self._auditor else None),
                int(self.block),
                (_profile.oracle.schema_cutover_block if _profile else 0),
            )
            self._log_audit(miner_hotkey,
                            (validation_result or {}).get("audit_observations"),
                            live=oracle_live)
            if gscorer is not None:
                # Merged with triage grades below when grading succeeded
                # (first-obs-wins dedup in the reputation store).
                if not triage_active and not audit_supersedes:
                    # Replaced, not supplemented: the audit measures the same thing
                    # against the article rather than against our own re-run.
                    self._record_observations(
                        miner_hotkey, (validation_result or {}).get("observations") or [])
                # Faithfulness cooldown update (min over sampled articles).
                faiths = (validation_result or {}).get("faithfulness_scores") or []
                if faiths:
                    min_faith = min(faiths)
                    bt.logging.info(
                        f"[FAITHFULNESS] hk={miner_hotkey} min={min_faith:.3f} "
                        f"scores={[round(f, 3) for f in faiths]} passed={is_valid}")
                    if not verification:
                        self._article_cooldown.record_faithfulness(miner_hotkey, min_faith)
        else:
            is_valid, validation_result = await loop.run_in_executor(
                self._validation_executor,
                validate_miner_article_batch, track_batch, self._news_analyzer, 1,
            )

        # Reference-relevance verdicts feed FP events and the negative pool.
        fp_ids = set()
        if triage_active:
            sent_by_id = {int(a.id): a for a in sent_batch}
            confirmed_irrelevant = await loop.run_in_executor(
                self._validation_executor, self._confirm_clearly_irrelevant,
                (validation_result or {}).get("reference_irrelevant", []),
                sent_by_id)
            for aid in sorted(confirmed_irrelevant):
                fp_ids.add(aid)
                triage_res.events.append(fp_soft_event(aid))
                if (self._canary_pool.size("neg") < TRIAGE_CFG.neg_pool_target
                        and aid in sent_by_id):
                    self._canary_pool.add(aid, "neg", deterministic=False)
                    self._canary_articles[aid] = sent_by_id[aid].model_copy(
                        update={"analysis": None})
            self._record_triage_observations(
                miner_hotkey, triage_res, article_batch,
                [] if audit_supersedes else ((validation_result or {}).get("observations") or []),
                allow_clean=full_push)

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

            current_epoch = self._miner_reward._get_current_epoch()
            missing_split = adaptive and getattr(config, "ADAPTIVE_MISSING_ANALYSIS_SPLIT_ENABLED", False)
            failure_class = classify_article_batch_failure(discrepancies)
            if verification:
                # Verification lane: penalty only on integrity failures;
                # no dispatch-window state moves.
                if failure_class == "integrity":
                    self._miner_penalty.add_penalty(miner_hotkey, 1)
                return False
            # Consecutive validation-fail park — skip validator-side (our analyzer's fault).
            if adaptive and failure_class != "validator_side":
                self._article_cooldown.record_validation_fail(miner_hotkey, failure_class)

            if missing_split and failure_class != "integrity":
                if failure_class == "validator_side":
                    # Our own analyzer failed — the miner did nothing wrong and isn't even
                    # overloaded: no penalty, no window change, just retry.
                    bt.logging.info(
                        f"[VALIDATION] validator-side analysis failure for {miner_hotkey} — no penalty, retrying")
                else:
                    # Capacity: back off the dispatch window (so we stop dumping work it
                    # can't drain), but no integrity penalty / no emission-gate hit.
                    bt.logging.info(
                        f"[VALIDATION] incomplete analysis from {miner_hotkey} (capacity, not integrity) "
                        f"— backing off window, no penalty")
                    self._article_cooldown.record_invalid(miner_hotkey)
                    self._adaptive_metrics.incr("incomplete")
                    self._article_cooldown.record_batch_shrink(miner_hotkey)
            else:
                # Genuine integrity failure (or the split is disabled) — unchanged:
                # display-only attribution + integrity penalty + window shrink.
                self._buffer_penalty_detail(
                    self._penalty_rows_from_discrepancies(discrepancies, miner_hotkey, current_epoch, "article"))
                self._miner_penalty.add_penalty(miner_hotkey, 1)
                if adaptive:
                    self._article_cooldown.record_invalid(miner_hotkey)
                    self._adaptive_metrics.incr("invalid")
                self._article_cooldown.record_batch_shrink(miner_hotkey)

            if not verification:
                for article in article_batch:
                    try:
                        self._article_store.reset_to_unprocessed(article.id)
                    except Exception:
                        pass
            return False

        bt.logging.info(f"[VALIDATION] Article batch validation PASSED for miner {miner_hotkey}")
        if not verification:
            self._article_cooldown.record_success(miner_hotkey)
            self._article_cooldown.record_validation_pass(miner_hotkey)  # clears the fail streak
        # Adaptive: grow the window if the round-trip was comfortably on-time, else
        # freeze (objective 8 — find capacity without ramping into a timeout).
        # A batch where nothing was claimed relevant proves triage throughput,
        # not analysis capacity, and returns almost instantly — ramping on it
        # would let an all-irrelevant miner win dispatch share it can't use.
        deep_validated = not (triage_active and not track_batch)
        if adaptive and deep_validated:
            self._article_cooldown.record_timely_valid(miner_hotkey, latency_s)
            self._adaptive_metrics.incr("valid")
            self._adaptive_metrics.mark_scored(miner_hotkey)
        self._article_cooldown.record_batch_valid(miner_hotkey, latency_s)
        if triage_active and verification:
            self._apply_verification_outcome(
                article_batch, miner_hotkey, triage_res, fp_ids,
                miner_signatures, nonces, self._miner_reward._get_current_epoch())
        elif triage_active:
            self._apply_triage_outcome(article_batch, miner_hotkey, triage_res, fp_ids,
                                       {int(a.id): a for a in sent_batch},
                                       full_push=full_push)
        else:
            floor_results = (validation_result or {}).get("floor_results") or {}
            floor_gating = bool(_profile.settlement.floor_gating) if _profile else False
            floor_failed = [aid for aid, ok in floor_results.items() if not ok]
            if floor_failed:
                bt.logging.info(
                    f"[FLOOR] hk={miner_hotkey} failed={len(floor_failed)}/"
                    f"{len(floor_results)} gating={'on' if floor_gating else 'shadow'}")
            self._observe_ration(miner_hotkey, article_batch, floor_results)

            for article in article_batch:
                if self._canary_pool.label_of(int(article.id)) is not None:
                    # Canaries are graded only, in every lane and era.
                    try:
                        self._article_store.reset_to_unprocessed(article.id)
                    except Exception:
                        pass
                    continue
                try:
                    self._article_store.update_article(article.id, article)
                except Exception:
                    self._article_store.add_article(article, article_id=article.id, hotkey=miner_hotkey, set_as_processing=False, overwrite=True)

                try:
                    self._article_store.set_processed(article.id)
                except Exception:
                    pass

                if not self._article_store.is_rewarded(article.id):
                    if floor_gating and floor_results.get(int(article.id)) is False:
                        # Cleared validation as part of the batch, but this article did
                        # not clear the floor on its own, so it earns nothing.
                        continue
                    content_len = len(article.content or "") if article.content else 0
                    if content_len >= 2000:
                        weight = 3
                    elif content_len >= 500:
                        weight = 2
                    else:
                        weight = 1
                    self._miner_reward.add_reward(miner_hotkey, weight)
                    self._article_pay[str(article.id)] = float(weight)
                    try:
                        self._article_store.mark_rewarded(article.id)
                    except Exception:
                        pass

        if not verification:
            current_epoch = self._miner_reward._get_current_epoch()
            self._verdict_meta.update(
                collect_verdict_meta(article_batch, miner_signatures, nonces, "valid", current_epoch))
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
        cooled_hotkeys = self._tweet_cooldown.get_cooled_down_hotkeys()
        cooled_uids = [
            uid for uid in range(self.metagraph.n.item())
            if self.metagraph.hotkeys[uid] in cooled_hotkeys
        ]
        exclude = [int(self.uid)] + cooled_uids
        uids = list(get_random_uids(self, k=len(miner_batches), exclude=exclude))
        tracked, on_cd = self._tweet_cooldown.stats()
        if on_cd > 0:
            available = len(uids)
            bt.logging.debug(f"[COOLDOWN/tweet] {on_cd} miners on cooldown, {available} available for dispatch")

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
        cooled_hotkeys = self._telegram_cooldown.get_cooled_down_hotkeys()
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
        # Split what is available across UIDs once, before anything is sliced.
        self._refresh_ration_plan(list(self.metagraph.hotkeys), len(articles))
        cooled_hotkeys = self._article_cooldown.get_cooled_down_hotkeys()
        cooled_uids = [
            uid for uid in range(self.metagraph.n.item())
            if self.metagraph.hotkeys[uid] in cooled_hotkeys
        ]
        exclude = [int(self.uid)] + cooled_uids

        if getattr(config, "ADAPTIVE_BATCH_SIZE_ENABLED", False):
            # Draw each selected miner's slice at its per-miner batch_size from the pool, in order.
            base = max(1, int(getattr(config, "MINER_BATCH_SIZE", 12)))
            n_slots = max(1, -(-len(articles) // base))  # upper bound on batches this tick
            ordered = self._select_article_targets([None] * n_slots, exclude)
            targets = []
            cursor = 0
            for uid, _placeholder in ordered:
                if cursor >= len(articles):
                    break
                try:
                    hk = self.metagraph.hotkeys[int(uid)]
                except Exception:
                    continue
                miner_batch = articles[cursor:cursor + self._article_cooldown.batch_size(hk)]
                if not miner_batch:
                    break
                cursor += len(miner_batch)
                targets.append((int(uid), miner_batch))
        else:
            miner_batches = []
            for i in range(0, len(articles), config.MINER_BATCH_SIZE):
                miner_batches.append(articles[i:i + config.MINER_BATCH_SIZE])
            targets = self._select_article_targets(miner_batches, exclude)

        # Blocking (gazetteer-bound): run in the validation executor.
        await asyncio.get_running_loop().run_in_executor(
            self._validation_executor, self._refresh_canaries, articles)
        canary_rng = random.Random()

        adaptive = getattr(config, "ADAPTIVE_DISPATCH_ENABLED", False)
        epoch = self._current_epoch() if adaptive else 0
        # Overlap activates with enforcement: each batch also goes to k-1
        # verifier miners, graded and paid at the split rate, never stored.
        overlap_k = (TRIAGE_CFG.overlap_k
                     if getattr(config, "TRIAGE_ENFORCED", False) else 1)
        self._prune_verification()
        exclude_set = set(exclude)
        # Verifiers must be alive (push-back seen recently) and not
        # blacklisted.
        eligible = []
        for u in range(self.metagraph.n.item()):
            if u in exclude_set:
                continue
            try:
                hk = self.metagraph.hotkeys[u]
            except Exception:
                continue
            if hk in config.BLACKLISTED_MINER_HOTKEYS:
                continue
            if not self._liveness.is_alive(hk):
                continue
            eligible.append(u)
        for uid, miner_batch in targets:
            if len(self._pending_miner_tasks) >= self._max_pending_miner_tasks:
                bt.logging.warning(
                    f"[VALIDATION] Too many pending miner dispatch tasks ({len(self._pending_miner_tasks)}); "
                    f"skipping scheduling remaining article batches this tick."
                )
                break
            miner_batch = list(miner_batch)
            self._inject_canaries(miner_batch, canary_rng, charge=overlap_k)
            pool = [u for u in eligible if u != int(uid)]
            n_verify = min(overlap_k - 1, len(pool)) if overlap_k > 1 else 0
            # Record each article's actual assignment count for pay.
            k_actual = 1 + n_verify
            now = time.time()
            for a in miner_batch:
                self._article_k[str(a.id)] = (k_actual, now)
            task = asyncio.create_task(self._dispatch_article_miner_batch(miner_batch, int(uid)))
            self._track_task(task)
            for vuid in (canary_rng.sample(pool, n_verify) if n_verify else []):
                vbatch = [a.model_copy(update={"analysis": None}) for a in miner_batch]
                vtask = asyncio.create_task(
                    self._dispatch_verification_batch(vbatch, int(vuid)))
                self._track_task(vtask)
            # Mark covered on actual dispatch (not at allocation time): if the pending-cap
            # break above drops a coverage assignment, the miner must NOT be recorded as
            # covered for this epoch without having been sent work.
            if adaptive:
                try:
                    self._article_cooldown.mark_covered(self.metagraph.hotkeys[int(uid)], epoch)
                except Exception:
                    pass

    def _reconcile_article_inflight(self):
        """Rebuild per-miner in-flight from the article store's PROCESSING set, so a
        missed or duplicated completion event can never strand the window. PROCESSING
        is per-article; the window is per-batch, so convert with
        ceil(articles / batch_size). This is the sole source of truth for in-flight
        under adaptive dispatch (the dispatch coroutine no longer releases at the ack)."""
        default_size = max(1, int(getattr(config, "MINER_BATCH_SIZE", 12)))
        # Count PROCESSING articles grouped by (hotkey, size the batch was DISPATCHED at). Each
        # dispatched batch contributes exactly `size` PROCESSING articles that move together
        # (a batch is set_processed / reset as a unit), so within a group the count is a whole
        # number of batches: ceil(count / size) == count // size batches. Summing per hotkey
        # gives an EXACT in-flight batch count that does not drift when the per-miner size ramps
        # while old-size batches are still outstanding (the reconcile checklist item). Legacy /
        # unstamped items fall back to the served baseline.
        group_counts = {}  # (hotkey, size) -> article count
        for item in self._article_store.get_processing_articles():
            hk = getattr(item, "hotkey", None)
            if not hk:
                continue
            size = getattr(item, "dispatch_batch_size", None) or default_size
            key = (hk, max(1, int(size)))
            group_counts[key] = group_counts.get(key, 0) + 1
        batch_counts = {}
        for (hk, size), c in group_counts.items():
            batch_counts[hk] = batch_counts.get(hk, 0) + -(-c // size)
        self._article_cooldown.reconcile_inflight(batch_counts)

    def _current_epoch(self) -> int:
        try:
            return int(self.block) // int(getattr(config, "BLOCK_LENGTH", 100))
        except Exception:
            return 0

    def _build_dispatch_status(self) -> list:
        """Per-miner adaptive-dispatch status snapshot for the dashboard diagnostics
        flush (display-only, consensus-decoupled). Covers every currently-live miner
        plus any miner we hold cooldown/window state for."""
        hotkeys = list(self.metagraph.hotkeys)
        hk_to_uid = {hk: u for u, hk in enumerate(hotkeys)}
        ct = self._article_cooldown.snapshot()
        live_hks = {hotkeys[u] for u in self._liveness.live_uids(self.metagraph)}
        w_min = float(getattr(config, "DISPATCH_WINDOW_MIN", 1))
        # Per-miner reputation emission multiplier (display-only): ~0 below the cliff, ~1
        # cleared, up to ~1.3 with the bonus. None when reputation scoring is off.
        rep_on = getattr(config, "REPUTATION_SCORING_ENABLED", False)
        _params = emission_params.resolve(
            self._mechanism_profile.resolve(int(self.block)))
        _em_args = _params.as_args()
        _n_min = _params.n_min
        rows = []
        for hk in (set(ct) | live_hks):
            st = ct.get(hk, {})
            emission_mult = None
            if rep_on:
                try:
                    r = self._reputation_store.reputation(hk)
                    n = self._reputation_store.samples(hk)
                    emission_mult = round(
                        float(_rep_emission(r, *_em_args, n=n, n_min=_n_min)), 3)
                except Exception:
                    emission_mult = None
            rows.append({
                "hotkey": hk,
                "uid": int(hk_to_uid.get(hk, -1)),
                "alive": bool(self._liveness.is_alive(hk)),
                "window": float(st.get("window", w_min)),
                "inflight": int(st.get("inflight", 0)),
                "consec_to": int(st.get("consec_to", 0)),
                "batch_size": int(st.get("batch_size", self._article_cooldown.batch_size(hk))),
                "covered_epoch": int(st.get("covered_epoch", -1)),
                "on_cooldown": bool(st.get("on_cooldown", False)),
                "cooldown_remaining_s": int(st.get("cooldown_remaining_s", 0)),
                # New display-only telemetry (the fields snapshot() carries but this dropped):
                "consec_inv": int(st.get("consec_inv", 0)),
                "consec_fail": int(st.get("consec_fail", 0)),
                "inv_level": int(st.get("inv_level", 0)),
                "last_faith": st.get("last_faith"),
                "median_latency_s": st.get("median_latency_s"),
                "emission_mult": emission_mult,
            })
        return rows

    def _select_article_targets(self, miner_batches, exclude):
        """
        Choose (uid, batch) dispatch targets.

        Flag off: unchanged random selection. Flag on: coverage-then-depth over the
        live roster (see utils/dispatch.coverage_depth_select). Read-only on the
        tracker here — the real per-miner reservation stays in
        _dispatch_article_miner_batch.try_acquire, so a pending-cap truncation cannot
        leak a reserved slot.
        """
        n_batches = len(miner_batches)
        if not getattr(config, "ADAPTIVE_DISPATCH_ENABLED", False):
            uids = list(get_random_uids(self, k=n_batches, exclude=exclude))
            return [(int(u), b) for b, u in zip(miner_batches, uids)]

        # Sync in-flight to ground truth before allocating (leak-proof; see method).
        self._reconcile_article_inflight()

        hotkeys = list(self.metagraph.hotkeys)
        blacklisted = getattr(config, "BLACKLISTED_MINER_HOTKEYS", set()) or set()
        exclude_set = {int(u) for u in exclude}
        live = [
            u for u in self._liveness.live_uids(self.metagraph)
            if u not in exclude_set and 0 <= u < len(hotkeys) and hotkeys[u] not in blacklisted
        ]
        if not live:
            bt.logging.warning("[DISPATCH] adaptive dispatch on but live roster is empty this tick; no targets")
            return []

        # Anti-monopoly cap, recomputed per tick: cap_pct of the validator's total
        # in-flight send budget (a stable, exogenous quantity — NOT the sum of
        # windows, which would feed back and run away). The budget is its own knob
        # (config.dispatch_window_budget) rather than the dispatch semaphore, so send
        # concurrency can be raised for throughput without widening per-miner depth.
        # Note the allocator floors this cap, so depth only opens up once it reaches
        # 2.0 — below that every miner is coverage-only at one batch each.
        cap_pct = float(getattr(config, "DISPATCH_WINDOW_CAP_PCT", 0.15))
        w_min = float(getattr(config, "DISPATCH_WINDOW_MIN", 1))
        budget = config.dispatch_window_budget()
        self._article_cooldown.set_cap(max(w_min, cap_pct * budget))

        epoch = self._current_epoch()
        assignments = coverage_depth_select(live, hotkeys, self._article_cooldown, epoch, n_batches)

        n_assigned = len(assignments)
        if n_assigned < n_batches:
            bt.logging.info(
                f"[DISPATCH] adaptive: {n_batches - n_assigned}/{n_batches} batch(es) unassigned this "
                f"tick (live windows full); they stay unprocessed and retry next tick."
            )
        else:
            distinct = len({u for u, _ in assignments})
            bt.logging.info(f"[DISPATCH] adaptive: assigned {n_assigned} batch(es) across {distinct} live miner(s)")
        return [(uid, miner_batches[bi]) for uid, bi in assignments]

    async def _dispatch_telegram_miner_batch(self, miner_batch: List[TelegramMessageForScoring], uid: int) -> None:
        hotkey = None
        try:
            hotkey = self.metagraph.hotkeys[int(uid)]
        except Exception:
            pass
        if hotkey and not self._telegram_cooldown.try_acquire(hotkey):
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
                self._telegram_cooldown.release(hotkey)

    async def _dispatch_verification_batch(self, miner_batch, uid: int) -> None:
        """Send a verification copy. No lease is held and nothing is reset on
        failure — the primary assignment owns the article lifecycle. The
        assignment is registered only after the miner acks the send."""
        try:
            try:
                hotkey = self.metagraph.hotkeys[int(uid)]
            except Exception:
                return
            async with self._miner_dispatch_semaphore:
                resp = await self._process_article_miner_batch(
                    miner_batch, uid, verification=True)
            if resp is not None:
                self._register_verification(hotkey, miner_batch)
            else:
                # No ack: shrink the article's pot divisor.
                for a in miner_batch:
                    entry = self._article_k.get(str(a.id))
                    if entry and entry[0] > 1:
                        self._article_k[str(a.id)] = (entry[0] - 1, entry[1])
        except Exception as e:
            bt.logging.debug(f"[OVERLAP] verification dispatch failed uid={uid}: {e}")

    async def _dispatch_article_miner_batch(self, miner_batch: List[NewsArticleForScoring], uid: int) -> None:
        hotkey = None
        try:
            hotkey = self.metagraph.hotkeys[int(uid)]
        except Exception:
            pass
        if hotkey and not self._article_cooldown.try_acquire(hotkey):
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
            # Static path releases at the ack as before. Under adaptive dispatch the ack
            # is not "work done": in-flight is reconciled from the article store's
            # PROCESSING set each cycle (_reconcile_article_inflight), so releasing here
            # would double-free against the reconcile. A failed send resets its articles
            # to UNPROCESSED, so they leave PROCESSING and the next reconcile reclaims
            # the slot automatically.
            if hotkey and not getattr(config, "ADAPTIVE_DISPATCH_ENABLED", False):
                self._article_cooldown.release(hotkey)

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
                    self._tweet_cooldown.record_failure(miner_hotkey)
                for tweet in miner_batch:
                    try:
                        self._tweet_store.reset_to_unprocessed(tweet.id)
                    except Exception:
                        pass
                return None

            if miner_hotkey:
                self._tweet_cooldown.record_success(miner_hotkey)
            return responses[0]
        except Exception as e:
            bt.logging.error(f"[VALIDATION] Failed to process miner batch: {e}", exc_info=True)
            if miner_hotkey:
                self._tweet_cooldown.record_failure(miner_hotkey)
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
                    self._telegram_cooldown.record_failure(miner_hotkey)
                for msg in miner_batch:
                    try:
                        self._telegram_store.reset_to_unprocessed(msg.id)
                    except Exception:
                        pass
                return None

            if miner_hotkey:
                self._telegram_cooldown.record_success(miner_hotkey)
            return responses[0]
        except Exception as e:
            bt.logging.error(f"[VALIDATION] Failed to process telegram miner batch: {e}", exc_info=True)
            if miner_hotkey:
                self._telegram_cooldown.record_failure(miner_hotkey)
            for msg in miner_batch:
                try:
                    self._telegram_store.reset_to_unprocessed(msg.id)
                except Exception:
                    pass
            return None

    async def _process_article_miner_batch(
        self,
        miner_batch: List[NewsArticleForScoring],
        uid: int,
        verification: bool = False,
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

            dispatch_size = len(miner_batch)  # record the ACTUAL dispatched size on the lease
            if not verification:
                for article in miner_batch:
                    self._article_store.add_article(article, article_id=article.id, hotkey=miner_hotkey, set_as_processing=False, overwrite=False)
                    try:
                        self._article_store.set_processing(article.id, hotkey=miner_hotkey, batch_size=dispatch_size)
                    except Exception:
                        pass

            article_batch = ArticleBatch(
                article_batch=miner_batch
            )
            axon = self.metagraph.axons[uid]
            # Adaptive dispatch: a short ack timeout replaces the 30 s blocking send so
            # dead axons stop holding a dispatch slot. The latency signal the window
            # needs comes from the push-back, not this ack.
            adaptive = getattr(config, "ADAPTIVE_DISPATCH_ENABLED", False)
            send_timeout = (
                float(getattr(config, "DISPATCH_ACK_TIMEOUT_S", 3.0)) if adaptive
                else float(getattr(config, "ARTICLE_SEND_TIMEOUT", 30.0))
            )
            responses = await self.dendrite.forward(
                axons=[axon],
                synapse=article_batch,
                timeout=send_timeout,
                deserialize=True
            )
            if adaptive:
                self._adaptive_metrics.incr("dispatched")
            if not responses[0].dendrite.status_code == 200:
                bt.logging.error(f"[VALIDATION] Failed to process article miner batch: {responses[0].dendrite.status_message}")
                # Under adaptive dispatch a failed send = unreachable miner, not a cheater:
                # it simply ages out of the liveness roster (no push-back, no heartbeat).
                # No integrity penalty. The static path keeps the legacy cooldown behaviour.
                if adaptive:
                    self._adaptive_metrics.incr("ack_fail")
                if miner_hotkey and not adaptive and not verification:
                    self._article_cooldown.record_failure(miner_hotkey)
                if not verification:
                    for article in miner_batch:
                        try:
                            self._article_store.reset_to_unprocessed(article.id)
                        except Exception:
                            pass
                return None

            if adaptive:
                self._adaptive_metrics.incr("ack_ok")
                # Ack round-trip — reveals whether the send semaphore is being held
                # across slow acks (busy miner axons), which bounds the depth ramp.
                try:
                    _pt = getattr(responses[0].dendrite, "process_time", None)
                    if _pt is not None:
                        self._adaptive_metrics.record_ack(float(_pt))
                except Exception:
                    pass
            if miner_hotkey and not verification:
                self._article_cooldown.record_success(miner_hotkey)
            return responses[0]
        except Exception as e:
            bt.logging.error(f"[VALIDATION] Failed to process article miner batch: {e}", exc_info=True)
            # See above: under adaptive dispatch a send-path failure is not an integrity
            # penalty; the miner ages out of the liveness roster instead.
            if (miner_hotkey and not verification
                    and not getattr(config, "ADAPTIVE_DISPATCH_ENABLED", False)):
                self._article_cooldown.record_failure(miner_hotkey)
            if not verification:
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

            try:
                hotkey = self._tweet_store.get_hotkey(tweet.id)
            except (KeyError, Exception):
                hotkey = None

            # T5 wiring: build base submission dict so extra verdict fields can be merged.
            # miner_signature, nonce, validator_verdict, points_awarded, and epoch are NOT
            # available here — they live on the TweetBatch response synapse (consumed in
            # _handle_miner_batch_response) and in validation_client.py's epoch loop.
            # To complete the wiring, store sig/nonce/verdict on TweetStoreItem when the
            # batch is accepted in _handle_miner_batch_response, then read them back here.
            base = CompletedTweetSubmission(
                tweet_id=tweet.id,
                sentiment=tweet.analysis.sentiment or "neutral",
                asset_id=tweet.analysis.asset_id,
                asset_symbol=tweet.analysis.asset_symbol,
                content_type=tweet.analysis.content_type,
                technical_quality=tweet.analysis.technical_quality,
                market_analysis=tweet.analysis.market_analysis,
                impact_potential=tweet.analysis.impact_potential,
                relevance_confidence=getattr(tweet.analysis, "relevance_confidence", None),
                miner_hotkey=hotkey,
            ).model_dump(exclude_none=True)
            meta = self._verdict_meta.pop(str(tweet.id), None)
            if meta and hotkey:
                base.update(build_verdict_fields(
                    miner_hotkey=hotkey, miner_signature=meta["miner_signature"],
                    nonce=meta["nonce"], analysis=tweet.analysis,
                    validator_verdict=meta["validator_verdict"],
                    points_awarded=1.0, epoch=meta["epoch"]))
            completed_tweets.append(base)
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

            try:
                hotkey = self._telegram_store.get_hotkey(msg.id)
            except (KeyError, Exception):
                hotkey = None

            # T5 wiring: same as tweet path — sig/nonce/verdict/epoch not in scope here.
            # Needs TelegramStoreItem to carry miner_signature, nonce, validator_verdict,
            # and epoch (stored in _handle_telegram_miner_batch_response when accepted).
            base = CompletedTelegramMessageSubmission(
                message_id=msg.id,
                sentiment=msg.analysis.sentiment or "neutral",
                asset_id=msg.analysis.asset_id,
                asset_symbol=msg.analysis.asset_symbol,
                content_type=msg.analysis.content_type,
                technical_quality=msg.analysis.technical_quality,
                market_analysis=msg.analysis.market_analysis,
                impact_potential=msg.analysis.impact_potential,
                relevance_confidence=getattr(msg.analysis, "relevance_confidence", None),
                miner_hotkey=hotkey,
            ).model_dump(exclude_none=True)
            meta = self._verdict_meta.pop(str(msg.id), None)
            if meta and hotkey:
                base.update(build_verdict_fields(
                    miner_hotkey=hotkey, miner_signature=meta["miner_signature"],
                    nonce=meta["nonce"], analysis=msg.analysis,
                    validator_verdict=meta["validator_verdict"],
                    points_awarded=1.0, epoch=meta["epoch"]))
            completed_messages.append(base)
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

            try:
                hotkey = self._article_store.get_hotkey(str(article.id))
            except (KeyError, Exception):
                hotkey = None

            # T5 wiring: same as tweet/telegram paths — sig/nonce/verdict/epoch not in scope.
            # Needs ArticleStoreItem to carry miner_signature, nonce, validator_verdict,
            # and epoch (stored in _handle_article_miner_batch_response when accepted).
            base = CompletedNewsArticleSubmission(
                article_id=article.id,
                sentiment=article.analysis.sentiment or "neutral",
                sector_id=article.analysis.sector_id,
                sector_symbol=article.analysis.sector_symbol,
                content_type=article.analysis.content_type,
                technical_quality=article.analysis.technical_quality,
                market_analysis=article.analysis.market_analysis,
                impact_potential=article.analysis.impact_potential,
                relevance_confidence=getattr(article.analysis, "relevance_confidence", None),
                analysis_data=getattr(article.analysis, "analysis_data", None),
                miner_hotkey=hotkey,
            ).model_dump(exclude_none=True)
            meta = self._verdict_meta.pop(str(article.id), None)
            if meta and hotkey:
                base.update(build_verdict_fields(
                    miner_hotkey=hotkey, miner_signature=meta["miner_signature"],
                    nonce=meta["nonce"], analysis=article.analysis,
                    validator_verdict=meta["validator_verdict"],
                    points_awarded=self._article_pay.pop(str(article.id), 1.0),
                    epoch=meta["epoch"]))
            completed_articles.append(base)
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

        # Liveness heartbeat (adaptive dispatch). Off the dispatch path; the loop
        # itself no-ops while the flag is disabled, so this is inert by default.
        if self._liveness_task is None or self._liveness_task.done():
            self._liveness_task = asyncio.create_task(self._liveness_sweep_loop())

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

        # Adaptive dispatch pilot metrics: one parseable line per cycle, then reset.
        if self.step % 100 == 0 and getattr(config, "ADAPTIVE_DISPATCH_ENABLED", False):
            try:
                _, on_cd = self._article_cooldown.stats()
                _, live = self._liveness.stats()
                bt.logging.info(self._adaptive_metrics.format_line(
                    self._article_cooldown.window_values(), live, on_cd))
                self._adaptive_metrics.reset()
            except Exception as e:
                bt.logging.warning(f"[ADAPTIVE_METRICS] failed to emit: {e}")

        return await forward(self)

    async def _liveness_sweep_loop(self):
        """
        Background heartbeat that keeps the liveness roster fresh, fully OFF the
        dispatch path. While ADAPTIVE_DISPATCH_ENABLED is off it just sleeps, so it
        is inert by default and toggling the remote flag turns it on within one
        interval without a restart. Maps alive UIDs → hotkeys (roster is hotkey-keyed).
        """
        while True:
            interval = max(5, int(getattr(config, "LIVENESS_SWEEP_INTERVAL_S", 60)))
            try:
                if getattr(config, "ADAPTIVE_DISPATCH_ENABLED", False):
                    alive_uids = await get_alive_uids(self.metagraph, self.dendrite)
                    hotkeys = list(self.metagraph.hotkeys)
                    alive_hotkeys = [hotkeys[u] for u in alive_uids if 0 <= u < len(hotkeys)]
                    self._liveness.update_from_heartbeat(alive_hotkeys)
                    self._liveness.prune(set(hotkeys))
                    tracked, live = self._liveness.stats()
                    bt.logging.info(
                        f"[LIVENESS] heartbeat: {len(alive_hotkeys)} alive via IsAlive; "
                        f"roster {live}/{tracked} live"
                    )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                bt.logging.warning(f"[LIVENESS] sweep error: {e}")
            await asyncio.sleep(interval)

    def _prune_stores(self):
        """Prune old data from stores to maintain bounded memory usage."""
        try:
            # Prune tweet store: remove submitted tweets and old unprocessed ones
            self._tweet_store.prune_old_tweets(max_age_seconds=3600, max_tweets=1000)
            self._tweet_store.save_to_file()
            
            # Prune telegram store: remove submitted messages and old unprocessed ones
            self._telegram_store.prune_old_messages(max_age_seconds=3600, max_messages=1000)
            self._telegram_store.save_to_file()

            # Prune article store: remove submitted articles and old unprocessed ones.
            # max_articles is remote-config tunable so the buffer can be sized subnet-wide.
            self._article_store.prune_old_articles(
                max_age_seconds=3600, max_articles=config.ARTICLE_STORE_MAX_ARTICLES)
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
            hotkey_to_uid = {hk: i for i, hk in enumerate(self.metagraph.hotkeys)}
            from alpharidge_ai.validator.reward_broadcast_store import route_reward_broadcast
            accepted, reason = route_reward_broadcast(
                store=self._reward_broadcasts,
                sender_hotkey=authenticated_hotkey,
                epoch=synapse.epoch,
                seq=synapse.seq,
                uid_points=synapse.uid_points,
                attestation=getattr(synapse, "attestation", None),
                attestation_sig=getattr(synapse, "attestation_sig", None),
                hotkey_to_uid=hotkey_to_uid,
                pinned_pubkey=config.API_ATTESTATION_PUBKEY,
                blacklisted=set(config.BLACKLISTED_MINER_HOTKEYS),
                enforce_signed=config.ENFORCE_SIGNED_ATTESTATIONS,
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

    async def forward_validator_reputation_obs(self, synapse: ValidatorReputationObs) -> ValidatorReputationObs:
        """Receive graded observations from other validators and buffer for aggregation."""
        if not getattr(config, "REPUTATION_SCORING_ENABLED", False):
            return synapse
        try:
            sender = synapse.dendrite.hotkey
            targets = {t: [tuple(o) for o in lst] for t, lst in (synapse.observations or {}).items()}
            accepted, reason = self._reputation_store.ingest(
                sender, int(synapse.epoch), targets, seq=int(synapse.seq))
            self._reputation_store.save()
            if accepted:
                bt.logging.info(
                    f"[REPUTATION_BROADCAST] Ingested from {sender[:12]}.. "
                    f"epoch={synapse.epoch} targets={len(targets)} {reason}")
            else:
                bt.logging.warning(
                    f"[REPUTATION_BROADCAST] Rejected from {sender[:12]}.. "
                    f"epoch={synapse.epoch} reason={reason}")
        except Exception as e:
            bt.logging.debug(f"[REPUTATION_BROADCAST] Failed to ingest: {e}")
        return synapse


# Entrypoint
if __name__ == "__main__":
    with Validator() as validator:
        while True:
            bt.logging.info(f"Validator running... {time.time()}")
            time.sleep(5)
