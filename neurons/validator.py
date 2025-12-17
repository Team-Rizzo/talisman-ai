# neurons/validator.py
# The MIT License (MIT)
# Copyright © 2023 Team Rizzo

import asyncio
import time
from typing import List, Dict, Any, Optional
import numpy as np
import bittensor as bt
from talisman_ai.base.validator import BaseValidatorNeuron
from talisman_ai.validator.forward import forward
from talisman_ai.validator.validation_client import ValidationClient
from talisman_ai.validator.grader import grade_hotkey, CONSENSUS_VALID, CONSENSUS_INVALID
from talisman_ai.analyzer import setup_analyzer
import talisman_ai.protocol
from talisman_ai import config
from talisman_ai.utils.api_models import TweetWithUser, CompletedTweetSubmission   
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
    Validator neuron for API v2 probabilistic validation system.
    
    The validator:
    1. Gets validation payloads from /v2/validation
    2. Grades individual posts using the grading system
    3. Submits results to /v2/validation_result
    4. Gets scores from /v2/scores every N blocks and sets hotkey rewards (only source of score updates)
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
        self._pending_results: List[Dict[str, Any]] = []
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
        # ensure the batch size is equal to the MINER_BATCH_SIZE
        if len(synapse.tweet_batch) != config.MINER_BATCH_SIZE:
            bt.logging.error(f"[VALIDATION] Tweet batch size is not equal to MINER_BATCH_SIZE: {len(synapse.tweet_batch)} != {config.MINER_BATCH_SIZE}")
            self._miner_penalty.add_penalty(synapse.dendrite.hotkey, "Invalid batch size")
            return synapse
        
        # Score the tweets
        is_valid, result = await validate_miner_batch(synapse.tweet_batch, self._analyzer)
        if not is_valid:
            hotkey = synapse.dendrite.hotkey
            for tweet in synapse.tweet_batch:
                self._tweet_store.reset_to_unprocessed(tweet.id)
            self._miner_penalty.add_penalty(hotkey, "Invalid score")
            return synapse

        for tweet in synapse.tweet_batch:
            self._tweet_store.set_processed(tweet.id)
            self._miner_reward.add_reward(synapse.dendrite.hotkey, 1)
        return synapse
        
    async def _on_tweets(self, tweets: List[TweetWithUser]):
        """
        Process multiple validation payloads in batch (sequentially).
        
        Args:
            tweets: List of tweets
        """
        if not tweets:
            return
        
        bt.logging.info(f"[VALIDATION] Processing {len(tweets)} tweets in batch")
        for tweet in tweets:
            self._tweet_store.add_tweet(tweet, set_as_processing=False)
        # Process all tweets sequentially (one at a time)
        results = []
        validation_results_by_hotkey = {}  # Group validation results by hotkey for batching
        miner_batches = []  
        for i in range(0, len(tweets.tweets), config.MINER_BATCH_SIZE):
            miner_batches.append(tweets.tweets[i:i + config.MINER_BATCH_SIZE])
        uids = get_random_uids(self.metagraph, self.dendrite, k=len(miner_batches), is_alive=True)

        for miner_batch, uid in zip(miner_batches, uids):
            await self._process_miner_batch(miner_batch, uid)
            
    async def _process_miner_batch( 
        self, 
        miner_batch: List[TweetWithUser],
        uid: int
    ) -> TweetBatch:
        """
        Process a miner batch.
        
        Args:
            validation: Validation payload with validation_id, miner_hotkey, post, selected_at
            validation_results_by_hotkey: Dictionary to accumulate validation results by hotkey for batching
        
        Returns:
            Result dict ready for submission
        """
        try:
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
                return None
            hotkey = self.metagraph.hotkeys[uid] # TODO no clue if this works
            for tweet in tweet_batch.tweet_batch:
                self._tweet_store.set_processing(tweet.id, hotkey)
            
            return responses[0]
        except Exception as e:
            bt.logging.error(f"[VALIDATION] Failed to process miner batch: {e}", exc_info=True)
            return None
    
    async def _submit_tweet_batch(self, tweet_batch: List[TweetWithUser]):
        """Submit a tweet batch to the API"""
        completed_tweets = []
        for tweet in tweet_batch:
            completed_tweets.append(CompletedTweetSubmission(
                tweet_id=tweet.id,
                sentiment=tweet.sentiment
            ))
        response = await self._validation_client.api_client.submit_completed_tweets(completed_tweets)
        return response
    
    async def _submit_pending_results(self):
        """Submit pending validation results to the API"""
        if not self._pending_results:
            return
        
        results = self._pending_results.copy()
        self._pending_results.clear()
        
        bt.logging.info(f"[VALIDATION] Submitting {len(results)} validation result(s)")
        
        try:
            response = await self._validation_client.submit_results(results)
            bt.logging.info(f"[VALIDATION] ✓ Submitted results: {response}")
        except Exception as e:
            bt.logging.error(f"[VALIDATION] ✗ Failed to submit results: {e}", exc_info=True)
            # Re-add results to pending for retry
            self._pending_results.extend(results)

    async def _on_scores(self, scores_data: Dict[str, Any]):
        """
        Handle scores fetched from /v2/scores endpoint.
        
        Sets hotkey rewards based on scores from the previous completed block window.
        This is called once per window when a new window starts, ensuring rewards are
        set based on complete window data rather than incomplete current window data.
        
        Args:
            scores_data: Scores response with scores dict, block_window metadata, etc.
        """
        scores = scores_data.get("scores", {})
        current_block = scores_data.get("current_block")
        block_window_start = scores_data.get("block_window_start")
        block_window_end = scores_data.get("block_window_end")
        
        bt.logging.info(
            f"[SCORES] Processing scores for block window {block_window_start}-{block_window_end} "
            f"(current={current_block}, {len(scores)} hotkeys)"
        )
        
        if not scores:
            bt.logging.warning("[SCORES] No scores in response")
            return
        
        uids_to_update = []
        rewards_array = []
        
        for hotkey, score in scores.items():
            try:
                uid = self.metagraph.hotkeys.index(hotkey)
                uids_to_update.append(uid)
                rewards_array.append(float(score))
            except ValueError:
                bt.logging.debug(f"[SCORES] Hotkey {hotkey} not found in metagraph, skipping")
        
        if uids_to_update and rewards_array:
            rewards_np = np.array(rewards_array)
            uids_np = np.array(uids_to_update)
            bt.logging.info(f"[SCORES] Setting rewards for {len(uids_to_update)} miner(s)...")
            self.update_scores(rewards_np, uids_np.tolist())
            bt.logging.info(
                f"[SCORES] ✓ Set rewards: range {rewards_np.min():.3f} - {rewards_np.max():.3f}, "
                f"mean={rewards_np.mean():.3f}"
            )
            
            # Send Score synapses to each miner
            await self._send_scores_to_miners(scores, block_window_start, block_window_end)
        else:
            bt.logging.warning("[SCORES] No valid hotkeys found to update")
    
    async def _send_scores_to_miners(
        self, 
        scores: Dict[str, float], 
        block_window_start: int, 
        block_window_end: int
    ):
        """
        Send Score synapses to each miner with their score for the 100-block interval.
        
        Args:
            scores: Dictionary mapping hotkey to score
            block_window_start: Start block of the interval
            block_window_end: End block of the interval
        """
        if not scores:
            return
        
        bt.logging.info(
            f"[SCORES] Sending Score synapses to {len(scores)} miner(s) for block window {block_window_start}-{block_window_end}"
        )
        
        # Get axons for all miners with scores
        axons_to_query = []
        score_synapses = []
        
        for hotkey, score in scores.items():
            try:
                uid = self.metagraph.hotkeys.index(hotkey)
                axon = self.metagraph.axons[uid]
                axons_to_query.append(axon)
                
                # Create Score synapse for this miner
                synapse = talisman_ai.protocol.Score(
                    block_window_start=block_window_start,
                    block_window_end=block_window_end,
                    score=float(score),
                    validator_hotkey=str(self.wallet.hotkey.ss58_address)
                )
                score_synapses.append(synapse)
            except ValueError:
                bt.logging.debug(f"[SCORES] Hotkey {hotkey} not found in metagraph, skipping Score synapse")
                continue
        
        if not axons_to_query:
            bt.logging.warning("[SCORES] No valid axons found to send Score synapses")
            return
        
        # Send Score synapses to all miners serially
        # Each synapse is matched to its corresponding axon
        try:
            success_count = 0
            for axon, synapse in zip(axons_to_query, score_synapses):
                try:
                    responses = await self.dendrite.forward(
                        axons=[axon],
                        synapse=synapse,
                        timeout=12.0,
                        deserialize=True
                    )
                    if responses and responses[0].dendrite.status_code == 200:
                        success_count += 1
                except Exception as e:
                    bt.logging.debug(f"[SCORES] Failed to send Score synapse to {axon.hotkey}: {e}")
            
            bt.logging.info(
                f"[SCORES] ✓ Sent Score synapses: {success_count}/{len(axons_to_query)} successful"
            )
        except Exception as e:
            bt.logging.error(f"[SCORES] ✗ Failed to send Score synapses: {e}", exc_info=True)

    async def _send_batched_validation_results(
        self,
        validation_results_by_hotkey: Dict[str, List[Dict[str, Any]]]
    ):
        """
        Send batched ValidationResult synapses to miners, grouped by hotkey.
        
        Args:
            validation_results_by_hotkey: Dictionary mapping hotkey to list of validation results
        """
        if not validation_results_by_hotkey:
            return
        
        # Send validation results to each hotkey
        for miner_hotkey, results in validation_results_by_hotkey.items():
            try:
                # Find the miner's axon in the metagraph
                uid = self.metagraph.hotkeys.index(miner_hotkey)
                axon = self.metagraph.axons[uid]
                
                # Create ValidationResult synapses for all posts validated for this hotkey
                synapses = []
                for result in results:
                    synapse = talisman_ai.protocol.ValidationResult(
                        validation_id=result["validation_id"],
                        post_id=result["post_id"],
                        success=result["success"],
                        validator_hotkey=str(self.wallet.hotkey.ss58_address),
                        failure_reason=result["failure_reason"]
                    )
                    synapses.append(synapse)
                
                # Send all synapses to the miner serially
                success_count = 0
                for synapse in synapses:
                    try:
                        responses = await self.dendrite.forward(
                            axons=[axon],
                            synapse=synapse,
                            timeout=12.0,
                            deserialize=True
                        )
                        if responses and responses[0].dendrite.status_code == 200:
                            success_count += 1
                    except Exception as e:
                        bt.logging.debug(
                            f"[VALIDATION] Failed to send ValidationResult to {miner_hotkey} "
                            f"for post {synapse.post_id}: {e}"
                        )
                
                bt.logging.info(
                    f"[VALIDATION] ✓ Sent {success_count}/{len(synapses)} ValidationResult(s) to {miner_hotkey} "
                    f"({len(results)} post(s) validated)"
                )
            except ValueError:
                bt.logging.warning(
                    f"[VALIDATION] Miner hotkey {miner_hotkey} not found in metagraph, "
                    f"skipping ValidationResult synapses"
                )
            except Exception as e:
                bt.logging.error(
                    f"[VALIDATION] Failed to send ValidationResult synapses to {miner_hotkey}: {e}",
                    exc_info=True
                )

    async def forward(self):
        """
        Main validator forward loop.
        
        Starts the validation client on first invocation. The client runs independently
        in the background, processing validations and fetching scores as needed.
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
