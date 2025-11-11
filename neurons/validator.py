# neurons/validator.py
# The MIT License (MIT)
# Copyright © 2023 Team Rizzo

import asyncio
import time
from typing import List, Dict, Any, Optional
import sys
import os

import numpy as np
import bittensor as bt
from talisman_ai.base.validator import BaseValidatorNeuron
from talisman_ai.validator.forward import forward

from talisman_ai import config
from talisman_ai.validator.batch_client import BatchClient
from talisman_ai.validator.grader import grade_hotkey, CONSENSUS_VALID, CONSENSUS_INVALID

import httpx
import talisman_ai.protocol as protocol

# Auth utilities for creating signed headers (defined inline, not imported from API)
def create_auth_message(timestamp=None):
    """Create a standardized authentication message"""
    if timestamp is None:
        timestamp = time.time()
    return f"talisman-ai-auth:{int(timestamp)}"

def sign_message(wallet, message):
    """Sign a message with the wallet's hotkey"""
    signature = wallet.hotkey.sign(message)
    return signature.hex()


# API configuration for batch validation system
API_URL = config.MINER_API_URL
VOTE_ENDPOINT = config.VOTE_ENDPOINT
HTTP_TIMEOUT = config.BATCH_HTTP_TIMEOUT


class Validator(BaseValidatorNeuron):
    """
    Validator neuron that processes batches of miner posts for validation.
    
    The validator operates in two phases:
    1. Batch Processing: Polls the API for batches of miner posts, grades each miner's
       submissions using the grading system, and submits votes back to the API.
    2. Score Management: Updates miner scores based on validation results, which are
       used by the base validator to set weights on-chain.
    
    The batch client runs asynchronously in the background, processing batches
    independently of the main validator forward loop.
    """

    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)

        bt.logging.info("load_state()")
        self.load_state()

        # Initialize batch client for polling miner submissions
        # The client will be started in forward() once an asyncio event loop is available
        # Pass wallet for authentication
        self._batch_client = BatchClient(wallet=self.wallet)
        self._batch_task: asyncio.Task | None = None
        self._current_batch_id: Optional[str] = None  # Track current batch_id for querying miners
        self._miner_scores: Dict[str, float] = {}  # Track scores per miner hotkey to send back

    async def _submit_hotkey_votes(self, batch_id: int, votes: List[Dict[str, Any]]):
        """
        Submit validator votes for a batch to the API.
        
        Each vote contains:
        - miner_hotkey: The miner's hotkey address
        - label: Binary classification (1=VALID, 0=INVALID)
        - score: Final incentive score (0.0-1.0) from grading
        
        Args:
            batch_id: The batch identifier being voted on
            votes: List of vote dictionaries, one per miner in the batch
        """
        payload = {
            "validator_hotkey": str(self.wallet.hotkey.ss58_address),
            "batch_id": int(batch_id),
            "votes": votes,
        }
        
        # Create authentication headers if wallet is available
        headers = {}
        if self.wallet:
            try:
                timestamp = time.time()
                message = create_auth_message(timestamp)
                signature = sign_message(self.wallet, message)
                headers = {
                    "X-Auth-SS58Address": self.wallet.hotkey.ss58_address,
                    "X-Auth-Signature": signature,
                    "X-Auth-Message": message,
                    "X-Auth-Timestamp": str(timestamp)
                }
                bt.logging.debug(f"[VALIDATE] Created authentication headers for hotkey: {self.wallet.hotkey.ss58_address}")
            except Exception as e:
                bt.logging.warning(f"[VALIDATE] Failed to create auth headers: {e}, proceeding without auth")
                headers = {}

        bt.logging.info(f"[VALIDATE] Submitting {len(votes)} votes for batch_id={batch_id} to {VOTE_ENDPOINT}")
        bt.logging.debug(f"[VALIDATE] Payload: validator_hotkey={payload['validator_hotkey']}, batch_id={batch_id}")
        for i, vote in enumerate(votes):
            failure_info = ""
            if vote.get('failure_reason'):
                fr = vote['failure_reason']
                failure_info = f", failure={fr.get('code', 'unknown')} - {fr.get('message', 'N/A')}"
            bt.logging.debug(f"[VALIDATE] Vote {i+1}: miner={vote.get('miner_hotkey')}, label={vote.get('label')}, score={vote.get('score', 'N/A')}{failure_info}")

        try:
            bt.logging.info(f"[VALIDATE] Sending POST request with timeout={HTTP_TIMEOUT}s...")
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
                r = await client.post(VOTE_ENDPOINT, json=payload, headers=headers)
                r.raise_for_status()
                response_data = r.json()
                bt.logging.info(f"[VALIDATE] ✓ Vote submission successful: {response_data}")
        except httpx.TimeoutException as e:
            bt.logging.error(f"[VALIDATE] ✗ Vote submission timed out after {HTTP_TIMEOUT}s: {e}")
        except Exception as e:
            bt.logging.error(f"[VALIDATE] ✗ Vote submission error: {e}", exc_info=True)

    async def _on_batch(self, batch_id: int, batch: List[Dict[str, Any]]):
        """
        Process a batch of miner submissions.
        
        This callback is invoked by BatchClient when a new batch is available.
        For each miner in the batch:
        1. Grades their posts using the grading system (checks analysis accuracy and scores quality)
        2. Calculates reward based on label (VALID/INVALID) and final_score
        3. Maps hotkey to UID for score updates
        4. Collects votes for API submission
        
        After processing all miners:
        - Updates validator scores using the base validator's update_scores method
        - Submits votes to the API for consensus tracking
        
        Args:
            batch_id: Unique identifier for this batch
            batch: List of miner entries, each containing hotkey and posts to validate
        """
        bt.logging.info(f"[BATCH] ========== Processing batch {batch_id} ==========")
        bt.logging.info(f"[BATCH] Batch contains {len(batch)} hotkey(s)")
        total_posts = sum(h.get("total_posts", 0) for h in batch)
        bt.logging.info(f"[BATCH] Total posts in batch: {total_posts}")
        
        # Store current batch_id so forward() can use it to query miners
        self._current_batch_id = str(batch_id)

        votes: List[Dict[str, Any]] = []
        rewards_dict: Dict[str, float] = {}
        uids_to_update: List[int] = []
        rewards_array = []
        
        for idx, miner_entry in enumerate(batch):
            hotkey = miner_entry.get("hotkey")
            posts = miner_entry.get("posts", []) # API selected sample of posts to grade
            avg_score_all_posts = miner_entry.get("avg_score_all_posts") # API-calculated average of ALL posts

            bt.logging.info(f"[BATCH] [{idx+1}/{len(batch)}] Processing hotkey={hotkey} with {len(posts)} post(s)")
            bt.logging.debug(f"[BATCH] Posts for {hotkey}: {[p.get('post_id', 'N/A') for p in posts]}")
            if avg_score_all_posts is not None:
                bt.logging.info(f"[BATCH] API-provided avg_score_all_posts for {hotkey}: {avg_score_all_posts:.3f}")
            
            bt.logging.info(f"[BATCH] Grading hotkey {hotkey}...")
            label, grade_result = grade_hotkey(posts)
            # The grader validates sampled posts to determine VALID/INVALID.
            # We use avg_score_all_posts (API-calculated average of ALL posts) for scoring.
            
            # Use API-provided average score if batch is VALID, otherwise use 0.0
            # The validator still validates sampled posts to determine VALID/INVALID
            if label == CONSENSUS_VALID and avg_score_all_posts is not None:
                # Use the API-calculated average score of ALL posts (not just sampled)
                final_score = float(avg_score_all_posts)
                bt.logging.info(f"[BATCH] Using API-provided avg_score_all_posts={final_score:.3f} for {hotkey} (batch is VALID)")
            else:
                # If batch is INVALID or no avg_score_all_posts provided, use 0.0
                final_score = 0.0
                if label == CONSENSUS_VALID:
                    bt.logging.warning(f"[BATCH] Batch is VALID but no avg_score_all_posts provided for {hotkey}, using 0.0")
                else:
                    bt.logging.info(f"[BATCH] Batch is INVALID for {hotkey}, final_score=0.0")
            
            # Extract failure reason if batch failed
            failure_reason = None
            if label == CONSENSUS_INVALID:
                error_info = grade_result.get("error", {})
                error_code = error_info.get("code", "unknown_error")
                error_message = error_info.get("message", "Unknown error")
                post_id = error_info.get("post_id", "unknown")
                post_index = error_info.get("post_index", None)
                
                # Format failure reason for logging and API
                failure_reason = {
                    "code": error_code,
                    "message": error_message,
                    "post_id": post_id,
                    "post_index": post_index,
                    "details": error_info.get("details", {})
                }
                
                # Log failure reason in validator logs
                post_info = f"post_id={post_id}"
                if post_index is not None:
                    post_info += f", post_index={post_index}"
                bt.logging.warning(
                    f"[BATCH] ✗ Batch FAILED for {hotkey}: {error_code} - {error_message} "
                    f"({post_info})"
                )
                if error_info.get("details"):
                    bt.logging.debug(f"[BATCH] Failure details for {hotkey}: {error_info['details']}")
            
            # Apply reward modifier based on label:
            # VALID miners receive full incentive score, INVALID miners receive 10% penalty
            # This ensures INVALID miners still receive some reward but are heavily penalized
            if label == CONSENSUS_VALID:
                reward = final_score
            else:
                reward = final_score * 0.1
            
            rewards_dict[hotkey] = reward
            
            # Store final_score for this miner so we can send it back via Batchscore synapse
            self._miner_scores[hotkey] = final_score
            
            log_msg = (
                f"[BATCH] ✓ Graded {hotkey}: label={label} (posts={len(posts)}) "
                f"final_score={final_score:.3f} reward={reward:.3f}"
            )
            if failure_reason:
                log_msg += f" | Failure: {failure_reason['code']} - {failure_reason['message']}"
            bt.logging.info(log_msg)

            # Validators only vote VALID/INVALID
            vote_dict = {
                "miner_hotkey": hotkey,
                "label": int(label),
            }
            if failure_reason:
                vote_dict["failure_reason"] = failure_reason
            votes.append(vote_dict)
            
            # Map hotkey to UID for validator score updates
            try:
                #TODO need to test that this actually works
                uid = self.metagraph.hotkeys.index(hotkey)
                uids_to_update.append(uid)
                rewards_array.append(reward)
                bt.logging.debug(f"[BATCH] Mapped {hotkey} to UID {uid}")
            except ValueError:
                bt.logging.warning(f"[BATCH] ✗ Hotkey {hotkey} not found in metagraph, skipping score update")
        
        # Update validator scores for all miners in this batch
        # These scores are used by the base validator to set weights on-chain
        if uids_to_update and rewards_array:
            rewards_np = np.array(rewards_array)
            uids_np = np.array(uids_to_update)
            bt.logging.info(f"[BATCH] Updating scores for {len(uids_to_update)} miner(s)...")
            self.update_scores(rewards_np, uids_np.tolist())
            bt.logging.info(
                f"[BATCH] ✓ Updated scores: rewards range {rewards_np.min():.3f} - {rewards_np.max():.3f}"
            )
        else:
            bt.logging.warning(f"[BATCH] No scores to update (uids={len(uids_to_update)}, rewards={len(rewards_array)})")

        # Submit votes to API for consensus tracking and batch finalization
        # TODO vote might not be a good word, maybe attestations?
        if votes:
            bt.logging.info(f"[BATCH] Preparing to submit {len(votes)} vote(s) for batch {batch_id}")
            await self._submit_hotkey_votes(batch_id, votes)
        else:
            bt.logging.warning(f"[BATCH] ✗ No votes to submit for batch {batch_id}")

        # Send calculated scores back to miners via Batchscore synapse
        await self._send_scores_to_miners(str(batch_id))

        bt.logging.info(f"[BATCH] ========== Completed batch {batch_id} ==========")


    async def _send_scores_to_miners(self, batch_id: str, timeout: float = 8.0):
        """
        Send scores back to miners via Batchscore synapse.
        
        For each miner that was graded in the batch, sends a Batchscore synapse
        with the batch_id and avg_score (API-calculated average of ALL posts).
        For VALID batches, sends the API-provided avg_score_all_posts.
        For INVALID batches, sends 0.0.
        
        Args:
            batch_id: Batch identifier
            timeout: Timeout for the query in seconds
        """
        if not self._miner_scores:
            bt.logging.debug("[BatchScore] No miner scores to send")
            return
        
        bt.logging.info(f"[BatchScore] Sending scores back to {len(self._miner_scores)} miner(s) for batch_id={batch_id}")
        
        # Map hotkeys to axons for sending scores
        miner_entries = []  # List of (hotkey, axon, score) tuples
        
        for hotkey, score in self._miner_scores.items():
            try:
                uid = self.metagraph.hotkeys.index(hotkey)
                axon = self.metagraph.axons[uid]
                miner_entries.append((hotkey, axon, score))
                bt.logging.debug(f"[BatchScore] Will send score {score:.3f} to miner {hotkey} (UID {uid})")
            except ValueError:
                bt.logging.warning(f"[BatchScore] Hotkey {hotkey} not found in metagraph, skipping score send")
        
        if not miner_entries:
            bt.logging.warning("[BatchScore] No valid axons found to send scores to")
            return
        
        # Create and send synapses with scores for each miner individually
        # Since each miner gets a different score, we send them individually
        async def send_score_to_miner(hotkey, axon, score):
            """Send a Batchscore synapse to a single miner"""
            syn = protocol.Batchscore(batch_id=batch_id)
            syn.avg_score = score  # API-calculated average of ALL posts (or 0.0 if INVALID)
            try:
                response = await self.dendrite(
                    axons=[axon],
                    synapse=syn,
                    timeout=timeout,
                )
                if response and len(response) > 0:
                    bt.logging.debug(f"[BatchScore] Sent score {score:.3f} to {hotkey}")
                    return response[0]
                return None
            except asyncio.TimeoutError:
                bt.logging.warning(f"[BatchScore] Timeout sending score to {hotkey} (timeout={timeout}s)")
                return None
            except Exception as e:
                bt.logging.warning(f"[BatchScore] Error sending score to {hotkey}: {e}")
                return None
        
        # Send scores to all miners in parallel
        try:
            tasks = [send_score_to_miner(hotkey, axon, score) for hotkey, axon, score in miner_entries]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            successful = sum(1 for r in responses if r is not None and not isinstance(r, Exception))
            bt.logging.info(f"[BatchScore] Sent scores to {successful}/{len(tasks)} miner(s)")
        except Exception as e:
            bt.logging.error(f"[BatchScore] Error sending scores to miners: {e}", exc_info=True)
        
        # Clear scores after sending
        self._miner_scores.clear()

    async def forward(self):
        """
        Main validator forward loop.
        
        Starts the batch polling client on first invocation (once an asyncio
        event loop is available). The batch client runs independently in the
        background, processing batches as they become available.
        
        The base validator handles weight setting automatically based on epoch
        timing. Miner scores are updated as batches are processed via _on_batch().
        """
        #TODO can we just remove the await and not do these async things?
        if self._batch_task is None:
            self._batch_task = asyncio.create_task(self._batch_client.run(self._on_batch))
            bt.logging.info("[BATCH] Started batch client poller")

        # Note: Scores are sent to miners in _on_batch() after grading completes
        # No need to query here - the Batchscore synapse is sent from _on_batch()

        # Delegate to base validator forward logic
        # Miner scores are updated in _on_batch() as batches are processed
        return await forward(self)

    def close(self):
        """
        Gracefully shutdown the validator by canceling the batch polling task.
        
        Call this method when shutting down the validator to ensure the batch
        client stops cleanly.
        """
        if self._batch_task and not self._batch_task.done():
            self._batch_task.cancel()


# Entrypoint
if __name__ == "__main__":
    with Validator() as validator:
        while True:
            bt.logging.info(f"Validator running... {time.time()}")
            time.sleep(5)
