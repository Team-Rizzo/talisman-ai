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
        self._validation_client = ValidationClient(wallet=self.wallet)
        self._validation_task: Optional[asyncio.Task] = None
        self._pending_results: List[Dict[str, Any]] = []

    async def _on_validations(self, validations: List[Dict[str, Any]]):
        """
        Process multiple validation payloads in batch (sequentially).
        
        Args:
            validations: List of validation payloads, each with validation_id, miner_hotkey, post, selected_at
        """
        if not validations:
            return
        
        bt.logging.info(f"[VALIDATION] Processing {len(validations)} validation(s) in batch")
        
        # Process all validations sequentially (one at a time)
        results = []
        validation_results_by_hotkey = {}  # Group validation results by hotkey for batching
        
        for validation in validations:
            result = await self._process_single_validation(validation, validation_results_by_hotkey)
            results.append(result)
        
        # Add all results to pending
        self._pending_results.extend(results)
        
        # Submit all results to API FIRST (before sending synapses)
        await self._submit_pending_results()
        
        # Send batched ValidationResult synapses to miners AFTER API submission
        if validation_results_by_hotkey:
            bt.logging.info(
                f"[VALIDATION] Sending ValidationResult synapses to {len(validation_results_by_hotkey)} miner(s)"
            )
            await self._send_batched_validation_results(validation_results_by_hotkey)
        else:
            bt.logging.warning("[VALIDATION] No validation results to send to miners")
    
    async def _process_single_validation(
        self, 
        validation: Dict[str, Any],
        validation_results_by_hotkey: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        Process a single validation payload.
        
        Args:
            validation: Validation payload with validation_id, miner_hotkey, post, selected_at
            validation_results_by_hotkey: Dictionary to accumulate validation results by hotkey for batching
        
        Returns:
            Result dict ready for submission
        """
        validation_id = validation.get("validation_id")
        miner_hotkey = validation.get("miner_hotkey")
        post = validation.get("post", {})
        
        bt.logging.info(f"[VALIDATION] Processing validation_id={validation_id}, miner_hotkey={miner_hotkey}")
        
        # Grade the post (run in executor to avoid blocking, reuse analyzer)
        loop = asyncio.get_event_loop()
        # Use lambda to pass analyzer as second positional argument
        label, grade_result = await loop.run_in_executor(
            None, 
            lambda: grade_hotkey([post], analyzer=self._analyzer)
        )
        
        # Determine success and failure_reason
        success = label == CONSENSUS_VALID
        failure_reason = None
        post_id = post.get("post_id", "unknown")
        
        if not success:
            error_info = grade_result.get("error", {})
            failure_reason = {
                "code": error_info.get("code", "unknown_error"),
                "message": error_info.get("message", "Unknown error"),
                "post_id": error_info.get("post_id", post_id),
                "details": error_info.get("details", {})
            }
            bt.logging.warning(
                f"[VALIDATION] ✗ Validation FAILED for {miner_hotkey}: "
                f"{failure_reason['code']} - {failure_reason['message']}"
            )
        else:
            bt.logging.info(f"[VALIDATION] ✓ Validation PASSED for {miner_hotkey}")
        
        # Accumulate validation result for batching (instead of sending immediately)
        if miner_hotkey not in validation_results_by_hotkey:
            validation_results_by_hotkey[miner_hotkey] = []
        
        validation_results_by_hotkey[miner_hotkey].append({
            "validation_id": validation_id,
            "post_id": post_id,
            "success": success,
            "failure_reason": failure_reason
        })
        
        # Return result dict
        return {
            "validator_hotkey": str(self.wallet.hotkey.ss58_address),
            "validation_id": validation_id,
            "miner_hotkey": miner_hotkey,
            "success": success,
            "failure_reason": failure_reason,
        }

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
        
        # Build hotkey→UID mapping from current metagraph to avoid race conditions
        # This prevents issues if metagraph updates between score fetch and processing
        hotkey_to_uid = {hotkey: uid for uid, hotkey in enumerate(self.metagraph.hotkeys)}
        
        uids_to_update = []
        rewards_array = []
        
        for hotkey, score in scores.items():
            uid = hotkey_to_uid.get(hotkey)
            if uid is not None:
                # Validate score range before applying
                try:
                    score_float = float(score)
                    if not (0.0 <= score_float <= 1.0):
                        bt.logging.warning(
                            f"[SCORES] Invalid score {score_float} for hotkey {hotkey[:8]}..., "
                            f"clamping to [0.0, 1.0]"
                        )
                        score_float = max(0.0, min(1.0, score_float))
                    
                    if np.isnan(score_float) or np.isinf(score_float):
                        bt.logging.error(
                            f"[SCORES] NaN/Inf score for hotkey {hotkey[:8]}..., setting to 0.0"
                        )
                        score_float = 0.0
                    
                    uids_to_update.append(uid)
                    rewards_array.append(score_float)
                except (ValueError, TypeError) as e:
                    bt.logging.error(f"[SCORES] Failed to process score for {hotkey[:8]}...: {e}")
            else:
                bt.logging.debug(f"[SCORES] Hotkey {hotkey[:8]}... not found in metagraph, skipping")
        
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
                    on_validations=self._on_validations,
                    on_scores=self._on_scores,
                )
            )
            bt.logging.info("[VALIDATION] Started validation client")

        self.save_state()
        return await forward(self)


# Entrypoint
if __name__ == "__main__":
    with Validator() as validator:
        while True:
            bt.logging.info(f"Validator running... {time.time()}")
            time.sleep(5)
