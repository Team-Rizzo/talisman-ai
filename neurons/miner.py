import time
import typing
import bittensor as bt

# Bittensor Miner Template:
import talisman_ai

# import base miner class which takes care of most of the boilerplate
from talisman_ai.base.miner import BaseMinerNeuron
from talisman_ai.analyzer import setup_analyzer
from talisman_ai.utils.api_models import TweetAnalysisBase


class Miner(BaseMinerNeuron):
    """
    V3 Miner: Processes TweetBatch requests from validators.
    
    The miner receives batches of tweets from validators, analyzes each tweet
    for subnet relevance and sentiment, and returns the enriched batch.
    """

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)

        # Initialize analyzer for tweet classification
        bt.logging.info("[Miner] Initializing analyzer...")
        self.analyzer = setup_analyzer()
        bt.logging.info("[Miner] Analyzer initialized")

        # IMPORTANT: Register a concrete TweetBatch handler on the axon.
        # Bittensor routes requests by synapse class name; attaching only `forward(self, bt.Synapse)`
        # registers the generic `Synapse` endpoint and does *not* register `TweetBatch`.
        self.axon.attach(
            forward_fn=self.forward_tweets,
            blacklist_fn=self.blacklist_tweet_batch,
            priority_fn=self.priority_tweet_batch,
        )
        
        hotkey = self.wallet.hotkey.ss58_address
        bt.logging.info(f"[Miner] V3 miner started with hotkey: {hotkey}")

    async def blacklist_tweet_batch(
        self, synapse: talisman_ai.protocol.TweetBatch
    ) -> typing.Tuple[bool, str]:
        """Typed wrapper so bittensor's axon signature checks pass for TweetBatch."""
        return await self.blacklist(synapse)

    async def priority_tweet_batch(self, synapse: talisman_ai.protocol.TweetBatch) -> float:
        """Typed wrapper so bittensor's axon signature checks pass for TweetBatch."""
        return await self.priority(synapse)
    
    async def forward_is_alive(self, synapse: talisman_ai.protocol.IsAlive) -> talisman_ai.protocol.IsAlive:
        """
        Processes incoming IsAlive synapses from validators.
        """
        synapse.is_alive = True
        return synapse
    
    async def forward(self, synapse: bt.Synapse) -> bt.Synapse:
        """
        Processes incoming synapses. Routes TweetBatch requests to forward_tweets.
        
        Args:
            synapse (bt.Synapse): The incoming synapse request.
            
        Returns:
            bt.Synapse: The processed synapse response.
        """
        if isinstance(synapse, talisman_ai.protocol.TweetBatch):
            return await self.forward_tweets(synapse)
        
        bt.logging.warning(f"Received synapse type: {type(synapse).__name__}, but no handler implemented")
        return synapse

    async def forward_tweets(self, synapse: talisman_ai.protocol.TweetBatch) -> talisman_ai.protocol.TweetBatch:
        """
        Processes TweetBatch requests from validators.
        
        Analyzes each tweet in the batch and enriches it with classification data.
        
        Args:
            synapse: TweetBatch containing list of tweets to analyze
            
        Returns:
            TweetBatch with enriched tweets (analysis field populated)
        """
        bt.logging.info(f"[Miner] Received TweetBatch with {len(synapse.tweet_batch)} tweet(s)")
        
        for tweet in synapse.tweet_batch:
            if not tweet.text:
                bt.logging.warning(f"[Miner] Skipping tweet {tweet.id} - no text content")
                continue
            
            # Classify the tweet
            classification = self.analyzer.classify_post(tweet.text)
            
            if classification is None:
                bt.logging.warning(f"[Miner] Failed to classify tweet {tweet.id}")
                continue
            
            # Create analysis object with required fields for validator
            tweet.analysis = TweetAnalysisBase(
                sentiment=classification.sentiment.value,
                subnet_id=classification.subnet_id,
                subnet_name=classification.subnet_name,
                content_type=classification.content_type.value,
                technical_quality=classification.technical_quality.value,
                market_analysis=classification.market_analysis.value,
                impact_potential=classification.impact_potential.value,
            )
        
        bt.logging.info(f"[Miner] Processed TweetBatch - returning enriched tweets")
        return synapse


    async def forward_score(self, synapse: talisman_ai.protocol.Score) -> talisman_ai.protocol.Score:
        """
        Processes incoming Score synapses from validators.
        
        Receives the score that the validator has given this hotkey for a 100-block interval.
        """
        block_window_start = synapse.block_window_start
        block_window_end = synapse.block_window_end
        score = synapse.score
        validator_hotkey = synapse.validator_hotkey
        bt.logging.info(
            f"[Score] Received score: {score:.6f} from validator {validator_hotkey} for block window {block_window_start}-{block_window_end}"
        )
        return synapse

    async def forward_validation_result(self, synapse: talisman_ai.protocol.ValidationResult) -> talisman_ai.protocol.ValidationResult:
        """
        Processes incoming ValidationResult synapses from validators.
        
        Receives validation results for a specific post, including whether it passed or failed and why.
        """
        validation_id = synapse.validation_id
        post_id = synapse.post_id
        success = synapse.success
        validator_hotkey = synapse.validator_hotkey
        failure_reason = synapse.failure_reason
        
        if success:
            bt.logging.info(
                f"[ValidationResult] ✓ Post {post_id} PASSED validation from validator {validator_hotkey} "
                f"(validation_id: {validation_id})"
            )
        else:
            failure_code = failure_reason.get("code", "unknown") if failure_reason else "unknown"
            failure_message = failure_reason.get("message", "Unknown error") if failure_reason else "Unknown error"
            bt.logging.warning(
                f"[ValidationResult] ✗ Post {post_id} FAILED validation from validator {validator_hotkey} "
                f"(validation_id: {validation_id}): {failure_code} - {failure_message}"
            )
        
        return synapse

    async def blacklist(
        self, synapse: bt.Synapse
    ) -> typing.Tuple[bool, str]:
        """
        Determines whether an incoming request should be blacklisted and thus ignored. Your implementation should
        define the logic for blacklisting requests based on your needs and desired security parameters.

        Blacklist runs before the synapse data has been deserialized (i.e. before synapse.data is available).
        The synapse is instead contracted via the headers of the request. It is important to blacklist
        requests before they are deserialized to avoid wasting resources on requests that will be ignored.

        Args:
            synapse (bt.Synapse): A synapse object constructed from the headers of the incoming request.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating whether the synapse's hotkey is blacklisted,
                            and a string providing the reason for the decision.

        This function is a security measure to prevent resource wastage on undesired requests. It should be enhanced
        to include checks against the metagraph for entity registration, validator status, and sufficient stake
        before deserialization of synapse data to minimize processing overhead.

        Example blacklist logic:
        - Reject if the hotkey is not a registered entity within the metagraph.
        - Consider blacklisting entities that are not validators or have insufficient stake.

        In practice it would be wise to blacklist requests from entities that are not validators, or do not have
        enough stake. This can be checked via metagraph.S and metagraph.validator_permit. You can always attain
        the uid of the sender via a metagraph.hotkeys.index( synapse.dendrite.hotkey ) call.

        Otherwise, allow the request to be processed further.
        """

        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning(
                "Received a request without a dendrite or hotkey."
            )
            return True, "Missing dendrite or hotkey"

        # TODO(developer): Define how miners should blacklist requests.
        # Check if hotkey is registered BEFORE trying to get its index
        if (
            not self.config.blacklist.allow_non_registered
            and synapse.dendrite.hotkey not in self.metagraph.hotkeys
        ):
            # Ignore requests from un-registered entities.
            bt.logging.trace(
                f"Blacklisting un-registered hotkey {synapse.dendrite.hotkey}"
            )
            return True, "Unrecognized hotkey"

        # Only get uid if hotkey is in metagraph (to avoid IndexError)
        try:
            uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        except ValueError:
            # Hotkey not found in metagraph (shouldn't happen if check above passed, but be safe)
            bt.logging.warning(f"Hotkey {synapse.dendrite.hotkey} not found in metagraph")
            return True, "Hotkey not in metagraph"

        if self.config.blacklist.force_validator_permit:
            # If the config is set to force validator permit, then we should only allow requests from validators.
            if not self.metagraph.validator_permit[uid]:
                bt.logging.warning(
                    f"Blacklisting a request from non-validator hotkey {synapse.dendrite.hotkey}"
                )
                return True, "Non-validator hotkey"

        bt.logging.trace(
            f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}"
        )
        return False, "Hotkey recognized!"

    async def priority(self, synapse: bt.Synapse) -> float:
        """
        The priority function determines the order in which requests are handled. More valuable or higher-priority
        requests are processed before others. You should design your own priority mechanism with care.

        This implementation assigns priority to incoming requests based on the calling entity's stake in the metagraph.

        Args:
            synapse (bt.Synapse): The synapse object that contains metadata about the incoming request.

        Returns:
            float: A priority score derived from the stake of the calling entity.

        Miners may receive messages from multiple entities at once. This function determines which request should be
        processed first. Higher values indicate that the request should be processed first. Lower values indicate
        that the request should be processed later.

        Example priority logic:
        - A higher stake results in a higher priority value.
        """
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning(
                "Received a request without a dendrite or hotkey."
            )
            return 0.0

        # TODO(developer): Define how miners should prioritize requests.
        caller_uid = self.metagraph.hotkeys.index(
            synapse.dendrite.hotkey
        )  # Get the caller index.
        priority = float(
            self.metagraph.S[caller_uid]
        )  # Return the stake as the priority.
        bt.logging.trace(
            f"Prioritizing {synapse.dendrite.hotkey} with value: {priority}"
        )
        return priority


    def __exit__(self, exc_type, exc_value, traceback):
        """Clean up when miner exits."""
        super().__exit__(exc_type, exc_value, traceback)
        return False


# This is the main function, which runs the miner.
if __name__ == "__main__":
    with Miner() as miner:
        while True:
            bt.logging.info(f"Miner running... {time.time()}")
            time.sleep(5)
