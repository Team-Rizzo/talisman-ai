# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Set your name
# Copyright © 2023 <your name>

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import bittensor as bt
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from talisman_ai.utils.api_models import TweetWithAuthor

class Score(bt.Synapse):
    """
    Synapse for sending scores from validator to miner.
    
    The validator sends this synapse to inform miners of their score for a 100-block interval.
    The miner receives the block_window_start, block_window_end, their score for that interval, and the validator hotkey.
    """
    # Input set by validator - start block of the 100-block interval
    block_window_start: int

    # Input set by validator - end block of the 100-block interval
    block_window_end: int

    # Input set by validator - score for this hotkey in the 100-block interval
    score: float

    # Input set by validator - hotkey of the validator sending this score
    validator_hotkey: str
    
class IsAlive(bt.Synapse):
    """
    Synapse for sending is alive signal from miner to validator.
    """
    is_alive: bool 
    
class TweetBatch(bt.Synapse):
    """
    Synapse for sending tweet batch from miner to validator.
    """
    tweet_batch: List[TweetWithAuthor] 


class ValidatorRewards(bt.Synapse):
    """
    Synapse for broadcasting validator-scored rewards to other validators.

    Rewards are sent as a compact mapping of miner UID -> integer points for a single epoch.
    This avoids the 128-byte limitation of knowledge commitments and supports moderate volume.

    Fields:
      - epoch: scoring epoch index (e.g., block//BLOCK_LENGTH)
      - uid_points: { miner_uid: points }
      - sender_hotkey: validator hotkey broadcasting this data
      - seq: monotonically increasing sequence number per sender (we use epoch by default)
    """
    epoch: int
    uid_points: Dict[int, int]
    sender_hotkey: str
    seq: int


class ValidatorPenalties(bt.Synapse):
    """
    Synapse for broadcasting validator-scored penalties to other validators.

    Penalties are sent as a compact mapping of miner UID -> penalty count for a single epoch.
    Miners with penalties will have their rewards zeroed out.

    Fields:
      - epoch: scoring epoch index (e.g., block//BLOCK_LENGTH)
      - uid_penalties: { miner_uid: penalty_count }
      - sender_hotkey: validator hotkey broadcasting this data
      - seq: monotonically increasing sequence number per sender (we use epoch by default)
    """
    epoch: int
    uid_penalties: Dict[int, int]
    sender_hotkey: str
    seq: int

class ValidationResult(bt.Synapse):
    """
    Synapse for sending validation results from validator to miner.
    
    The validator sends this synapse to inform miners about the validation result of a specific post.
    The miner receives information about which post was validated, whether it passed or failed, and why.
    """
    # Input set by validator - validation identifier from the API
    validation_id: str

    # Input set by validator - ID of the post that was validated
    post_id: str

    # Input set by validator - whether the validation passed (True) or failed (False)
    success: bool

    # Input set by validator - hotkey of the validator sending this result
    validator_hotkey: str

    # Input set by validator - failure reason if success=False, None if success=True
    failure_reason: Optional[Dict[str, Any]] = None