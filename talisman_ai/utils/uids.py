import random
import bittensor as bt
import numpy as np
from typing import List
from talisman_ai.protocol import IsAlive

def check_uid_availability(
    metagraph: "bt.metagraph.Metagraph", uid: int, vpermit_tao_limit: int
) -> bool:
    """Check if uid is available. The UID should be available if it is serving and has less than vpermit_tao_limit stake
    Args:
        metagraph (:obj: bt.metagraph.Metagraph): Metagraph object
        uid (int): uid to be checked
        vpermit_tao_limit (int): Validator permit tao limit
    Returns:
        bool: True if uid is available, False otherwise
    """
    # Filter non serving axons.
    if not metagraph.axons[uid].is_serving:
        return False
    # Filter validator permit > 1024 stake.
    if metagraph.validator_permit[uid]:
        if metagraph.S[uid] > vpermit_tao_limit:
            return False
    # Available otherwise.
    return True


def get_random_uids(self, k: int, exclude: List[int] = None, include: List[int] = None, is_alive: bool = False) -> np.ndarray:
    """Returns k available random uids from the metagraph.
    Args:
        k (int): Number of uids to return.
        exclude (List[int]): List of uids to exclude from the random sampling.
        include (List[int]): List of uids to include in the random sampling (always inserted if available).
        is_alive (bool): Whether to include only alive uids.
    Returns:
        uids (np.ndarray): Randomly sampled available uids.
    Notes:
        If `k` is larger than the number of available `uids`, set `k` to the number of available `uids`.
    """
    avail_uids = []
    candidate_uids = []

    # Gather available uids and optionally mark those for inclusion/exclusion
    if is_alive:
        include = get_alive_uids(self.metagraph, self.dendrite)
    for uid in range(self.metagraph.n.item()):
        uid_is_available = check_uid_availability(
            self.metagraph, uid, self.config.neuron.vpermit_tao_limit
        )
        if uid_is_available:
            avail_uids.append(uid)

    # Build the set of candidate uids
    exclude_set = set(exclude) if exclude is not None else set()
    include_set = set(include) if include is not None else set()
    # Only consider includes that are also available and not excluded
    final_include = [uid for uid in include_set if uid in avail_uids and uid not in exclude_set]
    # Other available, non-excluded, and not part of the includes
    rest_candidates = [uid for uid in avail_uids if uid not in exclude_set and uid not in final_include]

    # Calculate total candidates for sampling
    total_candidates = final_include + rest_candidates

    # If k is larger than the number of available uids, set k to the number of available uids.
    k = min(k, len(avail_uids))

    # Ensure we always include 'include' uids if possible
    if len(final_include) > k:
        # If more includes than k, just sample from includes
        selected = random.sample(final_include, k)
    else:
        # Take all includes, fill the rest from rest_candidates
        if len(rest_candidates) >= (k - len(final_include)):
            selected_rest = random.sample(rest_candidates, k - len(final_include))
        else:
            selected_rest = rest_candidates
        selected = final_include + selected_rest

    uids = np.array(selected)
    return uids

async def get_alive_uids(metagraph, dendrite) -> List[int]:
    """
    Get the list of alive miners from the metagraph.
    """
    alive_uids = []
    axons = []
    for uid in range(metagraph.n.item()):
        if metagraph.axons[uid].is_serving:
            axons.append(metagraph.axons[uid])
    responses = await dendrite.forward(
        axons=axons,
        synapse=IsAlive(is_alive=False),
        timeout=12.0,
        deserialize=True
    )
    for response in responses:
        if response.dendrite.status_code == 200:
            if response.synapse.is_alive:
                alive_uids.append(response.axon.uid)
    return alive_uids