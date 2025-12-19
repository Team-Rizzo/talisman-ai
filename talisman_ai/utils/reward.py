from typing import List, Callable, Optional
import json
from pathlib import Path
from talisman_ai import config


# TODO , before resetting the epoch we need to store the rewards for the previous epoch in the knowledge commitments
class MinerReward: 
    def __init__(self, block_length: int, block):
        """
        block is the self.block function that returns the current block number
        """
        self.block_length = block_length or config.BLOCK_LENGTH
        self.current_epoch = config.START_BLOCK // self.block_length
        self.current_epoch_old = self.current_epoch
        self.block = block  # function that returns the current block number

        # Store epoch -> {hotkey: reward}
        self.epoch_rewards = {
            self.current_epoch: {}
        }

    def _get_current_epoch(self):
        return self.block() // self.block_length

    def _resolve_epoch(self, epoch: int = None):
        """Allow epoch=-1, -2 for past epochs. None means current epoch."""
        self.update_current_epoch()
        epochs_sorted = sorted(self.epoch_rewards)
        if epoch is None:
            return self.current_epoch
        if isinstance(epoch, int) and epoch < 0:
            # Allow -1 for previous, -2 for 2 epochs ago, etc.
            index = epoch
            if abs(index) > len(epochs_sorted):
                raise IndexError(f"Epoch {epoch} out of stored range (only {len(epochs_sorted)} epochs kept).")
            return epochs_sorted[index]
        # Positive or exact epoch
        if epoch in self.epoch_rewards:
            return epoch
        raise KeyError(f"Epoch {epoch} not found in stored epochs")

    def update_current_epoch(self):
        epoch = self._get_current_epoch()
        if epoch != self.current_epoch:
            # Add new epoch
            self.epoch_rewards[epoch] = {}
            self.current_epoch_old = self.current_epoch
            self.current_epoch = epoch
    
    def delete_oldest_epoch(self):
        """Delete the oldest epoch from the rewards store."""
        if len(self.epoch_rewards) == 0:
            return
        oldest_epoch = min(self.epoch_rewards)
        del self.epoch_rewards[oldest_epoch]
        return oldest_epoch

    def add_reward(self, hotkey: str, reward: int):
        self.update_current_epoch()
        rewards = self.epoch_rewards[self.current_epoch]
        rewards[hotkey] = rewards.get(hotkey, 0) + reward

    def get_reward(self, hotkey: str, epoch: int = None):
        self.update_current_epoch()
        resolved_epoch = self._resolve_epoch(epoch)
        rewards = self.epoch_rewards.get(resolved_epoch, {})
        return rewards.get(hotkey, 0)

    def get_rewards(self, epoch: int = None):
        self.update_current_epoch()
        resolved_epoch = self._resolve_epoch(epoch)
        return dict(self.epoch_rewards.get(resolved_epoch, {}))

    def get_past_epochs(self) -> list:
        """Return a list of all stored epoch numbers, most recent last."""
        self.update_current_epoch()
        return sorted(self.epoch_rewards)

    def get_rewards_for_all_epochs(self) -> dict:
        """Return dict mapping epoch -> {hotkey: reward} for all stored epochs."""
        self.update_current_epoch()
        return {e: dict(r) for e, r in self.epoch_rewards.items()}
    
    def save_to_file(self, file_path: Optional[str] = None):
        """
        Saves the reward store to a JSON file.
        
        Note: The block function is not saved and must be provided when loading.
        
        Args:
            file_path: Path to the file. Defaults to config.REWARD_STORE_LOCATION
        """
        if file_path is None:
            file_path = config.REWARD_STORE_LOCATION
        
        file_path = Path(file_path)
        # Create parent directories if they don't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for serialization
        # Convert epoch keys to strings for JSON compatibility
        data = {
            "block_length": self.block_length,
            "current_epoch": self.current_epoch,
            "current_epoch_old": self.current_epoch_old,
            "epoch_rewards": {str(e): r for e, r in self.epoch_rewards.items()}
        }
        
        # Write to file
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

    # Backward compatibility: some callers use `.save()`.
    def save(self, file_path: Optional[str] = None):
        return self.save_to_file(file_path=file_path)
    
    def load_from_file(self, block: Callable[[], int], file_path: Optional[str] = None):
        """
        Loads the reward store from a JSON file.
        
        Args:
            block: Function that returns the current block number (required, as it cannot be serialized)
            file_path: Path to the file. Defaults to config.REWARD_STORE_LOCATION
        """
        if file_path is None:
            file_path = config.REWARD_STORE_LOCATION
        
        file_path = Path(file_path)
        
        # If file doesn't exist, initialize with defaults
        if not file_path.exists():
            self.block = block
            self.block_length = config.BLOCK_LENGTH
            self.current_epoch = config.START_BLOCK // self.block_length
            self.current_epoch_old = self.current_epoch
            self.epoch_rewards = {self.current_epoch: {}}
            return
        
        # Read from file
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Restore state
        self.block = block
        self.block_length = data.get("block_length", config.BLOCK_LENGTH)
        self.current_epoch = data.get("current_epoch", config.START_BLOCK // self.block_length)
        self.current_epoch_old = data.get("current_epoch_old", self.current_epoch)
        
        # Convert epoch keys back to integers
        epoch_rewards_raw = data.get("epoch_rewards", {})
        self.epoch_rewards = {int(e): r for e, r in epoch_rewards_raw.items()}
        
        # Ensure current epoch exists
        if self.current_epoch not in self.epoch_rewards:
            self.epoch_rewards[self.current_epoch] = {}