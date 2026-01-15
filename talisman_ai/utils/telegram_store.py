from enum import Enum
import time
import json
from pathlib import Path
from typing import Optional, Dict, List, Any
from talisman_ai import config
from talisman_ai.utils.api_models import TelegramMessageForScoring
from pydantic import BaseModel


class TelegramMessageStatus(Enum):
    UNPROCESSED = "Unprocessed"
    PROCESSING = "Processing"
    PROCESSED = "Processed"


class TelegramStoreItem(BaseModel):
    message: TelegramMessageForScoring
    status: TelegramMessageStatus
    start_time: Optional[float] = None
    hotkey: Optional[str] = None
    # Idempotency helpers
    submitted_to_api: bool = False
    rewarded: bool = False


class TelegramStore:
    def __init__(self):
        # key: message_id, value: TelegramStoreItem
        self._messages: Dict[str, TelegramStoreItem] = {}

    def add_message(
        self,
        message: TelegramMessageForScoring,
        message_id: Optional[str] = None,
        hotkey: Optional[str] = None,
        set_as_processing: bool = True,
        overwrite: bool = False,
    ):
        """
        Adds a telegram message to the store as Unprocessed.
        If message_id is provided, uses it as the key; else uses message.id.
        
        Args:
            message: TelegramMessageForScoring object to store
            message_id: Optional message ID. If not provided, uses message.id
            hotkey: Optional miner hotkey processing this message
            overwrite: If True, overwrite an existing entry. Defaults to False.
        """
        if message_id is None:
            message_id = message.id
        # Normalize keys so persistence round-trips correctly (JSON object keys are strings).
        message_id = str(message_id)
        # Ensure message is a TelegramMessageForScoring instance
        if not isinstance(message, TelegramMessageForScoring):
            raise TypeError(f"message must be a TelegramMessageForScoring instance, got {type(message)}")
        if (message_id in self._messages) and (not overwrite):
            # Preserve existing lifecycle/idempotency flags; optionally update message/hotkey if missing.
            existing = self._messages[message_id]
            if existing.message is None:
                existing.message = message
            # Only fill hotkey if not already set.
            if hotkey is not None and existing.hotkey is None:
                existing.hotkey = hotkey
            return
        self._messages[message_id] = TelegramStoreItem(
            message=message,
            status=TelegramMessageStatus.PROCESSING if set_as_processing else TelegramMessageStatus.UNPROCESSED,
            start_time=None,
            hotkey=hotkey,
            submitted_to_api=False,
            rewarded=False,
        )

    def update_message(self, message_id: str, message: TelegramMessageForScoring):
        """Update the stored message object (e.g. attach miner analysis) without changing lifecycle flags."""
        message_id = str(message_id)
        if message_id not in self._messages:
            raise KeyError(f"Message ID {message_id} not found")
        self._messages[message_id].message = message

    def set_processing(self, message_id: str, hotkey: Optional[str] = None):
        """
        Sets the message as Processing and stores the current time as start_time.
        
        Args:
            message_id: ID of the message to set as processing
            hotkey: Optional miner hotkey processing this message
        """
        message_id = str(message_id)
        if message_id in self._messages:
            self._messages[message_id].status = TelegramMessageStatus.PROCESSING
            self._messages[message_id].start_time = time.time()
            if hotkey is not None:
                self._messages[message_id].hotkey = hotkey
        else:
            raise KeyError(f"Message ID {message_id} not found")

    def set_processed(self, message_id: str):
        """
        Sets the message as Processed and clears start_time.
        """
        message_id = str(message_id)
        if message_id in self._messages:
            self._messages[message_id].status = TelegramMessageStatus.PROCESSED
            self._messages[message_id].start_time = None
        else:
            raise KeyError(f"Message ID {message_id} not found")

    def mark_submitted(self, message_id: str):
        """Mark a processed message as successfully submitted to the API."""
        message_id = str(message_id)
        if message_id not in self._messages:
            raise KeyError(f"Message ID {message_id} not found")
        self._messages[message_id].submitted_to_api = True

    def mark_rewarded(self, message_id: str):
        """Mark a message as having already contributed reward to a miner."""
        message_id = str(message_id)
        if message_id not in self._messages:
            raise KeyError(f"Message ID {message_id} not found")
        self._messages[message_id].rewarded = True

    def is_rewarded(self, message_id: str) -> bool:
        message_id = str(message_id)
        if message_id not in self._messages:
            return False
        return bool(self._messages[message_id].rewarded)

    def get_ready_to_submit(self) -> List[TelegramStoreItem]:
        """Return processed messages that have not yet been submitted to the API."""
        return [m for m in self._messages.values() if m.status == TelegramMessageStatus.PROCESSED and not m.submitted_to_api]

    def get_status(self, message_id: str) -> TelegramMessageStatus:
        """
        Returns the current status of the message.
        """
        message_id = str(message_id)
        if message_id in self._messages:
            return self._messages[message_id].status
        else:
            raise KeyError(f"Message ID {message_id} not found")

    def get_message(self, message_id: str) -> TelegramMessageForScoring:
        """
        Returns the stored message object.
        
        Returns:
            TelegramMessageForScoring: The stored message object
        """
        message_id = str(message_id)
        if message_id in self._messages:
            return self._messages[message_id].message
        else:
            raise KeyError(f"Message ID {message_id} not found")

    def get_all(self, status: TelegramMessageStatus = None) -> List[TelegramStoreItem]:
        """
        Returns a list of all messages.
        If status is given, filters by that status.
        """
        if status is None:
            return list(self._messages.values())
        return [info for info in self._messages.values() if info.status == status]
    
    def get_hotkey(self, message_id: str) -> Optional[str]:
        """
        Returns the hotkey of the miner processing the message, if set.
        
        Returns:
            Optional[str]: The miner hotkey, or None if not set
        """
        message_id = str(message_id)
        if message_id in self._messages:
            return self._messages[message_id].hotkey
        else:
            raise KeyError(f"Message ID {message_id} not found")
    
    def get_messages_by_hotkey(self, hotkey: str, status: Optional[TelegramMessageStatus] = None) -> List[TelegramStoreItem]:
        """
        Returns list of message items processed by the given hotkey.
        
        Args:
            hotkey: Miner hotkey to filter by
            status: Optional status to filter by. If None, returns messages of all statuses.
        
        Returns:
            List of TelegramStoreItem matching the hotkey (and optionally status)
        """
        result = []
        for msg_info in self._messages.values():
            if msg_info.hotkey == hotkey:
                if status is None or msg_info.status == status:
                    result.append(msg_info)
        return result

    def get_timeouts(self) -> List[TelegramStoreItem]:
        """
        Returns list of messages that are in Processing and have been processing longer than config.MESSAGE_MAX_PROCESS_TIME seconds.
        """
        now = time.time()
        # Use MESSAGE_MAX_PROCESS_TIME with fallback to TWEET_MAX_PROCESS_TIME for backward compatibility
        max_time = getattr(config, 'MESSAGE_MAX_PROCESS_TIME', getattr(config, 'TWEET_MAX_PROCESS_TIME', 300.0))
        result = []
        for m in self._messages.values():
            if (
                m.status == TelegramMessageStatus.PROCESSING
                and m.start_time is not None
                and (now - m.start_time) > max_time
            ):
                result.append(m)
        return result

    def reset_to_unprocessed(self, message_id: str):
        """
        Resets status to Unprocessed and clears start_time.
        Note: hotkey is preserved when resetting.
        """
        message_id = str(message_id)
        if message_id in self._messages:
            self._messages[message_id].status = TelegramMessageStatus.UNPROCESSED
            self._messages[message_id].start_time = None
        else:
            raise KeyError(f"Message ID {message_id} not found")
    
    def get_processed_messages(self) -> List[TelegramStoreItem]:
        """
        Returns list of messages that are in Processed.
        """
        return [m for m in self._messages.values() if m.status == TelegramMessageStatus.PROCESSED]

    def get_unprocessed_messages(self) -> List[TelegramStoreItem]:
        """
        Returns list of TelegramStoreItem that are in Unprocessed.
        """
        return [m for m in self._messages.values() if m.status == TelegramMessageStatus.UNPROCESSED]
    
    def get_processing_messages(self) -> List[TelegramStoreItem]:
        """
        Returns list of messages that are in Processing.
        """
        return [m for m in self._messages.values() if m.status == TelegramMessageStatus.PROCESSING]

    def delete_processed_messages(self):
        """
        Deletes all messages that are in Processed.
        """
        message_ids_to_delete = [
            message_id for message_id, m in self._messages.items()
            if m.status == TelegramMessageStatus.PROCESSED
        ]
        for message_id in message_ids_to_delete:
            del self._messages[message_id]

    def delete_submitted_messages(self):
        """Deletes all messages which have been submitted_to_api=True."""
        message_ids_to_delete = [message_id for message_id, m in self._messages.items() if bool(m.submitted_to_api)]
        for message_id in message_ids_to_delete:
            del self._messages[message_id]

    def delete_message(self, message_id: str):
        """
        Deletes a message from the store.
        """
        message_id = str(message_id)
        if message_id in self._messages:
            del self._messages[message_id]
        else:
            raise KeyError(f"Message ID {message_id} not found")

    def prune_old_messages(self, max_age_seconds: float = 3600, max_messages: int = 1000):
        """
        Prune old messages to prevent unbounded memory growth.
        
        Removes:
        1. All submitted messages (submitted_to_api=True)
        2. Unprocessed messages older than max_age_seconds
        3. If still over max_messages, remove oldest messages
        
        Args:
            max_age_seconds: Maximum age for unprocessed messages (default: 1 hour)
            max_messages: Maximum number of messages to keep (default: 1000)
        """
        now = time.time()
        
        # First pass: remove submitted and old unprocessed messages
        message_ids_to_delete = []
        for message_id, item in self._messages.items():
            # Remove submitted messages
            if item.submitted_to_api:
                message_ids_to_delete.append(message_id)
                continue
            # Remove old unprocessed messages
            if item.status == TelegramMessageStatus.UNPROCESSED:
                # Use start_time if available, otherwise check if message is older than max_age
                if item.start_time is not None and (now - item.start_time) > max_age_seconds:
                    message_ids_to_delete.append(message_id)
        
        for message_id in message_ids_to_delete:
            del self._messages[message_id]
        
        # Second pass: if still over limit, remove oldest messages
        if len(self._messages) > max_messages:
            # Sort by start_time (oldest first), keeping None at the end
            sorted_items = sorted(
                self._messages.items(),
                key=lambda x: x[1].start_time if x[1].start_time is not None else float('inf')
            )
            # Keep only the newest max_messages
            to_remove = len(self._messages) - max_messages
            for message_id, _ in sorted_items[:to_remove]:
                del self._messages[message_id]

    def get_message_by_id(self, message_id: str) -> Optional[TelegramMessageForScoring]:
        """
        Returns the message with the given ID, if it exists.
        """
        message_id = str(message_id)
        if message_id in self._messages:
            return self._messages[message_id].message
        else:
            return None
    
    def save_to_file(self, file_path: Optional[str] = None):
        """
        Saves the telegram store to a JSON file.
        
        Args:
            file_path: Path to the file. Defaults to config.TELEGRAM_STORE_LOCATION
        """
        if file_path is None:
            file_path = getattr(config, 'TELEGRAM_STORE_LOCATION', '.telegram_store.json')
        
        file_path = Path(file_path)
        # Create parent directories if they don't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for serialization
        data = {
            "messages": {}
        }
        
        for message_id, msg_info in self._messages.items():
            # Serialize TelegramMessageForScoring to dict, then add status, start_time, and hotkey
            data["messages"][message_id] = msg_info.model_dump(mode='json')
        
        # Write to file
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def load_from_file(self, file_path: Optional[str] = None):
        """
        Loads the telegram store from a JSON file.
        
        Args:
            file_path: Path to the file. Defaults to config.TELEGRAM_STORE_LOCATION
        """
        if file_path is None:
            file_path = getattr(config, 'TELEGRAM_STORE_LOCATION', '.telegram_store.json')
        
        file_path = Path(file_path)
        
        # If file doesn't exist, start with empty store
        if not file_path.exists():
            self._messages = {}
            return
        
        # Read from file
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Clear existing messages
        self._messages = {}
        
        # Deserialize messages
        for message_id, msg_info in data.get("messages", {}).items():
            message_id = str(message_id)
            # Reconstruct TelegramMessageForScoring from dict
            message = TelegramMessageForScoring.model_validate(msg_info["message"])
            # Reconstruct status enum
            status = TelegramMessageStatus(msg_info["status"])
            start_time = msg_info.get("start_time")
            hotkey = msg_info.get("hotkey")  # May be None for older saved files
            submitted_to_api = bool(msg_info.get("submitted_to_api", False))
            rewarded = bool(msg_info.get("rewarded", False))
            
            self._messages[message_id] = TelegramStoreItem(
                message=message,
                status=status,
                start_time=start_time,
                hotkey=hotkey,
                submitted_to_api=submitted_to_api,
                rewarded=rewarded,
            )

