from enum import Enum
import time
import json
from pathlib import Path
from typing import Optional, Dict, List, Any
from talisman_ai import config
from talisman_ai.utils.api_models import TweetWithAuthor
from pydantic import BaseModel

class TweetStatus(Enum):
    UNPROCESSED = "Unprocessed"
    PROCESSING = "Processing"
    PROCESSED = "Processed"

class TweetStoreItem(BaseModel):
    tweet: TweetWithAuthor
    status: TweetStatus
    start_time: Optional[float] = None
    hotkey: Optional[str] = None
    # Idempotency helpers
    submitted_to_api: bool = False
    rewarded: bool = False

class TweetStore:
    def __init__(self):
        # key: tweet_id, value: dict with 'tweet' (TweetWithAuthor), 'status', 'start_time', 'hotkey'
        self._tweets: Dict[Any, TweetStoreItem] = {}

    def add_tweet(
        self,
        tweet: TweetWithAuthor,
        tweet_id: Optional[str] = None,
        hotkey: Optional[str] = None,
        set_as_processing: bool = True,
        overwrite: bool = False,
    ):
        """
        Adds a tweet to the store as Unprocessed.
        If tweet_id is provided, uses it as the key; else uses tweet.id.
        
        Args:
            tweet: TweetWithAuthor object to store
            tweet_id: Optional tweet ID. If not provided, uses tweet.id
            hotkey: Optional miner hotkey processing this tweet
            overwrite: If True, overwrite an existing entry. Defaults to False.
        """
        if tweet_id is None:
            tweet_id = tweet.id
        # Normalize keys so persistence round-trips correctly (JSON object keys are strings).
        tweet_id = str(tweet_id)
        # Ensure tweet is a TweetWithAuthor instance
        if not isinstance(tweet, TweetWithAuthor):
            raise TypeError(f"tweet must be a TweetWithAuthor instance, got {type(tweet)}")
        if (tweet_id in self._tweets) and (not overwrite):
            # Preserve existing lifecycle/idempotency flags; optionally update tweet/hotkey if missing.
            existing = self._tweets[tweet_id]
            if existing.tweet is None:
                existing.tweet = tweet
            # Only fill hotkey if not already set.
            if hotkey is not None and existing.hotkey is None:
                existing.hotkey = hotkey
            return
        self._tweets[tweet_id] = TweetStoreItem(
            tweet=tweet,
            status=TweetStatus.PROCESSING if set_as_processing else TweetStatus.UNPROCESSED,
            start_time=None,
            hotkey=hotkey,
            submitted_to_api=False,
            rewarded=False,
        )

    def update_tweet(self, tweet_id, tweet: TweetWithAuthor):
        """Update the stored tweet object (e.g. attach miner analysis) without changing lifecycle flags."""
        tweet_id = str(tweet_id)
        if tweet_id not in self._tweets:
            raise KeyError(f"Tweet ID {tweet_id} not found")
        self._tweets[tweet_id].tweet = tweet

    def set_processing(self, tweet_id, hotkey: Optional[str] = None):
        """
        Sets the tweet as Processing and stores the current time as start_time.
        
        Args:
            tweet_id: ID of the tweet to set as processing
            hotkey: Optional miner hotkey processing this tweet
        """
        tweet_id = str(tweet_id)
        if tweet_id in self._tweets:
            self._tweets[tweet_id].status = TweetStatus.PROCESSING
            self._tweets[tweet_id].start_time = time.time()
            if hotkey is not None:
                self._tweets[tweet_id].hotkey = hotkey
        else:
            raise KeyError(f"Tweet ID {tweet_id} not found")

    def set_processed(self, tweet_id):
        """
        Sets the tweet as Processed and clears start_time.
        """
        tweet_id = str(tweet_id)
        if tweet_id in self._tweets:
            self._tweets[tweet_id].status = TweetStatus.PROCESSED
            self._tweets[tweet_id].start_time = None
        else:
            raise KeyError(f"Tweet ID {tweet_id} not found")

    def mark_submitted(self, tweet_id):
        """Mark a processed tweet as successfully submitted to the API."""
        tweet_id = str(tweet_id)
        if tweet_id not in self._tweets:
            raise KeyError(f"Tweet ID {tweet_id} not found")
        self._tweets[tweet_id].submitted_to_api = True

    def mark_rewarded(self, tweet_id):
        """Mark a tweet as having already contributed reward to a miner."""
        tweet_id = str(tweet_id)
        if tweet_id not in self._tweets:
            raise KeyError(f"Tweet ID {tweet_id} not found")
        self._tweets[tweet_id].rewarded = True

    def is_rewarded(self, tweet_id) -> bool:
        tweet_id = str(tweet_id)
        if tweet_id not in self._tweets:
            return False
        return bool(self._tweets[tweet_id].rewarded)

    def get_ready_to_submit(self) -> List[TweetStoreItem]:
        """Return processed tweets that have not yet been submitted to the API."""
        return [t for t in self._tweets.values() if t.status == TweetStatus.PROCESSED and not t.submitted_to_api]

    def get_status(self, tweet_id):
        """
        Returns the current status of the tweet.
        """
        tweet_id = str(tweet_id)
        if tweet_id in self._tweets:
            return self._tweets[tweet_id].status
        else:
            raise KeyError(f"Tweet ID {tweet_id} not found")

    def get_tweet(self, tweet_id) -> TweetWithAuthor:
        """
        Returns the stored tweet object.
        
        Returns:
            TweetWithAuthor: The stored tweet object
        """
        tweet_id = str(tweet_id)
        if tweet_id in self._tweets:
            return self._tweets[tweet_id].tweet
        else:
            raise KeyError(f"Tweet ID {tweet_id} not found")

    def get_all(self, status: TweetStatus = None):
        """
        Returns a list of all tweets (dict with 'tweet', 'status', 'start_time', 'hotkey').
        If status is given, filters by that status.
        """
        if status is None:
            return list(self._tweets.values())
        return [info for info in self._tweets.values() if info.status == status]
    
    def get_hotkey(self, tweet_id) -> Optional[str]:
        """
        Returns the hotkey of the miner processing the tweet, if set.
        
        Returns:
            Optional[str]: The miner hotkey, or None if not set
        """
        tweet_id = str(tweet_id)
        if tweet_id in self._tweets:
            return self._tweets[tweet_id].hotkey
        else:
            raise KeyError(f"Tweet ID {tweet_id} not found")
    
    def get_tweets_by_hotkey(self, hotkey: str, status: Optional[TweetStatus] = None):
        """
        Returns list of tweet dicts processed by the given hotkey.
        
        Args:
            hotkey: Miner hotkey to filter by
            status: Optional status to filter by. If None, returns tweets of all statuses.
        
        Returns:
            List of tweet dicts matching the hotkey (and optionally status)
        """
        result = []
        for tweet_info in self._tweets.values():
            if tweet_info.hotkey == hotkey:
                if status is None or tweet_info.status == status:
                    result.append(tweet_info)
        return result

    def get_timeouts(self):
        """
        Returns list of tweet dicts that are in Processing and have been processing longer than config.TWEET_MAX_PROCESS_TIME seconds.
        """
        now = time.time()
        result = []
        for t in self._tweets.values():
            if (
                t.status == TweetStatus.PROCESSING
                and t.start_time is not None
                and (now - t.start_time) > config.TWEET_MAX_PROCESS_TIME
            ):
                result.append(t)
        return result

    def reset_to_unprocessed(self, tweet_id):
        """
        Resets status to Unprocessed and clears start_time.
        Note: hotkey is preserved when resetting.
        """
        tweet_id = str(tweet_id)
        if tweet_id in self._tweets:
            self._tweets[tweet_id].status = TweetStatus.UNPROCESSED
            self._tweets[tweet_id].start_time = None
        else:
            raise KeyError(f"Tweet ID {tweet_id} not found")
    
    def get_processed_tweets(self):
        """
        Returns list of tweet dicts that are in Processed.
        """
        return [t for t in self._tweets.values() if t.status == TweetStatus.PROCESSED]

    def get_unprocessed_tweets(self):
        """
        Returns list of TweetStoreItem that are in Unprocessed.
        """
        return [t for t in self._tweets.values() if t.status == TweetStatus.UNPROCESSED]
    
    def get_processing_tweets(self):
        """
        Returns list of tweet dicts that are in Processing.
        """
        return [t for t in self._tweets.values() if t.status == TweetStatus.PROCESSING]

    def delete_processed_tweets(self):
        """
        Deletes all tweets that are in Processed.
        """
        tweet_ids_to_delete = [
            tweet_id for tweet_id, t in self._tweets.items()
            if t.status == TweetStatus.PROCESSED
        ]
        for tweet_id in tweet_ids_to_delete:
            del self._tweets[tweet_id]

    def delete_submitted_tweets(self):
        """Deletes all tweets which have been submitted_to_api=True."""
        tweet_ids_to_delete = [tweet_id for tweet_id, t in self._tweets.items() if bool(t.submitted_to_api)]
        for tweet_id in tweet_ids_to_delete:
            del self._tweets[tweet_id]

    def prune_old_tweets(self, max_age_seconds: float = 3600, max_tweets: int = 1000):
        """
        Prune old tweets to maintain bounded memory usage.
        
        Removes:
        - Tweets that have been submitted to API (already processed)
        - Tweets older than max_age_seconds that are still unprocessed
        - Oldest tweets if store exceeds max_tweets
        
        Args:
            max_age_seconds: Maximum age for unprocessed tweets (default: 1 hour)
            max_tweets: Maximum number of tweets to keep (default: 1000)
        """
        import time as _time
        now = _time.time()
        
        # First, delete all submitted tweets
        self.delete_submitted_tweets()
        
        # Delete old unprocessed tweets (likely stale/abandoned)
        tweet_ids_to_delete = []
        for tweet_id, item in self._tweets.items():
            # Skip tweets that are actively processing
            if item.status == TweetStatus.PROCESSING:
                continue
            # Delete unprocessed tweets older than max_age
            if item.start_time is not None and (now - item.start_time) > max_age_seconds:
                tweet_ids_to_delete.append(tweet_id)
        
        for tweet_id in tweet_ids_to_delete:
            del self._tweets[tweet_id]
        
        # If still too many tweets, delete oldest ones (by start_time or insertion order)
        if len(self._tweets) > max_tweets:
            # Sort by start_time (None = very old), keep newest
            sorted_items = sorted(
                self._tweets.items(),
                key=lambda x: x[1].start_time or 0
            )
            excess = len(self._tweets) - max_tweets
            for tweet_id, _ in sorted_items[:excess]:
                del self._tweets[tweet_id]

    def delete_tweet(self, tweet_id):
        """
        Deletes a tweet from the store.
        """
        tweet_id = str(tweet_id)
        if tweet_id in self._tweets:
            del self._tweets[tweet_id]
        else:
            raise KeyError(f"Tweet ID {tweet_id} not found")

    def get_tweet_by_id(self, tweet_id: str) -> Optional[TweetWithAuthor]:
        """
        Returns the tweet with the given ID, if it exists.
        """
        tweet_id = str(tweet_id)
        if tweet_id in self._tweets:
            return self._tweets[tweet_id].tweet
        else:
            return None
    
    def save_to_file(self, file_path: Optional[str] = None):
        """
        Saves the tweet store to a JSON file.
        
        Args:
            file_path: Path to the file. Defaults to config.TWEET_STORE_LOCATION
        """
        if file_path is None:
            file_path = config.TWEET_STORE_LOCATION
        
        file_path = Path(file_path)
        # Create parent directories if they don't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for serialization
        data = {
            "tweets": {}
        }
        
        for tweet_id, tweet_info in self._tweets.items():
            # Serialize TweetWithAuthor to dict, then add status, start_time, and hotkey
            data["tweets"][tweet_id] = tweet_info.model_dump(mode='json')
        
        # Write to file
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def load_from_file(self, file_path: Optional[str] = None):
        """
        Loads the tweet store from a JSON file.
        
        Args:
            file_path: Path to the file. Defaults to config.TWEET_STORE_LOCATION
        """
        if file_path is None:
            file_path = config.TWEET_STORE_LOCATION
        
        file_path = Path(file_path)
        
        # If file doesn't exist, start with empty store
        if not file_path.exists():
            self._tweets = {}
            return
        
        # Read from file
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Clear existing tweets
        self._tweets = {}
        
        # Deserialize tweets
        for tweet_id, tweet_info in data.get("tweets", {}).items():
            tweet_id = str(tweet_id)
            # Reconstruct TweetWithAuthor from dict
            tweet = TweetWithAuthor.model_validate(tweet_info["tweet"])
            # Reconstruct status enum
            status = TweetStatus(tweet_info["status"])
            start_time = tweet_info.get("start_time")
            hotkey = tweet_info.get("hotkey")  # May be None for older saved files
            submitted_to_api = bool(tweet_info.get("submitted_to_api", False))
            rewarded = bool(tweet_info.get("rewarded", False))
            
            self._tweets[tweet_id] = TweetStoreItem(
                tweet=tweet,
                status=status,
                start_time=start_time,
                hotkey=hotkey,
                submitted_to_api=submitted_to_api,
                rewarded=rewarded,
            )