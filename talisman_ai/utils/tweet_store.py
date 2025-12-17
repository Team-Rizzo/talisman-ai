from enum import Enum
import time
import json
from pathlib import Path
from typing import Optional
from talisman_ai import config
from talisman_ai.utils.api_models import TweetWithUser
from pydantic import BaseModel
from typing import List
class TweetStatus(Enum):
    UNPROCESSED = "Unprocessed"
    PROCESSING = "Processing"
    PROCESSED = "Processed"

class TweetStoreItem(BaseModel):
    tweet: TweetWithUser
    status: TweetStatus
    start_time: Optional[float] = None
    hotkey: Optional[str] = None

class TweetStore:
    def __init__(self):
        # key: tweet_id, value: dict with 'tweet' (TweetWithUser), 'status', 'start_time', 'hotkey'
        self._tweets = {}

    def add_tweet(self, tweet: TweetWithUser, tweet_id: Optional[str] = None, hotkey: Optional[str] = None, set_as_processing: bool = True):
        """
        Adds a tweet to the store as Unprocessed.
        If tweet_id is provided, uses it as the key; else uses tweet.id.
        
        Args:
            tweet: TweetWithUser object to store
            tweet_id: Optional tweet ID. If not provided, uses tweet.id
            hotkey: Optional miner hotkey processing this tweet
        """
        if tweet_id is None:
            tweet_id = tweet.id
        # Ensure tweet is a TweetWithUser instance
        if not isinstance(tweet, TweetWithUser):
            raise TypeError(f"tweet must be a TweetWithUser instance, got {type(tweet)}")
        self._tweets[tweet_id] = TweetStoreItem(
            tweet=tweet, 
            status=TweetStatus.PROCESSING if set_as_processing else TweetStatus.UNPROCESSED, 
            start_time=None,
            hotkey=hotkey
        )

    def set_processing(self, tweet_id, hotkey: Optional[str] = None):
        """
        Sets the tweet as Processing and stores the current time as start_time.
        
        Args:
            tweet_id: ID of the tweet to set as processing
            hotkey: Optional miner hotkey processing this tweet
        """
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
        if tweet_id in self._tweets:
            self._tweets[tweet_id].status = TweetStatus.PROCESSED
            self._tweets[tweet_id].start_time = None
        else:
            raise KeyError(f"Tweet ID {tweet_id} not found")

    def get_status(self, tweet_id):
        """
        Returns the current status of the tweet.
        """
        if tweet_id in self._tweets:
            return self._tweets[tweet_id].status
        else:
            raise KeyError(f"Tweet ID {tweet_id} not found")

    def get_tweet(self, tweet_id) -> TweetWithUser:
        """
        Returns the stored tweet object.
        
        Returns:
            TweetWithUser: The stored tweet object
        """
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

    def delete_tweet(self, tweet_id):
        """
        Deletes a tweet from the store.
        """
        if tweet_id in self._tweets:
            del self._tweets[tweet_id]
        else:
            raise KeyError(f"Tweet ID {tweet_id} not found")
    
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
            # Serialize TweetWithUser to dict, then add status, start_time, and hotkey
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
            # Reconstruct TweetWithUser from dict
            tweet = TweetWithUser.model_validate(tweet_info["tweet"])
            # Reconstruct status enum
            status = TweetStatus(tweet_info["status"])
            start_time = tweet_info.get("start_time")
            hotkey = tweet_info.get("hotkey")  # May be None for older saved files
            
            self._tweets[tweet_id] = TweetStoreItem(
                tweet=tweet,
                status=status,
                start_time=start_time,
                hotkey=hotkey
            )