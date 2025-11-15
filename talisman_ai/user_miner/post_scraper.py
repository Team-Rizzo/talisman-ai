"""
Scraper: fetches posts from X/Twitter API using tweepy.
"""

import bittensor as bt
import random
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional
import tweepy

from talisman_ai import config


class PostScraper:
    def __init__(self):
        self.post_count = 0
        self.tweets: List[Dict] = []
        self.seen_tweet_ids: set[str] = set()  # Track tweets we've already returned
        self.client: Optional[tweepy.Client] = None
        
        # ========================================================================
        # CONFIGURABLE KEYWORDS: Modify this list to change search terms
        # ========================================================================
        self.keywords = ["bittensor"]
        
        self._init_client()
        self._fetch_tweets()

    def _init_client(self):
        """Initialize the X API client."""
        bearer_token = config.X_BEARER_TOKEN
        if bearer_token == "null" or not bearer_token:
            bt.logging.warning("[PostScraper] X_BEARER_TOKEN not configured; X API unavailable.")
            return
        
        try:
            self.client = tweepy.Client(bearer_token=bearer_token)
            bt.logging.info("[PostScraper] X API client initialized")
        except Exception as e:
            bt.logging.error(f"[PostScraper] Failed to initialize X API client: {e}")
            self.client = None

    def _fetch_tweets(self):
        """Fetch tweets from X API."""
        if not self.client:
            bt.logging.warning("[PostScraper] No X API client available; skipping fetch.")
            return

        try:
            # Validate MAX_RESULTS (X API v2 requires 10-100)
            max_results = config.MAX_RESULTS
            if max_results < 10:
                bt.logging.warning(f"[PostScraper] MAX_RESULTS ({max_results}) is below X API minimum (10), using 10")
                max_results = 10
            elif max_results > 100:
                bt.logging.warning(f"[PostScraper] MAX_RESULTS ({max_results}) exceeds X API maximum (100), using 100")
                max_results = 100
            
            # Build search query from keywords
            query = " OR ".join(self.keywords) + " -is:retweet lang:en"
            start_time = datetime.now(timezone.utc) - timedelta(hours=72)
            
            bt.logging.info(f"[PostScraper] Fetching tweets with keywords: {self.keywords}")
            bt.logging.debug(f"[PostScraper] Query: {query}")
            
            # Search for tweets with all necessary fields and expansions
            response = self.client.search_recent_tweets(
                query=query,
                start_time=start_time.isoformat().replace("+00:00", "Z"),
                max_results=max_results,  # Fetch up to MAX_RESULTS tweets at once (clamped to 10-100)
                expansions=["author_id"],  # Expand to get user information
                tweet_fields=["created_at", "public_metrics", "text"],
                user_fields=["username", "name", "created_at", "public_metrics"]
            )
            
            if not response.data:
                bt.logging.warning("[PostScraper] No tweets found from X API")
                return
            
            # Map users by ID for easy lookup
            users = {}
            if response.includes and "users" in response.includes:
                users = {user.id: user for user in response.includes["users"]}
            
            # Convert tweets to our internal format
            new_tweets = []
            for tweet in response.data:
                author = users.get(tweet.author_id) if tweet.author_id else None
                
                # Calculate account age in days
                account_age_days = 0
                if author and author.created_at:
                    delta = datetime.now(timezone.utc) - author.created_at
                    account_age_days = delta.days
                
                # Get metrics
                public_metrics = tweet.public_metrics or {}
                author_metrics = author.public_metrics or {} if author else {}
                
                # Convert timestamp to unix timestamp
                timestamp = 0
                if tweet.created_at:
                    timestamp = int(tweet.created_at.timestamp())
                
                tweet_id = str(tweet.id)
                # Skip if we've already seen this tweet
                if tweet_id in self.seen_tweet_ids:
                    continue
                
                post_data = {
                    "id": tweet_id,
                    "content": tweet.text or "",
                    "author": author.username if author else "unknown",
                    "author_name": author.name if author else "Unknown User",
                    "timestamp": timestamp,
                    "account_age": account_age_days,
                    "retweets": public_metrics.get("retweet_count", 0),
                    "likes": public_metrics.get("like_count", 0),
                    "responses": public_metrics.get("reply_count", 0),
                    "followers": author_metrics.get("followers_count", 0),
                }
                new_tweets.append(post_data)
                self.seen_tweet_ids.add(tweet_id)
            
            # Append new tweets to existing pool
            self.tweets.extend(new_tweets)
            bt.logging.info(f"[PostScraper] Fetched {len(new_tweets)} new tweets (total: {len(self.tweets)})")
            
        except Exception as e:
            bt.logging.error(f"[PostScraper] Failed to fetch tweets from X API: {e}")
            self.tweets = []

    def scrape_posts(self, count: int = None) -> List[Dict]:
        """
        Returns a random sample of posts from the fetched tweets.
        If we're running low on tweets, fetch more from the API.
        
        Args:
            count: Number of posts to return. If None, returns random.randint(1, 5)
        
        Returns:
            List of post dictionaries in the expected format
        """
        # Refresh tweets if we're running low or have none
        # Dynamic threshold: refetch when pool is less than 2x POSTS_PER_SCRAPE (min 5)
        # This ensures we have enough tweets for the next cycle
        refetch_threshold = max(5, config.POSTS_PER_SCRAPE * 2)
        
        # Warn if MAX_RESULTS is too small relative to threshold (could cause excessive refetching)
        if config.MAX_RESULTS < refetch_threshold:
            bt.logging.warning(
                f"[PostScraper] MAX_RESULTS ({config.MAX_RESULTS}) is less than refetch threshold ({refetch_threshold}). "
                f"Consider increasing MAX_RESULTS to at least {refetch_threshold} to avoid excessive API calls."
            )
        
        if len(self.tweets) < refetch_threshold:
            bt.logging.debug(f"[PostScraper] Running low on tweets ({len(self.tweets)} < {refetch_threshold}), fetching more...")
            self._fetch_tweets()
        
        if not self.tweets:
            bt.logging.debug("[PostScraper] No tweets available")
            return []

        num_posts = count or random.randint(1, 5)
        # Sample randomly from available tweets
        selected = random.sample(self.tweets, min(num_posts, len(self.tweets)))
        
        # Remove selected tweets from our pool to avoid returning duplicates
        # (The miner also tracks seen_post_ids, but this helps at the scraper level)
        selected_ids = {post["id"] for post in selected}
        self.tweets = [t for t in self.tweets if t["id"] not in selected_ids]
        
        # Increment post count for tracking
        self.post_count += len(selected)
        
        return selected
