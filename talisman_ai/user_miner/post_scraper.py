"""
Post scraper module for fetching posts from X/Twitter API using tweepy.

This module implements a simplified block-based approach that fetches posts on-demand
without maintaining a pool. It's designed to work with the API v2 rate limits:
- Scrapes posts every 100 blocks (aligned with rate limit windows)
- Fetches up to 5 posts per window (matching the maximum submission limit)
"""

import bittensor as bt
import random
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional
import tweepy

from talisman_ai import config
from talisman_ai.utils.misc import safe_int


class PostScraper:
    """
    Scraper for fetching posts from X/Twitter API.
    
    Fetches posts on-demand based on configured keywords. No initial fetch is performed
    at initialization - posts are fetched when requested by the miner.
    """
    
    def __init__(self):
        """
        Initialize the scraper.
        
        Sets up the X API client and configures search keywords. No posts are fetched
        at initialization - fetching happens on-demand when scrape_posts() is called.
        """
        self.client: Optional[tweepy.Client] = None
        
        # Keywords are configured via SCRAPER_KEYWORDS environment variable
        # (comma-separated list, e.g., "sn45,talismanai,sn13,sn64")
        # Defaults are set in config.py
        self.keywords = config.SCRAPER_KEYWORDS
        
        self._init_client()

    def _init_client(self):
        """
        Initialize the X API client using the bearer token from configuration.
        
        The bearer token must be configured in the environment or config file.
        If initialization fails, the scraper will not be able to fetch posts.
        """
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

    def scrape_posts(self, count: int = 5) -> List[Dict]:
        """
        Fetch posts from X API on-demand.
        
        This method fetches fresh posts each time it's called, aligned with the
        block-based rate limiting system (scrape every 100 blocks). Posts are searched
        using the configured keywords and filtered to exclude retweets and non-English posts.
        
        Args:
            count: Number of posts to fetch and return (default: 5, matching rate limit max).
                   The API requires fetching at least 10 posts, so we fetch 10+ and sample
                   down to the requested count if needed.
        
        Returns:
            List of post dictionaries in the expected format, each containing:
            - id: Post ID
            - content: Post text content
            - author: Author username
            - timestamp: Unix timestamp of post creation
            - account_age: Account age in days
            - retweets: Number of retweets
            - likes: Number of likes
            - responses: Number of replies
            - followers: Author's follower count
        """
        if not self.client:
            bt.logging.warning("[PostScraper] No X API client available; cannot fetch posts.")
            return []
        
        # Validate count: X API v2 requires 10-100 for max_results
        # We'll fetch at least 10 to meet API minimum, then sample down to count if needed
        fetch_count = max(10, count)
        if fetch_count > 100:
            bt.logging.warning(f"[PostScraper] Requested count ({count}) exceeds X API maximum (100), using 100")
            fetch_count = 100

        try:
            # Build search query from keywords
            # Exclude retweets (-is:retweet) and limit to English posts (lang:en)
            query = " OR ".join(self.keywords) + " -is:retweet lang:en"
            # Search for posts from the last 72 hours
            start_time = datetime.now(timezone.utc) - timedelta(hours=72)
            
            bt.logging.info(f"[PostScraper] Fetching up to {fetch_count} tweets with keywords: {self.keywords}")
            bt.logging.debug(f"[PostScraper] Query: {query}")
            
            # Search for tweets with all necessary fields and expansions
            # Expansions allow us to get user information (author details, metrics)
            response = self.client.search_recent_tweets(
                query=query,
                start_time=start_time.isoformat().replace("+00:00", "Z"),
                max_results=fetch_count,
                expansions=["author_id"],  # Expand to get user information
                tweet_fields=["created_at", "public_metrics", "text"],
                user_fields=["username", "name", "created_at", "public_metrics"]
            )
            
            if not response.data:
                bt.logging.warning("[PostScraper] No tweets found from X API")
                return []
            
            # Map users by ID for easy lookup when processing tweets
            users = {}
            if response.includes and "users" in response.includes:
                users = {user.id: user for user in response.includes["users"]}
            
            # Convert tweets to our internal format
            posts = []
            for tweet in response.data:
                author = users.get(tweet.author_id) if tweet.author_id else None
                
                # Calculate account age in days from account creation date
                account_age_days = 0
                if author and author.created_at:
                    delta = datetime.now(timezone.utc) - author.created_at
                    account_age_days = delta.days
                
                # Get engagement metrics from tweet and author
                public_metrics = tweet.public_metrics or {}
                author_metrics = author.public_metrics or {} if author else {}
                
                # Convert timestamp to Unix timestamp (seconds since epoch)
                timestamp = 0
                if tweet.created_at:
                    timestamp = int(tweet.created_at.timestamp())
                
                # Build post data dictionary in the expected format
                # NOTE: This format is standardized regardless of which API is used (X API or TwitterAPI.io)
                # The API validation expects this exact format, so all miners should submit in this format
                post_data = {
                    "id": str(tweet.id),
                    "content": tweet.text or "",  # Raw content - API will normalize it
                    "author": author.username if author else "unknown",  # Username only (no @ prefix)
                    "timestamp": timestamp,  # Unix timestamp in seconds
                    "account_age": account_age_days,
                    "retweets": safe_int(public_metrics.get("retweet_count", 0)),
                    "likes": safe_int(public_metrics.get("like_count", 0)),
                    "responses": safe_int(public_metrics.get("reply_count", 0)),
                    "followers": safe_int(author_metrics.get("followers_count", 0)),  # Standardized: always int
                }
                posts.append(post_data)
            
            # If we fetched more than requested, randomly sample down to count
            # This ensures we return exactly the requested number of posts
            if len(posts) > count:
                posts = random.sample(posts, count)
            
            bt.logging.info(f"[PostScraper] Fetched {len(posts)} post(s) from X API")
            return posts
            
        except Exception as e:
            bt.logging.error(f"[PostScraper] Failed to fetch tweets from X API: {e}")
            return []
