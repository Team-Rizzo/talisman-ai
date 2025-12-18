#!/usr/bin/env python3
"""
Talisman AI API Client for Validators.

This client provides a simple interface for validators to interact with
the Talisman AI API. It handles authentication automatically using the
validator's Bittensor wallet.

Usage:
    import bittensor as bt
    from client import TalismanAPIClient
    
    # Initialize with your validator wallet
    wallet = bt.Wallet(name="validator", hotkey="default")
    client = TalismanAPIClient(
        base_url="http://localhost:8000",
        wallet=wallet,
    )
    
    # Get tweets to score
    tweets = await client.get_unscored_tweets(limit=3)
    
    # Submit completed tweets
    await client.submit_completed_tweets([
        {"tweet_id": 123456789, "sentiment": "bullish"},
    ])
    
    # Submit rewards
    await client.submit_rewards([
        {"start_block": 100, "stop_block": 200, "hotkey": "5xxx...", "points": 1.5},
    ])
"""

import time
import logging
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass

import httpx
import bittensor as bt

from talisman_ai.utils.api_models import (
    TweetWithAuthor, Account, TweetAnalysis,
    Penalty, PenaltyCreate,
    Reward, RewardCreate,
    BlacklistedHotkey,
    TweetsForScoringResponse,
    CompletedTweetSubmission,
    SubmissionResponse,
)

logger = logging.getLogger(__name__)


class TalismanAPIError(Exception):
    """Base exception for Talisman API errors."""
    
    def __init__(self, message: str, status_code: Optional[int] = None, detail: Optional[str] = None):
        self.message = message
        self.status_code = status_code
        self.detail = detail
        super().__init__(self.message)


class AuthenticationError(TalismanAPIError):
    """Raised when authentication fails."""
    pass


class AuthorizationError(TalismanAPIError):
    """Raised when the user is not authorized (not a validator)."""
    pass


class NotFoundError(TalismanAPIError):
    """Raised when a resource is not found."""
    pass


@dataclass
class ClientConfig:
    """Configuration for the Talisman API Client."""
    base_url: str
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0


class TalismanAPIClient:
    """
    Async client for the Talisman AI API.
    
    This client handles authentication automatically and provides
    typed methods for all API endpoints.
    """
    
    def __init__(
        self,
        base_url: str,
        wallet: bt.Wallet,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize the Talisman API client.
        
        Args:
            base_url: The base URL of the API (e.g., "http://localhost:8000")
            wallet: The Bittensor wallet to use for authentication
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
            retry_delay: Delay between retries in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.wallet = wallet
        self.ss58_address = wallet.hotkey.ss58_address
        self.config = ClientConfig(
            base_url=self.base_url,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
        self._client: Optional[httpx.AsyncClient] = None
        
        logger.info(f"Initialized TalismanAPIClient for validator {self.ss58_address}")
    
    def _create_auth_message(self, timestamp: float) -> str:
        """Create a standardized authentication message."""
        return f"talisman-ai-auth:{int(timestamp)}"
    
    def _sign_message(self, message: str) -> str:
        """Sign a message with the wallet's hotkey."""
        signature = self.wallet.hotkey.sign(message)
        return signature.hex()
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """Generate authentication headers for API requests."""
        timestamp = time.time()
        message = self._create_auth_message(timestamp)
        signature = self._sign_message(message)
        
        return {
            "X-Auth-SS58Address": self.ss58_address,
            "X-Auth-Signature": signature,
            "X-Auth-Message": message,
            "X-Auth-Timestamp": str(timestamp),
            "Content-Type": "application/json",
        }
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.config.timeout,
            )
        return self._client
    
    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    def _handle_response_error(self, response: httpx.Response):
        """Handle error responses from the API."""
        status_code = response.status_code
        
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text
        
        if status_code == 401:
            raise AuthenticationError(
                "Authentication failed",
                status_code=status_code,
                detail=detail,
            )
        elif status_code == 403:
            raise AuthorizationError(
                "Not authorized - only validators can access this API",
                status_code=status_code,
                detail=detail,
            )
        elif status_code == 404:
            raise NotFoundError(
                "Resource not found",
                status_code=status_code,
                detail=detail,
            )
        else:
            raise TalismanAPIError(
                f"API request failed with status {status_code}",
                status_code=status_code,
                detail=detail,
            )
    
    async def _request(
        self,
        method: str,
        endpoint: str,
        json: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Make an authenticated request to the API.
        
        Args:
            method: HTTP method (GET, POST, DELETE, etc.)
            endpoint: API endpoint (e.g., "/tweets/unscored")
            json: JSON body for POST requests
            params: Query parameters
            
        Returns:
            Response JSON as a dictionary
        """
        client = await self._get_client()
        headers = self._get_auth_headers()
        
        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                response = await client.request(
                    method=method,
                    url=endpoint,
                    headers=headers,
                    json=json,
                    params=params,
                )
                
                if response.status_code >= 400:
                    self._handle_response_error(response)
                
                return response.json()
                
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    logger.warning(
                        f"Request failed (attempt {attempt + 1}/{self.config.max_retries}): {e}"
                    )
                    await self._sleep(self.config.retry_delay * (attempt + 1))
                    # Refresh auth headers for retry
                    headers = self._get_auth_headers()
        
        raise TalismanAPIError(
            f"Request failed after {self.config.max_retries} attempts: {last_error}"
        )
    
    async def _sleep(self, seconds: float):
        """Async sleep helper."""
        import asyncio
        await asyncio.sleep(seconds)
    
    # =========================================================================
    # Health Check
    # =========================================================================
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check if the API is healthy.
        
        Returns:
            Health status dictionary
        """
        client = await self._get_client()
        response = await client.get("/health")
        return response.json()
    
    # =========================================================================
    # Tweet Methods
    # =========================================================================
    
    async def get_unscored_tweets(self, limit: int = 3) -> List[TweetWithAuthor]:
        """
        Get tweets that haven't been scored yet.
        
        This will mark the returned tweets as "in_progress" and assign them
        to your validator hotkey.
        
        Args:
            limit: Maximum number of tweets to return (default: 3)
            
        Returns:
            List of TweetWithAuthor objects
        """
        data = await self._request("GET", "/tweets/unscored", params={"limit": limit})
        
        tweets = []
        for tweet_data in data.get("tweets", []):
            author_data = tweet_data.pop("author", None)
            analysis_data = tweet_data.pop("analysis", None)
            
            author = Account(**author_data) if author_data else None
            analysis = TweetAnalysis(**analysis_data) if analysis_data else None
            
            tweet = TweetWithAuthor(**tweet_data, author=author, analysis=analysis)
            tweets.append(tweet)
        
        return tweets
    
    async def submit_completed_tweets(
        self,
        completed_tweets: List[Union[CompletedTweetSubmission, Dict[str, Any]]],
    ) -> SubmissionResponse:
        """
        Submit completed scored tweets.
        
        Args:
            completed_tweets: List of completed tweets with tweet_id (int) and sentiment
            
        Returns:
            SubmissionResponse with success status
            
        Example:
            await client.submit_completed_tweets([
                {"tweet_id": 123456789, "sentiment": "bullish"},
                {"tweet_id": 987654321, "sentiment": "bearish"},
            ])
        """
        # Convert dicts to CompletedTweetSubmission if needed
        submissions = []
        for item in completed_tweets:
            if isinstance(item, dict):
                submissions.append(item)
            else:
                submissions.append({"tweet_id": item.tweet_id, "sentiment": item.sentiment})
        
        data = await self._request(
            "POST",
            "/tweets/completed",
            json={"completed_tweets": submissions},
        )
        
        return SubmissionResponse(**data)
    
    # =========================================================================
    # Reward Methods
    # =========================================================================
    
    async def submit_rewards(
        self,
        rewards: List[Union[RewardCreate, Dict[str, Any]]],
    ) -> SubmissionResponse:
        """
        Submit rewards for miners.
        
        Args:
            rewards: List of rewards with start_block, stop_block, hotkey, and points
            
        Returns:
            SubmissionResponse with success status
            
        Example:
            await client.submit_rewards([
                {"start_block": 100, "stop_block": 200, "hotkey": "5xxx...", "points": 1.5},
            ])
        """
        reward_dicts = []
        for item in rewards:
            if isinstance(item, dict):
                reward_dicts.append(item)
            else:
                reward_dicts.append({
                    "start_block": item.start_block,
                    "stop_block": item.stop_block,
                    "hotkey": item.hotkey,
                    "points": item.points,
                })
        
        data = await self._request(
            "POST",
            "/rewards",
            json={"rewards": reward_dicts},
        )
        
        return SubmissionResponse(**data)
    
    async def get_rewards(
        self,
        hotkey: Optional[str] = None,
        limit: int = 100,
    ) -> List[Reward]:
        """
        Get rewards, optionally filtered by hotkey.
        
        Args:
            hotkey: Optional hotkey to filter by
            limit: Maximum number of rewards to return
            
        Returns:
            List of Reward objects
        """
        params = {"limit": limit}
        if hotkey:
            params["hotkey"] = hotkey
        
        data = await self._request("GET", "/rewards", params=params)
        
        return [Reward(**r) for r in data]
    
    # =========================================================================
    # Penalty Methods
    # =========================================================================
    
    async def submit_penalties(
        self,
        penalties: List[Union[PenaltyCreate, Dict[str, str]]],
    ) -> SubmissionResponse:
        """
        Submit penalties for miners.
        
        Args:
            penalties: List of penalties with hotkey and reason
            
        Returns:
            SubmissionResponse with success status
            
        Example:
            await client.submit_penalties([
                {"hotkey": "5xxx...", "reason": "Invalid tweet submission"},
            ])
        """
        penalty_dicts = []
        for item in penalties:
            if isinstance(item, dict):
                penalty_dicts.append(item)
            else:
                penalty_dicts.append({
                    "hotkey": item.hotkey,
                    "reason": item.reason,
                })
        
        data = await self._request(
            "POST",
            "/penalties",
            json={"penalties": penalty_dicts},
        )
        
        return SubmissionResponse(**data)
    
    async def get_penalties(
        self,
        hotkey: Optional[str] = None,
        limit: int = 100,
    ) -> List[Penalty]:
        """
        Get penalties, optionally filtered by hotkey.
        
        Args:
            hotkey: Optional hotkey to filter by
            limit: Maximum number of penalties to return
            
        Returns:
            List of Penalty objects
        """
        params = {"limit": limit}
        if hotkey:
            params["hotkey"] = hotkey
        
        data = await self._request("GET", "/penalties", params=params)
        
        return [Penalty(**p) for p in data]
    
    # =========================================================================
    # Blacklist Methods
    # =========================================================================
    
    async def get_blacklisted_hotkeys(self) -> List[BlacklistedHotkey]:
        """
        Get all blacklisted hotkeys.
        
        Returns:
            List of BlacklistedHotkey objects
        """
        data = await self._request("GET", "/blacklist")
        
        return [BlacklistedHotkey(**b) for b in data]
    
    async def add_blacklisted_hotkeys(
        self,
        hotkeys: List[str],
        reason: Optional[str] = None,
    ) -> SubmissionResponse:
        """
        Add hotkeys to the blacklist.
        
        Args:
            hotkeys: List of hotkey SS58 addresses to blacklist
            reason: Optional reason for blacklisting
            
        Returns:
            SubmissionResponse with success status
            
        Example:
            await client.add_blacklisted_hotkeys(["5xxx...", "5yyy..."], reason="Spam")
        """
        payload = {"hotkeys": hotkeys}
        if reason:
            payload["reason"] = reason
        
        data = await self._request(
            "POST",
            "/blacklist",
            json=payload,
        )
        
        return SubmissionResponse(**data)
    
    async def remove_blacklisted_hotkey(self, hotkey: str) -> SubmissionResponse:
        """
        Remove a hotkey from the blacklist.
        
        Args:
            hotkey: The hotkey SS58 address to remove
            
        Returns:
            SubmissionResponse with success status
        """
        data = await self._request("DELETE", f"/blacklist/{hotkey}")
        
        return SubmissionResponse(**data)


# =============================================================================
# Synchronous Wrapper (for convenience)
# =============================================================================

class TalismanAPIClientSync:
    """
    Synchronous wrapper for TalismanAPIClient.
    
    This is a convenience class for validators who prefer synchronous code.
    It wraps the async client and runs operations in an event loop.
    
    Usage:
        wallet = bt.Wallet(name="validator", hotkey="default")
        client = TalismanAPIClientSync("http://localhost:8000", wallet)
        
        tweets = client.get_unscored_tweets(limit=3)
        client.submit_completed_tweets([{"tweet_id": 123, "sentiment": "bullish"}])
    """
    
    def __init__(
        self,
        base_url: str,
        wallet: bt.Wallet,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """Initialize the synchronous client wrapper."""
        self._async_client = TalismanAPIClient(
            base_url=base_url,
            wallet=wallet,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
    
    def _run(self, coro):
        """Run a coroutine in the event loop."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    
    def close(self):
        """Close the client."""
        self._run(self._async_client.close())
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def health_check(self) -> Dict[str, Any]:
        """Check if the API is healthy."""
        return self._run(self._async_client.health_check())
    
    def get_unscored_tweets(self, limit: int = 3) -> List[TweetWithAuthor]:
        """Get tweets that haven't been scored yet."""
        return self._run(self._async_client.get_unscored_tweets(limit))
    
    def submit_completed_tweets(
        self,
        completed_tweets: List[Union[CompletedTweetSubmission, Dict[str, Any]]],
    ) -> SubmissionResponse:
        """Submit completed scored tweets."""
        return self._run(self._async_client.submit_completed_tweets(completed_tweets))
    
    def submit_rewards(
        self,
        rewards: List[Union[RewardCreate, Dict[str, Any]]],
    ) -> SubmissionResponse:
        """Submit rewards for miners."""
        return self._run(self._async_client.submit_rewards(rewards))
    
    def get_rewards(
        self,
        hotkey: Optional[str] = None,
        limit: int = 100,
    ) -> List[Reward]:
        """Get rewards, optionally filtered by hotkey."""
        return self._run(self._async_client.get_rewards(hotkey, limit))
    
    def submit_penalties(
        self,
        penalties: List[Union[PenaltyCreate, Dict[str, str]]],
    ) -> SubmissionResponse:
        """Submit penalties for miners."""
        return self._run(self._async_client.submit_penalties(penalties))
    
    def get_penalties(
        self,
        hotkey: Optional[str] = None,
        limit: int = 100,
    ) -> List[Penalty]:
        """Get penalties, optionally filtered by hotkey."""
        return self._run(self._async_client.get_penalties(hotkey, limit))
    
    def get_blacklisted_hotkeys(self) -> List[BlacklistedHotkey]:
        """Get all blacklisted hotkeys."""
        return self._run(self._async_client.get_blacklisted_hotkeys())
    
    def add_blacklisted_hotkeys(
        self,
        hotkeys: List[str],
        reason: Optional[str] = None,
    ) -> SubmissionResponse:
        """Add hotkeys to the blacklist."""
        return self._run(self._async_client.add_blacklisted_hotkeys(hotkeys, reason))
    
    def remove_blacklisted_hotkey(self, hotkey: str) -> SubmissionResponse:
        """Remove a hotkey from the blacklist."""
        return self._run(self._async_client.remove_blacklisted_hotkey(hotkey))


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    import asyncio
    
    async def main():
        """Example usage of the Talisman API Client."""
        # Initialize wallet (update with your validator wallet details)
        wallet = bt.Wallet(name="validator", hotkey="default")
        
        # Create client
        async with TalismanAPIClient(
            base_url="http://localhost:8000",
            wallet=wallet,
        ) as client:
            # Health check
            print("Checking API health...")
            health = await client.health_check()
            print(f"API Status: {health}")
            
            # Get unscored tweets
            print("\nGetting unscored tweets...")
            tweets = await client.get_unscored_tweets(limit=3)
            print(f"Got {len(tweets)} tweets:")
            for tweet in tweets:
                text_preview = (tweet.text[:50] + "...") if tweet.text and len(tweet.text) > 50 else tweet.text
                print(f"  - {tweet.id}: {text_preview}")
            
            # Example: Submit completed tweets (uncomment to use)
            # if tweets:
            #     completed = [
            #         {"tweet_id": tweet.id, "sentiment": "bullish"}
            #         for tweet in tweets
            #     ]
            #     result = await client.submit_completed_tweets(completed)
            #     print(f"Submitted: {result.message}")
            
            # Get blacklisted hotkeys
            print("\nGetting blacklisted hotkeys...")
            blacklisted = await client.get_blacklisted_hotkeys()
            print(f"Blacklisted hotkeys: {len(blacklisted)}")
    
    # Run the example
    asyncio.run(main())
