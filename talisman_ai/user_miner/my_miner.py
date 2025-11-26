"""
User miner implementation that orchestrates the post processing pipeline:
scrape -> analyze -> submit.

Block-based approach: scrapes every 100 blocks (aligned with API v2 rate limit windows),
fetches exactly 5 posts (the rate limit max), and submits them all.

Runs in a single background thread for simplicity and reliability.
Rate limiting is handled server-side by the API. The miner relies on API responses
(including 429 rate limit errors) rather than tracking limits locally to avoid
desynchronization issues.
"""

import threading
import time
import bittensor as bt
from typing import Dict, Optional
from datetime import datetime, timezone

from talisman_ai import config
from talisman_ai.user_miner.api_client import APIClient
from talisman_ai.user_miner.post_scraper import PostScraper
from talisman_ai.analyzer import setup_analyzer
from talisman_ai.analyzer.scoring import score_post_entry
from talisman_ai.utils.normalization import norm_text

class MyMiner:
    """
    User miner that processes posts in a background thread.
    
    Block-based scraping: scrapes every 100 blocks (aligned with API v2 rate limit windows),
    fetches exactly 5 posts (the rate limit max), and submits them all.
    
    The miner follows a simple pipeline:
    1. Waits for new block window (every 100 blocks)
    2. Scrapes exactly 5 posts from X API
    3. Analyzes each post for subnet relevance and sentiment
    4. Submits all analyzed posts to the API server
    
    Runs continuously until stopped explicitly.
    """
    
    def __init__(self, hotkey: str = "HOTKEY_PLACEHOLDER", wallet: bt.wallet = None, subtensor: bt.subtensor = None):
        """
        Initialize the miner with required components.
        
        Args:
            hotkey: Miner hotkey identifier, typically provided by the parent neuron.
                   Defaults to placeholder if not provided.
            wallet: Optional Bittensor wallet for API authentication.
            subtensor: Optional Bittensor subtensor for getting current block number.
        """
        self.scraper = PostScraper()
        self.analyzer = setup_analyzer()
        self.api_client = APIClient(wallet=wallet)

        self.miner_hotkey = hotkey
        self.subtensor = subtensor

        # Track post IDs we've already processed to avoid duplicate submissions
        self._seen_post_ids: set[str] = set()

        # Threading state management
        self.lock = threading.Lock()
        self.running = False
        self.thread: threading.Thread | None = None

        # Post processing statistics
        self.posts_processed = 0
        
        # Block-based scraping configuration
        # The miner scrapes posts every N blocks (aligned with API rate limit windows)
        self.blocks_per_window = config.BLOCKS_PER_WINDOW  # Default: 100 blocks
        self.posts_per_window = config.MAX_SUBMISSIONS_PER_WINDOW  # Default: 5 posts
        self._last_block_window: Optional[int] = None
        
        # API synchronization: track API's block number for accurate window alignment.
        # The API is the source of truth for rate limiting, so we use its block number
        # to ensure our window calculations match the server's expectations.
        self._api_block_number: Optional[int] = None
        
        # Rate limiting is handled server-side by the API v2 endpoint.
        # The API will return HTTP 429 (rate limit exceeded) if we exceed the limit.
        # We rely on the API's response rather than tracking limits locally to avoid
        # desynchronization issues between client and server.
    
    def _get_current_block(self) -> int:
        """
        Get the current block number.
        
        Prefers the API's block number if available (since it's the source of truth
        for rate limiting). Falls back to subtensor if the API block number hasn't
        been synchronized yet.
        
        Returns:
            Current block number, or 0 if unavailable.
        """
        # Prefer API's block number (source of truth for rate limiting)
        if self._api_block_number is not None:
            return self._api_block_number
        
        # Fallback to subtensor if API not yet synchronized
        if not self.subtensor:
            bt.logging.warning("[MyMiner] No subtensor available and API not synchronized, cannot get block number")
            return 0
        try:
            return self.subtensor.get_current_block()
        except Exception as e:
            bt.logging.error(f"[MyMiner] Failed to get current block: {e}")
            return 0
    
    def _sync_with_api_block(self, api_block: int, window_start_block: int):
        """
        Synchronize miner's block tracking with API's block number.
        
        Called when receiving block/window info from API responses.
        This ensures the miner uses the same block number as the API for window calculations.
        
        Args:
            api_block: Current block number from API
            window_start_block: Start block of current window from API (used for logging)
        """
        if api_block > 0:
            old_block = self._api_block_number
            self._api_block_number = api_block
            
            if old_block is None:
                bt.logging.info(f"[MyMiner] Synchronized with API: block={api_block}, window_start={window_start_block}")
            elif api_block != old_block:
                bt.logging.debug(f"[MyMiner] Updated API block: {old_block} -> {api_block}, window_start={window_start_block}")
    
    def _get_block_window(self, block: int) -> int:
        """
        Calculate which block window the given block belongs to.
        
        Windows are calculated by dividing the block number by blocks_per_window.
        For example, with blocks_per_window=100: block 0-99 = window 0, 100-199 = window 1, etc.
        
        Args:
            block: The block number to calculate the window for.
            
        Returns:
            The window number (0-indexed), or 0 if block is 0.
        """
        if block == 0:
            return 0
        return block // self.blocks_per_window
    
    def _should_scrape(self, current_block: int) -> tuple[bool, int]:
        """
        Determine if we should scrape posts for the current block window.
        
        Scraping occurs once per block window. This method checks if we've already
        scraped for the current window, or if this is a new window that needs scraping.
        
        Args:
            current_block: The current block number.
        
        Returns:
            Tuple of (should_scrape: bool, current_window: int):
            - should_scrape: True if we should scrape now, False otherwise
            - current_window: The window number for the current block
        """
        if current_block == 0:
            return False, 0
        
        current_window = self._get_block_window(current_block)
        
        # Scrape if we haven't scraped for this window yet
        should_scrape = self._last_block_window is None or self._last_block_window < current_window
        return should_scrape, current_window

    def _sync_with_api_on_startup(self):
        """
        Attempt to synchronize with the API's block number on startup.
        
        This ensures the miner starts with accurate block tracking. If synchronization
        fails, it will be attempted again on the first post submission.
        """
        try:
            status_info = self.api_client.get_status()
            if status_info and status_info.get("current_block"):
                api_block = status_info.get("current_block")
                window_start = status_info.get("window_start_block")
                if api_block and window_start:
                    self._sync_with_api_block(api_block, window_start)
                    bt.logging.info(f"[MyMiner] Initial sync with API: block={api_block}, window_start={window_start}")
                else:
                    bt.logging.warning("[MyMiner] API status response missing block info, will sync on first submission")
            else:
                bt.logging.debug("[MyMiner] Could not get API status on startup, will sync on first submission")
        except Exception as e:
            bt.logging.debug(f"[MyMiner] Failed to sync with API on startup: {e}, will sync on first submission")
    
    def start(self):
        """
        Starts the miner's background processing thread.
        Safe to call multiple times (idempotent).
        """
        if not self.running:
            # Attempt to synchronize with API before starting
            self._sync_with_api_on_startup()
            
            self.running = True
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
            bt.logging.info("[MyMiner] Started in background thread")

    def stop(self):
        """
        Stops the miner's background processing thread.
        Waits up to 5 seconds for the thread to finish gracefully.
        """
        if self.running:
            self.running = False
            if self.thread:
                self.thread.join(timeout=5)
            bt.logging.info("[MyMiner] Stopped")
    

    def _run(self):
        """
        Main processing loop running in the background thread.
        
        Block-based approach: waits for new block window (every 100 blocks),
        scrapes exactly 5 posts, analyzes them, and submits them all.
        
        Continuously runs until running is set to False.
        Handles errors gracefully and logs progress.
        """
        bt.logging.info("[MyMiner] Background thread started")
        bt.logging.info(f"[MyMiner] Hotkey: {self.miner_hotkey}")
        bt.logging.info(f"[MyMiner] Configuration: blocks_per_window={self.blocks_per_window}, posts_per_window={self.posts_per_window}")
        
        # Poll interval: check for new blocks every ~12 seconds (approximate block time)
        poll_interval = 12.0
        
        while self.running:
            try:
                current_block = self._get_current_block()
                
                if current_block == 0:
                    bt.logging.warning("[MyMiner] Cannot get block number, waiting before retry...")
                    time.sleep(poll_interval)
                    continue
                
                # Check if we should scrape (new block window)
                should_scrape, current_window = self._should_scrape(current_block)
                if should_scrape:
                    bt.logging.info(f"[MyMiner] ========== New block window detected ==========")
                    bt.logging.info(f"[MyMiner] Current block: {current_block}, Window: {current_window}")
                    bt.logging.info(f"[MyMiner] Scraping {self.posts_per_window} post(s) for this window...")
                    
                    # Scrape exactly the number of posts allowed per window
                    posts = self.scraper.scrape_posts(count=self.posts_per_window) or []
                    bt.logging.info(f"[MyMiner] Scraped {len(posts)} post(s)")
                    
                    if not posts:
                        bt.logging.warning("[MyMiner] No posts scraped, skipping this window")
                        self._last_block_window = current_window
                        time.sleep(poll_interval)
                        continue
                    
                    # Filter out posts we've already submitted to avoid duplicates
                    new_posts = []
                    for post in posts:
                        pid = str(post.get("id", "unknown"))
                        if pid not in self._seen_post_ids:
                            new_posts.append(post)
                        else:
                            bt.logging.debug(f"[MyMiner] Post {pid} already submitted, skipping")
                    
                    bt.logging.info(f"[MyMiner] {len(new_posts)} new post(s) to process (after filtering duplicates)")
                    
                    if len(new_posts) == 0:
                        bt.logging.info("[MyMiner] No new posts to process, skipping this window")
                        self._last_block_window = current_window
                        time.sleep(poll_interval)
                        continue
                    
                    # Process and submit all new posts
                    submitted_this_window = 0
                    failed_posts = []  # Track failed posts for retry logic
                    
                    for post in new_posts:
                        pid = str(post.get("id", "unknown"))
                        bt.logging.info(f"[MyMiner] Processing post ID: {pid}")

                        # Get raw content - API will normalize it via PostSubmission model
                        # We send raw content to avoid double normalization issues
                        raw_content = post.get("content", "")
                        if not raw_content:
                            bt.logging.warning(f"[MyMiner] Post {pid} content is empty, skipping")
                            continue
                        
                        # Normalize content for analysis (analyzer expects normalized text)
                        content = norm_text(raw_content)
                        if not content:
                            bt.logging.warning(f"[MyMiner] Post {pid} content is empty after normalization, skipping")
                            continue

                        # Analyze post content for subnet relevance and sentiment
                        # This determines which subnets the post is relevant to and its overall sentiment
                        bt.logging.info(f"[MyMiner] Analyzing post {pid} (content length: {len(content)} chars)")
                        analysis = self.analyzer.analyze_post_complete(content)
                        bt.logging.debug(f"[MyMiner] Analysis complete for {pid}: {analysis}")
                        
                        # Extract subnet relevance scores (range: 0.0 to 1.0) for each subnet
                        # Higher scores indicate greater relevance to that subnet
                        tokens = {}
                        for subnet_name, relevance_data in analysis.get("subnet_relevance", {}).items():
                            relevance_score = relevance_data.get("relevance", 0.0)
                            tokens[subnet_name] = float(relevance_score)
                        
                        # Ensure tokens is never empty - API requires at least one subnet with relevance > 0.0
                        # If tokens is empty or all values are 0.0, add a default subnet with minimum relevance
                        if not tokens or max(tokens.values(), default=0.0) <= 0.0:
                            # Add "sn45" subnet with minimum relevance to ensure validation passes
                            tokens["sn45"] = 0.01  # Minimum non-zero relevance
                            bt.logging.warning(f"[MyMiner] No subnet relevance found for {pid}, adding default 'sn45' with relevance 0.01")
                        
                        bt.logging.info(f"[MyMiner] Extracted tokens for {pid}: {list(tokens.keys())} (scores: {tokens})")
                        
                        # Extract sentiment score: -1.0 (very negative) to 1.0 (very positive)
                        sentiment = float(analysis.get("sentiment", 0.0))
                        bt.logging.info(f"[MyMiner] Extracted sentiment for {pid}: {sentiment:.3f}")

                        # Calculate post score using the same scoring logic as the validator
                        # This ensures consistency in how posts are evaluated
                        post_date = post.get("timestamp", 0)
                        if isinstance(post_date, int):
                            dt = datetime.fromtimestamp(post_date, tz=timezone.utc)
                            post_date_iso = dt.isoformat()
                        else:
                            post_date_iso = datetime.now(timezone.utc).isoformat()
                        
                        post_entry = {
                            "url": f"post_{pid}",
                            "post_info": {
                                "post_text": content,
                                "post_date": post_date_iso,
                                "like_count": int(post.get("likes", 0) or 0),
                                "retweet_count": int(post.get("retweets", 0) or 0),
                                "quote_count": 0,
                                "reply_count": int(post.get("responses", 0) or 0),
                                "author_followers": int(post.get("followers", 0) or 0),
                                "account_age_days": int(post.get("account_age", 0)),
                            }
                        }
                        
                        try:
                            scored_result = score_post_entry(post_entry, self.analyzer, k=5, analysis_result=analysis)
                            post_score = scored_result.get("score", 0.0)
                            bt.logging.info(f"[MyMiner] Calculated score for {pid}: {post_score:.3f}")
                        except Exception as e:
                            bt.logging.warning(f"[MyMiner] Error calculating score for {pid}: {e}, using 0.0")
                            post_score = 0.0

                        post_data = {
                            "miner_hotkey": self.miner_hotkey,
                            "post_id": pid,
                            "content": raw_content,  # Send raw content - API will normalize it
                            "date": int(post.get("timestamp", 0)),
                            "author": str(post.get("author", "unknown")),
                            "account_age": int(post.get("account_age", 0)),
                            "retweets": int(post.get("retweets", 0) or 0),
                            "likes": int(post.get("likes", 0) or 0),
                            "responses": int(post.get("responses", 0) or 0),
                            "followers": int(post.get("followers", 0) or 0),
                            "tokens": tokens,
                            "sentiment": sentiment,
                            "score": post_score,
                        }
                        bt.logging.info(f"[MyMiner] Prepared post_data for {pid}: hotkey={post_data['miner_hotkey']}, author={post_data['author']}, score={post_score:.3f}")

                        # Submit post to API server
                        bt.logging.info(f"[MyMiner] Submitting post {pid} to API...")
                        success, block_info, error_status = self.api_client.submit_post(post_data)
                        
                        # Synchronize with API's block number (source of truth for rate limiting)
                        # This ensures our window calculations stay aligned with the server
                        if block_info and block_info.get("current_block"):
                            api_block = block_info.get("current_block")
                            window_start = block_info.get("window_start_block")
                            if api_block and window_start:
                                self._sync_with_api_block(api_block, window_start)
                        
                        if success:
                            # Successfully submitted - track the post and update statistics
                            self._seen_post_ids.add(pid)
                            self.posts_processed += 1
                            submitted_this_window += 1
                            
                            bt.logging.info(
                                f"[MyMiner] ✓ Submitted post '{pid}' successfully "
                                f"(total: {self.posts_processed}, window: {submitted_this_window}/{len(new_posts)})"
                            )
                        else:
                            # Submission failed - check error type to determine retry strategy
                            if error_status == 409:
                                # HTTP 409 Conflict: permanent failure (duplicate/conflict), don't retry
                                bt.logging.warning(f"[MyMiner] ✗ Post '{pid}' rejected with 409 Conflict (permanent failure, not retrying)")
                                self._seen_post_ids.add(pid)  # Mark as seen to avoid retrying
                            elif error_status == 429:
                                # HTTP 429 Rate limit: wait for next window, don't retry immediately
                                bt.logging.warning(f"[MyMiner] ✗ Post '{pid}' hit rate limit (429), will retry in next window")
                                failed_posts.append((post_data, error_status))
                            else:
                                # Other errors (network issues, server errors, etc.) - retry later
                                bt.logging.warning(f"[MyMiner] ✗ Failed to submit post '{pid}' (HTTP {error_status}), will retry")
                                failed_posts.append((post_data, error_status))
                    
                    # Retry failed submissions (excluding 409 conflicts and 429 rate limits)
                    # Uses exponential backoff to avoid overwhelming the API
                    if failed_posts:
                        max_retries = 3
                        retry_delay = 2.0  # Initial delay in seconds
                        
                        for retry_attempt in range(max_retries):
                            if not failed_posts:
                                break
                            
                            bt.logging.info(f"[MyMiner] Retrying {len(failed_posts)} failed submission(s) (attempt {retry_attempt + 1}/{max_retries})...")
                            # Exponential backoff: wait 2s, 4s, 8s between retries
                            time.sleep(retry_delay * (2 ** retry_attempt))
                            
                            remaining_failed = []
                            for post_data, original_error_status in failed_posts:
                                pid = post_data.get("post_id", "unknown")
                                
                                # Skip 409 errors (permanent failures - don't retry)
                                if original_error_status == 409:
                                    bt.logging.debug(f"[MyMiner] Skipping retry for {pid} (409 Conflict)")
                                    self._seen_post_ids.add(pid)
                                    continue
                                
                                # Skip 429 errors (rate limit - wait for next window)
                                if original_error_status == 429:
                                    bt.logging.debug(f"[MyMiner] Skipping retry for {pid} (429 Rate limit - wait for next window)")
                                    remaining_failed.append((post_data, original_error_status))
                                    continue
                                
                                # Retry submission for transient errors
                                bt.logging.info(f"[MyMiner] Retrying submission for post {pid}...")
                                success, block_info, error_status = self.api_client.submit_post(post_data)
                                
                                # Synchronize with API's block number
                                if block_info and block_info.get("current_block"):
                                    api_block = block_info.get("current_block")
                                    window_start = block_info.get("window_start_block")
                                    if api_block and window_start:
                                        self._sync_with_api_block(api_block, window_start)
                                
                                if success:
                                    # Retry succeeded - update tracking
                                    self._seen_post_ids.add(pid)
                                    self.posts_processed += 1
                                    submitted_this_window += 1
                                    bt.logging.info(
                                        f"[MyMiner] ✓ Retry succeeded for post '{pid}' "
                                        f"(total: {self.posts_processed})"
                                    )
                                else:
                                    # Still failed - check error type
                                    if error_status == 409:
                                        # Now it's a 409 - permanent failure, don't retry again
                                        bt.logging.warning(f"[MyMiner] ✗ Post '{pid}' now returns 409 Conflict (permanent failure)")
                                        self._seen_post_ids.add(pid)
                                    elif error_status == 429:
                                        # Rate limit hit during retry - wait for next window
                                        bt.logging.warning(f"[MyMiner] ✗ Post '{pid}' hit rate limit during retry (429)")
                                        remaining_failed.append((post_data, error_status))
                                    else:
                                        # Other error - will retry again if attempts remain
                                        bt.logging.warning(f"[MyMiner] ✗ Retry failed for post '{pid}' (HTTP {error_status})")
                                        remaining_failed.append((post_data, error_status))
                            
                            failed_posts = remaining_failed
                        
                        # Log final status of any posts that still failed after all retries
                        if failed_posts:
                            bt.logging.warning(f"[MyMiner] {len(failed_posts)} post(s) still failed after {max_retries} retry attempts")
                            for post_data, error_status in failed_posts:
                                pid = post_data.get("post_id", "unknown")
                                if error_status == 429:
                                    bt.logging.info(f"[MyMiner]   - {pid}: Rate limit (429) - will retry in next window")
                                else:
                                    bt.logging.warning(f"[MyMiner]   - {pid}: HTTP {error_status} - giving up")
                    
                    # Mark this window as processed to avoid re-scraping
                    self._last_block_window = current_window
                    bt.logging.info(f"[MyMiner] Completed window {current_window}: submitted {submitted_this_window}/{len(new_posts)} posts")
                else:
                    # Still in the same window - wait and check again later
                    current_window = self._get_block_window(current_block)
                    blocks_until_next = self.blocks_per_window - (current_block % self.blocks_per_window)
                    estimated_seconds = blocks_until_next * 12  # ~12 seconds per block
                    bt.logging.debug(f"[MyMiner] Block {current_block}, window {current_window}. Next window in ~{blocks_until_next} blocks (~{estimated_seconds}s)")
                
                # Poll for block changes before next iteration
                time.sleep(poll_interval)

            except Exception as e:
                bt.logging.error(f"[MyMiner] Error in loop: {e}")
                time.sleep(poll_interval)

        bt.logging.info(f"[MyMiner] Background thread stopped. Processed {self.posts_processed} posts")

    def get_stats(self) -> Dict:
        """
        Returns current statistics about the miner's operation.
        
        Returns:
            Dict containing:
                - posts_processed: Number of posts successfully submitted
                - running: Whether the miner is currently running
                - thread_alive: Whether the background thread is alive
        """
        with self.lock:
            return {
                "posts_processed": self.posts_processed,
                "running": self.running,
                "thread_alive": self.thread.is_alive() if self.thread else False,
            }
