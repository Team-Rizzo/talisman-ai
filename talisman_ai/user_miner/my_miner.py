"""
Miner that scrapes posts, runs analysis, and submits results to the subnet API.

Posts are scraped each window, with up to a configured maximum submitted per window.
Rate limiting and window tracking are coordinated with the server.
"""

import threading
import time
import bittensor as bt
from typing import Dict, Optional, TypedDict
from datetime import datetime, timezone
from dataclasses import dataclass, field

from talisman_ai import config
from talisman_ai.user_miner.api_client import APIClient, SubmissionResult, BlockInfoDict
from talisman_ai.user_miner.post_scraper import PostScraper
from talisman_ai.analyzer import setup_analyzer
from talisman_ai.analyzer.scoring import score_post_entry
from talisman_ai.utils.normalization import norm_text
from talisman_ai.utils.misc import safe_int


class PostDict(TypedDict, total=False):
    """
    Post format produced by `PostScraper` and consumed by `MyMiner`.

    All fields are expected in normal operation, but are marked optional for robustness.
    """
    id: str  # Post ID
    content: str  # Raw post text content
    author: str  # Author username
    timestamp: int  # Unix timestamp in seconds
    account_age: int  # Account age in days
    retweets: int  # Number of retweets
    likes: int  # Number of likes
    responses: int  # Number of replies/responses
    followers: int  # Author's follower count


@dataclass
class WindowTracker:
    """Tracks synchronization state with the API's submission windows."""
    last_processed: Optional[int] = None
    loops_since_check: int = 0
    check_interval: int = field(default_factory=lambda: config.STATUS_CHECK_INTERVAL)


class MyMiner:
    """
    Miner loop that runs in a background thread and performs:
    scrape → analyze → score → submit.
    """
    
    def __init__(self, hotkey: str = "HOTKEY_PLACEHOLDER", wallet: bt.wallet = None):
        """
        Initialize the miner and supporting components.

        Args:
            hotkey: Miner hotkey identifier.
            wallet: Optional Bittensor wallet used for API authentication.
        """
        self.scraper = PostScraper()
        self.analyzer = setup_analyzer()
        self.api_client = APIClient(wallet=wallet)

        self.miner_hotkey = hotkey

        # Thread-safe, bounded set for tracking already-seen post IDs.
        self._seen_post_ids: set[str] = set()
        self._max_seen_ids = 10000  # Maximum number of post IDs to track
        
        self.lock = threading.Lock()
        self.running = False
        self.thread: threading.Thread | None = None
        self.posts_processed = 0
        
        self.blocks_per_window = config.BLOCKS_PER_WINDOW
        self.posts_per_window = config.MAX_SUBMISSIONS_PER_WINDOW
        
        # Window tracking for rate limit synchronization
        self._window = WindowTracker()
    
    def _get_current_window(self) -> Optional[int]:
        """Return the current window as reported by the API status endpoint."""
        try:
            status_info = self.api_client.get_status()
            if status_info and status_info.get("current_window") is not None:
                return status_info.get("current_window")
        except Exception as e:
            bt.logging.debug(f"[MyMiner] Failed to get status for window detection: {e}")
        
        return None
    
    def _update_window_from_response(self, block_info: Optional[BlockInfoDict] = None, window: Optional[int] = None) -> None:
        """
        Update window tracking from a submission response or explicit window value.

        Args:
            block_info: Optional block/window info from an API response.
            window: Optional explicit window number (takes precedence over `block_info`).
        """
        # Extract window value from either parameter
        if window is not None:
            response_window = window
        elif block_info and block_info.get("current_window") is not None:
            response_window = block_info.get("current_window")
        else:
            return
        
        if self._window.last_processed is None:
            bt.logging.info(f"[MyMiner] Synchronized with API: window={response_window}")
            self._window.last_processed = response_window
        elif response_window > self._window.last_processed:
            bt.logging.info(f"[MyMiner] New window detected: {response_window} (was {self._window.last_processed})")
            self._window.last_processed = response_window
        elif response_window < self._window.last_processed:
            bt.logging.warning(
                f"[MyMiner] Received window {response_window} which is less than "
                f"last processed {self._window.last_processed}"
            )
    
    def _should_scrape(self) -> tuple[bool, Optional[int]]:
        """
        Decide whether to scrape based on window synchronization state.

        Combines initial status sync, periodic status polling, and submission-driven
        updates to detect new windows.

        Returns:
            (should_scrape, current_window)
        """
        # Initial sync: get current window immediately
        if self._window.last_processed is None:
            current_window = self._get_current_window()
            if current_window is not None:
                self._update_window_from_response(window=current_window)
                return True, current_window
            return False, None
        
        # Periodic status endpoint check: handles case when not actively submitting
        # Window updates from submission responses handle the active case
        self._window.loops_since_check += 1
        if self._window.loops_since_check >= self._window.check_interval:
            self._window.loops_since_check = 0
            current_window = self._get_current_window()
            if current_window is not None and current_window > self._window.last_processed:
                self._update_window_from_response(window=current_window)
                return True, current_window
            elif current_window is not None:
                bt.logging.debug(f"[MyMiner] Still in window {current_window} (last processed: {self._window.last_processed})")
        
        return False, None

    def _validate_post_content(self, post: PostDict) -> tuple[bool, str, str]:
        """
        Validate and normalize post content.

        Returns:
            (is_valid, post_id, normalized_content)
        """
        pid = str(post.get("id", "unknown"))
        raw_content = post.get("content", "")
        
        if not raw_content:
            bt.logging.warning(f"[MyMiner] Post {pid} content is empty, skipping")
            return False, pid, ""
        
        content = norm_text(raw_content)
        if not content:
            bt.logging.warning(f"[MyMiner] Post {pid} content is empty after normalization, skipping")
            return False, pid, ""
        
        return True, pid, content
    
    def _prepare_post_data(self, post: PostDict, pid: str, content: str) -> tuple[Dict, Dict]:
        """
        Build data structures used for scoring and submission.

        Does a single pass over metadata to avoid redundant normalization.

        Returns:
            (post_entry_for_scoring, metadata_for_submission)
        """
        # Convert timestamp to ISO format
        post_date = post.get("timestamp", 0)
        if isinstance(post_date, int):
            dt = datetime.fromtimestamp(post_date, tz=timezone.utc)
            post_date_iso = dt.isoformat()
        else:
            post_date_iso = datetime.now(timezone.utc).isoformat()
        
        # Normalize all metadata once
        metadata = {
            "post_date_iso": post_date_iso,
            "timestamp": int(post.get("timestamp", 0)),
            "account_age": safe_int(post.get("account_age", 0)),
            "likes": safe_int(post.get("likes", 0)),
            "retweets": safe_int(post.get("retweets", 0)),
            "responses": safe_int(post.get("responses", 0)),
            "followers": safe_int(post.get("followers", 0)),
        }
        
        # Build scoring entry
        post_entry = {
            "url": f"post_{pid}",
            "post_info": {
                "post_text": content,
                "post_date": metadata["post_date_iso"],
                "like_count": metadata["likes"],
                "retweet_count": metadata["retweets"],
                "quote_count": 0,
                "reply_count": metadata["responses"],
                "author_followers": metadata["followers"],
                "account_age_days": metadata["account_age"],
            }
        }
        
        return post_entry, metadata
    
    def _analyze_post(self, content: str, pid: str) -> tuple[Optional[Dict], Dict[str, float], float]:
        """
        Run analyzer and extract subnet relevance tokens and sentiment.

        Returns:
            (analysis, tokens, sentiment); returns (None, {}, 0.0) on failure or no relevance.
        """
        bt.logging.info(f"[MyMiner] Analyzing post {pid} (content length: {len(content)} chars)")
        
        # Run analyzer with defensive error handling.
        try:
            analysis = self.analyzer.analyze_post_complete(content)
        except Exception as e:
            bt.logging.warning(f"[MyMiner] Analyzer failed for post {pid}: {e}, skipping")
            return None, {}, 0.0
        
        # Require a dictionary-like result from the analyzer.
        if not isinstance(analysis, dict):
            bt.logging.warning(
                f"[MyMiner] Analyzer returned unexpected type for post {pid}: "
                f"{type(analysis).__name__}, skipping"
            )
            return None, {}, 0.0
        
        tokens = {}
        for subnet_name, relevance_data in analysis.get("subnet_relevance", {}).items():
            tokens[subnet_name] = float(relevance_data.get("relevance", 0.0))
        
        # Do not submit posts without subnet relevance.
        if not tokens or max(tokens.values(), default=0.0) <= 0.0:
            bt.logging.warning(f"[MyMiner] Post {pid} has no subnet relevance, skipping")
            return None, {}, 0.0
        
        sentiment = float(analysis.get("sentiment", 0.0))
        bt.logging.info(f"[MyMiner] Extracted tokens: {list(tokens.keys())}, sentiment: {sentiment:.3f}")
        
        return analysis, tokens, sentiment
    
    def _build_submission_data(self, post: PostDict, pid: str, content: str, tokens: Dict[str, float], 
                              sentiment: float, post_score: float, metadata: Dict) -> Dict:
        """
        Construct the payload sent to the API.
        """
        return {
            "miner_hotkey": self.miner_hotkey,
            "post_id": pid,
            "content": post.get("content", ""),  # Raw content for API
            "date": metadata["timestamp"],
            "author": str(post.get("author", "unknown")),
            "account_age": metadata["account_age"],
            "retweets": metadata["retweets"],
            "likes": metadata["likes"],
            "responses": metadata["responses"],
            "followers": metadata["followers"],
            "tokens": tokens,
            "sentiment": sentiment,
            "score": post_score,
        }
    
    def _process_and_submit_post(self, post: PostDict) -> SubmissionResult:
        """
        Process a single post end-to-end: validate, analyze, score, and submit.
        """
        # Validate content
        is_valid, pid, content = self._validate_post_content(post)
        if not is_valid:
            return SubmissionResult(success=False, block_info=None, error_status=None)
        
        # Analyze post
        analysis, tokens, sentiment = self._analyze_post(content, pid)
        if analysis is None:
            return SubmissionResult(success=False, block_info=None, error_status=None)
        
        # Prepare post data for scoring and submission
        post_entry, metadata = self._prepare_post_data(post, pid, content)
        
        # Calculate score
        try:
            scored_result = score_post_entry(post_entry, self.analyzer, k=5, analysis_result=analysis)
            post_score = scored_result.get("score", 0.0)
        except Exception as e:
            bt.logging.warning(f"[MyMiner] Error calculating score for {pid}: {e}, using 0.0")
            post_score = 0.0
        
        # Build submission data
        post_data = self._build_submission_data(post, pid, content, tokens, sentiment, post_score, metadata)
        
        # Submit to API
        bt.logging.info(f"[MyMiner] Submitting post {pid}...")
        result = self.api_client.submit_post(post_data)
        
        self._update_window_from_response(result.block_info)
        
        # Thread-safe update of `seen_post_ids` and `posts_processed`.
        with self.lock:
            # Bound the `seen_post_ids` set to prevent unbounded memory growth.
            if len(self._seen_post_ids) > self._max_seen_ids:
                # Clear the tracking set when the limit is exceeded to keep memory bounded.
                self._seen_post_ids.clear()
                bt.logging.debug(f"[MyMiner] Cleared seen_post_ids set (exceeded max of {self._max_seen_ids})")
            
            if result.success:
                self._seen_post_ids.add(pid)
                self.posts_processed += 1
                bt.logging.info(
                    f"[MyMiner] ✓ Submitted post '{pid}' successfully "
                    f"(total: {self.posts_processed})"
                )
            elif result.error_status == 409:
                # Permanent failure (409 Conflict) — mark as seen to avoid retrying.
                self._seen_post_ids.add(pid)
                bt.logging.warning(f"[MyMiner] ✗ Post '{pid}' rejected with 409 Conflict")
            else:
                # Other errors (e.g. network issues).
                bt.logging.warning(
                    f"[MyMiner] ✗ Failed to submit post '{pid}'"
                    f"{f' (HTTP {result.error_status})' if result.error_status else ''}"
                )
        
        # For 409 errors, return success=False but keep `error_status` so the caller can treat
        # them as processed (do not retry) while still reflecting the failure.
        return result

    def start(self):
        """Start the miner in a background thread."""
        with self.lock:
            if not self.running:
                self.running = True
                self.thread = threading.Thread(target=self._run, daemon=True)
                self.thread.start()
                bt.logging.info("[MyMiner] Started in background thread")

    def stop(self):
        """Stop the background thread and clean up resources."""
        with self.lock:
            if self.running:
                self.running = False
                thread = self.thread
            else:
                thread = None
        
        if thread:
            thread.join(timeout=5)
        
        # Clean up API client session.
        if hasattr(self, 'api_client'):
            self.api_client.close()
        
        bt.logging.info("[MyMiner] Stopped")
    

    def _run(self):
        """Main loop: wait for a new window, then scrape, analyze, and submit posts."""
        bt.logging.info("[MyMiner] Background thread started")
        bt.logging.info(f"[MyMiner] Hotkey: {self.miner_hotkey}")
        bt.logging.info(f"[MyMiner] Configuration: blocks_per_window={self.blocks_per_window}, posts_per_window={self.posts_per_window}")
        
        poll_interval = config.POLL_INTERVAL_SECONDS
        
        while self.running:
            try:
                should_scrape, current_window = self._should_scrape()
                
                if should_scrape:
                    if current_window is None:
                        bt.logging.warning("[MyMiner] Cannot determine current window, waiting before retry...")
                        time.sleep(poll_interval)
                        continue
                    
                    self._process_window(current_window)
                    # Window state is updated via _update_window_from_response() during submissions
                    # No need to redundantly set it here
                else:
                    if self._window.last_processed is None:
                        bt.logging.debug("[MyMiner] Waiting for initial window synchronization...")
                    else:
                        bt.logging.debug(f"[MyMiner] Still in window {self._window.last_processed}, waiting for next window...")
                
                time.sleep(poll_interval)

            except Exception as e:
                bt.logging.error(f"[MyMiner] Error in loop: {e}")
                time.sleep(poll_interval)

        with self.lock:
            final_count = self.posts_processed
        bt.logging.info(f"[MyMiner] Background thread stopped. Processed {final_count} posts")

    def _process_window(self, current_window: int) -> None:
        """
        Process a single submission window: scrape, filter duplicates, and submit.
        """
        bt.logging.info(f"[MyMiner] ========== New block window detected ==========")
        bt.logging.info(f"[MyMiner] Current window: {current_window}")
        bt.logging.info(f"[MyMiner] Scraping {self.posts_per_window} post(s)...")

        posts = self.scraper.scrape_posts(count=self.posts_per_window) or []
        bt.logging.info(f"[MyMiner] Scraped {len(posts)} post(s)")

        # No posts available for this window.
        if not posts:
            bt.logging.warning("[MyMiner] No posts scraped, skipping this window")
            return

        # Thread-safe duplicate filtering (hold lock only for ID checks).
        with self.lock:
            new_posts = [p for p in posts if str(p.get("id", "unknown")) not in self._seen_post_ids]
        duplicate_count = len(posts) - len(new_posts)
        if duplicate_count > 0:
            bt.logging.info(
                f"[MyMiner] {duplicate_count} duplicate post(s) filtered out (already submitted)"
            )
        bt.logging.info(f"[MyMiner] {len(new_posts)} new post(s) to process")

        if len(new_posts) == 0:
            bt.logging.info("[MyMiner] No new posts to process, skipping this window")
            return

        # Process and submit all new posts for this window.
        submitted_this_window = 0

        for post in new_posts:
            result = self._process_and_submit_post(post)

            # Count successful submissions; treat 409 as terminal but not successful.
            if result.success or result.error_status == 409:
                if result.success:
                    submitted_this_window += 1
                # Stop once the local per-window submission limit is reached.
                if submitted_this_window >= self.posts_per_window:
                    bt.logging.info(
                        f"[MyMiner] Reached local rate limit "
                        f"({self.posts_per_window} posts), marking window as complete"
                    )
                    break

            # If the API reports rate limiting (429), stop submitting in this window.
            if result.error_status == 429:
                bt.logging.info(
                    "[MyMiner] Hit API rate limit (429) for this window, "
                    "stopping submissions until next window"
                )
                # Rely on `block_info` / `_update_window_from_response` to keep
                # the window state in sync with the API.
                break

        bt.logging.info(
            f"[MyMiner] Completed window {current_window}: "
            f"submitted {submitted_this_window}/{len(new_posts)} posts"
        )

    def get_stats(self) -> Dict:
        """Return a snapshot of current miner statistics."""
        with self.lock:
            return {
                "posts_processed": self.posts_processed,
                "running": self.running,
                "thread_alive": self.thread.is_alive() if self.thread else False,
            }
