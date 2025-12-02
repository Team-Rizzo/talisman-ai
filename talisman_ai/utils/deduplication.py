"""
Post deduplication utilities for Talisman AI subnet.

This module provides functions to prevent miners from gaming the system
by resubmitting the same posts multiple times within the scoring window.
"""

import time
from typing import Dict, Optional, Tuple


class PostDeduplicator:
    """
    Tracks submitted posts to prevent duplicate submissions within a time window.
    
    This prevents miners from gaming the reward system by submitting the same
    high-value post multiple times. Each post_id can only be submitted once
    within the deduplication window (default: 10 days).
    
    Attributes:
        seen_posts: Dictionary mapping post_id to first submission timestamp
        window_days: Number of days to track posts for deduplication
    
    Example:
        >>> dedup = PostDeduplicator(window_days=10)
        >>> is_dup, msg = dedup.is_duplicate("post_12345", miner_hotkey="5G...")
        >>> if is_dup:
        ...     return {"status": "duplicate", "message": msg}
    """
    
    def __init__(self, window_days: int = 10):
        """
        Initialize the post deduplicator.
        
        Args:
            window_days: Number of days to track posts (default: 10, matches scoring window)
        """
        self.seen_posts: Dict[str, Dict[str, float]] = {}  # {post_id: {miner_hotkey: timestamp}}
        self.window_days = window_days
        self.window_seconds = window_days * 24 * 3600
    
    def is_duplicate(
        self, 
        post_id: str, 
        miner_hotkey: str,
        current_time: Optional[float] = None
    ) -> Tuple[bool, str]:
        """
        Check if a post has already been submitted by this miner within the window.
        
        Args:
            post_id: The unique post identifier (e.g., Twitter post ID)
            miner_hotkey: The miner's hotkey submitting the post
            current_time: Optional timestamp (defaults to current time)
        
        Returns:
            Tuple of (is_duplicate: bool, message: str)
            - (True, message) if post is a duplicate
            - (False, "") if post is new or outside window
        
        Notes:
            - Same post_id from different miners is allowed (different perspectives)
            - Same post_id from same miner within window is rejected
            - Posts outside the window are allowed (can be resubmitted after expiry)
        """
        if current_time is None:
            current_time = time.time()
        
        # Check if we've seen this post_id before
        if post_id not in self.seen_posts:
            # First time seeing this post from any miner
            return False, ""
        
        # Check if THIS miner has submitted this post before
        if miner_hotkey not in self.seen_posts[post_id]:
            # This miner hasn't submitted this post before (another miner did)
            return False, ""
        
        # This miner has submitted this post before - check if within window
        first_submission = self.seen_posts[post_id][miner_hotkey]
        age_seconds = current_time - first_submission
        age_days = age_seconds / (24 * 3600)
        
        if age_seconds < self.window_seconds:
            # Duplicate within window - reject
            return True, (
                f"Post {post_id} already submitted by miner {miner_hotkey[:8]}... "
                f"{age_days:.1f} days ago (within {self.window_days}-day window)"
            )
        
        # Outside window - allow resubmission
        return False, ""
    
    def mark_as_seen(
        self, 
        post_id: str, 
        miner_hotkey: str,
        current_time: Optional[float] = None
    ) -> None:
        """
        Mark a post as seen by a specific miner.
        
        This should be called after validating that the post is not a duplicate
        and before accepting the submission.
        
        Args:
            post_id: The unique post identifier
            miner_hotkey: The miner's hotkey submitting the post
            current_time: Optional timestamp (defaults to current time)
        
        Example:
            >>> is_dup, msg = dedup.is_duplicate(post_id, miner_hotkey)
            >>> if not is_dup:
            ...     dedup.mark_as_seen(post_id, miner_hotkey)
            ...     # Process submission
        """
        if current_time is None:
            current_time = time.time()
        
        if post_id not in self.seen_posts:
            self.seen_posts[post_id] = {}
        
        self.seen_posts[post_id][miner_hotkey] = current_time
    
    def cleanup_old_entries(self, current_time: Optional[float] = None) -> int:
        """
        Remove posts that are outside the deduplication window.
        
        This prevents memory growth by removing old entries that are no longer
        relevant for deduplication checks.
        
        Args:
            current_time: Optional timestamp (defaults to current time)
        
        Returns:
            Number of post entries removed
        
        Example:
            >>> # Run periodically (e.g., every hour)
            >>> removed = dedup.cleanup_old_entries()
            >>> print(f"Cleaned up {removed} old post entries")
        """
        if current_time is None:
            current_time = time.time()
        
        cutoff_time = current_time - self.window_seconds
        removed_count = 0
        
        # Find posts where ALL miner submissions are outside the window
        posts_to_remove = []
        
        for post_id, miner_submissions in self.seen_posts.items():
            # Remove individual miner entries that are too old
            miners_to_remove = [
                miner for miner, timestamp in miner_submissions.items()
                if timestamp < cutoff_time
            ]
            
            for miner in miners_to_remove:
                del miner_submissions[miner]
                removed_count += 1
            
            # If no miners left for this post, mark post for removal
            if not miner_submissions:
                posts_to_remove.append(post_id)
        
        # Remove posts with no miner submissions
        for post_id in posts_to_remove:
            del self.seen_posts[post_id]
        
        return removed_count
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get statistics about the deduplicator state.
        
        Returns:
            Dictionary with statistics:
            - total_posts: Number of unique posts tracked
            - total_submissions: Total number of (post_id, miner) pairs tracked
        
        Example:
            >>> stats = dedup.get_stats()
            >>> print(f"Tracking {stats['total_posts']} posts, {stats['total_submissions']} submissions")
        """
        total_posts = len(self.seen_posts)
        total_submissions = sum(len(miners) for miners in self.seen_posts.values())
        
        return {
            "total_posts": total_posts,
            "total_submissions": total_submissions,
        }


def check_post_age(post_timestamp: float, max_age_days: int = 10) -> Tuple[bool, Optional[str]]:
    """
    Check if a post is within the acceptable age range for scoring.
    
    Posts older than max_age_days receive a score of 0 and should be rejected
    to prevent miners from submitting stale content.
    
    Args:
        post_timestamp: Unix timestamp of when the post was created
        max_age_days: Maximum age in days (default: 10, matches scoring window)
    
    Returns:
        Tuple of (is_valid: bool, error_message: Optional[str])
        - (True, None) if post is within age limit
        - (False, error_message) if post is too old
    
    Example:
        >>> is_valid, error = check_post_age(post_timestamp, max_age_days=10)
        >>> if not is_valid:
        ...     return {"status": "rejected", "reason": error}
    """
    current_time = time.time()
    age_seconds = current_time - post_timestamp
    age_days = age_seconds / (24 * 3600)
    
    if age_days > max_age_days:
        return False, f"Post is too old ({age_days:.1f} days, max: {max_age_days} days)"
    
    # Also check if post is in the future (clock skew or manipulation)
    if age_days < -1:  # Allow 1 day of clock skew
        return False, f"Post timestamp is in the future ({-age_days:.1f} days ahead)"
    
    return True, None
