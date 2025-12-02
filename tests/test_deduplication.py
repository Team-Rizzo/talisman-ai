"""
Unit tests for post deduplication utilities.

Tests the prevention of duplicate post submissions within the scoring window.
"""

import time
import pytest
from talisman_ai.utils.deduplication import (
    PostDeduplicator,
    check_post_age,
)


class TestPostDeduplicator:
    """Tests for PostDeduplicator class."""
    
    def test_first_submission_not_duplicate(self):
        """Test that first submission of a post is not marked as duplicate."""
        dedup = PostDeduplicator(window_days=10)
        
        is_dup, msg = dedup.is_duplicate("post_123", "miner_hotkey_1")
        
        assert is_dup is False
        assert msg == ""
    
    def test_same_miner_duplicate_rejected(self):
        """Test that same miner cannot submit same post twice within window."""
        dedup = PostDeduplicator(window_days=10)
        post_id = "post_123"
        miner = "miner_hotkey_1"
        
        # First submission
        is_dup, _ = dedup.is_duplicate(post_id, miner)
        assert is_dup is False
        dedup.mark_as_seen(post_id, miner)
        
        # Second submission (duplicate)
        is_dup, msg = dedup.is_duplicate(post_id, miner)
        assert is_dup is True
        assert "already submitted" in msg.lower()
    
    def test_different_miner_same_post_allowed(self):
        """Test that different miners can submit the same post."""
        dedup = PostDeduplicator(window_days=10)
        post_id = "post_123"
        
        # Miner 1 submits
        is_dup, _ = dedup.is_duplicate(post_id, "miner_1")
        assert is_dup is False
        dedup.mark_as_seen(post_id, "miner_1")
        
        # Miner 2 submits same post - should be allowed
        is_dup, _ = dedup.is_duplicate(post_id, "miner_2")
        assert is_dup is False
    
    def test_resubmission_after_window_allowed(self):
        """Test that post can be resubmitted after window expires."""
        dedup = PostDeduplicator(window_days=10)
        post_id = "post_123"
        miner = "miner_hotkey_1"
        
        # Submit at old timestamp (11 days ago)
        old_time = time.time() - (11 * 24 * 3600)
        is_dup, _ = dedup.is_duplicate(post_id, miner, current_time=old_time)
        assert is_dup is False
        dedup.mark_as_seen(post_id, miner, current_time=old_time)
        
        # Try to submit again now (11 days later, outside 10-day window)
        current_time = time.time()
        is_dup, _ = dedup.is_duplicate(post_id, miner, current_time=current_time)
        assert is_dup is False
    
    def test_duplicate_within_window_rejected(self):
        """Test that duplicate within window is rejected with correct message."""
        dedup = PostDeduplicator(window_days=10)
        post_id = "post_123"
        miner = "miner_hotkey_1"
        
        # Submit 5 days ago
        old_time = time.time() - (5 * 24 * 3600)
        dedup.mark_as_seen(post_id, miner, current_time=old_time)
        
        # Try to submit again now (5 days later, within 10-day window)
        is_dup, msg = dedup.is_duplicate(post_id, miner)
        
        assert is_dup is True
        assert "5" in msg  # Should mention ~5 days
        assert "10-day window" in msg


class TestPostDeduplicatorCleanup:
    """Tests for cleanup functionality."""
    
    def test_cleanup_removes_old_entries(self):
        """Test that cleanup removes posts outside the window."""
        dedup = PostDeduplicator(window_days=10)
        current_time = time.time()
        
        # Add old submissions (11 days ago)
        old_time = current_time - (11 * 24 * 3600)
        dedup.mark_as_seen("post_old_1", "miner_1", current_time=old_time)
        dedup.mark_as_seen("post_old_2", "miner_1", current_time=old_time)
        
        # Add recent submission (5 days ago)
        recent_time = current_time - (5 * 24 * 3600)
        dedup.mark_as_seen("post_recent", "miner_1", current_time=recent_time)
        
        # Cleanup
        removed = dedup.cleanup_old_entries(current_time=current_time)
        
        assert removed == 2
        stats = dedup.get_stats()
        assert stats["total_posts"] == 1  # Only recent post remains
        assert stats["total_submissions"] == 1
    
    def test_cleanup_handles_multiple_miners(self):
        """Test that cleanup correctly handles posts with multiple miner submissions."""
        dedup = PostDeduplicator(window_days=10)
        current_time = time.time()
        post_id = "post_123"
        
        # Miner 1 submitted 11 days ago (should be removed)
        old_time = current_time - (11 * 24 * 3600)
        dedup.mark_as_seen(post_id, "miner_1", current_time=old_time)
        
        # Miner 2 submitted 5 days ago (should be kept)
        recent_time = current_time - (5 * 24 * 3600)
        dedup.mark_as_seen(post_id, "miner_2", current_time=recent_time)
        
        # Cleanup
        removed = dedup.cleanup_old_entries(current_time=current_time)
        
        assert removed == 1  # Only miner_1's submission removed
        stats = dedup.get_stats()
        assert stats["total_posts"] == 1  # Post still tracked
        assert stats["total_submissions"] == 1  # Only miner_2's submission remains
    
    def test_cleanup_empty_deduplicator(self):
        """Test that cleanup works on empty deduplicator."""
        dedup = PostDeduplicator(window_days=10)
        
        removed = dedup.cleanup_old_entries()
        
        assert removed == 0
        stats = dedup.get_stats()
        assert stats["total_posts"] == 0


class TestPostDeduplicatorStats:
    """Tests for statistics functionality."""
    
    def test_stats_empty(self):
        """Test stats on empty deduplicator."""
        dedup = PostDeduplicator(window_days=10)
        
        stats = dedup.get_stats()
        
        assert stats["total_posts"] == 0
        assert stats["total_submissions"] == 0
    
    def test_stats_single_post_single_miner(self):
        """Test stats with one post from one miner."""
        dedup = PostDeduplicator(window_days=10)
        dedup.mark_as_seen("post_1", "miner_1")
        
        stats = dedup.get_stats()
        
        assert stats["total_posts"] == 1
        assert stats["total_submissions"] == 1
    
    def test_stats_single_post_multiple_miners(self):
        """Test stats with one post from multiple miners."""
        dedup = PostDeduplicator(window_days=10)
        dedup.mark_as_seen("post_1", "miner_1")
        dedup.mark_as_seen("post_1", "miner_2")
        dedup.mark_as_seen("post_1", "miner_3")
        
        stats = dedup.get_stats()
        
        assert stats["total_posts"] == 1
        assert stats["total_submissions"] == 3
    
    def test_stats_multiple_posts(self):
        """Test stats with multiple posts from multiple miners."""
        dedup = PostDeduplicator(window_days=10)
        dedup.mark_as_seen("post_1", "miner_1")
        dedup.mark_as_seen("post_1", "miner_2")
        dedup.mark_as_seen("post_2", "miner_1")
        dedup.mark_as_seen("post_3", "miner_3")
        
        stats = dedup.get_stats()
        
        assert stats["total_posts"] == 3
        assert stats["total_submissions"] == 4


class TestPostAgeValidation:
    """Tests for post age validation."""
    
    def test_recent_post_valid(self):
        """Test that a recent post is valid."""
        current_time = time.time()
        post_timestamp = current_time - (5 * 24 * 3600)  # 5 days ago
        
        is_valid, error = check_post_age(post_timestamp, max_age_days=10)
        
        assert is_valid is True
        assert error is None
    
    def test_old_post_rejected(self):
        """Test that an old post is rejected."""
        current_time = time.time()
        post_timestamp = current_time - (15 * 24 * 3600)  # 15 days ago
        
        is_valid, error = check_post_age(post_timestamp, max_age_days=10)
        
        assert is_valid is False
        assert "too old" in error.lower()
        assert "15" in error  # Should mention age
    
    def test_future_post_rejected(self):
        """Test that a post with future timestamp is rejected."""
        current_time = time.time()
        post_timestamp = current_time + (2 * 24 * 3600)  # 2 days in future
        
        is_valid, error = check_post_age(post_timestamp, max_age_days=10)
        
        assert is_valid is False
        assert "future" in error.lower()
    
    def test_boundary_post_valid(self):
        """Test that a post at the boundary is valid."""
        current_time = time.time()
        post_timestamp = current_time - (9.9 * 24 * 3600)  # 9.9 days ago
        
        is_valid, error = check_post_age(post_timestamp, max_age_days=10)
        
        assert is_valid is True
        assert error is None


class TestGamingScenario:
    """Integration tests simulating gaming scenarios."""
    
    def test_gaming_via_resubmission_blocked(self):
        """Test that gaming via resubmission is blocked."""
        dedup = PostDeduplicator(window_days=10)
        post_id = "high_value_post_123"
        miner = "gaming_miner"
        
        # Miner finds high-value post and submits it
        is_dup, _ = dedup.is_duplicate(post_id, miner)
        assert is_dup is False
        dedup.mark_as_seen(post_id, miner)
        
        # Miner tries to submit same post again next day (gaming attempt)
        next_day = time.time() + (1 * 24 * 3600)
        is_dup, msg = dedup.is_duplicate(post_id, miner, current_time=next_day)
        
        # Should be blocked
        assert is_dup is True
        assert "already submitted" in msg.lower()
    
    def test_multiple_resubmission_attempts_blocked(self):
        """Test that multiple resubmission attempts are all blocked."""
        dedup = PostDeduplicator(window_days=10)
        post_id = "high_value_post_123"
        miner = "gaming_miner"
        
        # Initial submission
        dedup.mark_as_seen(post_id, miner)
        
        # Try to submit on days 2, 3, 4, 5 (all should be blocked)
        for day in range(2, 6):
            future_time = time.time() + (day * 24 * 3600)
            is_dup, _ = dedup.is_duplicate(post_id, miner, current_time=future_time)
            assert is_dup is True
    
    def test_legitimate_different_posts_allowed(self):
        """Test that legitimate submissions of different posts are allowed."""
        dedup = PostDeduplicator(window_days=10)
        miner = "honest_miner"
        
        # Miner submits 5 different posts
        for i in range(5):
            post_id = f"post_{i}"
            is_dup, _ = dedup.is_duplicate(post_id, miner)
            assert is_dup is False
            dedup.mark_as_seen(post_id, miner)
        
        stats = dedup.get_stats()
        assert stats["total_posts"] == 5
        assert stats["total_submissions"] == 5
