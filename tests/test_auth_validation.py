"""
Unit tests for authentication validation utilities.

Tests the replay attack prevention mechanisms including timestamp validation
and signature freshness checks.
"""

import time
import pytest
from talisman_ai.utils.auth_validation import (
    validate_auth_timestamp,
    validate_signature_freshness,
    cleanup_seen_signatures,
)


class TestTimestampValidation:
    """Tests for timestamp validation to prevent replay attacks."""
    
    def test_valid_recent_timestamp(self):
        """Test that a recent timestamp is accepted."""
        current_time = time.time()
        timestamp_str = str(current_time - 10)  # 10 seconds ago
        
        is_valid, error = validate_auth_timestamp(timestamp_str)
        
        assert is_valid is True
        assert error is None
    
    def test_expired_timestamp_rejected(self):
        """Test that an old timestamp is rejected."""
        current_time = time.time()
        old_timestamp = str(current_time - 400)  # 400 seconds ago (> 5 min default)
        
        is_valid, error = validate_auth_timestamp(old_timestamp)
        
        assert is_valid is False
        assert "expired" in error.lower()
    
    def test_future_timestamp_rejected(self):
        """Test that a future timestamp is rejected (clock skew protection)."""
        current_time = time.time()
        future_timestamp = str(current_time + 120)  # 2 minutes in future
        
        is_valid, error = validate_auth_timestamp(future_timestamp)
        
        assert is_valid is False
        assert "future" in error.lower()
    
    def test_invalid_format_rejected(self):
        """Test that invalid timestamp formats are rejected."""
        invalid_timestamps = ["not_a_number", "", "abc123", None]
        
        for invalid_ts in invalid_timestamps:
            is_valid, error = validate_auth_timestamp(str(invalid_ts))
            
            assert is_valid is False
            assert "invalid" in error.lower() or "format" in error.lower()
    
    def test_custom_max_age(self):
        """Test that custom max_age_seconds parameter works."""
        current_time = time.time()
        timestamp_str = str(current_time - 150)  # 150 seconds ago
        
        # Should fail with default 300s window
        is_valid, _ = validate_auth_timestamp(timestamp_str, max_age_seconds=100)
        assert is_valid is False
        
        # Should pass with 200s window
        is_valid, _ = validate_auth_timestamp(timestamp_str, max_age_seconds=200)
        assert is_valid is True
    
    def test_clock_skew_tolerance(self):
        """Test that reasonable clock skew is tolerated."""
        current_time = time.time()
        slightly_future = str(current_time + 30)  # 30 seconds in future
        
        # Should pass with default 60s clock skew tolerance
        is_valid, error = validate_auth_timestamp(slightly_future)
        
        assert is_valid is True
        assert error is None


class TestSignatureFreshness:
    """Tests for signature freshness validation (nonce-based replay protection)."""
    
    def test_fresh_signature_accepted(self):
        """Test that a new signature is accepted."""
        seen_sigs = {}
        signature = "sig_12345"
        timestamp = time.time()
        
        is_valid, error = validate_signature_freshness(signature, timestamp, seen_sigs)
        
        assert is_valid is True
        assert error is None
    
    def test_reused_signature_rejected(self):
        """Test that a reused signature is rejected."""
        seen_sigs = {}
        signature = "sig_12345"
        timestamp = time.time()
        
        # First use - should be accepted
        is_valid, _ = validate_signature_freshness(signature, timestamp, seen_sigs)
        assert is_valid is True
        
        # Mark as used
        seen_sigs[signature] = timestamp
        
        # Second use - should be rejected
        is_valid, error = validate_signature_freshness(signature, timestamp, seen_sigs)
        assert is_valid is False
        assert "already used" in error.lower() or "replay" in error.lower()
    
    def test_expired_signature_allowed(self):
        """Test that a signature outside the window can be reused."""
        seen_sigs = {}
        signature = "sig_12345"
        old_timestamp = time.time() - 400  # 400 seconds ago
        
        # Mark as used 400 seconds ago
        seen_sigs[signature] = old_timestamp
        
        # Try to use again with 300s window - should be allowed
        current_timestamp = time.time()
        is_valid, error = validate_signature_freshness(
            signature, current_timestamp, seen_sigs, window_seconds=300
        )
        
        assert is_valid is True
        assert error is None


class TestSignatureCleanup:
    """Tests for signature cleanup to prevent memory growth."""
    
    def test_cleanup_removes_old_signatures(self):
        """Test that old signatures are removed during cleanup."""
        current_time = time.time()
        seen_sigs = {
            "sig_old_1": current_time - 400,  # 400s ago (should be removed)
            "sig_old_2": current_time - 350,  # 350s ago (should be removed)
            "sig_recent": current_time - 100,  # 100s ago (should be kept)
        }
        
        removed = cleanup_seen_signatures(seen_sigs, window_seconds=300)
        
        assert removed == 2
        assert "sig_old_1" not in seen_sigs
        assert "sig_old_2" not in seen_sigs
        assert "sig_recent" in seen_sigs
    
    def test_cleanup_empty_dict(self):
        """Test that cleanup works on empty dictionary."""
        seen_sigs = {}
        
        removed = cleanup_seen_signatures(seen_sigs)
        
        assert removed == 0
        assert len(seen_sigs) == 0
    
    def test_cleanup_all_recent(self):
        """Test that cleanup keeps all recent signatures."""
        current_time = time.time()
        seen_sigs = {
            "sig_1": current_time - 50,
            "sig_2": current_time - 100,
            "sig_3": current_time - 200,
        }
        
        removed = cleanup_seen_signatures(seen_sigs, window_seconds=300)
        
        assert removed == 0
        assert len(seen_sigs) == 3


class TestReplayAttackScenario:
    """Integration tests simulating replay attack scenarios."""
    
    def test_replay_attack_blocked(self):
        """Test that a complete replay attack is blocked."""
        seen_sigs = {}
        
        # Attacker captures valid request
        signature = "captured_sig_12345"
        timestamp = time.time()
        timestamp_str = str(timestamp)
        
        # First request (legitimate) - should succeed
        ts_valid, _ = validate_auth_timestamp(timestamp_str)
        sig_valid, _ = validate_signature_freshness(signature, timestamp, seen_sigs)
        
        assert ts_valid is True
        assert sig_valid is True
        
        # Mark signature as used
        seen_sigs[signature] = timestamp
        
        # Attacker replays same request 10 seconds later
        time.sleep(0.1)  # Small delay for test
        
        # Timestamp still valid (within 5 min window)
        ts_valid, _ = validate_auth_timestamp(timestamp_str)
        assert ts_valid is True
        
        # But signature is reused - should be blocked
        sig_valid, error = validate_signature_freshness(signature, timestamp, seen_sigs)
        assert sig_valid is False
        assert "replay" in error.lower() or "already used" in error.lower()
    
    def test_replay_attack_after_expiry(self):
        """Test that replay attack after expiry window is still blocked by timestamp."""
        # Attacker captures request
        old_timestamp = time.time() - 400  # 400 seconds ago
        timestamp_str = str(old_timestamp)
        
        # Try to replay after 400 seconds
        ts_valid, error = validate_auth_timestamp(timestamp_str, max_age_seconds=300)
        
        # Should be blocked by timestamp expiry
        assert ts_valid is False
        assert "expired" in error.lower()
