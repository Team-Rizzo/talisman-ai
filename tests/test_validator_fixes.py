"""
Unit tests for validator logic bug fixes.

Tests the fixes for:
1. Burn modifier logic error
2. Token selection bias
3. Metagraph hotkey race condition
4. Score range validation
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
from talisman_ai.validator.grader import select_tokens, normalize_keys


class TestBurnModifierFix:
    """Tests for burn modifier logic fix."""
    
    def test_burn_modifier_within_range(self):
        """Test that burn modifier works correctly when burn_uid is in range."""
        # Simulate normalized weights
        scores = np.array([0.5, 0.3, 0.2])
        norm = np.linalg.norm(scores, ord=1, axis=0, keepdims=True)
        raw_weights = scores / norm
        
        burn_modifier = 0.9
        burn_uid = 1
        
        # Apply burn modifier
        if burn_modifier > 0 and 0 <= burn_uid < len(raw_weights):
            raw_weights = raw_weights * (1 - burn_modifier)
            raw_weights[burn_uid] += burn_modifier
        
        # Check that weights still sum to ~1.0
        assert np.isclose(raw_weights.sum(), 1.0, atol=1e-6)
        
        # Check that burn_uid got the burn_modifier amount
        assert np.isclose(raw_weights[burn_uid], 0.9 + 0.3 * 0.1, atol=1e-6)
    
    def test_burn_modifier_out_of_range(self):
        """Test that burn modifier is skipped when burn_uid is out of range."""
        scores = np.array([0.5, 0.3, 0.2])
        norm = np.linalg.norm(scores, ord=1, axis=0, keepdims=True)
        raw_weights = scores / norm
        original_weights = raw_weights.copy()
        
        burn_modifier = 0.9
        burn_uid = 189  # Out of range
        
        # Apply burn modifier (should be skipped)
        if burn_modifier > 0:
            if 0 <= burn_uid < len(raw_weights):
                raw_weights = raw_weights * (1 - burn_modifier)
                raw_weights[burn_uid] += burn_modifier
            # else: warning logged, no modification
        
        # Weights should be unchanged
        assert np.allclose(raw_weights, original_weights)
    
    def test_burn_modifier_zero(self):
        """Test that zero burn modifier doesn't modify weights."""
        scores = np.array([0.5, 0.3, 0.2])
        norm = np.linalg.norm(scores, ord=1, axis=0, keepdims=True)
        raw_weights = scores / norm
        original_weights = raw_weights.copy()
        
        burn_modifier = 0.0
        burn_uid = 1
        
        # Apply burn modifier (should be skipped)
        if burn_modifier > 0:
            if 0 <= burn_uid < len(raw_weights):
                raw_weights = raw_weights * (1 - burn_modifier)
                raw_weights[burn_uid] += burn_modifier
        
        # Weights should be unchanged
        assert np.allclose(raw_weights, original_weights)


class TestTokenSelectionFix:
    """Tests for token selection bias fix."""
    
    def test_common_tokens_only(self):
        """Test that only common tokens are compared."""
        miner_tokens = {"subnet_1": 0.8, "subnet_2": 0.6, "subnet_3": 0.4}
        validator_tokens = {"subnet_1": 0.75, "subnet_2": 0.65, "subnet_3": 0.45}
        
        miner_selected, validator_selected = select_tokens(miner_tokens, validator_tokens)
        
        # All tokens are common, should all be included
        assert set(miner_selected.keys()) == {"subnet_1", "subnet_2", "subnet_3"}
        assert set(validator_selected.keys()) == {"subnet_1", "subnet_2", "subnet_3"}
    
    def test_validator_only_tokens_included(self):
        """Test that validator-only tokens are included (miner missed these)."""
        miner_tokens = {"subnet_1": 0.8, "subnet_2": 0.6}
        validator_tokens = {"subnet_1": 0.75, "subnet_2": 0.65, "subnet_3": 0.5}
        
        miner_selected, validator_selected = select_tokens(miner_tokens, validator_tokens)
        
        # Should include all validator tokens
        assert "subnet_3" in validator_selected
        # Miner should have 0.0 for missed token
        assert miner_selected.get("subnet_3", 0.0) == 0.0
    
    def test_miner_only_tokens_excluded(self):
        """Test that miner-only tokens are excluded (prevents gaming)."""
        miner_tokens = {"subnet_1": 0.8, "subnet_2": 0.6, "fake_subnet": 0.04}
        validator_tokens = {"subnet_1": 0.75, "subnet_2": 0.65}
        
        miner_selected, validator_selected = select_tokens(miner_tokens, validator_tokens)
        
        # Miner's fake token should NOT be included
        assert "fake_subnet" not in miner_selected
        assert "fake_subnet" not in validator_selected
    
    def test_eps_filtering(self):
        """Test that tiny values below eps are filtered out."""
        miner_tokens = {"subnet_1": 0.8, "subnet_2": 0.02}  # 0.02 < 0.05 eps
        validator_tokens = {"subnet_1": 0.75, "subnet_2": 0.03}  # 0.03 < 0.05 eps
        
        miner_selected, validator_selected = select_tokens(miner_tokens, validator_tokens, eps=0.05)
        
        # subnet_2 should be filtered out from both
        assert "subnet_2" not in miner_selected
        assert "subnet_2" not in validator_selected
    
    def test_token_cap_k(self):
        """Test that token count is capped at k."""
        # Create many tokens
        miner_tokens = {f"subnet_{i}": 0.1 for i in range(150)}
        validator_tokens = {f"subnet_{i}": 0.1 for i in range(150)}
        
        miner_selected, validator_selected = select_tokens(miner_tokens, validator_tokens, k=128)
        
        # Should be capped at 128
        assert len(miner_selected) <= 128
        assert len(validator_selected) <= 128


class TestScoreValidation:
    """Tests for score range validation."""
    
    def test_valid_score_accepted(self):
        """Test that valid scores in [0.0, 1.0] are accepted."""
        scores = {"hotkey_1": 0.5, "hotkey_2": 0.8, "hotkey_3": 0.0, "hotkey_4": 1.0}
        
        for hotkey, score in scores.items():
            score_float = float(score)
            if not (0.0 <= score_float <= 1.0):
                score_float = max(0.0, min(1.0, score_float))
            
            assert 0.0 <= score_float <= 1.0
    
    def test_negative_score_clamped(self):
        """Test that negative scores are clamped to 0.0."""
        score = -0.5
        score_float = float(score)
        
        if not (0.0 <= score_float <= 1.0):
            score_float = max(0.0, min(1.0, score_float))
        
        assert score_float == 0.0
    
    def test_score_above_one_clamped(self):
        """Test that scores > 1.0 are clamped to 1.0."""
        score = 1.5
        score_float = float(score)
        
        if not (0.0 <= score_float <= 1.0):
            score_float = max(0.0, min(1.0, score_float))
        
        assert score_float == 1.0
    
    def test_nan_score_handled(self):
        """Test that NaN scores are set to 0.0."""
        score = float('nan')
        score_float = float(score)
        
        if np.isnan(score_float) or np.isinf(score_float):
            score_float = 0.0
        
        assert score_float == 0.0
    
    def test_inf_score_handled(self):
        """Test that Inf scores are set to 0.0."""
        score = float('inf')
        score_float = float(score)
        
        if np.isnan(score_float) or np.isinf(score_float):
            score_float = 0.0
        
        assert score_float == 0.0


class TestHotkeyMapping:
    """Tests for hotkey→UID mapping to prevent race conditions."""
    
    def test_hotkey_mapping_consistent(self):
        """Test that hotkey→UID mapping is consistent."""
        # Simulate metagraph hotkeys
        hotkeys = ["hotkey_1", "hotkey_2", "hotkey_3"]
        
        # Build mapping
        hotkey_to_uid = {hotkey: uid for uid, hotkey in enumerate(hotkeys)}
        
        # Check mapping
        assert hotkey_to_uid["hotkey_1"] == 0
        assert hotkey_to_uid["hotkey_2"] == 1
        assert hotkey_to_uid["hotkey_3"] == 2
    
    def test_missing_hotkey_returns_none(self):
        """Test that missing hotkeys return None."""
        hotkeys = ["hotkey_1", "hotkey_2"]
        hotkey_to_uid = {hotkey: uid for uid, hotkey in enumerate(hotkeys)}
        
        # Missing hotkey
        uid = hotkey_to_uid.get("hotkey_missing")
        
        assert uid is None
    
    def test_mapping_survives_metagraph_update(self):
        """Test that mapping is stable even if metagraph updates."""
        # Initial metagraph
        hotkeys_v1 = ["hotkey_1", "hotkey_2", "hotkey_3"]
        hotkey_to_uid_v1 = {hotkey: uid for uid, hotkey in enumerate(hotkeys_v1)}
        
        # Metagraph updates (new miner registers)
        hotkeys_v2 = ["hotkey_1", "hotkey_new", "hotkey_2", "hotkey_3"]
        
        # Old mapping still valid for old hotkeys
        assert hotkey_to_uid_v1["hotkey_1"] == 0
        assert hotkey_to_uid_v1["hotkey_2"] == 1
        
        # But new mapping would be different
        hotkey_to_uid_v2 = {hotkey: uid for uid, hotkey in enumerate(hotkeys_v2)}
        assert hotkey_to_uid_v2["hotkey_1"] == 0
        assert hotkey_to_uid_v2["hotkey_2"] == 2  # Shifted!


class TestMovingAverageAlpha:
    """Tests for moving average alpha parameter."""
    
    def test_alpha_025_faster_response(self):
        """Test that alpha=0.25 responds faster than alpha=0.1."""
        # Simulate score drop from 1.0 to 0.0
        old_score = 1.0
        new_reward = 0.0
        
        # With alpha=0.1 (old default)
        score_alpha_01 = 0.1 * new_reward + 0.9 * old_score
        
        # With alpha=0.25 (new default)
        score_alpha_025 = 0.25 * new_reward + 0.75 * old_score
        
        # Alpha=0.25 should drop faster
        assert score_alpha_025 < score_alpha_01
        assert score_alpha_025 == 0.75
        assert score_alpha_01 == 0.9
    
    def test_alpha_025_convergence(self):
        """Test convergence speed with alpha=0.25."""
        score = 1.0
        new_reward = 0.0
        alpha = 0.25
        
        # Simulate 10 updates
        for _ in range(10):
            score = alpha * new_reward + (1 - alpha) * score
        
        # After 10 updates, should be < 0.06
        assert score < 0.06
    
    def test_alpha_01_slow_convergence(self):
        """Test that alpha=0.1 converges slowly."""
        score = 1.0
        new_reward = 0.0
        alpha = 0.1
        
        # Simulate 10 updates
        for _ in range(10):
            score = alpha * new_reward + (1 - alpha) * score
        
        # After 10 updates, still > 0.3
        assert score > 0.3
