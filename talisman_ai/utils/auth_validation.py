"""
Authentication validation utilities for Talisman AI subnet.

This module provides security functions to validate authentication headers
and prevent replay attacks.
"""

import time
from typing import Optional, Tuple


def validate_auth_timestamp(
    timestamp_str: str, 
    max_age_seconds: int = 300,
    max_clock_skew_seconds: int = 60
) -> Tuple[bool, Optional[str]]:
    """
    Validate that an authentication timestamp is recent and not replayed.
    
    This prevents replay attacks by ensuring authentication headers cannot
    be reused after a reasonable time window (default: 5 minutes).
    
    Args:
        timestamp_str: Timestamp string from X-Auth-Timestamp header
        max_age_seconds: Maximum age in seconds before timestamp expires (default: 300 = 5 minutes)
        max_clock_skew_seconds: Maximum allowed clock skew in seconds (default: 60 = 1 minute)
    
    Returns:
        Tuple of (is_valid: bool, error_message: Optional[str])
        - (True, None) if timestamp is valid
        - (False, error_message) if timestamp is invalid
    
    Security Notes:
        - Prevents replay attacks by rejecting old timestamps
        - Allows reasonable clock skew between client and server
        - Should be combined with nonce-based replay protection for defense-in-depth
    
    Example:
        >>> is_valid, error = validate_auth_timestamp("1701561600.123")
        >>> if not is_valid:
        ...     raise HTTPException(status_code=401, detail=error)
    """
    try:
        timestamp = float(timestamp_str)
    except (ValueError, TypeError):
        return False, "Invalid timestamp format"
    
    current_time = time.time()
    age = current_time - timestamp
    
    # Check if timestamp is too old (expired)
    if age > max_age_seconds:
        return False, f"Authentication timestamp expired (age: {int(age)}s, max: {max_age_seconds}s)"
    
    # Check if timestamp is in the future (clock skew protection)
    # This prevents attackers from using future timestamps to extend validity
    if age < -max_clock_skew_seconds:
        return False, f"Authentication timestamp is too far in the future (skew: {int(-age)}s, max: {max_clock_skew_seconds}s)"
    
    return True, None


def validate_signature_freshness(
    signature: str,
    timestamp: float,
    seen_signatures: dict,
    window_seconds: int = 300
) -> Tuple[bool, Optional[str]]:
    """
    Validate that a signature hasn't been used before (nonce-based replay protection).
    
    This provides an additional layer of security beyond timestamp validation
    by tracking signatures that have been used within the time window.
    
    Args:
        signature: The signature from X-Auth-Signature header
        timestamp: The timestamp from X-Auth-Timestamp header (already validated)
        seen_signatures: Dictionary tracking used signatures {signature: timestamp}
        window_seconds: Time window to track signatures (default: 300 = 5 minutes)
    
    Returns:
        Tuple of (is_valid: bool, error_message: Optional[str])
        - (True, None) if signature is fresh (not seen before)
        - (False, error_message) if signature was already used
    
    Note:
        The seen_signatures dict should be periodically cleaned to prevent memory growth.
        Consider using Redis with TTL for production deployments.
    
    Example:
        >>> seen_sigs = {}
        >>> is_valid, error = validate_signature_freshness(sig, ts, seen_sigs)
        >>> if is_valid:
        ...     seen_sigs[sig] = ts  # Mark as used
    """
    current_time = time.time()
    
    # Check if signature was already used
    if signature in seen_signatures:
        first_used = seen_signatures[signature]
        age = current_time - first_used
        
        if age < window_seconds:
            return False, f"Signature already used {int(age)}s ago (replay attack detected)"
    
    return True, None


def cleanup_seen_signatures(
    seen_signatures: dict,
    window_seconds: int = 300
) -> int:
    """
    Remove expired signatures from the tracking dictionary.
    
    This prevents memory growth by removing signatures that are outside
    the replay protection window.
    
    Args:
        seen_signatures: Dictionary tracking used signatures {signature: timestamp}
        window_seconds: Time window to keep signatures (default: 300 = 5 minutes)
    
    Returns:
        Number of signatures removed
    
    Example:
        >>> removed = cleanup_seen_signatures(seen_sigs, window_seconds=300)
        >>> print(f"Cleaned up {removed} expired signatures")
    """
    current_time = time.time()
    cutoff_time = current_time - window_seconds
    
    # Find expired signatures
    expired = [sig for sig, ts in seen_signatures.items() if ts < cutoff_time]
    
    # Remove them
    for sig in expired:
        del seen_signatures[sig]
    
    return len(expired)
