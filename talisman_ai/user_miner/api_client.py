"""
API client for submitting posts to the API v2 server.

This module handles HTTP communication with the subnet API server, including:
- POST requests to the /v2/submit endpoint
- Retry logic with exponential backoff (3s, 6s, 12s)
- Header authentication (X-Auth-* headers)
- Response handling and status validation
- Prominent logging of validation selection status
"""

import time
import requests
import bittensor as bt
from typing import Dict, Optional, TypedDict
from dataclasses import dataclass

from talisman_ai import config


class BlockInfoDict(TypedDict, total=False):
    """
    Block/window information dictionary from API responses.
    
    Used for synchronizing miner window state with the API server.
    All fields are optional as they may not all be present in every response.
    """
    current_block: int
    window_start_block: int
    window_end_block: int
    next_window_start_block: int
    blocks_per_window: int
    current_window: int
    blocks_until_next_window: int


class RateLimitInfoDict(TypedDict, total=False):
    """
    Rate limit information dictionary from API 429 responses.
    
    Provides details about rate limit status and reset timing.
    All fields are optional as they may not all be present in every response.
    """
    current_count: int
    max_submissions: int
    next_window_start_block: int
    estimated_seconds_until_reset: int
    blocks_per_window: int
    current_window: int


@dataclass
class SubmissionResult:
    """
    Result of a post submission attempt.
    
    Attributes:
        success: True if submission succeeded, False otherwise
        block_info: BlockInfoDict with block/window info from API (for synchronization),
                   or None if not available
        error_status: HTTP status code if submission failed, None if successful
    """
    success: bool
    block_info: Optional[BlockInfoDict]
    error_status: Optional[int]


def _extract_block_info(data: Dict) -> BlockInfoDict:
    """
    Extract block/window info from API response.
    
    Args:
        data: Dictionary containing API response data
        
    Returns:
        BlockInfoDict with block/window info containing:
        - current_block: Current block number
        - window_start_block: Start block of current window
        - window_end_block: End block of current window
        - next_window_start_block: Start block of next window
        - blocks_per_window: Number of blocks per window
        - current_window: Current window number
    """
    return {
        "current_block": data.get("current_block"),
        "window_start_block": data.get("window_start_block"),
        "window_end_block": data.get("window_end_block"),
        "next_window_start_block": data.get("next_window_start_block"),
        "blocks_per_window": data.get("blocks_per_window"),
        "current_window": data.get("current_window"),
    }


def _parse_error_response(resp: requests.Response) -> tuple[str, Optional[Dict]]:
    """
    Parse error detail and extra info from API error response.
    
    Args:
        resp: Response object from requests library
        
    Returns:
        Tuple of (error_detail: str, extra_info: Optional[Dict])
    """
    error_detail = "Unknown error"
    extra_info = None
    try:
        error_data = resp.json()
        detail = error_data.get("detail")
        if isinstance(detail, dict):
            error_detail = detail.get("message", "Error")
            extra_info = detail
        elif isinstance(detail, str):
            error_detail = detail
        elif isinstance(detail, list):
            # List of validation errors
            error_messages = []
            for err in detail:
                field = err.get("field", "unknown")
                msg = err.get("message", "validation error")
                error_messages.append(f"{field}: {msg}")
            error_detail = "; ".join(error_messages)
            extra_info = {"validation_errors": detail}
    except Exception:
        pass
    return error_detail, extra_info


class APIClient:
    """
    Client for submitting analyzed posts to the subnet API server.
    
    Handles HTTP communication, retries, and error handling for post submissions.
    The API endpoint is idempotent, so duplicate submissions return a success status.
    """
    
    @staticmethod
    def _create_auth_message(timestamp=None) -> str:
        """Create a standardized authentication message for API requests."""
        if timestamp is None:
            timestamp = time.time()
        return f"talisman-ai-auth:{int(timestamp)}"
    
    @staticmethod
    def _sign_message(wallet, message: str) -> str:
        """Sign a message with the wallet's hotkey."""
        signature = wallet.hotkey.sign(message)
        return signature.hex()
    
    def __init__(self, wallet: Optional[bt.wallet] = None):
        """
        Initialize the API client.
        
        Sets up the base URL pointing to the subnet API server and prepares
        authentication if a wallet is provided.
        
        Args:
            wallet: Optional Bittensor wallet for authentication. If provided,
                   requests will include signed authentication headers.
        """
        # Track total submission attempts (includes both successes and failures)
        self.submission_count = 0
        self.base_url = config.MINER_API_URL
        self.wallet = wallet
        # Use session for connection pooling and better performance
        self.session = requests.Session()
    
    def close(self):
        """Close the HTTP session."""
        if self.session:
            self.session.close()

    def _create_auth_headers(self) -> Dict[str, str]:
        """
        Create authentication headers for API requests.
        
        Returns:
            Dictionary of auth headers, or empty dict if wallet not available.
        """
        if not self.wallet:
            return {}
        
        try:
            timestamp = time.time()
            message = self._create_auth_message(timestamp)
            signature = self._sign_message(self.wallet, message)
            return {
                "X-Auth-SS58Address": self.wallet.hotkey.ss58_address,
                "X-Auth-Signature": signature,
                "X-Auth-Message": message,
                "X-Auth-Timestamp": str(timestamp)
            }
        except Exception as e:
            bt.logging.warning(f"[APIClient] Failed to create auth headers: {e}, proceeding without auth")
            return {}

    def _handle_response(self, resp: requests.Response, post_id: str, hotkey: str) -> Optional[SubmissionResult]:
        """
        Handle HTTP response and return SubmissionResult if terminal, None to continue/retry.
        
        Args:
            resp: Response object from requests
            post_id: Post ID for logging
            hotkey: Miner hotkey for logging
        
        Returns:
            SubmissionResult if response is terminal (success or non-retryable error),
            None if the request should be retried.
        """
        # Handle rate limit (429)
        if resp.status_code == 429:
            error_detail, rate_limit_info = _parse_error_response(resp)
            if rate_limit_info and isinstance(rate_limit_info.get("rate_limit"), dict):
                rate_limit_info = rate_limit_info["rate_limit"]
            
            reset_block = rate_limit_info.get("next_window_start_block") if rate_limit_info else None
            reset_seconds = rate_limit_info.get("estimated_seconds_until_reset") if rate_limit_info else None
            reset_block = reset_block or resp.headers.get("X-RateLimit-Reset-Block")
            reset_seconds = reset_seconds or resp.headers.get("X-RateLimit-Reset-Seconds")
            
            log_msg = f"[APIClient][429] Rate limit exceeded for post {post_id} (hotkey: {hotkey}): {error_detail}"
            if rate_limit_info:
                current = rate_limit_info.get("current_count", "?")
                max_subs = rate_limit_info.get("max_submissions", "?")
                log_msg += f" ({current}/{max_subs} submissions used)"
            if reset_seconds:
                reset_minutes = int(reset_seconds) / 60
                log_msg += f". Limit resets in ~{int(reset_seconds)}s (~{reset_minutes:.1f}min)"
            if reset_block:
                log_msg += f" at block {reset_block}"
            log_msg += ". Will retry in next cycle."
            
            bt.logging.warning(log_msg)
            block_info = _extract_block_info(rate_limit_info) if rate_limit_info else None
            return SubmissionResult(success=False, block_info=block_info, error_status=429)
        
        # Handle 409 Conflict (permanent failure)
        if resp.status_code == 409:
            error_detail, _ = _parse_error_response(resp)
            bt.logging.warning(f"[APIClient][409] Conflict for post {post_id} (hotkey: {hotkey}): {error_detail}. Not retrying.")
            block_info = None
            try:
                error_data = resp.json()
                block_info = _extract_block_info(error_data)
            except Exception:
                pass
            return SubmissionResult(success=False, block_info=block_info, error_status=409)
        
        # Handle 422 Validation Error
        if resp.status_code == 422:
            error_detail, extra_info = _parse_error_response(resp)
            validation_errors = extra_info.get("validation_errors") if extra_info else None
            bt.logging.error(f"[APIClient][422] Validation error for post {post_id} (hotkey: {hotkey}): {error_detail}")
            if validation_errors:
                bt.logging.error(f"[APIClient][422] Validation details: {validation_errors}")
            return SubmissionResult(success=False, block_info=None, error_status=422)
        
        # Not a special status code - return None to let caller handle normally
        return None

    def _submit_to_api(self, post_data: Dict) -> SubmissionResult:
        """
        Internal method to submit a post to the API server (v2).
        
        Implements retry logic with exponential backoff:
        - Up to 3 attempts total (initial + 2 retries)
        - Wait times between retries: 3s after first failure, 6s after second failure
        - Wait time calculated as: 3 * (2^attempt) seconds
        - 10 second timeout per request
        - Handles 429 rate limit errors with longer backoff
        - Does NOT retry on 409 (Conflict) errors - these are permanent failures
        
        Args:
            post_data: Dictionary containing post submission data including:
                      miner_hotkey, post_id, content, date, author, tokens, sentiment, etc.
        
        Returns:
            SubmissionResult containing:
            - success: True if submission succeeded (status "new", "duplicate", or "ok"), False otherwise.
                       Note: "duplicate" is treated as success since the API is idempotent.
                       Returns False if rate limited (429) - caller should wait before retrying.
            - block_info: BlockInfoDict with block/window info from API (for synchronization), or None if not available.
                          Contains: current_block, window_start_block, window_end_block, next_window_start_block, blocks_per_window, current_window
            - error_status: HTTP status code if submission failed, None if successful. Useful for determining retry behavior.
        """
        url = f"{self.base_url}/v2/submit"
        post_id = post_data.get("post_id", "unknown")
        hotkey = post_data.get("miner_hotkey", "")
        
        # Create authentication headers if wallet is available
        headers = self._create_auth_headers()
        if headers:
            bt.logging.debug(f"[APIClient] Added authentication headers for hotkey: {hotkey}")
        
        bt.logging.info(f"[APIClient] Submitting post {post_id} to {url} (hotkey: {hotkey})")
        
        # Retry logic: up to MAX_SUBMIT_ATTEMPTS attempts with exponential backoff
        # Wait time = SUBMIT_BACKOFF_BASE_SECONDS * (2^attempt) seconds (gives 3s, 6s, 12s)
        # For rate limits (429), we return immediately without retrying
        # (caller should wait for the next window)
        for attempt in range(config.MAX_SUBMIT_ATTEMPTS):
            try:
                bt.logging.debug(f"[APIClient] Attempt {attempt+1}/{config.MAX_SUBMIT_ATTEMPTS} for post {post_id}")
                resp = self.session.post(url, json=post_data, headers=headers, timeout=10)
                
                # Check for special status codes (429, 409, 422)
                result = self._handle_response(resp, post_id, hotkey)
                if result is not None:
                    return result
                
                resp.raise_for_status()
                data = resp.json()
                status = data.get("status")
                selected_for_validation = data.get("selected_for_validation", False)
                x_validation_passed = data.get("x_validation_passed")
                validation_id = data.get("validation_id")
                
                # Extract block/window info for synchronization (API is source of truth for rate limiting)
                # This ensures the miner's window calculations stay aligned with the server
                block_info = _extract_block_info(data)
                
                # Log validation selection status prominently for user visibility
                if selected_for_validation:
                    if x_validation_passed:
                        bt.logging.info(
                            f"[APIClient][OK] Post {post_id} (hotkey: {hotkey}) SELECTED for validation (validation_id: {validation_id})"
                        )
                    else:
                        x_error = data.get("x_validation_error", {})
                        error_code = x_error.get("code", "unknown") if x_error else "unknown"
                        error_msg = x_error.get("message", "N/A") if x_error else "N/A"
                        bt.logging.warning(
                            f"[APIClient][OK] Post {post_id} (hotkey: {hotkey}) selected but X validation FAILED: "
                            f"{error_code} - {error_msg}"
                        )
                else:
                    bt.logging.debug(f"[APIClient][OK] Post {post_id} (hotkey: {hotkey}) submitted (not selected for validation)")
                
                # Log full response details at debug level
                bt.logging.debug(
                    f"[APIClient] Full response for {post_id}: status={status}, "
                    f"selected={selected_for_validation}, "
                    f"x_validation_passed={x_validation_passed}, "
                    f"validation_id={validation_id}, "
                    f"message={data.get('message', 'N/A')}, "
                    f"block={block_info.get('current_block')}"
                )
                
                # The API returns "new" for first-time submissions, "duplicate" for already-submitted posts
                # Both are considered success since the API is idempotent per (miner_hotkey, post_id)
                # This means submitting the same post multiple times is safe and won't cause errors
                success = status in ("new", "duplicate", "ok")
                
                return SubmissionResult(success=success, block_info=block_info, error_status=None)
            except requests.HTTPError as e:
                # Note: 409 and 429 are handled above before raise_for_status() is called,
                # so this block only handles other HTTP errors (500, 502, etc.)
                error_status = e.response.status_code if hasattr(e, 'response') and e.response else None
                bt.logging.warning(f"[APIClient][HTTP] HTTP error on attempt {attempt+1}/{config.MAX_SUBMIT_ATTEMPTS} for post {post_id} (hotkey: {hotkey}): {e}")
                if attempt < config.MAX_SUBMIT_ATTEMPTS - 1:
                    wait_time = config.SUBMIT_BACKOFF_BASE_SECONDS * (2 ** attempt)
                    bt.logging.debug(f"[APIClient] Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    # Last attempt failed, return error status
                    return SubmissionResult(success=False, block_info=None, error_status=error_status)
            except requests.RequestException as e:
                # Network errors, timeouts, etc. - retry with exponential backoff
                bt.logging.warning(f"[APIClient][NET] Submit attempt {attempt+1}/{config.MAX_SUBMIT_ATTEMPTS} for post {post_id} (hotkey: {hotkey}) failed: {e}")
                if attempt < config.MAX_SUBMIT_ATTEMPTS - 1:
                    wait_time = config.SUBMIT_BACKOFF_BASE_SECONDS * (2 ** attempt)
                    bt.logging.debug(f"[APIClient] Retrying in {wait_time}s...")
                    time.sleep(wait_time)
        bt.logging.error(f"[APIClient][ERROR] All {config.MAX_SUBMIT_ATTEMPTS} attempts failed for post {post_id} (hotkey: {hotkey})")
        return SubmissionResult(success=False, block_info=None, error_status=None)

    def get_status(self) -> Optional[BlockInfoDict]:
        """
        Query the API /v2/status endpoint to get current block and window information.
        
        This is useful for initial synchronization when the miner starts up, or for
        periodic sync checks to ensure the miner's block tracking stays aligned with
        the server.
        
        Returns:
            BlockInfoDict with status info including:
            - current_block: Current block number (API's view)
            - window_start_block: Start block of current window
            - window_end_block: End block of current window
            - next_window_start_block: Start block of next window
            - blocks_per_window: Number of blocks per window
            - blocks_until_next_window: Blocks until next window starts
            - current_window: Current window number
            Or None if the request failed.
        """
        url = f"{self.base_url}/v2/status"
        try:
            resp = self.session.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") == "ok":
                block_info = _extract_block_info(data)
                block_info["blocks_until_next_window"] = data.get("blocks_until_next_window")
                return block_info
        except Exception as e:
            bt.logging.debug(f"[APIClient] Failed to get status from API: {e}")
        return None
    
    def submit_post(self, post_data: Dict) -> SubmissionResult:
        """
        Submit a post to the API server.
        
        This is the public interface for submitting posts. It tracks submission attempts
        and logs the result. The actual submission logic with retries is handled by
        the internal _submit_to_api method.
        
        Args:
            post_data: Dictionary containing post submission data including:
                     - miner_hotkey: Miner's hotkey identifier
                     - post_id: Unique post identifier
                     - content: Post text content
                     - date: Post timestamp
                     - author: Post author username
                     - tokens: Dictionary of subnet relevance scores
                     - sentiment: Sentiment score (-1.0 to 1.0)
                     - score: Calculated post score
                     - And other metadata fields
        
        Returns:
            SubmissionResult containing:
            - success: True if submission succeeded, False otherwise
            - block_info: BlockInfoDict with block/window info from API (for synchronization),
                         or None if not available
            - error_status: HTTP status code if submission failed, None if successful
        """
        self.submission_count += 1
        result = self._submit_to_api(post_data)
        if result.success:
            bt.logging.trace(f"[APIClient] Submitted post '{post_data.get('post_id')}' successfully")
        else:
            error_msg = f"[APIClient] API submission failed for post '{post_data.get('post_id')}'"
            if result.error_status:
                error_msg += f" (HTTP {result.error_status})"
            bt.logging.warning(error_msg)
        return result
