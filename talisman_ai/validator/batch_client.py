# talisman_ai/validator/batch_client.py
import asyncio
from typing import Any, Dict, List, Callable, Optional, Union
import httpx
import bittensor as bt
import time
import sys
import os

from talisman_ai import config

# Auth utilities for creating signed headers (defined inline, not imported from API)
def create_auth_message(timestamp=None):
    """Create a standardized authentication message"""
    if timestamp is None:
        timestamp = time.time()
    return f"talisman-ai-auth:{int(timestamp)}"

def sign_message(wallet, message):
    """Sign a message with the wallet's hotkey"""
    signature = wallet.hotkey.sign(message)
    return signature.hex()

# Type alias for the batch callback function
# Signature: (batch_id: int, batch: List[Dict[str, Any]]) -> Union[asyncio.Future, None, Any]
# The callback can be sync or async; async callbacks are automatically awaited
OnBatch = Callable[[int, List[Dict[str, Any]]], Union[asyncio.Future, None, Any]]

class BatchClient:
    """
    Periodically polls a FastAPI endpoint for the current validation batch.
    
    The BatchClient continuously polls the `/v1/batch` endpoint at a configurable
    interval. When a new batch is detected (identified by batch_id), it calls the
    provided `on_batch` callback with the batch_id and batch data exactly once.
    
    Each batch contains a list of miner entries, where each entry includes:
    - hotkey: The miner's hotkey identifier
    - posts: List of post submissions to validate
    - total_posts: Total count of posts for this miner
    
    The client tracks the last processed batch_id to avoid duplicate processing
    and handles HTTP errors gracefully by logging warnings and continuing to poll.
    
    This class is async-only. The caller must run this in an active event loop,
    typically by creating a background task: `asyncio.create_task(client.run(callback))`.
    To stop polling, cancel the task that awaited `run()`.
    """

    def __init__(
        self,
        api_url: Optional[str] = None,
        poll_seconds: Optional[int] = None,
        http_timeout: Optional[float] = None,
        wallet: Optional[bt.wallet] = None,
    ):
        """
        Initialize the BatchClient with configuration.
        
        Args:
            api_url: Base URL for the miner API. Defaults to MINER_API_URL env var
                    or "http://127.0.0.1:8000"
            poll_seconds: Seconds between poll attempts. Defaults to BATCH_POLL_SECONDS
                         env var or 10 seconds
            http_timeout: HTTP request timeout in seconds. Defaults to BATCH_HTTP_TIMEOUT
                         env var or 10.0 seconds
            wallet: Optional Bittensor wallet for authentication. If provided, requests will include signed auth headers.
        """
        self.api_url = api_url or config.MINER_API_URL
        self.endpoint = f"{self.api_url}/v1/batch"
        self.poll_seconds = int(poll_seconds or config.BATCH_POLL_SECONDS)
        self.http_timeout = float(http_timeout or config.BATCH_HTTP_TIMEOUT)
        self.wallet = wallet

        self._last_batch_id: Optional[int] = None
        self._running: bool = False

    def _create_auth_headers(self) -> Dict[str, str]:
        """Create authentication headers if wallet is available"""
        headers = {}
        if self.wallet:
            try:
                timestamp = time.time()
                message = create_auth_message(timestamp)
                signature = sign_message(self.wallet, message)
                headers = {
                    "X-Auth-SS58Address": self.wallet.hotkey.ss58_address,
                    "X-Auth-Signature": signature,
                    "X-Auth-Message": message,
                    "X-Auth-Timestamp": str(timestamp)
                }
                bt.logging.debug(f"[BATCH] Created authentication headers for hotkey: {self.wallet.hotkey.ss58_address}")
            except Exception as e:
                bt.logging.warning(f"[BATCH] Failed to create auth headers: {e}, proceeding without auth")
        return headers

    async def _fetch_once(self) -> Dict[str, Any]:
        """
        Fetch the current batch from the API endpoint.
        
        Returns:
            Dictionary containing:
            - available: Boolean indicating if a batch is available
            - batch_id: Integer identifier for the batch (if available)
            - batch: List of miner entries with posts to validate
            - timestamp: Unix timestamp when batch was created
            - Other metadata fields
            
        Raises:
            httpx.HTTPStatusError: If the HTTP request fails with a non-2xx status
        """
        bt.logging.debug(f"[BATCH] Fetching batch from {self.endpoint} (timeout={self.http_timeout}s)")
        headers = self._create_auth_headers()
        async with httpx.AsyncClient(timeout=self.http_timeout) as client:
            r = await client.get(self.endpoint, headers=headers)
            r.raise_for_status()
            data = r.json()
            bt.logging.debug(f"[BATCH] Fetch response: available={data.get('available')}, batch_id={data.get('batch_id')}")
            return data

    async def run(self, on_batch: OnBatch):
        """
        Start polling for batches until cancelled.
        
        This method runs indefinitely, polling the API endpoint at the configured
        interval. When a new batch is detected, it calls `on_batch(batch_id, batch)`.
        The callback can be a regular function or an async coroutine; both are
        handled automatically.
        
        The polling loop handles errors gracefully:
        - HTTP errors are logged as warnings and polling continues
        - Other exceptions are logged as warnings and polling continues
        - The loop can be stopped by cancelling the task that awaited this method
        
        Args:
            on_batch: Callback function that receives (batch_id: int, batch: List[Dict])
                     when a new batch is available. Can be async or sync.
        """
        self._running = True
        bt.logging.info(f"[BATCH] Polling {self.endpoint} every {self.poll_seconds}s")

        try:
            while self._running:
                try:
                    data = await self._fetch_once()

                    bt.logging.debug(f"[BATCH] Poll response: available={data.get('available')}, batch_id={data.get('batch_id')}, timestamp={data.get('timestamp')}")

                    if data.get("available"):
                        # Extract batch_id from explicit field or fall back to timestamp
                        # This handles API versions that may use either field for identification
                        batch_id = int(data.get("batch_id") or int(data.get("timestamp", 0)))
                        bt.logging.debug(f"[BATCH] Extracted batch_id={batch_id}, last_batch_id={self._last_batch_id}")
                        
                        if batch_id and batch_id != self._last_batch_id:
                            self._last_batch_id = batch_id
                            batch = data.get("batch", [])
                            total_posts = sum(h.get("total_posts", 0) for h in batch)
                            bt.logging.info(f"[BATCH] New batch {batch_id}: miners={len(batch)}, posts={total_posts}")

                            # Call the callback - handle both sync and async callbacks
                            maybe_coro = on_batch(batch_id, batch)
                            if asyncio.iscoroutine(maybe_coro):
                                await maybe_coro
                        else:
                            bt.logging.debug(f"[BATCH] Already processed batch {batch_id}")
                    else:
                        bt.logging.debug("[BATCH] No batch available yet")

                except httpx.HTTPStatusError as e:
                    bt.logging.warning(f"[BATCH] HTTP {e.response.status_code}: {e}")
                except Exception as e:
                    bt.logging.warning(f"[BATCH] Poll error: {e}")

                await asyncio.sleep(self.poll_seconds)

        except asyncio.CancelledError:
            bt.logging.info("[BATCH] Poller cancelled")
            raise
        finally:
            self._running = False
            bt.logging.info("[BATCH] Poller stopped")
