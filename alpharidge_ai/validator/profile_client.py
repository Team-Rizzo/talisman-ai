"""Fetching, verifying and holding the mechanism profile.

The decision logic lives in mechanism/profile.py and stays pure. This is the part that
touches the network and the disk: pull the served object, check it was signed by the key
this validator already trusts, hand it to the resolver, and keep the last accepted copy
so a restart does not silently fall back to a different mechanism.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Optional

import bittensor as bt
import requests

from alpharidge_ai import config
from alpharidge_ai.mechanism import profile as mp
from alpharidge_ai.utils import attestation_crypto as ac

PROFILE_PATH_ENV = "MECHANISM_PROFILE_LOCATION"
PROFILE_ENDPOINT = "/mechanism/profile"


def _default_path() -> Path:
    return Path(getattr(config, PROFILE_PATH_ENV,
                        str(Path(__file__).resolve().parent.parent / ".mechanism_profile.json")))


def verify_signature(message: str, signature_hex: str) -> bool:
    pubkey = getattr(config, "API_ATTESTATION_PUBKEY", "") or ""
    if not pubkey:
        return False
    return ac.verify_attestation(pubkey, message, signature_hex)


class ProfileClient:
    """Holds the resolver, refreshes it on a timer, and persists what it accepted."""

    def __init__(self, path: Optional[Path] = None, *, require_signature: bool = True):
        self.path = Path(path) if path else _default_path()
        self.require_signature = require_signature
        self.resolver = mp.ProfileResolver(
            refresh_seconds=int(getattr(config, "REMOTE_CONFIG_REFRESH_SECONDS", 3600)))
        self._last_fetch = 0.0
        self._lock = threading.Lock()
        self._check_epoch_length()

    @staticmethod
    def _check_epoch_length() -> None:
        """Day-quoted constants convert to epochs against a fixed epoch length.

        That length is a mechanism constant, identical fleet-wide, so it is not read
        from local config. If this box disagrees, every ration converts wrongly here
        and nowhere else, which is worth saying out loud.
        """
        local = int(getattr(config, "BLOCK_LENGTH", mp.BLOCKS_PER_EPOCH))
        if local != mp.BLOCKS_PER_EPOCH:
            bt.logging.warning(
                f"[PROFILE] BLOCK_LENGTH={local} differs from the mechanism's "
                f"{mp.BLOCKS_PER_EPOCH} blocks per epoch; per-epoch conversions "
                f"assume {mp.EPOCHS_PER_DAY} epochs/day")

    # ---- persistence ----

    def load(self) -> None:
        try:
            if not self.path.exists():
                return
            data = json.loads(self.path.read_text())
        except Exception as e:
            bt.logging.warning(f"[PROFILE] Could not read stored profile: {e}")
            return

        for slot in ("current", "next"):
            raw = data.get(slot)
            if not raw:
                continue
            try:
                parsed = mp.parse(raw)
            except mp.ProfileError as e:
                bt.logging.warning(f"[PROFILE] Stored {slot} profile is invalid: {e}")
                continue
            setattr(self.resolver, slot, parsed)

        if self.resolver.current:
            bt.logging.info(
                f"[PROFILE] Loaded version={self.resolver.current.version} "
                f"activation_block={self.resolver.current.activation_block}")

    def save(self) -> None:
        try:
            payload = {
                "current": self.resolver.current.raw if self.resolver.current else None,
                "next": self.resolver.next.raw if self.resolver.next else None,
            }
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload))
            tmp.replace(self.path)
        except Exception as e:
            bt.logging.warning(f"[PROFILE] Could not persist profile: {e}")

    # ---- refresh ----

    def refresh(self, block: int = 0, *, force: bool = False) -> bool:
        """Fetch and consider the served profile. Returns True when one was staged."""
        interval = int(getattr(config, "REMOTE_CONFIG_REFRESH_SECONDS", 3600))
        now = time.time()
        if not force and (now - self._last_fetch) < interval:
            return False

        with self._lock:
            if not force and (time.time() - self._last_fetch) < interval:
                return False
            self._last_fetch = time.time()

            api_url = (getattr(config, "MINER_API_URL", "") or "").rstrip("/")
            if not api_url:
                return False
            try:
                resp = requests.get(f"{api_url}{PROFILE_ENDPOINT}",
                                    headers=config._build_auth_headers(), timeout=15)
                resp.raise_for_status()
                body = resp.json() or {}
            except Exception as e:
                bt.logging.warning(f"[PROFILE] Fetch failed: {e}")
                return False

            changed = False
            active = body.get("current")
            if isinstance(active, dict) and self._adopt(active, block):
                changed = True
            staged = body.get("next")
            if isinstance(staged, dict) and self._offer(staged):
                changed = True
            if changed:
                self.save()
            return changed

    def _adopt(self, raw: dict, block: int) -> bool:
        """Take the profile the API says is already active."""
        verify = verify_signature if self.require_signature else None
        accepted, reason = self.resolver.adopt(raw, int(block), verify=verify)
        if accepted:
            bt.logging.info(f"[PROFILE] {reason}")
        elif not reason.startswith(("stale_version", "not_yet_active")):
            bt.logging.warning(f"[PROFILE] Rejected active profile: {reason}")
        return accepted

    def _offer(self, raw: dict) -> bool:
        verify = verify_signature if self.require_signature else None
        accepted, reason = self.resolver.offer(raw, verify=verify)
        if accepted:
            bt.logging.info(f"[PROFILE] {reason}")
        elif not reason.startswith("stale_version"):
            bt.logging.warning(f"[PROFILE] Rejected: {reason}")
        return accepted

    # ---- read ----

    def resolve(self, block: int) -> Optional[mp.MechanismProfile]:
        """The profile in force at `block`, or None while none has been accepted."""
        previous = self.resolver.current
        active = self.resolver.resolve(block)
        if active is not None and active is not previous:
            bt.logging.info(
                f"[PROFILE] Activated version={active.version} at block={block}")
            self.save()
        return active
