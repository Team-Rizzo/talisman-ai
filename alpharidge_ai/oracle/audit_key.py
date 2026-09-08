"""This validator's audit key.

Unique to this validator, and never leaves the process. Generated on first use and
persisted locally, so there is nothing for an operator to distribute.

Rotating it is deleting the file, or setting AUDIT_KEY to something new.
"""
from __future__ import annotations

import os
import secrets
import stat
from pathlib import Path

import bittensor as bt

from alpharidge_ai import config

KEY_BYTES = 32


def _default_path() -> Path:
    return Path(getattr(config, "AUDIT_KEY_LOCATION",
                        str(Path(__file__).resolve().parent.parent / ".audit_key")))


def load(path: Path = None) -> bytes:
    """Return this validator's key, generating and persisting one if needed."""
    from_env = os.getenv("AUDIT_KEY")
    if from_env:
        try:
            return bytes.fromhex(from_env.strip())
        except ValueError:
            return from_env.strip().encode("utf-8")

    path = Path(path) if path else _default_path()
    try:
        if path.exists():
            stored = path.read_text().strip()
            if stored:
                return bytes.fromhex(stored)
    except Exception as e:
        bt.logging.warning(f"[AUDIT] Could not read the audit key: {e}")

    key = secrets.token_bytes(KEY_BYTES)
    try:
        path.write_text(key.hex())
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        bt.logging.info("[AUDIT] Generated a new audit key")
    except Exception as e:
        # Still usable this run; selection would change on restart, which only costs
        # coverage, never correctness.
        bt.logging.warning(f"[AUDIT] Could not persist the audit key: {e}")
    return key
