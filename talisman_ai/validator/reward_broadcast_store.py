"""
Persistent store for validator↔validator reward broadcasts.

We cache received broadcasts because:
- Validators may miss messages while offline.
- We apply rewards with a delay (e.g. apply epoch E-2).

Data model:
  last_seen_seq: {validator_hotkey: seq}
  by_epoch_by_sender: {epoch: {validator_hotkey: {uid: points}}}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import bittensor as bt

from talisman_ai import config


def _default_path() -> Path:
    return Path(getattr(config, "BROADCAST_STATE_LOCATION", str(Path(__file__).resolve().parent.parent / ".broadcast_state.json")))


@dataclass
class RewardBroadcastStore:
    path: Path = field(default_factory=_default_path)
    keep_epochs: int = 3
    last_seen_seq: Dict[str, int] = field(default_factory=dict)
    by_epoch_by_sender: Dict[int, Dict[str, Dict[int, int]]] = field(default_factory=dict)

    # ---------------------------------------------------------------------
    # Persistence
    # ---------------------------------------------------------------------
    def load(self) -> None:
        try:
            if not self.path.exists():
                return
            data = json.loads(self.path.read_text())
            self.last_seen_seq = {str(k): int(v) for k, v in (data.get("last_seen_seq") or {}).items()}
            raw = data.get("by_epoch_by_sender") or {}
            parsed: Dict[int, Dict[str, Dict[int, int]]] = {}
            for epoch_s, senders in raw.items():
                epoch = int(epoch_s)
                if not isinstance(senders, dict):
                    continue
                parsed[epoch] = {}
                for sender, uid_points in senders.items():
                    if not isinstance(uid_points, dict):
                        continue
                    parsed[epoch][str(sender)] = {int(uid): int(pts) for uid, pts in uid_points.items()}
            self.by_epoch_by_sender = parsed
        except Exception as e:
            bt.logging.debug(f"[BROADCAST] Failed to load state {self.path}: {e}")

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "last_seen_seq": dict(self.last_seen_seq),
                "by_epoch_by_sender": {
                    str(epoch): {sender: {str(uid): int(pts) for uid, pts in uid_points.items()}
                                 for sender, uid_points in senders.items()}
                    for epoch, senders in self.by_epoch_by_sender.items()
                },
            }
            self.path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            bt.logging.debug(f"[BROADCAST] Failed to save state {self.path}: {e}")

    # ---------------------------------------------------------------------
    # Ingest
    # ---------------------------------------------------------------------
    def ingest(self, *, sender_hotkey: str, epoch: int, seq: int, uid_points: Dict[int, int]) -> Tuple[bool, str]:
        """
        Ingest a broadcast. Returns (accepted, reason).
        """
        sender = str(sender_hotkey)
        epoch_i = int(epoch)
        seq_i = int(seq)

        last = int(self.last_seen_seq.get(sender, -1))
        if seq_i <= last:
            return False, f"duplicate_or_old_seq(last={last}, got={seq_i})"

        cleaned = {int(uid): int(pts) for uid, pts in (uid_points or {}).items() if int(pts) > 0}
        if not cleaned:
            # Still advance last_seen_seq to prevent spam with empty payloads.
            self.last_seen_seq[sender] = seq_i
            return False, "empty_payload"

        # Store sender contribution for this epoch.
        self.by_epoch_by_sender.setdefault(epoch_i, {})[sender] = cleaned
        self.last_seen_seq[sender] = seq_i

        # Keep only the most recent N epochs.
        if self.keep_epochs > 0 and len(self.by_epoch_by_sender) > self.keep_epochs:
            for old_epoch in sorted(self.by_epoch_by_sender.keys())[:-self.keep_epochs]:
                self.by_epoch_by_sender.pop(old_epoch, None)

        return True, "accepted"

    # ---------------------------------------------------------------------
    # Aggregate
    # ---------------------------------------------------------------------
    def aggregate_epoch(self, epoch: int) -> Dict[int, int]:
        """
        Aggregate uid->points for a given epoch by summing across senders.
        """
        epoch_i = int(epoch)
        senders = self.by_epoch_by_sender.get(epoch_i) or {}
        agg: Dict[int, int] = {}
        for _sender, uid_points in senders.items():
            for uid, pts in uid_points.items():
                uid_i = int(uid)
                agg[uid_i] = agg.get(uid_i, 0) + int(pts)
        return agg


