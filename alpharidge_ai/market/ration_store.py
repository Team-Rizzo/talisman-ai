"""Durable ration state for one validator.

Rations are local. A ration never sets a weight, so two validators holding different
rations for the same UID is harmless and needs no agreement between them. That is why
this persists like the batch sizes it replaces, and is never broadcast.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence

import bittensor as bt

from alpharidge_ai import config
from alpharidge_ai.mechanism import profile as mp
from alpharidge_ai.mechanism import rations


def _default_path() -> Path:
    return Path(getattr(config, "RATION_STATE_LOCATION",
                        str(Path(__file__).resolve().parent.parent / ".ration_state.json")))


@dataclass
class RationStore:
    path: Path = field(default_factory=_default_path)
    book: rations.RationBook = field(default_factory=rations.RationBook)

    def load(self) -> None:
        try:
            if self.path.exists():
                self.book = rations.RationBook.from_dict(
                    json.loads(self.path.read_text()) or {})
        except Exception as e:
            bt.logging.warning(f"[RATION] Could not read state: {e}")

    def save(self) -> None:
        try:
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.book.to_dict()))
            tmp.replace(self.path)
        except Exception as e:
            bt.logging.warning(f"[RATION] Could not persist state: {e}")

    def observe(self, hotkey: str, *, epoch: int, validated: float, dispatched: float,
                profile: Optional[mp.MechanismProfile]) -> None:
        """Record one batch outcome. `validated` is work that cleared the floor."""
        if profile is None:
            return
        self.book.observe(hotkey, epoch=epoch, validated=validated,
                          dispatched=dispatched,
                          alpha_epoch=profile.rations.alpha_epoch)

    def plan(self, hotkeys: Sequence[str], *, epoch: int, supply: float,
             profile: Optional[mp.MechanismProfile]) -> Dict[str, float]:
        """Rations for this tick, in articles per epoch. Empty without a profile."""
        if profile is None or not hotkeys:
            return {}
        r = profile.rations
        return self.book.plan(
            hotkeys, epoch=epoch, supply=supply,
            explore_epoch=r.explore_epoch, boost_epoch=r.boost_epoch,
            boost_days=r.boost_days, epochs_per_day=mp.EPOCHS_PER_DAY,
            probe_epoch=r.probe_epoch, fill_gate=r.fill_gate,
            cap_epoch=r.cap_epoch, boost_tranche_max=r.boost_tranche_max)

    def prune(self, live_hotkeys) -> None:
        self.book.prune(live_hotkeys)
