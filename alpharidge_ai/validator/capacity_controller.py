"""Running the capacity controller: measure the return, arm the rule, report a step.

The rule itself is pure and lives in mechanism/controller.py. This is the part that
reads prices off the chain, keeps a day's samples together, and persists what it has
seen so a restart does not reset the arming clock.

A proposal reported from here changes nothing. Capacity moves when an operator
publishes a profile carrying the new value, which is the only path there is.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import bittensor as bt

from alpharidge_ai import config
from alpharidge_ai.mechanism import controller
from alpharidge_ai.mechanism import profile as mp
from alpharidge_ai.utils import burn


def _default_path() -> Path:
    return Path(getattr(config, "CONTROLLER_STATE_LOCATION",
                        str(Path(__file__).resolve().parent.parent / ".controller_state.json")))


@dataclass
class CapacityController:
    path: Path = field(default_factory=_default_path)
    state: controller.ControllerState = field(default_factory=controller.ControllerState)
    day: Optional[int] = None
    samples: List[float] = field(default_factory=list)
    last_capacity: Optional[float] = None

    # ---- persistence ----

    def load(self) -> None:
        try:
            if not self.path.exists():
                return
            data = json.loads(self.path.read_text()) or {}
            self.state = controller.ControllerState.from_dict(data.get("state"))
            self.day = data.get("day")
            self.samples = [float(x) for x in (data.get("samples") or [])]
            lc = data.get("last_capacity")
            self.last_capacity = None if lc is None else float(lc)
        except Exception as e:
            bt.logging.warning(f"[CONTROLLER] Could not read state: {e}")

    def save(self) -> None:
        try:
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"state": self.state.to_dict(), "day": self.day,
                                       "samples": self.samples,
                                       "last_capacity": self.last_capacity}))
            tmp.replace(self.path)
        except Exception as e:
            bt.logging.warning(f"[CONTROLLER] Could not persist state: {e}")

    # ---- measurement ----

    def measure(self, profile) -> Optional[float]:
        """The return on a point this epoch, or None when the chain cannot be read.

        Capacity is in points per epoch, so the emission it is compared against is the
        miner emission over one epoch, not one block.
        """
        try:
            emission = burn.get_miner_alpha_per_block() * float(config.BLOCK_LENGTH)
            return controller.roi(emission, profile.settlement.C, burn.alpha_usd(),
                                  profile.controller.cost_per_point)
        except Exception as e:
            bt.logging.debug(f"[CONTROLLER] Could not measure the return: {e}")
            return None

    # ---- the daily step ----

    def observe(self, epoch: int, profile) -> Optional[controller.Proposal]:
        """Fold this epoch into the current day, and close the day when it rolls.

        The rule is stated per day, so epochs are averaged rather than each one being
        treated as an observation: a single epoch's price is noise the band should not
        see.
        """
        if profile is None:
            return None

        today = int(epoch) // mp.EPOCHS_PER_DAY
        if self.day is None:
            self.day = today

        if today == self.day:
            value = self.measure(profile)
            if value is not None:
                self.samples.append(value)
            return None

        proposal = None
        if self.samples:
            mean = sum(self.samples) / len(self.samples)
            rule = profile.controller
            self.state, proposal = controller.advance(
                self.state, day=self.day, roi_today=mean,
                capacity=profile.settlement.C,
                roi_lo=rule.roi_lo, roi_hi=rule.roi_hi, arm_days=rule.arm_days,
                max_step=rule.max_step, gap_days=rule.gap_days,
                last_capacity=self.last_capacity)
            self.last_capacity = float(profile.settlement.C)
            bt.logging.info(
                f"[CONTROLLER] day={self.day} roi={mean:.2f} "
                f"ema={self.state.roi_ema:.2f} band=[{rule.roi_lo},{rule.roi_hi}] "
                f"days_outside={self.state.days_outside}")

        self.day = today
        self.samples = []
        value = self.measure(profile)
        if value is not None:
            self.samples.append(value)
        self.save()

        if proposal is not None:
            bt.logging.warning(
                f"[CONTROLLER] step armed: capacity {proposal.from_capacity:.0f} -> "
                f"{proposal.to_capacity:.0f} ({'up' if proposal.is_increase else 'down'}) "
                f"on roi_ema={proposal.roi_ema:.2f}; reporting, not applying")
        return proposal
