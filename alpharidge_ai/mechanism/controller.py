"""The capacity controller: a crawling peg on the one governed number.

Capacity is the only value in the mechanism that is set rather than measured, so it
moves by rule instead of by judgment. When the return on a point sits outside a wide
band for several days running, and enough time has passed since the last move, the
controller proposes a bounded step in the direction that closes the gap.

The band is wide and the steps are few and large on purpose. Small frequent steps
track the price more closely but leave miners underwater for longer during a fast move,
and they turn every week into a negotiation.

A proposal is not a change. Capacity only moves when an operator publishes a profile
carrying the new value, with the usual activation lead.

Pure: no chain reads, no clock.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

# Days in the return average, and the corresponding EMA step.
ROI_EMA_DAYS = 7
ROI_EMA_ALPHA = 2.0 / (ROI_EMA_DAYS + 1)

DOWN = -1
UP = 1


@dataclass(frozen=True)
class ControllerState:
    roi_ema: Optional[float] = None
    days_outside: int = 0
    last_step_day: Optional[int] = None

    def to_dict(self) -> dict:
        return {"roi_ema": self.roi_ema, "days_outside": self.days_outside,
                "last_step_day": self.last_step_day}

    @staticmethod
    def from_dict(d) -> "ControllerState":
        d = d or {}
        return ControllerState(
            roi_ema=(None if d.get("roi_ema") is None else float(d["roi_ema"])),
            days_outside=int(d.get("days_outside", 0)),
            last_step_day=(None if d.get("last_step_day") is None
                           else int(d["last_step_day"])),
        )


@dataclass(frozen=True)
class Proposal:
    day: int
    direction: int
    from_capacity: float
    to_capacity: float
    roi_ema: float

    @property
    def is_increase(self) -> bool:
        return self.direction > 0


def pay_per_point(alpha_emission: float, capacity: float, alpha_usd: float) -> float:
    """What one point pays, in USD, at the current capacity and alpha price."""
    if capacity <= 0:
        raise ValueError("capacity must be positive")
    return float(alpha_emission) * float(alpha_usd) / float(capacity)


def roi(alpha_emission: float, capacity: float, alpha_usd: float,
        cost_per_point: float) -> float:
    """Return on a point against a reference cost of producing one."""
    if cost_per_point <= 0:
        raise ValueError("cost_per_point must be positive")
    return pay_per_point(alpha_emission, capacity, alpha_usd) / float(cost_per_point)


def observe(state: ControllerState, roi_today: float, *, roi_lo: float,
            roi_hi: float) -> ControllerState:
    """Fold one day's return into the average and the consecutive-days-outside count."""
    roi_today = float(roi_today)
    ema = roi_today if state.roi_ema is None else (
        (1.0 - ROI_EMA_ALPHA) * state.roi_ema + ROI_EMA_ALPHA * roi_today)
    inside = float(roi_lo) <= ema <= float(roi_hi)
    return replace(state, roi_ema=ema,
                   days_outside=0 if inside else state.days_outside + 1)


def propose(state: ControllerState, *, day: int, capacity: float,
            roi_lo: float, roi_hi: float, arm_days: int, max_step: float,
            gap_days: int) -> Optional[Proposal]:
    """A step, if the breach has lasted long enough and the last one is far enough back."""
    if state.roi_ema is None or state.days_outside < int(arm_days):
        return None
    if state.last_step_day is not None and (int(day) - state.last_step_day) < int(gap_days):
        return None

    direction = DOWN if state.roi_ema < float(roi_lo) else UP
    target = float(capacity) * (1.0 + direction * float(max_step))
    return Proposal(day=int(day), direction=direction, from_capacity=float(capacity),
                    to_capacity=target, roi_ema=state.roi_ema)


def accept(state: ControllerState, proposal: Proposal) -> ControllerState:
    """Record a proposal that was published, so the gap and arming clocks restart."""
    return replace(state, days_outside=0, last_step_day=int(proposal.day))


def observed_step(state: ControllerState, *, day: int, capacity: float,
                  last_capacity: Optional[float]) -> ControllerState:
    """Record that capacity actually moved, which is what starts the gap clock.

    A proposal is only a suggestion; the gap is between real steps. Starting the clock
    on a proposal nobody published would silence the controller for the gap while the
    breach it flagged went uncorrected.
    """
    if last_capacity is None or capacity == last_capacity:
        return state
    return replace(state, days_outside=0, last_step_day=int(day))


def advance(state: ControllerState, *, day: int, roi_today: float, capacity: float,
            roi_lo: float, roi_hi: float, arm_days: int, max_step: float,
            gap_days: int, last_capacity: Optional[float] = None):
    """One day of the controller: note any real step, observe, then propose if armed.

    Returns (state, proposal). The arming count is cleared when a proposal is made so a
    single breach does not re-arm daily, but the gap clock only moves when capacity
    itself changed.
    """
    state = observed_step(state, day=day, capacity=capacity,
                          last_capacity=last_capacity)
    state = observe(state, roi_today, roi_lo=roi_lo, roi_hi=roi_hi)
    proposal = propose(state, day=day, capacity=capacity, roi_lo=roi_lo, roi_hi=roi_hi,
                       arm_days=arm_days, max_step=max_step, gap_days=gap_days)
    if proposal is not None:
        state = replace(state, days_outside=0)
    return state, proposal
