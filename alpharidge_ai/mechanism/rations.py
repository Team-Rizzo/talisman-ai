"""Earned rations: how much work dispatch leases a UID each epoch.

A ration is per-UID and identity-agnostic, driven only by that UID's own demonstrated
delivery. It advances on work that cleared the floor, never on what was submitted or on
how many slots were occupied.

Every UID keeps a floor, so a UID with no history is treated as small rather than as
failed, and a new one starts above that floor and decays to it.

Pure: state in, state out. Persistence and dispatch belong to the caller.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

# Consecutive epochs of full fill required before a UID is probed with more.
FILL_GATE_EPOCHS = 3


@dataclass(frozen=True)
class MinerRation:
    ema: float = 0.0
    fills: Tuple[float, ...] = ()
    first_epoch: Optional[int] = None
    last_epoch: Optional[int] = None

    def to_dict(self) -> dict:
        return {"ema": self.ema, "fills": list(self.fills),
                "first_epoch": self.first_epoch, "last_epoch": self.last_epoch}

    @staticmethod
    def from_dict(d: Mapping) -> "MinerRation":
        return MinerRation(
            ema=float(d.get("ema", 0.0)),
            fills=tuple(float(f) for f in (d.get("fills") or ()))[-FILL_GATE_EPOCHS:],
            first_epoch=(None if d.get("first_epoch") is None else int(d["first_epoch"])),
            last_epoch=(None if d.get("last_epoch") is None else int(d["last_epoch"])),
        )


def observe(state: Optional[MinerRation], *, epoch: int, validated: float,
            dispatched: float, alpha_epoch: float) -> MinerRation:
    """Fold one epoch of outcome into a UID's state.

    `validated` is work that cleared the deterministic floor. Nothing else belongs
    here: not submissions, not acknowledgements, not occupied slots.
    """
    state = state or MinerRation()
    validated = max(0.0, float(validated))
    dispatched = max(0.0, float(dispatched))
    a = min(1.0, max(0.0, float(alpha_epoch)))

    ema = (1.0 - a) * state.ema + a * validated
    fill = (validated / dispatched) if dispatched > 0 else 0.0
    fills = (state.fills + (min(1.0, fill),))[-FILL_GATE_EPOCHS:]

    return MinerRation(
        ema=ema,
        fills=fills,
        first_epoch=state.first_epoch if state.first_epoch is not None else int(epoch),
        last_epoch=int(epoch),
    )


def seen(state: Optional[MinerRation], epoch: int) -> MinerRation:
    """Register a UID without recording delivery, so its boost clock starts."""
    if state is not None and state.first_epoch is not None:
        return state
    return MinerRation(ema=(state.ema if state else 0.0),
                       fills=(state.fills if state else ()),
                       first_epoch=int(epoch),
                       last_epoch=(state.last_epoch if state else None))


def is_saturated(state: Optional[MinerRation], fill_gate: float) -> bool:
    """A UID that has filled everything it was given, for long enough to believe it."""
    if state is None or len(state.fills) < FILL_GATE_EPOCHS:
        return False
    return all(f >= fill_gate for f in state.fills)


def floor_for(state: Optional[MinerRation], *, epoch: int, explore_epoch: float,
              boost_epoch: float, boost_days: int, epochs_per_day: int) -> float:
    """The ration a UID gets regardless of history.

    A new UID starts at the boost and decays linearly to the explore floor. The floor
    tranche is protected from supply scaling, which is what lets a newcomer climb even
    when incumbents can absorb everything.
    """
    explore = max(0.0, float(explore_epoch))
    boost = max(explore, float(boost_epoch))
    span = int(boost_days) * int(epochs_per_day)
    if span <= 0 or boost <= explore:
        return explore
    if state is None or state.first_epoch is None:
        return boost

    age = int(epoch) - int(state.first_epoch)
    if age < 0:
        return boost
    if age >= span:
        return explore
    return boost - (boost - explore) * (age / span)


def want(state: Optional[MinerRation], *, probe_epoch: float,
         fill_gate: float, cap_epoch: float) -> float:
    """A UID's growth bid: its own delivery, probed upward only once saturated.

    A UID that cannot fill what it already has bids its delivery unchanged, so it drops
    out of the growth path on its own rather than being ranked out of it.

    The bid excludes the floor. The floor is added once, in allocate(), so a capped
    newcomer tranche cannot return through the growth path.
    """
    ema = state.ema if state else 0.0
    bid = ema * float(probe_epoch) if is_saturated(state, fill_gate) else ema
    return min(float(cap_epoch), max(0.0, bid))


def allocate(wants: Mapping[str, float], floors: Mapping[str, float], supply: float,
             *, explore_epoch: float = 0.0,
             boost_tranche_max: float = 1.0) -> Dict[str, float]:
    """Split the articles available this tick across UIDs.

    Allocation above the floor is linear in demonstrated delivery. Anything concave, or
    granted equally per UID, pays for holding a UID rather than for doing work, which is
    the subsidy this design removes.
    """
    supply = max(0.0, float(supply))
    keys = list(wants.keys())
    if not keys or supply <= 0.0:
        return {k: 0.0 for k in keys}

    floors = {k: max(0.0, float(floors.get(k, 0.0))) for k in keys}
    explore = max(0.0, float(explore_epoch))

    # Bound the newcomer tranche so a registration wave cannot starve incumbents.
    boost_total = sum(max(0.0, floors[k] - explore) for k in keys)
    allowed = max(0.0, float(boost_tranche_max)) * supply
    if boost_total > allowed and boost_total > 0:
        shrink = allowed / boost_total
        floors = {k: explore + (floors[k] - explore) * shrink if floors[k] > explore
                  else floors[k] for k in keys}

    floor_total = sum(floors.values())
    if floor_total > supply:
        # Cannot honour every floor; hold their relative sizes.
        shrink = supply / floor_total
        return {k: floors[k] * shrink for k in keys}

    headroom = supply - floor_total
    excess = {k: max(0.0, float(wants.get(k, 0.0)) - floors[k]) for k in keys}
    demand = sum(excess.values())
    scale = min(1.0, headroom / demand) if demand > 0 else 0.0

    return {k: floors[k] + excess[k] * scale for k in keys}


@dataclass
class RationBook:
    """Per-validator ration state. Local by construction: a ration never sets a weight,
    so two validators holding different rations for the same UID is harmless and needs
    no agreement between them."""

    states: Dict[str, MinerRation] = field(default_factory=dict)

    def observe(self, hotkey: str, *, epoch: int, validated: float,
                dispatched: float, alpha_epoch: float) -> None:
        self.states[hotkey] = observe(self.states.get(hotkey), epoch=epoch,
                                      validated=validated, dispatched=dispatched,
                                      alpha_epoch=alpha_epoch)

    def seen(self, hotkey: str, epoch: int) -> None:
        self.states[hotkey] = seen(self.states.get(hotkey), epoch)

    def plan(self, hotkeys: Sequence[str], *, epoch: int, supply: float,
             explore_epoch: float, boost_epoch: float, boost_days: int,
             epochs_per_day: int, probe_epoch: float, fill_gate: float,
             cap_epoch: float, boost_tranche_max: float) -> Dict[str, float]:
        """Rations for one tick, in articles per epoch."""
        floors, wants = {}, {}
        for hk in hotkeys:
            state = self.states.get(hk)
            f = floor_for(state, epoch=epoch, explore_epoch=explore_epoch,
                          boost_epoch=boost_epoch, boost_days=boost_days,
                          epochs_per_day=epochs_per_day)
            floors[hk] = f
            wants[hk] = want(state, probe_epoch=probe_epoch,
                             fill_gate=fill_gate, cap_epoch=cap_epoch)
        return allocate(wants, floors, supply, explore_epoch=explore_epoch,
                        boost_tranche_max=boost_tranche_max)

    def to_dict(self) -> dict:
        return {hk: st.to_dict() for hk, st in self.states.items()}

    @staticmethod
    def from_dict(d: Mapping) -> "RationBook":
        return RationBook({str(k): MinerRation.from_dict(v) for k, v in (d or {}).items()})

    def prune(self, keep: Iterable[str]) -> None:
        keep = set(keep)
        for hk in [k for k in self.states if k not in keep]:
            del self.states[hk]
