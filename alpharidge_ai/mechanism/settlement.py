"""Settlement: what each UID is paid, and what is burned.

    pay_i = E · w_i / max(W, C)
    burn  = max(0, 1 - W/C)

Below capacity every point pays the same E/C, so a UID's pay depends only on its own
work. At or above capacity it is a proportional split of the whole emission. The two
branches meet at W = C, so nothing changes regime discontinuously.

Burn is whatever capacity went unfilled. It is denominated in points, not in a currency,
so the alpha price cannot move the target that work is measured against.

Pure: no chain reads, no config, no clock.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence


@dataclass(frozen=True)
class Settlement:
    shares: Dict[str, float]
    burn: float
    work: float
    capacity: float

    @property
    def paid(self) -> float:
        return sum(self.shares.values())


def settle(work: Mapping[str, float], capacity: float) -> Settlement:
    """Split one epoch's emission across UIDs by verified work.

    `work` is per-UID verified points, already carrying the quality multiplier.
    """
    capacity = float(capacity)
    if capacity <= 0.0:
        raise ValueError(f"capacity must be positive, got {capacity}")

    cleaned = {k: max(0.0, float(v)) for k, v in (work or {}).items()}
    total = sum(cleaned.values())
    denominator = max(total, capacity)

    shares = {k: v / denominator for k, v in cleaned.items()}
    burn = max(0.0, 1.0 - total / capacity)
    return Settlement(shares=shares, burn=burn, work=total, capacity=capacity)


def weight_vector(shares: Mapping[str, float], burn: float,
                  hotkeys: Sequence[str], burn_uid: int,
                  size: Optional[int] = None) -> list:
    """Lay the shares onto a UID-indexed vector, with the remainder on the burn UID.

    A hotkey no longer in the metagraph is dropped, and whatever it would have been
    paid stays unallocated, so it is burned rather than redistributed.
    """
    n = int(size if size is not None else len(hotkeys))
    weights = [0.0] * n

    index = {hk: i for i, hk in enumerate(hotkeys)}
    for hk, share in shares.items():
        uid = index.get(hk)
        if uid is not None and 0 <= uid < n:
            weights[uid] += max(0.0, float(share))

    allocated = sum(weights)
    if 0 <= int(burn_uid) < n:
        weights[int(burn_uid)] += max(0.0, min(1.0, 1.0 - allocated))
    return weights


def capacity_for_continuity(alpha_emission: float, alpha_usd: float,
                            usd_per_point: float) -> float:
    """The capacity that leaves per-point pay unchanged at cutover.

    Used once, to set the starting capacity from what the previous pegged mechanism was
    already paying. Afterwards capacity only moves by controller steps.
    """
    if usd_per_point <= 0:
        raise ValueError("usd_per_point must be positive")
    return float(alpha_emission) * float(alpha_usd) / float(usd_per_point)
