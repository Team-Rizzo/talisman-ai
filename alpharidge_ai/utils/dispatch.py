"""
Coverage-then-depth article allocator (RFC 2026-06-28).

Pure assignment logic, separated from the validator so it can be tested in
isolation. Given the live miner UIDs and a window-aware tracker, it returns
``(uid, batch_index)`` assignments:

  1. Coverage pass — every live miner that has not been covered this epoch and
     has a free window slot gets exactly one batch (the coverage floor — every live
     miner is scored each epoch).
  2. Depth pass — remaining batches go to miners with window headroom, highest
     window first, round-robin so a single miner can't drain the queue before
     others get a turn.

It is intentionally **fully read-only** on the tracker — coverage is marked by the
caller on *actual dispatch* (not here), and the real per-miner reservation
(``try_acquire``/``release``) stays in the validator's dispatch coroutine, so a
pending-task-cap truncation can neither leak a reserved slot nor mark a miner
covered without sending it work. ``provisional`` mirrors what those acquisitions
will be so we don't assign a miner more than ``floor(window)`` within one tick.
"""

import random
from typing import Dict, List, Optional, Sequence, Tuple


def _slot_limit(tracker, hotkey: str) -> int:
    """How many batches a UID may hold at once this tick.

    Under earned rations the cap comes from what the UID has delivered, not from the
    adaptive window, so a UID whose ration exceeds one batch can hold several.
    """
    # Only when a ration is actually in force. The source is installed at startup and
    # returns None until its switch is published, so its presence says nothing.
    ration = getattr(tracker, "ration_for", None)
    if ration is not None and ration(hotkey) is not None:
        return max(1, int(tracker.batches_per_epoch(hotkey)))
    # Selection and reservation must reach the same number, so ask the tracker for its
    # limit rather than reconstructing it from one of the inputs.
    limit = getattr(tracker, "inflight_limit", None)
    if limit is not None:
        return max(1, int(limit(hotkey)))
    return max(1, int(tracker.window(hotkey)))


def coverage_depth_select(
    live_uids: Sequence[int],
    hotkeys: Sequence[str],
    tracker,
    epoch: int,
    n_batches: int,
    rng: Optional[random.Random] = None,
) -> List[Tuple[int, int]]:
    # Serve in a randomized order so priority does not track UID: when a tick has
    # fewer batches than uncovered targets the front of the list is served first,
    # and window ties in the depth pass break by list order. Selection is a local,
    # non-consensus choice, so a fresh shuffle each tick is safe.
    order = list(live_uids)
    (rng or random).shuffle(order)

    provisional: Dict[str, int] = {}

    def has_slot(uid: int) -> bool:
        hk = hotkeys[uid]
        return tracker.inflight(hk) + provisional.get(hk, 0) < _slot_limit(tracker, hk)

    def take(uid: int) -> None:
        hk = hotkeys[uid]
        provisional[hk] = provisional.get(hk, 0) + 1

    assignments: List[Tuple[int, int]] = []
    bi = 0

    # Coverage pass.
    for uid in order:
        if bi >= n_batches:
            break
        hk = hotkeys[uid]
        if tracker.covered_epoch(hk) < epoch and has_slot(uid):
            assignments.append((uid, bi))
            take(uid)
            bi += 1

    # Depth pass.
    if bi < n_batches:
        depth_order = sorted(order, key=lambda u: tracker.window(hotkeys[u]), reverse=True)
        while bi < n_batches:
            progressed = False
            for uid in depth_order:
                if bi >= n_batches:
                    break
                if has_slot(uid):
                    assignments.append((uid, bi))
                    take(uid)
                    bi += 1
                    progressed = True
            if not progressed:
                break  # every live window is full; remaining batches retry next tick

    return assignments
