"""Durable, consensus-safe reputation state.

Reputation is a substantive-weighted recency EMA of per-article graded scores, kept per
hotkey. To keep both validators identical, per-article observations are broadcast and the
UNION (local + received) is applied at a delayed epoch close in a DETERMINISTIC order
(sort by article_id, then source) — EMA is order-dependent, so arrival order must not
matter. Mirrors the reward/penalty broadcast-store pattern (delayed application, keep a
few epochs, persist to JSON).

State is authoritative on disk; losing it resets all history, so it must be persisted and
backed up like the reward store.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

from alpharidge_ai import config
from alpharidge_ai.validator import reputation as rep


def _default_path() -> Path:
    return Path(getattr(config, "REPUTATION_STATE_LOCATION",
                        str(Path(__file__).resolve().parent.parent / ".reputation_state.json")))


def _prior() -> float:
    """Cold-start reputation for an unseen hotkey. Read at the use-site (not bound at
    import) so the hourly served-config refresh takes effect without a restart."""
    try:
        return float(getattr(config, "REPUTATION_PRIOR", rep.PRIOR))
    except (TypeError, ValueError):
        return rep.PRIOR


# one observation: (article_id, graded, weight)
Obs = Tuple[int, float, float]

# Defense-in-depth bounds on ingested observations. A sender broadcasts once per epoch
# with seq == epoch, so a distant seq is a rogue or replayed payload; the volume caps
# bound how much EMA movement one sender can buy in a single epoch.
MAX_SEQ_EPOCH_SKEW = 100
MAX_OBS_PER_TARGET = 512
MAX_TARGETS_PER_SENDER = 1024


@dataclass
class ReputationStore:
    path: Path = field(default_factory=_default_path)
    keep_epochs: int = 4

    # durable per-hotkey state: hotkey -> {"r": reputation, "n": sample count}
    state: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # pending observations: epoch -> sender_hotkey -> target_hotkey -> [Obs]
    obs: Dict[int, Dict[str, Dict[str, List[Obs]]]] = field(default_factory=dict)
    finalized: List[int] = field(default_factory=list)
    # highest accepted seq per sender
    last_seen_seq: Dict[str, int] = field(default_factory=dict)

    def load(self) -> None:
        try:
            if not self.path.exists():
                return
            data = json.loads(self.path.read_text())
            self.state = {str(k): {"r": float(v["r"]), "n": int(v.get("n", 0))}
                          for k, v in (data.get("state") or {}).items()}
            self.finalized = [int(e) for e in (data.get("finalized") or [])][-64:]
            self.last_seen_seq = {str(k): int(v)
                                  for k, v in (data.get("last_seen_seq") or {}).items()}
            raw = data.get("obs") or {}
            self.obs = {int(e): {s: {t: [tuple(o) for o in lst] for t, lst in tgts.items()}
                                 for s, tgts in senders.items()}
                        for e, senders in raw.items()}
        except Exception:
            pass

    def save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({
            "state": self.state,
            "finalized": self.finalized[-64:],
            "obs": self.obs,
            "last_seen_seq": dict(self.last_seen_seq),
        }))
        tmp.replace(self.path)

    # ---- ingest ----
    def _add(self, epoch: int, sender: str, target: str, o: Obs) -> None:
        aid, g, w = int(o[0]), float(o[1]), float(o[2])
        if not (0.0 <= g <= 1.0) or not (0.0 < w <= 10.0):  # bounds guard
            return
        self.obs.setdefault(epoch, {}).setdefault(sender, {}).setdefault(target, []).append((aid, g, w))

    def record_local(self, epoch: int, self_hotkey: str, target: str, article_id: int,
                     graded: float, weight: float) -> None:
        """Own observation — buffered for aggregation and for broadcast (via export)."""
        self._add(epoch, self_hotkey, target, (article_id, graded, weight))

    def ingest(self, sender: str, epoch: int, targets: Dict[str, List[Obs]],
               seq: int = None) -> Tuple[bool, str]:
        """A peer validator's observations for an epoch. Returns (accepted, reason)."""
        sender = str(sender)
        epoch_i = int(epoch)
        if epoch_i in self.finalized:
            return False, f"epoch_already_finalized({epoch_i})"

        if seq is not None:
            seq_i = int(seq)
            if abs(seq_i - epoch_i) > MAX_SEQ_EPOCH_SKEW:
                return False, f"seq_epoch_skew(seq={seq_i}, epoch={epoch_i})"
            last = int(self.last_seen_seq.get(sender, -1))
            if seq_i <= last:
                return False, f"duplicate_or_old_seq(last={last}, got={seq_i})"
            self.last_seen_seq[sender] = seq_i

        targets = targets or {}
        kept = 0
        # Truncate rather than reject: one oversized target must not suppress the rest.
        for target in sorted(targets)[:MAX_TARGETS_PER_SENDER]:
            for o in (targets.get(target) or [])[:MAX_OBS_PER_TARGET]:
                try:
                    self._add(epoch_i, sender, str(target), o)
                    kept += 1
                except (TypeError, ValueError, IndexError):
                    continue
        if not kept:
            return False, "empty_payload"
        return True, f"accepted({kept})"

    def export(self, epoch: int, self_hotkey: str) -> Dict[str, List[Obs]]:
        """Own observations for `epoch`, to broadcast to peers."""
        return dict((self.obs.get(epoch, {}) or {}).get(self_hotkey, {}))

    # ---- finalize (delayed) ----
    def finalize(self, epoch: int, alpha: float = None) -> None:
        """Apply the union of all senders' observations for `epoch` to the EMA in a
        deterministic order. Call for a delayed epoch (e.g. E-2) so broadcasts have settled."""
        if epoch in self.finalized or epoch not in self.obs:
            return
        alpha = rep.ALPHA if alpha is None else alpha
        # union per target: dedup identical (sender, obs), then order by (article_id, sender)
        per_target: Dict[str, List[Tuple[int, str, float, float]]] = {}
        for sender, targets in self.obs[epoch].items():
            for target, lst in targets.items():
                # One observation per (article, sender): keep the worst
                # score (ties: larger weight).
                best: Dict[Tuple[int, str], Tuple[float, float]] = {}
                for aid, g, w in lst:
                    key = (aid, sender)
                    cur = best.get(key)
                    if cur is None or g < cur[0] or (g == cur[0] and w > cur[1]):
                        best[key] = (float(g), float(w))
                for (aid, sender_), (g, w) in best.items():
                    per_target.setdefault(target, []).append((aid, sender_, g, w))
        for target, rows in per_target.items():
            rows.sort(key=lambda x: (x[0], x[1]))  # (article_id, sender) — deterministic
            st = self.state.setdefault(target, {"r": _prior(), "n": 0})
            for _aid, _sender, g, w in rows:
                st["r"] = rep.update(st["r"], g, w, alpha)
                st["n"] += 1
        self.finalized.append(epoch)
        del self.obs[epoch]
        self._prune()
        self.save()

    def _prune(self) -> None:
        for e in sorted(self.obs)[:-self.keep_epochs]:
            del self.obs[e]
        self.finalized = self.finalized[-64:]

    # ---- read ----
    def reputation(self, hotkey: str) -> float:
        return self.state.get(hotkey, {}).get("r", _prior())

    def samples(self, hotkey: str) -> int:
        return int(self.state.get(hotkey, {}).get("n", 0))

    def sender_observations(self, epoch: int, sender: str) -> Dict[str, List[Obs]]:
        """One sender's buffered observations for an epoch, by target."""
        return dict((self.obs.get(int(epoch), {}) or {}).get(str(sender), {}))

    def senders(self, epoch: int) -> List[str]:
        return sorted((self.obs.get(int(epoch), {}) or {}).keys())

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        """Per-hotkey {r, n} for telemetry / emission."""
        return {k: dict(v) for k, v in self.state.items()}
