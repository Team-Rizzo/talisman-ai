"""The mechanism profile: one signed, versioned object holding every economic value.

A validator holds `current` and `next` and resolves by block height, so config and code
land together at a single activation block and a half-applied mechanism state cannot be
represented. No economic value is read from anywhere else.

Pure: no I/O, no clock, no chain access. Signature verification is injected by the
caller so this module stays importable without the wallet stack.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

SUPPORTED_SCHEMA_VERSIONS = ("1.2.0",)

# Blocks per epoch and epochs per day. Mechanism constants are quoted per day; the code
# runs per epoch. Convert here, never at a call site.
BLOCKS_PER_EPOCH = 100
BLOCKS_PER_DAY = 7200
EPOCHS_PER_DAY = BLOCKS_PER_DAY // BLOCKS_PER_EPOCH

SECONDS_PER_BLOCK = 12

_SEMVER = re.compile(r"^\d+\.\d+\.\d+$")


class ProfileError(ValueError):
    """A profile that must not be applied. The message is the reason."""


def min_lead_blocks(refresh_seconds: int) -> int:
    """Blocks a publish must lead its activation by.

    Every validator has to have fetched `next` before any validator activates it, so the
    lead must cover a full config refresh interval.
    """
    return max(1, int(refresh_seconds) // SECONDS_PER_BLOCK)


def canonical_json(payload: dict) -> str:
    """Byte-identical to the API's canonical_json. Both sides sign this exact string."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def signing_payload(raw: dict) -> str:
    """The canonical string a profile's signature covers: everything but the signature."""
    return canonical_json({k: v for k, v in raw.items() if k != "signature"})


# ---- range helpers ----------------------------------------------------------------

def _num(section: str, d: dict, key: str, lo: float, hi: float,
         *, lo_open: bool = False) -> float:
    if key not in d:
        raise ProfileError(f"{section}.{key} missing")
    try:
        v = float(d[key])
    except (TypeError, ValueError):
        raise ProfileError(f"{section}.{key} not a number: {d[key]!r}")
    if v != v:
        raise ProfileError(f"{section}.{key} is NaN")
    if (v <= lo if lo_open else v < lo) or v > hi:
        bound = "(" if lo_open else "["
        raise ProfileError(f"{section}.{key}={v} outside {bound}{lo}, {hi}]")
    return v


def _bool(section: str, d: dict, key: str, default: bool = False) -> bool:
    if key not in d:
        return default
    v = d[key]
    if isinstance(v, bool):
        return v
    raise ProfileError(f"{section}.{key} must be true or false, got {v!r}")


def _int(section: str, d: dict, key: str, lo: int, hi: int) -> int:
    if key not in d:
        raise ProfileError(f"{section}.{key} missing")
    try:
        v = int(d[key])
    except (TypeError, ValueError):
        raise ProfileError(f"{section}.{key} not an integer: {d[key]!r}")
    if v < lo or v > hi:
        raise ProfileError(f"{section}.{key}={v} outside [{lo}, {hi}]")
    return v


# ---- sections ---------------------------------------------------------------------

@dataclass(frozen=True)
class Settlement:
    C: float
    # Credit volume per article that clears the floor. Unlike the other switches this
    # withholds pay for work that failed a check rather than changing how pay is
    # computed, so it belongs on early and on its own — there is no operating state in
    # which paying for articles that failed the floor is wanted.
    floor_gating: bool = True
    live: bool = False

    @staticmethod
    def parse(d: dict) -> "Settlement":
        return Settlement(
            C=_num("settlement", d, "C", 0.0, 1e15, lo_open=True),
            floor_gating=_bool("settlement", d, "floor_gating", default=True),
            live=_bool("settlement", d, "live"),
        )


@dataclass(frozen=True)
class Emission:
    midpoint: float
    gain: float
    ceiling: float
    bonus_start: float
    bonus_full: float
    n_min: int
    ema_alpha: float

    @staticmethod
    def parse(d: dict) -> "Emission":
        start = _num("emission", d, "bonus_start", 0.0, 1.0)
        full = _num("emission", d, "bonus_full", 0.0, 1.0)
        if full < start:
            raise ProfileError(f"emission.bonus_full={full} below bonus_start={start}")
        return Emission(
            midpoint=_num("emission", d, "midpoint", 0.0, 1.0),
            gain=_num("emission", d, "gain", 1.0, 50.0),
            ceiling=_num("emission", d, "ceiling", 0.0, 3.0),
            bonus_start=start,
            bonus_full=full,
            n_min=_int("emission", d, "n_min", 0, 1_000_000),
            ema_alpha=_num("emission", d, "ema_alpha", 0.0, 1.0, lo_open=True),
        )


@dataclass(frozen=True)
class Rations:
    explore: float
    probe_day: float
    alpha_day: float
    cap: float
    slack_target: float
    fill_gate: float
    boost: float
    boost_days: int
    boost_tranche_max: float
    dispatch: bool = False

    @staticmethod
    def parse(d: dict) -> "Rations":
        explore = _num("rations", d, "explore", 0.0, 1e6, lo_open=True)
        boost = _num("rations", d, "boost", 0.0, 1e6, lo_open=True)
        if boost < explore:
            raise ProfileError(f"rations.boost={boost} below explore={explore}")
        cap = _num("rations", d, "cap", 0.0, 1e9, lo_open=True)
        if cap < explore:
            raise ProfileError(f"rations.cap={cap} below explore={explore}")
        return Rations(
            explore=explore,
            probe_day=_num("rations", d, "probe_day", 1.0, 100.0),
            alpha_day=_num("rations", d, "alpha_day", 0.0, 1.0, lo_open=True),
            cap=cap,
            slack_target=_num("rations", d, "slack_target", 0.0, 1.0),
            fill_gate=_num("rations", d, "fill_gate", 0.0, 1.0),
            boost=boost,
            boost_days=_int("rations", d, "boost_days", 0, 365),
            boost_tranche_max=_num("rations", d, "boost_tranche_max", 0.0, 1.0),
            dispatch=_bool("rations", d, "dispatch"),
        )

    @property
    def alpha_epoch(self) -> float:
        """Per-epoch EMA step equivalent to alpha_day over EPOCHS_PER_DAY steps."""
        return 1.0 - (1.0 - self.alpha_day) ** (1.0 / EPOCHS_PER_DAY)

    @property
    def probe_epoch(self) -> float:
        """Per-epoch probe factor whose compounded daily effect is probe_day."""
        return self.probe_day ** (1.0 / EPOCHS_PER_DAY)

    @property
    def explore_epoch(self) -> float:
        return self.explore / EPOCHS_PER_DAY

    @property
    def boost_epoch(self) -> float:
        return self.boost / EPOCHS_PER_DAY

    @property
    def cap_epoch(self) -> float:
        return self.cap / EPOCHS_PER_DAY


@dataclass(frozen=True)
class GraderModel:
    id: str
    weight: float


@dataclass(frozen=True)
class Oracle:
    pool_tiers: Tuple[str, ...]
    keyed_rate_pool: float
    keyed_rate_keeper: float
    claim_cap: int
    keeper_weight: float
    grader_models: Tuple[GraderModel, ...]
    schema_cutover_block: int
    live: bool = False

    @staticmethod
    def parse(d: dict) -> "Oracle":
        tiers = d.get("pool_tiers")
        if not isinstance(tiers, (list, tuple)) or not tiers:
            raise ProfileError("oracle.pool_tiers must be a non-empty list")
        if not all(isinstance(t, str) and t for t in tiers):
            raise ProfileError("oracle.pool_tiers must be strings")

        raw_models = d.get("grader_models")
        if not isinstance(raw_models, (list, tuple)) or not raw_models:
            raise ProfileError("oracle.grader_models must be a non-empty list")
        models: List[GraderModel] = []
        for i, m in enumerate(raw_models):
            if not isinstance(m, dict):
                raise ProfileError(f"oracle.grader_models[{i}] must be an object")
            mid = m.get("id")
            if not isinstance(mid, str) or not mid:
                raise ProfileError(f"oracle.grader_models[{i}].id missing")
            models.append(GraderModel(
                id=mid,
                weight=_num(f"oracle.grader_models[{i}]", m, "weight", 0.0, 1e6),
            ))
        if sum(m.weight for m in models) <= 0:
            raise ProfileError("oracle.grader_models weights sum to zero")

        return Oracle(
            pool_tiers=tuple(tiers),
            keyed_rate_pool=_num("oracle", d, "keyed_rate_pool", 0.0, 1.0),
            keyed_rate_keeper=_num("oracle", d, "keyed_rate_keeper", 0.0, 1.0),
            claim_cap=_int("oracle", d, "claim_cap", 1, 10_000),
            keeper_weight=_num("oracle", d, "keeper_weight", 0.0, 10.0),
            grader_models=tuple(models),
            schema_cutover_block=_int("oracle", d, "schema_cutover_block", 0, 2**63 - 1),
            live=_bool("oracle", d, "live"),
        )


@dataclass(frozen=True)
class Controller:
    roi_lo: float
    roi_hi: float
    arm_days: int
    max_step: float
    gap_days: int
    cost_per_point: float

    @staticmethod
    def parse(d: dict) -> "Controller":
        lo = _num("controller", d, "roi_lo", 0.0, 1e6, lo_open=True)
        hi = _num("controller", d, "roi_hi", 0.0, 1e6, lo_open=True)
        if hi <= lo:
            raise ProfileError(f"controller.roi_hi={hi} not above roi_lo={lo}")
        return Controller(
            roi_lo=lo,
            roi_hi=hi,
            arm_days=_int("controller", d, "arm_days", 1, 365),
            max_step=_num("controller", d, "max_step", 0.0, 1.0, lo_open=True),
            gap_days=_int("controller", d, "gap_days", 1, 365),
            cost_per_point=_num("controller", d, "cost_per_point", 0.0, 1e6, lo_open=True),
        )


@dataclass(frozen=True)
class MechanismProfile:
    version: int
    publish_block: int
    activation_block: int
    schema_version: str
    settlement: Settlement
    emission: Emission
    rations: Rations
    oracle: Oracle
    controller: Controller
    signature: str = ""
    raw: Dict[str, Any] = field(default_factory=dict, repr=False, compare=False)


def parse(raw: dict) -> MechanismProfile:
    """Parse and range-check a profile. Raises ProfileError with the reason."""
    if not isinstance(raw, dict):
        raise ProfileError("profile must be an object")

    schema_version = raw.get("schema_version")
    if not isinstance(schema_version, str) or not _SEMVER.match(schema_version):
        raise ProfileError(f"schema_version malformed: {schema_version!r}")
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ProfileError(f"schema_version {schema_version} not supported by this validator")

    for section in ("settlement", "emission", "rations", "oracle", "controller"):
        if not isinstance(raw.get(section), dict):
            raise ProfileError(f"{section} section missing")

    return MechanismProfile(
        version=_int("profile", raw, "version", 1, 2**63 - 1),
        publish_block=_int("profile", raw, "publish_block", 0, 2**63 - 1),
        activation_block=_int("profile", raw, "activation_block", 0, 2**63 - 1),
        schema_version=schema_version,
        settlement=Settlement.parse(raw["settlement"]),
        emission=Emission.parse(raw["emission"]),
        rations=Rations.parse(raw["rations"]),
        oracle=Oracle.parse(raw["oracle"]),
        controller=Controller.parse(raw["controller"]),
        signature=str(raw.get("signature") or ""),
        raw=dict(raw),
    )


# ---- resolution -------------------------------------------------------------------

VerifyFn = Callable[[str, str], bool]  # (canonical_message, signature_hex) -> bool


class ProfileResolver:
    """Holds `current` and `next`, and picks between them by block height.

    A profile is only ever replaced forward. Rolling back is publishing the old values
    under a higher version, so the version number never decreases and there is no edit
    path that could differ between validators.
    """

    def __init__(self, current: Optional[MechanismProfile] = None,
                 refresh_seconds: int = 3600):
        self.current = current
        self.next: Optional[MechanismProfile] = None
        self.refresh_seconds = int(refresh_seconds)

    @property
    def min_lead(self) -> int:
        return min_lead_blocks(self.refresh_seconds)

    def adopt(self, raw: dict, block: int,
              verify: Optional[VerifyFn] = None) -> Tuple[bool, str]:
        """Take a profile already in force, rather than staging it as the next one.

        A validator starting up is handed both the active profile and any staged
        successor. Routing both through offer() would put the active one in the single
        `next` slot and then overwrite it, leaving nothing in force at all.
        """
        try:
            candidate = parse(raw)
        except ProfileError as e:
            return False, f"invalid: {e}"
        if verify is not None:
            if not candidate.signature:
                return False, "unsigned"
            if not verify(signing_payload(raw), candidate.signature):
                return False, "bad_signature"
        if candidate.activation_block > int(block):
            return False, f"not_yet_active(at={candidate.activation_block})"
        if self.current is not None and candidate.version <= self.current.version:
            return False, f"stale_version(have={self.current.version})"

        self.current = candidate
        if self.next is not None and self.next.version <= candidate.version:
            self.next = None
        return True, f"adopted(version={candidate.version})"

    def offer(self, raw: dict, verify: Optional[VerifyFn] = None) -> Tuple[bool, str]:
        """Consider a fetched profile. Returns (accepted, reason).

        A rejected profile changes nothing; whatever is in force stays in force.

        The lead is measured from the profile's own signed publish block, not from
        whenever this validator happened to fetch it. Measuring from the fetch would
        make an already-active profile unadoptable, which is exactly what a restarting
        or previously-offline validator needs to pick up.
        """
        try:
            candidate = parse(raw)
        except ProfileError as e:
            return False, f"invalid: {e}"

        if verify is not None:
            if not candidate.signature:
                return False, "unsigned"
            if not verify(signing_payload(raw), candidate.signature):
                return False, "bad_signature"

        highest = max([p.version for p in (self.current, self.next) if p is not None],
                      default=0)
        if candidate.version <= highest:
            return False, f"stale_version(have={highest}, got={candidate.version})"

        lead = candidate.activation_block - candidate.publish_block
        if lead < self.min_lead:
            return False, (f"insufficient_lead(blocks={lead}, "
                           f"required={self.min_lead})")

        self.next = candidate
        return True, f"staged(version={candidate.version}, at={candidate.activation_block})"

    def resolve(self, block: int) -> Optional[MechanismProfile]:
        """The profile in force at `block`. Promotes `next` once its block is reached."""
        if self.next is not None and int(block) >= self.next.activation_block:
            self.current = self.next
            self.next = None
        return self.current
