"""Earned rations: growth on validated delivery only, protected floors, linear split."""

import pytest

from alpharidge_ai.mechanism import rations as r

EPOCHS_PER_DAY = 72
ALPHA_E = 1.0 - (1.0 - 0.5) ** (1.0 / EPOCHS_PER_DAY)
PROBE_E = 2.0 ** (1.0 / EPOCHS_PER_DAY)
EXPLORE_E = 25.0 / EPOCHS_PER_DAY
BOOST_E = 200.0 / EPOCHS_PER_DAY
CAP_E = 5000.0 / EPOCHS_PER_DAY
FILL_GATE = 0.97
BOOST_DAYS = 14


def run(validated, dispatched, epochs=200, start=0):
    """Drive one UID for `epochs` epochs and return its final state."""
    state = None
    for e in range(start, start + epochs):
        state = r.observe(state, epoch=e, validated=validated,
                          dispatched=dispatched, alpha_epoch=ALPHA_E)
    return state


POST_BOOST_EPOCH = (BOOST_DAYS + 5) * EPOCHS_PER_DAY


def want_of(state):
    return r.want(state, probe_epoch=PROBE_E, fill_gate=FILL_GATE, cap_epoch=CAP_E)


def ration_of(state, epoch=POST_BOOST_EPOCH, supply=10_000.0):
    """What one UID is actually leased, floor included."""
    floor = r.floor_for(state, epoch=epoch, explore_epoch=EXPLORE_E,
                        boost_epoch=BOOST_E, boost_days=BOOST_DAYS,
                        epochs_per_day=EPOCHS_PER_DAY)
    return r.allocate({"m": want_of(state)}, {"m": floor}, supply=supply,
                      explore_epoch=EXPLORE_E, boost_tranche_max=0.05)["m"]


# ---- standing advances on validated delivery ---------------------------------------

def test_occupying_slots_without_delivering_grows_nothing():
    """Filling every slot with work that fails the floor earns nothing."""
    padder = run(validated=0.0, dispatched=50.0)
    assert padder.ema == pytest.approx(0.0)
    assert want_of(padder) == pytest.approx(0.0)
    assert ration_of(padder) == pytest.approx(EXPLORE_E)


def test_partial_junk_only_earns_the_valid_part():
    honest = run(validated=20.0, dispatched=20.0)
    padder = run(validated=20.0, dispatched=50.0)
    assert padder.ema == pytest.approx(honest.ema)


def test_junk_blocks_the_probe_gate():
    """A UID that cannot fill what it holds does not get probed with more."""
    padder = run(validated=20.0, dispatched=50.0)
    assert not r.is_saturated(padder, FILL_GATE)
    assert want_of(padder) == pytest.approx(padder.ema)


# ---- growth -----------------------------------------------------------------------

def test_a_saturated_miner_is_probed_upward():
    full = run(validated=30.0, dispatched=30.0)
    assert r.is_saturated(full, FILL_GATE)
    assert want_of(full) > full.ema


def test_the_probe_compounds_to_the_daily_rate():
    assert PROBE_E ** EPOCHS_PER_DAY == pytest.approx(2.0)


def test_growth_needs_sustained_fill_not_one_good_epoch():
    state = run(validated=0.0, dispatched=30.0, epochs=10)
    state = r.observe(state, epoch=99, validated=30.0, dispatched=30.0,
                      alpha_epoch=ALPHA_E)
    assert not r.is_saturated(state, FILL_GATE)


def test_a_quiet_miner_deflates_toward_the_floor():
    busy = run(validated=100.0, dispatched=100.0)
    quiet = busy
    for e in range(1000, 3200):
        quiet = r.observe(quiet, epoch=e, validated=0.0, dispatched=0.0,
                          alpha_epoch=ALPHA_E)
    assert quiet.ema < busy.ema * 0.05
    assert ration_of(quiet) == pytest.approx(EXPLORE_E, rel=0.05)


def test_the_cap_bounds_the_bid():
    huge = run(validated=100_000.0, dispatched=100_000.0)
    assert want_of(huge) == pytest.approx(CAP_E)


# ---- floors and the newcomer boost ------------------------------------------------

def test_every_uid_gets_at_least_the_explore_floor():
    assert r.floor_for(None, epoch=0, explore_epoch=EXPLORE_E, boost_epoch=EXPLORE_E,
                       boost_days=0, epochs_per_day=EPOCHS_PER_DAY) == EXPLORE_E


def test_a_new_uid_starts_at_the_boost_and_decays_to_the_floor():
    state = r.seen(None, epoch=0)
    kw = dict(explore_epoch=EXPLORE_E, boost_epoch=BOOST_E,
              boost_days=BOOST_DAYS, epochs_per_day=EPOCHS_PER_DAY)
    day0 = r.floor_for(state, epoch=0, **kw)
    day7 = r.floor_for(state, epoch=7 * EPOCHS_PER_DAY, **kw)
    day14 = r.floor_for(state, epoch=14 * EPOCHS_PER_DAY, **kw)
    day30 = r.floor_for(state, epoch=30 * EPOCHS_PER_DAY, **kw)

    assert day0 == pytest.approx(BOOST_E)
    assert day7 == pytest.approx((BOOST_E + EXPLORE_E) / 2, rel=0.01)
    assert day14 == pytest.approx(EXPLORE_E)
    assert day30 == pytest.approx(EXPLORE_E)


def test_the_boost_decays_monotonically():
    state = r.seen(None, epoch=0)
    prev = None
    for day in range(0, 15):
        f = r.floor_for(state, epoch=day * EPOCHS_PER_DAY, explore_epoch=EXPLORE_E,
                        boost_epoch=BOOST_E, boost_days=BOOST_DAYS,
                        epochs_per_day=EPOCHS_PER_DAY)
        if prev is not None:
            assert f <= prev
        prev = f


# ---- allocation -------------------------------------------------------------------

def test_scarcity_never_hands_out_more_than_supply():
    wants = {f"m{i}": 100.0 for i in range(10)}
    floors = {k: EXPLORE_E for k in wants}
    given = r.allocate(wants, floors, supply=200.0, explore_epoch=EXPLORE_E)
    assert sum(given.values()) <= 200.0 + 1e-9


def test_nobody_is_given_more_than_they_bid():
    wants = {"a": 10.0, "b": 20.0}
    floors = {"a": EXPLORE_E, "b": EXPLORE_E}
    given = r.allocate(wants, floors, supply=10_000.0, explore_epoch=EXPLORE_E)
    assert given["a"] == pytest.approx(10.0)
    assert given["b"] == pytest.approx(20.0)


def test_the_floor_is_protected_from_scaling():
    """Under scarcity a newcomer still receives its floor."""
    wants = {"incumbent": 10_000.0, "newcomer": 0.0}
    floors = {"incumbent": EXPLORE_E, "newcomer": BOOST_E}
    given = r.allocate(wants, floors, supply=500.0, explore_epoch=EXPLORE_E,
                       boost_tranche_max=0.05)
    assert given["newcomer"] >= min(BOOST_E, 0.05 * 500.0)


def test_allocation_above_the_floor_is_linear_in_delivery():
    """Two UIDs with the same total delivery earn the same, however they are split."""
    floors = {k: 0.0 for k in ("one", "a", "b")}
    given = r.allocate({"one": 100.0, "a": 50.0, "b": 50.0}, floors,
                       supply=100.0, explore_epoch=0.0)
    assert given["one"] == pytest.approx(given["a"] + given["b"])


def test_splitting_across_uids_does_not_increase_the_total():
    """The sybil property, at the allocation step."""
    floors_one = {"solo": EXPLORE_E}
    solo = r.allocate({"solo": 300.0}, floors_one, supply=1000.0,
                      explore_epoch=EXPLORE_E)["solo"]

    split_keys = [f"s{i}" for i in range(3)]
    floors_many = {k: EXPLORE_E for k in split_keys}
    split = r.allocate({k: 100.0 for k in split_keys}, floors_many, supply=1000.0,
                       explore_epoch=EXPLORE_E)
    assert sum(split.values()) <= solo + 3 * EXPLORE_E + 1e-9


def test_the_boost_tranche_is_capped_under_a_registration_wave():
    # A UID that has delivered nothing bids nothing; its boost is the whole claim.
    newcomers = {f"n{i}": 0.0 for i in range(200)}
    floors = {k: BOOST_E for k in newcomers}
    supply = 1000.0
    given = r.allocate(newcomers, floors, supply=supply, explore_epoch=EXPLORE_E,
                       boost_tranche_max=0.05)
    above_explore = sum(max(0.0, v - EXPLORE_E) for v in given.values())
    assert above_explore <= 0.05 * supply + 1e-6


def test_incumbents_are_not_starved_by_a_registration_wave():
    hotkeys = {"incumbent": 400.0}
    floors = {"incumbent": EXPLORE_E}
    for i in range(300):
        hotkeys[f"n{i}"] = 0.0
        floors[f"n{i}"] = BOOST_E
    given = r.allocate(hotkeys, floors, supply=1000.0, explore_epoch=EXPLORE_E,
                       boost_tranche_max=0.05)
    assert given["incumbent"] > 0.5 * 400.0


def test_allocation_degrades_gracefully_when_floors_exceed_supply():
    keys = [f"m{i}" for i in range(100)]
    given = r.allocate({k: 10.0 for k in keys}, {k: 10.0 for k in keys},
                       supply=50.0, explore_epoch=EXPLORE_E)
    assert sum(given.values()) == pytest.approx(50.0)


def test_zero_supply_gives_nothing():
    given = r.allocate({"a": 5.0}, {"a": EXPLORE_E}, supply=0.0)
    assert given["a"] == 0.0


# ---- book -------------------------------------------------------------------------

def test_book_round_trips_through_a_dict():
    book = r.RationBook()
    book.observe("hk", epoch=1, validated=10.0, dispatched=10.0, alpha_epoch=ALPHA_E)
    restored = r.RationBook.from_dict(book.to_dict())
    assert restored.states["hk"].ema == pytest.approx(book.states["hk"].ema)
    assert restored.states["hk"].fills == book.states["hk"].fills


def test_plan_returns_rations_bounded_by_supply():
    book = r.RationBook()
    for i in range(20):
        for e in range(10):
            book.observe(f"m{i}", epoch=e, validated=float(i), dispatched=float(i),
                         alpha_epoch=ALPHA_E)
    plan = book.plan([f"m{i}" for i in range(20)], epoch=10, supply=300.0,
                     explore_epoch=EXPLORE_E, boost_epoch=BOOST_E,
                     boost_days=BOOST_DAYS, epochs_per_day=EPOCHS_PER_DAY,
                     probe_epoch=PROBE_E, fill_gate=FILL_GATE, cap_epoch=CAP_E,
                     boost_tranche_max=0.05)
    assert sum(plan.values()) <= 300.0 + 1e-9
    assert all(v >= 0 for v in plan.values())


def test_prune_drops_departed_hotkeys():
    book = r.RationBook()
    book.observe("stays", epoch=1, validated=1, dispatched=1, alpha_epoch=ALPHA_E)
    book.observe("goes", epoch=1, validated=1, dispatched=1, alpha_epoch=ALPHA_E)
    book.prune(["stays"])
    assert set(book.states) == {"stays"}
