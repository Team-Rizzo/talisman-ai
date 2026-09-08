"""Mechanism profile: parsing, range validation, and block-height resolution."""

import copy

import pytest

from alpharidge_ai.mechanism import profile as mp


def _valid() -> dict:
    return {
        "version": 2,
        "publish_block": 9_000,
        "activation_block": 10_000,
        "schema_version": "1.2.0",
        "settlement": {"C": 301_743.0},
        "emission": {
            "midpoint": 0.766, "gain": 15.0, "ceiling": 1.0,
            "bonus_start": 0.716, "bonus_full": 0.816,
            "n_min": 100, "ema_alpha": 0.03,
        },
        "rations": {
            "explore": 25.0, "probe_day": 2.0, "alpha_day": 0.5, "cap": 5000.0,
            "slack_target": 0.15, "fill_gate": 0.97,
            "boost": 200.0, "boost_days": 14, "boost_tranche_max": 0.05,
        },
        "oracle": {
            "pool_tiers": ["number_bearing"],
            "keyed_rate_pool": 0.90, "keyed_rate_keeper": 0.03,
            "claim_cap": 40, "keeper_weight": 0.3,
            "grader_models": [{"id": "model-a", "weight": 0.7},
                              {"id": "model-b", "weight": 0.3}],
            "schema_cutover_block": 12_000,
        },
        "controller": {
            "roi_lo": 1.5, "roi_hi": 6.0, "arm_days": 3,
            "max_step": 0.20, "gap_days": 14, "cost_per_point": 0.000292,
        },
        "signature": "",
    }


def test_parses_a_valid_profile():
    p = mp.parse(_valid())
    assert p.version == 2
    assert p.settlement.C == pytest.approx(301_743.0)
    assert p.emission.gain == 15.0
    assert p.oracle.grader_models[0].id == "model-a"
    assert p.controller.gap_days == 14


@pytest.mark.parametrize("section,key,bad", [
    ("emission", "gain", 0.5),          # below the sane floor
    ("emission", "gain", 51.0),
    ("emission", "ceiling", 3.5),
    ("emission", "midpoint", 1.5),
    ("settlement", "C", 0.0),
    ("oracle", "keyed_rate_pool", 1.5),
    ("oracle", "keyed_rate_keeper", -0.1),
    ("controller", "max_step", 0.0),
    ("rations", "alpha_day", 0.0),
    ("rations", "probe_day", 0.5),
])
def test_out_of_range_values_are_rejected(section, key, bad):
    raw = _valid()
    raw[section][key] = bad
    with pytest.raises(mp.ProfileError) as e:
        mp.parse(raw)
    assert key in str(e.value)


def test_bonus_full_below_start_is_rejected():
    raw = _valid()
    raw["emission"]["bonus_full"] = 0.5
    with pytest.raises(mp.ProfileError):
        mp.parse(raw)


def test_roi_band_must_be_ordered():
    raw = _valid()
    raw["controller"]["roi_hi"] = 1.0
    with pytest.raises(mp.ProfileError):
        mp.parse(raw)


def test_unknown_schema_version_is_rejected():
    raw = _valid()
    raw["schema_version"] = "9.9.9"
    with pytest.raises(mp.ProfileError) as e:
        mp.parse(raw)
    assert "not supported" in str(e.value)


def test_missing_section_is_rejected():
    raw = _valid()
    del raw["rations"]
    with pytest.raises(mp.ProfileError) as e:
        mp.parse(raw)
    assert "rations" in str(e.value)


def test_grader_models_must_carry_weight():
    raw = _valid()
    raw["oracle"]["grader_models"] = [{"id": "m", "weight": 0.0}]
    with pytest.raises(mp.ProfileError) as e:
        mp.parse(raw)
    assert "weights sum to zero" in str(e.value)


# ---- day to epoch conversion ------------------------------------------------------

def test_day_constants_convert_to_epochs_once():
    r = mp.parse(_valid()).rations
    assert (1.0 - r.alpha_epoch) ** mp.EPOCHS_PER_DAY == pytest.approx(1.0 - r.alpha_day)
    assert r.probe_epoch ** mp.EPOCHS_PER_DAY == pytest.approx(r.probe_day)
    assert r.explore_epoch * mp.EPOCHS_PER_DAY == pytest.approx(r.explore)


def test_epochs_per_day_matches_the_chain_layout():
    assert mp.EPOCHS_PER_DAY == 72


# ---- resolution -------------------------------------------------------------------

def _resolver_with_current():
    cur = mp.parse(_valid())
    return mp.ProfileResolver(current=cur, refresh_seconds=3600)


def test_min_lead_covers_a_full_refresh_interval():
    assert mp.min_lead_blocks(3600) == 300


def test_profile_activates_only_at_its_block():
    r = _resolver_with_current()
    nxt = _valid()
    nxt["version"] = 3
    nxt["activation_block"] = 20_000
    nxt["settlement"]["C"] = 400_000.0

    assert r.offer(nxt)[0]
    assert r.resolve(19_999).settlement.C == pytest.approx(301_743.0)
    assert r.resolve(20_000).settlement.C == pytest.approx(400_000.0)


def test_a_publish_without_enough_lead_is_rejected():
    r = _resolver_with_current()
    nxt = _valid()
    nxt["version"] = 3
    nxt["publish_block"] = 19_800
    nxt["activation_block"] = 20_000
    ok, reason = r.offer(nxt)  # 200 blocks of lead, needs 300
    assert not ok and "insufficient_lead" in reason
    assert r.next is None


def test_a_lower_version_is_rejected():
    r = _resolver_with_current()
    older = _valid()
    older["version"] = 1
    older["activation_block"] = 20_000
    ok, reason = r.offer(older)
    assert not ok and "stale_version" in reason


def test_the_same_version_is_rejected():
    r = _resolver_with_current()
    same = _valid()
    same["activation_block"] = 20_000
    ok, reason = r.offer(same)
    assert not ok and "stale_version" in reason


def test_rollback_is_a_forward_publish():
    r = _resolver_with_current()
    bad = _valid()
    bad["version"] = 3
    bad["activation_block"] = 20_000
    bad["settlement"]["C"] = 999_999.0
    r.offer(bad)
    assert r.resolve(20_000).settlement.C == pytest.approx(999_999.0)

    restore = _valid()
    restore["version"] = 4
    restore["activation_block"] = 30_000
    assert r.offer(restore)[0]
    assert r.resolve(30_000).settlement.C == pytest.approx(301_743.0)


def test_an_invalid_profile_leaves_the_current_one_in_force():
    r = _resolver_with_current()
    broken = _valid()
    broken["version"] = 3
    broken["activation_block"] = 20_000
    broken["emission"]["gain"] = 500.0
    ok, reason = r.offer(broken)
    assert not ok and "invalid" in reason
    assert r.resolve(20_000).emission.gain == 15.0


def test_the_lead_is_measured_from_the_signed_publish_block():
    """An already-active profile must still be adoptable, however late it is fetched."""
    r = mp.ProfileResolver(refresh_seconds=3600)
    old = _valid()
    old["version"] = 9
    old["publish_block"] = 1_000
    old["activation_block"] = 1_400
    assert r.offer(old)[0]
    assert r.resolve(9_000_000).version == 9


def test_resolution_is_deterministic_across_repeated_calls():
    r = _resolver_with_current()
    nxt = _valid()
    nxt["version"] = 3
    nxt["activation_block"] = 20_000
    r.offer(nxt)
    first = r.resolve(20_000)
    for _ in range(5):
        assert r.resolve(20_000) is first
    assert r.resolve(19_000) is first  # promotion does not un-apply


# ---- signature --------------------------------------------------------------------

def test_signature_is_checked_when_a_verifier_is_supplied():
    r = _resolver_with_current()
    nxt = _valid()
    nxt["version"] = 3
    nxt["activation_block"] = 20_000
    nxt["signature"] = "deadbeef"

    ok, reason = r.offer(nxt, verify=lambda m, s: False)
    assert not ok and reason == "bad_signature"

    ok, _ = r.offer(nxt, verify=lambda m, s: True)
    assert ok


def test_unsigned_profile_is_rejected_when_verification_is_on():
    r = _resolver_with_current()
    nxt = _valid()
    nxt["version"] = 3
    nxt["activation_block"] = 20_000
    ok, reason = r.offer(nxt, verify=lambda m, s: True)
    assert not ok and reason == "unsigned"


def test_signed_payload_excludes_the_signature_and_is_order_independent():
    raw = _valid()
    raw["signature"] = "abc123"
    msg = mp.signing_payload(raw)
    assert "abc123" not in msg

    shuffled = {k: copy.deepcopy(v) for k, v in reversed(list(raw.items()))}
    assert mp.signing_payload(shuffled) == msg


def test_canonical_json_matches_the_attestation_encoder():
    """Both sides sign the same bytes; a divergence here is a fleet-wide reject."""
    from alpharidge_ai.utils import attestation_crypto as ac
    sample = {"b": 1, "a": {"z": [1, 2], "y": "ünicode"}, "c": 2.5}
    assert mp.canonical_json(sample) == ac.canonical_json(sample)
