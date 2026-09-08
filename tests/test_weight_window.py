"""Weight-window aggregation.

Covers the parts the archive replay cannot reach: the reward store's retention
and range read, the broadcast re-key filter, and the config clamp. The chain
coefficients are pinned so K=1 and K=7 are compared on equal terms.
"""
import numpy as np
import pytest

from alpharidge_ai import config
from alpharidge_ai.models.reward import Reward
from alpharidge_ai.utils import burn
from alpharidge_ai.utils.reward import MinerReward
from alpharidge_ai.validator.reward_broadcast_store import RewardBroadcastStore

POINTS = [3, 7, 11, 2, 5]
ALPHA_PER_POINT = 500.0
MINER_ALPHA_PER_BLOCK = 0.02


@pytest.fixture(autouse=True)
def pinned_coefficients(monkeypatch):
    monkeypatch.setattr(burn, "get_alpha_per_point", lambda: ALPHA_PER_POINT)
    monkeypatch.setattr(burn, "get_miner_alpha_per_block", lambda: MINER_ALPHA_PER_BLOCK)


class FakeMetagraph:
    def __init__(self, n=256, registered_at=None):
        self.n = n
        self.hotkeys = [f"hk{i}" for i in range(n)]
        self.block_at_registration = registered_at if registered_at is not None else [0] * n


def rewards_for(points, multiplier=1):
    return [Reward(hotkey=f"hk{i}", reward=p * multiplier, epoch=0) for i, p in enumerate(points)]


# --- normalisation -------------------------------------------------------

@pytest.mark.parametrize("k", [1, 4, 7, 12, 24])
def test_same_production_rate_gives_same_weights_at_any_k(k):
    """A miner producing at a steady rate must be paid the same whatever K is.

    Points scale with the window, so the divisor has to as well. If either the
    calculated percent or the minimum-percent floor loses its divisor, this
    fails and the burn control drifts with K.
    """
    mg = FakeMetagraph()
    at_k1 = burn.calculate_weights(rewards_for(POINTS), mg, config.BLOCK_LENGTH)
    at_k = burn.calculate_weights(rewards_for(POINTS, k), mg, k * config.BLOCK_LENGTH)
    assert np.allclose(at_k, at_k1, atol=1e-12)


def test_single_epoch_window_reproduces_the_pre_change_arithmetic():
    """Landing the code at K=1 is a no-op, so step 1 of the rollout is safe.

    The old code divided by the EPOCH_LENGTH constant regardless of how long an
    epoch actually was, so parity is stated against that same span. In service
    the caller passes present_epochs * BLOCK_LENGTH; the two agree only while
    BLOCK_LENGTH == EPOCH_LENGTH, which config warns about at import.
    """
    mg = FakeMetagraph()
    got = burn.calculate_weights(rewards_for(POINTS), mg, config.EPOCH_LENGTH)

    expected = np.zeros(mg.n, dtype=np.float64)
    per, tpn = {}, 0.0
    for i, p in enumerate(POINTS):
        pct = (p / ALPHA_PER_POINT) / (MINER_ALPHA_PER_BLOCK * config.EPOCH_LENGTH) * 100
        pct = max(pct, p * config.MIN_PERCENT_PER_POINT)  # pre-change floor: no divisor
        per[i], tpn = pct, tpn + pct
    scale = 100.0 / tpn if tpn > 100 else 1.0
    for i, pct in per.items():
        expected[i] = pct * scale / 100.0
    expected[config.BURN_UID] = 1 - (min(tpn, 100) / 100)

    assert np.allclose(got, expected, atol=1e-15)


def test_window_blocks_is_required_and_must_be_positive():
    """A silent default here would under-divide and drive burn to zero."""
    mg = FakeMetagraph()
    with pytest.raises(TypeError):
        burn.calculate_weights([], mg)
    with pytest.raises(ValueError):
        burn.calculate_weights([], mg, 0)


# --- reward store --------------------------------------------------------

def test_range_read_skips_absent_epochs_rather_than_zeroing_them():
    """A gap must reduce the reported span, not silently count as zero points."""
    store = MinerReward(block_length=config.BLOCK_LENGTH, block=lambda: 1000)
    store.epoch_rewards = {5: {"a": 2}, 6: {"a": 3, "b": 1}, 8: {"b": 4}}
    store.update_current_epoch = lambda: None

    totals, present = store.get_rewards_range(4, 8)

    assert totals == {"a": 5, "b": 5}
    assert present == 3  # 5, 6 and 8 are held; 4 and 7 are not


def test_retention_follows_the_widest_configured_window(monkeypatch):
    """Retention is sized from the raw keys, never the block-gated active K.

    At rollout step 1 both window keys are still 1 while shadow mode runs at 7,
    and the epochs a window needs after the activation block were recorded
    before it. Either case would come up short if the active K were used.
    """
    monkeypatch.setattr(config, "WEIGHT_WINDOW_EPOCHS", 1)
    monkeypatch.setattr(config, "WEIGHT_WINDOW_EPOCHS_PREV", 1)
    monkeypatch.setattr(config, "WEIGHT_WINDOW_SHADOW_EPOCHS", 0)
    assert config.weight_window_retention() == 10  # never below the historical value

    monkeypatch.setattr(config, "WEIGHT_WINDOW_SHADOW_EPOCHS", 7)
    assert config.weight_window_retention() == 12

    monkeypatch.setattr(config, "WEIGHT_WINDOW_SHADOW_EPOCHS", 0)
    monkeypatch.setattr(config, "WEIGHT_WINDOW_EPOCHS", 7)
    assert config.weight_window_retention() == 12


# --- restart -------------------------------------------------------------

def _populated_store(tmp_path, epochs, points=3):
    """A reward store holding `points` for hk0 in each of `epochs`, saved to disk."""
    block = [0]
    store = MinerReward(config.BLOCK_LENGTH, lambda: block[0])
    for epoch in epochs:
        block[0] = epoch * config.BLOCK_LENGTH
        store.add_reward("hk0", points)
    path = tmp_path / "reward_store.json"
    store.save_to_file(path)
    return store, path


def _reloaded_at(path, epoch):
    """A fresh store loaded from disk, as if the process had just restarted."""
    block = [epoch * config.BLOCK_LENGTH]
    store = MinerReward(config.BLOCK_LENGTH, lambda: block[0])
    store.load_from_file(block=lambda: block[0], file_path=path)
    return store


def test_clean_restart_preserves_the_whole_window(tmp_path):
    """A restart must not shorten the window or lose its points.

    The store is written every poll loop, so a clean restart should come back
    holding every epoch it held before.
    """
    store, path = _populated_store(tmp_path, range(100, 107))
    before = store.get_rewards_range(100, 106)

    after = _reloaded_at(path, 108).get_rewards_range(100, 106)

    assert after == before == ({"hk0": 21}, 7)


def test_downtime_costs_sample_size_not_rate(tmp_path):
    """The point of the 4.2 divisor: epochs missed while down shrink the divisor too.

    Down for 107-109, back at 110. The window holds 4 of 7 epochs, so it divides
    by 4 epochs' worth of blocks — the miner's points-per-block is unchanged and
    its weight is unchanged. Dividing by the full span instead would understate
    every miner by 43% and spike burn for the rest of the window.
    """
    _, path = _populated_store(tmp_path, range(100, 107))
    store = _reloaded_at(path, 110)
    store.add_reward("hk0", 3)  # only epoch 110 is recorded after coming back

    totals, present = store.get_rewards_range(104, 110)

    assert present == 4  # 104, 105, 106 and 110
    assert totals == {"hk0": 12}
    full_rate = 21 / (7 * config.BLOCK_LENGTH)
    gap_rate = totals["hk0"] / (present * config.BLOCK_LENGTH)
    assert full_rate == gap_rate


def test_broadcast_points_survive_a_restart(tmp_path):
    """Other validators' points are cached on disk precisely so a restart keeps them."""
    path = tmp_path / "broadcast.json"
    store = RewardBroadcastStore(path=path, keep_epochs=99)
    for epoch in range(100, 107):
        store.ingest(sender_hotkey="v1", epoch=epoch, seq=epoch, uid_points={0: 2})
    before, _ = store.aggregate_range(100, 106)
    store.save()

    reloaded = RewardBroadcastStore(path=path, keep_epochs=99)
    reloaded.load()

    assert reloaded.aggregate_range(100, 106)[0] == before == {0: 14}


# --- broadcast store -----------------------------------------------------

def test_uid_that_changed_holder_inside_the_window_is_dropped(tmp_path):
    """Broadcast points are keyed by UID, so a re-keyed UID would pay the wrong miner."""
    store = RewardBroadcastStore(path=tmp_path / "b.json", keep_epochs=99)
    store.by_epoch_by_sender = {10: {"v1": {0: 5, 1: 7}}, 11: {"v1": {0: 5, 2: 9}}}
    window_start_block = 10 * config.BLOCK_LENGTH
    mg = FakeMetagraph(registered_at=[0, window_start_block + 50] + [0] * 254)

    agg, rekeyed = store.aggregate_range(10, 11, mg)

    assert agg == {0: 10, 2: 9}
    assert rekeyed == 1


def test_range_aggregate_without_a_metagraph_drops_nothing(tmp_path):
    store = RewardBroadcastStore(path=tmp_path / "b.json", keep_epochs=99)
    store.by_epoch_by_sender = {10: {"v1": {0: 5, 1: 7}}, 11: {"v1": {0: 5, 2: 9}}}

    agg, rekeyed = store.aggregate_range(10, 11)

    assert agg == {0: 10, 1: 7, 2: 9}
    assert rekeyed == 0


# --- config --------------------------------------------------------------

@pytest.mark.parametrize("raw", ["0", "500", "-3", "abc", ""])
def test_out_of_range_window_falls_back_to_the_default(raw):
    assert config._clamp_window(raw, 1) == 1


def test_shadow_window_may_be_zero_to_disable():
    assert config._clamp_window("0", 0, minimum=0) == 0
    assert config._clamp_window("7", 0, minimum=0) == 7


@pytest.mark.parametrize("raw", ["0", "500", "-3"])
def test_served_window_value_out_of_range_raises_so_the_previous_is_kept(raw):
    """refresh_remote_config catches ValueError and leaves the old value in place."""
    with pytest.raises(ValueError):
        config._cast_window_epochs(raw)


def test_active_block_selects_between_current_and_previous_k(monkeypatch):
    monkeypatch.setattr(config, "WEIGHT_WINDOW_EPOCHS", 7)
    monkeypatch.setattr(config, "WEIGHT_WINDOW_EPOCHS_PREV", 1)
    monkeypatch.setattr(config, "WEIGHT_WINDOW_ACTIVE_BLOCK", 1_000_000)

    assert config.weight_window_epochs(999_999) == 1
    assert config.weight_window_epochs(1_000_000) == 7
    assert config.weight_window_epochs(1_000_001) == 7


def test_a_transition_with_no_activation_block_is_refused(monkeypatch):
    """The likeliest operator slip once the API env is the only lever.

    Serving a new K without an activation block would switch every validator
    whenever it next polled — spread over the refresh interval, which is the
    divergence the activation block exists to prevent. Hold the previous value
    and say so, rather than switch on an unscheduled block.
    """
    monkeypatch.setattr(config, "WEIGHT_WINDOW_EPOCHS", 7)
    monkeypatch.setattr(config, "WEIGHT_WINDOW_EPOCHS_PREV", 1)
    monkeypatch.setattr(config, "WEIGHT_WINDOW_ACTIVE_BLOCK", 0)

    assert config.weight_window_epochs(9_000_000) == 1
    assert config.weight_window_epochs(0) == 1


def test_defaults_need_no_activation_block(monkeypatch):
    """Pull-and-restart with nothing served must not trip the guard above."""
    monkeypatch.setattr(config, "WEIGHT_WINDOW_EPOCHS", 1)
    monkeypatch.setattr(config, "WEIGHT_WINDOW_EPOCHS_PREV", 1)
    monkeypatch.setattr(config, "WEIGHT_WINDOW_ACTIVE_BLOCK", 0)

    assert config.weight_window_epochs(9_000_000) == 1


def test_activation_block_in_the_past_still_applies(monkeypatch):
    """A validator that polled late sees the block behind it and must still switch."""
    monkeypatch.setattr(config, "WEIGHT_WINDOW_EPOCHS", 7)
    monkeypatch.setattr(config, "WEIGHT_WINDOW_EPOCHS_PREV", 1)
    monkeypatch.setattr(config, "WEIGHT_WINDOW_ACTIVE_BLOCK", 8_900_000)

    assert config.weight_window_epochs(9_000_000) == 7


def test_window_keys_are_marked_as_consensus_keys():
    """A local OVERRIDE_ on these is a weight split no config push can correct."""
    assert {
        "WEIGHT_WINDOW_EPOCHS",
        "WEIGHT_WINDOW_EPOCHS_PREV",
        "WEIGHT_WINDOW_ACTIVE_BLOCK",
    } <= config._CONSENSUS_KEYS
    for key in config._CONSENSUS_KEYS:
        assert key in config._REMOTE_CONFIG_KEYS


def _serve(monkeypatch, payload):
    """Point refresh_remote_config at a fixed payload instead of the API."""
    class Response:
        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return {"config": payload}

    monkeypatch.setattr(config, "MINER_API_URL", "http://test.invalid")
    monkeypatch.setattr(config.requests, "get", lambda *a, **kw: Response())
    monkeypatch.setattr(config, "_remote_config_last_fetch", 0.0)


def test_override_is_ignored_for_the_window_keys(monkeypatch, caplog):
    """GUARD 5. The whole failure mode is silence, so assert the behaviour, not the constant.

    A stale OVERRIDE_ on one validator is a permanent weight difference that no
    config push can correct, and section 4.5 cannot detect it.
    """
    monkeypatch.setattr(config, "WEIGHT_WINDOW_EPOCHS", 1)
    monkeypatch.setenv("OVERRIDE_WEIGHT_WINDOW_EPOCHS", "9")
    _serve(monkeypatch, {"WEIGHT_WINDOW_EPOCHS": 7})

    config.refresh_remote_config(force=True)

    assert config.WEIGHT_WINDOW_EPOCHS == 7, "the served value must win over a local override"


def test_override_is_still_honoured_for_a_non_consensus_key(monkeypatch):
    """Control for the test above: the override path itself still works."""
    monkeypatch.setattr(config, "MIN_PERCENT_PER_POINT", 0.003)
    monkeypatch.setenv("OVERRIDE_MIN_PERCENT_PER_POINT", "0.001")
    _serve(monkeypatch, {"MIN_PERCENT_PER_POINT": 0.005})

    config.refresh_remote_config(force=True)

    assert config.MIN_PERCENT_PER_POINT == 0.001, "tuning keys still take the local override"


# --- reputation gate order ----------------------------------------------

class EpochStore:
    """Reward store yielding POINTS_PER_EPOCH per epoch, proportional to the range.

    Deliberately range-aware: a fixture that returns a flat total regardless of
    the span cannot tell a per-epoch gate from a pooled one.
    """
    POINTS_PER_EPOCH = 3

    def get_rewards_range(self, start, end):
        epochs = end - start + 1
        return {"hk0": self.POINTS_PER_EPOCH * epochs}, epochs

    def get_penalties_range(self, start, end):
        return {}, end - start + 1


class FixedBroadcasts:
    def __init__(self, points=None):
        self.points = points or {}

    def aggregate_range(self, start, end, metagraph=None):
        return dict(self.points), 0


def _client_with(monkeypatch, reward_store, reward_broadcasts, multiplier=0.5):
    from alpharidge_ai.validator import validation_client as vc

    monkeypatch.setattr(config, "REPUTATION_GATING_ENABLED", True)
    monkeypatch.setattr(vc.reputation, "emission", lambda *a, **kw: multiplier)

    validator = type("V", (), {})()
    validator.metagraph = FakeMetagraph()
    validator._miner_reward = reward_store
    validator._miner_penalty = reward_store
    validator._reward_broadcasts = reward_broadcasts
    validator._penalty_broadcasts = FixedBroadcasts()
    validator._reputation_store = type(
        "R", (), {"reputation": lambda self, hk: 0.9,
                  "samples": lambda self, hk: 1000})()

    client = object.__new__(vc.ValidationClient)
    client._validator = validator
    return client


# --- the §4.2 divisor ----------------------------------------------------

def _window_blocks(present, k=7):
    from alpharidge_ai.validator.validation_client import ValidationClient
    return ValidationClient._window_blocks(present, k, 100, 100 + k - 1)


def test_divisor_is_the_present_span_never_the_full_window():
    """The failure with the largest blast radius in this change.

    Dividing a partial point total by the full window span understates every
    miner in proportion — tpn falls and burn rises. The divisor must shrink with
    the sample.
    """
    assert _window_blocks(7) == 7 * config.BLOCK_LENGTH  # complete window
    assert _window_blocks(3) == 3 * config.BLOCK_LENGTH  # NOT 7 * BLOCK_LENGTH
    assert _window_blocks(1) == 1 * config.BLOCK_LENGTH


def test_no_epochs_present_keeps_the_previous_vector():
    """None means "do not recompute" — a near-empty vector must never be sent."""
    assert _window_blocks(0) is None


def test_a_short_window_pays_the_same_rate_as_a_full_one():
    """The property the divisor exists to preserve.

    A miner producing 3 points per epoch is paid the same whether the store held
    all 7 epochs of the window or only 3 of them. Only the sample size differs.
    """
    mg = FakeMetagraph()
    full = burn.calculate_weights(
        [Reward(hotkey="hk0", reward=3 * 7, epoch=0)], mg, _window_blocks(7)
    )
    short = burn.calculate_weights(
        [Reward(hotkey="hk0", reward=3 * 3, epoch=0)], mg, _window_blocks(3)
    )
    assert np.allclose(short, full, atol=1e-12)


def test_reputation_gate_is_applied_once_to_the_window_not_per_epoch(monkeypatch):
    """GUARD 4, on the axis 5.7 names: sum the window, then gate. Never gate each epoch and add.

    Two epochs of 3 points at a multiplier of 0.5. Gating the pooled 6 gives
    round(3.0) = 3; gating each epoch gives round(1.5) + round(1.5) = 4. The
    store is range-aware, so the two orders are genuinely distinguishable here.
    """
    client = _client_with(monkeypatch, EpochStore(), FixedBroadcasts())

    rewards, present, _ = client._aggregate_window(10, 9, 10)

    assert present == 2, "the fixture must span two epochs or this proves nothing"
    assert len(rewards) == 1
    assert rewards[0].reward == 3, "gate the window total once, never each epoch then add"


def test_reputation_gate_sees_local_and_broadcast_pooled(monkeypatch):
    """GUARD 4 on the other axis: both point sources are pooled before the gate.

    One epoch of 3 local points plus 3 broadcast points. Gating the pooled 6
    gives 3; gating each source gives round(1.5) + round(1.5) = 4.
    """
    client = _client_with(monkeypatch, EpochStore(), FixedBroadcasts({0: 3}))

    rewards, present, _ = client._aggregate_window(10, 10, 10)

    assert present == 1
    assert len(rewards) == 1
    assert rewards[0].reward == 3, "pool local and broadcast points before gating"
