"""Scoring and emission keys are consensus keys, so a local OVERRIDE_ is ignored.

These values decide points and the multiplier applied to them. A validator running a
different one produces weights no config push can reconcile, and the served kill switch
stops working on the machine that most needs it.
"""

import pytest

import alpharidge_ai.config as config


SCORING_KEYS = [
    "REPUTATION_SCORING_ENABLED",
    "REPUTATION_GATING_ENABLED",
    "REPUTATION_EMA_ALPHA",
    "REPUTATION_PRIOR",
    "EMISSION_MIDPOINT",
    "EMISSION_GAIN",
    "EMISSION_BONUS_CEILING",
    "EMISSION_BONUS_START",
    "EMISSION_BONUS_FULL",
    "EMISSION_N_MIN",
    "SAMPLING_SUBSTANTIVE_WEIGHT",
]


@pytest.mark.parametrize("key", SCORING_KEYS)
def test_scoring_keys_are_consensus_keys(key):
    assert key in config._CONSENSUS_KEYS
    assert key in config._REMOTE_CONFIG_KEYS


def _serve(monkeypatch, payload):
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


def test_local_override_of_a_scoring_key_is_ignored(monkeypatch):
    monkeypatch.setattr(config, "EMISSION_GAIN", 100.0, raising=False)
    monkeypatch.setenv("OVERRIDE_EMISSION_GAIN", "3.0")
    _serve(monkeypatch, {"EMISSION_GAIN": 15.0})

    config.refresh_remote_config(force=True)

    assert config.EMISSION_GAIN == 15.0


def test_scoring_kill_switch_cannot_be_pinned_locally(monkeypatch):
    monkeypatch.setattr(config, "REPUTATION_SCORING_ENABLED", True, raising=False)
    monkeypatch.setenv("OVERRIDE_REPUTATION_SCORING_ENABLED", "true")
    _serve(monkeypatch, {"REPUTATION_SCORING_ENABLED": False})

    config.refresh_remote_config(force=True)

    assert config.REPUTATION_SCORING_ENABLED is False
