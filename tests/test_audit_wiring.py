"""The audit is opt-in, costs nothing while off, and never records an observation."""

import types

import pytest

import alpharidge_ai.config as config
from alpharidge_ai.oracle import audit_key
from alpharidge_ai.oracle.runner import Auditor, Observation


def test_shadow_auditing_is_off_by_default():
    assert config.AUDIT_SHADOW_ENABLED is False


def test_the_flag_is_served_but_not_a_consensus_key():
    """It changes only local logging and spend, so validators may differ on it."""
    assert "AUDIT_SHADOW_ENABLED" in config._REMOTE_CONFIG_KEYS
    assert "AUDIT_SHADOW_ENABLED" not in config._CONSENSUS_KEYS


# ---- the key ----------------------------------------------------------------------

def test_a_key_is_generated_and_reused(tmp_path, monkeypatch):
    monkeypatch.delenv("AUDIT_KEY", raising=False)
    path = tmp_path / "key"
    first = audit_key.load(path)
    assert len(first) == audit_key.KEY_BYTES
    assert audit_key.load(path) == first


def test_two_validators_get_different_keys(tmp_path, monkeypatch):
    monkeypatch.delenv("AUDIT_KEY", raising=False)
    assert audit_key.load(tmp_path / "a") != audit_key.load(tmp_path / "b")


def test_an_explicit_key_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIT_KEY", "ab" * 32)
    assert audit_key.load(tmp_path / "key") == bytes.fromhex("ab" * 32)


def test_the_key_file_is_not_world_readable(tmp_path, monkeypatch):
    monkeypatch.delenv("AUDIT_KEY", raising=False)
    path = tmp_path / "key"
    audit_key.load(path)
    assert path.stat().st_mode & 0o077 == 0


# ---- the auditor ------------------------------------------------------------------

def test_without_a_profile_the_auditor_does_nothing():
    auditor = Auditor(b"key", lambda block: None)
    result = auditor.audit(1, "text", object(), object(),
                           types.SimpleNamespace(floor_pass=True), block=0)
    assert result is None


def test_a_failure_inside_the_audit_does_not_escape():
    def explode(block):
        raise RuntimeError("profile unavailable")
    auditor = Auditor(b"key", explode)
    assert auditor.audit(1, "text", object(), object(),
                         types.SimpleNamespace(floor_pass=True), block=0) is None


def test_observations_are_returned_not_recorded():
    """Nothing in the audit path may reach the reputation store during shadow."""
    import inspect
    from alpharidge_ai.oracle import runner
    source = inspect.getsource(runner)
    assert "record_local" not in source
