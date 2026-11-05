"""Model weight asset contract (``percepta.weights``).

The library does not manage a model's *dependencies* — that is the deployer's job. It does
enforce the *asset* contract, because a wrong or truncated checkpoint is the failure mode
that hurts most: it can load successfully and then produce quietly wrong segments.

These tests pin down all three verification modes and, crucially, that ``verified`` never
overstates what actually happened.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from percepta.weights import (
    WeightError,
    WeightMismatch,
    WeightNotFound,
    file_sha256,
    is_local_path,
    resolve_weights,
)


@pytest.fixture()
def checkpoint(tmp_path: Path) -> Path:
    path = tmp_path / "model.pt"
    path.write_bytes(b"fake weights" * 1000)
    return path


def _sha(path: Path) -> str:
    return file_sha256(path)


# ---------------------------------------------------------------------------
# Location classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "location,local",
    [
        ("/models/sam3.pt", True),
        ("models/sam3.pt", True),
        ("./weights/x.bin", True),
        ("http://host/x.pt", False),
        ("https://host/x.pt", False),
        ("hf://facebook/sam3", False),
        ("s3://bucket/key", False),
    ],
)
def test_is_local_path(location, local):
    assert is_local_path(location) is local


def test_none_location_yields_empty_spec():
    spec = resolve_weights(None)
    assert spec.path is None and spec.verified is False
    assert spec.describe()["remote"] is True


# ---------------------------------------------------------------------------
# Existence
# ---------------------------------------------------------------------------


def test_missing_checkpoint_is_not_found_not_a_silent_pass(tmp_path):
    missing = tmp_path / "nope.pt"
    with pytest.raises(WeightNotFound, match="not found"):
        resolve_weights(missing)
    # The message must tell the operator what to do, not just that it failed.
    with pytest.raises(WeightNotFound, match="[Mm]ount"):
        resolve_weights(missing)


def test_remote_location_is_passed_through_unverified():
    spec = resolve_weights("hf://facebook/sam3")
    assert spec.path is None
    assert spec.verified is False
    assert spec.location == "hf://facebook/sam3"


# ---------------------------------------------------------------------------
# Hash verification modes
# ---------------------------------------------------------------------------


def test_if_declared_without_expected_records_only(checkpoint):
    spec = resolve_weights(checkpoint, verify="if-declared")
    assert spec.observed_sha256 == _sha(checkpoint)
    assert spec.verified is False  # nothing was compared — must not claim otherwise


def test_if_declared_with_expected_verifies(checkpoint):
    spec = resolve_weights(
        checkpoint, expected_sha256=_sha(checkpoint), verify="if-declared"
    )
    assert spec.verified is True
    assert spec.observed_sha256 == spec.expected_sha256


def test_mismatch_refuses_to_load(checkpoint):
    wrong = "0" * 64
    with pytest.raises(WeightMismatch) as excinfo:
        resolve_weights(checkpoint, expected_sha256=wrong)
    message = str(excinfo.value)
    assert wrong in message  # expected
    assert _sha(checkpoint) in message  # observed
    # The error must explain *why* this matters, not just that it differed.
    assert "wrong" in message.lower()


def test_always_requires_expected_hash(checkpoint):
    with pytest.raises(WeightError, match="requires checkpoint_sha256"):
        resolve_weights(checkpoint, verify="always")


def test_always_verifies_when_expected_present(checkpoint):
    spec = resolve_weights(checkpoint, expected_sha256=_sha(checkpoint), verify="always")
    assert spec.verified is True


def test_never_skips_hashing_entirely(checkpoint):
    spec = resolve_weights(checkpoint, expected_sha256=_sha(checkpoint), verify="never")
    assert spec.observed_sha256 is None
    assert spec.verified is False


def test_unknown_mode_is_a_config_error(checkpoint):
    with pytest.raises(WeightError, match="unknown verify_weights"):
        resolve_weights(checkpoint, verify="sometimes")


def test_malformed_expected_hash_is_rejected(checkpoint):
    for bad in ("abc", "z" * 64, "0" * 63):
        with pytest.raises(WeightError, match="64 hex"):
            resolve_weights(checkpoint, expected_sha256=bad)


def test_config_dict_can_carry_the_expected_hash(checkpoint):
    spec = resolve_weights(
        checkpoint, config={"checkpoint_sha256": _sha(checkpoint).upper()}
    )
    assert spec.verified is True  # upper-case hex is normalised, not rejected


def test_symlinks_are_resolved_to_the_real_artifact(tmp_path, checkpoint):
    link = tmp_path / "link.pt"
    link.symlink_to(checkpoint)
    spec = resolve_weights(link)
    assert spec.path == checkpoint.resolve()


# ---------------------------------------------------------------------------
# Corruption detection — the case that motivates all of this
# ---------------------------------------------------------------------------


def test_truncated_checkpoint_is_caught(tmp_path):
    original = tmp_path / "model.pt"
    original.write_bytes(b"real weights" * 100)
    expected = _sha(original)

    truncated = tmp_path / "truncated.pt"
    truncated.write_bytes(original.read_bytes()[:200])  # half the file

    with pytest.raises(WeightMismatch):
        resolve_weights(truncated, expected_sha256=expected)
    # Without an expected hash nothing can catch it — documented behaviour, not a bug.
    spec = resolve_weights(truncated)
    assert spec.observed_sha256 != expected


def test_file_sha256_matches_hashlib(tmp_path):
    import hashlib

    path = tmp_path / "x.bin"
    payload = b"\x00\x01\x02" * 5000
    path.write_bytes(payload)
    assert file_sha256(path) == hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# CLI: `percepta models check`
# ---------------------------------------------------------------------------


def _run_cli(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "percepta.cli", *args],
        capture_output=True,
        text=True,
    )


def test_cli_models_check_reports_ok_for_valid_config(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"weights")
    config = tmp_path / "c.yaml"
    config.write_text(
        "tools:\n"
        "  sam3:\n"
        "    enabled: true\n"
        f"    checkpoint: {checkpoint}\n"
        f"    checkpoint_sha256: {_sha(checkpoint)}\n",
        encoding="utf-8",
    )
    result = _run_cli(["models", "check", "--config", str(config), "--json"])
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["ok"] is True
    assert report["models"][0]["verified"] is True


def test_cli_models_check_fails_on_mismatch(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"weights")
    config = tmp_path / "c.yaml"
    config.write_text(
        "tools:\n"
        "  sam3:\n"
        "    enabled: true\n"
        f"    checkpoint: {checkpoint}\n"
        "    checkpoint_sha256: '" + "0" * 64 + "'\n",  # quoted: a hash is a string
        encoding="utf-8",
    )
    result = _run_cli(["models", "check", "--config", str(config), "--json"])
    assert result.returncode == 1
    report = json.loads(result.stdout)
    assert report["ok"] is False
    assert "mismatch" in report["models"][0]["error"].lower()


def test_cli_models_check_rejects_unquoted_numeric_hash(tmp_path):
    """An all-digit hash without quotes becomes a YAML integer.

    Before this was guarded, ``int(0)`` was falsy, the ``or`` chain dropped the expected
    hash, and verification was skipped *silently* — a mismatched checkpoint would load.
    The contract is now: a numeric-looking hash is either quoted (string) or loudly rejected.
    """
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"weights")
    config = tmp_path / "c.yaml"
    config.write_text(
        "tools:\n"
        "  sam3:\n"
        "    enabled: true\n"
        f"    checkpoint: {checkpoint}\n"
        "    checkpoint_sha256: " + "0" * 64 + "\n",  # unquoted -> integer 0
        encoding="utf-8",
    )
    result = _run_cli(["models", "check", "--config", str(config), "--json"])
    assert result.returncode == 1
    report = json.loads(result.stdout)
    assert report["ok"] is False
    error = report["models"][0]["error"].lower()
    assert "64 hex" in error
    assert "quote" in error  # the message must say how to fix it


def test_cli_models_check_skips_disabled_tools(tmp_path):
    config = tmp_path / "c.yaml"
    config.write_text(
        "tools:\n"
        "  sam3:\n"
        "    enabled: false\n"
        "    checkpoint: /does/not/exist.pt\n",
        encoding="utf-8",
    )
    result = _run_cli(["models", "check", "--config", str(config), "--json"])
    assert result.returncode == 0
    assert json.loads(result.stdout)["models"] == []
