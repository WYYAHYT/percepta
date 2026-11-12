"""Pinned offline source changes are repeatable and never overwrite unexpected bytes."""

import hashlib

import pytest

from scripts import bootstrap_mvsanywhere as bootstrap


def test_source_patch_is_repeatable_and_rejects_corruption(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    target = source / "encoder.py"
    target.write_text("remote_load()\n")
    monkeypatch.setattr(bootstrap, "SOURCE", source)
    monkeypatch.setattr(bootstrap, "prepare_repository", lambda *args: None)
    manifest = {
        "source": {"offline_patches": [{
            "path": "encoder.py", "original_sha256": hashlib.sha256(b"remote_load()\n").hexdigest(),
            "sha256": hashlib.sha256(b"local_load()\n").hexdigest(),
            "replacements": [{"original": "remote_load()", "replacement": "local_load()", "count": 1}],
        }]},
        "dinov2": {},
    }
    bootstrap.prepare_source(manifest)
    bootstrap.prepare_source(manifest)
    assert target.read_text() == "local_load()\n"
    target.write_text("changed()\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        bootstrap.prepare_source(manifest)
    assert target.read_text() == "changed()\n"


def test_asset_hash_checked_before_checkpoint_publication(tmp_path, monkeypatch):
    monkeypatch.setattr(bootstrap, "ASSETS", tmp_path)
    asset = {"filename": "hero.ckpt", "url": "https://example.invalid", "sha256": hashlib.sha256(b"official").hexdigest()}
    temporary = tmp_path / "hero.ckpt.part"
    temporary.write_bytes(b"truncated")
    with pytest.raises(ValueError, match="hash mismatch"):
        bootstrap.prepare_checkpoint({"checkpoint": asset})
    assert not (tmp_path / "hero.ckpt").exists()
    temporary.write_bytes(b"official")
    bootstrap.prepare_checkpoint({"checkpoint": asset})
    bootstrap.prepare_checkpoint({"checkpoint": asset})
    assert (tmp_path / "hero.ckpt").read_bytes() == b"official"
