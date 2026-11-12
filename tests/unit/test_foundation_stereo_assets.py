"""Asset preparation is repeatable and rejects changed bytes without overwriting them."""

import hashlib
import sys
from types import SimpleNamespace

import pytest

from scripts import bootstrap_foundation_stereo as bootstrap


def test_offline_source_patch_is_repeatable(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    target = source / "model.py"
    target.write_text("remote_load()\n")
    manifest = {
        "source": {
            "revision": "fixed-revision",
            "offline_dino_patch": {
                "path": "model.py",
                "original": "remote_load()",
                "replacement": "local_load()",
                "sha256": hashlib.sha256(b"local_load()\n").hexdigest(),
            },
        }
    }
    monkeypatch.setattr(bootstrap, "SOURCE", source)
    monkeypatch.setattr(bootstrap.subprocess, "check_output", lambda *a, **kw: "fixed-revision\n")
    bootstrap.prepare_source(manifest)
    bootstrap.prepare_source(manifest)
    assert target.read_text() == "local_load()\n"
    target.write_text("local_load()\nchanged()\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        bootstrap.prepare_source(manifest)


def test_checkpoint_download_checks_hash_before_publication(tmp_path, monkeypatch):
    monkeypatch.setattr(bootstrap, "ASSETS", tmp_path)
    calls = []

    def download(*, id, output, quiet):
        calls.append(id)
        from pathlib import Path

        Path(output).write_bytes(b"official bytes")
        return output

    monkeypatch.setitem(sys.modules, "gdown", SimpleNamespace(download=download))
    manifest = {
        "checkpoint_assets": [{
            "filename": "weights.pth",
            "google_drive_id": "official-id",
            "sha256": hashlib.sha256(b"official bytes").hexdigest(),
        }]
    }
    bootstrap.prepare_checkpoint(manifest)
    bootstrap.prepare_checkpoint(manifest)
    assert calls == ["official-id"]
    target = tmp_path / "weights.pth"
    target.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="hash mismatch"):
        bootstrap.prepare_checkpoint(manifest)
    assert target.read_bytes() == b"corrupted"
    assert calls == ["official-id"]


def test_backbone_main_alias_cannot_change_silently(tmp_path, monkeypatch):
    monkeypatch.setattr(bootstrap, "ASSETS", tmp_path)
    asset = tmp_path / "cached.bin"
    asset.write_bytes(b"backbone")
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(hf_hub_download=lambda **kwargs: str(asset)),
    )
    manifest = {
        "backbone": {
            "repo_id": "owner/backbone",
            "revision": "pinned-revision",
            "files": [{
                "filename": "model.bin",
                "sha256": hashlib.sha256(b"backbone").hexdigest(),
            }],
        }
    }
    bootstrap.prepare_backbone(manifest)
    bootstrap.prepare_backbone(manifest)
    reference = tmp_path / "hf_cache/models--owner--backbone/refs/main"
    assert reference.read_text() == "pinned-revision"
    reference.write_text("unexpected-revision")
    with pytest.raises(ValueError, match="another revision"):
        bootstrap.prepare_backbone(manifest)
    assert reference.read_text() == "unexpected-revision"
