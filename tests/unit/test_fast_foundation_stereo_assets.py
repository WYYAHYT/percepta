"""Pinned research assets are checked before any model unpickling or replacement."""

import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from percepta.backends import fast_foundation_stereo_worker as worker
from scripts import bootstrap_fast_foundation_stereo as bootstrap


def test_adapter_requires_official_bytes_before_loading_torch(tmp_path, monkeypatch):
    checkpoint = tmp_path / "renamed.pth"
    checkpoint.write_bytes(b"official bytes")
    manifest = tmp_path / "assets.json"
    manifest.write_text(json.dumps({"checkpoint_assets": [{
        "filename": "model_best_bp2_serialize.pth", "sha256": hashlib.sha256(b"official bytes").hexdigest(),
    }]}))
    monkeypatch.setattr(worker, "MANIFEST", manifest)
    assert worker.require_official_asset(checkpoint, "model_best_bp2_serialize.pth") == checkpoint
    checkpoint.write_bytes(b"untrusted pickle")
    with pytest.raises(ValueError, match="hash mismatch"):
        worker.FastFoundationStereoAdapter(source_dir=str(tmp_path), checkpoint=str(checkpoint))
    assert checkpoint.read_bytes() == b"untrusted pickle"


@pytest.mark.parametrize("options", [{"valid_iters": 0}, {"valid_iters": True}, {"max_disp": 0}, {"max_disp": 193}, {"max_disp": 640}])
def test_invalid_runtime_options_are_rejected_before_loading_assets(tmp_path, options):
    with pytest.raises(ValueError):
        worker.FastFoundationStereoAdapter(source_dir=str(tmp_path), checkpoint="missing.pth", **options)


def test_bootstrap_checks_hash_before_publishing_and_is_repeatable(tmp_path, monkeypatch):
    monkeypatch.setattr(bootstrap, "ASSETS", tmp_path)
    calls = []

    def download(*, id, output, quiet):
        from pathlib import Path

        calls.append(id)
        Path(output).write_bytes(b"official bytes")
        return output

    monkeypatch.setitem(sys.modules, "gdown", SimpleNamespace(download=download))
    manifest = {"checkpoint_assets": [{
        "filename": "model.pth", "google_drive_id": "official-id",
        "sha256": hashlib.sha256(b"official bytes").hexdigest(),
    }]}
    bootstrap.prepare_checkpoint(manifest)
    bootstrap.prepare_checkpoint(manifest)
    assert calls == ["official-id"]
    (tmp_path / "model.pth").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        bootstrap.prepare_checkpoint(manifest)
    assert (tmp_path / "model.pth").read_bytes() == b"corrupt"
    manifest["checkpoint_assets"][0]["filename"] = "bad-download.pth"
    manifest["checkpoint_assets"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        bootstrap.prepare_checkpoint(manifest)
    assert not (tmp_path / "bad-download.pth").exists()


def test_source_changes_are_rejected_without_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(bootstrap, "SOURCE", tmp_path)
    manifest = {"source": {"revision": "pinned"}}

    def output(argv, **kwargs):
        return "pinned\n" if "rev-parse" in argv else " M core/model.py\n"

    monkeypatch.setattr(bootstrap.subprocess, "check_output", output)
    with pytest.raises(ValueError, match="modified tracked files"):
        bootstrap.prepare_source(manifest)
