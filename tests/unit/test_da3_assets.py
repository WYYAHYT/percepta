"""DA3 deployment preparation rejects changed source and assets without overwriting."""

import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from scripts import bootstrap_da3 as bootstrap


@pytest.fixture
def deployment(tmp_path, monkeypatch):
    source, assets = tmp_path / "source", tmp_path / "assets"
    source.mkdir()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"source": {"url": "official", "revision": "pinned"}, "weights": {
        "repo_id": "official/DA3-BASE", "revision": "pinned-weights", "files": [{
            "filename": "model.safetensors", "sha256": hashlib.sha256(b"official weights").hexdigest(),
        }],
    }}))
    monkeypatch.setattr(bootstrap, "MANIFEST", manifest)
    monkeypatch.setattr(bootstrap, "SOURCE", source)
    monkeypatch.setattr(bootstrap, "ASSETS", assets)
    monkeypatch.setattr(bootstrap.subprocess, "check_output", lambda argv, **kwargs: "" if "status" in argv else "pinned\n")
    cached = tmp_path / "cached"
    cached.write_bytes(b"official weights")
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(cached)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download))
    return source, assets, cached, calls


def test_prepare_downloads_pinned_asset_once_and_checks_existing_bytes(deployment):
    _source, assets, _cached, calls = deployment
    bootstrap.prepare()
    bootstrap.prepare()
    assert len(calls) == 1
    assert calls[0]["revision"] == "pinned-weights"
    model = assets / "model.safetensors"
    assert model.read_bytes() == b"official weights"
    model.write_bytes(b"wrong existing bytes")
    with pytest.raises(ValueError, match="hash mismatch"):
        bootstrap.prepare()
    assert model.read_bytes() == b"wrong existing bytes"
    assert len(calls) == 1


def test_bad_download_is_never_published(deployment):
    _source, assets, cached, _calls = deployment
    cached.write_bytes(b"corrupted download")
    with pytest.raises(ValueError, match="hash mismatch"):
        bootstrap.prepare()
    assert not (assets / "model.safetensors").exists()


@pytest.mark.parametrize("reply", ["wrong-revision", " M modified.py"])
def test_wrong_or_dirty_source_rejected_before_download(deployment, monkeypatch, reply):
    _source, _assets, _cached, calls = deployment
    monkeypatch.setattr(bootstrap.subprocess, "check_output", lambda *a, **kw: reply)
    with pytest.raises(ValueError, match="source must be clean"):
        bootstrap.prepare()
    assert calls == []
