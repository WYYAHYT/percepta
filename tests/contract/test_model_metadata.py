"""Model introspection reports the same verified weight hash as perception provenance."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from percepta.client import build_perceive_request
from percepta.config import AppConfig
from percepta.registry import ToolRegistry
from percepta.service.app import create_app
from percepta.tools.stereo_3d import Stereo3DTool
from percepta.tools.triview_3d import Triview3DTool


def test_sam3_info_hash_matches_subprocess_provenance(tmp_path: Path) -> None:
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"CPU metadata fixture")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    config = AppConfig.model_validate({
        "pipeline": {"default": ["sam3"]}, "warmup": {"enabled": False}, "cache": {"enabled": False},
        "tools": {"sam3": {"backend": "subprocess", "python": sys.executable,
                           "worker_backend_factory": "percepta.backends.reference:ThresholdAdapter",
                           "checkpoint": str(checkpoint), "checkpoint_sha256": digest,
                           "verify_weights": "always", "device": "cpu"}},
    })
    request = build_perceive_request({"cam": np.full((8, 8, 3), 255, dtype=np.uint8)}, ["object"], options={"fuse": False})
    with TestClient(create_app(config)) as client:
        response = client.post("/v1/perceive", json=request.model_dump(mode="json"))
        assert response.status_code == 200, response.text
        percept = response.json()["percepts"][0]
        assert percept["status"] == "ok"
        row = next(row for row in client.get("/v1/info").json()["tools"] if row["name"] == "sam3")
        assert row["loaded"] is True
        assert row["model_sha256"] == percept["provenance"]["model_sha256"] == digest


@pytest.mark.parametrize("factory", [Stereo3DTool, Triview3DTool])
def test_geometry_info_reads_the_provenance_hash_field(factory) -> None:
    tool = factory()
    tool._model_sha256 = "a" * 64
    registry = ToolRegistry(discover=False)
    registry.register(tool.name, factory)
    registry.entry(tool.name).instance = tool
    config = AppConfig.model_validate({"pipeline": {"default": [tool.name]}, "warmup": {"enabled": False}})
    with TestClient(create_app(config, registry=registry)) as client:
        row = client.get("/v1/info").json()["tools"][0]
    assert row["model_sha256"] == tool._model_sha256
