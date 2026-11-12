"""Neural triview tool integration through a real CPU subprocess, including HTTP errors."""

import sys

import pytest
from fastapi.testclient import TestClient

from percepta import Prompt
from percepta.config import AppConfig
from percepta.schema import Calibration, ToolContext
from percepta.service.app import create_app
from percepta.tools import triview_3d as module
from percepta.tools.base import ToolUnavailable
from tests import synth
from tests.unit.test_multiview_backend import scene, worker


def deployment(tmp_path, model="da3"):
    weights = tmp_path / "weights.bin"
    weights.write_bytes(b"local test weights")
    config = tmp_path / "config.json"
    config.write_text('{}')
    return {"triview_method": "mvs", "triview_backend": model, "checkpoint": str(weights),
            "model_config": str(config), "python": sys.executable, "source_dir": str(tmp_path)}


def context():
    images, K, transforms = scene()
    names = ("a", "b", "c")
    return ToolContext(request_id="triview-neural", images=dict(zip(names, images, strict=True)),
                       calibration={name: Calibration(camera=name, width=13, height=9, K=K[i].tolist(), T_world_cam=transforms[i].tolist())
                                    for i, name in enumerate(names)})


@pytest.mark.parametrize("model", ["da3", "mvsanywhere"])
def test_metric_depth_yields_world_geometry_and_model_provenance(tmp_path, monkeypatch, model):
    backend = worker(monkeypatch)
    monkeypatch.setattr(module, "MultiViewBackend", lambda **kwargs: backend)
    tool = module.Triview3DTool()
    try:
        tool.load(deployment(tmp_path, model))
        result = tool.perceive(context(), [])[0]
        assert result.status == "ok" and result.geometry.frame == "world"
        assert result.geometry.views_used == 3 and result.geometry.centroid[2] == pytest.approx(1.25)
        assert result.extras["geometry_scope"] == "whole_scene_visible_surface"
        assert result.extras["confidence_available"] is True
        assert result.provenance.model_sha256 and result.provenance.model
        assert result.geometry.method == f"triview_{model}"
        assert backend.describe()["calls"] == 1
    finally:
        tool.close()


def test_bad_da3_camera_geometry_is_rejected_before_inference(tmp_path, monkeypatch):
    backend = worker(monkeypatch)
    monkeypatch.setattr(module, "MultiViewBackend", lambda **kwargs: backend)
    tool = module.Triview3DTool()
    try:
        tool.load(deployment(tmp_path))
        ctx = context()
        ctx.calibration["c"].T_world_cam[1][3] = 0
        ctx.calibration["c"].T_world_cam[0][3] = 0.2
        result = tool.perceive(ctx, [])[0]
        assert result.status == "invalid_request" and result.geometry is None
        assert backend.describe()["calls"] == 0
        assert tool.perceive(context(), [])[0].status == "ok"
    finally:
        tool.close()


def test_always_verify_requires_declared_digest_before_worker_construction(tmp_path, monkeypatch):
    def forbidden(**kwargs):
        pytest.fail("invalid weights policy must fail before starting a worker")

    monkeypatch.setattr(module, "MultiViewBackend", forbidden)
    tool = module.Triview3DTool()
    with pytest.raises(ToolUnavailable, match="requires checkpoint_sha256"):
        tool.load({**deployment(tmp_path), "verify_weights": "always"})


def test_neural_deployment_can_report_missing_optional_pair_matcher(tmp_path, monkeypatch):
    backend = worker(monkeypatch)
    monkeypatch.setattr(module, "MultiViewBackend", lambda **kwargs: backend)
    tool = module.Triview3DTool()
    try:
        tool.load(deployment(tmp_path))
        monkeypatch.setitem(sys.modules, "cv2", None)
        ctx = context()
        ctx.context["triview_method"] = "pairs"
        with pytest.raises(ToolUnavailable, match="OpenCV"):
            tool.perceive(ctx, [])
        assert backend.describe()["calls"] == 0
        assert tool.perceive(context(), [])[0].status == "ok"
    finally:
        tool.close()


@pytest.mark.parametrize("mode", ["ok", "empty", "crash"])
def test_http_neural_geometry_and_terminal_unavailable(tmp_path, monkeypatch, mode):
    backend = worker(monkeypatch, mode)
    monkeypatch.setattr(module, "MultiViewBackend", lambda **kwargs: backend)
    config = AppConfig.model_validate({"pipeline": {"default": ["triview_3d"]}, "warmup": {"enabled": False},
                                       "cache": {"enabled": False}, "tools": {"triview_3d": deployment(tmp_path)}})
    ctx = context()
    request = {"images": {name: synth.image_ref(image).model_dump(mode="json") for name, image in ctx.images.items()},
               "calibration": {name: cal.model_dump(mode="json") for name, cal in ctx.calibration.items()},
               "prompts": [Prompt(type="text", text="scene").model_dump(mode="json")]}
    with TestClient(create_app(config)) as client:
        response = client.post("/v1/perceive", json=request)
        assert response.status_code == 200, response.text
        body = response.json()
        if mode == "ok":
            assert body["fused"]["geometry"]["centroid"][2] == pytest.approx(1.25)
        elif mode == "empty":
            assert body["fused"] is None
        else:
            assert body["fused"] is None and "tool_unavailable" in body["warnings"]
            assert all(p["status"] == "unavailable" for p in body["percepts"])
        health = next(row for row in client.get("/healthz").json()["tools"] if row["name"] == "triview_3d")
        assert health["status"] == ("error" if mode == "crash" else "ok")
    assert backend._process is None


def test_idle_worker_death_is_reported_without_waiting_for_inference(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    backend = worker(monkeypatch)
    monkeypatch.setattr(module, "MultiViewBackend", lambda **kwargs: backend)
    config = AppConfig.model_validate({"pipeline": {"default": ["triview_3d"]}, "warmup": {"enabled": False},
                                       "tools": {"triview_3d": deployment(tmp_path)}})
    with TestClient(create_app(config)) as client, ThreadPoolExecutor(max_workers=1) as pool:
        process = backend._process
        process.kill()
        process.wait(timeout=3)
        # Health must not take the lock held throughout model inference.
        backend._lock.acquire()
        try:
            health = pool.submit(client.get, "/healthz").result(timeout=2).json()
        finally:
            backend._lock.release()
        assert next(row for row in health["tools"] if row["name"] == "triview_3d")["status"] == "error"
        row = next(row for row in client.get("/v1/tools").json()["tools"] if row["name"] == "triview_3d")
        assert row["status"] == "error" and "exited" in row["error"]
        assert backend._process is process


def test_bad_da3_image_size_returns_invalid_request_then_normal_request_succeeds(tmp_path, monkeypatch):
    import numpy as np

    backend = worker(monkeypatch)
    backend.name = "da3"
    monkeypatch.setattr(module, "MultiViewBackend", lambda **kwargs: backend)
    config = AppConfig.model_validate({"pipeline": {"default": ["triview_3d"]}, "warmup": {"enabled": False},
                                       "cache": {"enabled": False}, "tools": {"triview_3d": deployment(tmp_path)}})
    ctx = context()
    bad_calibration = {name: cal.model_copy(update={"width": 1000, "height": 1}) for name, cal in ctx.calibration.items()}
    bad = {"images": {name: synth.image_ref(np.zeros((1, 1000, 3), np.uint8)).model_dump(mode="json") for name in ctx.images},
           "calibration": {name: cal.model_dump(mode="json") for name, cal in bad_calibration.items()},
           "prompts": [Prompt(type="text", text="scene").model_dump(mode="json")]}
    with TestClient(create_app(config)) as client:
        pid = backend.describe()["pid"]
        response = client.post("/v1/perceive", json=bad)
        assert response.status_code == 200
        assert response.json()["percepts"] and all(p["status"] == "invalid_request" for p in response.json()["percepts"])
        assert "tool_unavailable" not in response.json()["warnings"]
        assert backend.describe()["pid"] == pid and backend.describe()["calls"] == 0
        normal = {"images": {name: synth.image_ref(image).model_dump(mode="json") for name, image in ctx.images.items()},
                  "calibration": {name: cal.model_dump(mode="json") for name, cal in ctx.calibration.items()}, "prompts": bad["prompts"]}
        result = client.post("/v1/perceive", json=normal)
        assert result.status_code == 200 and result.json()["fused"]["status"] == "ok"
        assert backend.describe()["pid"] == pid
