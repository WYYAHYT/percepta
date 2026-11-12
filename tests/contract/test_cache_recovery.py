"""Temporary model failures cannot turn into cached failures after the service recovers."""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from percepta.backends.protocol import ResponseFrame, decode_request, encode_response
from percepta.client import build_perceive_request
from percepta.config import AppConfig
from percepta.registry import ToolRegistry
from percepta.service.app import create_app
from percepta.tools.sam3 import Sam3Tool


@pytest.fixture
def recovering_model_service():
    state = SimpleNamespace(fail_path="/segment", fail_kind="http", failed=False, calls=[], shape=(8, 8))

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            frame = decode_request(self.rfile.read(int(self.headers["Content-Length"])))
            state.calls.append(self.path)
            if frame.image is not None:
                state.shape = frame.image.shape[:2]
            status = 200
            if self.path == state.fail_path and not state.failed:
                state.failed = True
                if state.fail_kind == "http":
                    status, body = 503, b"temporarily busy"
                else:
                    body = encode_response(ResponseFrame("error", error="temporarily busy", error_kind="unavailable"))
            elif frame.image is not None:
                body = encode_response(ResponseFrame("ok"))
            else:
                body = encode_response(ResponseFrame(
                    "ok", masks=np.ones((1, *state.shape), dtype=bool), scores=np.array([1.0]),
                ))
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


@pytest.mark.parametrize("fail_path", ["/set_image", "/segment"])
@pytest.mark.parametrize("fail_kind", ["http", "frame"])
def test_model_recovers_with_default_cache_after_temporary_failure(recovering_model_service, fail_path, fail_kind):
    endpoint, state = recovering_model_service
    state.fail_path, state.fail_kind = fail_path, fail_kind
    registry = ToolRegistry(discover=False)
    registry.register("sam3", Sam3Tool)
    config = AppConfig.model_validate({
        "pipeline": {"default": ["sam3"]}, "warmup": {"enabled": False},
        "tools": {"sam3": {"backend": "http", "endpoint": endpoint}},
    })
    request = build_perceive_request(
        {"cam": np.full((8, 8, 3), 255, dtype=np.uint8)}, ["object"], options={"fuse": False},
    ).model_dump(mode="json")
    with TestClient(create_app(config, registry=registry)) as client:
        failed = client.post("/v1/perceive", json=request)
        assert failed.status_code == 200
        failed_percept = failed.json()["percepts"][0]
        assert failed_percept["status"] == "unavailable"
        assert failed_percept["provenance"]["cache_hit"] is False

        recovered = client.post("/v1/perceive", json=request)
        assert recovered.status_code == 200
        percept = recovered.json()["percepts"][0]
        assert percept["status"] == "ok"
        assert percept["provenance"]["cache_hit"] is False
        assert percept["observations"][0]["area_px"] == 64
        assert state.calls == (["/set_image"] if fail_path == "/set_image" else ["/set_image", "/segment"]) + [
            "/set_image", "/segment",
        ]

        calls = list(state.calls)
        cached = client.post("/v1/perceive", json=request)
        assert cached.status_code == 200
        cached_percept = cached.json()["percepts"][0]
        assert cached_percept["status"] == "ok"
        assert cached_percept["provenance"]["cache_hit"] is True
        assert cached_percept["observations"] == percept["observations"]
        assert state.calls == calls
