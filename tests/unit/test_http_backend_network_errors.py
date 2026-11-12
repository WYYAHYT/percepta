"""HTTP model errors retain their classification when the error body fails in transit."""

from __future__ import annotations

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from percepta.backends.base import BackendRequestError, BackendUnavailable
from percepta.backends.http_backend import HttpBackend
from percepta.backends.protocol import ResponseFrame, encode_response
from percepta.client import build_perceive_request
from percepta.config import AppConfig
from percepta.registry import ToolRegistry
from percepta.schema import Prompt
from percepta.service.app import create_app
from percepta.tools.sam3 import Sam3Tool


@pytest.fixture
def error_model_service():
    state = SimpleNamespace(mode="healthy", status=503, calls=0, error_path=None)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            state.calls += 1
            mode, status = state.mode, state.status
            if state.error_path is not None and self.path != state.error_path:
                mode = "healthy"
            if mode == "healthy":
                body = encode_response(ResponseFrame("ok"))
                status, size = 200, len(body)
            else:
                body = b"x" * 400 if mode == "bounded" else b"error detail"
                size = len(body) + (1 if mode == "short" else 400 if mode == "bounded" else 0)
            self.send_response(status)
            self.send_header("Content-Length", str(size))
            self.end_headers()
            if mode == "slow":
                time.sleep(0.15)
            try:
                self.wfile.write(body)
                self.wfile.flush()
                if mode == "bounded":
                    time.sleep(0.15)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    backend = HttpBackend(endpoint=f"http://127.0.0.1:{server.server_port}", timeout_s=0.04)
    try:
        yield backend, state
    finally:
        backend.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


@pytest.mark.parametrize("operation", ["prepare", "segment"])
@pytest.mark.parametrize("status", [400, 503])
@pytest.mark.parametrize("mode", ["complete", "short", "slow", "bounded"])
def test_http_error_bodies_use_backend_errors_and_preserve_recovery(error_model_service, operation, status, mode):
    backend, state = error_model_service
    image = np.full((8, 8, 3), 255, dtype=np.uint8)
    if operation == "segment":
        backend.set_image(image)
    state.mode, state.status = mode, status
    expected = BackendRequestError if status == 400 and mode in {"complete", "bounded"} else BackendUnavailable
    with pytest.raises(expected):
        if operation == "prepare":
            backend.set_image(image)
        else:
            backend.segment(Prompt(type="text", text="red"))
    assert bool(backend.health_error) is (mode == "short")

    calls = state.calls
    state.mode = "healthy"
    if mode == "short":
        with pytest.raises(BackendUnavailable):
            backend.set_image(image)
        assert state.calls == calls
    else:
        backend.set_image(image)
        assert state.calls == calls + 1


@pytest.mark.parametrize("mode", ["short", "slow"])
@pytest.mark.parametrize("path", ["/set_image", "/segment"])
def test_error_body_transport_failure_is_unavailable_percept_over_http(error_model_service, mode, path):
    backend, state = error_model_service
    state.mode, state.error_path = mode, path
    registry = ToolRegistry(discover=False)
    registry.register("sam3", lambda: Sam3Tool(backend_factory=lambda **_kwargs: backend))
    config = AppConfig.model_validate({
        "pipeline": {"default": ["sam3"]}, "warmup": {"enabled": False}, "cache": {"enabled": False},
    })
    request = build_perceive_request(
        {"cam": np.full((8, 8, 3), 255, dtype=np.uint8)}, ["object"], options={"fuse": False},
    )
    with TestClient(create_app(config, registry=registry), raise_server_exceptions=False) as client:
        response = client.post("/v1/perceive", json=request.model_dump(mode="json"))
    assert response.status_code == 200, response.text
    percept = response.json()["percepts"][0]
    assert percept["status"] == "unavailable"
    assert percept["observations"] == []
