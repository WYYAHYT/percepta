"""Client readiness deadlines and local runner replacement release their resources."""

from __future__ import annotations

import io
import json
import socket
import threading
import time
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

from percepta.client import PerceptaClient, PerceptaError
from percepta.config import AppConfig
from percepta.registry import ToolRegistry
from percepta.tools.fake import FakeTool


@pytest.mark.parametrize("mode", ["delayed", "busy", "warming", "trickle", "error_trickle"])
def test_readiness_budget_includes_socket_retry_backoff_and_poll_interval(mode: str) -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if mode == "delayed":
                time.sleep(0.3)
            body = json.dumps({"ok": mode != "warming"}).encode()
            self.send_response(503 if mode in {"busy", "error_trickle"} else 200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                if "trickle" in mode:
                    for value in body:
                        self.wfile.write(bytes([value]))
                        self.wfile.flush()
                        time.sleep(0.025)
                else:
                    self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *_args) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    try:
        client = PerceptaClient(f"http://127.0.0.1:{server.server_port}", timeout=1, backoff_s=0.3)
        started = time.monotonic()
        assert client.wait_until_ready(timeout_s=0.04, interval_s=0.3) is False
        assert time.monotonic() - started < 0.18
        assert client.timeout == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


@pytest.mark.parametrize("part", ["status", "headers"])
@pytest.mark.parametrize("status", [200, 503])
def test_readiness_deadline_includes_slow_status_and_headers(part: str, status: int) -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            body = b'{"ok":true}'
            status_line = f"HTTP/1.1 {status} Test\r\n".encode()
            length = f"Content-Length: {len(body)}\r\n".encode()
            if part == "status":
                prefix, slow, suffix = b"", status_line, length + b"\r\n" + body
            else:
                prefix, slow, suffix = status_line + length, b"X-Slow: abcde\r\n", b"\r\n" + body
            try:
                self.connection.sendall(prefix)
                for value in slow:
                    self.connection.sendall(bytes([value]))
                    time.sleep(0.015)
                self.connection.sendall(suffix)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *_args) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    try:
        client = PerceptaClient(f"http://127.0.0.1:{server.server_port}", timeout=1, retries=0)
        started = time.monotonic()
        assert client.wait_until_ready(timeout_s=0.08, interval_s=0.01) is False
        assert time.monotonic() - started < 0.18
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


def test_readiness_deadline_includes_chunk_delimiter_reads() -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            try:
                self.connection.sendall(b'HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\nb\r\n{"ok":true}')
                time.sleep(0.09)
                self.connection.sendall(b"\r")
                time.sleep(0.09)
                self.connection.sendall(b"\n0\r\n\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *_args) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    try:
        client = PerceptaClient(f"http://127.0.0.1:{server.server_port}", timeout=1, retries=0)
        started = time.monotonic()
        assert client.wait_until_ready(timeout_s=0.12, interval_s=0.01) is False
        assert time.monotonic() - started < 0.16
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


def test_redirected_connection_receives_only_remaining_readiness_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    import percepta.client as client_module

    timeouts = []
    original = client_module._DeadlineHTTPConnection.__init__

    def record_timeout(self, *_args, **kwargs):
        timeouts.append(kwargs["timeout"])
        original(self, *_args, **kwargs)

    monkeypatch.setattr(client_module._DeadlineHTTPConnection, "__init__", record_timeout)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/healthz":
                time.sleep(0.05)
                self.send_response(302)
                self.send_header("Location", "/ready")
                body = b""
            else:
                self.send_response(200)
                body = b'{"ok":true}'
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    try:
        client = PerceptaClient(f"http://127.0.0.1:{server.server_port}", timeout=1, retries=0)
        assert client.wait_until_ready(timeout_s=0.2) is True
        assert len(timeouts) == 2
        assert timeouts[1] < timeouts[0] - 0.03
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


def test_direct_socket_timeout_uses_the_client_error_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    def timed_out(*_args, **_kwargs):
        raise TimeoutError("timed out")

    monkeypatch.setattr("urllib.request.urlopen", timed_out)
    monkeypatch.setattr("percepta.client._deadline_urlopen", timed_out)
    client = PerceptaClient(retries=0)
    with pytest.raises(PerceptaError) as caught:
        client.health()
    assert caught.value.code == "connection_error"
    assert caught.value.status_code is None
    assert client.wait_until_ready(timeout_s=0.02, interval_s=0.01) is False


def test_socket_timeout_while_reading_http_error_is_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    class ErrorBody(io.BytesIO):
        def read(self, *_args):
            raise TimeoutError("slow error body")

    def slow_error(*_args, **_kwargs):
        raise urllib.error.HTTPError("http://local/healthz", 503, "busy", None, ErrorBody())

    monkeypatch.setattr("urllib.request.urlopen", slow_error)
    with pytest.raises(PerceptaError) as caught:
        PerceptaClient(retries=0).health()
    assert caught.value.code == "connection_error"


@pytest.mark.parametrize("status", [200, 503])
@pytest.mark.parametrize("buffered", [False, True])
def test_readiness_body_reader_accepts_bytesio_and_simple_mock_responses(
    monkeypatch: pytest.MonkeyPatch, status: int, buffered: bool,
) -> None:
    class MockResponse:
        def __init__(self) -> None:
            self.status = status
            self.closed = False

        def read(self) -> bytes:
            return b'{"ok": true}'

        def close(self) -> None:
            self.closed = True

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    streams = []

    def open_response(*_args, **_kwargs):
        stream = io.BytesIO(b'{"ok": true}') if buffered else MockResponse()
        stream.status = status
        streams.append(stream)
        if status == 503:
            raise urllib.error.HTTPError("http://local/healthz", status, "busy", None, stream)
        return stream

    monkeypatch.setattr("urllib.request.urlopen", open_response)
    monkeypatch.setattr("percepta.client._deadline_urlopen", open_response)
    client = PerceptaClient(retries=0)
    assert client.wait_until_ready(timeout_s=0.025, interval_s=0.01) is (status == 200)
    if status == 200:
        assert client.health() == {"ok": True}
    else:
        with pytest.raises(PerceptaError) as caught:
            client.health()
        assert caught.value.status_code == 503
        assert caught.value.body == {"ok": True}
    assert all(stream.closed for stream in streams)


class _TrackedFake(FakeTool):
    def __init__(self) -> None:
        super().__init__()
        self.closed = 0
        self.load_calls = 0

    def load(self, config) -> None:
        self.load_calls += 1
        super().load(config)

    def close(self) -> None:
        self.closed += 1
        super().close()


def _call_local(client: PerceptaClient, registry: ToolRegistry, config: AppConfig):
    return client.perceive_local(
        {"cam": np.zeros((4, 4, 3), dtype=np.uint8)}, ["red block"], registry=registry,
        config=config, options={"tools": ["fake"], "fuse": False},
    )


@pytest.mark.parametrize("same_registry", [False, True])
def test_replacing_local_runner_closes_previous_before_reusing_registry(same_registry: bool) -> None:
    first_tool, second_tool = _TrackedFake(), _TrackedFake()
    first_registry, second_registry = ToolRegistry(), ToolRegistry()
    first_registry.register("fake", first_tool)
    second_registry.register("fake", second_tool)
    config = AppConfig()
    with PerceptaClient() as client:
        _call_local(client, first_registry, config)
        old_runner = client._local_runner
        target = first_registry if same_registry else second_registry
        response = _call_local(client, target, config.model_copy(deep=True))
        assert client._local_runner is not old_runner
        assert old_runner._closed is True
        assert first_tool.closed == 1
        assert target.entry("fake").instance is not None
        assert response.percepts[0].status == "not_found"
        assert first_tool.load_calls == (2 if same_registry else 1)
    assert first_tool.closed == (2 if same_registry else 1)
    assert second_tool.closed == (0 if same_registry else 1)


def test_failed_local_runner_replacement_does_not_keep_previous_resources(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = ToolRegistry()
    tool = _TrackedFake()
    registry.register("fake", tool)
    client = PerceptaClient()
    _call_local(client, registry, AppConfig())

    def fail(*_args, **_kwargs):
        raise RuntimeError("cannot construct runner")

    monkeypatch.setattr("percepta.pipeline.PipelineRunner", fail)
    with pytest.raises(RuntimeError, match="cannot construct"):
        _call_local(client, ToolRegistry(), AppConfig())
    assert tool.closed == 1
    assert client._local_runner is None
    client.close()
    assert tool.closed == 1
