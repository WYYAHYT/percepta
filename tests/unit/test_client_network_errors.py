"""SDK transport failures use PerceptaError and cannot report readiness."""

from __future__ import annotations

import threading
import time
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from percepta.client import PerceptaClient, PerceptaError

_BROKEN_RESPONSES = [
    pytest.param(b"", id="disconnect-before-status"),
    pytest.param(b'HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\n{"ok":', id="truncated-success"),
    pytest.param(b'HTTP/1.1 503 Busy\r\nContent-Length: 100\r\n\r\n{"error":', id="truncated-error"),
    pytest.param(b'HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\n{"ok":true}', id="truncated-valid-json"),
]


@pytest.fixture
def network_client(wire_response: bytes):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.connection.sendall(wire_response)
            self.close_connection = True

        def log_message(self, *_args) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    try:
        yield PerceptaClient(f"http://127.0.0.1:{server.server_port}", timeout=1, retries=0)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


@pytest.mark.parametrize("wire_response", _BROKEN_RESPONSES)
def test_disconnected_or_truncated_response_uses_sdk_error_contract(network_client: PerceptaClient) -> None:
    with pytest.raises(PerceptaError) as caught:
        network_client.health()
    assert caught.value.code == "connection_error"
    assert caught.value.status_code is None


@pytest.mark.parametrize("wire_response", _BROKEN_RESPONSES)
def test_disconnected_or_truncated_response_cannot_report_readiness(network_client: PerceptaClient) -> None:
    started = time.monotonic()
    assert network_client.wait_until_ready(timeout_s=0.05, interval_s=0.01) is False
    assert time.monotonic() - started < 0.2


@pytest.mark.parametrize("wire_response", [
    pytest.param(b'HTTP/1.1 200 OK\r\nContent-Length: 11\r\n\r\n{"ok":true}', id="content-length"),
    pytest.param(b'HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\nb\r\n{"ok":true}\r\n0\r\n\r\n', id="chunked"),
    pytest.param(b'HTTP/1.0 200 OK\r\n\r\n{"ok":true}', id="close-delimited"),
])
def test_complete_response_framing_remains_supported(network_client: PerceptaClient) -> None:
    assert network_client.health() == {"ok": True}
    assert network_client.wait_until_ready(timeout_s=0.1, interval_s=0.01) is True


@pytest.mark.parametrize("failure_at", ["connect", "success-body", "error-body"])
def test_socket_reset_uses_sdk_error_contract(monkeypatch: pytest.MonkeyPatch, failure_at: str) -> None:
    class ResetBody:
        status = 200
        closed = False

        def read(self) -> bytes:
            raise ConnectionResetError("connection reset")

        def close(self) -> None:
            self.closed = True

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            self.close()

    def open_response(*_args, **_kwargs):
        if failure_at == "connect":
            raise ConnectionResetError("connection reset")
        if failure_at == "error-body":
            raise urllib.error.HTTPError("http://local/healthz", 503, "busy", None, ResetBody())
        return ResetBody()

    monkeypatch.setattr("urllib.request.urlopen", open_response)
    monkeypatch.setattr("percepta.client._deadline_urlopen", open_response)
    client = PerceptaClient(retries=0)
    with pytest.raises(PerceptaError) as caught:
        client.health()
    assert caught.value.code == "connection_error"
    assert caught.value.status_code is None
    assert client.wait_until_ready(timeout_s=0.025, interval_s=0.01) is False
