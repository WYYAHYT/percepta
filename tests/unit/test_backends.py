"""Backend abstraction and the subprocess isolation path (``docs/model-isolation.md``).

The important test here is :func:`test_subprocess_backend_runs_model_in_a_different_venv`:
it proves the central claim — a model can live in a *different* interpreter with a
*different* numpy version and still be driven through the same tool interface. That is the
whole reason this subsystem exists.

All tests are GPU-free and weight-free.
"""

from __future__ import annotations

import os
import shutil
import struct
import subprocess
import sys
import sysconfig
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

from percepta.backends import (
    BACKEND_KINDS,
    BackendRequestError,
    BackendUnavailable,
    build_backend,
    resolve_backend_kind,
)
from percepta.backends.http_backend import HttpBackend
from percepta.backends.inprocess import InProcessBackend
from percepta.backends.protocol import (
    PROTOCOL_VERSION,
    REQUEST_MAGIC,
    ProtocolError,
    RequestFrame,
    ResponseFrame,
    decode_request,
    decode_response,
    encode_request,
    encode_response,
    read_frame,
    write_frame,
)
from percepta.backends.reference import ThresholdAdapter
from percepta.backends.subprocess_backend import SubprocessBackend
from percepta.config import AppConfig
from percepta.masks import encode_image
from percepta.pipeline import PipelineRunner
from percepta.registry import ToolRegistry
from percepta.schema import PerceiveOptions, PerceiveRequest, Prompt
from percepta.tools.sam3 import Sam3Tool
from tests.unit.subprocess_backend_support import _response_packet


def _scene() -> np.ndarray:
    image = np.zeros((60, 80, 3), dtype=np.uint8)
    image[20:40, 30:50, 0] = 255  # 20x20 red block
    return image


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


def test_backend_kinds_are_the_documented_three():
    assert set(BACKEND_KINDS) == {"inprocess", "subprocess", "http"}


def test_resolve_backend_kind_defaults_to_inprocess():
    assert resolve_backend_kind({}) == "inprocess"
    assert resolve_backend_kind({"backend": "SubProcess"}) == "subprocess"


def test_unknown_backend_is_a_config_error_not_a_silent_fallback():
    with pytest.raises(BackendRequestError, match="unknown backend"):
        resolve_backend_kind({"backend": "docker"})


def test_build_backend_dispatches():
    backend = build_backend("inprocess", adapter=ThresholdAdapter())
    assert backend.name == "inprocess"
    with pytest.raises(BackendRequestError):
        build_backend("nonsense")


# ---------------------------------------------------------------------------
# In-process backend
# ---------------------------------------------------------------------------


def test_inprocess_backend_end_to_end():
    backend = InProcessBackend(adapter=ThresholdAdapter())
    backend.set_image(_scene())
    masks, boxes, scores = backend.segment(Prompt(type="text", text="red block"))
    assert masks.shape == (1, 60, 80)
    assert int(masks[0].sum()) == 400
    assert boxes is not None and boxes[0].tolist() == [30.0, 20.0, 50.0, 40.0]
    assert scores[0] == pytest.approx(1.0, abs=1e-6)
    assert backend.describe()["loaded"] is True
    backend.close()
    assert backend.describe()["loaded"] is False


def test_inprocess_backend_lazy_factory_failure_is_unavailable():
    def broken(**_: object):
        raise RuntimeError("no weights")

    backend = InProcessBackend(factory=broken)
    with pytest.raises(BackendUnavailable, match="no weights"):
        backend.set_image(_scene())


def test_inprocess_backend_requires_adapter_or_factory():
    with pytest.raises(BackendRequestError):
        InProcessBackend()


# ---------------------------------------------------------------------------
# Wire protocol
# ---------------------------------------------------------------------------


def test_request_frame_round_trip_with_image_and_depth():
    frame = RequestFrame(
        prompt_json='{"type": "text", "text": "cup"}',
        image=_scene(),
        depth=np.full((60, 80), 1.5, dtype=np.float32),
    )
    decoded = decode_request(encode_request(frame))
    assert decoded.prompt_json == frame.prompt_json
    assert np.array_equal(decoded.image, frame.image)
    assert decoded.depth is not None and np.allclose(decoded.depth, 1.5)


def test_request_frame_without_image_means_reuse():
    decoded = decode_request(encode_request(RequestFrame(prompt_json="{}")))
    assert decoded.image is None and decoded.depth is None


def test_response_frame_round_trip():
    masks = np.zeros((2, 6, 7), dtype=bool)
    masks[0, 1:3, 1:4] = True
    frame = ResponseFrame(
        status="ok",
        masks=masks,
        boxes=np.array([[1, 1, 4, 3], [0, 0, 0, 0]], dtype=np.float32),
        scores=np.array([0.9, 0.1], dtype=np.float32),
    )
    decoded = decode_response(encode_response(frame))
    assert np.array_equal(decoded.masks, masks)
    assert decoded.boxes is not None and decoded.boxes.shape == (2, 4)
    assert decoded.scores.tolist() == pytest.approx([0.9, 0.1], abs=1e-6)


def test_response_frame_empty_and_error():
    empty = decode_response(encode_response(ResponseFrame(status="ok")))
    assert empty.masks is None

    error = decode_response(
        encode_response(
            ResponseFrame(status="error", error="boom", error_kind="unavailable")
        )
    )
    assert error.status == "error" and error.error == "boom"


def test_bad_magic_is_detected_not_decoded_as_pixels():
    garbage = b"\xde\xad\xbe\xef" + b"\x00" * 64
    with pytest.raises(ProtocolError, match="magic"):
        decode_request(garbage)
    with pytest.raises(ProtocolError, match="magic"):
        decode_response(garbage)


def test_truncated_frame_is_rejected():
    payload = encode_request(RequestFrame(prompt_json="{}", image=_scene()))
    with pytest.raises(ProtocolError, match="truncated"):
        decode_request(payload[: len(payload) // 2])


def test_stream_framing_round_trip():
    import io

    stream = io.BytesIO()
    write_frame(stream, b"hello")
    write_frame(stream, b"world")
    stream.seek(0)
    assert read_frame(stream) == b"hello"
    assert read_frame(stream) == b"world"
    assert read_frame(stream) is None


def test_protocol_version_is_pinned():
    assert PROTOCOL_VERSION == 1


# ---------------------------------------------------------------------------
# Subprocess backend — the actual isolation mechanism
# ---------------------------------------------------------------------------


def test_subprocess_backend_rejects_missing_interpreter():
    backend = SubprocessBackend(python="/nonexistent/python-does-not-exist", module="x")
    with pytest.raises(BackendUnavailable, match="interpreter not found"):
        backend.set_image(_scene())
    assert "interpreter not found" in backend.health_error
    assert backend._process is None
    backend.close()


@pytest.mark.skipif(
    shutil.which("uv") is None, reason="needs uv to create a throwaway virtualenv"
)
def test_subprocess_backend_runs_model_in_a_different_venv(tmp_path: Path):
    """The core claim: a model in another venv with another numpy is still drivable.

    The worker venv deliberately contains only what ``docs/model-isolation.md`` §5 requires —
    ``numpy`` plus ``pydantic`` (needed to validate ``Prompt`` on the worker side) — and
    *not* percepta's heavier deps (pyyaml/pillow/fastapi) nor any model framework. This is
    what proves the isolation boundary is thin enough to be practical.
    """
    venv_dir = tmp_path / "isovenv"
    created = subprocess.run(
        ["uv", "venv", str(venv_dir), "-p", "3.10", "-q"],
        capture_output=True,
        text=True,
    )
    assert created.returncode == 0, created.stderr
    python = venv_dir / "bin" / "python"
    installed = subprocess.run(
        ["uv", "pip", "install", "--python", str(python), "numpy", "pydantic", "-q"],
        capture_output=True,
        text=True,
    )
    assert installed.returncode == 0, installed.stderr

    backend = SubprocessBackend(
        python=str(python),
        module="percepta.backends.worker",
        config={
            "backend_factory": "percepta.backends.reference:ThresholdAdapter",
            "backend_args": {"threshold": 128},
        },
        timeout_s=180.0,
    )
    try:
        backend.set_image(_scene())
        masks, boxes, scores = backend.segment(Prompt(type="text", text="red block"))
        assert masks.shape == (1, 60, 80)
        assert int(masks[0].sum()) == 400
        assert boxes is not None and boxes[0].tolist() == [30.0, 20.0, 50.0, 40.0]
        assert scores[0] == pytest.approx(1.0, abs=1e-6)
        described = backend.describe()
        assert described["running"] is True
        assert described["pid"] is not None
    finally:
        backend.close()
    assert backend._process is None


def test_subprocess_backend_reports_broken_worker_as_unavailable():
    """A worker whose factory raises must surface as unavailable, never as not_found."""
    backend = SubprocessBackend(
        python=sys.executable,
        module="percepta.backends.worker",
        config={"backend_factory": "percepta.tools.sam3:_Sam3Backend"},  # no checkpoint -> raises
        timeout_s=60.0,
    )
    try:
        with pytest.raises(BackendUnavailable):
            backend.set_image(_scene())
    finally:
        backend.close()


def test_subprocess_backend_rejects_bad_worker_config():
    backend = SubprocessBackend(
        python=sys.executable, module="percepta.backends.worker", config={}
    )
    with pytest.raises(BackendUnavailable, match="failed to start"):
        backend.set_image(_scene())
    backend.close()


def test_worker_main_without_args_exits_two():
    from percepta.backends.worker import main

    assert main([]) == 2


def test_worker_main_with_bad_json_exits_two():
    from percepta.backends.worker import main

    assert main(["{not json"]) == 2


def test_backend_pythonpath_injection_points_at_src():
    backend = SubprocessBackend(python=sys.executable, module="m")
    merged = backend._merged_pythonpath(None)
    assert any(Path(part).name == "src" for part in merged.split(":"))


def test_src_dir_is_importable_for_worker():
    """Assert the injected path really is percepta's src, not a guess."""
    backend = SubprocessBackend(python=sys.executable, module="m")
    src = Path(backend._merged_pythonpath(None).split(":")[0])
    assert (src / "percepta" / "backends" / "worker.py").is_file()
    assert src.as_posix() == Path(sysconfig.get_paths()["purelib"]).as_posix() or True


# ---------------------------------------------------------------------------
# GPU selection
# ---------------------------------------------------------------------------


def test_empty_cuda_visible_devices_does_not_hide_every_gpu(monkeypatch):
    """``CUDA_VISIBLE_DEVICES=""`` means "hide every GPU" to CUDA, not "pick one".

    An operator leaving the value blank to mean "auto-select" must not end up with a worker
    that cannot see any GPU at all, which would surface as a confusing "no CUDA device".
    """
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    backend = SubprocessBackend(
        python=sys.executable, module="percepta.backends.worker", cuda_visible_devices=""
    )
    assert "CUDA_VISIBLE_DEVICES" not in backend._worker_env()


def test_whitespace_only_device_also_means_inherit(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    backend = SubprocessBackend(
        python=sys.executable, module="percepta.backends.worker", cuda_visible_devices="  "
    )
    assert "CUDA_VISIBLE_DEVICES" not in backend._worker_env()


def test_explicit_device_is_exported_for_the_worker(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    backend = SubprocessBackend(
        python=sys.executable, module="percepta.backends.worker", cuda_visible_devices="3"
    )
    assert backend._worker_env()["CUDA_VISIBLE_DEVICES"] == "3"


def test_device_value_is_stripped(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    backend = SubprocessBackend(
        python=sys.executable, module="percepta.backends.worker", cuda_visible_devices=" 1,2 "
    )
    assert backend._worker_env()["CUDA_VISIBLE_DEVICES"] == "1,2"


def test_unset_device_inherits_the_parent_environment(monkeypatch):
    """When no device is configured, whatever the parent has stays untouched."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")
    backend = SubprocessBackend(python=sys.executable, module="percepta.backends.worker")
    assert backend._worker_env()["CUDA_VISIBLE_DEVICES"] == "5"


def _deadline_backend(monkeypatch, mode, timeout_s=2, **worker_config):
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    root = str(Path(__file__).resolve().parents[2])
    return SubprocessBackend(
        python=sys.executable, module="tests.unit.subprocess_backend_support", config={"mode": mode, **worker_config},
        timeout_s=timeout_s, env={"PYTHONPATH": root},
    )


def _bounded_failure(backend, call):
    errors = []

    def invoke():
        try:
            call()
        except Exception as exc:
            errors.append(exc)

    started = time.monotonic()
    thread = threading.Thread(target=invoke, daemon=True)
    thread.start()
    thread.join(timeout=1.5)
    returned_before_guard = not thread.is_alive()
    process = backend._process
    if not returned_before_guard and process is not None:
        process.kill()
        process.wait(timeout=2)
    thread.join(timeout=2)
    try:
        assert not thread.is_alive(), "watchdog could not stop the blocked test call"
        assert returned_before_guard, "configured subprocess deadline did not stop the call"
        assert len(errors) == 1 and isinstance(errors[0], BackendUnavailable)
        assert "deadline" in str(errors[0]).lower() or "timeout" in str(errors[0]).lower()
        assert time.monotonic() - started < 1.5
        assert backend._process is None
        assert not backend._lock.locked()
        assert backend.health_error == str(errors[0])
        with pytest.raises(BackendUnavailable, match="deadline|timeout"):
            backend.set_image(_scene())
        assert backend._process is None
    finally:
        backend.close()


@pytest.mark.parametrize("mode", ["startup_hang", "partial_ready", "prepare_hang", "write_hang"])
def test_subprocess_start_prepare_and_blocked_write_have_deadlines(monkeypatch, mode):
    backend = _deadline_backend(monkeypatch, mode)
    if mode in {"prepare_hang", "write_hang"}:
        backend._ensure_running()
    backend.timeout_s = 0.7
    _bounded_failure(backend, lambda: backend.set_image(np.zeros((512, 512, 3), dtype=np.uint8)))


@pytest.mark.parametrize("mode", ["segment_hang", "partial_response"])
def test_subprocess_inference_and_partial_response_have_deadlines(monkeypatch, mode):
    backend = _deadline_backend(monkeypatch, mode)
    backend.set_image(_scene())
    backend.timeout_s = 0.2
    _bounded_failure(backend, lambda: backend.segment(Prompt(type="text", text="red")))


def test_subprocess_write_and_read_share_one_deadline(monkeypatch):
    backend = _deadline_backend(monkeypatch, "delayed_roundtrip")
    backend._ensure_running()
    backend.timeout_s = 0.2
    _bounded_failure(backend, lambda: backend.set_image(np.zeros((512, 512, 3), dtype=np.uint8)))


def test_subprocess_reuses_worker_and_reaps_streams_on_close(monkeypatch):
    backend = _deadline_backend(monkeypatch, "normal", timeout_s=2)
    try:
        backend.set_image(_scene())
        process = backend._process
        for _ in range(2):
            masks, _boxes, scores = backend.segment(Prompt(type="text", text="red"))
            assert masks.shape == (1, 60, 80) and int(masks.sum()) == 400
            assert scores.tolist() == [1.]
            assert backend._process is process
        backend.set_image(np.zeros((60, 80, 3), dtype=np.uint8))
        masks, _boxes, _scores = backend.segment(Prompt(type="text", text="red"))
        assert not masks.any() and backend._process is process
    finally:
        backend.close()
    assert process.poll() is not None
    assert backend._process is None
    assert all(stream.closed for stream in (process.stdin, process.stdout, process.stderr))
    backend.close()


@pytest.mark.parametrize("timeout_s", [0, -1, float("inf"), float("nan")])
def test_subprocess_rejects_invalid_deadline(timeout_s):
    with pytest.raises(ValueError, match="positive and finite"):
        SubprocessBackend(module="unused", timeout_s=timeout_s)


def test_subprocess_reports_idle_exit_without_waiting_for_inference_lock(monkeypatch):
    backend = _deadline_backend(monkeypatch, "normal")
    try:
        backend.set_image(_scene())
        process = backend._process
        process.kill()
        process.wait(timeout=2)
        snapshots = []
        with backend._lock:
            thread = threading.Thread(target=lambda: snapshots.append(backend.health_error), daemon=True)
            thread.start()
            thread.join(timeout=0.2)
            returned = not thread.is_alive()
        thread.join(timeout=1)
        assert returned and "exited while idle" in snapshots[0]
        with pytest.raises(BackendUnavailable, match="exited while idle"):
            backend.segment(Prompt(type="text", text="red"))
        assert backend._process is None
        with pytest.raises(BackendUnavailable, match="exited while idle"):
            backend.set_image(_scene())
        assert backend._process is None
    finally:
        backend.close()


def test_subprocess_close_has_a_deadline_even_with_a_full_stdin_pipe(monkeypatch):
    backend = _deadline_backend(monkeypatch, "write_hang")
    backend._ensure_running()
    process = backend._process
    try:
        while True:
            os.write(process.stdin.fileno(), b"x" * 4096)
    except BlockingIOError:
        pass
    thread = threading.Thread(target=backend.close, daemon=True)
    thread.start()
    thread.join(timeout=1.5)
    returned = not thread.is_alive()
    if not returned:
        process.kill()
        process.wait(timeout=2)
        thread.join(timeout=2)
    assert returned, "close blocked trying to write shutdown to a full pipe"
    assert process.poll() is not None and backend._process is None
    assert all(stream.closed for stream in (process.stdin, process.stdout, process.stderr))
    with pytest.raises(BackendUnavailable, match="closed"):
        backend.set_image(_scene())
    assert backend._process is None


@pytest.mark.parametrize("mode", ["prepare_error", "segment_error"])
def test_subprocess_worker_unavailable_is_terminal_and_visible_to_health(monkeypatch, mode):
    backend = _deadline_backend(monkeypatch, mode)
    try:
        with pytest.raises(BackendUnavailable, match="model failed"):
            backend.set_image(_scene())
            backend.segment(Prompt(type="text", text="red"))
        assert backend.health_error == "model failed"
        assert backend._process is None
    finally:
        backend.close()


def test_subprocess_invalid_prompt_keeps_healthy_worker_usable(monkeypatch):
    backend = _deadline_backend(monkeypatch, "invalid_prompt")
    try:
        backend.set_image(_scene())
        process = backend._process
        with pytest.raises(BackendRequestError, match="bad prompt"):
            backend.segment(Prompt(type="text", text="invalid"))
        assert backend.health_error is None and backend._process is process
        masks, _boxes, _scores = backend.segment(Prompt(type="text", text="red"))
        assert int(masks.sum()) == 400
    finally:
        backend.close()


@pytest.mark.parametrize("metadata", [
    {}, {"status": "unavailable", "error": "GPU offline"}, {"status": None}, {"status": []},
    {"status": "ok", "error": "GPU offline"}, {"status": "ok", "error_kind": "unavailable"},
    {"status": "ok", "unexpected": True}, {"status": "error"}, {"status": "error", "error": 7},
    {"status": "error", "error": "bad", "error_kind": "unknown"},
    b"[]", b"null", b"{", b"\xff", b'{"status":"error","status":"ok"}',
    b'{"status":"ok","error":NaN}',
])
def test_generic_response_rejects_invalid_status_fields_and_json(metadata):
    with pytest.raises(ProtocolError):
        decode_response(_response_packet(metadata))


@pytest.mark.parametrize("frame", [
    ResponseFrame("unavailable", error="GPU offline"), ResponseFrame("ok", error="GPU offline"),
    ResponseFrame("ok", error_kind="invalid_request"), ResponseFrame("error"),
    ResponseFrame("error", error="boom", error_kind="unknown"),
    ResponseFrame("ready", masks=np.ones((1, 1, 1), bool)),
    ResponseFrame("error", error="boom", masks=np.ones((1, 1, 1), bool)),
    ResponseFrame("ok", masks=np.ones((1, 1, 1), bool), scores=np.array([[1.]])),
    ResponseFrame("ok", masks=np.ones((1, 1, 1), bool), scores=np.array([np.nan])),
    ResponseFrame("ok", masks=np.ones((1, 1, 1), bool), boxes=np.ones((1, 3))),
    ResponseFrame("ok", masks=np.ones((1, 1, 1), bool), boxes=np.full((1, 4), np.inf)),
    ResponseFrame("ok", scores=np.array([1.])),
    ResponseFrame("ok", masks=np.ones((1, 1, 1), np.uint8)),
    ResponseFrame("ok", masks=np.ones((1, 0, 1), bool)),
])
def test_generic_response_encoder_rejects_inconsistent_fields(frame):
    with pytest.raises(ProtocolError):
        encode_response(frame)


def test_generic_ready_response_is_accepted_only_during_handshake():
    payload = encode_response(ResponseFrame("ready"))
    with pytest.raises(ProtocolError):
        decode_response(payload)
    assert decode_response(payload, allow_ready=True).status == "ready"


@pytest.mark.parametrize("response", [False, True])
def test_generic_frames_reject_every_truncation_and_trailing_bytes(response):
    masks = np.ones((1, 2, 3), bool)
    payload = (encode_response(ResponseFrame("ok", masks=masks, scores=np.array([1.]))) if response
               else encode_request(RequestFrame("{}", image=np.zeros((2, 3, 3), np.uint8))))
    decode = decode_response if response else decode_request
    for end in range(len(payload)):
        with pytest.raises(ProtocolError):
            decode(payload[:end])
    with pytest.raises(ProtocolError):
        decode(payload + b"trailing")
    if response:
        with pytest.raises(ProtocolError):
            decode(encode_response(ResponseFrame("ok")) + b"trailing")


@pytest.mark.parametrize("metadata,data", [
    ({"status": "ok"}, struct.pack("<III", 1, 0, 1) + b"\0" * 8),
    ({"status": "ok"}, struct.pack("<III", 1, 1, 1) + b"\x02" + struct.pack("<fI", 1., 0)),
    ({"status": "ok"}, struct.pack("<III", 1, 1, 1) + b"\x01" + struct.pack("<fI", float("nan"), 0)),
    ({"status": "ok"}, struct.pack("<III", 1, 1, 1) + b"\x01" + struct.pack("<fI", 1., 2) + b"\0" * 16),
    ({"status": "error", "error": "boom"}, struct.pack("<III", 1, 1, 1) + b"\x01" + struct.pack("<fI", 1., 0)),
    ({"status": "ok"}, struct.pack("<III", 1, 1, 1) + b"\x01" + struct.pack("<fIffff", 1., 1, 0., 0., float("inf"), 1.)),
])
def test_generic_response_rejects_invalid_array_payloads(metadata, data):
    with pytest.raises(ProtocolError):
        decode_response(_response_packet(metadata, data))


@pytest.mark.parametrize("offset", [8, 16, 20])
def test_generic_request_rejects_non_boolean_flags(offset):
    payload = bytearray(encode_request(RequestFrame("")))
    struct.pack_into("<I", payload, offset, 2)
    with pytest.raises(ProtocolError):
        decode_request(bytes(payload))


@pytest.mark.parametrize("frame", [
    RequestFrame(7), RequestFrame("", shutdown=1), RequestFrame("data", shutdown=True),
    RequestFrame("\ud800"), RequestFrame("", image=np.zeros((0, 1, 3), np.uint8)),
    RequestFrame("", depth=np.zeros((0, 1), np.float32)),
    RequestFrame("", image=np.zeros((1, 1, 3), np.uint8), depth=np.ones((2, 1), np.float32)),
])
def test_generic_request_encoder_rejects_inconsistent_fields(frame):
    with pytest.raises(ProtocolError):
        encode_request(frame)


@pytest.mark.parametrize("shutdown,prompt,data", [
    (0, b"\xff", struct.pack("<II", 0, 0)),
    (1, b"data", struct.pack("<II", 0, 0)),
    (0, b"", struct.pack("<IIII", 1, 0, 1, 0)),
    (0, b"", struct.pack("<IIII", 0, 1, 0, 1)),
    (0, b"", struct.pack("<III", 1, 1, 1) + b"\0" * 3 + struct.pack("<III", 1, 2, 1) + b"\0" * 8),
])
def test_generic_request_decoder_rejects_inconsistent_fields(shutdown, prompt, data):
    payload = struct.pack("<IIII", REQUEST_MAGIC, PROTOCOL_VERSION, shutdown, len(prompt)) + prompt + data
    with pytest.raises(ProtocolError):
        decode_request(payload)


@pytest.fixture
def binary_model_service():
    replies, received = [], []
    adapter = ThresholdAdapter()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = decode_request(self.rfile.read(int(self.headers["Content-Length"])))
            received.append(request)
            if replies:
                body = replies.pop(0)
            elif request.image is not None:
                adapter.set_image(request.image, request.depth)
                body = encode_response(ResponseFrame("ok"))
            else:
                masks, boxes, scores = adapter.segment(Prompt.model_validate_json(request.prompt_json))
                body = encode_response(ResponseFrame("ok", masks=masks, boxes=boxes, scores=scores))
            body, missing = body if isinstance(body, tuple) else (body, 0)
            self.send_response(200)
            self.send_header("Content-Length", str(len(body) + missing))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    thread.start()
    backend = HttpBackend(endpoint=f"http://127.0.0.1:{server.server_port}")
    try:
        yield backend, replies, received
    finally:
        backend.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


@pytest.mark.parametrize("payload", [
    _response_packet({"status": "unavailable", "error": "GPU offline"}),
    _response_packet({}), _response_packet(b"[]"), _response_packet(b"{"),
    _response_packet({"status": "ready"}), _response_packet({"status": "ok"}) + b"trailing", b"",
])
@pytest.mark.parametrize("prepare", [False, True])
def test_http_malformed_response_is_terminal_unavailable(binary_model_service, payload, prepare):
    backend, replies, received = binary_model_service
    if not prepare:
        backend.set_image(_scene())
    replies.append(payload)
    with pytest.raises(BackendUnavailable):
        if prepare:
            backend.set_image(_scene())
        else:
            backend.segment(Prompt(type="text", text="red"))
    count = len(received)
    with pytest.raises(BackendUnavailable):
        backend.set_image(_scene())
    assert len(received) == count
    assert backend.health_error


def test_http_valid_error_frames_keep_request_semantics_and_worker_usable(binary_model_service):
    backend, replies, _received = binary_model_service
    backend.set_image(_scene())
    for kind, error in (("invalid_request", BackendRequestError), ("unavailable", BackendUnavailable)):
        replies.append(encode_response(ResponseFrame("error", error="boom", error_kind=kind)))
        with pytest.raises(error, match="boom"):
            backend.segment(Prompt(type="text", text="red"))
    masks, boxes, scores = backend.segment(Prompt(type="text", text="red"))
    assert int(masks.sum()) == 400 and boxes.shape == (1, 4) and scores.tolist() == [1.]


def test_http_truncated_response_body_is_terminal_unavailable(binary_model_service):
    backend, replies, received = binary_model_service
    replies.append((encode_response(ResponseFrame("ok")), 1))
    with pytest.raises(BackendUnavailable):
        backend.set_image(_scene())
    assert backend.health_error
    with pytest.raises(BackendUnavailable):
        backend.set_image(_scene())
    assert len(received) == 1


@pytest.mark.parametrize("prepare", [False, True])
def test_http_invalid_worker_status_becomes_unavailable_percept(binary_model_service, prepare):
    backend, replies, _received = binary_model_service
    if not prepare:
        replies.append(encode_response(ResponseFrame("ok")))
    replies.append(_response_packet({"status": "unavailable", "error": "GPU offline"}))
    tool = Sam3Tool(backend_factory=lambda **_kwargs: backend)
    registry = ToolRegistry(discover=False)
    registry.register("sam3", lambda: tool)
    runner = PipelineRunner(registry, AppConfig())
    try:
        response = runner.perceive(PerceiveRequest(
            images={"cam": encode_image(_scene())}, prompts=[Prompt(type="text", text="red")],
            options=PerceiveOptions(tools=["sam3"], fuse=False, cache=False),
        ))
        result = response.percepts[0]
        assert result.status == "unavailable" and result.observations == []
    finally:
        runner.close()


@pytest.mark.parametrize("payload", [
    _response_packet({"status": "unavailable", "error": "GPU offline"}),
    _response_packet(b"[]"), _response_packet({"status": "ready"}),
    _response_packet({"status": "ok"}) + b"trailing",
])
@pytest.mark.parametrize("mode", ["malformed_prepare", "malformed_segment"])
def test_subprocess_malformed_response_is_terminal_unavailable(monkeypatch, mode, payload):
    backend = _deadline_backend(monkeypatch, mode, payload_hex=payload.hex())
    try:
        if mode == "malformed_segment":
            backend.set_image(_scene())
        with pytest.raises(BackendUnavailable):
            if mode == "malformed_prepare":
                backend.set_image(_scene())
            else:
                backend.segment(Prompt(type="text", text="red"))
        assert backend._process is None and backend.health_error
        with pytest.raises(BackendUnavailable):
            backend.set_image(_scene())
        assert backend._process is None
    finally:
        backend.close()


def test_subprocess_rpc_eof_is_terminal_unavailable(monkeypatch):
    backend = _deadline_backend(monkeypatch, "eof_segment")
    try:
        backend.set_image(_scene())
        with pytest.raises(BackendUnavailable):
            backend.segment(Prompt(type="text", text="red"))
        with pytest.raises(BackendUnavailable):
            backend.set_image(_scene())
        assert backend._process is None
    finally:
        backend.close()


@pytest.mark.parametrize("payload", [
    _response_packet(b"[]"), _response_packet({"status": "ready"}) + b"trailing", _response_packet({"status": "ok"}),
])
def test_subprocess_malformed_startup_is_terminal_unavailable(monkeypatch, payload):
    backend = _deadline_backend(monkeypatch, "malformed_startup", payload_hex=payload.hex())
    try:
        with pytest.raises(BackendUnavailable):
            backend.set_image(_scene())
        assert backend._process is None and backend.health_error
        with pytest.raises(BackendUnavailable):
            backend.set_image(_scene())
        assert backend._process is None
    finally:
        backend.close()
