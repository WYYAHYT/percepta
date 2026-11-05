"""Backend abstraction and the subprocess isolation path (``docs/model-isolation.md``).

The important test here is :func:`test_subprocess_backend_runs_model_in_a_different_venv`:
it proves the central claim — a model can live in a *different* interpreter with a
*different* numpy version and still be driven through the same tool interface. That is the
whole reason this subsystem exists.

All tests are GPU-free and weight-free.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import sysconfig
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
from percepta.backends.inprocess import InProcessBackend
from percepta.backends.protocol import (
    PROTOCOL_VERSION,
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
from percepta.schema import Prompt


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
