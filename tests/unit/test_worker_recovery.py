"""Real CPU worker recovery and pipe IO in hosts with high-numbered file descriptors."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from percepta.backends.base import BackendRequestError
from percepta.backends.foundation_stereo_backend import STDERR_LIMIT_BYTES, FoundationStereoBackend
from percepta.backends.multiview_backend import MultiViewBackend
from percepta.backends.reference import ThresholdAdapter
from percepta.backends.subprocess_backend import SubprocessBackend
from percepta.schema import Prompt, ToolContext
from percepta.tools.sam3 import Sam3BackendError, Sam3Tool
from tests.unit.multiview_support import FakeMultiViewAdapter

ROOT = Path(__file__).resolve().parents[2]
MARKER = b"high-fd-stderr-drained"


class _RecoveryAdapter(ThresholdAdapter):
    def __init__(self, error="backend", chatty=False):
        super().__init__()
        self.error = error
        if chatty:
            os.write(2, b"x" * (512 * 1024) + MARKER)

    def segment(self, prompt):
        if prompt.text == "invalid":
            error = BackendRequestError if self.error == "backend" else Sam3BackendError
            raise error("unsupported test prompt")
        return super().segment(prompt)


class _ChattyMultiViewAdapter(FakeMultiViewAdapter):
    def __init__(self):
        super().__init__()
        os.write(2, b"x" * (512 * 1024) + MARKER)


@pytest.mark.parametrize("error", ["backend", "sam3"])
@pytest.mark.parametrize("transport", ["inprocess", "subprocess"])
def test_request_error_keeps_worker_and_sam3_status_consistent(error, transport):
    if transport == "inprocess":
        backend = _RecoveryAdapter(error=error)
    else:
        backend = SubprocessBackend(
            python=sys.executable, module="percepta.backends.worker",
            config={"backend_factory": "tests.unit.test_worker_recovery:_RecoveryAdapter", "backend_args": {"error": error}},
            timeout_s=3, env={"PYTHONPATH": str(ROOT), "OPENBLAS_NUM_THREADS": "1"},
        )
    tool = Sam3Tool(backend_factory=lambda **_kwargs: backend)
    tool.load({})
    ctx = ToolContext(request_id="request-recovery", images={"cam": np.full((8, 8, 3), 255, np.uint8)})
    invalid, valid = Prompt(type="text", text="invalid"), Prompt(type="text", text="red")
    try:
        backend.set_image(ctx.images["cam"])
        process = getattr(backend, "_process", None)
        with pytest.raises(BackendRequestError, match="unsupported test prompt"):
            backend.segment(invalid)
        results = tool.perceive(ctx, [invalid, valid])
        assert [result.status for result in results] == ["invalid_request", "ok"]
        assert results[1].observations[0].area_px == 64
        assert tool.perceive(ctx, [valid])[0].status == "ok"
        if process is not None:
            assert backend._process is process and process.poll() is None
            assert backend.health_error is None
    finally:
        tool.close()


def _high_fd_probe(kind):
    opened = []
    backend = None
    try:
        while not opened or opened[-1] < 1050:
            opened.append(os.open(os.devnull, os.O_RDONLY))
        image = np.full((8, 8, 3), 255, np.uint8)
        if kind == "generic":
            backend = SubprocessBackend(
                python=sys.executable, module="percepta.backends.worker", timeout_s=5,
                config={"backend_factory": "tests.unit.test_worker_recovery:_RecoveryAdapter", "backend_args": {"chatty": True}},
            )
            backend.set_image(image)
            assert int(backend.segment(Prompt(type="text", text="red"))[0].sum()) == 64
            assert MARKER in backend._stderr_tail
            assert len(backend._stderr_tail) <= STDERR_LIMIT_BYTES
        elif kind == "stereo":
            backend = FoundationStereoBackend(
                python=sys.executable, backend_factory="tests.unit.foundation_stereo_support:FakeStereoAdapter",
                backend_args={"mode": "chatty"}, startup_timeout_s=5, timeout_s=3,
            )
            assert np.all(backend.infer_pair(image, image) == 0.25)
            assert b"fake inference" in backend._stderr_tail
            assert len(backend._stderr_tail) <= STDERR_LIMIT_BYTES
        else:
            backend = MultiViewBackend(
                name="cpu-test", python=sys.executable,
                backend_factory="tests.unit.test_worker_recovery:_ChattyMultiViewAdapter", backend_args={},
                startup_timeout_s=5, timeout_s=3,
            )
            images = np.repeat(image[None], 3, axis=0)
            intrinsics = np.tile([[20., 0, 4], [0, 20, 4], [0, 0, 1]], (3, 1, 1))
            transforms = np.tile(np.eye(4), (3, 1, 1))
            transforms[1, 0, 3], transforms[2, 1, 3] = 0.1, 0.2
            assert np.all(backend.infer(images, intrinsics, transforms).depth == 1.25)
            assert MARKER in backend._stderr_tail
        process = backend._process
        assert min(stream.fileno() for stream in (process.stdin, process.stdout, process.stderr)) > 1023
        backend.close()
        assert process.poll() is not None and backend._process is None
        assert all(stream.closed for stream in (process.stdin, process.stdout, process.stderr))
        assert backend._stderr_thread is None or not backend._stderr_thread.is_alive()
    finally:
        if backend is not None:
            backend.close()
        for fd in opened:
            os.close(fd)


@pytest.mark.parametrize("kind", ["generic", "stereo", "multiview"])
def test_high_fd_worker_transport_drains_stderr_and_closes(kind):
    import resource

    if resource.getrlimit(resource.RLIMIT_NOFILE)[0] < 1100:
        pytest.skip("requires a host fd limit above 1100")
    env = {**os.environ, "PYTHONPATH": str(ROOT), "OPENBLAS_NUM_THREADS": "1"}
    code = "from tests.unit.test_worker_recovery import _high_fd_probe; _high_fd_probe(" + repr(kind) + ")"
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
