"""Subprocess backend: run a model in its own virtualenv, talk over a binary pipe.

This is the answer to dependency conflicts without containers (``docs/model-isolation.md``).
Two models that need incompatible ``torch``/``numpy`` versions each get a virtualenv; the
parent process only needs numpy (a core dependency) to build frames and read results back.

Lifecycle: the worker is spawned on first use, loads the model once, then stays resident and
answers requests. Model load is the expensive part (seconds); per-request IPC is milliseconds.

Failure mapping follows spec §16.4 — a dead/unavailable worker is ``unavailable``, never
``not_found``.
"""

from __future__ import annotations

import json
import logging
import math
import os
import select
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..schema import Prompt
from .base import BackendRequestError, BackendUnavailable
from .foundation_stereo_backend import _DeadlinePipe
from .protocol import (
    ProtocolError,
    RequestFrame,
    ResponseFrame,
    decode_response,
    encode_request,
    read_frame,
    write_frame,
)

__all__ = ["SubprocessBackend"]

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 120.0


class SubprocessBackend:
    """Run a resident model in another interpreter with bounded pipe operations.

    Args:
        python: Worker interpreter path or executable name.
        module: Importable module implementing the framed worker protocol.
        config: JSON configuration passed to the worker at startup.
        timeout_s: Positive finite deadline for startup and each complete write/read RPC.
        cuda_visible_devices: GPU selection; empty values inherit the parent environment.
        env: Additional worker environment variables.

    Example:
        >>> backend = SubprocessBackend(module="percepta.backends.worker")
        >>> backend.describe()["running"]
        False
        >>> backend.close()
    """

    name = "subprocess"

    def __init__(
        self,
        *,
        python: str = "python3",
        module: str,
        config: dict[str, Any] | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        cuda_visible_devices: str | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        self.python = str(python)
        self.module = str(module)
        self.worker_config = dict(config or {})
        self.timeout_s = float(timeout_s)
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise ValueError("timeout_s must be positive and finite")
        self.cuda_visible_devices = cuda_visible_devices
        self.extra_env = dict(env or {})
        self._process: subprocess.Popen[bytes] | None = None
        self._lock = threading.Lock()
        self._frame_prepared = False
        self._stderr_tail = bytearray()
        self._stderr_stop = threading.Event()
        self._stderr_thread: threading.Thread | None = None
        self._failure: str | None = None
        self._closed = False

    # -- lifecycle -------------------------------------------------------

    def _resolve_python(self) -> str:
        """Resolve the interpreter, failing with an actionable message (not a traceback)."""
        candidate = Path(self.python)
        if candidate.is_absolute() or os.sep in self.python:
            if not candidate.exists():
                raise BackendUnavailable(
                    f"worker interpreter not found: {self.python} "
                    f"(create it with `uv venv {candidate.parent} -p 3.10`)"
                )
            return str(candidate)
        found = shutil.which(self.python)
        if found:
            return found
        raise BackendUnavailable(f"worker interpreter not found on PATH: {self.python}")

    def _worker_env(self) -> dict[str, str]:
        """Build the worker's environment. Extracted so the GPU rule is directly testable."""
        env = dict(os.environ)
        env.update(self.extra_env)
        # Respect the caller's environment unless an explicit device was configured; the
        # "one worker per GPU" rule from docs/model-isolation.md §7 is enforced here.
        #
        # NOTE: an EMPTY string is not a device list. ``CUDA_VISIBLE_DEVICES=""`` means "hide
        # every GPU" to CUDA, so setting it would silently disable the model instead of
        # letting it choose. Empty therefore means "inherit the parent's environment".
        device = None if self.cuda_visible_devices is None else str(self.cuda_visible_devices).strip()
        if device:
            env["CUDA_VISIBLE_DEVICES"] = device
        return env

    def _spawn(self) -> None:
        deadline = time.monotonic() + self.timeout_s
        python = self._resolve_python()
        env = self._worker_env()
        env.setdefault("PYTHONUNBUFFERED", "1")
        # The worker imports numpy from *its own* env; make sure our src/ is importable so
        # the shared protocol module can be reused when the worker is percepta's own.
        env["PYTHONPATH"] = self._merged_pythonpath(env.get("PYTHONPATH"))

        # NOTE: `-u` must come *before* `-m`; `python -u -m mod` is valid but
        # `python -m -u mod` is not, and getting this wrong makes Python treat the module
        # name as a script path ("can't open file 'percepta.backends.worker'").
        argv = [python, "-u", "-m", self.module, json.dumps(self.worker_config)]
        log.debug("spawning model worker: %s", argv)
        try:
            self._process = subprocess.Popen(  # noqa: S603 - argv is config-controlled, no shell
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                bufsize=0,
            )
        except OSError as exc:
            raise BackendUnavailable(f"failed to start worker {self.module}: {exc}") from exc
        self._frame_prepared = False
        try:
            for stream in (self._process.stdin, self._process.stdout, self._process.stderr):
                os.set_blocking(stream.fileno(), False)
            self._drain_stderr_async()
            self._await_ready(deadline)
        except (OSError, ValueError) as exc:
            raise self._fail(f"failed to start worker {self.module}: {exc}") from exc

    def _merged_pythonpath(self, existing: str | None) -> str:
        parts = [str(Path(__file__).resolve().parents[2])]  # .../src
        if existing:
            parts.append(existing)
        return os.pathsep.join(parts)

    def _drain_stderr_async(self) -> None:
        """Keep stderr drained so a chatty worker can never block on a full pipe."""
        process = self._process
        if process is None or process.stderr is None:
            return
        fd = process.stderr.fileno()

        def pump() -> None:
            try:
                while not self._stderr_stop.is_set():
                    readable, _, _ = select.select([fd], [], [], 0.05)
                    if not readable:
                        continue
                    try:
                        chunk = os.read(fd, 4096)
                    except BlockingIOError:
                        continue
                    if not chunk:
                        return
                    self._stderr_tail.extend(chunk)
                    del self._stderr_tail[:-16 * 1024]
                    log.debug("model worker: %s", chunk.decode("utf-8", "replace").rstrip())
            except (OSError, ValueError):
                return  # pipe closed during teardown

        self._stderr_thread = threading.Thread(target=pump, name=f"worker-stderr-{self.module}", daemon=True)
        self._stderr_thread.start()

    def _await_ready(self, deadline: float) -> None:
        """The worker's first frame is a status frame with ``status="ready"``."""
        response = self._read_frame_or_fail("worker exited before signalling readiness", deadline, allow_ready=True)
        if response.status != "ready":
            detail = response.error or (self._stderr_tail_tail() or "unknown error")
            raise self._fail(f"model worker failed to start: {detail}")

    def _stderr_tail_tail(self) -> str:
        return " | ".join(self._stderr_tail.decode("utf-8", "replace").splitlines()[-5:])

    def _read_frame_or_fail(self, message: str, deadline: float, *, allow_ready: bool = False) -> ResponseFrame:
        process = self._process
        if process is None or process.stdout is None:
            raise self._fail(message)
        try:
            payload = read_frame(_DeadlinePipe(process.stdout.fileno(), deadline, limit=512 * 1024 * 1024))
            if payload is not None:
                response = decode_response(payload, allow_ready=allow_ready)
                if time.monotonic() >= deadline:
                    raise TimeoutError("worker deadline exceeded")
                return response
        except TimeoutError as exc:
            raise self._fail(f"{message}: worker deadline exceeded") from exc
        except (ProtocolError, OSError, ValueError) as exc:
            raise self._fail(f"{message}: {exc}") from exc
        if payload is None:
            code = process.poll()
            raise self._fail(f"{message} (exit code {code})")

    def _request(self, frame: RequestFrame) -> ResponseFrame:
        """Send one frame and read exactly one response frame."""
        process = self._process
        if process is None or process.stdin is None:
            raise self._fail("model worker is not running")
        deadline = time.monotonic() + self.timeout_s
        try:
            write_frame(_DeadlinePipe(process.stdin.fileno(), deadline), encode_request(frame))
        except TimeoutError as exc:
            raise self._fail("model worker write failed: worker deadline exceeded") from exc
        except (OSError, ValueError) as exc:
            raise self._fail(f"model worker write failed: {exc}") from exc
        return self._read_frame_or_fail("model worker produced no response", deadline)

    # -- ModelBackend ----------------------------------------------------

    def set_image(self, image_rgb: np.ndarray, depth: np.ndarray | None = None) -> None:
        """Prepare RGB/depth without running a prompt.

        Args:
            image_rgb: uint8 RGB array of shape (H, W, 3).
            depth: Optional float32 camera depth in metres, shape (H, W).

        Raises:
            BackendUnavailable: The worker fails or exceeds its deadline.

        Example:
            >>> backend.set_image(np.zeros((60, 80, 3), dtype=np.uint8))  # prepare a black frame
        """
        with self._lock:
            self._ensure_running()
            self._frame_prepared = False
            response = self._request(
                RequestFrame(prompt_json="", image=image_rgb, depth=depth)
            )
            if response.status == "error":
                raise self._map_error(response)
            self._frame_prepared = True

    def segment(self, prompt: Prompt) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
        """Run one prompt against a previously prepared frame.

        Args:
            prompt: Text or geometric prompt supported by the worker.

        Returns:
            Boolean masks (N, H, W), optional pixel boxes (N, 4), and scores (N,).

        Raises:
            BackendRequestError: The worker rejects the prompt.
            BackendUnavailable: The worker fails or exceeds its deadline.

        Example:
            >>> masks, boxes, scores = backend.segment(Prompt(type="text", text="red block"))
            >>> masks.dtype == bool
            True
        """
        with self._lock:
            self._ensure_running()
            response = self._request(
                RequestFrame(prompt_json=prompt.model_dump_json(), image=None)
            )
            if response.status == "error":
                raise self._map_error(response)
            if response.masks is None:
                return (
                    np.zeros((0, 1, 1), dtype=bool),
                    None,
                    np.zeros(0, dtype=float),
                )
            return response.masks, response.boxes, response.scores

    def _map_error(self, response: ResponseFrame) -> Exception:
        message = response.error or "model worker reported an error"
        if response.error_kind == "invalid_request":
            return BackendRequestError(message)
        return self._fail(message)

    def _ensure_running(self) -> None:
        """Start once; transport failures are terminal so stale image state cannot restart."""
        if self._failure or self._closed:
            raise BackendUnavailable(self._failure or "model worker is closed")
        dead = self._process is not None and self._process.poll() is not None
        if dead:
            code = self._process.poll() if self._process is not None else None
            raise self._fail(f"model worker exited while idle (exit code {code})")
        if self._process is None:
            try:
                self._spawn()
            except BackendUnavailable as exc:
                if self._failure is None:
                    raise self._fail(str(exc)) from exc
                raise

    def _fail(self, message: str) -> BackendUnavailable:
        self._failure = message
        try:
            self._dispose(graceful=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            self._failure += f"; worker cleanup failed: {exc}"
        detail = self._stderr_tail_tail()
        if detail:
            self._failure += f": {detail}"
        return BackendUnavailable(self._failure)

    def _dispose(self, *, graceful: bool) -> None:
        process, self._process = self._process, None
        self._frame_prepared = False
        if process is None:
            return
        try:
            if graceful and process.stdin is not None and process.poll() is None:
                try:
                    write_frame(
                        _DeadlinePipe(process.stdin.fileno(), time.monotonic() + 0.2),
                        encode_request(RequestFrame(prompt_json="", shutdown=True)),
                    )
                    process.wait(timeout=0.2)
                except (OSError, ValueError, subprocess.TimeoutExpired):
                    pass
            if process.poll() is None:
                process.kill()
            process.wait(timeout=1)
        finally:
            if process.poll() is None:  # retain ownership if a kernel-stuck child cannot yet be reaped
                self._process = process
            self._stderr_stop.set()
            if self._stderr_thread is not None:
                self._stderr_thread.join(timeout=0.2)
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    stream.close()

    @property
    def health_error(self) -> str | None:
        """Read worker failure without waiting for an in-flight inference lock.

        Returns:
            Failure or exit detail, or None before startup and while healthy.

        Example:
            >>> SubprocessBackend(module="percepta.backends.worker").health_error is None
            True
        """
        if self._failure:
            return self._failure
        if self._closed:
            return "model worker is closed"
        process = self._process
        if process is not None and process.poll() is not None:
            return f"model worker exited while idle (exit code {process.returncode})"
        return None

    def close(self) -> None:
        """Stop and reap the worker within bounded pipe/write/wait operations.

        Example:
            >>> backend.close()
            >>> backend.describe()["running"]
            False
        """
        with self._lock:
            self._closed = True
            self._dispose(graceful=True)

    def describe(self) -> dict[str, Any]:
        """Describe interpreter configuration and the current worker process.

        Returns:
            Backend name, interpreter, module, PID, GPU selection, and running flag.

        Example:
            >>> SubprocessBackend(module="percepta.backends.worker").describe()["pid"] is None
            True
        """
        process = self._process
        return {
            "backend": self.name,
            "python": self.python,
            "module": self.module,
            "pid": None if process is None else process.pid,
            "cuda_visible_devices": self.cuda_visible_devices,
            "running": process is not None and process.poll() is None,
        }

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        try:
            self.close()
        except Exception:  # noqa: BLE001 - never raise from __del__
            pass


def _python_for_venv(venv_dir: str | Path) -> str:
    """Best-effort interpreter path inside a virtualenv (used by configs and tests)."""
    venv = Path(venv_dir)
    candidate = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    return str(candidate)


_ = sys  # kept for symmetry with the worker module's imports
