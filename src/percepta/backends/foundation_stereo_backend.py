"""Resident FoundationStereo subprocess; constructor values are trusted deployment config."""

from __future__ import annotations

import json
import math
import os
import select
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from .base import BackendRequestError, BackendUnavailable
from .stereo_protocol import (
    MAX_RESPONSE_BYTES,
    ProtocolError,
    StereoRequest,
    decode_response,
    encode_request,
    read_frame,
    write_frame,
)

__all__ = ["FoundationStereoBackend"]

STDERR_LIMIT_BYTES = 16 * 1024


class _DeadlinePipe:
    """Nonblocking pipe IO with one absolute deadline across every partial read/write."""

    def __init__(self, fd: int, deadline: float, limit: int = MAX_RESPONSE_BYTES):
        self.fd, self.deadline, self.limit = fd, deadline, limit

    def _wait(self, *, writing: bool = False) -> None:
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("stereo worker deadline exceeded")
        readable, writable, _ = select.select(
            [] if writing else [self.fd], [self.fd] if writing else [], [], remaining,
        )
        if not (readable or writable) or time.monotonic() >= self.deadline:
            raise TimeoutError("stereo worker deadline exceeded")

    def read(self, size: int) -> bytes:
        if size > self.limit:
            raise ProtocolError("stereo frame exceeds length limit")
        result = bytearray()
        while len(result) < size:
            self._wait()
            try:
                chunk = os.read(self.fd, min(size - len(result), 65536))
            except BlockingIOError:
                continue
            if not chunk:
                break
            result.extend(chunk)
        return bytes(result)

    def write(self, data: bytes) -> int:
        view = memoryview(data)
        offset = 0
        while offset < len(view):
            self._wait(writing=True)
            try:
                written = os.write(self.fd, view[offset:offset + 65536])
            except BlockingIOError:
                continue
            if not written:
                raise BrokenPipeError("stereo worker pipe closed")
            offset += written
        return offset

    def flush(self) -> None:
        pass  # os.write is unbuffered


class FoundationStereoBackend:
    """Rectified RGB uint8 HWC pairs -> float32 HW pixel disparity; failure is terminal."""

    name = "foundation_stereo"
    worker_module = "percepta.backends.foundation_stereo_worker"
    response_limit = MAX_RESPONSE_BYTES
    response_decoder = staticmethod(decode_response)

    @staticmethod
    def shutdown_payload() -> bytes:
        """Encode the worker's shutdown control message.

        Returns:
            A stereo protocol shutdown payload.

        Example:
            >>> bool(FoundationStereoBackend.shutdown_payload())
            True
        """
        return encode_request(StereoRequest(op="shutdown"))

    def __init__(
        self,
        *,
        python: str = "python3",
        source_dir: str | None = None,
        checkpoint: str | None = None,
        model_config: str | None = None,
        hf_cache: str | None = None,
        valid_iters: int = 32,
        cuda_visible_devices: str | None = None,
        startup_timeout_s: float = 180.0,
        timeout_s: float = 120.0,
        backend_factory: str | None = None,
        backend_args: dict[str, Any] | None = None,
    ) -> None:
        if type(valid_iters) is not int or valid_iters <= 0:
            raise BackendRequestError("valid_iters must be a positive integer")
        for value in (startup_timeout_s, timeout_s):
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise BackendRequestError("stereo timeouts must be positive and finite")
        self.python = str(python)
        self.hf_cache = str(Path(hf_cache).resolve()) if hf_cache else None
        self.cuda_visible_devices = cuda_visible_devices
        self.startup_timeout_s, self.timeout_s = float(startup_timeout_s), float(timeout_s)
        checkpoint = str(Path(checkpoint).resolve()) if checkpoint else None
        self.worker_config = {
            "source_dir": str(Path(source_dir).resolve()) if source_dir else None,
            "checkpoint": checkpoint,
            "model_config": str(Path(model_config).resolve()) if model_config else (
                str(Path(checkpoint).with_name("cfg.yaml")) if checkpoint else None
            ),
            "valid_iters": valid_iters,
            "backend_factory": backend_factory,
            "backend_args": dict(backend_args or {}),
        }
        self._lock = threading.Lock()
        self._stderr_lock = threading.Lock()
        self._stderr_stop = threading.Event()
        self._stderr_thread: threading.Thread | None = None
        self._stderr_tail = bytearray()
        self._process: subprocess.Popen | None = None
        self._state = "new"
        self._failure = ""
        self._info: dict[str, Any] = {}

    def _worker_env(self) -> dict[str, str]:
        env = dict(os.environ)
        src = str(Path(__file__).resolve().parents[2])
        env["PYTHONPATH"] = os.pathsep.join(filter(None, (src, env.get("PYTHONPATH"))))
        device = str(self.cuda_visible_devices).strip() if self.cuda_visible_devices is not None else ""
        if device:
            env["CUDA_VISIBLE_DEVICES"] = device
        env["HF_HUB_OFFLINE"] = "1"
        env["HF_DATASETS_OFFLINE"] = "1"
        env["HF_HUB_DISABLE_TELEMETRY"] = "1"
        if self.hf_cache:
            env["HF_HUB_CACHE"] = self.hf_cache
            env["HUGGINGFACE_HUB_CACHE"] = self.hf_cache
        return env

    def _drain_stderr(self, fd: int) -> None:
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
                with self._stderr_lock:
                    self._stderr_tail.extend(chunk)
                    del self._stderr_tail[:-STDERR_LIMIT_BYTES]
        except OSError:
            return  # pipe closed during teardown

    def _receive(self, deadline: float):
        payload = read_frame(_DeadlinePipe(self._process.stdout.fileno(), deadline, self.response_limit))
        if payload is None:
            raise ProtocolError("stereo worker EOF (exited without a response)")
        response = self.response_decoder(payload)
        if response.status == "error":
            raise BackendUnavailable(response.error)
        if time.monotonic() >= deadline:
            raise TimeoutError("stereo worker deadline exceeded")
        return response

    def _fail(self, exc: Exception) -> BackendUnavailable:
        self._state = "failed"
        try:
            self._dispose(graceful=False)
        except (OSError, subprocess.TimeoutExpired) as cleanup_error:
            # Even a teardown failure must preserve the public unavailable semantics.
            exc = RuntimeError(f"{exc}; worker cleanup failed: {cleanup_error}")
        with self._stderr_lock:
            detail = self._stderr_tail.decode("utf-8", "replace")
        self._failure = f"{self.name} unavailable: {exc}"
        if detail:
            self._failure += f"; stderr: {detail}"
        return BackendUnavailable(self._failure)

    def _start_locked(self) -> None:
        if self._state == "ready":
            if self._process.poll() is not None:
                raise self._fail(RuntimeError(f"worker exited with code {self._process.returncode}"))
            return
        if self._state != "new":
            raise BackendUnavailable(self._failure or "FoundationStereo backend is closed")
        deadline = time.monotonic() + self.startup_timeout_s
        self._state = "starting"
        try:
            argv = [self.python, "-u", "-m", self.worker_module,
                    json.dumps(self.worker_config, allow_nan=False)]
            self._process = subprocess.Popen(
                argv, shell=False, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, bufsize=0, env=self._worker_env(),
            )
            for stream in (self._process.stdin, self._process.stdout, self._process.stderr):
                os.set_blocking(stream.fileno(), False)
            self._stderr_thread = threading.Thread(
                target=self._drain_stderr, args=(self._process.stderr.fileno(),),
                name=f"{self.name}-stderr-{self._process.pid}", daemon=True,
            )
            self._stderr_thread.start()
            response = self._receive(deadline)
            if response.status != "ready":
                raise ProtocolError("expected stereo readiness after model load")
            self._info = response.diagnostics
            self._state = "ready"
        except Exception as exc:
            raise self._fail(exc) from exc

    def start(self) -> None:
        """Load one resident worker, or raise BackendUnavailable; idempotent while alive."""
        with self._lock:
            self._start_locked()

    def infer_pair(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Return float32 (H,W) pixel disparity; invalid input is BackendRequestError."""
        with self._lock:
            # Validate before spawning; malformed caller arrays never poison a live worker.
            try:
                payload = encode_request(StereoRequest(left=left, right=right))
            except ProtocolError as exc:
                raise BackendRequestError(str(exc)) from exc
            self._start_locked()
            deadline = time.monotonic() + self.timeout_s
            try:
                write_frame(_DeadlinePipe(self._process.stdin.fileno(), deadline), payload)
                response = self._receive(deadline)
                if response.status != "ok" or response.disparity.shape != left.shape[:2]:
                    raise ProtocolError("expected float32 disparity matching the input shape")
                self._info = response.diagnostics
                return response.disparity
            except Exception as exc:
                raise self._fail(exc) from exc

    def describe(self) -> dict[str, Any]:
        with self._lock:
            process = self._process
            return {
                **self._info,
                "backend": self.name,
                "python": self.python,
                "pid": process.pid if process else None,
                "running": self._state == "ready" and process is not None and process.poll() is None,
                "state": self._state,
                "cuda_visible_devices": self.cuda_visible_devices,
                "startup_timeout_s": self.startup_timeout_s,
                "timeout_s": self.timeout_s,
            }

    @property
    def health_error(self) -> str | None:
        """Read terminal failure or worker exit without waiting for the inference lock.

        Returns:
            Failure detail for an unusable worker, otherwise None. No worker is restarted.

        Example:
            >>> backend = FoundationStereoBackend()
            >>> backend.health_error is None
            True
            >>> backend.close()
            >>> backend.health_error
            'foundation_stereo backend is closed'
        """
        if self._state == "failed":
            return self._failure or f"{self.name} worker failed"
        if self._state == "closed":
            return f"{self.name} backend is closed"
        process = self._process
        if self._state == "ready" and process is not None:
            code = process.poll()
            if code is not None:
                return f"{self.name} worker exited with code {code}"
        return None

    def _dispose(self, *, graceful: bool) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        try:
            if graceful and process.poll() is None:
                try:
                    write_frame(
                        _DeadlinePipe(process.stdin.fileno(), time.monotonic() + 0.2),
                        self.shutdown_payload(),
                    )
                    process.wait(timeout=0.2)
                except (OSError, ValueError, TimeoutError, subprocess.TimeoutExpired):
                    pass
            if process.poll() is None:
                process.kill()  # only this Popen child; never a process group / other models
            process.wait(timeout=1.0)
        finally:
            if process.poll() is None:
                # A child stuck in the kernel may outlive SIGKILL's bounded wait. Keep
                # ownership so a later close() can finish reaping; never restart it.
                self._process = process
            self._stderr_stop.set()
            if self._stderr_thread is not None:
                self._stderr_thread.join(timeout=0.2)
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    stream.close()

    def close(self) -> None:
        """Bounded shutdown and reap of our own worker; safe to call repeatedly."""
        with self._lock:
            if self._state != "failed":
                self._state = "closed"
            self._dispose(graceful=True)

    def __del__(self) -> None:  # pragma: no cover - best effort at interpreter shutdown
        try:
            self.close()
        except Exception:
            pass
