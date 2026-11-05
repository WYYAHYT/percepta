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
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np

from ..schema import Prompt
from .base import BackendRequestError, BackendUnavailable
from .protocol import (
    ProtocolError,
    RequestFrame,
    decode_response,
    encode_request,
    read_frame,
    write_frame,
)

__all__ = ["SubprocessBackend"]

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 120.0


class SubprocessBackend:
    """A :class:`~percepta.backends.base.ModelBackend` that lives in another interpreter."""

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
        self.cuda_visible_devices = cuda_visible_devices
        self.extra_env = dict(env or {})
        self._process: subprocess.Popen[bytes] | None = None
        self._lock = threading.Lock()
        self._frame_prepared = False
        self._stderr_tail: list[str] = []

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
            )
        except OSError as exc:
            raise BackendUnavailable(f"failed to start worker {self.module}: {exc}") from exc
        self._frame_prepared = False
        self._drain_stderr_async()
        self._await_ready()

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

        def pump() -> None:
            for raw in process.stderr:
                line = raw.decode("utf-8", "replace").rstrip()
                self._stderr_tail.append(line)
                del self._stderr_tail[:-40]  # keep the last 40 lines for diagnostics
                log.debug("model worker: %s", line)

        threading.Thread(target=pump, name=f"worker-stderr-{self.module}", daemon=True).start()

    def _await_ready(self) -> None:
        """The worker's first frame is a status frame with ``status="ready"``."""
        payload = self._read_frame_or_fail("worker exited before signalling readiness")
        from .protocol import decode_response

        response = decode_response(payload)
        if response.status != "ready":
            detail = response.error or (self._stderr_tail_tail() or "unknown error")
            self.close()
            raise BackendUnavailable(f"model worker failed to start: {detail}")

    def _stderr_tail_tail(self) -> str:
        return " | ".join(self._stderr_tail[-5:])

    def _read_frame_or_fail(self, message: str) -> bytes:
        process = self._process
        if process is None or process.stdout is None:
            raise BackendUnavailable(message)
        try:
            payload = read_frame(process.stdout)
        except ProtocolError as exc:
            self.close()
            raise BackendUnavailable(f"{message}: {exc}") from exc
        if payload is None:
            code = process.poll()
            self.close()
            detail = self._stderr_tail_tail()
            raise BackendUnavailable(
                f"{message} (exit code {code})" + (f": {detail}" if detail else "")
            )
        return payload

    def _request(self, frame: RequestFrame) -> bytes:
        """Send one frame and read exactly one response frame."""
        process = self._process
        if process is None or process.stdin is None:
            raise BackendUnavailable("model worker is not running")
        try:
            write_frame(process.stdin, encode_request(frame))
        except (BrokenPipeError, OSError) as exc:
            self.close()
            raise BackendUnavailable(f"model worker pipe closed: {exc}") from exc
        return self._read_frame_or_fail("model worker produced no response")

    # -- ModelBackend ----------------------------------------------------

    def set_image(self, image_rgb: np.ndarray, depth: np.ndarray | None = None) -> None:
        """Hand a new frame to the worker. The response is discarded (prepare only)."""
        with self._lock:
            self._ensure_running()
            payload = self._request(
                RequestFrame(prompt_json="", image=image_rgb, depth=depth)
            )
            response = decode_response(payload)
            if response.status == "error":
                raise self._map_error(response)
            self._frame_prepared = True

    def segment(self, prompt: Prompt) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
        """Run one prompt. Requires a successful :meth:`set_image` first."""
        with self._lock:
            self._ensure_running(for_segment=True)
            payload = self._request(
                RequestFrame(prompt_json=prompt.model_dump_json(), image=None)
            )
            response = decode_response(payload)
            if response.status == "error":
                raise self._map_error(response)
            if response.masks is None:
                return (
                    np.zeros((0, 1, 1), dtype=bool),
                    None,
                    np.zeros(0, dtype=float),
                )
            return response.masks, response.boxes, response.scores

    @staticmethod
    def _map_error(response) -> Exception:
        message = response.error or "model worker reported an error"
        if response.error_kind == "invalid_request":
            return BackendRequestError(message)
        return BackendUnavailable(message)

    def _ensure_running(self, *, for_segment: bool = False) -> None:
        """Start the worker if it is not running.

        ``for_segment`` makes a crash visible instead of silently papering over it. Without
        this, a worker killed mid-request would be respawned by the *next* ``segment`` call and
        the model would fail with ``"set_image() must be called before segment()"`` — which
        points at the caller's code while the real cause is a dead process. Reporting
        ``unavailable`` is both truthful and what spec §16.4 requires.
        """
        dead = self._process is not None and self._process.poll() is not None
        if dead and for_segment:
            code = self._process.poll() if self._process is not None else None
            detail = self._stderr_tail_tail()
            self.close()
            raise BackendUnavailable(
                f"model worker exited while idle (exit code {code})"
                + (f": {detail}" if detail else "")
            )
        if self._process is None or dead:
            self._spawn()

    def close(self) -> None:
        """Ask the worker to exit, then make sure it actually does."""
        process, self._process = self._process, None
        self._frame_prepared = False
        if process is None:
            return
        try:
            if process.stdin is not None and process.poll() is None:
                write_frame(process.stdin, encode_request(RequestFrame(prompt_json="", shutdown=True)))
        except (BrokenPipeError, OSError, ValueError):
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:  # pragma: no cover - stubborn process
                log.warning("model worker %s did not exit after SIGKILL", self.module)
        for stream in (process.stdin, process.stdout, process.stderr):
            try:
                if stream is not None:
                    stream.close()
            except OSError:
                pass

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "python": self.python,
            "module": self.module,
            "pid": None if self._process is None else self._process.pid,
            "cuda_visible_devices": self.cuda_visible_devices,
            "running": self._process is not None and self._process.poll() is None,
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
