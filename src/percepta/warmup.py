"""Startup warm-up: pay the one-time model cost before the first real request.

Why this exists: the first SAM3 call costs ~16 s (worker spawn + 3.2 GiB checkpoint read +
CUDA init + cuDNN autotuning), while later calls take 250-520 ms. An Agent Policy that sends
its first request into that cold path either times out or has to carry a special "first call
is slow" branch. Both are avoidable: do the expensive work at startup.

Design decisions, and why:

* **In a background thread, not blocking startup.** A service that refuses connections for
  20 s looks broken to a supervisor, a `docker run` healthcheck, or a load balancer. The port
  binds immediately; ``/healthz`` reports the warm-up state separately so callers can tell
  "listening but cold" from "wrong port".
* **Through the real pipeline, not a fake tool call.** The cost we are removing lives in
  image decode → prompt → model forward → result assembly. Warming only the model would
  leave the rest cold.
* **Never fatal.** A failed warm-up is recorded and reported; the service stays up, because
  a broken model is a ``/healthz`` problem, not a reason to refuse to boot (decision D9).
* **A synthetic image is correct here.** SAM3 finds nothing in a flat colour-field and that is
  fine — warm-up exercises the expensive forward pass, not object detection quality. We
  discard the response entirely.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import numpy as np

from .schema import PerceiveOptions, PerceiveRequest, Prompt

__all__ = ["WarmupState", "run_warmup", "start_warmup", "build_warmup_request"]

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 300.0
DEFAULT_SIZE = (480, 640)


class WarmupState:
    """Thread-safe record of what the warm-up did, for ``/healthz`` and logs."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.status = "pending"  # pending | running | ok | error | disabled
        self.duration_ms: float | None = None
        self.error: str | None = None
        self.started_at: float | None = None
        self.finished_at: float | None = None

    def set(self, **fields: Any) -> None:
        with self._lock:
            for key, value in fields.items():
                setattr(self, key, value)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            data: dict[str, Any] = {
                "status": self.status,
                "duration_ms": self.duration_ms,
            }
            if self.error:
                data["error"] = self.error
            return data

    @property
    def ready(self) -> bool:
        """True once warm-up finished *or* failed — either way the service is usable."""
        with self._lock:
            return self.status in {"ok", "error", "disabled"}


def build_warmup_request(size: tuple[int, int] = DEFAULT_SIZE, prompt: str = "object") -> PerceiveRequest:
    """A minimal, valid request whose only purpose is to exercise the hot path."""
    from .masks import encode_image

    height, width = int(size[0]), int(size[1])
    # A soft gradient rather than a flat field: flat images can short-circuit some kernels
    # (e.g. all-zero inputs), which would leave part of the path cold.
    ramp = np.linspace(40, 200, width, dtype=np.float32)
    image = np.repeat(ramp[None, :, None], height, axis=0).repeat(3, axis=2)
    image = np.clip(image, 0, 255).astype(np.uint8)

    return PerceiveRequest(
        request_id="warmup",
        images={"warmup": encode_image(image, encoding="png", camera="warmup")},
        prompts=[Prompt(type="text", text=prompt, min_score=0.05, top_k=1)],
        # No calibration and no fusion: warm-up must not depend on a camera rig existing.
        options=PerceiveOptions(fuse=False, masks="rle", cache=False),
    )


def run_warmup(
    runner: Any,
    state: WarmupState,
    *,
    size: tuple[int, int] = DEFAULT_SIZE,
    prompt: str = "object",
) -> None:
    """Run one request through the pipeline and record the outcome. Never raises."""
    state.set(status="running", started_at=time.time())
    started = time.perf_counter()
    try:
        response = runner.perceive(build_warmup_request(size, prompt))
        elapsed = (time.perf_counter() - started) * 1000.0
        statuses = [percept.status for percept in response.percepts]
        # ``not_found`` is a *successful* warm-up: the model ran and reported nothing, which is
        # exactly what a synthetic image should produce. Only real failures count.
        broken = [
            percept
            for percept in response.percepts
            if percept.status in {"unavailable", "invalid_request", "timeout"}
        ]
        if broken:
            state.set(
                status="error",
                duration_ms=elapsed,
                error="; ".join(
                    f"{percept.provenance.tool}: {percept.message or percept.status}"
                    for percept in broken[:3]
                ),
                finished_at=time.time(),
            )
            log.warning("warm-up finished with errors in %.0f ms: %s", elapsed, state.error)
            return
        state.set(status="ok", duration_ms=elapsed, error=None, finished_at=time.time())
        log.info(
            "warm-up ok in %.0f ms (statuses=%s)", elapsed, ",".join(statuses) or "none"
        )
    except Exception as exc:  # noqa: BLE001 - warm-up must never take the service down
        elapsed = (time.perf_counter() - started) * 1000.0
        state.set(
            status="error",
            duration_ms=elapsed,
            error=f"{type(exc).__name__}: {exc}",
            finished_at=time.time(),
        )
        log.warning("warm-up failed after %.0f ms: %s", elapsed, exc)


def start_warmup(
    runner: Any,
    *,
    size: tuple[int, int] = DEFAULT_SIZE,
    prompt: str = "object",
    enabled: bool = True,
) -> tuple[WarmupState, threading.Thread | None]:
    """Begin warming in the background; returns the state and the thread (or ``None``)."""
    state = WarmupState()
    if not enabled:
        state.set(status="disabled")
        return state, None

    thread = threading.Thread(
        target=run_warmup,
        args=(runner, state),
        kwargs={"size": size, "prompt": prompt},
        name="percepta-warmup",
        daemon=True,
    )
    thread.start()
    return state, thread
