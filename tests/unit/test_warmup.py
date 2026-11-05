"""Startup warm-up (``percepta.warmup``).

Warm-up exists to remove one specific cost: the first model call is ~16 s (worker spawn +
3.2 GiB checkpoint + CUDA init) versus 250-520 ms afterwards. These tests pin the properties
that make it safe and useful, all without a GPU:

* the service still serves while warming (it runs in a thread);
* a failing warm-up never kills the service, and never reports ``ok``;
* ``not_found`` counts as success — a synthetic frame legitimately yields no instances;
* the reported state is honest (``pending``/``running``/``ok``/``error``/``disabled``).
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from percepta.config import AppConfig, WarmupConfig
from percepta.schema import (
    PerceiveResponse,
    Percept,
    Provenance,
)
from percepta.warmup import (
    WarmupState,
    build_warmup_request,
    run_warmup,
    start_warmup,
)


def _percept(status: str, message: str | None = None, tool: str = "sam3") -> Percept:
    return Percept(
        percept_id=f"p-{tool}-{status}",
        status=status,
        message=message,
        provenance=Provenance(tool=tool, tool_version="0.1.0", kind="segmenter"),
    )


class _Runner:
    """Stands in for PipelineRunner; records calls and returns a canned response."""

    def __init__(self, percepts: list[Percept] | None = None, delay: float = 0.0) -> None:
        self.calls: list[object] = []
        self._percepts = percepts if percepts is not None else [_percept("not_found")]
        self._delay = delay

    def perceive(self, request):  # noqa: ANN001 - mirrors the real signature
        self.calls.append(request)
        if self._delay:
            time.sleep(self._delay)
        return PerceiveResponse(request_id=getattr(request, "request_id", "x"), percepts=self._percepts)


class _ExplodingRunner(_Runner):
    def perceive(self, request):  # noqa: ANN001
        raise RuntimeError("CUDA out of memory")


# ---------------------------------------------------------------------------
# The warm-up request itself
# ---------------------------------------------------------------------------


def test_warmup_request_is_valid_and_self_contained():
    request = build_warmup_request()
    assert list(request.images) == ["warmup"]
    assert request.prompts[0].type == "text"
    # No calibration and no fusion: warm-up must not require a camera rig to exist.
    assert request.calibration == {}
    assert request.options.fuse is False
    # Caching a throwaway request would only pollute the cache.
    assert request.options.cache is False


def test_warmup_image_is_not_a_flat_field():
    """A flat image can short-circuit kernels; a gradient exercises the real path."""
    from percepta.masks import decode_image

    request = build_warmup_request((64, 96))
    image = decode_image(request.images["warmup"])
    assert image.shape == (64, 96, 3)
    assert int(np.unique(image[:, :, 0]).size) > 1


# ---------------------------------------------------------------------------
# Outcome reporting
# ---------------------------------------------------------------------------


def test_not_found_counts_as_a_successful_warmup():
    """The model ran and found nothing in a synthetic frame — that is a pass."""
    state = WarmupState()
    run_warmup(_Runner([_percept("not_found")]), state)
    assert state.status == "ok"
    assert state.error is None
    assert state.duration_ms is not None


def test_ok_status_is_success():
    state = WarmupState()
    run_warmup(_Runner([_percept("ok")]), state)
    assert state.status == "ok"


@pytest.mark.parametrize("bad", ["unavailable", "invalid_request", "timeout"])
def test_broken_statuses_are_reported_as_error(bad):
    """A dead model must not be reported as a ready service."""
    state = WarmupState()
    run_warmup(_Runner([_percept(bad, message="no weights")]), state)
    assert state.status == "error"
    assert "no weights" in (state.error or "")
    assert state.duration_ms is not None


def test_exception_is_swallowed_and_recorded():
    """Warm-up must never take the service down."""
    state = WarmupState()
    run_warmup(_ExplodingRunner(), state)  # must not raise
    assert state.status == "error"
    assert "CUDA out of memory" in (state.error or "")


def test_warmup_passes_a_real_request_through_the_pipeline():
    runner = _Runner()
    state = WarmupState()
    run_warmup(runner, state, size=(32, 48), prompt="fixture")
    assert len(runner.calls) == 1
    request = runner.calls[0]
    assert request.prompts[0].text == "fixture"
    from percepta.masks import decode_image

    assert decode_image(request.images["warmup"]).shape == (32, 48, 3)


# ---------------------------------------------------------------------------
# Threading / lifecycle
# ---------------------------------------------------------------------------


def test_disabled_warmup_does_no_work():
    runner = _Runner()
    state, thread = start_warmup(runner, enabled=False)
    assert state.status == "disabled"
    assert thread is None
    assert runner.calls == []


def test_warmup_runs_in_the_background_so_the_service_serves_immediately():
    """The whole point: startup must not block for ~16 s."""
    runner = _Runner(delay=0.25)
    started = time.perf_counter()
    state, thread = start_warmup(runner)
    elapsed = time.perf_counter() - started
    assert elapsed < 0.2, "start_warmup blocked the caller"
    assert thread is not None and thread.is_alive()
    assert state.status in {"pending", "running"}
    assert not state.ready
    thread.join(timeout=5)
    assert state.status == "ok"
    assert state.ready


def test_state_snapshot_is_json_friendly():
    import json

    state = WarmupState()
    run_warmup(_Runner(), state)
    snapshot = state.snapshot()
    json.dumps(snapshot)  # must serialise
    assert snapshot["status"] == "ok"
    assert isinstance(snapshot["duration_ms"], float)


def test_snapshot_omits_error_when_clean():
    state = WarmupState()
    run_warmup(_Runner(), state)
    assert "error" not in state.snapshot()


def test_ready_is_true_after_failure_too():
    """A failed warm-up still means "done waiting" — the caller should not hang."""
    state = WarmupState()
    run_warmup(_ExplodingRunner(), state)
    assert state.ready is True


def test_state_is_thread_safe_under_concurrent_observation():
    state = WarmupState()
    runner = _Runner(delay=0.05)
    _, thread = start_warmup(runner, enabled=True)
    observations: list[str] = []

    def observe() -> None:
        for _ in range(50):
            observations.append(state.snapshot()["status"])

    watchers = [threading.Thread(target=observe) for _ in range(4)]
    for watcher in watchers:
        watcher.start()
    thread.join(timeout=5)
    for watcher in watchers:
        watcher.join(timeout=5)
    assert all(status in {"pending", "running", "ok"} for status in observations)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_config_has_warmup_defaults():
    config = WarmupConfig()
    assert config.enabled is True
    assert (config.height, config.width) == (480, 640)
    assert config.prompt


def test_appconfig_includes_warmup():
    assert AppConfig().warmup.enabled is True


def test_warmup_config_rejects_unknown_keys():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        WarmupConfig.model_validate({"nonsense": 1})
