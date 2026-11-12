"""HTTP routes (spec §6) and the shared service state.

Frozen endpoints:

=================== ==========================================================
GET  /healthz       ``{"ok": true, "uptime_s": …, "tools": [{name, version, status}]}``
GET  /v1/info       schema/package version, tool capabilities, device, model sha
GET  /v1/tools      tools + capabilities
POST /v1/perceive   ``PerceiveRequest`` -> ``PerceiveResponse``
GET  /metrics       Prometheus text (404 when ``metrics.enabled`` is false)
=================== ==========================================================

Error mapping follows appendix A: ``400 invalid_request``, ``401 unauthorized``,
``413 payload_too_large``, ``503 busy``, ``504 timeout``, ``500 internal``. A ``not_found``
Percept is **not** an HTTP error: it travels inside a 200 response.

A missing model or an unloadable tool is likewise a **Percept status**:
``POST /v1/perceive`` answers 200 with ``percepts[i].status == "unavailable"`` (and a
``tool_unavailable`` warning), because spec §6 lists only invalid/unauthorized/too-large/
busy/timeout/internal as HTTP errors, while ``/healthz`` reports the tool as ``error``.
That way one unavailable tool in a multi-tool pipeline does not hide the results the other
tools produced.

Authentication (optional): when ``config.server.resolve_token()`` returns a token, every
``/v1/*`` endpoint requires ``Authorization: Bearer <token>``. ``/healthz`` and ``/metrics``
stay open so probes and scrapers work without a secret.
"""

from __future__ import annotations

import asyncio
import hmac
import importlib.util
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from fastapi import APIRouter, Depends, Request
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import JSONResponse, Response

from .. import __version__
from ..config import AppConfig
from ..pipeline import PipelineRequestError, PipelineRunner
from ..registry import ToolEntry, ToolRegistry
from ..schema import SCHEMA_VERSION, TOOL_API_VERSION, PerceiveRequest, PerceiveResponse
from . import metrics as metrics_mod
from .logging import log_event

__all__ = ["ServiceState", "router", "percept_statuses", "tool_capabilities_dict", "tool_row"]

log = logging.getLogger("percepta.service")

router = APIRouter()

#: Registry status -> healthz status (spec §11.3 allows ``ok|loading|error`` only).
#: ``lazy`` means "imported but not loaded yet": it is available and will load on demand.
_STATUS_MAP = {"ok": "ok", "lazy": "ok", "loading": "loading", "error": "error"}


# ---------------------------------------------------------------------------
# Service state
# ---------------------------------------------------------------------------


@dataclass
class ServiceState:
    """Everything the routes need: config, registry, runner, admission control.

    ``max_queue`` bounds how many requests may be inside the pipeline at once; a request
    that cannot be admitted is rejected with ``503 busy`` instead of queueing without a
    bound. ``max_queue <= 0`` therefore admits nothing — a deliberate, documented way to
    take a worker out of service without stopping the process (``/healthz`` still works).
    """

    config: AppConfig
    registry: ToolRegistry
    runner: PipelineRunner
    semaphore: threading.BoundedSemaphore | None = None
    started_at: float = field(default_factory=time.monotonic)
    # Set once the app's lifespan starts a background warm-up; ``None`` means no lifespan ran
    # (e.g. a bare ``create_app()`` in a unit test), which /healthz reports as unknown.
    warmup: Any = None
    pending: set[asyncio.Task[PerceiveResponse]] = field(default_factory=set)
    closing: bool = False

    @property
    def uptime_s(self) -> float:
        return round(time.monotonic() - self.started_at, 3)

    def try_acquire(self) -> bool:
        if self.closing or self.semaphore is None:
            return False
        return bool(self.semaphore.acquire(blocking=False))

    def release(self) -> None:
        if self.semaphore is None:
            return
        try:
            self.semaphore.release()
        except ValueError:  # pragma: no cover - defensive: more releases than acquires
            log.warning("released an admission slot that was never acquired")

    # -- introspection ---------------------------------------------------

    def entries(self) -> list[tuple[ToolEntry, Any]]:
        """Tools that are part of *this* deployment: enabled in config or unlisted.

        ``AppConfig.is_enabled`` returns ``True`` for names absent from ``config.tools`` (the
        tool keeps its own defaults), so only an explicit ``enabled: false`` is hidden.
        A disabled tool is *absent* from the listing rather than reported with a made-up
        status — the frozen §11.3 vocabulary is ``ok|loading|error`` and "absent" is the
        correct representation for "not part of this deployment".
        """
        rows: list[tuple[ToolEntry, Any]] = []
        for name in self.registry.names():
            if not self.config.is_enabled(name):
                continue
            entry = self.registry.entry(name)
            rows.append((entry, entry.instance))
        return rows

    def health_rows(self) -> list[dict[str, Any]]:
        rows = []
        for entry, _instance in self.entries():
            declared = entry.describe()
            rows.append(
                {
                    "name": entry.name,
                    # ``describe()`` reads version/kind off the tool *class*, so an
                    # unloaded tool still reports its real version instead of 0.0.0.
                    "version": declared["version"],
                    "status": _STATUS_MAP.get(entry.status, "error"),
                }
            )
        return rows


# NOTE: there is deliberately no local ``entry_version`` helper any more. Duplicating the
# registry's logic here is what made /healthz and /v1/tools report ``0.0.0`` for tools that
# were registered but never loaded (no GPU/weights) while the CLI reported the real version.
# All introspection now goes through ``ToolEntry.describe()`` — one source of truth.


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_state(request: Request) -> ServiceState:
    state = getattr(request.app.state, "service", None)
    if state is None:  # pragma: no cover - create_app always installs it
        raise RuntimeError("percepta app has no service state; build it with create_app()")
    return state


def error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": code, "message": message})


def tool_capabilities_dict(instance: Any) -> dict[str, Any] | None:
    """Serialise ``ToolCapabilities`` (``set`` -> sorted list) for JSON."""
    if instance is None:
        return None
    try:
        caps = instance.capabilities()
    except Exception:  # noqa: BLE001 - introspection must never fail a route
        return None
    return {
        "kind": caps.kind,
        "prompts": sorted(caps.prompts),
        "masks": caps.masks,
        "boxes": caps.boxes,
        "points": caps.points,
        "depth": caps.depth,
        "multi_view": caps.multi_view,
        "geometry": caps.geometry,
    }


def _model_sha256(instance: Any) -> str | None:
    """Use the tool's provenance metadata without touching the loaded model."""
    try:
        metadata = getattr(instance, "_model_meta", None)
        if callable(metadata):
            return metadata()[1]
        return getattr(instance, "model_sha256", None) or getattr(instance, "_model_sha256", None)
    except Exception:  # noqa: BLE001 - introspection must remain available for broken tools
        return None


def tool_row(entry: ToolEntry) -> dict[str, Any]:
    instance = entry.instance
    declared = entry.describe()
    row: dict[str, Any] = {
        "name": entry.name,
        "version": declared["version"],
        "status": _STATUS_MAP.get(entry.status, "error"),
        "kind": declared["kind"],
        "source": entry.source,
        "capabilities": tool_capabilities_dict(instance),
        "model_sha256": _model_sha256(instance),
        "loaded": instance is not None,
    }
    if entry.error:
        row["error"] = entry.error
    return row


def authorize(request: Request) -> None:
    """Optional bearer-token check against ``server.resolve_token()`` (spec §6)."""
    token = get_state(request).config.server.resolve_token()
    if not token:
        return
    header = request.headers.get("authorization") or ""
    scheme, _, supplied = header.partition(" ")
    if scheme.lower() != "bearer" or not hmac.compare_digest(
        supplied.strip().encode("utf-8"), token.encode("utf-8")
    ):
        raise StarletteHTTPException(status_code=401, detail="unauthorized")


def _record_percepts(response: PerceiveResponse) -> None:
    for percept in response.percepts + ([response.fused] if response.fused else []):
        provenance = percept.provenance
        tool = provenance.tool if provenance is not None else "unknown"
        metrics_mod.METRICS.inc(
            "percepta_requests_total", {"tool": tool, "status": percept.status}
        )
        metrics_mod.METRICS.observe_ms(
            "percepta_latency_ms",
            float(provenance.latency_ms) if provenance is not None else 0.0,
            {"tool": tool},
        )
        if provenance is not None and provenance.cache_hit:
            metrics_mod.METRICS.inc("percepta_cache_hits_total")


def _torch_available() -> bool:
    try:
        return importlib.util.find_spec("torch") is not None
    except (ImportError, ValueError):  # pragma: no cover - broken install
        return False


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/healthz")
def healthz(request: Request) -> dict[str, Any]:
    """Liveness/readiness: always 200 while the process serves, tools carry the detail.

    ``warmup`` separates "the port is listening" from "the model is actually hot". A caller
    that cares about the ~16 s cold path can wait for ``warmup.status == "ok"``; a supervisor
    that only wants liveness can ignore it. ``ok`` stays true either way — a failed warm-up
    means the model is broken, which the ``tools[].status`` field already reports, and it is
    not a reason to fail liveness.
    """
    state = get_state(request)
    body: dict[str, Any] = {
        "ok": True,
        "uptime_s": state.uptime_s,
        "tools": state.health_rows(),
    }
    warmup = getattr(state, "warmup", None)
    body["warmup"] = warmup.snapshot() if warmup is not None else {"status": "unknown"}
    return body


@router.get("/v1/info", dependencies=[Depends(authorize)])
def info(request: Request) -> dict[str, Any]:
    state = get_state(request)
    config = state.config
    token = config.server.resolve_token()
    torch_installed = _torch_available()
    return {
        "name": "percepta",
        "version": __version__,
        "schema_version": SCHEMA_VERSION,
        "tool_api_version": TOOL_API_VERSION,
        "uptime_s": state.uptime_s,
        "server": {
            "host": config.server.host,
            "port": config.server.port,
            "max_body_bytes": config.server.max_body_bytes,
            "max_queue": config.server.max_queue,
            "request_timeout_ms": config.server.request_timeout_ms,
            "auth_required": token is not None,
        },
        "pipeline": {"default": list(config.pipeline.default)},
        "cache": {
            "enabled": config.cache.enabled,
            "max_entries": config.cache.max_entries,
            "ttl_s": config.cache.ttl_s,
        },
        "metrics_enabled": config.metrics.enabled,
        "device": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_installed": torch_installed,
            "gpu": "unknown" if torch_installed else "none",
        },
        "tools": [tool_row(entry) for entry, _instance in state.entries()],
    }


@router.get("/v1/tools", dependencies=[Depends(authorize)])
def tools(request: Request) -> dict[str, Any]:
    state = get_state(request)
    return {"tools": [tool_row(entry) for entry, _instance in state.entries()]}


@router.post("/v1/perceive", dependencies=[Depends(authorize)])
async def perceive(request: Request, payload: PerceiveRequest) -> Response:
    """Run the pipeline; errors use appendix A, ``not_found`` stays a 200 Percept status."""
    state = get_state(request)
    if not state.try_acquire():
        metrics_mod.METRICS.inc("percepta_http_responses_total", {"code": "503"})
        return error_response(
            503, "busy", f"queue full: max_queue={state.config.server.max_queue}"
        )

    metrics_mod.METRICS.add_gauge("percepta_inflight", 1)
    started = time.perf_counter()
    # The server limit wins over the per-request one: a caller may shorten, never extend.
    timeout_ms = min(
        int(state.config.server.request_timeout_ms), int(payload.options.timeout_ms)
    )
    timeout_s = max(timeout_ms, 1) / 1000.0
    task = asyncio.create_task(asyncio.to_thread(state.runner.perceive, payload))
    state.pending.add(task)

    def completed(done: asyncio.Task[PerceiveResponse]) -> None:
        state.pending.discard(done)
        state.release()
        metrics_mod.METRICS.add_gauge("percepta_inflight", -1)
        if not done.cancelled():
            done.exception()  # Retrieve failures even after the HTTP caller has left.

    task.add_done_callback(completed)
    try:
        response = await asyncio.wait_for(
            asyncio.shield(task), timeout=timeout_s
        )
    except asyncio.TimeoutError:
        metrics_mod.METRICS.inc("percepta_http_responses_total", {"code": "504"})
        log.warning(
            "perceive timeout",
            extra={"request_id": payload.request_id, "status": "timeout",
                   "latency_ms": round((time.perf_counter() - started) * 1000.0, 3)},
        )
        return error_response(504, "timeout", f"request exceeded {timeout_ms} ms")
    except PipelineRequestError as exc:
        metrics_mod.METRICS.inc("percepta_http_responses_total", {"code": str(exc.status_code)})
        return JSONResponse(status_code=exc.status_code, content=exc.as_dict())
    except Exception:  # noqa: BLE001 - appendix A: unexpected -> 500 internal
        metrics_mod.METRICS.inc("percepta_http_responses_total", {"code": "500"})
        log.exception(
            "perceive failed",
            extra={"request_id": payload.request_id, "status": "internal"},
        )
        return error_response(500, "internal", "unexpected error while perceiving")
    elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
    metrics_mod.METRICS.observe_ms("percepta_request_latency_ms", elapsed_ms)
    metrics_mod.METRICS.inc("percepta_http_responses_total", {"code": "200"})
    _record_percepts(response)
    log_event(
        log,
        "perceive",
        request_id=response.request_id,
        status="ok",
        latency_ms=elapsed_ms,
        cache_hit=any(p.provenance.cache_hit for p in response.percepts),
    )
    return JSONResponse(content=response.model_dump(mode="json"))


@router.get("/metrics")
def metrics(request: Request) -> Response:
    """Prometheus text, or 404 when ``metrics.enabled`` is false (spec §6, §11.3)."""
    state = get_state(request)
    if not state.config.metrics.enabled:
        return error_response(404, "not_found", "metrics are disabled (metrics.enabled=false)")
    return Response(
        content=metrics_mod.render_prometheus(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


def percept_statuses(response: PerceiveResponse) -> list[str]:
    """Percept statuses in response order (used by tests and embedding callers)."""
    return [percept.status for percept in response.percepts]
