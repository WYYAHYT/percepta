"""FastAPI application factory (spec §6).

``create_app(config)`` wires together the frozen seams:

* the tool registry, configured **and warmed** from ``config.tools`` / ``config.pipeline``
  so ``/healthz`` and ``/v1/tools`` tell the truth from the first request — decision D9: a
  missing ``sam3`` must show up as ``error`` without crashing the process;
* one :class:`~percepta.pipeline.PipelineRunner`, shared by every request;
* admission control (``server.max_queue``), body-size limiting (``server.max_body_bytes``)
  and per-request timeouts (``server.request_timeout_ms``);
* appendix-A error mapping, including ``RequestValidationError`` -> ``400 invalid_request``
  because the frozen contract says 400 for a malformed body (FastAPI's default is 422).

Bind address is *not* handled here: ``percepta serve`` owns the loopback-only policy.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .. import __version__
from ..config import AppConfig
from ..pipeline import PipelineRequestError, PipelineRunner, prepare_tools
from ..registry import ToolRegistry
from ..warmup import WarmupState, start_warmup
from .logging import configure_logging
from .routes import ServiceState, error_response, router

__all__ = ["BodySizeLimitMiddleware", "create_app"]

log = logging.getLogger("percepta.service")

#: appendix A: HTTP status -> stable error code.
ERROR_CODES = {
    400: "invalid_request",
    401: "unauthorized",
    403: "unauthorized",
    404: "not_found",
    405: "invalid_request",
    413: "payload_too_large",
    415: "invalid_request",
    422: "invalid_request",
    429: "busy",
    503: "busy",
    504: "timeout",
    500: "internal",
}


class BodySizeLimitMiddleware:
    """Reject bodies larger than ``max_body_bytes`` with ``413 payload_too_large`` (spec §6).

    The ``Content-Length`` header is checked first (the normal case). Bodies that arrive
    without it — chunked uploads — are buffered up to the limit before the app sees them,
    and once the limit is crossed the request is answered with a 413 immediately. Buffering
    is what Starlette's ``request.body()`` does anyway, and deciding *before* the app runs
    keeps FastAPI's own body-parsing error handling from turning the rejection into a 400.
    """

    def __init__(self, app: ASGIApp, max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = int(max_body_bytes)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        limit = self.max_body_bytes
        declared = _content_length(scope)
        if declared is not None and declared > limit:
            await _send_payload_too_large(send, limit)
            return

        buffered: list[Message] = []
        total = 0
        while True:
            message = await receive()
            if message["type"] == "http.request":
                total += len(message.get("body", b""))
                if total > limit:
                    await _send_payload_too_large(send, limit)
                    return
                buffered.append(message)
                if not message.get("more_body", False):
                    break
                continue
            buffered.append(message)
            if message["type"] == "http.disconnect":
                break

        index = 0

        async def replay_receive() -> Message:
            nonlocal index
            if index < len(buffered):
                message = buffered[index]
                index += 1
                return message
            return await receive()

        await self.app(scope, replay_receive, send)


def _content_length(scope: Scope) -> int | None:
    for key, value in scope.get("headers", []):
        if key == b"content-length":
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
    return None


async def _send_payload_too_large(send: Send, limit: int) -> None:
    body = (
        b'{"error":"payload_too_large","message":"request body exceeds max_body_bytes='
        + str(limit).encode()
        + b'"}'
    )
    await send(
        {
            "type": "http.response.start",
            "status": 413,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


def create_app(
    config: AppConfig | None = None,
    *,
    registry: ToolRegistry | None = None,
    cache: Any = None,
) -> FastAPI:
    """Build the percepta ASGI app for ``config`` (defaults to :class:`AppConfig`)."""
    config = config or AppConfig()
    configure_logging(config.logging.level, config.logging.json_output)

    registry = registry if registry is not None else ToolRegistry()
    # Eager configuration + warm-up makes /healthz report the real load state
    # (ok|loading|error) and never raises: a broken/missing tool is recorded as `error`
    # (decision D9).
    prepare_tools(registry, config)

    max_queue = int(config.server.max_queue)
    state = ServiceState(
        config=config,
        registry=registry,
        runner=PipelineRunner(registry, config, cache),
        semaphore=threading.BoundedSemaphore(max_queue) if max_queue > 0 else None,
    )

    warmup_cfg = getattr(config, "warmup", None)
    warmup_enabled = bool(getattr(warmup_cfg, "enabled", True))
    warmup_state: WarmupState | None = None
    warmup_thread: threading.Thread | None = None

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        nonlocal warmup_state, warmup_thread
        # Warm up *inside* the lifespan, not at import: importing the app must stay free of
        # side effects, and this is where the process is actually committed to serving.
        warmup_state, warmup_thread = start_warmup(
            state.runner,
            size=(
                int(getattr(warmup_cfg, "height", 480)),
                int(getattr(warmup_cfg, "width", 640)),
            ),
            prompt=str(getattr(warmup_cfg, "prompt", "object")),
            enabled=warmup_enabled,
        )
        state.warmup = warmup_state
        _app.state.warmup = warmup_state
        try:
            yield
        finally:
            state.closing = True
            if state.pending:
                await asyncio.shield(asyncio.gather(*state.pending, return_exceptions=True))
            if warmup_thread is not None:
                await asyncio.to_thread(warmup_thread.join)
            await asyncio.to_thread(state.runner.close)

    app = FastAPI(
        title="percepta",
        version=__version__,
        summary="Pluggable robot-perception layer",
        lifespan=lifespan,
    )
    app.state.service = state
    app.state.config = config
    app.state.registry = registry
    app.state.runner = state.runner

    app.add_middleware(BodySizeLimitMiddleware, max_body_bytes=config.server.max_body_bytes)

    @app.exception_handler(RequestValidationError)
    async def _invalid_request(_request, exc: RequestValidationError) -> JSONResponse:
        details = [
            {"loc": [str(part) for part in error.get("loc", ())], "msg": error.get("msg", "")}
            for error in exc.errors()
        ]
        return JSONResponse(
            status_code=400,
            content={
                "error": "invalid_request",
                "message": "request body failed validation",
                "details": jsonable_encoder(details),
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_request, exc: StarletteHTTPException) -> JSONResponse:
        code = ERROR_CODES.get(exc.status_code, "internal")
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": code, "message": str(exc.detail)},
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(PipelineRequestError)
    async def _pipeline_error(_request, exc: PipelineRequestError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=exc.as_dict())

    @app.exception_handler(Exception)
    async def _internal_error(_request, exc: Exception) -> JSONResponse:  # pragma: no cover
        log.exception("unhandled error: %s", exc)
        return error_response(500, "internal", "unexpected error")

    app.include_router(router)
    return app
