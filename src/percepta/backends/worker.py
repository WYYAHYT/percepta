"""Worker entry point: hosts one model inside its own interpreter.

Run as ``python -m percepta.backends.worker '<config-json>'``. It reads framed requests from
stdin and writes framed responses to stdout, so **nothing may print to stdout** — use stderr
for logging (the parent drains it).

The worker is deliberately thin: it knows how to build a model adapter from a factory name,
then loops on the protocol. All model-specific knowledge lives behind
:class:`~percepta.backends.base.ModelBackend`.

Why a *factory name* rather than an import path: the isolated virtualenv may not have
percepta installed, but it always has numpy (models need it). Percepta itself is importable
because the parent puts its ``src/`` on ``PYTHONPATH`` — see
``docs/model-isolation.md`` §6.
"""

from __future__ import annotations

import importlib
import json
import logging
import sys
import traceback

import numpy as np

from .base import BackendRequestError
from .protocol import (
    ProtocolError,
    RequestFrame,
    ResponseFrame,
    decode_request,
    encode_response,
    read_frame,
    write_frame,
)

__all__ = ["main", "serve", "build_backend"]

log = logging.getLogger("percepta.backends.worker")


def build_backend(config: dict) -> object:
    """Instantiate the model adapter named by ``config["backend_factory"]``.

    The value is ``"module:attribute"``. Keeping it as data (not a hard import) means the
    worker can host any model tool without knowing about it at import time.
    """
    spec = config.get("backend_factory")
    if not spec or ":" not in str(spec):
        raise ValueError(f"worker config needs 'backend_factory' as 'module:attribute', got {spec!r}")
    module_name, _, attribute = str(spec).partition(":")
    module = importlib.import_module(module_name)
    try:
        factory = getattr(module, attribute)
    except AttributeError as exc:
        raise ValueError(f"{module_name} has no attribute {attribute!r}") from exc
    return factory(**dict(config.get("backend_args") or {}))


def _emit(frame: ResponseFrame) -> None:
    write_frame(sys.stdout.buffer, encode_response(frame))


def _error_frame(exc: BaseException, kind: str) -> ResponseFrame:
    return ResponseFrame(status="error", error=f"{type(exc).__name__}: {exc}", error_kind=kind)


def serve(config: dict) -> int:
    """Main loop. Returns a process exit code."""
    try:
        backend = build_backend(config)
    except Exception as exc:  # noqa: BLE001 - report, do not crash silently
        traceback.print_exc(file=sys.stderr)
        _emit(_error_frame(exc, "unavailable"))
        return 1

    _emit(ResponseFrame(status="ready"))
    stdin = sys.stdin.buffer

    try:
        while True:
            payload = read_frame(stdin)
            if payload is None:  # parent closed the pipe
                return 0
            try:
                request: RequestFrame = decode_request(payload)
            except ProtocolError as exc:
                _emit(_error_frame(exc, "invalid_request"))
                continue

            if request.shutdown:
                return 0

            try:
                if request.image is not None:
                    backend.set_image(request.image, request.depth)
                    if request.prompt_json == "":
                        # prepare-only frame: acknowledge so set_image() can return.
                        _emit(ResponseFrame(status="ok"))
                        continue
                from percepta.schema import Prompt  # imported lazily: worker may be slim

                prompt = Prompt.model_validate_json(request.prompt_json)
                masks, boxes, scores = backend.segment(prompt)
                _emit(
                    ResponseFrame(
                        status="ok",
                        masks=None if masks is None or masks.size == 0 else masks,
                        boxes=boxes,
                        scores=scores,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - one bad request must not kill the worker
                kind = "invalid_request" if isinstance(exc, (BackendRequestError, ValueError, KeyError)) else "unavailable"
                traceback.print_exc(file=sys.stderr)
                _emit(_error_frame(exc, kind))
    finally:
        try:
            backend.close()
        except Exception:  # noqa: BLE001
            pass


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,  # never stdout: it carries the protocol
        format="%(levelname)s %(name)s: %(message)s",
    )
    if not argv:
        print("usage: python -m percepta.backends.worker '<config-json>'", file=sys.stderr)
        return 2
    try:
        config = json.loads(argv[0])
    except json.JSONDecodeError as exc:
        print(f"invalid worker config JSON: {exc}", file=sys.stderr)
        return 2
    return serve(config)


if __name__ == "__main__":  # pragma: no cover - exercised through the backend
    raise SystemExit(main())


_ = np  # numpy is a hard requirement of every model adapter
