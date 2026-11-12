"""Isolated three-view model worker; deployment argv selects the adapter, requests carry only data."""

from __future__ import annotations

import os
import sys
import traceback

from .foundation_stereo_worker import build_backend
from .multiview_protocol import (
    MAX_REQUEST_BYTES,
    MultiViewResponse,
    ProtocolError,
    decode_request,
    encode_response,
)
from .stereo_protocol import read_frame, write_frame


class _RequestReader:
    def __init__(self, stream):
        self.stream = stream

    def read(self, size: int) -> bytes:
        if size > MAX_REQUEST_BYTES:
            raise ProtocolError("multiview request exceeds length limit")
        return self.stream.read(size)


def serve(config: dict, stdin, stdout) -> int:
    """Run one calibrated model with terminal failures and explicit metric units.

    Args:
        config: Trusted deployment factory and constructor keywords.
        stdin: Binary request stream.
        stdout: Private binary response stream, separate from model logging.

    Returns:
        Zero for clean shutdown, one for model/protocol failure.

    Example:
        >>> serve(deployment_config, request_stream, response_stream)  # doctest: +SKIP
        0
    """
    backend = None
    try:
        if not config.get("backend_factory"):
            raise ValueError("multiview worker requires a deployment adapter")
        backend = build_backend(config)
        write_frame(stdout, encode_response(MultiViewResponse("ready", diagnostics=backend.describe())))
        reader = _RequestReader(stdin)
        while True:
            payload = read_frame(reader)
            if payload is None:
                return 0
            request = decode_request(payload)
            if request.op == "shutdown":
                return 0
            result = backend.infer(request.images, request.intrinsics, request.T_world_cam)
            if not isinstance(result, dict) or result.get("depth_unit") != "m":
                raise ProtocolError("adapter must explicitly return metric optical-axis depth")
            if result["depth"].shape != request.images.shape[:3]:
                raise ProtocolError("adapter depth must match the original pixel grid")
            write_frame(stdout, encode_response(MultiViewResponse(
                "ok", result["depth"], result.get("confidence"), backend.describe(),
            )))
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        try:
            write_frame(stdout, encode_response(MultiViewResponse("error", error=f"{type(exc).__name__}: {exc}"[:8192])))
        except (OSError, ProtocolError):
            pass
        return 1
    finally:
        if backend is not None:
            backend.close()


def main(argv: list[str] | None = None) -> int:
    """Start a worker whose stdout is reserved for binary RPC.

    Args:
        argv: One trusted deployment JSON argument; defaults to command-line arguments.

    Returns:
        Worker exit status.

    Example:
        >>> main([])
        2
    """
    import json

    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("expected one deployment config JSON argument", file=sys.stderr)
        return 2
    try:
        config = json.loads(argv[0])
        if not isinstance(config, dict):
            raise ValueError("deployment config must be an object")
    except ValueError as exc:
        print(f"invalid deployment config: {exc}", file=sys.stderr)
        return 2
    with os.fdopen(os.dup(sys.stdout.fileno()), "wb") as wire:
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        sys.stdout = sys.stderr
        return serve(config, sys.stdin.buffer, wire)


if __name__ == "__main__":
    raise SystemExit(main())
