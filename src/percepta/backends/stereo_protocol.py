"""Framed stereo RPC: <4sIIII> header, JSON, then two RGB uint8 or one float32 HW array."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .protocol import ProtocolError, read_frame, write_frame  # noqa: F401 - shared framing

REQUEST_MAGIC = b"STPQ"
RESPONSE_MAGIC = b"STPR"
PROTOCOL_VERSION = 1
HEADER = struct.Struct("<4sIIII")
MAX_PIXELS = 4096 * 4096
MAX_METADATA_BYTES = 64 * 1024
MAX_REQUEST_BYTES = HEADER.size + MAX_METADATA_BYTES + 6 * MAX_PIXELS
MAX_RESPONSE_BYTES = HEADER.size + MAX_METADATA_BYTES + 4 * MAX_PIXELS


@dataclass
class StereoRequest:
    op: str = "infer"
    left: np.ndarray | None = None
    right: np.ndarray | None = None


@dataclass
class StereoResponse:
    status: str
    disparity: np.ndarray | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


def _pixels(height: int, width: int) -> int:
    if height <= 0 or width <= 0 or height * width > MAX_PIXELS:
        raise ProtocolError(f"invalid stereo dimensions {height}x{width}; limit {MAX_PIXELS} pixels")
    return height * width


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key}")
        result[key] = value
    return result


def _bad_constant(value):
    raise ValueError(f"non-finite JSON value {value}")


def _pack(magic: bytes, meta: dict, height: int = 0, width: int = 0, raw: bytes = b"") -> bytes:
    try:
        blob = json.dumps(meta, allow_nan=False, separators=(",", ":")).encode("utf-8")
    except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
        raise ProtocolError(f"invalid stereo metadata: {exc}") from exc
    if len(blob) > MAX_METADATA_BYTES:
        raise ProtocolError("stereo metadata exceeds length limit")
    return HEADER.pack(magic, PROTOCOL_VERSION, len(blob), height, width) + blob + raw


def _unpack(payload: bytes, magic: bytes, limit: int) -> tuple[dict, int, int, memoryview]:
    if len(payload) < HEADER.size:
        raise ProtocolError("truncated stereo header")
    if len(payload) > limit:
        raise ProtocolError("stereo frame exceeds length limit")
    received, version, size, height, width = HEADER.unpack_from(payload)
    if received != magic:
        raise ProtocolError("bad stereo magic")
    if version != PROTOCOL_VERSION:
        raise ProtocolError(f"unsupported stereo version {version}")
    if not 0 < size <= MAX_METADATA_BYTES:
        raise ProtocolError("invalid stereo metadata length")
    end = HEADER.size + size
    if end > len(payload):
        raise ProtocolError("truncated stereo metadata")
    try:
        meta = json.loads(
            payload[HEADER.size:end].decode("utf-8"),
            object_pairs_hook=_object,
            parse_constant=_bad_constant,
        )
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ProtocolError(f"invalid stereo JSON: {exc}") from exc
    if not isinstance(meta, dict):
        raise ProtocolError("stereo metadata must be an object")
    return meta, height, width, memoryview(payload)[end:]


def _exact_size(raw: memoryview, expected: int) -> None:
    if len(raw) != expected:
        raise ProtocolError(f"stereo byte length mismatch (truncated/trailing): {len(raw)} != {expected}")


def encode_request(frame: StereoRequest) -> bytes:
    if frame.op == "shutdown":
        if frame.left is not None or frame.right is not None:
            raise ProtocolError("shutdown cannot contain images")
        return _pack(REQUEST_MAGIC, {"op": "shutdown"})
    if frame.op != "infer":
        raise ProtocolError("unknown stereo operation")
    left, right = frame.left, frame.right
    for image in (left, right):
        if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
                or image.ndim != 3 or image.shape[2] != 3):
            raise ProtocolError("stereo images must be uint8 RGB HWC")
    if left.shape != right.shape:
        raise ProtocolError("stereo image shapes must match")
    height, width = left.shape[:2]
    _pixels(height, width)
    return _pack(
        REQUEST_MAGIC, {"op": "infer", "dtype": "uint8"}, height, width,
        left.tobytes(order="C") + right.tobytes(order="C"),
    )


def decode_request(payload: bytes) -> StereoRequest:
    meta, height, width, raw = _unpack(payload, REQUEST_MAGIC, MAX_REQUEST_BYTES)
    if meta == {"op": "shutdown"}:
        if height or width or raw:
            raise ProtocolError("shutdown cannot contain images or trailing bytes")
        return StereoRequest(op="shutdown")
    if meta != {"op": "infer", "dtype": "uint8"}:
        raise ProtocolError("invalid stereo operation/dtype/fields")
    size = _pixels(height, width) * 3
    _exact_size(raw, size * 2)
    return StereoRequest(
        left=np.frombuffer(raw[:size], dtype=np.uint8).reshape(height, width, 3).copy(),
        right=np.frombuffer(raw[size:], dtype=np.uint8).reshape(height, width, 3).copy(),
    )


def encode_response(frame: StereoResponse) -> bytes:
    if not isinstance(frame.diagnostics, dict):
        raise ProtocolError("diagnostics must be an object")
    if frame.status == "ok":
        disparity = frame.disparity
        if (not isinstance(disparity, np.ndarray) or disparity.dtype != np.float32
                or disparity.ndim != 2):
            raise ProtocolError("disparity must be finite float32 HW")
        height, width = disparity.shape
        _pixels(height, width)
        if not np.isfinite(disparity).all():
            raise ProtocolError("disparity must be finite float32 HW")
        if frame.error is not None:
            raise ProtocolError("ok cannot contain an error")
        return _pack(
            RESPONSE_MAGIC,
            {"status": "ok", "dtype": "float32_le", "diagnostics": frame.diagnostics},
            height, width, disparity.astype("<f4", copy=False).tobytes(order="C"),
        )
    if frame.disparity is not None:
        raise ProtocolError("control response cannot contain disparity")
    if frame.status == "ready" and frame.error is None:
        return _pack(RESPONSE_MAGIC, {"status": "ready", "diagnostics": frame.diagnostics})
    if frame.status == "error" and isinstance(frame.error, str) and frame.error:
        return _pack(RESPONSE_MAGIC, {"status": "error", "error": frame.error})
    raise ProtocolError("invalid stereo response status/error")


def decode_response(payload: bytes) -> StereoResponse:
    meta, height, width, raw = _unpack(payload, RESPONSE_MAGIC, MAX_RESPONSE_BYTES)
    status = meta.get("status")
    if status == "ok":
        if (set(meta) != {"status", "dtype", "diagnostics"}
                or meta["dtype"] != "float32_le" or not isinstance(meta["diagnostics"], dict)):
            raise ProtocolError("invalid disparity dtype/diagnostics/fields")
        _exact_size(raw, _pixels(height, width) * 4)
        disparity = np.frombuffer(raw, dtype="<f4").reshape(height, width).astype(np.float32, copy=True)
        if not np.isfinite(disparity).all():
            raise ProtocolError("non-finite disparity")
        return StereoResponse(status="ok", disparity=disparity, diagnostics=meta["diagnostics"])
    if height or width or raw:
        raise ProtocolError("control response contains data/trailing bytes")
    if status == "ready" and set(meta) == {"status", "diagnostics"}:
        if isinstance(meta["diagnostics"], dict):
            return StereoResponse(status="ready", diagnostics=meta["diagnostics"])
    if status == "error" and set(meta) == {"status", "error"}:
        if isinstance(meta["error"], str) and meta["error"]:
            return StereoResponse(status="error", error=meta["error"])
    raise ProtocolError("invalid stereo response status/fields")
