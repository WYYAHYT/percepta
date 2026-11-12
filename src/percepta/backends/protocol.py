"""Binary wire protocol shared by the subprocess backend and its worker.

Why a binary protocol instead of JSON: a 480x640x3 uint8 frame is 921 KB of raw bytes but
10–20 MB as JSON text. The frame is therefore length-prefixed binary; only the *prompt* and
*status* travel as JSON, because they are tiny and benefit from being debuggable.

All integers are little-endian ``uint32``/``float32``. Every frame starts with a magic so a
desynchronised stream (worker crashed mid-write, or a stray print() on stdout) is detected
immediately instead of being decoded as garbage pixels.

Keep this module dependency-free (stdlib + numpy only): the worker imports it from inside
the isolated virtualenv, which may have a completely different numpy/torch version.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass

import numpy as np

__all__ = [
    "REQUEST_MAGIC",
    "RESPONSE_MAGIC",
    "PROTOCOL_VERSION",
    "ProtocolError",
    "RequestFrame",
    "ResponseFrame",
    "encode_request",
    "decode_request",
    "encode_response",
    "decode_response",
    "read_frame",
    "write_frame",
]

REQUEST_MAGIC = 0x50524350  # 'PRCP'
RESPONSE_MAGIC = 0x50524352  # 'PRCR'
PROTOCOL_VERSION = 1

_U32 = struct.Struct("<I")
_I32 = struct.Struct("<i")
_F32 = struct.Struct("<f")


class ProtocolError(RuntimeError):
    """The peer sent something that does not match the protocol."""


@dataclass
class RequestFrame:
    """One ``segment`` call: a prepared image, an optional depth map, one prompt."""

    prompt_json: str
    image: np.ndarray | None = None  # uint8 (H, W, 3); None = reuse the frame in the worker
    depth: np.ndarray | None = None  # float32 (H, W), optional
    shutdown: bool = False


@dataclass
class ResponseFrame:
    """One ``segment`` result, or an error."""

    status: str  # "ok" | "error" | "ready" (startup handshake only)
    error: str | None = None
    error_kind: str | None = None  # "unavailable" | "invalid_request"
    masks: np.ndarray | None = None  # bool (N, H, W)
    boxes: np.ndarray | None = None  # float32 (N, 4)
    scores: np.ndarray | None = None  # float32 (N,)


# ---------------------------------------------------------------------------
# encoding
# ---------------------------------------------------------------------------


def encode_request(frame: RequestFrame) -> bytes:
    """Serialize a request, sending RGB/depth only when provided.

    Args:
        frame: Prompt text, optional RGB/depth arrays, or shutdown control message.

    Returns:
        A v1 binary request payload without its transport length prefix.

    Raises:
        ProtocolError: The fields or array dimensions are inconsistent.

    Example:
        >>> decode_request(encode_request(RequestFrame("{}"))).prompt_json
        '{}'
    """
    if not isinstance(frame.prompt_json, str) or not isinstance(frame.shutdown, bool):
        raise ProtocolError("request prompt must be text and shutdown must be boolean")
    if frame.shutdown and (frame.prompt_json or frame.image is not None or frame.depth is not None):
        raise ProtocolError("shutdown cannot contain prompt/image/depth data")
    try:
        prompt_bytes = frame.prompt_json.encode("utf-8")
    except UnicodeError as exc:
        raise ProtocolError(f"invalid prompt UTF-8: {exc}") from exc
    parts = [
        _U32.pack(REQUEST_MAGIC),
        _U32.pack(PROTOCOL_VERSION),
        _U32.pack(1 if frame.shutdown else 0),
        _U32.pack(len(prompt_bytes)),
        prompt_bytes,
    ]
    if frame.image is not None:
        image = np.ascontiguousarray(frame.image, dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ProtocolError(f"image must be (H, W, 3) uint8, got {image.shape}")
        height, width = int(image.shape[0]), int(image.shape[1])
        if height <= 0 or width <= 0:
            raise ProtocolError("image dimensions must be positive")
        parts += [
            _U32.pack(1),  # has_image
            _U32.pack(height),
            _U32.pack(width),
            image.tobytes(),
        ]
    else:
        parts.append(_U32.pack(0))  # has_image = 0 -> worker reuses its prepared frame

    if frame.depth is not None:
        depth = np.ascontiguousarray(frame.depth, dtype="<f4")
        if depth.ndim != 2:
            raise ProtocolError(f"depth must be (H, W), got {depth.shape}")
        if min(depth.shape) <= 0 or frame.image is not None and depth.shape != image.shape[:2]:
            raise ProtocolError("depth dimensions must be positive and match the image")
        parts += [
            _U32.pack(1),  # has_depth
            _U32.pack(int(depth.shape[0])),
            _U32.pack(int(depth.shape[1])),
            depth.tobytes(),
        ]
    else:
        parts.append(_U32.pack(0))
    return b"".join(parts)


def decode_request(payload: bytes) -> RequestFrame:
    """Decode and consume one complete v1 request payload.

    Args:
        payload: Binary request without its transport length prefix.

    Returns:
        The decoded prompt, RGB/depth arrays, and shutdown flag.

    Raises:
        ProtocolError: Fields, dimensions, UTF-8, or the exact byte length are invalid.

    Example:
        >>> decode_request(encode_request(RequestFrame("{}"))).image is None
        True
    """
    view = memoryview(payload)
    offset = 0

    def take_u32() -> int:
        nonlocal offset
        if offset + 4 > len(payload):
            raise ProtocolError("truncated request frame")
        (value,) = _U32.unpack_from(view, offset)
        offset += 4
        return int(value)

    if take_u32() != REQUEST_MAGIC:
        raise ProtocolError("bad request magic (stream desynchronised?)")
    version = take_u32()
    if version != PROTOCOL_VERSION:
        raise ProtocolError(f"unsupported protocol version {version}")
    shutdown = _flag(take_u32(), "shutdown")
    prompt_len = take_u32()
    if offset + prompt_len > len(payload):
        raise ProtocolError("truncated prompt payload")
    try:
        prompt_json = bytes(view[offset : offset + prompt_len]).decode("utf-8")
    except UnicodeError as exc:
        raise ProtocolError(f"invalid prompt UTF-8: {exc}") from exc
    offset += prompt_len

    image = None
    if _flag(take_u32(), "has_image"):
        height, width = take_u32(), take_u32()
        if height <= 0 or width <= 0:
            raise ProtocolError("image dimensions must be positive")
        size = height * width * 3
        if offset + size > len(payload):
            raise ProtocolError("truncated image payload")
        image = np.frombuffer(bytes(view[offset : offset + size]), dtype=np.uint8).reshape(
            height, width, 3
        )
        offset += size

    depth = None
    if _flag(take_u32(), "has_depth"):
        height, width = take_u32(), take_u32()
        if height <= 0 or width <= 0 or image is not None and (height, width) != image.shape[:2]:
            raise ProtocolError("depth dimensions must be positive and match the image")
        size = height * width * 4
        if offset + size > len(payload):
            raise ProtocolError("truncated depth payload")
        depth = (
            np.frombuffer(bytes(view[offset : offset + size]), dtype="<f4")
            .reshape(height, width)
            .copy()
        )
        offset += size

    if offset != len(payload):
        raise ProtocolError("trailing bytes in request frame")
    if shutdown and (prompt_json or image is not None or depth is not None):
        raise ProtocolError("shutdown cannot contain prompt/image/depth data")
    return RequestFrame(prompt_json=prompt_json, image=image, depth=depth, shutdown=shutdown)


def _flag(value: int, name: str) -> bool:
    if value not in (0, 1):
        raise ProtocolError(f"{name} must be 0 or 1")
    return bool(value)


def _response_fields(status, error, error_kind) -> None:
    if status not in ("ok", "error", "ready"):
        raise ProtocolError("invalid response status")
    if status == "error":
        if not isinstance(error, str) or not error or error_kind not in (None, "unavailable", "invalid_request"):
            raise ProtocolError("invalid response error/error_kind")
    elif error is not None or error_kind is not None:
        raise ProtocolError("successful/control response cannot contain an error")


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError(f"duplicate JSON field {key}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ProtocolError(f"non-finite JSON value {value}")


def _encode_json_blob(text: str) -> bytes:
    raw = text.encode("utf-8")
    return _U32.pack(len(raw)) + raw


def encode_response(frame: ResponseFrame) -> bytes:
    """Serialize a validated result, error, or startup handshake.

    Args:
        frame: Response fields with optional boolean masks, pixel boxes, and scores.

    Returns:
        A v1 payload; masks remain binary bytes and pixel boxes use exclusive x1/y1.

    Raises:
        ProtocolError: Status/error fields or arrays are inconsistent or non-finite.

    Example:
        >>> decode_response(encode_response(ResponseFrame("ok"))).masks is None
        True
    """
    _response_fields(frame.status, frame.error, frame.error_kind)
    if frame.status != "ok" and any(value is not None for value in (frame.masks, frame.boxes, frame.scores)):
        raise ProtocolError("control/error response cannot contain observations")
    masks = frame.masks
    if masks is not None and (not isinstance(masks, np.ndarray) or masks.ndim != 3 or masks.dtype != bool):
        raise ProtocolError("masks must be boolean (N, H, W)")
    count = 0 if masks is None else masks.shape[0]
    if masks is not None and min(masks.shape[1:]) <= 0:
        raise ProtocolError("mask dimensions must be positive")
    try:
        scores = np.zeros(count, dtype="<f4") if frame.scores is None else np.asarray(frame.scores, dtype="<f4")
        boxes = None if frame.boxes is None else np.asarray(frame.boxes, dtype="<f4")
    except (ValueError, TypeError, OverflowError) as exc:
        raise ProtocolError(f"invalid response arrays: {exc}") from exc
    if scores.shape != (count,) or not np.isfinite(scores).all():
        raise ProtocolError("scores must be finite (N,) matching the mask count")
    if boxes is not None and (boxes.shape != (count, 4) or not np.isfinite(boxes).all()):
        raise ProtocolError("boxes must be finite (N, 4) matching the mask count")

    status_json = json.dumps(
        {"status": frame.status, "error": frame.error, "error_kind": frame.error_kind}
    )
    parts = [
        _U32.pack(RESPONSE_MAGIC),
        _U32.pack(PROTOCOL_VERSION),
        _encode_json_blob(status_json),
    ]
    if count == 0:
        parts.append(_U32.pack(0))  # n_obs = 0
        return b"".join(parts)

    masks = np.ascontiguousarray(masks)
    count, height, width = (int(v) for v in masks.shape)

    parts.append(_U32.pack(count))
    parts += [_U32.pack(height), _U32.pack(width)]
    parts.append(np.ascontiguousarray(masks).tobytes())  # N*H*W bool bytes
    parts.append(np.ascontiguousarray(scores, dtype="<f4").tobytes())
    if boxes is None:
        parts.append(_U32.pack(0))
    else:
        parts.append(_U32.pack(1))
        parts.append(np.ascontiguousarray(boxes, dtype="<f4").tobytes())
    return b"".join(parts)


def decode_response(payload: bytes, *, allow_ready: bool = False) -> ResponseFrame:
    """Decode one complete result/error frame, allowing ready only at startup.

    Args:
        payload: Binary response without its transport length prefix.
        allow_ready: Accept the ready control status during a worker startup handshake.

    Returns:
        Validated response fields and optional masks, pixel boxes, and finite scores.

    Raises:
        ProtocolError: Metadata, status, arrays, or the exact byte length are invalid.

    Example:
        >>> decode_response(encode_response(ResponseFrame("ready")), allow_ready=True).status
        'ready'
    """

    view = memoryview(payload)
    offset = 0

    def take_u32() -> int:
        nonlocal offset
        if offset + 4 > len(payload):
            raise ProtocolError("truncated response frame")
        (value,) = _U32.unpack_from(view, offset)
        offset += 4
        return int(value)

    if take_u32() != RESPONSE_MAGIC:
        raise ProtocolError("bad response magic (worker crashed or wrote junk?)")
    version = take_u32()
    if version != PROTOCOL_VERSION:
        raise ProtocolError(f"unsupported protocol version {version}")
    status_len = take_u32()
    if offset + status_len > len(payload):
        raise ProtocolError("truncated status payload")
    try:
        meta = json.loads(
            bytes(view[offset : offset + status_len]).decode("utf-8"),
            object_pairs_hook=_json_object, parse_constant=_invalid_constant,
        )
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ProtocolError(f"invalid response JSON: {exc}") from exc
    if not isinstance(meta, dict) or "status" not in meta or set(meta) - {"status", "error", "error_kind"}:
        raise ProtocolError("invalid response metadata fields")
    status, error, error_kind = meta["status"], meta.get("error"), meta.get("error_kind")
    _response_fields(status, error, error_kind)
    if status == "ready" and not allow_ready:
        raise ProtocolError("ready response is only valid during startup")
    offset += status_len

    count = take_u32()
    if status != "ok" and count:
        raise ProtocolError("control/error response cannot contain observations")
    if count == 0:
        if offset != len(payload):
            raise ProtocolError("trailing bytes in empty response frame")
        return ResponseFrame(
            status=status, error=error, error_kind=error_kind,
        )

    height, width = take_u32(), take_u32()
    if height <= 0 or width <= 0:
        raise ProtocolError("mask dimensions must be positive")
    size = count * height * width
    if offset + size > len(payload):
        raise ProtocolError("truncated mask payload")
    raw_masks = np.frombuffer(view[offset : offset + size], dtype=np.uint8)
    if np.any(raw_masks > 1):
        raise ProtocolError("mask bytes must be 0 or 1")
    masks = raw_masks.reshape(count, height, width).astype(bool)
    offset += size

    if offset + count * 4 > len(payload):
        raise ProtocolError("truncated score payload")
    scores = np.frombuffer(bytes(view[offset : offset + count * 4]), dtype="<f4").copy()
    if not np.isfinite(scores).all():
        raise ProtocolError("scores must be finite")
    offset += count * 4

    boxes = None
    if _flag(take_u32(), "has_boxes"):
        size = count * 4 * 4
        if offset + size > len(payload):
            raise ProtocolError("truncated box payload")
        boxes = np.frombuffer(bytes(view[offset : offset + size]), dtype="<f4").reshape(count, 4)
        if not np.isfinite(boxes).all():
            raise ProtocolError("boxes must be finite")
        offset += size

    if offset != len(payload):
        raise ProtocolError("trailing bytes in response frame")
    return ResponseFrame(
        status=status, error=error, error_kind=error_kind,
        masks=masks,
        boxes=None if boxes is None else boxes.copy(),
        scores=scores,
    )


# ---------------------------------------------------------------------------
# stream framing
# ---------------------------------------------------------------------------


def write_frame(stream, payload: bytes) -> None:
    """Write ``<len><payload>`` and flush, so the peer sees it immediately."""
    stream.write(_U32.pack(len(payload)))
    stream.write(payload)
    stream.flush()


def read_frame(stream) -> bytes | None:
    """Read one length-prefixed payload, or ``None`` at clean EOF (worker exited)."""
    header = stream.read(4)
    if not header:
        return None
    if len(header) < 4:
        raise ProtocolError("truncated length prefix")
    (length,) = _U32.unpack(header)
    if length > 512 * 1024 * 1024:  # 512 MiB sanity bound
        raise ProtocolError(f"implausible frame length {length}")
    payload = stream.read(length)
    if len(payload) < length:
        raise ProtocolError("truncated frame payload")
    return payload
