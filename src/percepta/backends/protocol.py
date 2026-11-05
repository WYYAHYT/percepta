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

    status: str  # "ok" | "error"
    error: str | None = None
    error_kind: str | None = None  # "unavailable" | "invalid_request"
    masks: np.ndarray | None = None  # bool (N, H, W)
    boxes: np.ndarray | None = None  # float32 (N, 4)
    scores: np.ndarray | None = None  # float32 (N,)


# ---------------------------------------------------------------------------
# encoding
# ---------------------------------------------------------------------------


def encode_request(frame: RequestFrame) -> bytes:
    """Serialise a request. The image is sent only when the caller provides one."""
    prompt_bytes = frame.prompt_json.encode("utf-8")
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
    """Inverse of :func:`encode_request`; raises :class:`ProtocolError` on any mismatch."""
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
    shutdown = bool(take_u32())
    prompt_len = take_u32()
    if offset + prompt_len > len(payload):
        raise ProtocolError("truncated prompt payload")
    prompt_json = bytes(view[offset : offset + prompt_len]).decode("utf-8")
    offset += prompt_len

    image = None
    if take_u32():
        height, width = take_u32(), take_u32()
        size = height * width * 3
        if offset + size > len(payload):
            raise ProtocolError("truncated image payload")
        image = np.frombuffer(bytes(view[offset : offset + size]), dtype=np.uint8).reshape(
            height, width, 3
        )
        offset += size

    depth = None
    if take_u32():
        height, width = take_u32(), take_u32()
        size = height * width * 4
        if offset + size > len(payload):
            raise ProtocolError("truncated depth payload")
        depth = (
            np.frombuffer(bytes(view[offset : offset + size]), dtype="<f4")
            .reshape(height, width)
            .copy()
        )
        offset += size

    return RequestFrame(prompt_json=prompt_json, image=image, depth=depth, shutdown=shutdown)


def _encode_json_blob(text: str) -> bytes:
    raw = text.encode("utf-8")
    return _U32.pack(len(raw)) + raw


def encode_response(frame: ResponseFrame) -> bytes:
    """Serialise a response. Masks travel as raw bytes; RLE stays a parent-side concern."""
    import json

    status_json = json.dumps(
        {"status": frame.status, "error": frame.error, "error_kind": frame.error_kind}
    )
    parts = [
        _U32.pack(RESPONSE_MAGIC),
        _U32.pack(PROTOCOL_VERSION),
        _encode_json_blob(status_json),
    ]
    if frame.masks is None or frame.masks.size == 0:
        parts.append(_U32.pack(0))  # n_obs = 0
        return b"".join(parts)

    masks = np.ascontiguousarray(frame.masks, dtype=bool)
    if masks.ndim != 3:
        raise ProtocolError(f"masks must be (N, H, W), got {masks.shape}")
    count, height, width = (int(v) for v in masks.shape)
    boxes = frame.boxes
    scores = frame.scores if frame.scores is not None else np.zeros(count, dtype="<f4")
    if scores.shape[0] != count:
        raise ProtocolError("scores length does not match mask count")

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


def decode_response(payload: bytes) -> ResponseFrame:
    """Inverse of :func:`encode_response`."""
    import json

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
    status = json.loads(bytes(view[offset : offset + status_len]).decode("utf-8"))
    offset += status_len

    count = take_u32()
    if count == 0:
        return ResponseFrame(
            status=status.get("status", "ok"),
            error=status.get("error"),
            error_kind=status.get("error_kind"),
        )

    height, width = take_u32(), take_u32()
    size = count * height * width
    if offset + size > len(payload):
        raise ProtocolError("truncated mask payload")
    masks = np.frombuffer(bytes(view[offset : offset + size]), dtype=np.uint8).reshape(
        count, height, width
    ).astype(bool)
    offset += size

    if offset + count * 4 > len(payload):
        raise ProtocolError("truncated score payload")
    scores = np.frombuffer(bytes(view[offset : offset + count * 4]), dtype="<f4").copy()
    offset += count * 4

    boxes = None
    if take_u32():
        size = count * 4 * 4
        if offset + size > len(payload):
            raise ProtocolError("truncated box payload")
        boxes = np.frombuffer(bytes(view[offset : offset + size]), dtype="<f4").reshape(count, 4)
        offset += size

    return ResponseFrame(
        status=status.get("status", "ok"),
        error=status.get("error"),
        error_kind=status.get("error_kind"),
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
