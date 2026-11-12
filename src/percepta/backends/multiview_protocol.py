"""Bounded three-view RGB/calibration RPC, returning metric optical-axis depth on the input grid."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .stereo_protocol import (
    HEADER,
    MAX_METADATA_BYTES,
    MAX_PIXELS,
    ProtocolError,
    _exact_size,
    _pack,
    _pixels,
    _unpack,
)

REQUEST_MAGIC = b"MVPQ"
RESPONSE_MAGIC = b"MVPR"
MAX_REQUEST_BYTES = HEADER.size + MAX_METADATA_BYTES + 9 * MAX_PIXELS
MAX_RESPONSE_BYTES = HEADER.size + MAX_METADATA_BYTES + 24 * MAX_PIXELS


@dataclass
class MultiViewRequest:
    """Three RGB views and OpenCV camera-to-world calibration.

    Args:
        op: Infer or shutdown.
        images: Uint8 RGB array (3,H,W,3).
        intrinsics: Pixel intrinsics (3,3,3).
        T_world_cam: Rigid camera-to-world matrices (3,4,4), in metres.

    Example:
        >>> MultiViewRequest(op="shutdown").op
        'shutdown'
    """

    op: str = "infer"
    images: np.ndarray | None = None
    intrinsics: np.ndarray | None = None
    T_world_cam: np.ndarray | None = None


@dataclass
class MultiViewResponse:
    """Metric camera +Z depth, optional model confidence and JSON diagnostics.

    Args:
        status: Ready, ok or error.
        depth: Float32 (3,H,W); zero marks invalid depth.
        confidence: Optional finite nonnegative float32 array on the same grid.
        diagnostics: Model and runtime metadata.
        error: Failure message for an error response.

    Example:
        >>> MultiViewResponse("ready").status
        'ready'
    """

    status: str
    depth: np.ndarray | None = None
    confidence: np.ndarray | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


def _calibration(intrinsics, transforms) -> tuple[np.ndarray, np.ndarray]:
    try:
        K, T = np.asarray(intrinsics, dtype=np.float64), np.asarray(transforms, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProtocolError("multiview calibration must be numeric") from exc
    if K.shape != (3, 3, 3) or T.shape != (3, 4, 4) or not np.isfinite(K).all() or not np.isfinite(T).all():
        raise ProtocolError("expected finite three-view K (3,3,3) and T_world_cam (3,4,4)")
    if np.any(K[:, (0, 1), (0, 1)] <= 0) or not np.allclose(K[:, 2], [0, 0, 1]):
        raise ProtocolError("invalid pinhole intrinsics")
    R = T[:, :3, :3]
    if (not np.allclose(T[:, 3], [0, 0, 0, 1]) or not np.allclose(R @ R.transpose(0, 2, 1), np.eye(3), atol=1e-5)
            or not np.allclose(np.linalg.det(R), 1, atol=1e-5)):
        raise ProtocolError("T_world_cam must contain proper rigid camera-to-world transforms")
    if any(np.linalg.norm(T[a, :3, 3] - T[b, :3, 3]) <= 1e-6 for a, b in ((0, 1), (0, 2), (1, 2))):
        raise ProtocolError("metric multiview depth requires distinct camera centres")
    return K, T


def encode_request(frame: MultiViewRequest) -> bytes:
    """Encode RGB bytes with calibration; p_world = T_world_cam @ p_cam_h.

    Args:
        frame: Exactly three equal-size images with rigid calibration, or shutdown.

    Returns:
        Validated protocol bytes; no Python objects are serialized.

    Example:
        >>> decode_request(encode_request(MultiViewRequest(op="shutdown"))).op
        'shutdown'
    """
    if frame.op == "shutdown":
        if any(value is not None for value in (frame.images, frame.intrinsics, frame.T_world_cam)):
            raise ProtocolError("shutdown cannot contain images/calibration")
        return _pack(REQUEST_MAGIC, {"op": "shutdown"})
    images = frame.images
    if (frame.op != "infer" or not isinstance(images, np.ndarray) or images.dtype != np.uint8
            or images.ndim != 4 or images.shape[0] != 3 or images.shape[-1] != 3):
        raise ProtocolError("expected three uint8 RGB images (3,H,W,3)")
    height, width = images.shape[1:3]
    _pixels(height, width)
    K, T = _calibration(frame.intrinsics, frame.T_world_cam)
    return _pack(REQUEST_MAGIC, {"op": "infer", "K": K.tolist(), "T_world_cam": T.tolist()}, height, width, images.tobytes())


def decode_request(payload: bytes) -> MultiViewRequest:
    """Decode bounded images and calibrated OpenCV camera-to-world transforms.

    Args:
        payload: Bytes produced by encode_request.

    Returns:
        Writable RGB and float64 calibration arrays.

    Example:
        >>> decode_request(encode_request(MultiViewRequest(op="shutdown"))).images is None
        True
    """
    meta, height, width, raw = _unpack(payload, REQUEST_MAGIC, MAX_REQUEST_BYTES)
    if meta == {"op": "shutdown"} and not (height or width or raw):
        return MultiViewRequest(op="shutdown")
    if set(meta) != {"op", "K", "T_world_cam"} or meta["op"] != "infer":
        raise ProtocolError("invalid multiview operation/fields")
    _exact_size(raw, _pixels(height, width) * 9)
    K, T = _calibration(meta["K"], meta["T_world_cam"])
    images = np.frombuffer(raw, dtype=np.uint8).reshape(3, height, width, 3).copy()
    return MultiViewRequest(images=images, intrinsics=K, T_world_cam=T)


def _depth_array(value, shape=None) -> np.ndarray:
    if (not isinstance(value, np.ndarray) or value.dtype != np.float32 or value.ndim != 3 or value.shape[0] != 3
            or not np.isfinite(value).all() or np.any(value < 0) or (shape is not None and value.shape != shape)):
        raise ProtocolError("depth/confidence must be finite nonnegative float32 (3,H,W) on the input grid")
    _pixels(*value.shape[1:])
    return value


def encode_response(frame: MultiViewResponse) -> bytes:
    """Encode depth measured in metres along each camera's +Z axis.

    Args:
        frame: Result, readiness or error with JSON diagnostics.

    Returns:
        Bounded response payload.

    Example:
        >>> decode_response(encode_response(MultiViewResponse("ready"))).status
        'ready'
    """
    if not isinstance(frame.diagnostics, dict):
        raise ProtocolError("diagnostics must be an object")
    if frame.status == "ok" and frame.error is None:
        depth = _depth_array(frame.depth)
        confidence = None if frame.confidence is None else _depth_array(frame.confidence, depth.shape)
        raw = depth.astype("<f4", copy=False).tobytes()
        if confidence is not None:
            raw += confidence.astype("<f4", copy=False).tobytes()
        return _pack(RESPONSE_MAGIC, {"status": "ok", "depth_unit": "m", "confidence": confidence is not None,
                                     "diagnostics": frame.diagnostics}, *depth.shape[1:], raw)
    if frame.depth is not None or frame.confidence is not None:
        raise ProtocolError("control response cannot contain arrays")
    if frame.status == "ready" and frame.error is None:
        return _pack(RESPONSE_MAGIC, {"status": "ready", "diagnostics": frame.diagnostics})
    if frame.status == "error" and isinstance(frame.error, str) and frame.error:
        return _pack(RESPONSE_MAGIC, {"status": "error", "error": frame.error})
    raise ProtocolError("invalid multiview response")


def decode_response(payload: bytes) -> MultiViewResponse:
    """Decode finite metric depth and preserve optional confidence without rescaling it.

    Args:
        payload: Binary multiview response.

    Returns:
        Three-view response with independent float32 arrays.

    Example:
        >>> decode_response(encode_response(MultiViewResponse("error", error="missing weights"))).error
        'missing weights'
    """
    meta, height, width, raw = _unpack(payload, RESPONSE_MAGIC, MAX_RESPONSE_BYTES)
    if meta.get("status") == "ok":
        if (set(meta) != {"status", "depth_unit", "confidence", "diagnostics"} or meta["depth_unit"] != "m"
                or type(meta["confidence"]) is not bool or not isinstance(meta["diagnostics"], dict)):
            raise ProtocolError("invalid multiview response fields/units")
        size = _pixels(height, width) * 3 * 4
        _exact_size(raw, size * (2 if meta["confidence"] else 1))
        depth = np.frombuffer(raw[:size], dtype="<f4").reshape(3, height, width).astype(np.float32, copy=True)
        confidence = None
        if meta["confidence"]:
            confidence = np.frombuffer(raw[size:], dtype="<f4").reshape(depth.shape).astype(np.float32, copy=True)
            _depth_array(confidence, depth.shape)
        return MultiViewResponse("ok", _depth_array(depth), confidence, meta["diagnostics"])
    if height or width or raw:
        raise ProtocolError("control response contains data")
    if set(meta) == {"status", "diagnostics"} and meta["status"] == "ready" and isinstance(meta["diagnostics"], dict):
        return MultiViewResponse("ready", diagnostics=meta["diagnostics"])
    if set(meta) == {"status", "error"} and meta["status"] == "error" and isinstance(meta["error"], str) and meta["error"]:
        return MultiViewResponse("error", error=meta["error"])
    raise ProtocolError("invalid multiview response")
