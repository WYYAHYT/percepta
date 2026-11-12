"""Actual CPU worker transport, metric-unit boundaries and bounded failure cleanup."""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from percepta.backends.base import BackendRequestError, BackendUnavailable
from percepta.backends.multiview_backend import MultiViewBackend
from percepta.backends.multiview_protocol import (
    MAX_RESPONSE_BYTES,
    MultiViewRequest,
    MultiViewResponse,
    decode_request,
    decode_response,
    encode_request,
    encode_response,
)
from percepta.backends.stereo_protocol import HEADER, ProtocolError


def scene():
    images = np.arange(3 * 9 * 13 * 3, dtype=np.uint8).reshape(3, 9, 13, 3)
    K = np.tile([[20., 0, 6], [0, 20, 4], [0, 0, 1]], (3, 1, 1))
    transforms = np.tile(np.eye(4), (3, 1, 1))
    transforms[1, 0, 3], transforms[2, 1, 3] = 0.1, 0.2
    return images[:, :, ::-1], K, transforms


def worker(monkeypatch, mode="ok", **kwargs):
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, (str(Path(__file__).resolve().parents[2]), os.environ.get("PYTHONPATH")))))
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    return MultiViewBackend(name="cpu-test", python=sys.executable,
                            backend_factory="tests.unit.multiview_support:FakeMultiViewAdapter", backend_args={"mode": mode},
                            startup_timeout_s=5, timeout_s=kwargs.get("timeout_s", 5))


def test_wire_roundtrip_and_control_frames():
    images, K, T = scene()
    packet = encode_request(MultiViewRequest(images=images, intrinsics=K, T_world_cam=T))
    request = decode_request(packet)
    np.testing.assert_array_equal(request.images, images)
    np.testing.assert_array_equal(request.intrinsics, K)
    np.testing.assert_array_equal(request.T_world_cam, T)
    assert request.images.flags.writeable
    assert decode_request(encode_request(MultiViewRequest(op="shutdown"))).op == "shutdown"
    depth, confidence = np.ones(images.shape[:3], np.float32), np.full(images.shape[:3], 4, np.float32)
    result = decode_response(encode_response(MultiViewResponse("ok", depth, confidence, {"views": 3})))
    np.testing.assert_array_equal(result.depth, depth)
    np.testing.assert_array_equal(result.confidence, confidence)
    assert result.diagnostics == {"views": 3}
    assert decode_response(encode_response(MultiViewResponse("ok", depth))).confidence is None
    for status, error in (("ready", None), ("error", "missing checkpoint")):
        result = decode_response(encode_response(MultiViewResponse(status, error=error)))
        assert result.status == status and result.error == error


def test_every_truncation_and_extra_bytes_are_rejected():
    images, K, T = scene()
    for packet, decoder in (
        (encode_request(MultiViewRequest(images=images, intrinsics=K, T_world_cam=T)), decode_request),
        (encode_response(MultiViewResponse("ok", np.ones(images.shape[:3], np.float32))), decode_response),
    ):
        for cut in range(len(packet)):
            with pytest.raises(ProtocolError):
                decoder(packet[:cut])
        with pytest.raises(ProtocolError):
            decoder(packet + b"trailing bytes")


def test_units_dtype_nonfinite_and_dimensions_cannot_be_guessed():
    images, K, T = scene()
    with pytest.raises(ProtocolError):
        encode_request(MultiViewRequest(images=images[:2], intrinsics=K, T_world_cam=T))
    with pytest.raises(ProtocolError):
        encode_request(MultiViewRequest(images=images, intrinsics=K, T_world_cam=np.zeros_like(T)))
    with pytest.raises(ProtocolError):
        encode_request(MultiViewRequest(images=images, intrinsics=K, T_world_cam=np.tile(np.eye(4), (3, 1, 1))))
    for value in (np.ones(images.shape[:3], np.float64), np.full(images.shape[:3], np.nan, np.float32),
                  np.full(images.shape[:3], -1, np.float32)):
        with pytest.raises(ProtocolError):
            encode_response(MultiViewResponse("ok", value))
    payload = encode_response(MultiViewResponse("ok", np.ones(images.shape[:3], np.float32)))
    magic, version, length, height, width = HEADER.unpack_from(payload)
    meta = json.loads(payload[HEADER.size:HEADER.size + length])
    meta["depth_unit"] = "relative"
    blob = json.dumps(meta).encode()
    with pytest.raises(ProtocolError):
        decode_response(HEADER.pack(magic, version, len(blob), height, width) + blob + payload[HEADER.size + length:])
    assert MAX_RESPONSE_BYTES > len(payload)


def test_real_worker_reuses_model_and_invalid_request_does_not_poison_it(monkeypatch):
    backend = worker(monkeypatch)
    images, K, T = scene()
    try:
        first = backend.infer(images, K, T)
        pid = backend.describe()["pid"]
        assert first.depth.dtype == np.float32 and first.depth.shape == images.shape[:3]
        assert backend.describe()["calls"] == 1
        with pytest.raises(BackendRequestError):
            backend.infer(images.astype(float), K, T)
        coincident = T.copy()
        coincident[1] = coincident[0]
        with pytest.raises(BackendRequestError):
            backend.infer(images, K, coincident)
        second = backend.infer(images, K, T)
        np.testing.assert_array_equal(first.depth, second.depth)
        assert backend.describe()["pid"] == pid and backend.describe()["calls"] == 2
    finally:
        backend.close()
    assert backend._process is None


@pytest.mark.parametrize("mode", ["crash", "relative", "nan", "hang", "startup_error"])
def test_worker_failure_is_terminal_and_bounded(monkeypatch, mode):
    backend = worker(monkeypatch, mode, timeout_s=0.3)
    started = time.monotonic()
    try:
        with pytest.raises(BackendUnavailable):
            backend.infer(*scene())
        assert time.monotonic() - started < 7
        assert backend._process is None
        with pytest.raises(BackendUnavailable):
            backend.infer(*scene())
        assert backend.describe()["state"] == "failed"
    finally:
        backend.close()


def test_da3_rejects_thin_image_without_poisoning_resident_worker(monkeypatch):
    backend = worker(monkeypatch)
    backend.name = "da3"
    images, K, T = scene()
    try:
        backend.infer(images, K, T)
        pid = backend.describe()["pid"]
        with pytest.raises(BackendRequestError, match="patch"):
            backend.infer(np.zeros((3, 1, 1000, 3), np.uint8), K, T)
        assert backend.describe()["state"] == "ready" and backend.describe()["calls"] == 1
        backend.infer(images, K, T)
        assert backend.describe()["pid"] == pid and backend.describe()["calls"] == 2
    finally:
        backend.close()
