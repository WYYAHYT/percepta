"""Resident calibrated multi-view models sharing the bounded stereo subprocess lifecycle."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .base import BackendRequestError
from .da3_adapter import validate_image_shape
from .foundation_stereo_backend import FoundationStereoBackend, _DeadlinePipe
from .multiview_protocol import (
    MAX_RESPONSE_BYTES,
    MultiViewRequest,
    MultiViewResponse,
    decode_response,
    encode_request,
)
from .stereo_protocol import ProtocolError, write_frame


class MultiViewBackend(FoundationStereoBackend):
    """Three calibrated RGB views to resident-model metric depth.

    Args:
        name: Deployed model name for diagnostics.
        backend_factory: Trusted module:class adapter entry point.
        backend_args: Trusted adapter constructor keywords.
        runtime: Python, GPU, HF cache and deadline settings for the existing subprocess lifecycle.

    Example:
        >>> backend = MultiViewBackend(name="da3", backend_factory="percepta.backends.da3_adapter:DA3Adapter", backend_args={})
        >>> backend.describe()["state"]
        'new'
        >>> backend.close()
    """

    worker_module = "percepta.backends.multiview_worker"
    response_limit = MAX_RESPONSE_BYTES
    response_decoder = staticmethod(decode_response)

    def __init__(self, *, name: str, backend_factory: str, backend_args: dict[str, Any], **runtime) -> None:
        super().__init__(backend_factory=backend_factory, backend_args=backend_args, **runtime)
        self.name = name

    @staticmethod
    def shutdown_payload() -> bytes:
        """Encode a multi-view worker shutdown message.

        Returns:
            Protocol control payload.

        Example:
            >>> bool(MultiViewBackend.shutdown_payload())
            True
        """
        return encode_request(MultiViewRequest(op="shutdown"))

    def infer(self, images: np.ndarray, intrinsics: np.ndarray, T_world_cam: np.ndarray) -> MultiViewResponse:
        """Infer metric optical-axis depth; p_world = T_world_cam @ p_cam_h.

        Args:
            images: Uint8 RGB (3,H,W,3), in matching calibration order.
            intrinsics: Pixel intrinsics (3,3,3).
            T_world_cam: Camera-to-world matrices (3,4,4), translations in metres.

        Returns:
            Depth and optional confidence on the original pixel grid, plus model diagnostics.

        Raises:
            BackendRequestError: Invalid RGB/calibration input.
            BackendUnavailable: The resident worker failed or exceeded its deadline.

        Example:
            >>> result = backend.infer(images, K, T_world_cam)  # doctest: +SKIP
            >>> result.depth.shape, result.depth.dtype.name  # doctest: +SKIP
            ((3, 480, 640), 'float32')
        """
        with self._lock:
            try:
                payload = encode_request(MultiViewRequest(images=images, intrinsics=intrinsics, T_world_cam=T_world_cam))
                if self.name == "da3":
                    validate_image_shape(images.shape[1:3], self.worker_config["backend_args"].get("process_res", 504))
            except (ProtocolError, ValueError) as exc:
                raise BackendRequestError(str(exc)) from exc
            self._start_locked()
            deadline = time.monotonic() + self.timeout_s
            try:
                write_frame(_DeadlinePipe(self._process.stdin.fileno(), deadline, len(payload)), payload)
                response = self._receive(deadline)
                if response.status != "ok" or response.depth.shape != images.shape[:3]:
                    raise ProtocolError("expected metric depth matching the original input grid")
                self._info = response.diagnostics
                return response
            except Exception as exc:
                raise self._fail(exc) from exc
