"""Payload helpers shared by the HTTP contract tests (spec §3.5/§3.6, S1/S4)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from percepta.masks import encode_image
from percepta.schema import Calibration

ROOT = Path(__file__).resolve().parents[2]
CI_CONFIG = ROOT / "configs" / "ci.yaml"
"""The weight-free CI config (decision D9): ``fake`` enabled, ``sam3`` disabled."""


def image_payload(image: np.ndarray, camera: str) -> dict[str, Any]:
    """PNG-encode an RGB array as the JSON form of :class:`~percepta.schema.ImageRef`."""
    return encode_image(image, encoding="png", camera=camera).model_dump(mode="json")


def calibration_payload(calibration: Calibration) -> dict[str, Any]:
    return calibration.model_dump(mode="json")


def perceive_payload(
    images: dict[str, np.ndarray],
    prompts: list[dict[str, Any]],
    calibration: dict[str, Calibration] | None = None,
    *,
    tools: list[str] | None = None,
    fuse: bool = False,
    request_id: str = "req-contract",
    context: dict[str, Any] | None = None,
    masks: str = "rle",
) -> dict[str, Any]:
    """Build a ``POST /v1/perceive`` body around synthetic images.

    ``tools``/``fuse`` are pinned per test so contract assertions never depend on whether the
    optional geometry layer happens to be present in the workspace.
    """
    options: dict[str, Any] = {"fuse": fuse, "masks": masks}
    if tools is not None:
        options["tools"] = tools
    return {
        "request_id": request_id,
        "images": {camera: image_payload(image, camera) for camera, image in images.items()},
        "calibration": {
            camera: calibration_payload(calib) for camera, calib in (calibration or {}).items()
        },
        "prompts": prompts,
        "context": dict(context or {}),
        "options": options,
    }
