"""Reference model adapter: the smallest legal backend, and a worked example.

Two purposes, both practical:

1. **An executable spec for adapter authors.** §8 of ``docs/model-isolation.md`` asks every new
   model to ship a fake backend so its logic is testable on a GPU-less machine. This is that
   pattern, written out: only ``numpy`` + ``pydantic``, no framework, no weights.
2. **A real end-to-end probe of the subprocess boundary.** Because this adapter works in a
   virtualenv that contains nothing but those two packages, using it in a worker test proves
   the isolation boundary is genuinely thin — the worker does not secretly need percepta's
   full dependency set (pyyaml, pillow, fastapi…).

It segments by a per-channel threshold, which is exact by construction and therefore
assertable to the pixel.

Import cost: numpy + pydantic only. Never import torch here.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..schema import Prompt

__all__ = ["ThresholdAdapter"]


class ThresholdAdapter:
    """Mark pixels whose red channel exceeds ``threshold``.

    Implements :class:`percepta.backends.base.ModelBackend`:

    * ``set_image`` remembers the frame (the "expensive" step in a real model);
    * ``segment`` thresholds it and returns one instance per prompt, so the same frame can be
      reused across prompts exactly like SAM3's ``set_image``/``set_text_prompt`` flow.
    """

    name = "threshold"

    def __init__(self, threshold: int = 128, channel: int = 0, **_: Any) -> None:
        self.threshold = int(threshold)
        self.channel = int(channel)
        if not 0 <= self.channel <= 2:
            raise ValueError(f"channel must be 0, 1 or 2, got {channel}")
        self._image: np.ndarray | None = None

    # -- ModelBackend ----------------------------------------------------

    def set_image(self, image_rgb: np.ndarray, depth: np.ndarray | None = None) -> None:
        image = np.asarray(image_rgb)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"expected RGB (H, W, 3), got {image.shape}")
        self._image = image

    def segment(self, prompt: Prompt) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
        if self._image is None:
            raise ValueError("set_image() must be called before segment()")
        mask = self._image[:, :, self.channel] > self.threshold
        if not mask.any():
            return np.zeros((0, 1, 1), dtype=bool), None, np.zeros(0, dtype=np.float32)
        rows, cols = np.nonzero(mask)
        box = np.array(
            [[cols.min(), rows.min(), cols.max() + 1, rows.max() + 1]], dtype=np.float32
        )
        # Score is exact here (a threshold either matched or it did not), which keeps tests
        # deterministic instead of asserting on a model's confidence noise.
        return mask[None, ...], box, np.array([1.0], dtype=np.float32)

    def close(self) -> None:
        self._image = None

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "threshold": self.threshold,
            "channel": self.channel,
            "loaded": self._image is not None,
        }
