"""The ``fake`` tool — pure numpy synthesis, no models, no GPU, no network.

Purpose (spec §5.4): make the *whole* contract testable in CI. It segments by "colour
threshold + connected components", which is exact by construction, so the acceptance
scenarios can assert on areas, boxes and centroids to the pixel.

Prompt handling:

* ``text`` — the first colour word in the noun phrase selects that colour; otherwise the
  tool falls back to "salient blob" mode (anything unlike the background).
* ``box``  — restrict the search to the box (normalised ``[cx, cy, w, h]``, spec §3.2).
* ``point`` — the component containing that pixel (colour sampled at the point).

Scoring is deterministic: ``score = 1 - mean(|pixel - target|) / 255`` over the component,
where ``target`` is the requested colour, or the background in salient mode.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Sequence
from typing import Any

import numpy as np

from ..masks import mask_box_xyxy, mask_centroid, mask_iou
from ..schema import Percept, Prompt, ToolCapabilities, ToolContext
from .base import BaseTool, apply_top_k, make_observation, make_percept

__all__ = ["FakeTool", "label_components"]

# Nominal colours; tolerance absorbs anti-aliasing and lighting jitter in test fixtures.
DEFAULT_COLORS: dict[str, tuple[int, int, int]] = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "orange": (255, 140, 0),
    "purple": (128, 0, 128),
    "gray": (128, 128, 128),
    "grey": (128, 128, 128),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
}


# ---------------------------------------------------------------------------
# Connected components (pure numpy, run-length + union-find)
# ---------------------------------------------------------------------------


def _row_runs(row: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Half-open ``(start, end)`` column intervals of True runs in one row."""
    padded = np.concatenate(([False], np.asarray(row, dtype=bool), [False]))
    edges = np.diff(padded.astype(np.int8))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def label_components(mask: np.ndarray, connectivity: int = 4) -> tuple[np.ndarray, int]:
    """Label 4- or 8-connected components of a bool mask (background label 0).

    Returns ``(labels, count)`` with ``labels`` an ``int32`` array of the input shape and
    component ids ``1..count``. Implemented as row-run extraction plus union-find, which
    keeps it pure numpy while staying fast on test-sized images.
    """
    arr = np.asarray(mask, dtype=bool)
    height, width = arr.shape
    labels = np.zeros((height, width), dtype=np.int32)
    if not arr.any():
        return labels, 0

    parent: list[int] = []

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:  # path compression
            parent[x], x = root, parent[x]
        return root

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    runs: list[tuple[int, int, int, int]] = []  # (row, start, end, run_id)
    prev: list[tuple[int, int, int]] = []  # (start, end, run_id)
    for row in range(height):
        starts, ends = _row_runs(arr[row])
        current: list[tuple[int, int, int]] = []
        for start, end in zip(starts.tolist(), ends.tolist(), strict=False):
            run_id = len(parent)
            parent.append(run_id)
            runs.append((row, start, end, run_id))
            current.append((start, end, run_id))

        # Merge with the previous row's runs: 4-connectivity needs a strict overlap,
        # 8-connectivity also joins diagonal neighbours (touching corners).
        i = j = 0
        while i < len(prev) and j < len(current):
            p_start, p_end, p_id = prev[i]
            c_start, c_end, c_id = current[j]
            if connectivity >= 8:
                overlap = p_start <= c_end and c_start <= p_end
            else:
                overlap = p_start < c_end and c_start < p_end
            if overlap:
                union(p_id, c_id)
                if p_end < c_end:
                    i += 1
                elif c_end < p_end:
                    j += 1
                else:
                    i += 1
                    j += 1
            elif p_end <= c_start:
                i += 1
            else:
                j += 1
        prev = current

    # Compact roots into 1..n in order of first appearance (stable labelling).
    remap: dict[int, int] = {}
    for row, start, end, run_id in runs:
        root = find(run_id)
        label = remap.setdefault(root, len(remap) + 1)
        labels[row, start:end] = label
    return labels, len(remap)


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


def _distance(pixels: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-pixel mean absolute channel distance, ``pixels`` (N, 3), ``target`` (3,)."""
    return np.abs(pixels.astype(np.float32) - target.astype(np.float32)).mean(axis=-1)


class FakeTool(BaseTool):
    name = "fake"
    version = "0.1.0"
    kind = "fake"

    def capabilities(self) -> ToolCapabilities:
        return ToolCapabilities(
            kind="fake",
            prompts={"text", "box", "point"},
            masks=True,
            boxes=True,
            points=True,
            depth=False,
            multi_view=False,
            geometry=False,
        )

    # -- helpers ---------------------------------------------------------

    def _palette(self) -> dict[str, tuple[int, int, int]]:
        palette = dict(DEFAULT_COLORS)
        palette.update({str(k).lower(): tuple(v) for k, v in (self.config.get("colors") or {}).items()})
        return palette

    def _color_from_text(self, text: str) -> str | None:
        lowered = (text or "").lower()
        for name in sorted(self._palette(), key=len, reverse=True):
            if name in lowered:
                return name
        return None

    def _background(self, image: np.ndarray) -> np.ndarray:
        configured = self.config.get("background")
        if configured is not None:
            return np.asarray(configured, dtype=np.float32)
        # Most frequent colour of a coarse sample is the background in synthetic scenes.
        sampled = image[:: max(1, image.shape[0] // 64), :: max(1, image.shape[1] // 64)]
        flat = sampled.reshape(-1, 3).astype(np.uint8)
        colors, counts = np.unique(flat, axis=0, return_counts=True)
        return colors[int(np.argmax(counts))].astype(np.float32)

    def _search_mask(
        self,
        image: np.ndarray,
        prompt: Prompt,
        roi: tuple[int, int, int, int] | None,
        palette: dict[str, tuple[int, int, int]],
    ) -> tuple[np.ndarray, np.ndarray | None, str]:
        """Return ``(candidate_mask, target_rgb_or_None, mode)``."""
        tolerance = float(self.config.get("tolerance", 40.0))
        target: np.ndarray | None = None
        mode = "salient"

        if prompt.type == "text":
            color_name = self._color_from_text(prompt.text or "")
            if color_name is not None:
                target = np.asarray(palette[color_name], dtype=np.float32)
                mode = f"color:{color_name}"
        elif prompt.type == "point" and prompt.point is not None:
            u, v = int(prompt.point[0]), int(prompt.point[1])
            v = min(max(v, 0), image.shape[0] - 1)
            u = min(max(u, 0), image.shape[1] - 1)
            target = image[v, u].astype(np.float32)
            mode = "point_color"

        if target is None:
            target = self._background(image)

        dist = _distance(image, target)
        candidate = dist <= tolerance
        if mode == "salient" or mode == "point_color":
            # In salient mode "close to background" is the background, so invert.
            if mode == "salient":
                candidate = dist > tolerance

        if roi is not None:
            x0, y0, x1, y1 = roi
            gated = np.zeros_like(candidate)
            gated[y0:y1, x0:x1] = candidate[y0:y1, x0:x1]
            candidate = gated

        return candidate, (None if mode == "salient" else target), mode

    def _roi_from_box(self, prompt: Prompt, image: np.ndarray) -> tuple[int, int, int, int] | None:
        if prompt.type != "box" or prompt.box is None:
            return None
        height, width = image.shape[:2]
        cx, cy, bw, bh = (float(v) for v in prompt.box)
        x0 = int(round((cx - bw / 2.0) * width))
        x1 = int(round((cx + bw / 2.0) * width))
        y0 = int(round((cy - bh / 2.0) * height))
        y1 = int(round((cy + bh / 2.0) * height))
        x0, x1 = max(0, min(x0, width)), max(0, min(x1, width))
        y0, y1 = max(0, min(y0, height)), max(0, min(y1, height))
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, y0, x1, y1

    # -- perceive --------------------------------------------------------

    def perceive(self, ctx: ToolContext, prompts: Sequence[Prompt]) -> list[Percept]:
        percepts: list[Percept] = []
        for camera in sorted(ctx.images):
            image = ctx.images[camera]
            for prompt in prompts:
                percepts.append(self._perceive_one(ctx, camera, image, prompt))
        return percepts

    def _perceive_one(
        self, ctx: ToolContext, camera: str, image: np.ndarray, prompt: Prompt
    ) -> Percept:
        started = time.perf_counter()
        percept_id = str(uuid.uuid4())
        palette = self._palette()
        min_area_px = int(self.config.get("min_area_px", 16))
        max_instances = int(self.config.get("max_instances", 32))
        connectivity = int(self.config.get("connectivity", 4))

        roi = self._roi_from_box(prompt, image)
        candidate, target, mode = self._search_mask(image, prompt, roi, palette)

        labels, count = label_components(candidate, connectivity=connectivity)
        if count:
            areas = np.bincount(labels.ravel(), minlength=count + 1)
            order = [i for i in range(1, count + 1) if areas[i] >= min_area_px]
            order.sort(key=lambda i: int(areas[i]), reverse=True)
            order = order[:max_instances]
        else:
            areas = np.zeros(2, dtype=np.int64)
            order = []

        if prompt.type == "point" and prompt.point is not None:
            u = min(max(int(prompt.point[0]), 0), image.shape[1] - 1)
            v = min(max(int(prompt.point[1]), 0), image.shape[0] - 1)
            hit = int(labels[v, u])
            order = [i for i in order if i == hit]

        observations = []
        for rank, component in enumerate(order):
            mask = labels == component
            if target is not None:
                score = 1.0 - float(_distance(image[mask], target).mean()) / 255.0
            else:
                score = 0.9
            score = max(0.0, min(1.0, score))
            attributes: dict[str, float | str | bool] = {
                "mode": mode,
                "area_ratio": float(areas[component]) / float(image.shape[0] * image.shape[1]),
            }
            if prompt.type == "point":
                attributes["rank"] = float(rank)
            observations.append(
                make_observation(
                    mask,
                    score,
                    version_percept_id=percept_id,
                    index=rank,
                    encoding=ctx.options.masks,
                    label=(prompt.text if prompt.type == "text" else mode),
                    attributes=attributes,
                )
            )

        observations = apply_top_k(observations, prompt)
        status = "ok" if observations else "not_found"
        message = None if observations else "fake tool found no matching component"
        provenance = self._provenance(
            ctx,
            started,
            camera=camera,
            prompt=prompt,
            calibration_id=(ctx.calibration.get(camera).id if camera in ctx.calibration else None),
        )
        extras: dict[str, Any] = {"mode": mode}
        if order:
            extras["component_count"] = len(order)
        return make_percept(
            percept_id=percept_id,
            status=status,
            provenance=provenance,
            camera=camera,
            message=message,
            observations=observations,
            extras=extras,
        )


def _self_check() -> None:  # pragma: no cover - manual smoke helper
    """Tiny sanity script: ``python -m percepta.tools.fake``."""
    image = np.zeros((60, 80, 3), dtype=np.uint8)
    image[20:40, 30:50] = (255, 0, 0)
    tool = FakeTool()
    tool.load({})
    ctx = ToolContext(request_id="demo", images={"cam": image})
    percept = tool.perceive(ctx, [Prompt(type="text", text="red block")])[0]
    obs = percept.observations[0]
    print(percept.status, obs.area_px, obs.box_xyxy, obs.centroid_uv)
    assert obs.area_px == 400
    assert mask_iou(
        np.zeros((1, 1), dtype=bool), np.zeros((1, 1), dtype=bool)
    ) == 0.0
    _ = mask_box_xyxy, mask_centroid  # re-exported helpers used by tests


if __name__ == "__main__":  # pragma: no cover
    _self_check()
