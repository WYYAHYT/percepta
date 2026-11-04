"""Tool plugin protocol and shared helpers (spec §5.1).

Rules every tool implementation must follow (enforced by tests and code review):

1. **Import-safe without heavy deps.** A module in ``percepta.tools`` may be imported on a
   GPU-less, weight-less machine: ``import torch`` / ``import sam3`` belongs inside
   ``load()``, never at module level. ``registry.discover()`` imports every tool module.
2. **Declare capabilities.** ``capabilities()`` must describe what ``perceive()`` can
   actually do; the pipeline validates prompts against it *before* calling the tool.
3. **Never swallow errors into ``not_found``.** Raise :class:`ToolUnavailable` when the
   model/service is missing (Percept status ``unavailable``); return a ``not_found`` Percept
   only when the tool genuinely looked and found nothing (spec §16.4).
4. **Encode coordinates as specified.** Pixel ``u`` = column, ``v`` = row, origin top-left;
   box is ``[x0, y0, x1, y1]`` with ``x1``/``y1`` exclusive.
"""

from __future__ import annotations

import abc
import hashlib
import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from .. import masks as mask_utils
from ..schema import (
    TOOL_API_VERSION,
    Geometry,
    Observation,
    PerceiveOptions,
    Percept,
    Prompt,
    Provenance,
    ToolCapabilities,
    ToolContext,
)

__all__ = [
    "PerceptionTool",
    "BaseTool",
    "ToolError",
    "ToolUnavailable",
    "UnsupportedPrompt",
    "make_percept",
    "make_observation",
    "build_provenance",
    "select_prompts",
    "apply_top_k",
]


class ToolError(RuntimeError):
    """Base class for tool failures that must surface as a status, not an exception."""


class ToolUnavailable(ToolError):
    """The tool cannot run at all: missing weights, missing device, broken service."""


class UnsupportedPrompt(ToolError):
    """The tool was handed a prompt type it did not declare support for."""


def _stable_object_id(percept_id: str, index: int) -> str:
    """``sha1(percept_id + index)`` — stable *within* one percept (spec §3.4)."""
    return hashlib.sha1(f"{percept_id}:{index}".encode()).hexdigest()


def make_observation(
    mask: np.ndarray | None,
    score: float,
    *,
    version_percept_id: str,
    index: int,
    encoding: str = "rle",
    label: str | None = None,
    box_xyxy: Sequence[float] | None = None,
    attributes: dict[str, float | str | bool] | None = None,
) -> Observation:
    """Build an :class:`Observation`, filling area/box/centroid from the mask when given."""
    arr = None if mask is None else np.asarray(mask).astype(bool)
    if arr is not None:
        if box_xyxy is None:
            box_xyxy = mask_utils.mask_box_xyxy(arr)
        centroid = mask_utils.mask_centroid(arr)
        area = mask_utils.mask_area(arr)
        mask_payload = mask_utils.to_mask(arr, encoding) if encoding != "none" else None
    else:
        centroid, area, mask_payload = None, None, None
    return Observation(
        object_id=_stable_object_id(version_percept_id, index),
        score=float(score),
        box_xyxy=None if box_xyxy is None else [float(v) for v in box_xyxy],
        mask=mask_payload,
        area_px=area,
        centroid_uv=None if centroid is None else [float(centroid[0]), float(centroid[1])],
        label=label,
        attributes=dict(attributes or {}),
    )


def build_provenance(
    *,
    tool: str,
    tool_version: str,
    kind: str,
    started_at: float,
    camera: str | None = None,
    prompt: Prompt | None = None,
    images: dict[str, np.ndarray] | None = None,
    calibration_id: str | None = None,
    model: str | None = None,
    model_sha256: str | None = None,
    cache_hit: bool = False,
) -> Provenance:
    """Assemble provenance, including per-camera image hashes (spec §3.4)."""
    hashes: dict[str, str] = {}
    for cam, arr in (images or {}).items():
        hashes[cam] = mask_utils.array_sha256(arr)
    return Provenance(
        tool=tool,
        tool_version=tool_version,
        kind=kind,  # validated by the schema's ToolKind literal
        model=model,
        model_sha256=model_sha256,
        prompt=prompt,
        camera=camera,
        image_sha256=hashes,
        calibration_id=calibration_id,
        latency_ms=round((time.perf_counter() - started_at) * 1000.0, 3),
        cache_hit=cache_hit,
    )


def make_percept(
    *,
    percept_id: str,
    status: str,
    provenance: Provenance,
    camera: str | None = None,
    message: str | None = None,
    observations: Sequence[Observation] | None = None,
    geometry: Geometry | None = None,
    extras: dict[str, Any] | None = None,
) -> Percept:
    return Percept(
        percept_id=percept_id,
        status=status,  # validated by the schema's Status literal
        message=message,
        camera=camera,
        observations=list(observations or []),
        geometry=geometry,
        provenance=provenance,
        extras=dict(extras or {}),
    )


def select_prompts(prompts: Sequence[Prompt], caps: ToolCapabilities) -> list[Prompt]:
    """Split prompts into the supported ones; the pipeline reports the others."""
    return [p for p in prompts if p.type in caps.prompts]


def apply_top_k(
    observations: Sequence[Observation], prompt: Prompt
) -> list[Observation]:
    """Sort by score and keep ``prompt.top_k``; drops anything below ``prompt.min_score``."""
    kept = [o for o in observations if o.score >= prompt.min_score]
    kept.sort(key=lambda o: o.score, reverse=True)
    return kept[: max(0, prompt.top_k)]


class PerceptionTool(abc.ABC):
    """The protocol every tool implements (spec §5.1).

    A plain ``Protocol`` would be more pythonic, but an ABC gives tools a shared, concrete
    ``name``/``version``/``api_version`` and lets the registry instantiate by class.
    """

    name: str = "unnamed"
    version: str = "0.0.0"
    api_version: str = TOOL_API_VERSION

    @abc.abstractmethod
    def capabilities(self) -> ToolCapabilities: ...

    def load(self, config: dict[str, Any]) -> None:  # noqa: B027 - intentional no-op hook
        """Called exactly once per process before ``perceive``. Heavy imports go here."""

    @abc.abstractmethod
    def perceive(self, ctx: ToolContext, prompts: list[Prompt]) -> list[Percept]: ...

    def close(self) -> None:  # noqa: B027 - intentional no-op hook
        """Release weights/handles. Must be idempotent."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} name={self.name!r} version={self.version!r}>"


class BaseTool(PerceptionTool):
    """Convenience base: config storage, lazily-built provenance helpers, timing."""

    kind: str = "fake"

    def __init__(self) -> None:
        self.config: dict[str, Any] = {}
        self._loaded = False

    def load(self, config: dict[str, Any]) -> None:
        self.config = dict(config or {})
        self._loaded = True

    @property
    def loaded(self) -> bool:
        return self._loaded

    def _provenance(self, ctx: ToolContext, started_at: float, **kwargs: Any) -> Provenance:
        kwargs.setdefault("tool", self.name)
        kwargs.setdefault("tool_version", self.version)
        kwargs.setdefault("kind", self.kind)
        kwargs.setdefault("started_at", started_at)
        kwargs.setdefault("images", ctx.images)
        return build_provenance(**kwargs)

    def _options(self, ctx: ToolContext) -> PerceiveOptions:
        return ctx.options


def timed(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator used by tools that want ``latency_ms`` on every Percept."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            _ = (time.perf_counter() - started) * 1000.0

    return wrapper
