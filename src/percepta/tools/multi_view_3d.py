"""``multi_view_3d`` — the built-in geometry tool (spec §5.4, §9).

It is *pure geometry*: it segments nothing, ignores prompts, and consumes the masks the
pipeline already produced. Per decision **D6** the pipeline calls every tool **once per
request** with all cameras in ``ctx.images`` and the upstream percepts in
``ctx.extras["upstream"]`` (spec §5.3), so this tool fuses every calibrated view in one go
and returns at most one ``Percept``:

* ``[fused]`` — a world-frame ``Geometry`` (see :mod:`percepta.fusion` for the depth /
  prism / single-view priority of spec §9);
* ``[]`` — no geometry is possible (no calibration, no masks, no table for a single view).
  The pipeline turns that into the ``geometry_unavailable`` warning.

The tool never imports torch or any model package (``AGENTS.md`` §6.5): everything it needs
is numpy, already installed.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from ..fusion import TOOL_NAME, TOOL_VERSION, fuse
from ..schema import Percept, Prompt, ToolCapabilities, ToolContext
from .base import BaseTool

__all__ = ["MultiView3DTool"]

#: Request-context keys this tool's ``load(config)`` may supply defaults for (D4 tunables).
_CONTEXT_KEYS = (
    "table_plane",
    "table_z",
    "prism_max_height_m",
    "prism_step_m",
    "prism_iou_threshold",
    "single_view_rough_height_m",
)


class MultiView3DTool(BaseTool):
    """Multi-view mask + calibration -> ``Geometry`` (spec §9)."""

    name = TOOL_NAME
    version = TOOL_VERSION
    kind = "geometry"

    def capabilities(self) -> ToolCapabilities:
        """Geometry only: no prompts, no masks/boxes/points; needs depth and many views."""
        return ToolCapabilities(
            kind="geometry",
            prompts=set(),
            masks=False,
            boxes=False,
            points=False,
            depth=True,
            multi_view=True,
            geometry=True,
        )

    def perceive(self, ctx: ToolContext, prompts: Sequence[Prompt]) -> list[Percept]:
        """Fuse ``ctx.extras["upstream"]`` for every calibrated camera; ``[]`` if impossible.

        ``prompts`` is accepted for the frozen tool protocol but ignored: a geometry tool has
        no prompt type of its own (capabilities ``prompts=set()``), it only re-uses upstream
        observations.
        """
        _ = prompts
        started = time.perf_counter()
        upstream = list(ctx.extras.get("upstream") or [])
        context = dict(ctx.context or {})
        for key in _CONTEXT_KEYS:
            if key in self.config and key not in context:
                context[key] = self.config[key]
        fused = fuse(upstream, ctx.calibration, context, ctx.options)
        if fused is None:
            self._log(ctx, "multi_view_3d found no usable geometry", level="debug")
            return []
        warnings = list(fused.extras.get("warnings") or [])
        if warnings:
            self._log(
                ctx,
                f"multi_view_3d geometry warnings: {', '.join(str(w) for w in warnings)}",
                level="warning",
            )
        self._log(
            ctx,
            "multi_view_3d fused {n} view(s) via {method} in {ms:.1f} ms".format(
                n=fused.geometry.views_used if fused.geometry else 0,
                method=fused.extras.get("method", "unknown"),
                ms=(time.perf_counter() - started) * 1000.0,
            ),
            level="debug",
        )
        return [fused]

    @staticmethod
    def _log(ctx: ToolContext, message: str, *, level: str = "debug") -> None:
        """Best-effort structured log; a missing/odd logger must never fail a request."""
        logger: Any = getattr(ctx, "logger", None)
        if logger is None:
            return
        try:
            getattr(logger, level, logger.debug)(message, extra={"tool": TOOL_NAME})
        except (AttributeError, TypeError, ValueError):
            pass
