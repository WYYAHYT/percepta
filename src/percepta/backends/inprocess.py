"""In-process backend: the model's dependency is installed in percepta's own environment.

This is the default and the fastest path — no IPC, no copying. Use it when there is no
dependency conflict (``docs/model-isolation.md`` §4.1).

The class deliberately holds *no* model code: it wraps whatever adapter object the tool
supplies, so the same wrapper serves any model and keeps the tool free of backend logic.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..schema import Prompt
from .base import BackendRequestError, BackendUnavailable

__all__ = ["InProcessBackend"]


class InProcessBackend:
    """Wrap an already-constructible model adapter and call it directly."""

    name = "inprocess"

    def __init__(self, adapter: Any = None, factory: Any = None, config: dict | None = None) -> None:
        """Accept either a ready adapter or a ``factory(**config)`` to build one lazily."""
        if adapter is None and factory is None:
            raise BackendRequestError("InProcessBackend needs an 'adapter' or a 'factory'")
        self._adapter = adapter
        self._factory = factory
        self._config = dict(config or {})

    def _model(self) -> Any:
        if self._adapter is None:
            if self._factory is None:  # pragma: no cover - guarded in __init__
                raise BackendUnavailable("in-process backend has no model adapter")
            try:
                self._adapter = self._factory(**self._config)
            except Exception as exc:  # noqa: BLE001 - surface as unavailable, not not_found
                raise BackendUnavailable(f"failed to build in-process adapter: {exc}") from exc
        return self._adapter

    def set_image(self, image_rgb: np.ndarray, depth: np.ndarray | None = None) -> None:
        self._model().set_image(image_rgb, depth)

    def segment(self, prompt: Prompt) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
        return self._model().segment(prompt)

    def close(self) -> None:
        adapter, self._adapter = self._adapter, None
        if adapter is not None:
            closer = getattr(adapter, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception:  # noqa: BLE001 - closing must not raise
                    pass

    def describe(self) -> dict[str, Any]:
        adapter = self._adapter
        model = getattr(self, "_config", {}).get("checkpoint")
        return {
            "backend": self.name,
            "loaded": adapter is not None,
            "model": model,
            "adapter": None if adapter is None else type(adapter).__name__,
        }
