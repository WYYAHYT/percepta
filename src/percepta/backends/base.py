"""Model backend abstraction: where does a model actually run?

A perception model needs a consistent answer to "which interpreter runs this?", because two
models may require mutually exclusive dependencies (see ``docs/model-isolation.md``). The
frozen tool protocol (spec §5.1) only says ``perceive(ctx, prompts) -> list[Percept]``, so the
*where* can be a swappable detail.

Three backends exist:

``inprocess``
    The model's dependency is installed in the same environment as percepta. Fastest (no
    copy, no IPC), and the right default when there is no dependency conflict.

``subprocess``
    The model lives in its own virtualenv; percepta talks to it over a binary pipe. This is
    the escape hatch when two models disagree about ``torch``/``numpy`` versions — no
    container needed. Measured cost: 0.01 s interpreter start, then a resident worker.

``http``
    The model runs as a separate service, possibly on another machine. Same isolation, but
    independently restartable and observable.

Selection is configuration, not code: ``tools.<name>.backend``. Every backend implements the
same small protocol below, so a tool never learns which one it got.

Note on numpy in annotations: this module must stay import-safe without any model framework,
but it *does* depend on numpy, which is a core dependency, so a plain import is fine here.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ..schema import Prompt

__all__ = [
    "ModelBackend",
    "BackendError",
    "BackendUnavailable",
    "BackendRequestError",
    "BACKEND_KINDS",
    "DEFAULT_BACKEND",
    "resolve_backend_kind",
]

log = logging.getLogger(__name__)

#: Accepted values for ``tools.<name>.backend``.
BACKEND_KINDS = ("inprocess", "subprocess", "http")

DEFAULT_BACKEND = "inprocess"


class BackendError(RuntimeError):
    """A backend failed. Subclasses map onto Percept statuses (spec §16.4)."""


class BackendUnavailable(BackendError):
    """The model cannot run at all: missing weights, missing GPU, dead worker, no service.

    Maps to Percept ``status="unavailable"``. Never ``not_found`` — "I could not look" and
    "I looked and found nothing" must stay distinguishable (spec §16.4).
    """


class BackendRequestError(BackendError):
    """The backend rejected the request itself (bad prompt, protocol mismatch).

    Maps to Percept ``status="invalid_request"``.
    """


@runtime_checkable
class ModelBackend(Protocol):
    """What every backend must provide.

    The shape mirrors the SAM3 inference flow because that is the most demanding case:
    an expensive ``set_image`` that is worth reusing across prompts, then a cheap per-prompt
    ``segment``. Backends that do not need image reuse can implement ``set_image`` as a no-op.
    """

    name: str
    """Human-readable backend name, for logs and ``/v1/info``."""

    def set_image(self, image_rgb: np.ndarray, depth: np.ndarray | None = None) -> None:
        """Prepare a frame. Called once per (camera, image) and reused for every prompt."""

    def segment(self, prompt: Prompt) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
        """Run one prompt on the prepared frame.

        Returns ``(masks, boxes, scores)`` where ``masks`` is ``(N, H, W)`` boolean at the
        original image resolution, ``boxes`` is ``(N, 4)`` xyxy in pixels or ``None``, and
        ``scores`` is ``(N,)``. Empty ``N`` means "ran fine, found nothing".
        """

    def close(self) -> None:
        """Release weights/handles. Must be idempotent."""

    def describe(self) -> dict[str, Any]:
        """Metadata for ``/v1/info``: device, model path, weight hash, worker pid…"""


def resolve_backend_kind(config: dict[str, Any]) -> str:
    """Read ``backend`` from a tool config, validating it against :data:`BACKEND_KINDS`.

    Unknown values are a configuration error, not a silent fallback: silently using the
    wrong backend would run the model in an environment that cannot import it, and the
    failure would surface as a confusing ``ModuleNotFoundError`` much later.
    """
    raw = config.get("backend", DEFAULT_BACKEND)
    kind = str(raw).strip().lower()
    if kind not in BACKEND_KINDS:
        raise BackendRequestError(
            f"unknown backend {raw!r}; expected one of {', '.join(BACKEND_KINDS)}"
        )
    return kind
