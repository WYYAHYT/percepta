"""Model backends — where a perception model actually runs.

``backends`` exists so a tool never has to know whether its model is imported in-process,
hosted in a separate virtualenv, or reached over HTTP. The tool keeps calling
``set_image()``/``segment()``; the deployment decides the rest.

See ``docs/model-isolation.md`` for the design rationale and the wire protocol.
"""

from __future__ import annotations

from .base import (
    BACKEND_KINDS,
    DEFAULT_BACKEND,
    BackendError,
    BackendRequestError,
    BackendUnavailable,
    ModelBackend,
    resolve_backend_kind,
)

__all__ = [
    "BACKEND_KINDS",
    "DEFAULT_BACKEND",
    "BackendError",
    "BackendRequestError",
    "BackendUnavailable",
    "ModelBackend",
    "resolve_backend_kind",
    "build_backend",
]


def build_backend(kind: str, **kwargs: object):
    """Construct a backend by name, importing the implementation lazily.

    Lazy import matters: ``subprocess`` and ``http`` backends pull in nothing heavy, but a
    process that only ever uses ``inprocess`` should not even import the IPC modules.
    """
    normalised = str(kind).strip().lower()
    if normalised == "inprocess":
        from .inprocess import InProcessBackend

        return InProcessBackend(**kwargs)  # type: ignore[arg-type]
    if normalised == "subprocess":
        from .subprocess_backend import SubprocessBackend

        return SubprocessBackend(**kwargs)  # type: ignore[arg-type]
    if normalised == "http":
        from .http_backend import HttpBackend

        return HttpBackend(**kwargs)  # type: ignore[arg-type]
    raise BackendRequestError(
        f"unknown backend {kind!r}; expected one of {', '.join(BACKEND_KINDS)}"
    )
