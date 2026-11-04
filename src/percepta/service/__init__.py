"""Service package: FastAPI app factory, routes, metrics and logging (spec §6, §11.3).

Import surface::

    from percepta.service import create_app

Nothing here is imported by ``percepta`` itself at package import time, so installing
percepta without the ``service`` extra still works for in-process use.
"""

from __future__ import annotations

__all__ = ["create_app"]

_LAZY = {"create_app": ".app"}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        module = importlib.import_module(_LAZY[name], __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
