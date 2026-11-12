"""Tool discovery, instantiation and lifecycle (spec §5.2, §11.2).

Registration has two sources, entry points first:

1. the ``percepta.tools`` entry-point group (third-party packages and this package once
   installed), and
2. a static fallback map of the built-in tools, so the registry also works from a plain
   ``PYTHONPATH=src`` checkout without an install.

Every tool instance is created once, loaded once (:meth:`PerceptionTool.load`), and
guarded by a per-tool :class:`threading.Lock`: model inference is serialised per tool
(§11.2), while the service layer decides how much to queue.
"""

from __future__ import annotations

import importlib
import logging
import threading
from dataclasses import dataclass, field
from typing import Any

from .schema import ToolCapabilities, ToolKind
from .tools.base import PerceptionTool

__all__ = [
    "BUILTIN_TOOLS",
    "ToolEntry",
    "ToolRegistry",
    "UnknownTool",
    "ToolLoadError",
    "default_registry",
]

log = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "percepta.tools"

#: ``name -> "module:ClassName"`` for the tools shipped in this package.
BUILTIN_TOOLS: dict[str, str] = {
    "fake": "percepta.tools.fake:FakeTool",
    "sam3": "percepta.tools.sam3:Sam3Tool",
    "multi_view_3d": "percepta.tools.multi_view_3d:MultiView3DTool",
}


class UnknownTool(KeyError):
    """Requested tool name is not registered (maps to ``invalid_request``)."""


class ToolLoadError(RuntimeError):
    """A registered tool could not be imported, constructed or loaded."""


@dataclass
class ToolEntry:
    """One registered tool: its factory, optional instance, and health."""

    name: str
    factory: Any  # Callable[[], PerceptionTool] | type[PerceptionTool]
    source: str = "builtin"
    instance: PerceptionTool | None = None
    status: str = "lazy"  # lazy | ok | error
    error: str | None = None
    kind: ToolKind | None = None
    config: dict[str, Any] = field(default_factory=dict)
    lock: threading.RLock = field(default_factory=threading.RLock)

    def describe(self) -> dict[str, Any]:
        """Describe the tool, recording a resident backend's nonblocking failure snapshot.

        Returns:
            Tool metadata and current load/health status; a failed singleton remains loaded.

        Example:
            >>> ToolEntry(name="missing", factory=None).describe()["status"]
            'lazy'
        """
        error = getattr(getattr(self.instance, "_backend", None), "health_error", None)
        if isinstance(error, str) and error:
            self.status, self.error = "error", error
        return {
            "name": self.name,
            "version": _version(self),
            "status": self.status,
            "kind": _kind(self),
            "loaded": self.instance is not None,
        }


def _declared_meta(factory: Any) -> tuple[str | None, str | None]:
    """Read ``version``/``kind`` off the tool *class*, without instantiating it.

    Both are class attributes on :class:`~percepta.tools.base.PerceptionTool` /
    ``BaseTool``, so an unloaded tool can still report them. Without this, a tool that is
    registered but not loaded (no GPU, no weights, disabled) reports version ``0.0.0`` and
    kind ``None`` — which reads like a broken plugin rather than "known but not loaded".

    ``capabilities()`` cannot be used here because it needs an instance; the class-level
    ``kind`` attribute is the only metadata available before ``load()``.

    Returns ``(version, kind)``; each is ``None`` when the class does not declare it.
    """
    if factory is None:
        return None, None
    version = getattr(factory, "version", None)
    kind = getattr(factory, "kind", None)
    return (
        None if version is None else str(version),
        None if kind is None else str(kind),
    )


def _version(entry: ToolEntry) -> str:
    if entry.instance is not None:
        return str(getattr(entry.instance, "version", "0.0.0"))
    declared, _ = _declared_meta(entry.factory)
    return declared or "0.0.0"


def _kind(entry: ToolEntry) -> str | None:
    if entry.kind is not None:
        return entry.kind
    _, declared = _declared_meta(entry.factory)
    return declared


def _resolve(spec: str) -> Any:
    """Resolve ``"module:attribute"`` (or ``"module.attribute"``) into a Python object."""
    module_name, _, attr = spec.partition(":")
    if not attr:
        module_name, _, attr = spec.rpartition(".")
    if not module_name:
        raise ToolLoadError(f"bad tool spec {spec!r}")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr)
    except AttributeError as exc:  # pragma: no cover - misconfiguration path
        raise ToolLoadError(f"{module_name} has no attribute {attr!r}") from exc


class ToolRegistry:
    """Holds tool entries; loads lazily and reports per-tool health."""

    def __init__(self, discover: bool = True) -> None:
        self._entries: dict[str, ToolEntry] = {}
        self._discover_error: str | None = None
        if discover:
            self.discover()

    # -- registration ----------------------------------------------------

    def register(self, name: str, factory: Any, *, source: str = "manual") -> None:
        if name in self._entries and source != "manual":
            return  # entry points never shadow an explicit registration
        self._entries[name] = ToolEntry(name=name, factory=factory, source=source)

    def discover(self) -> None:
        """Register entry points, then fill gaps from :data:`BUILTIN_TOOLS`."""
        try:
            from importlib.metadata import entry_points

            found = entry_points(group=ENTRY_POINT_GROUP)
            for ep in found:
                try:
                    self.register(ep.name, ep.load(), source="entry_point")
                except Exception as exc:  # noqa: BLE001 - a broken plugin must not break startup
                    log.warning("failed to load entry point %s: %s", ep.name, exc)
        except Exception as exc:  # noqa: BLE001 - metadata may be unavailable
            self._discover_error = str(exc)
            log.debug("entry point discovery unavailable: %s", exc)

        for name, spec in BUILTIN_TOOLS.items():
            if name in self._entries:
                continue
            try:
                self.register(name, _resolve(spec), source="builtin")
            except Exception as exc:  # noqa: BLE001 - keep serving other tools
                log.warning("builtin tool %s not importable: %s", name, exc)
                self._entries[name] = ToolEntry(
                    name=name,
                    factory=None,
                    source="builtin",
                    status="error",
                    error=f"{type(exc).__name__}: {exc}",
                )

    # -- introspection ---------------------------------------------------

    def names(self) -> list[str]:
        return sorted(self._entries)

    def __contains__(self, name: object) -> bool:
        return name in self._entries

    def entry(self, name: str) -> ToolEntry:
        try:
            return self._entries[name]
        except KeyError:
            raise UnknownTool(name) from None

    def statuses(self) -> list[dict[str, Any]]:
        return [self._entries[name].describe() for name in self.names()]

    # -- lifecycle -------------------------------------------------------

    def instance(self, name: str, config: dict[str, Any] | None = None) -> PerceptionTool:
        """Return the loaded singleton, serializing first load with the tool's inference lock.

        Args:
            name: Registered tool name.
            config: Deployment configuration used on the first load.

        Returns:
            The same loaded tool for all concurrent callers of this registry.

        Raises:
            UnknownTool: The tool is not registered.
            ToolLoadError: Construction or loading failed.

        Example:
            >>> from percepta.tools.fake import FakeTool
            >>> registry = ToolRegistry(discover=False)
            >>> registry.register("fake", FakeTool)
            >>> registry.instance("fake") is registry.instance("fake")
            True
        """
        entry = self.entry(name)
        with entry.lock:
            if entry.instance is not None:
                return entry.instance
            if entry.factory is None:
                entry.status = "error"
                raise ToolLoadError(f"tool {name!r} is not importable: {entry.error}")

            entry.status = "loading"
            tool = None
            try:
                tool = entry.factory() if callable(entry.factory) else entry.factory
                if not isinstance(tool, PerceptionTool):
                    raise ToolLoadError(
                        f"tool {name!r} does not implement PerceptionTool (got {type(tool).__name__})"
                    )
                tool.load(dict(config or {}))
                if getattr(tool, "name", None) != name:
                    log.warning("tool %r reports name %r", name, getattr(tool, "name", None))
                kind = tool.capabilities().kind
                entry.instance = tool
                entry.config = dict(config or {})
                entry.kind = kind
                entry.status = "ok"
                entry.error = None
                return tool
            except Exception as exc:
                if isinstance(tool, PerceptionTool):
                    try:
                        tool.close()
                    except Exception as cleanup_error:  # noqa: BLE001 - preserve the original load error
                        log.warning("error closing failed tool %s: %s", name, cleanup_error)
                entry.status = "error"
                entry.error = f"{type(exc).__name__}: {exc}"
                raise ToolLoadError(f"failed to load tool {name!r}: {exc}") from exc

    def capabilities(self, name: str, config: dict[str, Any] | None = None) -> ToolCapabilities:
        return self.instance(name, config).capabilities()

    def lock_for(self, name: str) -> threading.RLock:
        return self.entry(name).lock

    def close_all(self) -> None:
        for name in self.names():
            entry = self._entries[name]
            if entry.instance is None:
                continue
            try:
                entry.instance.close()
            except Exception as exc:  # noqa: BLE001 - shutdown must not raise
                log.warning("error closing tool %s: %s", name, exc)
            finally:
                entry.instance = None
                entry.status = "lazy"

    # -- helpers ---------------------------------------------------------

    def configure(self, tools_config: dict[str, dict[str, Any]] | None) -> list[str]:
        """Instantiate every enabled tool up-front; return the enabled names.

        ``tools_config`` mirrors the YAML ``tools:`` block. A tool with ``enabled: false``
        is skipped; any other key is passed to :meth:`PerceptionTool.load`.
        """
        enabled: list[str] = []
        for name, raw in (tools_config or {}).items():
            cfg = dict(raw or {})
            if not cfg.get("enabled", True):
                continue
            if name not in self._entries:
                log.warning("configured tool %r is not registered", name)
                continue
            try:
                self.instance(name, cfg)
                enabled.append(name)
            except ToolLoadError as exc:
                log.warning("%s", exc)
        return enabled


_default: ToolRegistry | None = None
_default_lock = threading.Lock()


def default_registry() -> ToolRegistry:
    """Process-wide registry singleton."""
    global _default
    with _default_lock:
        if _default is None:
            _default = ToolRegistry()
        return _default
