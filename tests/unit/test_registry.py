"""Tool registry: discovery, lifecycle, capabilities, error surfaces (spec §5.2, §11.2)."""

from __future__ import annotations

import pytest

from percepta.registry import ToolLoadError, ToolRegistry, UnknownTool
from percepta.schema import Prompt, ToolCapabilities, ToolContext
from percepta.tools.base import BaseTool


class _BoxOnly(BaseTool):
    """A decoy tool that advertises box prompts only (used for S8 capability checks)."""

    name = "box_only"
    version = "9.9.9"
    kind = "detector"

    def capabilities(self) -> ToolCapabilities:
        return ToolCapabilities(
            kind="detector",
            prompts={"box"},
            masks=False,
            boxes=True,
            points=False,
            depth=False,
            multi_view=False,
            geometry=False,
        )

    def perceive(self, ctx, prompts):  # pragma: no cover - not called in these tests
        return []


@pytest.fixture()
def registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register("box_only", _BoxOnly)
    return reg


def test_builtin_fake_is_discovered(registry):
    assert "fake" in registry.names()
    assert "fake" in registry


def test_instance_is_a_singleton_and_loads_once(registry):
    first = registry.instance("fake", {"tolerance": 8})
    second = registry.instance("fake", {"tolerance": 8})
    assert first is second
    assert registry.entry("fake").status == "ok"
    assert first.capabilities().kind == "fake"


def test_unknown_tool_raises(registry):
    with pytest.raises(UnknownTool):
        registry.instance("does_not_exist")
    assert registry.statuses()


def test_manual_registration_and_capabilities(registry):
    caps = registry.capabilities("box_only")
    assert caps.prompts == {"box"}
    assert registry.entry("box_only").status == "ok"
    assert registry.entry("box_only").kind == "detector"


def test_configure_skips_disabled_tools(registry):
    enabled = registry.configure(
        {"fake": {"enabled": False}, "box_only": {"enabled": True, "x": 1}}
    )
    assert "fake" not in enabled
    assert "box_only" in enabled


def test_configure_reports_unregistered_names_without_raising(registry):
    assert registry.configure({"ghost": {"enabled": True}}) == []


def test_close_all_resets_instances(registry):
    registry.instance("fake")
    registry.close_all()
    assert registry.entry("fake").instance is None
    assert registry.entry("fake").status == "lazy"


def test_failed_tool_load_is_reported_as_tool_load_error(registry):
    class Broken(BaseTool):
        name = "broken"
        version = "0.0.1"
        kind = "fake"

        def load(self, config):
            raise RuntimeError("no weights")

        def capabilities(self):  # pragma: no cover - never reached
            return ToolCapabilities(kind="fake")

        def perceive(self, ctx, prompts):  # pragma: no cover
            return []

    registry.register("broken", Broken)
    with pytest.raises(ToolLoadError, match="no weights"):
        registry.instance("broken")
    assert registry.entry("broken").status == "error"
    assert "no weights" in (registry.entry("broken").error or "")


def test_tool_context_carries_upstream_extras():
    ctx = ToolContext(request_id="r", images={})
    ctx.extras["upstream"] = []
    assert ctx.extras["upstream"] == []
    assert ctx.logger is None and ctx.cache is None
    _ = Prompt(type="text", text="cup")


def test_describe_reports_declared_metadata_before_load(registry):
    """``describe()`` must not need a loaded instance to report version/kind.

    Both are class attributes, so an unloaded tool can still answer "what am I?". The
    default ``0.0.0`` is reserved for a class that genuinely declares no version.
    """
    registry.register("box_only", _BoxOnly)  # registered, deliberately not loaded
    entry = registry.entry("box_only")
    assert entry.instance is None
    described = entry.describe()
    assert described["version"] == "9.9.9"
    assert described["kind"] == "detector"
    assert described["loaded"] is False


def test_describe_falls_back_to_zero_version_when_undeclared(registry):
    class Nameless(BaseTool):
        name = "nameless"

        def capabilities(self):  # pragma: no cover - not called here
            return ToolCapabilities(kind="fake")

        def perceive(self, ctx, prompts):  # pragma: no cover
            return []

    registry.register("nameless", Nameless)
    described = registry.entry("nameless").describe()
    assert described["version"] == "0.0.0"
    assert described["kind"] == "fake"  # inherited from BaseTool's class attribute


def test_describe_prefers_loaded_instance_metadata(registry):
    registry.register("box_only", _BoxOnly)
    registry.instance("box_only")
    described = registry.entry("box_only").describe()
    assert described["version"] == "9.9.9"
    assert described["loaded"] is True
