"""``multi_view_3d`` tool behaviour: one Percept for the whole request, or nothing (D6).

The tool is the §9 geometry plugin: it reads ``ctx.extras["upstream"]`` (spec §5.3) and adds
exactly one fused Percept per request — never one per camera (decision D6).
"""

from __future__ import annotations

import sys

import numpy as np

from percepta.schema import PerceiveOptions, ToolContext
from percepta.tools.multi_view_3d import MultiView3DTool
from tests import synth_geom


def _tool(**config) -> MultiView3DTool:
    tool = MultiView3DTool()
    tool.load(dict(config))
    return tool


def _context(calibrations, images, percepts, **options) -> ToolContext:
    return ToolContext(
        request_id="req-geometry",
        images=images,
        calibration=calibrations,
        context={"table_z": 0.0},
        options=PerceiveOptions(**options),
        extras={"upstream": percepts},
    )


def test_capabilities_are_geometry_only():
    capabilities = _tool().capabilities()
    assert capabilities.kind == "geometry"
    assert capabilities.prompts == set()
    assert capabilities.masks is False
    assert capabilities.boxes is False
    assert capabilities.points is False
    assert capabilities.depth is True
    assert capabilities.multi_view is True
    assert capabilities.geometry is True


def test_tool_metadata():
    tool = _tool()
    assert tool.name == "multi_view_3d"
    assert tool.version
    assert tool.api_version == "percepta/tool/v1"


def test_perceive_returns_one_fused_percept_for_the_whole_request():
    calibrations, images, _ = synth_geom.two_view_roomcube()
    masks = synth_geom.cube_masks(calibrations, (0.0, 0.0, 0.05), 0.05)
    percepts = synth_geom.mask_percepts(calibrations, masks)
    ctx = _context(calibrations, images, percepts)

    percepts_out = _tool().perceive(ctx, [])

    assert len(percepts_out) == 1
    fused = percepts_out[0]
    assert fused.geometry is not None
    assert fused.geometry.frame == "world"
    assert fused.geometry.views_used == 2
    assert fused.geometry.uncertainty_m is not None
    error = np.linalg.norm(np.asarray(fused.geometry.centroid) - np.array([0.0, 0.0, 0.05]))
    assert error < 0.02
    assert fused.extras["method"] == "prism"
    assert fused.camera is None  # the fused percept spans cameras


def test_tool_honours_depth_when_the_calibration_has_it():
    calibrations, images, _ = synth_geom.two_view_roomcube(with_depth=True)
    masks = synth_geom.cube_masks(calibrations, (0.0, 0.0, 0.05), 0.05)
    percepts = synth_geom.mask_percepts(calibrations, masks)
    out = _tool().perceive(_context(calibrations, images, percepts), [])
    assert len(out) == 1
    assert out[0].extras["method"] == "depth"


def test_tool_returns_nothing_without_upstream():
    calibrations, images, _ = synth_geom.two_view_roomcube()
    assert _tool().perceive(_context(calibrations, images, []), []) == []


def test_tool_returns_nothing_without_calibration():
    calibrations, images, _ = synth_geom.two_view_roomcube()
    masks = synth_geom.cube_masks(calibrations, (0.0, 0.0, 0.05), 0.05)
    percepts = synth_geom.mask_percepts(calibrations, masks)
    assert _tool().perceive(_context({}, images, percepts), []) == []


def test_tool_returns_nothing_for_a_single_view_without_depth_or_table():
    calibrations, images, _ = synth_geom.two_view_roomcube()
    camera = sorted(calibrations)[0]
    masks = synth_geom.cube_masks(calibrations, (0.0, 0.0, 0.05), 0.05)
    percepts = synth_geom.mask_percepts(calibrations, {camera: masks[camera]})
    ctx = ToolContext(
        request_id="req-single",
        images=images,
        calibration=calibrations,
        context={},
        options=PerceiveOptions(),
        extras={"upstream": percepts},
    )
    assert _tool().perceive(ctx, []) == []


def test_config_supplies_request_context_defaults():
    """``load(config)`` may carry the D4 tunables; request context still wins."""
    calibrations, images, _ = synth_geom.two_view_roomcube()
    masks = synth_geom.cube_masks(calibrations, (0.0, 0.0, 0.05), 0.05)
    percepts = synth_geom.mask_percepts(calibrations, masks)
    out = _tool(prism_iou_threshold=0.99).perceive(
        _context(calibrations, images, percepts), []
    )
    assert len(out) == 1
    assert "geometry_low_confidence" in out[0].extras["warnings"]


def test_module_import_does_not_pull_torch():
    """The geometry tool must stay importable on a GPU-less, weight-less machine."""
    import importlib

    module_name = "percepta.tools.multi_view_3d"
    had_torch = "torch" in sys.modules
    module = importlib.reload(sys.modules[module_name])
    assert ("torch" in sys.modules) == had_torch
    assert not hasattr(module, "torch")
