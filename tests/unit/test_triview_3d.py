"""``triview_3d``: three views, fused, and honest when the fusion is not really three-way.

The tool's value is that three pairs of a three-camera rig give three independent depth
measurements of the same object. Two things must therefore be true and are tested here:

* the merged result lands where the geometry says it should, not merely somewhere plausible;
* when only some pairs work, the answer says so instead of wearing a three-view label.

The synthetic scene is the same analytic texture used for ``stereo_3d``, extended to a third
camera on a different baseline, so the expected depth is known exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from percepta.schema import Calibration, PerceiveOptions, ToolContext
from percepta.tools.triview_3d import (
    ERR_BACKEND_NOT_CONFIGURED,
    ERR_NEEDS_THREE_VIEWS,
    ERR_NO_USABLE_PAIR,
    TOOL_NAME,
    Triview3DTool,
)

FOCAL = 397.0413
WIDTH, HEIGHT = 320, 240
PLANE_Y = 1.0


def _rotation() -> np.ndarray:
    """World <- camera for a camera looking along +world-y, +x right, +y down."""
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ]
    )


def _calibration(camera: str, x: float) -> Calibration:
    T = np.eye(4)
    T[:3, :3] = _rotation()
    T[:3, 3] = [x, 0.0, 0.0]
    K = [[FOCAL, 0.0, WIDTH / 2], [0.0, FOCAL, HEIGHT / 2], [0.0, 0.0, 1.0]]
    return Calibration(camera=camera, width=WIDTH, height=HEIGHT, K=K, T_world_cam=T.tolist())


def _plane_image(offset_x: float, plane_y: float = PLANE_Y) -> np.ndarray:
    """The plane ``y = plane_y`` rendered from a camera at ``x = offset_x``."""
    rng = np.random.default_rng(1234)
    texture = rng.integers(20, 235, size=(512, 512), dtype=np.int16)
    img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    xs = np.arange(WIDTH)
    ys = np.arange(HEIGHT)
    world_x = (xs - WIDTH / 2) / FOCAL * plane_y + offset_x
    world_z = (ys - HEIGHT / 2) / FOCAL * plane_y
    tex_u = ((world_x * 160.0) % 512).astype(np.int32)
    tex_v = ((world_z * 160.0 + 256.0) % 512).astype(np.int32)
    grey = texture[tex_v][:, tex_u].astype(np.uint8)
    img[:, :, 0] = grey
    img[:, :, 1] = grey
    img[:, :, 2] = grey
    return img


def _context(images, calibration, **extra) -> ToolContext:
    return ToolContext(
        request_id="test",
        images=images,
        calibration=calibration,
        context=dict(extra),
        options=PerceiveOptions(),
    )


def _three_view_scene():
    """Three cameras on one line, all looking the same way, spaced 0.3 m apart.

    Every pair therefore has a different baseline (0.3, 0.3, 0.6 m), which is what makes the
    inverse-variance merge testable: the long pair should dominate.
    """
    cal = {
        "cam0": _calibration("cam0", 0.0),
        "cam1": _calibration("cam1", 0.3),
        "cam2": _calibration("cam2", 0.6),
    }
    images = {
        "cam0": _plane_image(0.0),
        "cam1": _plane_image(0.3),
        "cam2": _plane_image(0.6),
    }
    return images, cal


# ---------------------------------------------------------------------------
# The fusion itself
# ---------------------------------------------------------------------------


def test_three_views_recover_the_plane_depth():
    images, cal = _three_view_scene()
    tool = Triview3DTool()
    tool.load({})
    percepts = tool.perceive(
        _context(images, cal, stereo_num_disparities=256), []
    )
    assert len(percepts) == 1, "triview should answer when all pairs work"
    p = percepts[0]
    assert p.status == "ok", p.message
    assert p.geometry is not None
    # The plane is at y = 1.0; the reported centroid must sit there, not at a fraction of it.
    assert abs(p.geometry.centroid[1] - PLANE_Y) < 0.06, (
        f"centroid y = {p.geometry.centroid[1]:.3f} for a plane at {PLANE_Y}"
    )
    assert p.extras["pairs_used"] == 3
    assert p.extras["degraded_to_two_view"] is False


def test_all_three_pairs_are_reported():
    images, cal = _three_view_scene()
    tool = Triview3DTool()
    tool.load({})
    p = tool.perceive(_context(images, cal, stereo_num_disparities=256), [])[0]
    pairs = {row["pair"] for row in p.extras["per_pair"]}
    assert pairs == {"cam0-cam1", "cam0-cam2", "cam1-cam2"}
    # Each pair reports its own baseline, so the merge is auditable.
    baselines = {row["pair"]: row["baseline_m"] for row in p.extras["per_pair"]}
    assert baselines["cam0-cam2"] == pytest.approx(0.6, rel=0.02)
    assert baselines["cam0-cam1"] == pytest.approx(0.3, rel=0.02)


def test_the_longer_baseline_pair_is_more_precise():
    """Sigma falls with baseline, which is what justifies weighting the merge by 1/sigma^2."""
    images, cal = _three_view_scene()
    tool = Triview3DTool()
    tool.load({})
    p = tool.perceive(_context(images, cal, stereo_num_disparities=256), [])[0]
    sigma = {row["pair"]: row["median_sigma_z_m"] for row in p.extras["per_pair"]}
    assert sigma["cam0-cam2"] < sigma["cam0-cam1"]
    assert sigma["cam0-cam2"] < sigma["cam1-cam2"]


def test_views_used_reflects_the_cameras_that_contributed():
    images, cal = _three_view_scene()
    tool = Triview3DTool()
    tool.load({})
    p = tool.perceive(_context(images, cal, stereo_num_disparities=256), [])[0]
    assert p.geometry.views_used == 3


def test_limiting_pairs_is_reported_as_degradation():
    """``triview_max_pairs=1`` must mark the result as not actually three-way."""
    images, cal = _three_view_scene()
    tool = Triview3DTool()
    tool.load({})
    p = tool.perceive(
        _context(images, cal, stereo_num_disparities=256, triview_max_pairs=1), []
    )[0]
    assert p.extras["pairs_used"] == 1
    assert p.extras["degraded_to_two_view"] is True
    assert "triview_degraded_to_subset_of_pairs" in p.extras["warnings"]


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_two_views_are_refused():
    """Two views belong to stereo_3d; a triview tool that accepted them would blur the boundary."""
    cal = {"a": _calibration("a", 0.0), "b": _calibration("b", 0.3)}
    images = {"a": _plane_image(0.0), "b": _plane_image(0.3)}
    tool = Triview3DTool()
    tool.load({})
    assert tool.perceive(_context(images, cal), []) == []
    assert ERR_NEEDS_THREE_VIEWS == "triview_needs_three_calibrated_views"


def test_four_views_are_refused():
    cal = {f"c{i}": _calibration(f"c{i}", 0.3 * i) for i in range(4)}
    images = {f"c{i}": _plane_image(0.3 * i) for i in range(4)}
    tool = Triview3DTool()
    tool.load({})
    assert tool.perceive(_context(images, cal), []) == []


def test_unconfigured_mvs_backend_is_refused():
    """The native-MVS route must report itself unavailable, not fall back to pairwise."""
    images, cal = _three_view_scene()
    tool = Triview3DTool()
    tool.load({})
    assert tool.perceive(_context(images, cal, triview_method="mvs"), []) == []
    assert ERR_BACKEND_NOT_CONFIGURED == "backend_not_configured"


def test_when_no_pair_works_the_tool_returns_nothing():
    """Blank images carry no disparity, so every pair must fail and the tool must say so."""
    cal = {
        "a": _calibration("a", 0.0),
        "b": _calibration("b", 0.3),
        "c": _calibration("c", 0.6),
    }
    blank = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    images = {"a": blank, "b": blank, "c": blank}
    tool = Triview3DTool()
    tool.load({})
    assert tool.perceive(_context(images, cal), []) == []
    assert ERR_NO_USABLE_PAIR == "no_usable_pair"


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


def test_capabilities_describe_a_geometry_producer():
    caps = Triview3DTool().capabilities()
    assert caps.kind == "geometry"
    assert caps.multi_view is True
    assert caps.prompts == set()


def test_name_is_declared():
    assert Triview3DTool().name == TOOL_NAME == "triview_3d"
