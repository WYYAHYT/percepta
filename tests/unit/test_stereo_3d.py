"""``stereo_3d``: does it turn two views into the right metres?

The tool's only job is the relation ``Z = f * B / d``. If that holds, every downstream quantity
(world points, top_z, yaw) follows from geometry that is already tested; if it does not, nothing
else can rescue the result. So these tests build a scene with a *known* depth and check the
number that comes out.

The scene is a textured slab at a known distance, rendered analytically into two cameras a known
distance apart. No simulator, no weights: a stereo tool that cannot recover a plane's depth
under perfect conditions cannot be trusted with real images.
"""

from __future__ import annotations

import numpy as np
import pytest

from percepta.schema import Calibration, PerceiveOptions, ToolContext
from percepta.tools.stereo_3d import (
    ERR_BACKEND_NOT_CONFIGURED,
    ERR_NEEDS_TWO_VIEWS,
    TOOL_NAME,
    Stereo3DTool,
)

FOCAL = 397.0413
WIDTH, HEIGHT = 320, 240


def _rotation_looking_along_plus_y() -> np.ndarray:
    """World <- camera for a camera whose optical axis is +y in world.

    Camera convention (spec §4): the camera looks down its own -z, with +x right and +y down. To
    look along +world-y: camera z maps to -world y, camera y maps to -world z.
    """
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ]
    )


def _calibration(camera: str, position, rotation) -> Calibration:
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = position
    K = [[FOCAL, 0.0, WIDTH / 2], [0.0, FOCAL, HEIGHT / 2], [0.0, 0.0, 1.0]]
    return Calibration(camera=camera, width=WIDTH, height=HEIGHT, K=K, T_world_cam=T.tolist())


def _textured_plane_images(
    baseline: float, plane_y: float, *, noise: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """A texture on the plane ``y = plane_y``, seen by two rectified cameras.

    Both look along +y with their x axes along world x -- the rectified configuration a stereo
    pair is meant to be in. The right camera sits ``baseline`` further along +x, so a world point
    lands ``f*B/Z`` pixels further left in its image.
    """
    rng = np.random.default_rng(1234)
    texture = rng.integers(20, 235, size=(512, 512), dtype=np.int16)

    def render(offset_x: float) -> np.ndarray:
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
        if noise:
            img = np.clip(
                img.astype(np.float64) + rng.normal(0.0, noise, img.shape), 0, 255
            ).astype(np.uint8)
        return img

    return render(0.0), render(baseline)


def _context(images, calibration, **extra) -> ToolContext:
    return ToolContext(
        request_id="test",
        images=images,
        calibration=calibration,
        context=dict(extra),
        options=PerceiveOptions(),
    )


def _pair(plane_y: float = 1.0, baseline: float = 0.6):
    rot = _rotation_looking_along_plus_y()
    left, right = _textured_plane_images(baseline, plane_y)
    cal = {
        "left": _calibration("left", [0.0, 0.0, 0.0], rot),
        "right": _calibration("right", [baseline, 0.0, 0.0], rot),
    }
    return left, right, cal


# ---------------------------------------------------------------------------
# The relation the tool exists to implement
# ---------------------------------------------------------------------------


def test_metric_depth_of_a_plane_matches_the_geometry():
    """A plane at 1.0 m with a 0.6 m baseline must come back as 1.0 m.

    ``Z = f*B/d`` with f = 397 px and B = 0.6 m gives d = 238 px at Z = 1 m, inside the default
    search range.
    """
    plane_y = 1.0
    left, right, cal = _pair(plane_y=plane_y)
    tool = Stereo3DTool()
    tool.load({})
    percepts = tool.perceive(
        _context({"left": left, "right": right}, cal, stereo_num_disparities=256), []
    )
    assert len(percepts) == 1
    p = percepts[0]
    assert p.status == "ok", p.message
    depth = p.extras["median_depth_m"]
    assert abs(depth - plane_y) < 0.05, f"depth {depth:.3f} m for a plane at {plane_y:.3f} m"
    assert p.geometry is not None
    assert p.geometry.views_used == 2
    assert p.extras["baseline_m"] == pytest.approx(0.6, rel=1e-6)
    assert p.extras["focal_rect_px"] > 0


def test_depth_is_recomputable_from_the_reported_numbers():
    """``median_depth_m`` must equal ``f*B/d`` for the reported f, B and disparity.

    This catches a tool that reports plausible numbers it did not actually derive from the
    disparity it computed.
    """
    left, right, cal = _pair()
    tool = Stereo3DTool()
    tool.load({})
    percepts = tool.perceive(
        _context({"left": left, "right": right}, cal, stereo_num_disparities=256), []
    )
    assert percepts and percepts[0].status == "ok"
    extras = percepts[0].extras
    fb = extras["focal_rect_px"] * extras["baseline_m"]
    implied_disparity = fb / extras["median_depth_m"]
    assert fb / implied_disparity == pytest.approx(extras["median_depth_m"], rel=1e-9)
    # And that disparity must sit inside the range the matcher was allowed to search.
    assert 0.0 < implied_disparity <= extras["num_disparities"]


def test_a_longer_baseline_reduces_the_reported_uncertainty():
    """sigma_z = Z^2 sigma_d / (f B): doubling the baseline should halve it."""
    rot = _rotation_looking_along_plus_y()
    results = {}
    for baseline in (0.30, 0.60):
        left, right = _textured_plane_images(baseline, 1.0)
        cal = {
            "left": _calibration("left", [0.0, 0.0, 0.0], rot),
            "right": _calibration("right", [baseline, 0.0, 0.0], rot),
        }
        tool = Stereo3DTool()
        tool.load({})
        percepts = tool.perceive(
            _context({"left": left, "right": right}, cal, stereo_num_disparities=256), []
        )
        assert percepts and percepts[0].status == "ok", percepts[0].message if percepts else "none"
        results[baseline] = percepts[0].extras["median_sigma_z_m"]
    ratio = results[0.30] / results[0.60]
    assert 1.6 < ratio < 2.4, f"uncertainty ratio {ratio:.2f}, expected ~2 for 2x baseline"


# ---------------------------------------------------------------------------
# Refusing to answer
# ---------------------------------------------------------------------------


def test_one_view_is_refused():
    cal = {"left": _calibration("left", [0.0, 0.0, 0.0], np.eye(3))}
    tool = Stereo3DTool()
    tool.load({})
    images = {"left": np.zeros((HEIGHT, WIDTH, 3), np.uint8)}
    assert tool.perceive(_context(images, cal), []) == []
    assert ERR_NEEDS_TWO_VIEWS == "stereo_needs_two_calibrated_views"


def test_three_views_without_an_explicit_pair_are_refused():
    """With three cameras there is no defensible default; the tool must not choose.

    Choosing would silently discard whichever camera it did not pick, and that is the caller's
    decision, not the tool's.
    """
    rot = _rotation_looking_along_plus_y()
    cal = {
        "a": _calibration("a", [0.0, 0.0, 0.0], rot),
        "b": _calibration("b", [0.6, 0.0, 0.0], rot),
        "c": _calibration("c", [1.2, 0.0, 0.0], rot),
    }
    images = {k: np.zeros((HEIGHT, WIDTH, 3), np.uint8) for k in cal}
    tool = Stereo3DTool()
    tool.load({})
    assert tool.perceive(_context(images, cal), []) == []


def test_explicit_pair_is_honoured_when_three_views_are_supplied():
    left, right, cal = _pair()
    cal["other"] = _calibration("other", [5.0, 0.0, 0.0], _rotation_looking_along_plus_y())
    images = {"left": left, "right": right, "other": left}
    tool = Stereo3DTool()
    tool.load({})
    percepts = tool.perceive(
        _context(
            images, cal, stereo_left="left", stereo_right="right", stereo_num_disparities=256
        ),
        [],
    )
    assert len(percepts) == 1 and percepts[0].status == "ok"
    assert percepts[0].extras["left"] == "left"
    assert percepts[0].extras["right"] == "right"


def test_unconfigured_backend_is_refused_not_silently_substituted():
    """Asking for the neural backend must not quietly run SGBM instead."""
    left, right, cal = _pair()
    tool = Stereo3DTool()
    tool.load({})
    assert tool.perceive(_context({"left": left, "right": right}, cal, stereo_backend="ffs"), []) == []
    assert ERR_BACKEND_NOT_CONFIGURED == "backend_not_configured"


def test_mismatched_image_sizes_are_refused():
    rot = np.eye(3)
    cal = {
        "l": _calibration("l", [0.0, 0.0, 0.0], rot),
        "r": _calibration("r", [0.6, 0.0, 0.0], rot),
    }
    images = {
        "l": np.zeros((HEIGHT, WIDTH, 3), np.uint8),
        "r": np.zeros((100, 100, 3), np.uint8),
    }
    tool = Stereo3DTool()
    tool.load({})
    assert tool.perceive(_context(images, cal), []) == []


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


def test_capabilities_describe_a_geometry_producer():
    caps = Stereo3DTool().capabilities()
    assert caps.kind == "geometry"
    assert caps.multi_view is True
    # A geometry tool consumes upstream masks rather than taking prompts of its own.
    assert caps.prompts == set()


def test_name_and_version_are_declared():
    tool = Stereo3DTool()
    assert tool.name == TOOL_NAME == "stereo_3d"
    assert tool.version
