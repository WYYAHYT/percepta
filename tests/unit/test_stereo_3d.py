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
from percepta.tools import stereo_3d as stereo
from percepta.tools.stereo_3d import (
    ERR_BACKEND_NOT_CONFIGURED,
    ERR_NEEDS_TWO_VIEWS,
    TOOL_NAME,
    Stereo3DTool,
)

cv2 = pytest.importorskip("cv2", reason="stereo geometry tests require optional OpenCV")

FOCAL = 397.0413
WIDTH, HEIGHT = 320, 240


def _rotation_looking_along_plus_y() -> np.ndarray:
    """World <- camera for a camera whose optical axis is +y in world.

    OpenCV +z is forward: camera z maps to +world y, camera y to -world z.
    """
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
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
        world_z = -(ys - HEIGHT / 2) / FOCAL * plane_y
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


@pytest.mark.parametrize(
    "options",
    [
        {"stereo_sigma_d_px": -1}, {"stereo_sigma_d_px": 0}, {"stereo_sigma_d_px": "nan"},
        {"stereo_sigma_d_px": True}, {"min_valid_fraction": -0.1}, {"min_valid_fraction": 1.1},
        {"min_valid_fraction": None}, {"stereo_num_disparities": 0}, {"stereo_num_disparities": 17},
        {"stereo_num_disparities": "256"}, {"stereo_num_disparities": 256.5},
        {"stereo_block_size": -3}, {"stereo_block_size": 10_000}, {"stereo_block_size": True},
    ],
)
def test_invalid_numeric_options_are_rejected_before_matching(options, monkeypatch):
    left, right, calibration = _pair()
    monkeypatch.setattr(stereo, "_sgbm_disparity", lambda *a, **kw: pytest.fail("invalid options reached matcher"))
    result = Stereo3DTool().perceive(_context({"left": left, "right": right}, calibration, **options), [])
    _assert_invalid(result, "invalid_geometry_options")


@pytest.mark.parametrize("width", [1, 8, 16])
def test_sgbm_rejects_images_without_a_legal_disparity_range(width, monkeypatch):
    _, _, calibration = _pair()
    images = {name: np.zeros((24, width, 3), np.uint8) for name in calibration}
    calibration = {
        name: cal.model_copy(update={"width": width, "height": 24, "K": [[50, 0, width / 2], [0, 50, 12], [0, 0, 1]]})
        for name, cal in calibration.items()
    }
    monkeypatch.setattr(stereo, "_sgbm_disparity", lambda *a, **kw: pytest.fail("unsupported image reached matcher"))
    _assert_invalid(Stereo3DTool().perceive(_context(images, calibration), []), "sgbm_image_too_small")


def test_smallest_supported_width_preserves_legal_option_normalization(monkeypatch):
    _, _, calibration = _pair()
    images = {name: np.zeros((24, 17, 3), np.uint8) for name in calibration}
    calibration = {name: cal.model_copy(update={"width": 17, "height": 24}) for name, cal in calibration.items()}
    calls = []

    def match(left, right, **options):
        calls.append(options)
        return np.zeros(left.shape[:2], np.float32), np.zeros(left.shape[:2], bool)

    monkeypatch.setattr(stereo, "_sgbm_disparity", match)
    result = Stereo3DTool().perceive(
        _context(images, calibration, stereo_num_disparities=32.0, stereo_block_size=4.0, min_valid_fraction=0), []
    )
    assert result == []
    assert calls == [{"num_disparities": 16, "block_size": 5}]


def test_finite_sigma_median_does_not_overflow(monkeypatch):
    left, right, calibration = _pair()
    disparity = np.full(left.shape[:2], np.sqrt(FOCAL * 0.6))
    monkeypatch.setattr(stereo, "_sgbm_disparity", lambda *a, **kw: (disparity, np.ones(disparity.shape, bool)))
    result = Stereo3DTool().perceive(
        _context({"left": left, "right": right}, calibration, stereo_sigma_d_px=1e308), []
    )[0]
    assert result.status == "ok"
    assert np.isfinite(result.geometry.uncertainty_m)
    assert result.geometry.uncertainty_m == pytest.approx(1e308)
    smallest = np.nextafter(0.0, 1.0)
    assert stereo._median_sigma(np.full(2, smallest)) == smallest


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


def test_recovered_points_lie_in_front_of_the_camera_not_behind_it():
    """The camera looks down +z (spec §4); a positive depth must land on that side.

    This is the test that catches the sign error the earlier assertions missed: they only
    checked |depth|, so a point at -1.0 m passed a check for a plane at 1.0 m. Here the side of
    the camera is asserted directly.
    """
    left, right, cal = _pair(plane_y=1.0)
    tool = Stereo3DTool()
    tool.load({})
    percepts = tool.perceive(
        _context({"left": left, "right": right}, cal, stereo_num_disparities=256), []
    )
    assert percepts and percepts[0].status == "ok"
    geometry = percepts[0].geometry
    # The cameras look along +world-y from the origin, so the object must be at +y.
    assert geometry.centroid[1] > 0.0, (
        f"centroid y = {geometry.centroid[1]:.3f}: geometry landed behind the cameras"
    )
    assert geometry.centroid[1] == pytest.approx(1.0, abs=0.06)


# ---------------------------------------------------------------------------
# Refusing to answer
# ---------------------------------------------------------------------------


def test_one_view_is_refused():
    cal = {"left": _calibration("left", [0.0, 0.0, 0.0], np.eye(3))}
    tool = Stereo3DTool()
    tool.load({})
    images = {"left": np.zeros((HEIGHT, WIDTH, 3), np.uint8)}
    _assert_invalid(tool.perceive(_context(images, cal), []))
    assert ERR_NEEDS_TWO_VIEWS == "stereo_needs_two_calibrated_views"


def test_three_views_are_refused_outright():
    """stereo_3d takes exactly two views; three belong to triview_3d.

    Accepting three would force the tool to pick a pair, discarding a camera the caller chose to
    pass. Delegating that decision to another tool keeps the boundary where the caller put it.
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
    _assert_invalid(tool.perceive(_context(images, cal), []))


def test_two_views_are_used_without_being_named():
    """With exactly two views the pair is unambiguous, so no stereo_left/right is needed."""
    left, right, cal = _pair()
    tool = Stereo3DTool()
    tool.load({})
    percepts = tool.perceive(
        _context({"left": left, "right": right}, cal, stereo_num_disparities=256), []
    )
    assert len(percepts) == 1 and percepts[0].status == "ok"
    assert {percepts[0].extras["left"], percepts[0].extras["right"]} == {"left", "right"}


def test_unconfigured_backend_is_refused_not_silently_substituted():
    """Asking for the neural backend must not quietly run SGBM instead."""
    left, right, cal = _pair()
    tool = Stereo3DTool()
    tool.load({})
    p = tool.perceive(_context({"left": left, "right": right}, cal, stereo_backend="ffs"), [])[0]
    assert p.status == "unavailable"
    assert p.geometry is None
    assert p.extras["code"] == ERR_BACKEND_NOT_CONFIGURED
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
    _assert_invalid(tool.perceive(_context(images, cal), []))


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


def _assert_invalid(percepts, code=None):
    assert len(percepts) == 1
    p = percepts[0]
    assert p.status == "invalid_request", p.message
    assert p.geometry is None
    assert p.message
    if code is not None:
        assert p.extras["code"] == code
    p.model_dump_json()


def _rotated_pair():
    rotation = cv2.Rodrigues(np.array([0.07, -0.12, 0.05]))[0]
    centre = np.array([0.2, -0.1, 0.3])
    left = _calibration("left", centre, rotation)
    right = _calibration(
        "right",
        centre + rotation @ np.array([0.18, 0.015, 0.01]),
        rotation @ cv2.Rodrigues(np.array([0.025, -0.035, 0.02]))[0],
    )
    right.K = [[380.0, 0.0, 151.0], [0.0, 390.0, 117.0], [0.0, 0.0, 1.0]]
    return left, right


def test_fixture_and_renderer_use_opencv_axes():
    rotation = _rotation_looking_along_plus_y()
    np.testing.assert_array_equal(rotation @ [0, 0, 1], [0, 1, 0])
    np.testing.assert_array_equal(rotation @ [0, 1, 0], [0, 0, -1])
    left, _, _ = _pair()
    texture = np.random.default_rng(1234).integers(20, 235, (512, 512), dtype=np.int16)
    u, v = 201, 167
    x, z = (u - WIDTH / 2) / FOCAL, -(v - HEIGHT / 2) / FOCAL
    assert left[v, u, 0] == texture[int((z * 160 + 256) % 512), int(x * 160 % 512)]


def test_unprojection_pins_plus_z_and_off_axis_signs():
    K = np.array([[100.0, 0, 37], [0, 120.0, 29], [0, 0, 1]])
    points = stereo._rectified_to_world(
        [[37, 29], [47, 41], [17, 5]],
        [2, 2, 3],
        K,
        np.eye(3),
        np.eye(4),
    )
    np.testing.assert_allclose(points, [[0, 0, 2], [0.2, 0.2, 2], [-0.6, -0.6, 3]])


def test_rotated_world_camera_pixel_roundtrip_and_signed_projections(monkeypatch):
    a, b = _rotated_pair()
    Ta, Tb = np.array(a.T_world_cam), np.array(b.T_world_cam)
    original = cv2.stereoRectify

    def checked_rectify(Ka, Da, Kb, Db, size, R, t, **kwargs):
        np.testing.assert_allclose(R, Tb[:3, :3].T @ Ta[:3, :3], atol=1e-12)
        np.testing.assert_allclose(t.ravel(), Tb[:3, :3].T @ (Ta[:3, 3] - Tb[:3, 3]), atol=1e-12)
        return original(Ka, Da, Kb, Db, size, R, t, **kwargs)

    monkeypatch.setattr(cv2, "stereoRectify", checked_rectify)
    K_rect, R1, info = stereo._rectify(a, b, (HEIGHT, WIDTH))
    world = np.array([[0.12, -0.22, 2.1], [0.36, 0.03, 2.4], [-0.1, -0.08, 1.7]])
    rectified = []
    for cal, T, R, P in ((a, Ta, R1, info["P1"]), (b, Tb, info["R_b_to_rect"], info["P2"])):
        camera = (world - T[:3, 3]) @ T[:3, :3]
        assert np.all(camera[:, 2] > 0)
        K = np.array(cal.K)
        raw_uv = (camera @ K.T)[:, :2] / camera[:, 2, None]
        rect_uv = cv2.undistortPoints(raw_uv[:, None, :], K, None, R=R, P=P).reshape(-1, 2)
        rectified.append(rect_uv)
    np.testing.assert_allclose(rectified[0][:, 1], rectified[1][:, 1], atol=1e-8)
    d = rectified[0][:, 0] - rectified[1][:, 0]
    assert np.all(d > 0)
    depth = info["focal_rect_px"] * info["baseline_m"] / d
    np.testing.assert_allclose(depth, ((world - Ta[:3, 3]) @ Ta[:3, :3] @ R1.T)[:, 2])
    recovered = stereo._rectified_to_world(rectified[0], depth, K_rect, R1, Ta)
    np.testing.assert_allclose(recovered, world, atol=1e-9)
    q_points = np.column_stack((rectified[0], d, np.ones(len(d)))) @ info["Q"].T
    np.testing.assert_allclose(q_points[:, 2] / q_points[:, 3], depth)
    assert info["P2"][0, 3] < 0
    assert info["baseline_m"] == pytest.approx(np.linalg.norm(Ta[:3, 3] - Tb[:3, 3]))


def test_both_images_and_left_mask_are_remapped_before_matching(monkeypatch):
    a, b = _rotated_pair()
    rng = np.random.default_rng(5)
    left = rng.integers(0, 256, (HEIGHT, WIDTH, 3), dtype=np.uint8)
    right = rng.integers(0, 256, left.shape, dtype=np.uint8)
    mask = np.zeros((HEIGHT, WIDTH), bool)
    mask[35:91, 175:239] = True
    K, _, info = stereo._rectify(a, b, (HEIGHT, WIDTH))
    expected_l = cv2.remap(left, *info["map_a"], cv2.INTER_LINEAR)
    expected_r = cv2.remap(right, *info["map_b"], cv2.INTER_LINEAR)
    expected_mask = cv2.remap(mask.astype(np.uint8), *info["map_a"], cv2.INTER_NEAREST).astype(bool)
    assert not np.array_equal(left, expected_l)
    assert not np.array_equal(right, expected_r)
    assert not np.array_equal(mask, expected_mask)
    _, _, actual_mask = stereo._remap_pair(left, right, info, mask)
    np.testing.assert_array_equal(actual_mask, expected_mask)

    def matcher(rect_left, rect_right, **kwargs):
        np.testing.assert_array_equal(rect_left, expected_l)
        np.testing.assert_array_equal(rect_right, expected_r)
        return np.full((HEIGHT, WIDTH), 12.5), np.ones((HEIGHT, WIDTH), bool)

    monkeypatch.setattr(stereo, "_sgbm_disparity", matcher)
    p = Stereo3DTool().perceive(
        _context(
            {"left": left, "right": right},
            {"left": a, "right": b},
            stereo_mask={"left": mask},
            min_valid_fraction=0,
        ),
        [],
    )[0]
    assert p.status == "ok"
    rows, cols = np.indices(mask.shape)
    right_u = cols - 12.5
    valid = expected_mask & info["support_a"] & (right_u >= 0)
    lo = np.clip(np.floor(right_u).astype(int), 0, WIDTH - 1)
    hi = np.clip(np.ceil(right_u).astype(int), 0, WIDTH - 1)
    valid &= info["support_b"][rows, lo] & info["support_b"][rows, hi]
    assert p.extras["points"] == valid.sum()
    v, u = np.nonzero(valid)
    depth = np.full(u.size, info["focal_rect_px"] * info["baseline_m"] / 12.5)
    world = stereo._rectified_to_world(
        np.column_stack((u, v)), depth, K, info["R_a_to_rect"], a.T_world_cam
    )
    expected_centre = np.percentile(world, [1, 99], axis=0).mean(axis=0)
    np.testing.assert_allclose(p.geometry.centroid, expected_centre)
    assert p.extras["geometry_scope"] == "matched_visible_surface"
    assert p.geometry.height_uncertainty_m is None
    assert p.geometry.lateral_uncertainty_m is None
    p.model_dump_json()


def test_validity_requires_both_supports_and_subpixel_right_correspondence():
    a = _calibration("left", [0, 0, 0], np.eye(3))
    b = _calibration("right", [0.1, 0, 0], np.eye(3))
    shape = (4, 8)
    info = {
        "support_a": np.ones(shape, bool),
        "support_b": np.ones(shape, bool),
        "focal_rect_px": 4.0,
        "baseline_m": 0.1,
        "R_a_to_rect": np.eye(3),
    }
    info["support_a"][0, 3] = False
    info["support_b"][0, 5] = False
    info["support_b"][1, 7] = False
    info["support_b"][3, 6] = False  # right support is tested at u-d, not u
    d = np.zeros(shape)
    d[0, 3], d[0, 6], d[1, 0], d[1, 7] = 1, 1, 1, 0.5
    d[2, :4] = [np.nan, np.inf, -1, 0]
    d[3, 6] = 2
    K = np.array([[4.0, 0, 3], [0, 4.0, 2], [0, 0, 1]])
    points, depth, valid = stereo._disparity_points(d, np.ones(shape, bool), a, b, K, info)
    assert valid.sum() == 1 and valid[3, 6]
    np.testing.assert_allclose(depth, [0.2])
    np.testing.assert_allclose(points, [[0.15, 0.05, 0.2]])


@pytest.mark.parametrize("behind", ["left", "right"])
def test_points_behind_either_original_camera_are_rejected(behind):
    a = _calibration("left", [0, 0, 0], np.eye(3))
    b = _calibration("right", [0.1, 0, 0], np.eye(3))
    info = {
        "support_a": np.ones((2, 4), bool),
        "support_b": np.ones((2, 4), bool),
        "focal_rect_px": 4.0,
        "baseline_m": 0.1,
        "R_a_to_rect": np.eye(3),
    }
    if behind == "left":
        info["R_a_to_rect"] = np.diag([1, -1, -1])
    else:
        b.T_world_cam = np.diag([1.0, -1.0, -1.0, 1.0]).tolist()
    points, depth, valid = stereo._disparity_points(
        np.ones((2, 4)),
        np.ones((2, 4), bool),
        a,
        b,
        np.eye(3),
        info,
    )
    assert points.shape == (0, 3) and depth.size == 0 and not valid.any()


@pytest.mark.parametrize(
    "options",
    [
        {"stereo_left": "left"},
        {"stereo_right": "right"},
        {"stereo_left": "left", "stereo_right": "left"},
        {"stereo_left": "unknown", "stereo_right": "right"},
        {"stereo_left": [], "stereo_right": "right"},
        {"stereo_left": None, "stereo_right": "right"},
    ],
)
def test_invalid_explicit_names_are_structured_errors(options):
    left, right, cal = _pair()
    _assert_invalid(
        Stereo3DTool().perceive(_context({"left": left, "right": right}, cal, **options), []),
        stereo.ERR_INVALID_ORDER,
    )


def test_explicit_order_uses_exact_pair_without_auto_swapping():
    left, right, cal = _pair()
    images = {"z_left": left, "a_right": right}
    cal = {
        "z_left": cal["left"].model_copy(update={"camera": "z_left"}),
        "a_right": cal["right"].model_copy(update={"camera": "a_right"}),
    }
    tool = Stereo3DTool()
    _assert_invalid(tool.perceive(_context(images, cal), []), stereo.ERR_REVERSED_ORDER)
    p = tool.perceive(
        _context(
            images, cal, stereo_left="z_left", stereo_right="a_right", stereo_num_disparities=256
        ),
        [],
    )[0]
    assert p.status == "ok" and p.extras["left"] == "z_left"
    assert p.geometry.centroid[1] == pytest.approx(1.0, abs=0.05)


@pytest.mark.parametrize("extra", ["image", "calibration", "both"])
def test_extra_supplied_views_are_not_filtered_away(extra):
    left, right, cal = _pair()
    images = {"left": left, "right": right}
    if extra in ("image", "both"):
        images["third"] = left
    if extra in ("calibration", "both"):
        cal["third"] = _calibration("third", [1, 0, 0], np.eye(3))
    _assert_invalid(
        Stereo3DTool().perceive(
            _context(images, cal, stereo_left="left", stereo_right="right"), []
        ),
        ERR_NEEDS_TWO_VIEWS,
    )


@pytest.mark.parametrize(
    "fault",
    [
        "zero_focal",
        "negative_focal",
        "nan",
        "skew",
        "K_row",
        "scale",
        "reflection",
        "T_row",
        "T_nan",
        "size",
        "name",
        "coincident",
    ],
)
def test_malformed_calibrations_are_structured_errors(fault):
    left, right, cal = _pair()
    ctx = _context({"left": left, "right": right}, cal)
    a = cal["left"]
    if fault == "zero_focal":
        a.K[0][0] = 0
    elif fault == "negative_focal":
        a.K[1][1] = -1
    elif fault == "nan":
        a.K[0][0] = float("nan")
    elif fault == "skew":
        a.K[0][1] = 1
    elif fault == "K_row":
        a.K[2][2] = 2
    elif fault == "scale":
        a.T_world_cam[0][0] = 2
    elif fault == "reflection":
        a.T_world_cam[0][0] = -1
    elif fault == "T_row":
        a.T_world_cam[3][0] = 1
    elif fault == "T_nan":
        a.T_world_cam[0][3] = float("nan")
    elif fault == "size":
        a.width += 1
    elif fault == "name":
        a.camera = "wrong"
    elif fault == "coincident":
        cal["right"].T_world_cam = a.T_world_cam
    _assert_invalid(
        Stereo3DTool().perceive(ctx, []),
        stereo.ERR_INVALID_CALIBRATION,
    )


@pytest.mark.parametrize(
    "fault", ["negative_focal", "zero_focal", "nonfinite", "zero_baseline", "baseline_overflow"]
)
def test_invalid_rectification_outputs_are_rejected(fault, monkeypatch):
    left, right, cal = _pair()
    original = cv2.stereoRectify

    def broken_rectify(*args, **kwargs):
        R1, R2, P1, P2, Q, roi1, roi2 = original(*args, **kwargs)
        if fault == "negative_focal":
            P1[0, 0] = P2[0, 0] = -1
        elif fault == "zero_focal":
            P1[0, 0] = P2[0, 0] = 0
        elif fault == "nonfinite":
            Q[0, 0] = np.nan
        elif fault == "zero_baseline":
            P2[0, 3] = 0
        else:
            P1[0, 0] = P2[0, 0] = 1e-308
            P2[0, 3] = -1e308
        return R1, R2, P1, P2, Q, roi1, roi2

    monkeypatch.setattr(cv2, "stereoRectify", broken_rectify)
    with np.errstate(over="ignore", invalid="ignore"):
        result = Stereo3DTool().perceive(_context({"left": left, "right": right}, cal), [])
    _assert_invalid(result, stereo.ERR_INVALID_CALIBRATION)


def test_vertical_rectification_is_rejected():
    cal = {
        "left": _calibration("left", [0, 0, 0], np.eye(3)),
        "right": _calibration("right", [0, 0.2, 0], np.eye(3)),
    }
    images = {name: np.zeros((HEIGHT, WIDTH, 3), np.uint8) for name in cal}
    _assert_invalid(
        Stereo3DTool().perceive(_context(images, cal), []), stereo.ERR_VERTICAL_RECTIFICATION
    )


@pytest.mark.parametrize(
    "mask",
    [
        np.ones((4, 4)),
        np.ones((HEIGHT, WIDTH, 1)),
        np.full((HEIGHT, WIDTH), np.nan),
        [[1], [0, 1]],
        np.full((HEIGHT, WIDTH), 2),
    ],
)
def test_invalid_left_masks_are_not_ignored(mask, monkeypatch):
    left, right, cal = _pair()
    monkeypatch.setattr(
        stereo, "_sgbm_disparity", lambda *a, **kw: pytest.fail("must validate mask first")
    )
    _assert_invalid(
        Stereo3DTool().perceive(
            _context({"left": left, "right": right}, cal, stereo_mask={"left": mask}), []
        ),
        stereo.ERR_INVALID_MASK,
    )


@pytest.mark.parametrize("disparity", [0.0, -1.0, np.nan, np.inf, WIDTH + 1.0])
def test_no_usable_depth_returns_empty_geometry_even_with_zero_gate(disparity, monkeypatch):
    left, right, cal = _pair()
    monkeypatch.setattr(
        stereo,
        "_sgbm_disparity",
        lambda *a, **kw: (np.full((HEIGHT, WIDTH), disparity), np.ones((HEIGHT, WIDTH), bool)),
    )
    assert (
        Stereo3DTool().perceive(
            _context({"left": left, "right": right}, cal, min_valid_fraction=0), []
        )
        == []
    )


def test_real_sgbm_on_nonparallel_cameras_recovers_world_plane():
    a, b = _rotated_pair()
    texture = np.random.default_rng(20).integers(0, 256, (1024, 1024), dtype=np.uint8)
    texture = cv2.GaussianBlur(texture, (3, 3), 0.6)

    def render(cal):
        T, K = np.array(cal.T_world_cam), np.array(cal.K)
        v, u = np.indices((HEIGHT, WIDTH))
        rays = np.stack(
            ((u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)), axis=-1
        )
        rays = rays @ T[:3, :3].T
        points = T[:3, 3] + rays * ((2.0 - T[2, 3]) / rays[..., 2])[..., None]
        grey = cv2.remap(
            texture,
            (points[..., 0] * 240 + 512).astype(np.float32),
            (points[..., 1] * 240 + 512).astype(np.float32),
            cv2.INTER_LINEAR,
        )
        return np.repeat(grey[..., None], 3, axis=2)

    p = Stereo3DTool().perceive(
        _context(
            {"left": render(a), "right": render(b)},
            {"left": a, "right": b},
            stereo_num_disparities=64,
        ),
        [],
    )[0]
    assert p.status == "ok"
    assert p.geometry.centroid[2] == pytest.approx(2.0, abs=0.04)
    assert p.extras["valid_fraction"] > 0.5
