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
from percepta.tools import triview_3d as triview
from percepta.tools.triview_3d import (
    ERR_BACKEND_NOT_CONFIGURED,
    ERR_NEEDS_THREE_VIEWS,
    ERR_NO_USABLE_PAIR,
    TOOL_NAME,
    Triview3DTool,
)

cv2 = pytest.importorskip("cv2", reason="triview pair tests require optional OpenCV")

FOCAL = 397.0413
WIDTH, HEIGHT = 320, 240
PLANE_Y = 1.0


def _rotation() -> np.ndarray:
    """World <- camera for a camera looking along +world-y, +x right, +y down."""
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
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
    world_z = -(ys - HEIGHT / 2) / FOCAL * plane_y
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


@pytest.mark.parametrize(
    "options",
    [
        {"stereo_sigma_d_px": -1}, {"stereo_sigma_d_px": 0}, {"stereo_sigma_d_px": "nan"},
        {"stereo_sigma_d_px": True}, {"min_valid_fraction": -0.1}, {"min_valid_fraction": 1.1},
        {"stereo_num_disparities": 17}, {"stereo_num_disparities": "256"}, {"stereo_block_size": -3},
        {"stereo_block_size": 10_000}, {"triview_max_pairs": 0}, {"triview_max_pairs": 4},
        {"triview_max_pairs": 2.5}, {"triview_max_pairs": True},
    ],
)
def test_invalid_numeric_options_are_rejected_before_pair_matching(options, monkeypatch):
    images, calibration = _three_view_scene()
    monkeypatch.setattr(triview, "_pair_points", lambda *a, **kw: pytest.fail("invalid options reached pair matcher"))
    result = Triview3DTool().perceive(_context(images, calibration, **options), [])
    assert result and result[0].status == "invalid_request"
    assert result[0].extras["code"] == "invalid_geometry_options"


def test_triview_rejects_images_without_a_legal_sgbm_range(monkeypatch):
    _, calibration = _three_view_scene()
    images = {name: np.zeros((24, 8, 3), np.uint8) for name in calibration}
    calibration = {
        name: cal.model_copy(update={"width": 8, "height": 24, "K": [[50, 0, 4], [0, 50, 12], [0, 0, 1]]})
        for name, cal in calibration.items()
    }
    monkeypatch.setattr(triview, "_pair_points", lambda *a, **kw: pytest.fail("unsupported image reached pair matcher"))
    result = Triview3DTool().perceive(_context(images, calibration), [])
    assert result and result[0].status == "invalid_request"
    assert result[0].extras["code"] == "sgbm_image_too_small"


def test_inverse_variance_centroid_survives_a_large_finite_noise_scale():
    images, calibration = _three_view_scene()
    tool = Triview3DTool()
    normal = tool.perceive(_context(images, calibration, stereo_num_disparities=256), [])[0]
    large = tool.perceive(_context(images, calibration, stereo_num_disparities=256, stereo_sigma_d_px=1e200), [])[0]
    assert normal.status == large.status == "ok"
    np.testing.assert_allclose(large.geometry.centroid, normal.geometry.centroid, atol=1e-10)
    assert np.isfinite(large.geometry.uncertainty_m)
    assert large.geometry.uncertainty_m == pytest.approx(normal.geometry.uncertainty_m * 1e200)


def test_per_pair_sigma_medians_remain_finite(monkeypatch):
    images, calibration = _three_view_scene()
    disparity = np.full((HEIGHT, WIDTH), np.sqrt(FOCAL * 0.6))
    monkeypatch.setattr(triview, "_sgbm_disparity", lambda *a, **kw: (disparity, np.ones(disparity.shape, bool)))
    result = Triview3DTool().perceive(_context(images, calibration, stereo_sigma_d_px=1e308), [])[0]
    assert result.status == "ok"
    assert np.isfinite(result.geometry.uncertainty_m)
    assert all(np.isfinite(pair["median_sigma_z_m"]) for pair in result.extras["per_pair"])


def test_nonfinite_computed_uncertainty_is_an_invalid_request(monkeypatch):
    images, calibration = _three_view_scene()
    monkeypatch.setattr(
        triview, "_sgbm_disparity", lambda a, b, **kw: (np.full(a.shape[:2], 0.0625), np.ones(a.shape[:2], bool))
    )
    result = Triview3DTool().perceive(_context(images, calibration, stereo_sigma_d_px=1e308), [])
    assert result and result[0].status == "invalid_request"
    assert result[0].extras["code"] == "invalid_geometry_options"


def test_two_pairs_still_use_all_three_cameras():
    images, calibration = _three_view_scene()
    result = Triview3DTool().perceive(
        _context(images, calibration, stereo_num_disparities=256, triview_max_pairs=2), []
    )[0]
    assert result.geometry.views_used == 3
    assert result.extras["pairs_used"] == 2
    assert result.extras["degraded_to_two_view"] is False
    assert "triview_degraded_to_subset_of_pairs" in result.extras["warnings"]


# ---------------------------------------------------------------------------
# The fusion itself
# ---------------------------------------------------------------------------


def test_three_views_recover_the_plane_depth():
    images, cal = _three_view_scene()
    tool = Triview3DTool()
    tool.load({})
    percepts = tool.perceive(_context(images, cal, stereo_num_disparities=256), [])
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
    p = tool.perceive(_context(images, cal, stereo_num_disparities=256, triview_max_pairs=1), [])[0]
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
    p = tool.perceive(_context(images, cal), [])[0]
    assert p.status == "invalid_request" and p.geometry is None
    assert p.extras["code"] == ERR_NEEDS_THREE_VIEWS


def test_four_views_are_refused():
    cal = {f"c{i}": _calibration(f"c{i}", 0.3 * i) for i in range(4)}
    images = {f"c{i}": _plane_image(0.3 * i) for i in range(4)}
    tool = Triview3DTool()
    tool.load({})
    p = tool.perceive(_context(images, cal), [])[0]
    assert p.status == "invalid_request" and p.geometry is None
    assert p.extras["code"] == ERR_NEEDS_THREE_VIEWS


def test_unconfigured_mvs_backend_is_refused():
    """The native-MVS route must report itself unavailable, not fall back to pairwise."""
    images, cal = _three_view_scene()
    tool = Triview3DTool()
    tool.load({})
    result = tool.perceive(_context(images, cal, triview_method="mvs"), [])[0]
    assert result.status == "unavailable" and result.extras["code"] == ERR_BACKEND_NOT_CONFIGURED
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


def test_fixture_pins_forward_and_down_axes():
    np.testing.assert_array_equal(_rotation() @ [0, 0, 1], [0, 1, 0])
    np.testing.assert_array_equal(_rotation() @ [0, 1, 0], [0, 0, -1])
    texture = np.random.default_rng(1234).integers(20, 235, (512, 512), dtype=np.int16)
    u, v = 201, 167
    x, z = (u - WIDTH / 2) / FOCAL, -(v - HEIGHT / 2) / FOCAL
    assert _plane_image(0)[v, u, 0] == texture[int((z * 160 + 256) % 512), int(x * 160 % 512)]


def test_pair_rectifies_both_images_and_recovers_off_axis_world_points(monkeypatch):
    a, b = _calibration("a", 0), _calibration("b", 0.2)
    T = np.array(b.T_world_cam)
    T[:3, :3] = T[:3, :3] @ cv2.Rodrigues(np.array([0.03, -0.02, 0.015]))[0]
    b.T_world_cam = T.tolist()
    images = [_plane_image(0), _plane_image(0.2)]
    K, R1, info = triview._rectify(a, b, (HEIGHT, WIDTH))
    expected = [
        cv2.remap(image, *info[key], cv2.INTER_LINEAR)
        for image, key in zip(images, ("map_a", "map_b"), strict=True)
    ]
    assert all(not np.array_equal(raw, rect) for raw, rect in zip(images, expected, strict=True))
    uv = np.array([[190, 90], [210, 130], [240, 155]])
    depth = np.array([0.8, 1.1, 1.3])
    d = np.zeros((HEIGHT, WIDTH))
    d[uv[:, 1], uv[:, 0]] = info["focal_rect_px"] * info["baseline_m"] / depth

    def matcher(left, right, **kwargs):
        np.testing.assert_array_equal(left, expected[0])
        np.testing.assert_array_equal(right, expected[1])
        return d, d > 0

    monkeypatch.setattr(triview, "_sgbm_disparity", matcher)
    result = triview._pair_points(
        a, b, *images, num_disparities=128, block_size=5, sigma_d=1, min_valid_fraction=0
    )
    assert result is not None
    rect_points = np.column_stack(
        ((uv[:, 0] - K[0, 2]) / K[0, 0] * depth, (uv[:, 1] - K[1, 2]) / K[1, 1] * depth, depth)
    )
    T_a = np.array(a.T_world_cam)
    expected_world = rect_points @ R1 @ T_a[:3, :3].T + T_a[:3, 3]
    np.testing.assert_allclose(result["points"], expected_world, atol=1e-10)
    # Independent world->camera->pixel projection recovers both row and column.
    camera = (result["points"] - T_a[:3, 3]) @ T_a[:3, :3]
    assert np.all(camera[:, 2] > 0)
    rect_camera = camera @ R1.T
    projected = rect_camera @ K.T
    np.testing.assert_allclose(projected[:, :2] / projected[:, 2, None], uv, atol=1e-10)


@pytest.mark.parametrize("disparity", [0.0, -1.0, np.nan, np.inf, WIDTH + 1.0])
def test_invalid_pair_depth_is_empty_not_fallback(disparity, monkeypatch):
    monkeypatch.setattr(
        triview,
        "_sgbm_disparity",
        lambda *a, **kw: (np.full((HEIGHT, WIDTH), disparity), np.ones((HEIGHT, WIDTH), bool)),
    )
    images, cal = _three_view_scene()
    assert Triview3DTool().perceive(_context(images, cal, min_valid_fraction=0), []) == []


def test_reversed_pairs_are_reported_not_automatically_swapped():
    images, cal = _three_view_scene()
    images["cam0"], images["cam2"] = images["cam2"], images["cam0"]
    cal["cam0"].T_world_cam, cal["cam2"].T_world_cam = (
        cal["cam2"].T_world_cam,
        cal["cam0"].T_world_cam,
    )
    p = Triview3DTool().perceive(_context(images, cal), [])[0]
    assert p.status == "invalid_request" and p.geometry is None
    assert p.extras["code"] == "reversed_stereo_order"
    assert len(p.extras["pair_failures"]) == 3


def test_malformed_calibration_is_not_hidden_by_other_pairs():
    images, cal = _three_view_scene()
    ctx = _context(images, cal)
    cal["cam2"].K[0][0] = 0
    p = Triview3DTool().perceive(ctx, [])[0]
    assert p.status == "invalid_request" and p.geometry is None
    assert p.extras["code"] == "invalid_calibration"


def test_extra_uncalibrated_image_is_not_ignored():
    images, cal = _three_view_scene()
    images["extra"] = images["cam0"]
    p = Triview3DTool().perceive(_context(images, cal), [])[0]
    assert p.status == "invalid_request" and p.geometry is None
    assert p.extras["code"] == ERR_NEEDS_THREE_VIEWS


def test_surface_summary_does_not_claim_calibrated_world_axis_uncertainty():
    images, cal = _three_view_scene()
    p = Triview3DTool().perceive(_context(images, cal, stereo_num_disparities=256), [])[0]
    assert p.extras["geometry_scope"] == "matched_visible_surface"
    assert "pair_correlation" in p.extras["uncertainty_excludes"]
    assert p.geometry.height_uncertainty_m is None
    assert p.geometry.lateral_uncertainty_m is None
    p.model_dump_json()
