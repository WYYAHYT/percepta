"""S3 — projection/unprojection round trips, plane math, bbox/percentile, PCA yaw.

Acceptance (spec §14): world -> pixel -> (depth) world error ``< 1e-6``; with a plane
assumption (no depth) ``< 1e-3``. Coordinates per §4: pixel ``[u, v] = [col, row]``, camera
OpenCV, world metres with ``T_world_cam`` camera -> world.
"""

from __future__ import annotations

import numpy as np
import pytest

from percepta import geometry as geo
from tests import synth

K = np.asarray(synth.intrinsics(640, 480), dtype=np.float64)  # f = 576, cx = 319.5, cy = 239.5
T_WORLD_CAM = np.asarray(
    synth.look_at((0.7, -1.1, 0.9), (0.1, 0.05, 0.1)), dtype=np.float64
)
PLANE_Z = 0.4


# ---------------------------------------------------------------------------
# Projection formulas, hand-computable
# ---------------------------------------------------------------------------


def test_project_points_matches_the_pinhole_formula():
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    points = np.array([[1.0, 2.0, 4.0], [0.0, 0.0, 2.0]])
    uv = geo.project_points(points, K)
    assert uv[0, 0] == pytest.approx(fx * 1.0 / 4.0 + cx)
    assert uv[0, 1] == pytest.approx(fy * 2.0 / 4.0 + cy)
    assert uv[1, 0] == pytest.approx(cx)
    assert uv[1, 1] == pytest.approx(cy)


def test_project_points_behind_the_camera_is_nan():
    uv = geo.project_points(np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 0.0]]), K)
    assert np.all(np.isnan(uv))


def test_unproject_pixels_matches_the_pinhole_formula():
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    uv = np.array([[cx + fx, cy + fy]])
    cam = geo.unproject_pixels(uv, np.array([2.0]), K)
    np.testing.assert_allclose(cam[0], [2.0, 2.0, 2.0], atol=1e-12)


def test_unproject_marks_invalid_depth_as_nan_rows():
    uv = np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]])
    cam = geo.unproject_pixels(uv, np.array([0.0, np.nan, -2.0]), K)
    assert np.all(np.isnan(cam))


@pytest.mark.parametrize(
    "intrinsics",
    [
        [[0, 0, 0], [0, 100, 0], [0, 0, 1]],
        [[100, 1, 0], [0, 100, 0], [0, 0, 1]],
        [[100, 0, 0], [0, 100, 0], [0, 0, 2]],
    ],
)
def test_projection_helpers_reject_unsupported_pinhole_intrinsics(intrinsics):
    with pytest.raises(ValueError, match="pinhole"):
        geo.project_points([[0, 0, 1]], intrinsics)
    with pytest.raises(ValueError, match="pinhole"):
        geo.unproject_pixels([[0, 0]], [1], intrinsics)
    with pytest.raises(ValueError, match="pinhole"):
        geo.rays_from_pixels([[0, 0]], intrinsics, T_WORLD_CAM)


@pytest.mark.parametrize("transform", [np.zeros((4, 4)), np.diag([-1.0, 1.0, 1.0, 1.0])])
def test_world_helpers_reject_nonrigid_camera_transforms(transform):
    with pytest.raises(ValueError, match="rigid"):
        geo.pixel_to_world([[0, 0]], [1], K, transform)
    with pytest.raises(ValueError, match="rigid"):
        geo.world_to_camera([[0, 0, 1]], transform)
    with pytest.raises(ValueError, match="rigid"):
        geo.rays_from_pixels([[0, 0]], K, transform)


# ---------------------------------------------------------------------------
# S3: round trips
# ---------------------------------------------------------------------------


def test_depth_round_trip_error_is_below_one_micrometre():
    """S3: world -> camera -> pixel -> world with true depth, error < 1e-6 m."""
    points_world = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.1, -0.2, 0.35],
            [-0.4, 0.3, 0.8],
            [0.25, 0.15, 1.6],
        ]
    )
    cam = geo.world_to_camera(points_world, T_WORLD_CAM)
    uv = geo.project_points(cam, K)
    back = geo.pixel_to_world(uv, cam[:, 2], K, T_WORLD_CAM)
    error = np.linalg.norm(back - points_world, axis=1)
    assert error.max() < 1e-6


def test_round_trip_against_the_shared_synth_projector():
    """The same round trip through ``tests.synth.project`` (an independent projector)."""
    points_world = np.array(
        [[0.0, 0.0, 0.0], [0.2, -0.1, 0.1], [-0.3, 0.25, 0.5], [0.1, 0.1, 1.2]]
    )
    uv = synth.project(points_world, K, T_WORLD_CAM)
    camera_frame = geo.world_to_camera(points_world, T_WORLD_CAM)
    recovered = geo.pixel_to_world(uv, camera_frame[:, 2], K, T_WORLD_CAM)
    np.testing.assert_allclose(recovered, points_world, atol=1e-6)


def test_plane_based_unprojection_error_is_below_one_millimetre():
    """S3 without depth: cast the pixel's ray onto the plane z = PLANE_Z, error < 1e-3 m."""
    plane = geo.plane_from_table_z(PLANE_Z)
    target = np.array([0.05, -0.1, PLANE_Z])
    uv = geo.project_points(geo.world_to_camera(target[None, :], T_WORLD_CAM), K)
    origin, dirs = geo.rays_from_pixels(uv, K, T_WORLD_CAM)
    hit = geo.ray_plane_intersection(origin, dirs, plane)[0]
    assert np.linalg.norm(hit - target) < 1e-3


# ---------------------------------------------------------------------------
# Rays and planes
# ---------------------------------------------------------------------------


def test_ray_horizontal_plane_hand_case():
    origin = np.array([0.0, 0.0, 1.0])
    hits = geo.ray_horizontal_plane(origin, np.array([[0.0, 0.0, -2.0]]), 0.0)
    np.testing.assert_allclose(hits[0], [0.0, 0.0, 0.0], atol=1e-12)


def test_ray_plane_intersection_hand_case_tilted_plane():
    # Plane x = 2 -> [1, 0, 0, -2]; ray from the origin along +x hits it at (2, 0, 0).
    plane = np.array([1.0, 0.0, 0.0, -2.0])
    hits = geo.ray_plane_intersection(
        np.array([0.0, 0.0, 0.0]), np.array([[1.0, 0.0, 0.0]]), plane
    )
    np.testing.assert_allclose(hits[0], [2.0, 0.0, 0.0], atol=1e-12)


def test_ray_plane_is_scale_invariant_and_broadcasts_origin():
    plane = np.array([0.0, 0.0, 3.0, -1.5])  # 3z - 1.5 = 0, i.e. z = 0.5
    dirs = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, -2.0]])
    hits = geo.ray_plane_intersection(np.array([0.1, 0.2, 1.0]), dirs, plane)
    np.testing.assert_allclose(hits[:, 2], [0.5, 0.5], atol=1e-12)
    np.testing.assert_allclose(hits[:, 0], [0.1, 0.1], atol=1e-12)


def test_parallel_and_backwards_rays_are_nan():
    origin = np.array([0.0, 0.0, 1.0])
    directions = np.array(
        [
            [1.0, 0.0, 0.0],  # parallel to z = 0
            [0.0, 0.0, 1.0],  # points away from the plane
        ]
    )
    hits = geo.ray_horizontal_plane(origin, directions, 0.0)
    assert np.all(np.isnan(hits))


def test_plane_parsing_normalises_and_rejects_degenerate_input():
    np.testing.assert_allclose(geo.plane_from_table_z(0.4), [0.0, 0.0, 1.0, -0.4])
    np.testing.assert_allclose(geo.parse_plane([0.0, 0.0, 2.0, -0.5]), [0.0, 0.0, 1.0, -0.25])
    parsed = geo.parse_plane(np.array([0.0, 3.0, 0.0, -0.6]))
    assert abs(np.linalg.norm(parsed[:3]) - 1.0) < 1e-12
    with pytest.raises(ValueError):
        geo.parse_plane([0.0, 0.0, 0.0, 1.0])
    with pytest.raises(ValueError):
        geo.parse_plane([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        geo.parse_plane([1.0, 0.0, 0.0, np.nan])


# ---------------------------------------------------------------------------
# Point clouds: bbox, percentile bbox, top z, spread
# ---------------------------------------------------------------------------


def test_bbox3d_from_points_and_top_z():
    points = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [0.5, 0.5, 0.5]])
    low, high = geo.bbox3d_from_points(points)
    assert low == [0.0, 0.0, 0.0]
    assert high == [1.0, 2.0, 3.0]
    assert geo.top_z_from_points(points) == 3.0
    with pytest.raises(ValueError):
        geo.bbox3d_from_points(np.zeros((0, 3)))


def test_percentile_bbox3d_rejects_depth_outliers():
    x = np.linspace(0.0, 0.99, 100)
    points = np.column_stack((x, np.zeros(100), np.zeros(100)))
    low, high = geo.percentile_bbox3d(points, lo=1, hi=99)
    assert high[0] < 0.995
    assert high[0] > 0.9
    assert low[0] < 0.02


def test_percentile_bbox3d_validates_bounds():
    with pytest.raises(ValueError):
        geo.percentile_bbox3d(np.zeros((3, 3)), lo=90, hi=10)


# ---------------------------------------------------------------------------
# PCA yaw
# ---------------------------------------------------------------------------


def _rotated_rectangle_points(angle_rad: float, width: float = 0.2, height: float = 0.1) -> np.ndarray:
    xs = np.linspace(-width / 2, width / 2, 11)
    ys = np.linspace(-height / 2, height / 2, 6)
    grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    rotation = np.array(
        [[np.cos(angle_rad), -np.sin(angle_rad)], [np.sin(angle_rad), np.cos(angle_rad)]]
    )
    return grid @ rotation.T


def test_yaw_from_points_recovers_a_rotated_rectangle():
    for degrees in (0.0, 25.0, -40.0, 89.0):
        points = _rotated_rectangle_points(np.deg2rad(degrees))
        recovered = np.rad2deg(geo.yaw_from_points(points))
        assert recovered == pytest.approx(degrees, abs=1e-6)


def test_yaw_is_folded_into_the_half_open_interval():
    points = _rotated_rectangle_points(np.deg2rad(100.0))
    recovered = geo.yaw_from_points(points)
    assert -np.pi / 2 < recovered <= np.pi / 2
    assert np.rad2deg(recovered) == pytest.approx(-80.0, abs=1e-6)
    # An axis is sign-ambiguous: +180 degrees is the same axis as 0 degrees.
    axis = _rotated_rectangle_points(np.deg2rad(180.0))
    assert abs(geo.yaw_from_points(axis)) < 1e-6


def test_yaw_of_a_square_is_finite_and_normalised():
    points = _rotated_rectangle_points(np.deg2rad(30.0), width=0.1, height=0.1)
    yaw = geo.yaw_from_points(points)
    assert np.isfinite(yaw)
    assert -np.pi / 2 <= yaw <= np.pi / 2


def test_yaw_degenerate_inputs():
    assert geo.yaw_from_points(np.zeros((1, 2))) == 0.0
    assert geo.yaw_from_points(np.full((5, 2), 0.3)) == 0.0


# ---------------------------------------------------------------------------
# Masks <-> pixels, hulls, polygons, depth filtering
# ---------------------------------------------------------------------------


def test_points_in_mask_uv_is_row_major_uv():
    mask = np.zeros((4, 5), dtype=bool)
    mask[1, 2] = True
    mask[3, 4] = True
    uv = geo.points_in_mask_uv(mask)
    np.testing.assert_allclose(uv, [[2.0, 1.0], [4.0, 3.0]])


def test_bbox3d_from_masked_world_points_round_trip():
    mask = np.zeros((6, 6), dtype=bool)
    mask[2:5, 3:5] = True
    depth = np.zeros((6, 6), dtype=np.float64)
    depth[mask] = 1.0
    points = geo.mask_to_world_points(mask, depth, K, T_WORLD_CAM)
    assert points.shape == (6, 3)
    low, high = geo.bbox3d_from_points(points)
    assert all(np.asarray(low) <= np.asarray(high))


def test_mask_to_world_points_skips_invalid_depth():
    mask = np.ones((2, 2), dtype=bool)
    depth = np.array([[1.0, np.nan], [0.0, -3.0]])
    points = geo.mask_to_world_points(mask, depth, K, T_WORLD_CAM)
    assert points.shape == (1, 3)
    camera_point = geo.world_to_camera(points, T_WORLD_CAM)[0]
    assert camera_point[2] == pytest.approx(1.0)


def test_mask_to_world_points_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        geo.mask_to_world_points(np.ones((2, 2), bool), np.ones((3, 3)), K, T_WORLD_CAM)


def test_convex_hull_is_counter_clockwise_and_drops_interior_points():
    corners = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])
    with_inside = np.vstack((corners, [[1.0, 1.0], [0.5, 1.5]]))
    hull = geo.convex_hull_2d(with_inside)
    assert hull.shape == (4, 2)
    assert geo.polygon_area(hull) > 0.0
    assert geo.polygon_area(hull) == pytest.approx(4.0)


def test_polygon_mask_rasterises_the_hull():
    hull = geo.convex_hull_2d(np.array([[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]]))
    mask = geo.polygon_mask(hull, (30, 30))
    assert mask[10, 10] and mask[20, 20]
    assert not mask[0, 0]
    assert 100 <= int(mask.sum()) <= 121


def test_convex_polygon_intersection_hand_cases():
    a = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])
    b = np.array([[1.0, 1.0], [3.0, 1.0], [3.0, 3.0], [1.0, 3.0]])
    overlap = geo.convex_polygon_intersection(a, b)
    assert geo.polygon_area(overlap) == pytest.approx(1.0)
    np.testing.assert_allclose(geo.polygon_centroid(overlap), [1.5, 1.5], atol=1e-9)
    disjoint = geo.convex_polygon_intersection(a, b + np.array([5.0, 5.0]))
    assert disjoint.shape == (0, 2)


def test_polygon_centroid_of_a_square():
    square = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]])
    np.testing.assert_allclose(geo.polygon_centroid(square), [2.0, 2.0], atol=1e-12)


def test_rays_from_pixels_are_unit_and_start_at_the_camera_centre():
    principal = np.array([[K[0, 2], K[1, 2]], [10.0, 470.0]])
    origin, dirs = geo.rays_from_pixels(principal, K, T_WORLD_CAM)
    np.testing.assert_allclose(origin, T_WORLD_CAM[:3, 3], atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(dirs, axis=1), [1.0, 1.0], atol=1e-12)
    # The principal ray must look along the camera's optical axis.
    np.testing.assert_allclose(
        dirs[0], T_WORLD_CAM[:3, 2] / np.linalg.norm(T_WORLD_CAM[:3, 2]), atol=1e-9
    )
