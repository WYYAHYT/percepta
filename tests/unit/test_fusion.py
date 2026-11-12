"""S5 — multi-view fusion of two calibrated views of a known cube (spec §9, §14).

Covers both §9 paths: the depth point cloud (§9.1) and the D4 silhouette/prism search
(§9.2), plus the hard constraints — no calibration never yields ``Geometry``, a single view
without depth is never confident, and every result is a world-frame geometry with an
uncertainty.
"""

from __future__ import annotations

import numpy as np
import pytest

from percepta.fusion import fuse
from percepta.schema import DepthRef, Geometry, PerceiveOptions, Percept
from tests import synth_geom

CENTROID_TOLERANCE_M = 0.02
HALF_EXTENT = 0.05
CUBE_CENTRE = (0.0, 0.0, 0.05)


def _scene(
    *,
    with_depth: bool = False,
    centre: tuple[float, float, float] = CUBE_CENTRE,
    half_extent: float = HALF_EXTENT,
    elevation_deg: float = 20.0,
    azimuths_deg: tuple[float, float] = (0.0, 180.0),
):
    calibrations, images, truth = synth_geom.two_view_roomcube(
        centre,
        half_extent,
        elevation_deg=elevation_deg,
        azimuths_deg=azimuths_deg,
        with_depth=with_depth,
    )
    masks = synth_geom.cube_masks(calibrations, truth, half_extent)
    percepts = synth_geom.mask_percepts(calibrations, masks)
    return calibrations, images, masks, percepts, np.asarray(truth, dtype=np.float64)


def _centroid_error(percept: Percept, truth: np.ndarray) -> float:
    assert percept.geometry is not None and percept.geometry.centroid is not None
    return float(np.linalg.norm(np.asarray(percept.geometry.centroid) - truth))


# ---------------------------------------------------------------------------
# §9.1 depth path
# ---------------------------------------------------------------------------


def test_two_views_with_depth_fuse_to_a_world_geometry():
    calibrations, _, _, percepts, truth = _scene(with_depth=True)
    fused = fuse(percepts, calibrations, {"table_z": 0.0})
    assert fused is not None and fused.geometry is not None
    geometry = fused.geometry
    assert geometry.frame == "world"
    assert geometry.views_used == 2
    assert geometry.uncertainty_m is not None and geometry.uncertainty_m > 0.0
    assert geometry.top_z == pytest.approx(0.10, abs=0.005)
    assert _centroid_error(fused, truth) < CENTROID_TOLERANCE_M
    assert fused.extras["method"] == "depth"
    assert np.isfinite(geometry.yaw_rad)


def test_depth_takes_priority_over_the_prism_search():
    calibrations, _, _, percepts, _ = _scene(with_depth=True)
    fused = fuse(percepts, calibrations, {})
    assert fused is not None
    assert fused.extras["method"] == "depth"
    assert fused.geometry is not None and fused.geometry.views_used == 2


def test_single_view_with_depth_is_a_valid_one_view_geometry():
    """One depth view yields geometry, but it is a 2.5-D surface: the median of the *visible*
    face is biased towards that face by roughly the object radius (spec §9.1's median centre).
    """
    calibrations, _, _, percepts, truth = _scene(with_depth=True)
    camera = sorted(calibrations)[0]
    fused = fuse([p for p in percepts if p.camera == camera], calibrations, {"table_z": 0.0})
    assert fused is not None and fused.geometry is not None
    assert fused.geometry.views_used == 1
    assert fused.geometry.uncertainty_m is not None
    error = _centroid_error(fused, truth)
    assert error < 0.06
    assert fused.geometry.top_z == pytest.approx(0.10, abs=0.01)


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("T_world_cam", np.zeros((4, 4)).tolist(), "rigid"),
        ("K", [[0, 0, 0], [0, 0, 0], [0, 0, 1]], "pinhole"),
    ],
)
def test_direct_fuse_revalidates_calibration_that_bypassed_schema(field, value, error):
    calibrations, _, _, percepts, _ = _scene(with_depth=True)
    camera = sorted(calibrations)[0]
    # model_copy and mutable fields can bypass Pydantic construction-time validation.
    calibrations[camera] = calibrations[camera].model_copy(update={field: value})
    with pytest.raises(ValueError, match=error):
        fuse(percepts, calibrations, {"table_z": 0.0})


# ---------------------------------------------------------------------------
# §9.2 / D4 prism (silhouette) path
# ---------------------------------------------------------------------------


def test_two_views_without_depth_use_the_prism_search():
    calibrations, _, _, percepts, truth = _scene()
    fused = fuse(percepts, calibrations, {"table_z": 0.0})
    assert fused is not None and fused.geometry is not None
    geometry = fused.geometry
    assert geometry.frame == "world"
    assert geometry.views_used == 2
    assert geometry.uncertainty_m is not None and geometry.uncertainty_m > 0.0
    assert _centroid_error(fused, truth) < CENTROID_TOLERANCE_M
    assert fused.extras["method"] == "prism"
    # D4: a height inside (table_z, table_z + prism_max_height_m].
    assert 0.0 < fused.extras["prism_height_m"] <= 0.5
    assert fused.extras["prism_iou"] >= 0.5
    assert "geometry_low_confidence" not in fused.extras["warnings"]


def test_prism_survives_a_tilted_and_an_offset_view_pair():
    calibrations, _, _, percepts, truth = _scene(
        centre=(0.1, -0.05, 0.05), azimuths_deg=(0.0, 140.0)
    )
    fused = fuse(percepts, calibrations, {"table_z": 0.0})
    assert fused is not None and fused.geometry is not None
    assert fused.geometry.views_used == 2
    # Off-axis views are harder: the prism slice is a shadow of the silhouette, so the
    # acceptance bound is looser than the head-on S5 fixture.
    assert _centroid_error(fused, truth) < 0.05


def test_impossible_iou_threshold_falls_back_to_argmax_with_a_warning():
    calibrations, _, _, percepts, _ = _scene()
    fused = fuse(percepts, calibrations, {"table_z": 0.0, "prism_iou_threshold": 0.99})
    assert fused is not None and fused.geometry is not None
    warnings = fused.extras["warnings"]
    assert "geometry_low_confidence" in warnings
    assert fused.extras["prism_height_m"] == fused.extras["prism_best_height_m"]
    step = 0.01
    assert fused.geometry.uncertainty_m >= step * (1.0 + 1.0 / 2.0)


def test_prism_is_deterministic():
    calibrations, _, _, percepts, _ = _scene()
    first = fuse(percepts, calibrations, {"table_z": 0.0})
    second = fuse(percepts, calibrations, {"table_z": 0.0})
    assert first is not None and second is not None
    assert first.geometry is not None and second.geometry is not None
    assert first.geometry.centroid == second.geometry.centroid
    assert first.extras["prism_height_m"] == second.extras["prism_height_m"]


def test_table_plane_context_is_accepted():
    calibrations, _, _, percepts, truth = _scene()
    fused = fuse(percepts, calibrations, {"table_plane": [0.0, 0.0, 1.0, 0.0]})
    assert fused is not None and fused.geometry is not None
    assert _centroid_error(fused, truth) < CENTROID_TOLERANCE_M
    assert fused.extras["table_z"] == pytest.approx(0.0)


def test_degenerate_table_plane_warns_and_falls_back():
    calibrations, _, _, percepts, _ = _scene()
    fused = fuse(percepts, calibrations, {"table_plane": [0.0, 0.0, 0.0, 1.0]})
    assert fused is not None
    assert "invalid_table_plane" in fused.extras["warnings"]


# ---------------------------------------------------------------------------
# §9 hard constraints
# ---------------------------------------------------------------------------


def test_no_calibration_never_yields_geometry():
    _, _, _, percepts, _ = _scene(with_depth=True)
    assert fuse(percepts, {}, {}) is None
    assert fuse(percepts, {}, {"table_z": 0.0}) is None


def test_uncalibrated_cameras_are_skipped():
    calibrations, _, _, percepts, truth = _scene()
    partial = {"cam_a": calibrations["cam_a"]}
    fused = fuse(percepts, partial, {"table_z": 0.0})
    # One calibrated view left: either no table (None) or a low-confidence single view.
    if fused is not None:
        assert fused.geometry is not None
        assert fused.geometry.views_used == 1
        assert fused.geometry.uncertainty_m is not None
        assert fused.geometry.uncertainty_m >= 0.03
        assert _centroid_error(fused, truth) < 0.1


def test_single_view_without_depth_is_none_or_explicitly_uncertain():
    calibrations, _, _, percepts, _ = _scene()
    camera = sorted(calibrations)[0]
    single = [p for p in percepts if p.camera == camera]
    assert fuse(single, calibrations, {}) is None  # no table information at all
    fused = fuse(single, calibrations, {"table_z": 0.0})
    assert fused is not None and fused.geometry is not None
    assert fused.geometry.views_used == 1
    assert fused.geometry.uncertainty_m is not None
    assert fused.geometry.uncertainty_m >= 0.03
    assert fused.extras["method"] == "single_view"
    assert "geometry_height_assumed" in fused.extras["warnings"]


def test_empty_or_maskless_input_is_none():
    calibrations, _, _, percepts, _ = _scene()
    assert fuse([], calibrations, {}) is None
    assert fuse([], {}, {}) is None
    maskless = [p.model_copy(update={"observations": []}) for p in percepts]
    assert fuse(maskless, calibrations, {}) is None


def test_fuse_disabled_options_return_none():
    calibrations, _, _, percepts, _ = _scene()
    options = PerceiveOptions(fuse=False)
    assert fuse(percepts, calibrations, {"table_z": 0.0}, options) is None


def test_fused_percept_is_json_serialisable_and_consistent():
    calibrations, _, _, percepts, truth = _scene()
    fused = fuse(percepts, calibrations, {"table_z": 0.0})
    assert fused is not None
    payload = fused.model_dump_json()
    restored = Percept.model_validate_json(payload)
    assert restored.geometry is not None
    assert isinstance(restored.geometry, Geometry)
    assert restored.geometry.frame == "world"
    assert restored.geometry.centroid == fused.geometry.centroid
    assert restored.status == "ok"
    assert restored.provenance.tool == "multi_view_3d"
    assert restored.provenance.kind == "geometry"
    assert np.linalg.norm(np.asarray(restored.geometry.centroid) - truth) < CENTROID_TOLERANCE_M


# ---------------------------------------------------------------------------
# Tolerant-input behaviour (the tool must survive a messy pipeline)
# ---------------------------------------------------------------------------


def test_corrupt_depth_payload_falls_back_to_the_prism():
    calibrations, _, _, percepts, _ = _scene()
    broken = DepthRef(encoding="raw_f32", data="!!!not-base64!!!", shape=(480, 640))
    damaged = dict(calibrations)
    damaged["cam_a"] = calibrations["cam_a"].model_copy(update={"depth": broken})
    fused = fuse(percepts, damaged, {"table_z": 0.0})
    assert fused is not None
    assert fused.extras["method"] == "prism"


def test_single_pixel_masks_never_produce_a_confident_multi_view_geometry():
    calibrations, images, _, _, _ = _scene()
    tiny = np.zeros((480, 640), dtype=bool)
    tiny[240, 320] = True
    percepts = [
        synth_geom.mask_percepts(calibrations, {camera: tiny})[0]
        for camera in sorted(calibrations)
    ]
    assert images  # the scene is still a normal rig
    fused = fuse(percepts, calibrations, {"table_z": 0.0})
    if fused is not None:
        assert fused.geometry is not None
        assert fused.geometry.views_used == 1
        assert fused.geometry.uncertainty_m is not None
        assert fused.geometry.uncertainty_m >= 0.03


def test_non_convex_masks_do_not_raise():
    calibrations, _, _, _, _ = _scene()
    blobs = np.zeros((480, 640), dtype=bool)
    blobs[80:180, 80:180] = True
    blobs[300:400, 420:520] = True
    percepts = [
        synth_geom.mask_percepts(calibrations, {camera: blobs})[0]
        for camera in sorted(calibrations)
    ]
    fused = fuse(percepts, calibrations, {"table_z": 0.0})
    assert fused is not None and fused.geometry is not None
    assert fused.geometry.frame == "world"


def test_absurd_prism_tunables_are_clamped():
    calibrations, _, _, percepts, _ = _scene()
    fused = fuse(
        percepts,
        calibrations,
        {"table_z": 0.0, "prism_step_m": 100.0, "prism_max_height_m": 0.5},
    )
    if fused is not None:
        assert fused.geometry is not None
        assert fused.extras["prism_height_m"] <= 0.5


def test_invalid_table_z_falls_back_and_warns():
    calibrations, _, _, percepts, _ = _scene()
    fused = fuse(percepts, calibrations, {"table_z": "not-a-number"})
    assert fused is not None
    assert "invalid_table_z" in fused.extras["warnings"]
    assert "table_z_assumed" in fused.extras["warnings"]


def test_three_views_report_three_used_views():
    calibrations = synth_geom.camera_rig()
    truth = np.array([0.0, 0.0, 0.05])
    masks = synth_geom.cube_masks(calibrations, truth, HALF_EXTENT)
    percepts = synth_geom.mask_percepts(calibrations, masks)
    fused = fuse(percepts, calibrations, {"table_z": 0.0})
    assert fused is not None and fused.geometry is not None
    assert fused.geometry.views_used == 3
    assert fused.extras["views"] == ["cam_0", "cam_1", "cam_2"]
    assert np.linalg.norm(np.asarray(fused.geometry.centroid) - truth) < CENTROID_TOLERANCE_M

