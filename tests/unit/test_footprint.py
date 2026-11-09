"""P0: exact prism footprint, and P1: split uncertainty.

Two defects motivated this file, both measured before the change (see
``scripts/footprint_accuracy.py``):

**P0 — the footprint was the object's shadow, not its footprint.**  The old cross-section was the
cone slice at mid-height: where each camera's silhouette rays cross that plane. That polygon is
larger than the object by roughly ``h/2 * (tan(ray) - tan(base))``, and intersecting several such
shadows cancels only part of it. A 10 cm cube measured 178–293 cm² against a true 100 cm²
(**1.8–2.9×**).

**P1 — one uncertainty hid two very different numbers.**  ``uncertainty_m`` said nothing about
whether the *lateral* or the *height* figure was the weak one, and a single camera leaves the
height effectively guessed while the lateral position is still good to a few millimetres.

Two more defects were found *while* fixing P0, and are pinned here too:

* sampling the column at its exact ends (fractions 0.0 and 1.0) made **every** column fail, because
  a mask covers the object's interior and the base/top *are* the boundary. A 15 cm cube collapsed
  to a 2 cm² footprint scored at zero.
* a coarse ``prism_step_m`` collapsed the height search to one sample, which has no peak.
"""

from __future__ import annotations

import numpy as np
import pytest

from percepta import hull
from percepta.fusion import (
    HANG_MARGIN,
    MIN_PRISM_STEPS,
    _prism_grid,
    _prism_views,
    _View,
    fuse,
)
from percepta.schema import Geometry
from tests.synth_geom import camera_rig, cube_masks, mask_percepts

HALF = 0.05  # a 10 cm cube


def _views(azimuths, top_height, elevation=20.0, half=HALF):
    centre = np.array([0.0, 0.0, top_height / 2.0])
    cals = camera_rig(azimuths, elevation_deg=elevation, target=(0.0, 0.0, 0.05))
    masks = cube_masks(cals, centre, half)
    views = [
        _View(
            camera=name,
            mask=masks[name],
            K=np.asarray(cals[name].K),
            T=np.asarray(cals[name].T_world_cam),
            score=0.9,
            calibration_id=name,
        )
        for name in sorted(cals)
    ]
    return cals, centre, masks, views


# ---------------------------------------------------------------------------
# hull: the column test
# ---------------------------------------------------------------------------


def test_column_test_samples_interior_not_the_boundary():
    """Sampling the ends made every column fail — the bug that zeroed a 15 cm cube.

    A mask covers the object's *interior*; its base and top edges are exactly where the interior
    stops. A column test that samples z=base or z=top therefore lands on the boundary and reads
    "outside" for the very columns that are correct.
    """
    assert 0.0 not in hull.LEVEL_FRACTIONS
    assert 1.0 not in hull.LEVEL_FRACTIONS
    assert all(0.0 < f < 1.0 for f in hull.LEVEL_FRACTIONS)
    assert len(hull.LEVEL_FRACTIONS) >= 3, "one interior sample cannot catch a waisted silhouette"


def test_prism_footprint_of_a_cube_is_close_to_the_truth():
    cals, _, _, views = _views((0, 120, 240), 0.10)
    prism_views = _prism_views(views)
    grid = _prism_grid(prism_views, 0.0, 0.5)
    cells = hull.column_inside(prism_views, grid, 0.0, 0.10)
    metrics = hull.footprint_metrics(grid, cells)
    assert metrics is not None
    true_area = (2 * HALF) ** 2
    # A silhouette visual hull is an *upper bound* on a convex body, so the error is one-sided.
    assert metrics.area_m2 >= true_area * 0.6
    assert metrics.area_m2 <= true_area * 2.0, (
        f"footprint {metrics.area_m2 * 1e4:.1f} cm2 is far above the true {true_area * 1e4:.1f}"
    )


def test_three_views_beat_one_view():
    """The whole point of several cameras: the intersection is tighter than any single cone."""
    cals, _, _, views = _views((0, 120, 240), 0.10)
    prism_views = _prism_views(views)
    grid = _prism_grid(prism_views, 0.0, 0.5)

    one = hull.footprint_metrics(grid, hull.column_inside(prism_views[:1], grid, 0.0, 0.10))
    three = hull.footprint_metrics(grid, hull.column_inside(prism_views, grid, 0.0, 0.10))
    assert one is not None and three is not None
    assert three.area_m2 < one.area_m2


def test_grid_is_finite_even_for_a_wide_span():
    grid = hull.PlaneGrid.fit([-5.0, -5.0], [5.0, 5.0], max_cells=10_000)
    assert grid.size <= 12_000  # bounded despite the huge span
    assert grid.cell > 0.0


def test_grid_never_quantises_below_its_floor():
    grid = hull.PlaneGrid.around([0, 0], [0.1, 0.1], 1e-9)
    assert grid.cell >= hull.MIN_CELL_M


# ---------------------------------------------------------------------------
# The height search
# ---------------------------------------------------------------------------


def test_height_peak_is_taken_not_the_first_threshold_crossing():
    """Spec D4 said "smallest height reaching the threshold"; that biases every height low.

    Measured on a 10 cm cube the IoU curve rises monotonically and crosses 0.5 at h≈0.04 while
    peaking at 0.10 (IoU 0.96). Taking the crossing reported a 4 cm object, which also dragged
    ``top_z`` and the centroid's z down by half the object's height.
    """
    cals = camera_rig((0, 120, 240), elevation_deg=20.0, target=(0, 0, 0.05))
    centre = np.array([0.0, 0.0, 0.05])  # a 10 cm cube standing on the table
    masks = cube_masks(cals, centre, HALF)
    fused = fuse(mask_percepts(cals, masks), cals, {"table_z": 0.0})
    assert fused is not None and fused.geometry is not None
    height = fused.extras["prism_height_m"]
    assert 0.07 <= height <= 0.13, f"chose {height:.3f} m for a 0.10 m cube"
    assert abs(fused.geometry.top_z - 0.10) <= 0.03


def test_coarse_step_does_not_collapse_the_search():
    """``prism_step_m`` larger than the range must not leave a single (peakless) sample."""
    cals = camera_rig((0, 120, 240), elevation_deg=20.0, target=(0, 0, 0.05))
    centre = np.array([0.0, 0.0, 0.05])
    masks = cube_masks(cals, centre, HALF)
    fused = fuse(
        mask_percepts(cals, masks),
        cals,
        {"table_z": 0.0, "prism_step_m": 100.0, "prism_max_height_m": 0.5},
    )
    assert fused is not None
    assert fused.extras.get("prism_scanned_heights", MIN_PRISM_STEPS) >= MIN_PRISM_STEPS


# ---------------------------------------------------------------------------
# P1: the split uncertainty
# ---------------------------------------------------------------------------


def test_geometry_reports_lateral_and_height_separately():
    cals = camera_rig((0, 120, 240), elevation_deg=20.0, target=(0, 0, 0.05))
    centre = np.array([0.0, 0.0, 0.05])
    masks = cube_masks(cals, centre, HALF)
    fused = fuse(mask_percepts(cals, masks), cals, {"table_z": 0.0})
    geometry = fused.geometry
    assert geometry.lateral_uncertainty_m is not None
    assert geometry.height_uncertainty_m is not None
    assert geometry.lateral_uncertainty_m > 0.0
    assert geometry.height_uncertainty_m > 0.0
    # ``uncertainty_m`` keeps its frozen meaning: the worst of the two.
    assert geometry.uncertainty_m == pytest.approx(
        max(geometry.lateral_uncertainty_m, geometry.height_uncertainty_m)
    )


def test_single_view_height_is_flagged_as_a_guess():
    """One camera cannot read height from a silhouette; the split has to say so.

    A low deep object and a tall shallow one cast the same shadow, so the height from one mask is
    inferred, not measured — while the lateral position is still good to a few millimetres. This
    is exactly the distinction a single ``uncertainty_m`` could not express.
    """
    cals = camera_rig((0.0,), elevation_deg=20.0, target=(0, 0, 0.05))
    centre = np.array([0.0, 0.0, 0.05])
    masks = cube_masks(cals, centre, HALF)
    fused = fuse(
        mask_percepts(cals, masks), cals, {"table_z": 0.0, "table_plane": [0, 0, 1, 0]}
    )
    assert fused is not None and fused.geometry is not None
    geometry = fused.geometry
    assert geometry.views_used == 1
    assert geometry.height_uncertainty_m is not None
    # the height is a guess: it must not look as trustworthy as the lateral figure
    assert geometry.height_uncertainty_m >= geometry.lateral_uncertainty_m


def test_view_disagreement_is_reported_for_multiple_views():
    cals = camera_rig((0, 120, 240), elevation_deg=20.0, target=(0, 0, 0.05))
    centre = np.array([0.0, 0.0, 0.05])
    masks = cube_masks(cals, centre, HALF)
    fused = fuse(mask_percepts(cals, masks), cals, {"table_z": 0.0})
    assert fused.geometry.view_disagreement_m is not None
    assert fused.geometry.view_disagreement_m >= 0.0


def test_geometry_extra_fields_are_optional_and_default_none():
    """Additive-only schema change: an existing construction must still validate."""
    geometry = Geometry(frame="world", centroid=[0.0, 0.0, 0.1])
    assert geometry.lateral_uncertainty_m is None
    assert geometry.height_uncertainty_m is None
    assert geometry.view_disagreement_m is None
    assert geometry.method is None


def test_geometry_round_trips_with_the_new_fields():
    geometry = Geometry(
        frame="world",
        centroid=[0.01, 0.02, 0.03],
        lateral_uncertainty_m=0.002,
        height_uncertainty_m=0.021,
        view_disagreement_m=0.004,
        method="prism",
    )
    assert Geometry.model_validate_json(geometry.model_dump_json()) == geometry


def test_method_names_the_reading():
    cals = camera_rig((0, 120, 240), elevation_deg=20.0, target=(0, 0, 0.05))
    centre = np.array([0.0, 0.0, 0.05])
    masks = cube_masks(cals, centre, HALF)
    fused = fuse(mask_percepts(cals, masks), cals, {"table_z": 0.0})
    assert fused.geometry.method in {"prism", "hang"}


def test_hang_margin_is_not_zero():
    """An overhanging reading has to earn its place: anything on its own footprint is a prism, so
    the unusual reading needs a clear win."""
    assert 0.0 < HANG_MARGIN < 0.2


# ---------------------------------------------------------------------------
# Accuracy table (a coarse guard, not a precise assertion)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("azimuths", [(0, 120, 240), (0, 90, 180, 270)])
@pytest.mark.parametrize("top_height", [0.05, 0.10])
def test_centroid_is_within_a_few_centimetres(azimuths, top_height):
    cals = camera_rig(azimuths, elevation_deg=20.0, target=(0, 0, 0.05))
    centre = np.array([0.0, 0.0, top_height / 2.0])
    masks = cube_masks(cals, centre, HALF)
    fused = fuse(mask_percepts(cals, masks), cals, {"table_z": 0.0})
    assert fused is not None and fused.geometry is not None
    error = float(np.linalg.norm(np.asarray(fused.geometry.centroid) - centre))
    assert error < 0.03, f"centroid off by {error * 100:.1f} cm"
