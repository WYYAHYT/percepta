"""S6 — verification: followed/lifted/stable/visible semantics (spec §10).

All fixtures are synthetic: explicit masks and explicit ``Geometry`` values, no real 3D,
no GPU, no weights. Pixel centroid of a mask is the mean of its ``(row, col)`` indices, so
a rectangle moved by N columns changes ``u`` by exactly N.
"""

from __future__ import annotations

import numpy as np
import pytest

from percepta.masks import mask_centroid, to_mask
from percepta.schema import Geometry, Observation, Percept, Provenance
from percepta.verification import compare

H, W = 120, 160


def rect_mask(x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    mask = np.zeros((H, W), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def _percept(
    mask: np.ndarray | None,
    *,
    z: float | None = None,
    status: str = "ok",
    area_px: int | None = None,
    frame: str = "world",
) -> Percept:
    """Build a Percept with an optional first observation and optional world geometry."""
    observations: list[Observation] = []
    if mask is not None:
        centroid = mask_centroid(mask)
        observations.append(
            Observation(
                object_id="object-0",
                score=1.0,
                mask=to_mask(mask),
                area_px=int(np.count_nonzero(mask)) if area_px is None else area_px,
                centroid_uv=(
                    None if centroid is None else [float(centroid[0]), float(centroid[1])]
                ),
            )
        )
    geometry = None
    if z is not None:
        geometry = Geometry(
            frame=frame, centroid=[0.0, 0.0, float(z)], uncertainty_m=0.01, views_used=2
        )
    return Percept(
        percept_id="percept-test",
        status=status,
        observations=observations,
        geometry=geometry,
        provenance=Provenance(tool="fake", tool_version="0.1.0", kind="fake", latency_ms=0.5),
    )


# ---------------------------------------------------------------------------
# lifted (§10, S6's headline case)
# ---------------------------------------------------------------------------


def test_lift_of_12cm_passes():
    before = _percept(rect_mask(60, 40, 90, 70), z=0.50)
    after = _percept(rect_mask(60, 40, 90, 70), z=0.62)

    result = compare(before, after, {"ee_delta_px": [0.0, 0.0]})

    assert result.ok is True
    assert result.status == "ok"
    assert result.checks["lifted"] is True
    assert result.checks["followed"] is True
    assert result.checks["visible"] is True
    assert result.details["lift_m"] == pytest.approx(0.12, abs=1e-9)
    assert "stable" not in result.checks  # stable only runs when asked for


def test_lift_below_threshold_fails_verification():
    before = _percept(rect_mask(60, 40, 90, 70), z=0.50)
    after = _percept(rect_mask(60, 40, 90, 70), z=0.54)

    result = compare(before, after, {"ee_delta_px": [0.0, 0.0]})

    assert result.checks["lifted"] is False
    assert result.details["lift_m"] == pytest.approx(0.04, abs=1e-9)
    assert result.ok is False
    # Nothing about the percepts themselves is wrong: the verdict lives in ``ok``.
    assert result.status == "ok"


def test_required_lift_threshold_is_configurable():
    before = _percept(rect_mask(60, 40, 90, 70), z=0.50)
    after = _percept(rect_mask(60, 40, 90, 70), z=0.54)

    result = compare(before, after, {"ee_delta_px": [0.0, 0.0]}, {"required_lift_m": 0.03})

    assert result.checks["lifted"] is True
    assert result.ok is True


def test_missing_geometry_degrades_to_a_failed_check():
    without_geometry = _percept(rect_mask(60, 40, 90, 70))
    with_geometry = _percept(rect_mask(60, 40, 90, 70), z=0.62)

    for before, after in ((without_geometry, with_geometry), (with_geometry, without_geometry)):
        result = compare(before, after, {"ee_delta_px": [0.0, 0.0]})
        assert result.checks["lifted"] is False
        assert result.details["geometry_available"] == 0.0
        assert "lift_m" not in result.details
        assert result.ok is False


def test_mismatched_geometry_frames_do_not_pretend_to_lift():
    before = _percept(rect_mask(60, 40, 90, 70), z=0.50)
    after = _percept(rect_mask(60, 40, 90, 70), z=0.62, frame="camera")

    result = compare(before, after, {"ee_delta_px": [0.0, 0.0]})

    assert result.checks["lifted"] is False
    assert result.details["frame_mismatch"] == 1.0


@pytest.mark.parametrize("frame", ["camera", "pixel"])
def test_matching_non_world_frames_do_not_count_as_metric_lift(frame):
    before = _percept(None, z=1.0, frame=frame)
    after = _percept(None, z=1.2, frame=frame)
    result = compare(before, after, {}, {"required": ["lifted"]})
    assert result.ok is False
    assert result.checks["lifted"] is False
    assert result.details["world_frame_required"] == 1.0
    assert "lift_m" not in result.details
    assert "lift_m" not in result.details


# ---------------------------------------------------------------------------
# not_found is never a pass (§10, last line)
# ---------------------------------------------------------------------------


def test_not_found_after_never_passes_even_with_geometry():
    before = _percept(rect_mask(60, 40, 90, 70), z=0.50)
    after = _percept(rect_mask(60, 40, 90, 70), z=0.62, status="not_found")

    result = compare(before, after, {"ee_delta_px": [0.0, 0.0]})

    assert result.checks["lifted"] is True  # the geometry says "lifted" ...
    assert result.checks["visible"] is False
    assert result.ok is False  # ... but not_found must never verify
    assert result.status == "not_found"


def test_not_found_never_passes_even_when_required_is_narrowed():
    before = _percept(rect_mask(60, 40, 90, 70), z=0.50)
    after = _percept(rect_mask(60, 40, 90, 70), z=0.62, status="not_found")

    result = compare(before, after, {"ee_delta_px": [0.0, 0.0]}, {"required": ["lifted"]})

    assert result.ok is False
    assert result.status == "not_found"


def test_occluded_and_unavailable_are_propagated():
    before = _percept(rect_mask(60, 40, 90, 70), z=0.50)
    for status in ("occluded", "unavailable", "timeout", "invalid_request"):
        after = _percept(rect_mask(60, 40, 90, 70), z=0.62, status=status)
        result = compare(before, after, {"ee_delta_px": [0.0, 0.0]})
        assert result.ok is False
        assert result.status == status


# ---------------------------------------------------------------------------
# followed: direction + magnitude
# ---------------------------------------------------------------------------


def test_followed_accepts_matching_direction_and_rejects_the_reverse():
    before = _percept(rect_mask(60, 40, 90, 70), z=0.50)
    after = _percept(rect_mask(70, 40, 100, 70), z=0.62)  # +10 px in u

    forward = compare(before, after, {"ee_delta_px": [10.0, 0.0]})
    assert forward.checks["followed"] is True
    assert forward.details["centroid_shift_px"] == pytest.approx(10.0, abs=1e-9)
    assert forward.details["direction_cos"] == pytest.approx(1.0, abs=1e-9)
    assert forward.details["followed_error_px"] == pytest.approx(0.0, abs=1e-9)

    reverse = compare(before, after, {"ee_delta_px": [-10.0, 0.0]})
    assert reverse.checks["followed"] is False
    assert reverse.details["direction_cos"] == pytest.approx(-1.0, abs=1e-9)
    assert reverse.ok is False

    orthogonal = compare(before, after, {"ee_delta_px": [0.0, 10.0]})
    assert orthogonal.checks["followed"] is False
    assert orthogonal.details["direction_cos"] == pytest.approx(0.0, abs=1e-9)


def test_followed_checks_magnitude_within_tolerance():
    before = _percept(rect_mask(60, 40, 90, 70), z=0.50)
    after = _percept(rect_mask(70, 40, 100, 70), z=0.62)  # actual shift 10 px

    too_far = compare(before, after, {"ee_delta_px": [40.0, 0.0]})
    assert too_far.details["followed_error_px"] == pytest.approx(30.0, abs=1e-9)
    assert too_far.checks["followed"] is False

    loose = compare(before, after, {"ee_delta_px": [40.0, 0.0]}, {"followed_tol_px": 31.0})
    assert loose.checks["followed"] is True

    strict_direction = compare(
        before, after, {"ee_delta_px": [10.0, 0.0]}, {"followed_min_cos": 1.1}
    )
    assert strict_direction.status == "invalid_request"  # cos must stay in [-1, 1]


def test_missing_masks_degrade_without_raising():
    before = _percept(None, z=0.50)
    after = _percept(None, z=0.62)

    result = compare(before, after, {"ee_delta_px": [1.0, 0.0]})

    assert result.checks["followed"] is False
    assert result.checks["visible"] is False
    assert result.details["masks_available"] == 0.0
    assert result.ok is False


def test_empty_mask_does_not_raise_and_is_not_visible():
    empty = np.zeros((H, W), dtype=bool)
    before = _percept(empty, z=0.50)
    after = _percept(empty, z=0.62)

    result = compare(before, after, {"ee_delta_px": [0.0, 0.0]})

    assert result.checks["followed"] is False
    assert result.checks["visible"] is False
    assert result.details["area_px"] == 0.0


# ---------------------------------------------------------------------------
# visible
# ---------------------------------------------------------------------------


def test_visible_requires_ok_status_and_min_area():
    before = _percept(rect_mask(60, 40, 90, 70), z=0.50)

    too_small = _percept(rect_mask(60, 40, 90, 70), z=0.62, area_px=0)
    small_result = compare(before, too_small, {"ee_delta_px": [0.0, 0.0]})
    assert small_result.checks["visible"] is False
    assert small_result.ok is False

    fine = _percept(rect_mask(60, 40, 90, 70), z=0.62)
    raised_threshold = compare(
        before, fine, {"ee_delta_px": [0.0, 0.0]}, {"min_area_px": 10_000}
    )
    assert raised_threshold.checks["visible"] is False


# ---------------------------------------------------------------------------
# stable (mask IoU)
# ---------------------------------------------------------------------------


def test_stable_accepts_overlapping_mask_and_rejects_a_moved_one():
    before = _percept(rect_mask(60, 40, 90, 70), z=0.50)
    after = _percept(rect_mask(60, 40, 90, 70), z=0.62)

    previous = _percept(rect_mask(61, 40, 91, 70))  # one pixel of jitter
    stable = compare(before, after, {"ee_delta_px": [0.0, 0.0]}, {"stable_with": previous})
    assert stable.checks["stable"] is True
    assert stable.details["mask_iou"] > 0.9
    assert stable.ok is True

    moved = _percept(rect_mask(0, 0, 30, 30))  # disjoint
    unstable = compare(
        before, after, {"ee_delta_px": [0.0, 0.0]}, {"stable_with": moved, "required": ["stable"]}
    )
    assert unstable.checks["stable"] is False
    assert unstable.details["mask_iou"] == pytest.approx(0.0)
    assert unstable.ok is False


def test_stable_with_accepts_percept_observation_mask_and_array():
    before = _percept(rect_mask(60, 40, 90, 70), z=0.50)
    after = _percept(rect_mask(60, 40, 90, 70), z=0.62)
    expected = rect_mask(60, 40, 90, 70)

    for stable_with in (
        _percept(expected),
        after.observations[0],
        after.observations[0].mask,
        expected,
    ):
        result = compare(before, after, {"ee_delta_px": [0.0, 0.0]}, {"stable_with": stable_with})
        assert result.checks["stable"] is True
        assert result.details["mask_iou"] == pytest.approx(1.0)


def test_stable_without_evidence_degrades():
    before = _percept(rect_mask(60, 40, 90, 70), z=0.50)
    after = _percept(rect_mask(60, 40, 90, 70), z=0.62)

    result = compare(before, after, {"ee_delta_px": [0.0, 0.0]}, {"stable_with": 42})

    assert result.checks["stable"] is False
    assert result.details["stable_masks_available"] == 0.0
    assert result.ok is False


# ---------------------------------------------------------------------------
# options / malformed input
# ---------------------------------------------------------------------------


def test_required_list_narrows_the_checks():
    before = _percept(rect_mask(60, 40, 90, 70), z=0.50)
    after = _percept(rect_mask(100, 40, 130, 70), z=0.62)  # moved 40 px, ee says 10

    default = compare(before, after, {"ee_delta_px": [10.0, 0.0]})
    assert default.checks["followed"] is False
    assert default.ok is False

    narrowed = compare(
        before, after, {"ee_delta_px": [10.0, 0.0]}, {"required": ["visible", "lifted"]}
    )
    assert narrowed.ok is True


def test_missing_or_malformed_action_is_invalid_request():
    before = _percept(rect_mask(60, 40, 90, 70), z=0.50)
    after = _percept(rect_mask(60, 40, 90, 70), z=0.62)

    for action in ({}, {"ee_delta_px": None}, {"ee_delta_px": "right"}, {"ee_delta_px": [1, 2, 3]}):
        result = compare(before, after, action)
        assert result.ok is False
        assert result.status == "invalid_request"
        assert result.checks["followed"] is False

    not_a_mapping = compare(before, after, None)  # type: ignore[arg-type]
    assert not_a_mapping.status == "invalid_request"


def test_bad_options_are_invalid_request():
    before = _percept(rect_mask(60, 40, 90, 70), z=0.50)
    after = _percept(rect_mask(60, 40, 90, 70), z=0.62)

    for options in (
        {"followed_min_cos": 2.0},
        {"followed_tol_px": -1.0},
        {"stable_iou": "not-a-number"},
        {"required": ["vibes"]},
        {"required": []},
        "not-a-mapping",
    ):
        result = compare(before, after, {"ee_delta_px": [0.0, 0.0]}, options)  # type: ignore[arg-type]
        assert result.ok is False
        assert result.status == "invalid_request"


def test_requiring_stable_without_evidence_is_invalid_request():
    before = _percept(rect_mask(60, 40, 90, 70), z=0.50)
    after = _percept(rect_mask(60, 40, 90, 70), z=0.62)

    result = compare(before, after, {"ee_delta_px": [0.0, 0.0]}, {"required": ["stable"]})

    assert result.checks["stable"] is False
    assert result.status == "invalid_request"
    assert result.ok is False


def test_non_percept_inputs_return_invalid_request_instead_of_raising():
    result = compare({}, {}, {"ee_delta_px": [0.0, 0.0]})  # type: ignore[arg-type]
    assert result.ok is False
    assert result.status == "invalid_request"
