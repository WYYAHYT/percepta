"""S2 — RLE round-trip plus mask statistics and codecs (spec §4.7, appendix B)."""

from __future__ import annotations

import numpy as np
import pytest

from percepta.masks import (
    box_iou,
    decode_image,
    decode_rle,
    encode_image,
    encode_rle,
    mask_area,
    mask_box_xyxy,
    mask_centroid,
    mask_iou,
    to_array,
    to_mask,
)
from percepta.schema import Mask

# ---------------------------------------------------------------------------
# RLE (appendix B)
# ---------------------------------------------------------------------------


def test_rle_examples_from_appendix():
    assert encode_rle(np.array([[0, 0, 1, 1]], dtype=bool)) == [2, 2]
    assert encode_rle(np.ones((1, 3), dtype=bool)) == [0, 3]
    assert encode_rle(np.zeros((1, 3), dtype=bool)) == [3]
    assert encode_rle(np.array([[1, 0, 1]], dtype=bool)) == [0, 1, 1, 1]


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("shape", [(1, 1), (3, 7), (16, 16), (37, 53), (480, 640)])
def test_rle_round_trip_is_exact(seed, shape):
    rng = np.random.default_rng(seed)
    mask = rng.random(shape) > 0.5
    counts = encode_rle(mask)
    assert sum(counts) == shape[0] * shape[1]
    assert np.array_equal(decode_rle(counts, shape), mask)


def test_rle_round_trip_degenerate_masks():
    for mask in (
        np.zeros((0, 0), dtype=bool),
        np.zeros((5, 5), dtype=bool),
        np.ones((5, 5), dtype=bool),
        np.zeros((480, 640), dtype=bool),
        np.ones((480, 640), dtype=bool),
        np.array([[True]], dtype=bool),
        np.array([[False]], dtype=bool),
    ):
        counts = encode_rle(mask)
        assert sum(counts) == mask.size
        assert np.array_equal(decode_rle(counts, mask.shape), mask)


def test_rle_starts_with_zero_run_when_first_pixel_set():
    mask = np.array([[1, 1, 0, 0, 1]], dtype=bool)
    counts = encode_rle(mask)
    assert counts[0] == 0  # leading zero-length 0-run
    assert decode_rle(counts, mask.shape).tolist() == mask.tolist()


def test_decode_rle_rejects_bad_sum_and_negatives():
    with pytest.raises(ValueError, match="!= H\\*W"):
        decode_rle([1, 2], (2, 2))
    with pytest.raises(ValueError, match="non-negative"):
        decode_rle([-1, 5], (2, 2))


# ---------------------------------------------------------------------------
# Statistics (§4.7)
# ---------------------------------------------------------------------------


def test_area_centroid_box_of_rectangle():
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 3:7] = True  # rows 2..4, cols 3..6
    assert mask_area(mask) == 12
    u, v = mask_centroid(mask)
    assert (u, v) == pytest.approx((4.5, 3.0))
    assert mask_box_xyxy(mask) == [3.0, 2.0, 7.0, 5.0]  # x1/y1 exclusive


def test_empty_mask_stats_are_none():
    mask = np.zeros((8, 8), dtype=bool)
    assert mask_area(mask) == 0
    assert mask_centroid(mask) is None
    assert mask_box_xyxy(mask) is None


def test_mask_iou_cases():
    a = np.zeros((4, 4), dtype=bool)
    b = np.zeros((4, 4), dtype=bool)
    a[:2, :] = True
    b[:2, :] = True
    assert mask_iou(a, b) == 1.0
    assert mask_iou(a, np.zeros_like(a)) == 0.0
    assert mask_iou(np.zeros_like(a), np.zeros_like(b)) == 0.0  # empty never matches
    b[:] = False
    b[2:, :] = True
    assert mask_iou(a, b) == 0.0
    b[:] = True
    assert mask_iou(a, b) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="shapes differ"):
        mask_iou(a, np.zeros((5, 5), dtype=bool))


def test_box_iou():
    assert box_iou([0, 0, 2, 2], [0, 0, 2, 2]) == 1.0
    assert box_iou([0, 0, 1, 1], [2, 2, 3, 3]) == 0.0
    assert box_iou([0, 0, 2, 2], [1, 0, 3, 2]) == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# Codecs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("encoding", ["rle", "png", "none"])
def test_mask_container_round_trip(encoding):
    mask = np.zeros((12, 20), dtype=bool)
    mask[3:9, 5:15] = True
    payload = to_mask(mask, encoding)
    assert payload.encoding == encoding
    assert payload.size == (12, 20)
    decoded = to_array(payload)
    if encoding == "none":
        assert not decoded.any()  # "none" keeps size only
    else:
        assert np.array_equal(decoded, mask)


def test_rle_mask_rejects_inconsistent_counts():
    with pytest.raises(ValueError, match="counts sum"):
        Mask(encoding="rle", size=(2, 2), counts=[3, 2])  # 5 != 4
    with pytest.raises(ValueError, match="requires counts"):
        Mask(encoding="rle", size=(2, 2))


def test_image_round_trip_png_jpeg_raw():
    rng = np.random.default_rng(3)
    image = rng.integers(0, 256, size=(24, 32, 3), dtype=np.uint8)
    for encoding in ("png", "raw"):
        ref = encode_image(image, encoding=encoding, camera="cam")
        assert ref.camera == "cam"
        assert np.array_equal(decode_image(ref), image)
    # jpeg is lossy: use a smooth image and assert only shape + closeness
    ramp = np.linspace(0, 255, 32, dtype=np.float32)
    smooth = np.repeat(ramp[None, :, None], 24, axis=0).repeat(3, axis=2).astype(np.uint8)
    ref = encode_image(smooth, encoding="jpeg")
    decoded = decode_image(ref)
    assert decoded.shape == smooth.shape
    assert np.abs(decoded.astype(int) - smooth.astype(int)).mean() < 12
