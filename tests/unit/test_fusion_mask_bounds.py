"""Direct fusion rejects masks outside calibration dimensions before pixel allocation."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from percepta import masks
from percepta.fusion import fuse
from percepta.schema import Mask
from tests import synth, synth_geom


def _inputs(mask: Mask):
    calibration = {"cam": synth.make_calibration("cam", width=8, height=6)}
    percept = synth_geom.mask_percepts(calibration, {"cam": np.ones((6, 8), dtype=bool)})[0]
    percept.observations[0].mask = mask
    return [percept], calibration


@pytest.mark.parametrize("encoding", ["rle", "none"])
def test_direct_fuse_rejects_large_declared_mask_before_allocating(monkeypatch, encoding):
    mask = Mask(encoding=encoding, size=(10**6, 10**6), counts=[0, 10**12] if encoding == "rle" else None)
    percepts, calibration = _inputs(mask)
    zeros = np.zeros

    def bounded_zeros(shape, *args, **kwargs):
        pixels = shape if isinstance(shape, int) else int(np.prod(shape))
        assert pixels <= 48, "mismatched mask must be rejected before allocating its declared pixels"
        return zeros(shape, *args, **kwargs)

    monkeypatch.setattr(masks.np, "zeros", bounded_zeros)
    assert fuse(percepts, calibration, {}) is None


@pytest.mark.parametrize("declared_size", [None, (6, 8)])
def test_direct_fuse_checks_png_header_before_pixel_decode(monkeypatch, declared_size):
    encoded = masks.to_mask(np.ones((24, 32), dtype=bool), "png")
    encoded.size = declared_size
    percepts, calibration = _inputs(encoded)

    def forbidden_convert(*_args, **_kwargs):
        pytest.fail("PNG header dimensions must be checked before converting/decompressing pixels")

    monkeypatch.setattr(Image.Image, "convert", forbidden_convert)
    assert fuse(percepts, calibration, {}) is None


def test_direct_fuse_skips_corrupt_png():
    percepts, calibration = _inputs(Mask(encoding="png", data=masks.bytes_b64(b"not a PNG")))
    assert fuse(percepts, calibration, {}) is None


def test_direct_fuse_skips_pillow_decompression_bomb(monkeypatch):
    percepts, calibration = _inputs(masks.to_mask(np.ones((6, 8), dtype=bool), "png"))

    def decompression_bomb(*_args, **_kwargs):
        raise Image.DecompressionBombError("test oversized PNG header")

    monkeypatch.setattr(Image, "open", decompression_bomb)
    assert fuse(percepts, calibration, {}) is None


@pytest.mark.parametrize("encoding", ["rle", "png", "none"])
def test_mask_expected_size_preserves_valid_decoding(encoding):
    original = np.zeros((6, 8), dtype=bool)
    original[2:4, 3:6] = True
    encoded = masks.to_mask(original, encoding)
    expected = np.zeros_like(original) if encoding == "none" else original
    assert np.array_equal(masks.to_array(encoded, expected_size=(6, 8)), expected)
    assert np.array_equal(masks.to_array(encoded), expected)


def test_png_without_declared_size_remains_usable_in_direct_fuse():
    encoded = masks.to_mask(np.ones((6, 8), dtype=bool), "png")
    encoded.size = None
    percepts, calibration = _inputs(encoded)
    result = fuse(percepts, calibration, {"table_z": 0.0})
    assert result is not None and result.geometry is not None
    assert result.geometry.views_used == 1


@pytest.mark.parametrize("expected_size", [(0, 8), (-1, 8), (6,), (6, 8, 1), (6.0, 8), (True, 8), (np.bool_(True), 8), 6])
def test_expected_mask_size_rejects_invalid_dimensions(expected_size):
    declared_size = expected_size if isinstance(expected_size, tuple) and len(expected_size) == 2 else (6, 8)
    mask = Mask(encoding="none", size=declared_size)
    with pytest.raises(ValueError):
        masks.to_array(mask, expected_size=expected_size)


def test_negative_expected_dimensions_are_rejected_before_rle_allocation(monkeypatch):
    mask = Mask(encoding="rle", size=(-10**6, -10**6), counts=[0, 10**12])

    def forbidden_zeros(*_args, **_kwargs):
        pytest.fail("negative expected dimensions must be rejected before allocating their positive product")

    monkeypatch.setattr(masks.np, "zeros", forbidden_zeros)
    with pytest.raises(ValueError, match="expected_size"):
        masks.to_array(mask, expected_size=mask.size)


def test_expected_mask_size_accepts_numpy_integer_dimensions():
    original = np.ones((6, 8), dtype=bool)
    assert np.array_equal(masks.to_array(masks.to_mask(original), expected_size=(np.int64(6), np.int32(8))), original)
