"""Mask codecs and 2D mask statistics (spec §4.7, appendix B).

The RLE here is the *percepta* run-length format, not COCO's compressed string format:
row-major ``(H, W)`` flattened to 0/1, emitted as alternating run lengths **starting with
the 0-run**, so ``sum(counts) == H * W`` and a first pixel of 1 is encoded by a leading 0.

All mask helpers take ``np.ndarray`` booleans (or anything truthy-castable) and never
mutate their inputs.
"""

from __future__ import annotations

import base64
import hashlib
import io
from collections.abc import Iterable, Sequence

import numpy as np
from PIL import Image

from .schema import ImageRef, Mask

__all__ = [
    "encode_rle",
    "decode_rle",
    "mask_area",
    "mask_centroid",
    "mask_box_xyxy",
    "mask_iou",
    "box_iou",
    "to_mask",
    "to_array",
    "encode_image",
    "decode_image",
    "encode_depth",
    "decode_depth",
    "array_sha256",
    "bytes_b64",
    "b64_bytes",
]


# ---------------------------------------------------------------------------
# base64 / hashing helpers
# ---------------------------------------------------------------------------


def bytes_b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def b64_bytes(data: str) -> bytes:
    return base64.b64decode(data.encode("ascii") if isinstance(data, str) else data)


def array_sha256(arr: np.ndarray) -> str:
    """sha256 of an array's raw bytes, prefixed with dtype/shape for unambiguity."""
    arr = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(f"{arr.dtype.str}:{arr.shape}:".encode())
    h.update(arr.tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# RLE (appendix B)
# ---------------------------------------------------------------------------


def encode_rle(mask: np.ndarray) -> list[int]:
    """Encode a 2-D mask as alternating run lengths starting with the 0-run.

    >>> encode_rle(np.array([[0, 0, 1, 1]], dtype=bool))
    [2, 2]
    >>> encode_rle(np.ones((1, 3), dtype=bool))
    [0, 3]
    >>> encode_rle(np.zeros((1, 3), dtype=bool))
    [3]
    """
    flat = np.asarray(mask).astype(bool).reshape(-1)
    if flat.size == 0:
        return []
    change = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    starts = np.concatenate(([0], change))
    lengths = np.diff(np.concatenate((starts, [flat.size])))
    counts = [int(n) for n in lengths]
    if bool(flat[0]):
        counts.insert(0, 0)
    return counts


def decode_rle(counts: Sequence[int], shape: tuple[int, int]) -> np.ndarray:
    """Decode alternating run lengths (starting with the 0-run) into a bool mask."""
    height, width = int(shape[0]), int(shape[1])
    total = height * width
    if any(int(c) < 0 for c in counts):
        raise ValueError("RLE counts must be non-negative")
    if sum(int(c) for c in counts) != total:
        raise ValueError(f"RLE counts sum {sum(int(c) for c in counts)} != H*W {total}")
    out = np.zeros(total, dtype=bool)
    idx = 0
    value = False
    for count in counts:
        count = int(count)
        if count:
            if value:
                out[idx : idx + count] = True
            idx += count
        value = not value
    return out.reshape(height, width)


# ---------------------------------------------------------------------------
# 2D statistics
# ---------------------------------------------------------------------------


def mask_area(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask))


def mask_centroid(mask: np.ndarray) -> tuple[float, float] | None:
    """Pixel mean of a mask as ``(u, v)`` = ``(col, row)``; ``None`` for an empty mask."""
    arr = np.asarray(mask).astype(bool)
    rows, cols = np.nonzero(arr)
    if rows.size == 0:
        return None
    return float(cols.mean()), float(rows.mean())


def mask_box_xyxy(mask: np.ndarray) -> list[float] | None:
    """Tight box in pixels, ``x1``/``y1`` exclusive (spec §4.1); ``None`` if empty."""
    arr = np.asarray(mask).astype(bool)
    rows, cols = np.nonzero(arr)
    if rows.size == 0:
        return None
    return [
        float(cols.min()),
        float(rows.min()),
        float(cols.max() + 1),
        float(rows.max() + 1),
    ]


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two same-sized masks. Two empty masks score 0.0 (an empty mask never "matches")."""
    arr_a = np.asarray(a).astype(bool)
    arr_b = np.asarray(b).astype(bool)
    if arr_a.shape != arr_b.shape:
        raise ValueError(f"mask shapes differ: {arr_a.shape} vs {arr_b.shape}")
    union = int(np.count_nonzero(arr_a | arr_b))
    if union == 0:
        return 0.0
    return float(np.count_nonzero(arr_a & arr_b)) / union


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """IoU of two ``[x0, y0, x1, y1]`` boxes with exclusive outer edges."""
    ax0, ay0, ax1, ay1 = (float(v) for v in a[:4])
    bx0, by0, bx1, by1 = (float(v) for v in b[:4])
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    union = max(0.0, (ax1 - ax0)) * max(0.0, (ay1 - ay0)) + max(0.0, (bx1 - bx0)) * max(
        0.0, (by1 - by0)
    ) - inter
    return 0.0 if union <= 0.0 else float(inter / union)


# ---------------------------------------------------------------------------
# Mask <-> schema.Mask
# ---------------------------------------------------------------------------


def _encode_png_mask(mask: np.ndarray) -> str:
    arr = np.asarray(mask).astype(bool)
    image = Image.fromarray(arr.astype(np.uint8) * 255, mode="L")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return bytes_b64(buf.getvalue())


def _decode_png_mask(data: str) -> np.ndarray:
    with Image.open(io.BytesIO(b64_bytes(data))) as image:
        arr = np.asarray(image.convert("L"))
    return arr > 127


def to_mask(mask: np.ndarray, encoding: str = "rle") -> Mask:
    """Wrap a bool mask in the serialisable :class:`~percepta.schema.Mask` container."""
    arr = np.asarray(mask).astype(bool)
    size = (int(arr.shape[0]), int(arr.shape[1]))
    if encoding == "rle":
        return Mask(encoding="rle", size=size, counts=encode_rle(arr))
    if encoding == "png":
        return Mask(encoding="png", size=size, data=_encode_png_mask(arr))
    if encoding == "none":
        return Mask(encoding="none", size=size)
    raise ValueError(f"unsupported mask encoding {encoding!r}")


def to_array(mask: Mask) -> np.ndarray:
    """Decode a :class:`~percepta.schema.Mask` back to a bool ndarray."""
    if mask.encoding == "rle":
        if mask.counts is None or mask.size is None:
            raise ValueError("rle mask without counts/size")
        return decode_rle(mask.counts, mask.size)
    if mask.encoding == "png":
        if not mask.data:
            raise ValueError("png mask without data")
        arr = _decode_png_mask(mask.data)
        if mask.size is not None and arr.shape != tuple(mask.size):
            raise ValueError(f"png mask shape {arr.shape} != declared size {tuple(mask.size)}")
        return arr
    # encoding == "none": we know the size but not the pixels.
    if mask.size is None:
        raise ValueError("mask encoding='none' cannot be decoded without size")
    return np.zeros(tuple(mask.size), dtype=bool)


# ---------------------------------------------------------------------------
# Image / depth codecs
# ---------------------------------------------------------------------------


def _png_bytes(arr: np.ndarray) -> bytes:
    image = Image.fromarray(np.asarray(arr, dtype=np.uint8), mode="RGB")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def encode_image(image: np.ndarray, encoding: str = "png", camera: str | None = None) -> ImageRef:
    """Encode an RGB uint8 ``(H, W, 3)`` array as an :class:`ImageRef`."""
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected RGB image (H, W, 3), got shape {arr.shape}")
    arr = arr.astype(np.uint8, copy=False)
    if encoding == "png":
        return ImageRef(encoding="png", data=bytes_b64(_png_bytes(arr)), camera=camera)
    if encoding == "jpeg":
        image_pil = Image.fromarray(arr, mode="RGB")
        buf = io.BytesIO()
        image_pil.save(buf, format="JPEG", quality=95)
        return ImageRef(encoding="jpeg", data=bytes_b64(buf.getvalue()), camera=camera)
    if encoding == "raw":
        shape = (int(arr.shape[0]), int(arr.shape[1]), 3)
        return ImageRef(
            encoding="raw",
            data=bytes_b64(np.ascontiguousarray(arr).tobytes()),
            shape=shape,
            camera=camera,
        )
    raise ValueError(f"unsupported image encoding {encoding!r}")


def decode_image(ref: ImageRef) -> np.ndarray:
    """Decode an :class:`ImageRef` to an RGB uint8 ``(H, W, 3)`` array."""
    raw = b64_bytes(ref.data)
    if ref.encoding == "raw":
        if ref.shape is None:
            raise ValueError("raw ImageRef requires shape")
        height, width, channels = (int(v) for v in ref.shape)
        arr = np.frombuffer(raw, dtype=np.uint8)
        if arr.size != height * width * channels:
            raise ValueError(
                f"raw payload has {arr.size} bytes, expected {height * width * channels}"
            )
        return arr.reshape(height, width, channels).copy()
    with Image.open(io.BytesIO(raw)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def encode_depth(depth: np.ndarray) -> str:
    """base64 of float32 metres, little-endian, C-contiguous."""
    arr = np.ascontiguousarray(np.asarray(depth, dtype="<f4"))
    return bytes_b64(arr.tobytes())


def decode_depth(data: str, shape: Iterable[int]) -> np.ndarray:
    height, width = (int(v) for v in shape)
    arr = np.frombuffer(b64_bytes(data), dtype="<f4")
    if arr.size != height * width:
        raise ValueError(f"depth payload has {arr.size} values, expected {height * width}")
    return arr.reshape(height, width).copy()
