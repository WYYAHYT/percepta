"""Synthetic scenes, cameras and calibrations shared by the whole test suite.

Owned by the test harness (Lead). Geometry-specific fixtures live in ``synth_geom.py`` so
parallel work never edits this file.

Conventions match the spec exactly: pixel ``u`` = column, ``v`` = row, origin top-left;
camera frame is OpenCV (x right, y down, z forward); ``T_world_cam`` maps camera -> world.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from percepta.masks import encode_image
from percepta.schema import Calibration, ImageRef

__all__ = [
    "RGB",
    "BLACK",
    "WHITE",
    "RED",
    "GREEN",
    "BLUE",
    "blank_image",
    "rect_image",
    "square_image",
    "intrinsics",
    "look_at",
    "make_calibration",
    "image_ref",
    "red_square_scene",
    "cube_points",
    "project",
]

RGB = tuple[int, int, int]

BLACK: RGB = (0, 0, 0)
WHITE: RGB = (255, 255, 255)
RED: RGB = (255, 0, 0)
GREEN: RGB = (0, 255, 0)
BLUE: RGB = (0, 0, 255)


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------


def blank_image(height: int = 480, width: int = 640, background: RGB = BLACK) -> np.ndarray:
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:, :] = background
    return image


def rect_image(
    height: int = 480,
    width: int = 640,
    *,
    rect: tuple[int, int, int, int] = (280, 200, 360, 280),
    color: RGB = RED,
    background: RGB = BLACK,
) -> np.ndarray:
    """Background image with one filled axis-aligned rectangle ``(x0, y0, x1, y1)``."""
    image = blank_image(height, width, background)
    x0, y0, x1, y1 = rect
    image[y0:y1, x0:x1] = color
    return image


def square_image(
    height: int = 480,
    width: int = 640,
    *,
    center: tuple[int, int] = (320, 240),
    size: int = 80,
    color: RGB = RED,
    background: RGB = BLACK,
) -> np.ndarray:
    """The S1 fixture: a filled square centred on ``center`` (u, v)."""
    cx, cy = center
    half = size // 2
    return rect_image(
        height,
        width,
        rect=(cx - half, cy - half, cx - half + size, cy - half + size),
        color=color,
        background=background,
    )


def red_square_scene(
    height: int = 480, width: int = 640, *, center: tuple[int, int] = (320, 240), size: int = 80
) -> np.ndarray:
    return square_image(height, width, center=center, size=size, color=RED)


# ---------------------------------------------------------------------------
# Camera model
# ---------------------------------------------------------------------------


def intrinsics(
    width: int = 640, height: int = 480, *, f: float | None = None
) -> list[list[float]]:
    """Pinhole intrinsics with square pixels, principal point at the image centre."""
    focal = float(f) if f is not None else 0.9 * float(width)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    return [[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]]


def look_at(
    eye: Sequence[float],
    target: Sequence[float] = (0.0, 0.0, 0.0),
    up: Sequence[float] = (0.0, 0.0, 1.0),
) -> list[list[float]]:
    """Build ``T_world_cam`` (4x4) for an OpenCV camera at ``eye`` looking at ``target``.

    ``z_cam = normalize(target - eye)``, ``x_cam = normalize(z_cam x up)``,
    ``y_cam = z_cam x x_cam`` (image y points *down*, so ``up`` is roughly ``-y_cam``).
    """
    eye_v = np.asarray(eye, dtype=np.float64)
    target_v = np.asarray(target, dtype=np.float64)
    up_v = np.asarray(up, dtype=np.float64)

    z_cam = target_v - eye_v
    norm = np.linalg.norm(z_cam)
    if norm < 1e-9:
        raise ValueError("eye and target must differ")
    z_cam = z_cam / norm

    x_cam = np.cross(z_cam, up_v)
    if np.linalg.norm(x_cam) < 1e-9:  # looking straight along `up`
        x_cam = np.cross(z_cam, np.array([0.0, 1.0, 0.0]))
    x_cam = x_cam / np.linalg.norm(x_cam)
    y_cam = np.cross(z_cam, x_cam)

    transform = np.eye(4, dtype=np.float64)
    transform[:3, 0] = x_cam
    transform[:3, 1] = y_cam
    transform[:3, 2] = z_cam
    transform[:3, 3] = eye_v
    return transform.tolist()


def make_calibration(
    camera: str = "cam_head",
    *,
    width: int = 640,
    height: int = 480,
    eye: Sequence[float] = (0.0, -1.0, 1.0),
    target: Sequence[float] = (0.0, 0.0, 0.0),
    up: Sequence[float] = (0.0, 0.0, 1.0),
    f: float | None = None,
    calibration_id: str | None = "test-calib",
) -> Calibration:
    return Calibration(
        camera=camera,
        width=width,
        height=height,
        K=intrinsics(width, height, f=f),
        T_world_cam=look_at(eye, target, up),
        id=calibration_id,
    )


def image_ref(image: np.ndarray, camera: str | None = None, encoding: str = "png") -> ImageRef:
    return encode_image(image, encoding=encoding, camera=camera)


def project(
    points_world: np.ndarray, K: Sequence[Sequence[float]], T_world_cam: Sequence[Sequence[float]]
) -> np.ndarray:
    """Project world points ``(N, 3)`` to pixels ``(N, 2)``; z<=0 points become NaN."""
    k = np.asarray(K, dtype=np.float64)
    t = np.asarray(T_world_cam, dtype=np.float64)
    world = np.atleast_2d(np.asarray(points_world, dtype=np.float64))
    homogeneous = np.concatenate([world, np.ones((world.shape[0], 1))], axis=1)
    cam = (np.linalg.inv(t) @ homogeneous.T).T[:, :3]
    z = cam[:, 2]
    pixels = (cam @ k.T)
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = pixels[:, :2] / pixels[:, 2:3]
    uv[z <= 1e-9] = np.nan
    return uv


def cube_points(center: Sequence[float], half_extent: float = 0.05) -> np.ndarray:
    """The 8 corners of an axis-aligned cube in world metres."""
    cx, cy, cz = (float(v) for v in center)
    h = float(half_extent)
    corners = []
    for dx in (-h, h):
        for dy in (-h, h):
            for dz in (-h, h):
                corners.append((cx + dx, cy + dy, cz + dz))
    return np.asarray(corners, dtype=np.float64)
