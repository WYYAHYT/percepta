"""Geometry fixtures: calibrated camera rigs and analytic cube silhouettes.

Owned by the geometry milestone (``AGENTS.md`` §2); built on top of the shared
``tests/synth.py`` (read-only for everyone but the Lead).

Everything here is analytic and deterministic — no rasteriser library, no GPU, no network:

* a cube's silhouette in a calibrated view is the convex hull of its 8 projected corners
  (exact for a convex body whose vertices are all in front of the camera, spec §4);
* its depth map is the ray/box entry distance in camera ``z`` (metres), so the depth and
  silhouette paths are fed from the *same* ground-truth geometry.

Frames: pixel ``[u, v] = [column, row]`` (origin top-left); camera OpenCV
(``x`` right, ``y`` down, ``z`` forward); world metres, ``T_world_cam`` camera -> world.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence

import numpy as np

from percepta.geometry import (
    convex_hull_2d,
    points_in_mask_uv,
    polygon_mask,
    rays_from_pixels,
    world_to_camera,
)
from percepta.masks import encode_depth
from percepta.schema import Calibration, DepthRef, Percept, Prompt, Provenance
from percepta.tools.base import make_observation, make_percept
from tests import synth

__all__ = [
    "DEFAULT_PAIR_DISTANCE",
    "eye_from_spherical",
    "camera_pair",
    "camera_pair_distance",
    "camera_rig",
    "cube_silhouette",
    "cube_masks",
    "cube_image",
    "cube_depth",
    "cube_depth_ref",
    "two_view_roomcube",
    "mask_percepts",
]

#: Baseline that keeps a 10 cm cube roughly 75 px wide at the default 640x480 intrinsics.
DEFAULT_PAIR_DISTANCE = 0.75

_TABLE_Z = 0.0


# ---------------------------------------------------------------------------
# Camera rigs
# ---------------------------------------------------------------------------


def eye_from_spherical(
    azimuth_deg: float,
    elevation_deg: float,
    distance: float,
    target: Sequence[float],
) -> tuple[float, float, float]:
    """World position of a camera on a sphere around ``target`` (metres, world frame)."""
    az = np.deg2rad(float(azimuth_deg))
    el = np.deg2rad(float(elevation_deg))
    tx, ty, tz = (float(v) for v in target)
    return (
        tx + distance * np.cos(el) * np.cos(az),
        ty + distance * np.cos(el) * np.sin(az),
        tz + distance * np.sin(el),
    )


def camera_pair(
    distance: float = DEFAULT_PAIR_DISTANCE,
    *,
    elevation_deg: float = 20.0,
    azimuths_deg: tuple[float, float] = (0.0, 180.0),
    target: Sequence[float] = (0.0, 0.0, 0.05),
    width: int = 640,
    height: int = 480,
    f: float | None = None,
    names: tuple[str, str] = ("cam_a", "cam_b"),
) -> dict[str, Calibration]:
    """Two calibrated cameras on opposite sides of ``target`` (world frame).

    Both look at ``target`` with world ``+z`` up; elevation is measured from the table
    plane. Returns ``{name: Calibration}`` — the input of every fusion test.
    """
    calibrations: dict[str, Calibration] = {}
    for name, azimuth in zip(names, azimuths_deg, strict=True):
        eye = eye_from_spherical(azimuth, elevation_deg, distance, target)
        calibrations[name] = synth.make_calibration(
            name,
            width=width,
            height=height,
            eye=eye,
            target=tuple(float(v) for v in target),
            f=f,
        )
    return calibrations


def camera_pair_distance(calibrations: dict[str, Calibration]) -> float:
    """Baseline (metres) between the first two camera centres of a calibration dict."""
    centers = [np.asarray(c.T_world_cam, dtype=np.float64)[:3, 3] for _, c in sorted(calibrations.items())]
    if len(centers) < 2:
        raise ValueError("need at least two calibrations to measure a baseline")
    return float(np.linalg.norm(centers[0] - centers[1]))


def camera_rig(
    azimuths_deg: Sequence[float] = (0.0, 120.0, 240.0),
    *,
    distance: float = DEFAULT_PAIR_DISTANCE,
    elevation_deg: float = 20.0,
    target: Sequence[float] = (0.0, 0.0, 0.05),
    width: int = 640,
    height: int = 480,
    f: float | None = None,
) -> dict[str, Calibration]:
    """``N`` calibrated cameras on a circle around ``target`` (``cam_0``, ``cam_1``, ...)."""
    calibrations: dict[str, Calibration] = {}
    for index, azimuth in enumerate(azimuths_deg):
        eye = eye_from_spherical(azimuth, elevation_deg, distance, target)
        name = f"cam_{index}"
        calibrations[name] = synth.make_calibration(
            name,
            width=width,
            height=height,
            eye=eye,
            target=tuple(float(v) for v in target),
            f=f,
        )
    return calibrations


# ---------------------------------------------------------------------------
# Cube silhouettes / depth
# ---------------------------------------------------------------------------


def cube_silhouette(
    calibration: Calibration,
    center: Sequence[float] = (0.0, 0.0, 0.05),
    half_extent: float = 0.05,
) -> np.ndarray:
    """Analytic cube silhouette as a bool ``(H, W)`` mask in that camera's pixel frame.

    Projects the 8 world corners with ``K``/``T_world_cam`` and fills their convex hull:
    exact for a convex body whose vertices are all in front of the camera (spec §4).
    """
    corners = synth.cube_points(center, half_extent)
    uv = synth.project(corners, calibration.K, calibration.T_world_cam)
    uv = uv[np.all(np.isfinite(uv), axis=1)]
    if uv.shape[0] < 3:
        return np.zeros((calibration.height, calibration.width), dtype=bool)
    hull = convex_hull_2d(uv)
    return polygon_mask(hull, (calibration.height, calibration.width))


def cube_masks(
    calibrations: dict[str, Calibration],
    center: Sequence[float] = (0.0, 0.0, 0.05),
    half_extent: float = 0.05,
) -> dict[str, np.ndarray]:
    """Cube silhouettes for every calibration, ``{camera: bool (H, W)}``."""
    return {
        name: cube_silhouette(calib, center, half_extent)
        for name, calib in sorted(calibrations.items())
    }


def cube_image(
    mask: np.ndarray, color: synth.RGB = synth.RED, background: synth.RGB = synth.BLACK
) -> np.ndarray:
    """RGB image with the masked silhouette painted in ``color`` (uint8 ``(H, W, 3)``)."""
    image = np.empty((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    image[:, :] = background
    image[np.asarray(mask, dtype=bool)] = color
    return image


def cube_depth(
    calibration: Calibration,
    mask: np.ndarray,
    center: Sequence[float] = (0.0, 0.0, 0.05),
    half_extent: float = 0.05,
) -> np.ndarray:
    """Analytic depth (camera ``z`` metres) for the masked pixels of a cube.

    Ray/axis-aligned-box slab intersection from the camera centre; pixels outside the
    silhouette (or rays that miss the cube) are ``0.0``, which
    :func:`percepta.geometry.mask_to_world_points` treats as invalid. Output ``(H, W)``
    float32, camera frame.
    """
    arr = np.asarray(mask, dtype=bool)
    height, width = (int(arr.shape[0]), int(arr.shape[1]))
    depth = np.zeros((height, width), dtype=np.float32)
    uv = points_in_mask_uv(arr)
    if uv.shape[0] == 0:
        return depth
    origin, dirs = rays_from_pixels(uv, calibration.K, calibration.T_world_cam)
    lo = np.asarray(center, dtype=np.float64) - float(half_extent)
    hi = np.asarray(center, dtype=np.float64) + float(half_extent)
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (lo - origin) / dirs
        t2 = (hi - origin) / dirs
    t_near = np.nanmax(np.minimum(t1, t2), axis=1)
    t_far = np.nanmin(np.maximum(t1, t2), axis=1)
    hit = (t_far >= np.maximum(t_near, 0.0)) & np.isfinite(t_near) & (t_near > 0.0)
    points_world = origin + t_near[:, None] * dirs
    cam = world_to_camera(points_world, calibration.T_world_cam)
    depth[arr] = np.where(hit, cam[:, 2], 0.0).astype(np.float32)
    return depth


def cube_depth_ref(
    calibration: Calibration,
    mask: np.ndarray,
    center: Sequence[float] = (0.0, 0.0, 0.05),
    half_extent: float = 0.05,
) -> DepthRef:
    """``DepthRef`` (spec §3.3 ``raw_f32``) for :func:`cube_depth`, metres."""
    depth = cube_depth(calibration, mask, center, half_extent)
    return DepthRef(
        encoding="raw_f32",
        data=encode_depth(depth),
        shape=(int(calibration.height), int(calibration.width)),
    )


# ---------------------------------------------------------------------------
# Scenes
# ---------------------------------------------------------------------------


def two_view_roomcube(
    center: Sequence[float] = (0.0, 0.0, 0.05),
    half_extent: float = 0.05,
    *,
    distance: float = DEFAULT_PAIR_DISTANCE,
    elevation_deg: float = 20.0,
    azimuths_deg: tuple[float, float] = (0.0, 180.0),
    with_depth: bool = False,
    width: int = 640,
    height: int = 480,
) -> tuple[dict[str, Calibration], dict[str, np.ndarray], tuple[float, float, float]]:
    """The S5 fixture: two calibrated views of a known cube on the table.

    Returns ``(calibrations, images, cube_center)`` where ``images`` are RGB uint8 views of
    the painted silhouette (usable by the ``fake`` tool too). Pass ``with_depth=True`` to
    attach an analytic :class:`~percepta.schema.DepthRef` to each calibration, which
    exercises the depth path of :func:`percepta.fusion.fuse` instead of the prism search.
    """
    target = (float(center[0]), float(center[1]), float(center[2]))
    calibrations = camera_pair(
        distance,
        elevation_deg=elevation_deg,
        azimuths_deg=azimuths_deg,
        target=target,
        width=width,
        height=height,
    )
    center_v = (float(center[0]), float(center[1]), float(center[2]))
    masks = cube_masks(calibrations, center_v, half_extent)
    images = {name: cube_image(mask) for name, mask in masks.items()}
    if with_depth:
        for name, calib in calibrations.items():
            calibrations[name] = calib.model_copy(
                update={"depth": cube_depth_ref(calib, masks[name], center_v, half_extent)}
            )
    return calibrations, images, center_v


# ---------------------------------------------------------------------------
# Percept fixtures for fusion tests
# ---------------------------------------------------------------------------


def mask_percepts(
    calibrations: dict[str, Calibration],
    masks: dict[str, np.ndarray],
    *,
    score: float = 0.9,
    label: str = "cube",
    prompt: Prompt | None = None,
) -> list[Percept]:
    """Wrap ``{camera: mask}`` into upstream ``Percept``s (one per camera, status ``ok``)."""
    percepts: list[Percept] = []
    for camera in sorted(masks):
        calib = calibrations.get(camera)
        if calib is None:
            continue
        percept_id = str(uuid.uuid4())
        observation = make_observation(
            np.asarray(masks[camera], dtype=bool),
            score,
            version_percept_id=percept_id,
            index=0,
            encoding="rle",
            label=label,
        )
        provenance = Provenance(
            tool="synth",
            tool_version="0.0.0",
            kind="fake",
            prompt=prompt,
            camera=camera,
            image_sha256={},
            calibration_id=calib.id,
            latency_ms=0.0,
        )
        percepts.append(
            make_percept(
                percept_id=percept_id,
                status="ok",
                provenance=provenance,
                camera=camera,
                observations=[observation],
            )
        )
    return percepts
