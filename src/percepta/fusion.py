"""Multi-view 3-D fusion (spec §9, decision D4).

:func:`fuse` turns the upstream per-camera percepts of one request into a single
``Percept`` carrying a ``Geometry`` in the **world** frame. The priority order is exactly
spec §9:

1. **depth** — any calibrated camera whose :class:`~percepta.schema.Calibration` carries a
   ``DepthRef``. Each mask is back-projected to world points, the views are merged, and the
   geometry is the median centre, the 1/99-percentile bbox, the top ``z``, and the PCA yaw
   (spec §9.1).
2. **silhouette / prism** — no depth, at least two calibrated views (spec §9.2 / D4). Mask
   outlines are cast onto the table plane, a prism of candidate height ``h`` is built and
   re-projected into the *other* calibrated views, and the smallest ``h`` whose mean mask
   IoU reaches ``prism_iou_threshold`` wins.
3. **single view without depth** — 2-D only plus a *rough* height when the table plane is
   known, with ``uncertainty_m >= 0.03`` (spec §9.3). Without table information there is no
   ``Geometry`` at all.

Hard rules enforced here:

* no ``Calibration`` -> **never** a ``Geometry`` (spec §9: 不得用像素猜米);
* every returned geometry has ``frame="world"`` and a non-null ``uncertainty_m``;
* ``views_used`` counts only views that actually contributed;
* ``fuse`` never raises on empty/partial input — it returns ``None`` and lets the pipeline
  add ``geometry_unavailable``.

The prism-search tunables of D4 (``prism_max_height_m``, ``prism_step_m``,
``prism_iou_threshold``) are read from ``context`` with the D4 defaults, because
``PerceiveOptions`` is frozen (spec §3, ``AGENTS.md`` §3).

Warnings are carried in ``Percept.extras["warnings"]`` — the schema has no warning field on
a Percept, so the pipeline lifts them into ``PerceiveResponse.warnings``.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import geometry as geo
from .masks import decode_depth, to_array
from .schema import Geometry, PerceiveOptions, Percept
from .tools.base import build_provenance, make_percept

__all__ = [
    "TOOL_NAME",
    "TOOL_VERSION",
    "DEFAULT_PRISM_MAX_HEIGHT_M",
    "DEFAULT_PRISM_STEP_M",
    "DEFAULT_PRISM_IOU_THRESHOLD",
    "DEFAULT_SINGLE_VIEW_HEIGHT_M",
    "DEFAULT_SINGLE_VIEW_UNCERTAINTY_M",
    "fuse",
]

TOOL_NAME = "multi_view_3d"
TOOL_VERSION = "0.1.0"

#: D4 defaults. ``table_z`` defaults to 0 m (spec §9.2 "或默认 z=table_z").
DEFAULT_PRISM_MAX_HEIGHT_M = 0.5
DEFAULT_PRISM_STEP_M = 0.01
DEFAULT_PRISM_IOU_THRESHOLD = 0.5
DEFAULT_SINGLE_VIEW_HEIGHT_M = 0.10
DEFAULT_SINGLE_VIEW_UNCERTAINTY_M = 0.05

#: Hard cap on the number of height hypotheses, so a pathological ``step`` cannot stall a
#: request; when hit, the step is widened to cover the range with this many samples.
MAX_PRISM_HYPOTHESES = 512

#: A geometry is never reported with an uncertainty below this (depth noise floor).
MIN_UNCERTAINTY_M = 1e-3


# ---------------------------------------------------------------------------
# Input normalisation
# ---------------------------------------------------------------------------


@dataclass
class _View:
    """One calibrated mask that contributes to the fusion."""

    camera: str
    mask: np.ndarray
    K: np.ndarray
    T: np.ndarray
    score: float
    depth: np.ndarray | None = None
    calibration_id: str | None = None
    image_sha256: dict[str, str] = field(default_factory=dict)


def _collect_views(
    percepts: list[Percept] | None, calibration: dict[str, Any]
) -> list[_View]:
    """Pick the best-scoring mask per calibrated camera; skip everything unusable.

    Tolerant by contract: a percept without a mask, a camera without calibration, an
    undecodable RLE payload or a size mismatch is silently skipped rather than raising.
    """
    calibration = calibration or {}
    best: dict[str, _View] = {}
    for percept in percepts or []:
        if percept is None:
            continue
        camera = getattr(percept, "camera", None)
        if camera is None or camera not in calibration:
            continue
        calib = calibration[camera]
        expected = (int(calib.height), int(calib.width))
        depth: np.ndarray | None = None
        if calib.depth is not None:
            try:
                depth = decode_depth(calib.depth.data, calib.depth.shape)
                if depth.shape != expected:
                    depth = None
            except (ValueError, TypeError):
                depth = None
        hashes = dict(getattr(getattr(percept, "provenance", None), "image_sha256", {}) or {})
        for observation in getattr(percept, "observations", None) or []:
            if observation.mask is None:
                continue
            try:
                mask = to_array(observation.mask)
            except (ValueError, TypeError):
                continue
            if mask.ndim != 2 or tuple(mask.shape) != expected:
                continue
            if not bool(mask.any()):
                continue
            view = _View(
                camera=camera,
                mask=mask,
                K=np.asarray(calib.K, dtype=np.float64),
                T=np.asarray(calib.T_world_cam, dtype=np.float64),
                score=float(observation.score),
                depth=depth,
                calibration_id=calib.id,
                image_sha256=hashes,
            )
            current = best.get(camera)
            if current is None or view.score > current.score:
                best[camera] = view
    return [best[camera] for camera in sorted(best)]


def _context_value(context: dict[str, Any], key: str, default: float) -> float:
    """Read a positive float tunable from ``context``, falling back to the D4 default."""
    value = context.get(key, default)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(number) or number <= 0.0:
        return float(default)
    return number


# ---------------------------------------------------------------------------
# Table plane resolution
# ---------------------------------------------------------------------------


def _resolve_table(
    views: list[_View], context: dict[str, Any]
) -> tuple[float | None, list[str]]:
    """Resolve the table height ``z`` (world metres) for the silhouette paths.

    ``context["table_plane"] = [a, b, c, d]`` wins over ``context["table_z"]``. A tilted
    plane is honoured only by its height at the object (the median ``z`` where each view's
    mask-centroid ray meets the plane): v0.1 extrudes along world ``+z`` and assumes a
    near-horizontal table (spec §9's table assumption). Returns ``(z, warnings)``;
    ``z is None`` means "no table information was supplied".
    """
    warnings: list[str] = []
    plane_value = context.get("table_plane")
    if plane_value is not None:
        try:
            plane = geo.parse_plane(plane_value)
        except (TypeError, ValueError):
            warnings.append("invalid_table_plane")
            plane = None
        if plane is not None:
            if abs(float(plane[2])) < 0.999:
                warnings.append("table_plane_tilted")
            heights: list[float] = []
            for view in views:
                centroid_uv = geo.points_in_mask_uv(view.mask).mean(axis=0)
                origin, dirs = geo.rays_from_pixels(centroid_uv.reshape(1, 2), view.K, view.T)
                hit = geo.ray_plane_intersection(origin, dirs, plane)
                if np.all(np.isfinite(hit)):
                    heights.append(float(hit[0, 2]))
            if heights:
                return float(np.median(heights)), warnings
            warnings.append("table_plane_unreachable")

    table_z = context.get("table_z")
    if table_z is not None:
        try:
            value = float(table_z)
        except (TypeError, ValueError):
            warnings.append("invalid_table_z")
        else:
            if np.isfinite(value):
                return value, warnings
            warnings.append("invalid_table_z")
    return None, warnings


# ---------------------------------------------------------------------------
# Path 1: depth point cloud (spec §9.1)
# ---------------------------------------------------------------------------


def _fuse_depth(views: list[_View]) -> tuple[Geometry, dict[str, Any]] | None:
    """Fuse back-projected depth points from every calibrated view that has depth."""
    clouds: list[np.ndarray] = []
    view_centers: list[np.ndarray] = []
    used: list[str] = []
    warnings: list[str] = []
    for view in views:
        if view.depth is None:
            continue
        points = geo.mask_to_world_points(view.mask, view.depth, view.K, view.T)
        points = points[np.all(np.isfinite(points), axis=1)]
        if points.shape[0] < 3:
            warnings.append(f"depth_too_sparse:{view.camera}")
            continue
        clouds.append(points)
        view_centers.append(np.median(points, axis=0))
        used.append(view.camera)
    if not clouds:
        return None

    merged = np.concatenate(clouds, axis=0)
    if merged.shape[0] < 3:
        return None
    centroid = np.median(merged, axis=0)
    bbox_min, bbox_max = geo.percentile_bbox3d(merged)
    top_z = geo.top_z_from_points(merged)
    yaw = geo.yaw_from_points(merged[:, :2])

    centers = np.asarray(view_centers, dtype=np.float64)
    disagreement = 0.0
    if centers.shape[0] > 1:
        disagreement = float(np.max(np.linalg.norm(centers - centroid, axis=1)))
    robust_sigma = 1.4826 * float(np.median(np.linalg.norm(merged - centroid, axis=1)))
    noise = robust_sigma / float(np.sqrt(merged.shape[0]))
    uncertainty = max(float(np.hypot(disagreement, noise)), MIN_UNCERTAINTY_M)

    geometry = Geometry(
        frame="world",
        centroid=[float(v) for v in centroid],
        bbox3d_min=bbox_min,
        bbox3d_max=bbox_max,
        top_z=float(top_z),
        yaw_rad=float(yaw),
        uncertainty_m=float(uncertainty),
        views_used=len(used),
    )
    extras: dict[str, Any] = {
        "method": "depth",
        "views": used,
        "points": int(merged.shape[0]),
        "warnings": warnings,
    }
    return geometry, extras


# ---------------------------------------------------------------------------
# Path 2: silhouette / prism search (spec §9.2, decision D4)
# ---------------------------------------------------------------------------


@dataclass
class _PrismView:
    """A view reduced to the ingredients of the prism search."""

    camera: str
    mask: np.ndarray
    K: np.ndarray
    T: np.ndarray
    origin: np.ndarray
    dirs: np.ndarray
    box: tuple[int, int, int, int]


def _prism_views(views: list[_View]) -> list[_PrismView]:
    """Reduce each view to the rays of its silhouette hull, cast once.

    A projective map sends the convex hull of the mask to the convex hull of the hull
    vertices, so casting only the image-space hull corners gives exactly the same
    cross-section hull on every candidate plane, at a fraction of the cost.
    """
    prepared: list[_PrismView] = []
    for view in views:
        boundary = geo.mask_boundary_uv(view.mask)
        if boundary.shape[0] < 3:
            continue
        corners = geo.convex_hull_2d(boundary)
        if corners.shape[0] < 3:
            corners = boundary
        origin, dirs = geo.rays_from_pixels(corners, view.K, view.T)
        rows, cols = np.nonzero(view.mask)
        box = (
            int(rows.min()),
            int(rows.max()) + 1,
            int(cols.min()),
            int(cols.max()) + 1,
        )
        prepared.append(
            _PrismView(
                camera=view.camera,
                mask=view.mask,
                K=view.K,
                T=view.T,
                origin=origin,
                dirs=dirs,
                box=box,
            )
        )
    return prepared


def _slice_polygon(view: _PrismView, z: float) -> np.ndarray | None:
    """Cone cross-section of one view at the horizontal plane ``z``, world ``xy`` hull."""
    hits = geo.ray_horizontal_plane(view.origin, view.dirs, z)
    hits = hits[np.all(np.isfinite(hits), axis=1)]
    if hits.shape[0] < 3:
        return None
    hull = geo.convex_hull_2d(hits[:, :2])
    return hull if hull.shape[0] >= 3 else None


def _prism_vertices(poly_xy: np.ndarray, z_base: float, z_top: float) -> np.ndarray:
    """The 2N world vertices of a prism standing on ``z_base`` with top at ``z_top``."""
    base = np.column_stack((poly_xy, np.full(poly_xy.shape[0], float(z_base))))
    top = np.column_stack((poly_xy, np.full(poly_xy.shape[0], float(z_top))))
    return np.concatenate((base, top), axis=0)


def _project_to_view(points_w: np.ndarray, view: _PrismView) -> np.ndarray | None:
    """Convex hull of a world point set projected into one view, pixel frame."""
    camera_frame = geo.world_to_camera(points_w, view.T)
    uv = geo.project_points(camera_frame, view.K)
    uv = uv[np.all(np.isfinite(uv), axis=1)]
    if uv.shape[0] < 3:
        return None
    hull = geo.convex_hull_2d(uv)
    return hull if hull.shape[0] >= 3 else None


def _polygon_box(vertices_uv: np.ndarray, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    """Clipped integer ``(row0, row1, col0, col1)`` box of a pixel-frame polygon."""
    height, width = int(shape[0]), int(shape[1])
    cols = vertices_uv[:, 0]
    rows = vertices_uv[:, 1]
    row0 = max(0, min(height - 1, int(np.floor(rows.min()))))
    row1 = max(row0 + 1, min(height, int(np.ceil(rows.max())) + 1))
    col0 = max(0, min(width - 1, int(np.floor(cols.min()))))
    col1 = max(col0 + 1, min(width, int(np.ceil(cols.max())) + 1))
    return row0, row1, col0, col1


def _projected_iou(poly_uv: np.ndarray, target: _PrismView) -> float:
    """Mask IoU between a re-projected prism silhouette and the observed mask."""
    raster = geo.polygon_mask(poly_uv, target.mask.shape)
    row0, row1, col0, col1 = _polygon_box(poly_uv, target.mask.shape)
    m_row0, m_row1, m_col0, m_col1 = target.box
    row0, row1 = min(row0, m_row0), max(row1, m_row1)
    col0, col1 = min(col0, m_col0), max(col1, m_col1)
    model = raster[row0:row1, col0:col1]
    observed = target.mask[row0:row1, col0:col1]
    union = int(np.count_nonzero(model | observed))
    if union == 0:
        return 0.0
    return float(np.count_nonzero(model & observed)) / float(union)


def _height_curve(
    prism_views: list[_PrismView],
    table_z: float,
    heights: list[float],
    threshold: float,
) -> list[tuple[float, float]]:
    """Mean mask IoU for ascending candidate heights, stopping at the first match.

    D4 asks for the *smallest* height reaching ``threshold``, so once a height reaches it
    no larger height can change the answer (and the full curve is only needed for the
    arg-max fallback when nothing matches).
    """
    curve: list[tuple[float, float]] = []
    for height in heights:
        z_top = table_z + height
        scores: list[float] = []
        for index, source in enumerate(prism_views):
            cross_section = _slice_polygon(source, z_top)
            if cross_section is None:
                continue
            vertices = _prism_vertices(cross_section, table_z, z_top)
            for other_index, target in enumerate(prism_views):
                if other_index == index:
                    continue
                silhouette = _project_to_view(vertices, target)
                if silhouette is None:
                    continue
                scores.append(_projected_iou(silhouette, target))
        score = float(np.mean(scores)) if scores else 0.0
        curve.append((float(height), score))
        if score >= threshold:
            break
    return curve


def _fuse_prism(
    views: list[_View], context: dict[str, Any]
) -> tuple[Geometry, dict[str, Any]] | None:
    """D4 silhouette/prism height search over at least two calibrated views."""
    prism_views = _prism_views(views)
    if len(prism_views) < 2:
        return None

    table_z, warnings = _resolve_table(views, context)
    if table_z is None:
        table_z = 0.0
        warnings.append("table_z_assumed")

    max_height = _context_value(context, "prism_max_height_m", DEFAULT_PRISM_MAX_HEIGHT_M)
    step = _context_value(context, "prism_step_m", DEFAULT_PRISM_STEP_M)
    threshold = _context_value(context, "prism_iou_threshold", DEFAULT_PRISM_IOU_THRESHOLD)
    threshold = min(max(threshold, 0.0), 1.0)
    step = min(step, max_height)  # a step larger than the range would only sample nonsense

    steps = max(1, int(round(max_height / step)))
    if steps > MAX_PRISM_HYPOTHESES:
        steps = MAX_PRISM_HYPOTHESES
        step = max_height / float(steps)
    heights = [step * float(index) for index in range(1, steps + 1)]

    curve = _height_curve(prism_views, table_z, heights, threshold)
    if not curve:
        return None
    best_height, best_score = max(curve, key=lambda item: item[1])
    chosen = next((height for height, score in curve if score >= threshold), None)
    if chosen is None:
        chosen = best_height
        warnings.append("geometry_low_confidence")
    chosen_score = dict(curve)[chosen]

    # The winning height is D4's answer. The cross-section itself is taken at mid-height and
    # intersected across views (visual hull), which is symmetric in the views and therefore
    # cancels most of the single-view "shadow" bias of the prism slice.
    mid_z = table_z + 0.5 * chosen
    polygons = []
    for view in prism_views:
        polygon = _slice_polygon(view, mid_z)
        if polygon is not None:
            polygons.append(polygon)
    if not polygons:
        return None
    intersection = polygons[0]
    for polygon in polygons[1:]:
        intersection = geo.convex_polygon_intersection(intersection, polygon)
        if intersection.shape[0] < 3:
            break
    if intersection.shape[0] >= 3:
        cross_section = intersection
    else:
        cross_section = None
        warnings.append("geometry_inconsistent_views")

    if cross_section is not None:
        centroid_xy = geo.polygon_centroid(cross_section)
        yaw = geo.yaw_from_points(cross_section)
        cols, rows = cross_section[:, 0], cross_section[:, 1]
        # The silhouette primitive is a visual-hull slice, so this box is an *upper bound*
        # on the object's footprint, not a tight fit (no depth means no surface to fit).
        bbox_min = [float(cols.min()), float(rows.min()), float(table_z)]
        bbox_max = [float(cols.max()), float(rows.max()), float(table_z + chosen)]
    else:
        centroids = np.asarray([geo.polygon_centroid(polygon) for polygon in polygons])
        centroid_xy = centroids.mean(axis=0)
        yaw = geo.yaw_from_points(np.concatenate(polygons, axis=0))
        bbox_min = None
        bbox_max = None
    if not np.all(np.isfinite(centroid_xy)):
        return None

    # D4: step * (1 + 1/views_used), plus a margin proportional to the IoU shortfall.
    shortfall = max(0.0, threshold - float(chosen_score))
    uncertainty = step * (1.0 + 1.0 / float(len(prism_views)))
    uncertainty += shortfall * max_height * 0.5

    geometry = Geometry(
        frame="world",
        centroid=[float(centroid_xy[0]), float(centroid_xy[1]), float(table_z + 0.5 * chosen)],
        bbox3d_min=bbox_min,
        bbox3d_max=bbox_max,
        top_z=float(table_z + chosen),
        yaw_rad=float(yaw),
        uncertainty_m=float(uncertainty),
        views_used=len(prism_views),
    )
    extras: dict[str, Any] = {
        "method": "prism",
        "views": [view.camera for view in prism_views],
        "prism_height_m": float(chosen),
        "prism_iou": float(chosen_score),
        "prism_iou_threshold": float(threshold),
        # "best" refers to the heights actually scanned (the scan stops at the first match).
        "prism_best_height_m": float(best_height),
        "prism_best_iou": float(best_score),
        "prism_scanned_heights": len(curve),
        "table_z": float(table_z),
        "warnings": warnings,
    }
    return geometry, extras


# ---------------------------------------------------------------------------
# Path 3: single view without depth (spec §9.3)
# ---------------------------------------------------------------------------


def _fuse_single_view(
    view: _View, context: dict[str, Any]
) -> tuple[Geometry, dict[str, Any]] | None:
    """2-D only plus a *rough* height; requires table information (spec §9.3)."""
    table_z, warnings = _resolve_table([view], context)
    if table_z is None:
        return None
    rough_height = _context_value(
        context, "single_view_rough_height_m", DEFAULT_SINGLE_VIEW_HEIGHT_M
    )
    rough_height = min(max(rough_height, 0.01), 2.0)  # keep the assumption plausible
    pixels = geo.points_in_mask_uv(view.mask)
    if pixels.shape[0] == 0:
        return None
    bottom = pixels[pixels[:, 1] >= float(pixels[:, 1].max()) - 1.0]
    contact_uv = bottom.mean(axis=0)
    origin, dirs = geo.rays_from_pixels(contact_uv.reshape(1, 2), view.K, view.T)
    hit = geo.ray_horizontal_plane(origin, dirs, table_z)[0]
    if not np.all(np.isfinite(hit)):
        centroid_uv = pixels.mean(axis=0)
        origin, dirs = geo.rays_from_pixels(centroid_uv.reshape(1, 2), view.K, view.T)
        hit = geo.ray_horizontal_plane(origin, dirs, table_z)[0]
    if not np.all(np.isfinite(hit)):
        return None
    warnings.append("geometry_single_view")
    warnings.append("geometry_height_assumed")
    uncertainty = max(
        DEFAULT_SINGLE_VIEW_UNCERTAINTY_M, 0.5 * rough_height, 0.03
    )
    geometry = Geometry(
        frame="world",
        centroid=[float(hit[0]), float(hit[1]), float(table_z + 0.5 * rough_height)],
        bbox3d_min=None,
        bbox3d_max=None,
        top_z=float(table_z + rough_height),
        yaw_rad=None,
        uncertainty_m=float(uncertainty),
        views_used=1,
    )
    extras: dict[str, Any] = {
        "method": "single_view",
        "views": [view.camera],
        "assumed_height_m": float(rough_height),
        "table_z": float(table_z),
        "warnings": warnings,
    }
    return geometry, extras


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def fuse(
    percepts: list[Percept],
    calibration: dict[str, Any],
    context: dict[str, Any],
    options: PerceiveOptions | None = None,
) -> Percept | None:
    """Fuse upstream per-camera percepts into one world-frame ``Geometry``.

    ``percepts`` are the pipeline's earlier masks (spec §5.3, ``ctx.extras["upstream"]``);
    ``calibration`` maps camera -> :class:`~percepta.schema.Calibration`; ``context`` is the
    request context (``table_plane``/``table_z`` plus the D4 prism tunables);
    ``options.fuse=False`` disables geometry.

    Returns a ``Percept`` with ``status="ok"``, ``geometry.frame="world"``, a non-null
    ``uncertainty_m``, and the method/warnings in ``extras``. Returns ``None`` when no
    geometry can be produced — in particular whenever no calibration is supplied (spec §9:
    a calibrated camera is the only licence to convert pixels to metres).
    """
    started = time.perf_counter()
    if options is not None and getattr(options, "fuse", True) is False:
        return None
    views = _collect_views(percepts, calibration or {})
    if not views:
        return None
    context = dict(context or {})

    result: tuple[Geometry, dict[str, Any]] | None = None
    depth_views = [view for view in views if view.depth is not None]
    if depth_views:
        result = _fuse_depth(depth_views)
    if result is None and len(views) >= 2:
        result = _fuse_prism(views, context)
    if result is None:
        best_view = max(views, key=lambda view: view.score)
        result = _fuse_single_view(best_view, context)
    if result is None:
        return None

    geometry, extras = result
    if geometry.frame != "world" or geometry.uncertainty_m is None:
        return None
    if not np.isfinite(geometry.uncertainty_m):
        return None

    hashes: dict[str, str] = {}
    calibration_ids = set()
    for view in views:
        hashes.update(view.image_sha256)
        if view.calibration_id:
            calibration_ids.add(view.calibration_id)
    calibration_id = calibration_ids.pop() if len(calibration_ids) == 1 else None
    provenance = build_provenance(
        tool=TOOL_NAME,
        tool_version=TOOL_VERSION,
        kind="geometry",
        started_at=started,
        camera=None,
        calibration_id=calibration_id,
    ).model_copy(update={"image_sha256": hashes})
    return make_percept(
        percept_id=str(uuid.uuid4()),
        status="ok",
        provenance=provenance,
        camera=None,
        geometry=geometry,
        extras=extras,
    )
