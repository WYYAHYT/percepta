"""Multi-view 3-D fusion (spec §9, decisions D4 and D35).

:func:`fuse` turns the upstream per-camera percepts of one request into a single
``Percept`` carrying a ``Geometry`` in the **world** frame. Paths are tried in this order;
when one cannot produce geometry, the next path is attempted:

1. **depth** — any calibrated camera whose :class:`~percepta.schema.Calibration` carries a
   ``DepthRef``. Each mask is back-projected to world points, the views are merged, and the
   geometry is the median centre, the 1/99-percentile bbox, the top ``z``, and the PCA yaw
   (spec §9.1).
2. **silhouette / prism** — at least two calibrated masks (spec §9.2 / D35). Candidate
   footprints retain plane-grid cells whose sampled vertical columns project inside every
   mask. Each candidate prism is re-projected into all views, and the peak of the complete
   height/mean-IoU curve wins. ``prism_iou_threshold`` only gates the low-confidence warning.
   Lateral uncertainty, height uncertainty and per-view disagreement are reported separately.
3. **single view without depth** — the highest-scoring calibrated mask gives a table-contact
   ray intersection plus an assumed height (spec §9.3). The result explicitly warns that its
   height is assumed, with height uncertainty at least 0.05 m. Without table information this
   path produces no ``Geometry``.

Hard rules enforced here:

* no ``Calibration`` -> **never** a ``Geometry`` (spec §9: 不得用像素猜米);
* every returned geometry has ``frame="world"`` and a non-null ``uncertainty_m``;
* ``views_used`` counts only views that actually contributed;
* empty/partial mask input is tolerated; without usable geometry ``fuse`` returns ``None``
  and lets the pipeline add ``geometry_unavailable``;
* malformed camera matrices raise ``ValueError``; request schemas reject them before inference.

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
from . import hull
from ._calibration import pinhole_intrinsics, rigid_transform
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

#: A height search is only meaningful with several candidates: one sample cannot have a peak, so
#: a coarse ``prism_step_m`` must not collapse the search to a single guess.
MIN_PRISM_STEPS = 8

#: Minimum uncertainty for the depth point-cloud path; the silhouette paths use separate estimates.
MIN_UNCERTAINTY_M = 1e-3

#: A "hang" reading (an object wider above than below) is taken only when it explains the other
#: views better than the prism reading by this much of IoU. Anything standing on its own footprint
#: is a prism, so the burden of proof is on the unusual reading.
HANG_MARGIN = 0.03

#: Per-view IoU spread, in metres of silhouette disagreement, above which the views are reported
#: as inconsistent. Two views of one object rarely disagree by more than a mask edge; a large
#: spread means one of them is partly occluded or shows something else.
DISAGREEMENT_WARN_M = 0.02

#: A footprint wider than this is not an object standing on a table. Silhouettes seen at a
#: grazing angle project to enormous regions on the plane, and the grid has to stay finite.
MAX_FOOTPRINT_SPAN_M = 1.2


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
    """Pick the best-scoring usable mask per camera; reject malformed camera matrices.

    Tolerant by contract: a percept without a mask, a camera without calibration, an
    undecodable RLE payload or a size mismatch is silently skipped rather than raising.
    A supplied nonphysical calibration raises ``ValueError`` before it can produce geometry.
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
        K = pinhole_intrinsics(calib.K)
        T = rigid_transform(calib.T_world_cam)
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
                K=K,
                T=T,
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
    """Legacy cone-slice IoU curve, stopping at the first threshold match.

    This preserves the pre-D35 rule for the old helper; the production prism path uses
    :func:`_height_curve_grid` and the peak of its complete sampled-column IoU curve.
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
    """Search the sampled-column prism IoU peak and report split uncertainty (D35)."""
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
    step = min(step, max_height)

    grid = _prism_grid(prism_views, table_z, max_height)
    if grid is None:
        return None

    # A height search needs several candidates to have a peak at all: with one sample there is
    # nothing to compare, and the answer is whatever that single guess scored. A caller passing an
    # absurd ``prism_step_m`` (a bug, or a deliberate "just give me something") would otherwise
    # silently get a one-sample search; floor it at a resolution that can still find a peak.
    steps = max(MIN_PRISM_STEPS, int(round(min(max_height / step, MAX_PRISM_HYPOTHESES))))
    step = max_height / float(steps)
    heights = [step * float(index) for index in range(1, steps + 1)]

    # P0: the score of a candidate height now comes from the *exact* prism — the cells whose whole
    # column is inside every mask — projected back into the other views. The old code scored a
    # cone slice at z_top, which is the object's shadow at that depth, and kept it only for
    # scoring while building the answer from a *different* (mid-height) slice.
    curve = _height_curve_grid(grid, prism_views, table_z, heights)
    if not curve:
        return None
    best_height, best_score = max(curve, key=lambda item: item[1])
    # Take the *peak* of the curve, not the first height that clears the threshold.
    #
    # Spec D4 said "smallest height whose IoU reaches prism_iou_threshold". Measured against a
    # synthetic cube that rule stops far too early: the curve rises monotonically to the true
    # height (IoU 0.96 at h=0.10 for a 10 cm cube) but already crosses 0.5 at h=0.04, so taking
    # the crossing biases every height low — a systematic error of h/2 in top_z and in the
    # centroid's z. The threshold now only gates confidence (did anything fit at all?), and the
    # answer is the best-fitting height.
    chosen = best_height if best_score >= threshold else None
    if chosen is None:
        chosen = best_height
        warnings.append("geometry_low_confidence")
    chosen_score = dict(curve)[chosen]

    top_z = table_z + chosen
    cells = hull.column_inside(prism_views, grid, table_z, top_z)
    metrics = hull.footprint_metrics(grid, cells)
    if metrics is None:
        return None

    # A union of top-plane silhouette shadows for objects wider above than below, kept only when
    # its IoU is clearly better. It remains a separate heuristic reading of the same masks.
    hang_cells = _hang_cells(prism_views, grid, table_z, chosen, cells)
    hang_metrics = None if hang_cells is None else hull.footprint_metrics(grid, hang_cells)
    kind = "prism"
    if hang_metrics is not None and hang_metrics.cell_count >= max(4, metrics.cell_count // 4):
        hang_score = _cells_score(grid, hang_cells, prism_views, table_z, chosen)
        if hang_score is not None and hang_score > chosen_score + HANG_MARGIN:
            kind = "hang"
            metrics = hang_metrics
            chosen_score = hang_score

    width = float(metrics.bbox_max[0] - metrics.bbox_min[0])
    depth = float(metrics.bbox_max[1] - metrics.bbox_min[1])

    # P1: two uncertainties, because they differ by an order of magnitude and a caller has to
    # know which one is the problem. Lateral comes from the mask edge and the grid; height comes
    # from the height search's step and how well the views agreed on it.
    lateral = _lateral_uncertainty(grid, prism_views, table_z, chosen)
    height_unc = _height_uncertainty(step, chosen_score, threshold, max_height)
    disagreement = _view_disagreement(grid, prism_views, table_z, chosen)

    geometry = Geometry(
        frame="world",
        centroid=[
            float(metrics.centroid_xy[0]),
            float(metrics.centroid_xy[1]),
            float(table_z + 0.5 * chosen),
        ],
        bbox3d_min=[float(metrics.bbox_min[0]), float(metrics.bbox_min[1]), float(table_z)],
        bbox3d_max=[float(metrics.bbox_max[0]), float(metrics.bbox_max[1]), float(top_z)],
        top_z=float(top_z),
        yaw_rad=float(metrics.yaw_rad),
        # ``uncertainty_m`` keeps its frozen meaning (the worst axis) so existing callers do not
        # silently become more confident; the split lives in the dedicated fields below.
        uncertainty_m=float(max(lateral, height_unc)),
        views_used=len(prism_views),
        lateral_uncertainty_m=float(lateral),
        height_uncertainty_m=float(height_unc),
        view_disagreement_m=disagreement,
        method=kind,
    )
    extras: dict[str, Any] = {
        "method": "prism",
        "reading": kind,
        "views": [view.camera for view in prism_views],
        "prism_height_m": float(chosen),
        "prism_iou": float(chosen_score),
        "prism_iou_threshold": float(threshold),
        "prism_best_height_m": float(best_height),
        "prism_best_iou": float(best_score),
        "prism_scanned_heights": len(curve),
        "footprint_area_m2": float(metrics.area_m2),
        "footprint_extent_m": [width, depth],
        "footprint_cells": metrics.cell_count,
        "grid_cell_m": float(grid.cell),
        "table_z": float(table_z),
        "lateral_uncertainty_m": float(lateral),
        "height_uncertainty_m": float(height_unc),
        "view_disagreement_m": disagreement,
        "warnings": warnings,
    }
    if disagreement is not None and disagreement > DISAGREEMENT_WARN_M:
        warnings.append("geometry_inconsistent_views")
    return geometry, extras


@dataclass
class _PrismGridCache:
    """Placeholder type kept for the public surface; the grid itself is a ``hull.PlaneGrid``."""


def _prism_grid(
    prism_views: list[_PrismView], table_z: float, max_height: float
) -> Any:
    """Plane grid covering every view's shadow, from the base to the tallest candidate.

    The bounds come from the shadow at the base (where the silhouette is widest as a rule) unioned
    with the shadow at ``max_height``: an object cannot stand outside the region its own masks
    allow at either extreme, so the grid never clips a real footprint.
    """
    lo = np.array([np.inf, np.inf])
    hi = np.array([-np.inf, -np.inf])
    found = False
    for view in prism_views:
        for height in (0.0, max_height):
            hits = geo.ray_horizontal_plane(view.origin, view.dirs, table_z + height)
            hits = hits[np.all(np.isfinite(hits), axis=1)]
            if hits.shape[0] < 1:
                continue
            found = True
            lo = np.minimum(lo, hits[:, :2].min(axis=0))
            hi = np.maximum(hi, hits[:, :2].max(axis=0))
    if not found or not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)):
        return None
    margin = 0.01
    # A shadow seen at a grazing angle can be enormous (the plane is nearly parallel to the rays).
    # Cap the span: a footprint larger than this is not an object on a table.
    span = hi - lo
    if np.any(span > MAX_FOOTPRINT_SPAN_M):
        centre = 0.5 * (lo + hi)
        hi = np.where(span > MAX_FOOTPRINT_SPAN_M, centre + MAX_FOOTPRINT_SPAN_M / 2, hi)
        lo = np.where(span > MAX_FOOTPRINT_SPAN_M, centre - MAX_FOOTPRINT_SPAN_M / 2, lo)
    return hull.PlaneGrid.fit(lo - margin, hi + margin)


def _height_curve_grid(
    grid: Any,
    prism_views: list[_PrismView],
    table_z: float,
    heights: list[float],
) -> list[tuple[float, float]]:
    """Mean mask IoU of the exact prism at each candidate height."""
    curve: list[tuple[float, float]] = []
    for height in heights:
        top_z = table_z + height
        cells = hull.column_inside(prism_views, grid, table_z, top_z)
        if int(np.count_nonzero(cells)) < 4:
            curve.append((float(height), 0.0))
            continue
        score = _cells_score(grid, cells, prism_views, table_z, height)
        curve.append((float(height), 0.0 if score is None else float(score)))
    return curve


def _cells_score(
    grid: Any,
    cells: np.ndarray,
    prism_views: list[_PrismView],
    table_z: float,
    height: float,
) -> float | None:
    """Mean IoU of a prism built on ``cells`` against every view's mask.

    The prism is the cell set extruded from ``table_z`` to ``table_z + height``. Its silhouette in
    a view is the hull of its projected corners; comparing that with the observed mask is what
    scores a candidate, and it is the same measure the old code used — only now the prism under it
    is the real one.
    """
    xy = grid.xy()[cells]
    if len(xy) == 0:
        return None
    # the footprint outline plus its own base/top heights: a prism's silhouette is the hull of
    # these projected points, so the hull of the 2-D corners suffices for every view
    corners = geo.convex_hull_2d(xy)
    if corners.shape[0] < 3:
        return None
    vertices = _prism_vertices(corners, table_z, table_z + height)
    scores: list[float] = []
    for target in prism_views:
        silhouette = _project_to_view(vertices, target)
        if silhouette is None:
            continue
        scores.append(_projected_iou(silhouette, target))
    return float(np.mean(scores)) if scores else None


def _hang_cells(
    prism_views: list[_PrismView],
    grid: Any,
    table_z: float,
    height: float,
    prism_cells: np.ndarray,
) -> np.ndarray | None:
    """Build a union of top-plane silhouette shadows on a world-plane grid.

    Input rays and their plane intersections are in world metres. Polygon coordinates become
    grid pixels through ``uv = (xy - grid.lo) / grid.cell``; output is a row-major bool cell mask.
    The per-view masks are unioned. ``prism_cells`` remains accepted but unused by this reading;
    the caller selects it only when its mean silhouette IoU clearly improves on the prism.
    """
    top_z = table_z + height
    allowed = np.zeros(grid.size, dtype=bool)
    for view in prism_views:
        hits = geo.ray_horizontal_plane(view.origin, view.dirs, top_z)
        hits = hits[np.all(np.isfinite(hits), axis=1)]
        if hits.shape[0] < 3:
            return None
        ring = geo.convex_hull_2d(hits[:, :2])
        if ring.shape[0] < 3:
            return None
        # grid.lo is the centre of cell (0, 0); the rasterizer samples integer pixel centres.
        ring_uv = (ring - grid.lo) / grid.cell
        raster = geo.polygon_mask(ring_uv, (grid.ny, grid.nx))
        allowed |= raster.ravel()
        _ = prism_cells
    return allowed


def _lateral_uncertainty(
    grid: Any, prism_views: list[_PrismView], table_z: float, height: float
) -> float:
    """Estimate lateral uncertainty from one-pixel angular error and grid quantisation.

    Both ``hit`` and ``origin`` are world-frame metres, so ``distance = norm(hit - origin)``.
    The projected scale is ``distance / focal / sin(elevation)``, largest for a grazing camera.
    The worst view sets this heuristic estimate, with half a grid cell added for quantisation.
    """
    worst = 0.0
    for view in prism_views:
        rows, cols = np.nonzero(view.mask)
        if len(rows) == 0:
            continue
        focal = float(view.K[0, 0])
        if focal <= 0.0:
            continue
        # the ray through the middle of the mask, to the object's mid height
        mid_uv = np.array([[float(np.mean(cols)), float(np.mean(rows))]])
        origin, dirs = geo.rays_from_pixels(mid_uv, view.K, view.T)
        direction = dirs[0]
        elevation = abs(float(direction[2]))
        elevation = max(elevation, 0.05)
        hit = geo.ray_horizontal_plane(origin, dirs, table_z + 0.5 * height)
        if not np.all(np.isfinite(hit)):
            continue
        distance = float(np.linalg.norm(hit[0] - origin))
        worst = max(worst, distance / focal / elevation)
    return float(worst + 0.5 * grid.cell)


def _ray_plane_uncertainty(origin, dirs, hit, view: _View) -> float:
    """How far a ray/plane intersection moves for one pixel of mask error (P1).

    This path's lateral error is not the grid's: it is the mask edge, projected. One pixel moves
    the ray by ``1/f`` radians, and where the ray meets a horizontal plane that becomes
    ``distance / focal / sin(elevation)`` — largest for a grazing camera, which is exactly the
    case a caller should not trust without being told.
    """
    focal = float(np.asarray(view.K)[0, 0])
    if focal <= 0.0 or not np.all(np.isfinite(hit)):
        return DEFAULT_SINGLE_VIEW_UNCERTAINTY_M
    direction = np.asarray(dirs, dtype=np.float64).reshape(-1)[:3]
    elevation = max(abs(float(direction[2])), 0.05)
    distance = float(np.linalg.norm(np.asarray(hit).reshape(3) - np.asarray(origin).reshape(3)))
    return float(distance / focal / elevation)


def _height_uncertainty(
    step: float, score: float, threshold: float, max_height: float
) -> float:
    """Height uncertainty from the search step, widened for a below-threshold IoU peak (D35).

    ``geometry_low_confidence`` means no height reached the threshold, in which case the arg-max is
    a guess rather than a fit and the bound has to say so.
    """
    shortfall = max(0.0, float(threshold) - float(score))
    return float(step + shortfall * max_height * 0.5)


def _view_disagreement(
    grid: Any, prism_views: list[_PrismView], table_z: float, height: float
) -> float | None:
    """Largest disagreement between the per-view IoU scores of the chosen prism (P1).

    A single ``uncertainty_m`` cannot distinguish "all views agree, the answer is good" from "two
    views disagree by a lot, the average hides it". This reports the spread directly: a camera
    whose mask is partly occluded, or twice the object, shows up here and nowhere else.
    """
    top_z = table_z + height
    cells = hull.column_inside(prism_views, grid, table_z, top_z)
    xy = grid.xy()[cells]
    if len(xy) < 3:
        return None
    corners = geo.convex_hull_2d(xy)
    if corners.shape[0] < 3:
        return None
    vertices = _prism_vertices(corners, table_z, top_z)
    scores: list[float] = []
    for target in prism_views:
        silhouette = _project_to_view(vertices, target)
        if silhouette is None:
            continue
        scores.append(_projected_iou(silhouette, target))
    if len(scores) < 2:
        return None
    # IoU spread, converted to metres through the object's own extent: a 0.5 IoU gap on a 10 cm
    # object is a few centimetres of disagreement at the silhouette edge.
    spread = float(max(scores) - min(scores))
    extent = float(np.linalg.norm(xy.max(axis=0) - xy.min(axis=0)))
    return float(spread * max(extent, 0.01))


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
    # P1 applied to this path, where the split matters most. The ray/plane hit that gives (x, y)
    # is a geometric intersection: its error is one mask pixel, a few millimetres. The height, by
    # contrast, is not measured at all — one silhouette cannot tell a low deep object from a tall
    # shallow one — it is *assumed*. Reporting both under one number would let a caller believe
    # the height is as good as the position, which is exactly the mistake that makes a policy
    # descend to the wrong place.
    lateral = _ray_plane_uncertainty(origin, dirs, hit, view)
    height_unc = max(DEFAULT_SINGLE_VIEW_UNCERTAINTY_M, 0.5 * rough_height, 0.03)
    geometry = Geometry(
        frame="world",
        centroid=[float(hit[0]), float(hit[1]), float(table_z + 0.5 * rough_height)],
        bbox3d_min=None,
        bbox3d_max=None,
        top_z=float(table_z + rough_height),
        yaw_rad=None,
        uncertainty_m=float(max(lateral, height_unc)),
        views_used=1,
        lateral_uncertainty_m=float(lateral),
        height_uncertainty_m=float(height_unc),
        # A single view has nothing to disagree with itself; ``None`` says "unknown", which is
        # honest, rather than ``0.0`` which would read as "verified consistent".
        view_disagreement_m=None,
        method="single_view",
    )
    extras: dict[str, Any] = {
        "method": "single_view",
        "views": [view.camera],
        "assumed_height_m": float(rough_height),
        "lateral_uncertainty_m": float(lateral),
        "height_uncertainty_m": float(height_unc),
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

    Args:
        percepts: Earlier per-camera masks (spec §5.3, ``ctx.extras["upstream"]``).
        calibration: Camera names mapped to pixel intrinsics and rigid camera-to-world poses
            in metres, with ``p_world = T_world_cam @ p_cam_h``.
        context: Request context, including ``table_plane``/``table_z`` and prism tunables.
        options: Request options; ``fuse=False`` disables geometry.

    Returns:
        An ok ``Percept`` with world geometry, finite uncertainty and method/warnings in
        ``extras``, or ``None`` when no geometry is possible, including absent calibration.

    Raises:
        ValueError: A used camera has unsupported intrinsics or a nonrigid camera-to-world pose,
            including a calibration mutated after schema validation.

    Example:
        >>> fuse([], {}, {}) is None
        True
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
