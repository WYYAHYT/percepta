"""Pure-numpy geometry primitives for percepta (spec §4, §9; decision D4).

Frames used throughout (frozen — ``AGENTS.md`` §3 / spec §4):

* **pixel** — ``uv = [u, v] = [column, row]``, origin at the top-left corner, ``u`` grows
  right, ``v`` grows down, units of pixels; ``x1``/``y1`` box edges are exclusive.
* **camera** — OpenCV convention: ``x`` right, ``y`` down, ``z`` forward, metres.
* **world** — right-handed, metres, defined by ``T_world_cam`` (**camera -> world**,
  ``p_world = T_world_cam @ [X, Y, Z, 1]``).

Every public function states its input frame, output frame and formula (spec §16.5), never
mutates its inputs, and raises :class:`ValueError` on shapes it cannot interpret. No
dependency beyond numpy (no scipy, spec §16.3).
"""

from __future__ import annotations

import numpy as np

from ._calibration import pinhole_intrinsics, rigid_transform

__all__ = [
    "project_points",
    "unproject_pixels",
    "pixel_to_world",
    "mask_to_world_points",
    "ray_plane_intersection",
    "ray_horizontal_plane",
    "bbox3d_from_points",
    "percentile_bbox3d",
    "yaw_from_points",
    "points_in_mask_uv",
    "top_z_from_points",
    "plane_from_table_z",
    "parse_plane",
    "world_to_camera",
    "rays_from_pixels",
    "mask_boundary_uv",
    "convex_hull_2d",
    "polygon_mask",
    "polygon_area",
    "polygon_centroid",
    "convex_polygon_intersection",
]

_EPS = 1e-9
_PLANE_EPS = 1e-12


# ---------------------------------------------------------------------------
# Small shape/validation helpers
# ---------------------------------------------------------------------------


def _intrinsics(K) -> np.ndarray:
    """Return validated float64 zero-skew pinhole intrinsics in pixel units."""
    return pinhole_intrinsics(K)


def _transform(T) -> np.ndarray:
    """Return ``T`` as a validated ``(4, 4)`` float64 rigid transform."""
    return rigid_transform(T)


def _pixels(uv) -> np.ndarray:
    """Return ``uv`` as ``(N, 2)`` float64 pixel coordinates."""
    arr = np.asarray(uv, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"uv must be (N, 2) pixels, got shape {arr.shape}")
    return arr


def _points3(points, name: str = "points") -> np.ndarray:
    """Return ``points`` as ``(N, 3)`` float64."""
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{name} must be (N, 3), got shape {arr.shape}")
    return arr


def _plane4(plane) -> np.ndarray:
    """Validate ``[a, b, c, d]`` (``a*x + b*y + c*z + d = 0``); returns a new array."""
    arr = np.asarray(plane, dtype=np.float64).reshape(-1)
    if arr.size != 4:
        raise ValueError(f"plane must be [a, b, c, d], got {arr.size} values")
    if not np.all(np.isfinite(arr)):
        raise ValueError("plane contains non-finite values")
    if np.linalg.norm(arr[:3]) <= _EPS:
        raise ValueError("degenerate plane: normal [a, b, c] is (almost) zero")
    return arr


# ---------------------------------------------------------------------------
# Projection / unprojection (spec §4)
# ---------------------------------------------------------------------------


def project_points(points_cam: np.ndarray, K) -> np.ndarray:
    """Project camera-frame points to pixels (pinhole, no distortion).

    Input frame: **camera** (OpenCV), ``points_cam`` is ``(N, 3)`` metres.
    Output frame: **pixel**, ``(N, 2)`` ``[u, v]`` floats.
    Formula: ``u = fx * X / Z + cx``, ``v = fy * Y / Z + cy`` (``K = [[fx,0,cx],...]``).
    Points with ``Z <= 0`` (behind the camera) become ``NaN`` rows.
    """
    pts = _points3(points_cam, name="points_cam")
    k = _intrinsics(K)
    fx, fy = float(k[0, 0]), float(k[1, 1])
    cx, cy = float(k[0, 2]), float(k[1, 2])
    z = pts[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = fx * pts[:, 0] / z + cx
        v = fy * pts[:, 1] / z + cy
    uv = np.stack((u, v), axis=1)
    uv[~(z > 0.0)] = np.nan
    return uv


def unproject_pixels(uv, depth, K) -> np.ndarray:
    """Lift pixels to 3-D camera-frame points using a depth map.

    Input frame: **pixel** ``uv`` ``(N, 2)`` plus per-pixel depth ``(N,)`` metres
    (camera-frame ``Z``), and intrinsics ``K``.
    Output frame: **camera** (OpenCV), ``(N, 3)`` metres.
    Formula: ``X = (u - cx) * Z / fx``, ``Y = (v - cy) * Z / fy``, ``Z = Z``.
    Rows with non-finite or non-positive depth become ``NaN``.
    """
    pix = _pixels(uv)
    z = np.asarray(depth, dtype=np.float64).reshape(-1)
    if z.shape[0] != pix.shape[0]:
        raise ValueError(f"depth has {z.shape[0]} values for {pix.shape[0]} pixels")
    k = _intrinsics(K)
    fx, fy = float(k[0, 0]), float(k[1, 1])
    cx, cy = float(k[0, 2]), float(k[1, 2])
    with np.errstate(divide="ignore", invalid="ignore"):
        x = (pix[:, 0] - cx) * z / fx
        y = (pix[:, 1] - cy) * z / fy
    cam = np.stack((x, y, z), axis=1)
    bad = ~np.isfinite(z) | (z <= 0.0)
    cam[bad] = np.nan
    return cam


def pixel_to_world(uv, depth, K, T_world_cam) -> np.ndarray:
    """Lift pixels + depth into the world frame.

    Input frame: **pixel** ``uv`` ``(N, 2)`` and depth ``(N,)`` metres (camera ``Z``).
    Output frame: **world**, ``(N, 3)`` metres.
    Formula: ``p_world = T_world_cam @ [X, Y, Z, 1]^T`` where ``(X, Y, Z)`` comes from
    :func:`unproject_pixels`. ``T_world_cam`` is camera -> world.
    """
    cam = unproject_pixels(uv, depth, K)
    t = _transform(T_world_cam)
    ones = np.ones((cam.shape[0], 1), dtype=np.float64)
    homogeneous = np.concatenate((cam, ones), axis=1)
    world = (t @ homogeneous.T).T[:, :3]
    valid = np.all(np.isfinite(cam), axis=1)
    world[~valid] = np.nan
    return world


def mask_to_world_points(mask, depth, K, T_world_cam) -> np.ndarray:
    """World points for the pixels of a mask that carry a valid depth reading.

    Input frame: **pixel** — ``mask`` ``(H, W)`` bool and ``depth`` ``(H, W)`` metres.
    Output frame: **world**, ``(M, 3)`` metres, one row per valid masked pixel in
    row-major order. Valid means ``isfinite(depth) and depth > 0``.
    """
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError(f"mask must be 2-D (H, W), got shape {arr.shape}")
    arr = arr.astype(bool)
    d = np.asarray(depth, dtype=np.float64)
    if d.shape != arr.shape:
        raise ValueError(f"depth shape {d.shape} != mask shape {arr.shape}")
    uv = points_in_mask_uv(arr)
    z = d[arr]
    valid = np.isfinite(z) & (z > 0.0)
    uv = uv[valid]
    z = z[valid]
    if uv.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float64)
    return pixel_to_world(uv, z, K, T_world_cam)


# ---------------------------------------------------------------------------
# Rays and planes
# ---------------------------------------------------------------------------


def ray_plane_intersection(origin_w, dirs_w, plane) -> np.ndarray:
    """Intersect world-space rays with a plane.

    Input frame: **world** — ``origin_w`` ``(3,)`` (or ``(N, 3)``, broadcast), ``dirs_w``
    ``(N, 3)`` direction vectors (not required to be unit length), ``plane``
    ``[a, b, c, d]`` meaning ``a*x + b*y + c*z + d = 0``.
    Output frame: **world**, ``(N, 3)`` hit points.
    Formula: ``t = -(n . origin + d) / (n . dir)``, ``p = origin + t * dir`` with
    ``n = plane[:3]``. A ray parallel to the plane, or a hit behind the origin
    (``t < 0``), yields a ``NaN`` row (undefined / not a ray hit).
    """
    origin = np.asarray(origin_w, dtype=np.float64)
    if origin.ndim == 1:
        origin = origin.reshape(1, 3)
    if origin.ndim != 2 or origin.shape[1] != 3:
        raise ValueError(f"origin_w must be (3,) or (N, 3), got {origin.shape}")
    dirs = _points3(dirs_w, name="dirs_w")
    p = _plane4(plane)
    n, d = p[:3], float(p[3])

    if origin.shape[0] == 1 and dirs.shape[0] != 1:
        origin = np.broadcast_to(origin, dirs.shape)
    elif origin.shape[0] != dirs.shape[0]:
        raise ValueError(
            f"origin_w has {origin.shape[0]} rows but dirs_w has {dirs.shape[0]}"
        )

    denom = dirs @ n
    numer = origin @ n + d
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        t = -numer / denom
        hits = origin + t[:, None] * dirs
    bad = (~np.isfinite(t)) | (np.abs(denom) <= _PLANE_EPS) | (t < 0.0)
    hits[bad] = np.nan
    return hits


def ray_horizontal_plane(origin_w, dirs_w, z: float) -> np.ndarray:
    """Intersect world rays with the horizontal plane ``z = const``.

    Input/output frames: **world**. This is :func:`ray_plane_intersection` with the plane
    ``[0, 0, 1, -z]``; formula ``t = (z - o_z) / d_z``, ``p = o + t * d``.
    """
    return ray_plane_intersection(origin_w, dirs_w, plane_from_table_z(z))


def plane_from_table_z(z: float) -> np.ndarray:
    """Horizontal plane ``z = const`` as ``[0, 0, 1, -z]`` (world frame, metres)."""
    return np.asarray([0.0, 0.0, 1.0, -float(z)], dtype=np.float64)


def parse_plane(value) -> np.ndarray:
    """Parse a plane spec (``context["table_plane"]``) into a normalised ``[a,b,c,d]``.

    Accepts any 4-element sequence of finite numbers meaning ``a*x + b*y + c*z + d = 0``.
    The result is scaled so ``|(a, b, c)| == 1`` (which leaves the plane unchanged).
    Raises :class:`ValueError` for wrong arity, non-finite values, or a zero normal.
    """
    p = _plane4(value)
    norm = float(np.linalg.norm(p[:3]))
    return p / norm


# ---------------------------------------------------------------------------
# 3-D statistics (spec §9.1)
# ---------------------------------------------------------------------------


def bbox3d_from_points(points) -> tuple[list[float], list[float]]:
    """Axis-aligned bounding box of a world point cloud (min, max), metres.

    Input/output frames: **world**. Output ``(min_xyz, max_xyz)`` as plain lists for JSON.
    """
    pts = _points3(points)
    if pts.shape[0] == 0:
        raise ValueError("cannot bound an empty point cloud")
    finite = np.all(np.isfinite(pts), axis=1)
    if not finite.any():
        raise ValueError("point cloud has no finite points")
    pts = pts[finite]
    return (
        [float(v) for v in pts.min(axis=0)],
        [float(v) for v in pts.max(axis=0)],
    )


def percentile_bbox3d(points, lo: float = 1.0, hi: float = 99.0) -> tuple[list[float], list[float]]:
    """Per-axis percentile bounding box ("分位数包围盒", spec §9.1), world frame, metres.

    Percentiles reject depth outliers without assuming a noise model. ``lo``/``hi`` are
    percentages in ``[0, 100]`` with ``lo < hi``. Output ``(min_xyz, max_xyz)`` lists.
    """
    if not (0.0 <= float(lo) < float(hi) <= 100.0):
        raise ValueError(f"need 0 <= lo < hi <= 100, got lo={lo}, hi={hi}")
    pts = _points3(points)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if pts.shape[0] == 0:
        raise ValueError("point cloud has no finite points")
    mins = np.percentile(pts, float(lo), axis=0)
    maxs = np.percentile(pts, float(hi), axis=0)
    return (
        [float(v) for v in mins],
        [float(v) for v in maxs],
    )


def yaw_from_points(points_xy) -> float:
    """Principal-axis yaw of a planar point set, radians in ``(-pi/2, pi/2]``.

    Input: **world** (or any planar frame) ``(N, 2)`` ``xy`` coordinates, or ``(N, 3)`` whose
    first two columns are used. Output: rotation of the dominant axis about +z, i.e. the
    object's yaw in the world frame when the points are a top-down footprint.
    Method: PCA via ``np.linalg.eigh`` of the 2x2 covariance (no scipy), then
    ``yaw = atan2(vy, vx)`` of the major eigenvector, folded into ``(-pi/2, pi/2]`` because
    an axis is sign-ambiguous. Degenerate input (``< 2`` points or zero variance) -> ``0.0``.
    """
    pts = np.asarray(points_xy, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 2:
        raise ValueError(f"points_xy must be (N, 2), got shape {pts.shape}")
    xy = pts[:, :2]
    if xy.shape[0] < 2:
        return 0.0
    centered = xy - xy.mean(axis=0)
    cov = centered.T @ centered / float(xy.shape[0] - 1)
    if not np.any(np.abs(cov) > 0.0):
        return 0.0
    evals, evecs = np.linalg.eigh(cov)  # ascending eigenvalues, eigenvectors in columns
    major = evecs[:, int(np.argmax(evals))]
    angle = float(np.arctan2(major[1], major[0]))
    angle = angle % np.pi  # [0, pi): an axis, so +pi is the same direction
    if angle > np.pi / 2.0:
        angle -= np.pi
    return float(angle)


def top_z_from_points(points) -> float:
    """World-frame top of a point cloud: the maximum ``z`` (metres)."""
    pts = _points3(points)
    if pts.shape[0] == 0:
        raise ValueError("cannot take the top of an empty point cloud")
    finite = pts[np.all(np.isfinite(pts), axis=1)]
    if finite.shape[0] == 0:
        raise ValueError("point cloud has no finite points")
    return float(finite[:, 2].max())


# ---------------------------------------------------------------------------
# Mask / pixel helpers
# ---------------------------------------------------------------------------


def points_in_mask_uv(mask) -> np.ndarray:
    """Pixel coordinates of the ``True`` pixels of ``mask``.

    Input frame: **pixel** mask ``(H, W)``. Output frame: **pixel**, ``(N, 2)`` float64
    ``[u, v]`` in row-major order (``u`` = column = ``x``, ``v`` = row = ``y``).
    """
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError(f"mask must be 2-D (H, W), got shape {arr.shape}")
    rows, cols = np.nonzero(arr.astype(bool))
    return np.stack((cols, rows), axis=1).astype(np.float64)


def mask_boundary_uv(mask) -> np.ndarray:
    """Boundary pixels (4-connected outline) of a mask, pixel frame, ``(N, 2)`` ``[u, v]``.

    Used by the silhouette/prism search (spec §9, D4): only the outline carries the
    silhouette, so casting it avoids redundant rays.
    """
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError(f"mask must be 2-D (H, W), got shape {arr.shape}")
    arr = arr.astype(bool)
    if not arr.any():
        return np.zeros((0, 2), dtype=np.float64)
    interior = np.zeros_like(arr)
    if arr.shape[0] > 2 and arr.shape[1] > 2:
        interior[1:-1, 1:-1] = (
            arr[1:-1, 1:-1]
            & arr[:-2, 1:-1]
            & arr[2:, 1:-1]
            & arr[1:-1, :-2]
            & arr[1:-1, 2:]
        )
    return points_in_mask_uv(arr & ~interior)


# ---------------------------------------------------------------------------
# 2-D convex hull / polygon rasterisation (used by the prism search)
# ---------------------------------------------------------------------------


def convex_hull_2d(points_xy) -> np.ndarray:
    """Convex hull of 2-D points, counter-clockwise in the coordinates' numeric frame.

    Input/output: ``(N, 2)`` numeric coordinates (pixel or world ``xy``). Andrew's
    monotone chain, pure numpy + a Python loop over at most ``N`` points; collinear and
    duplicate points are dropped. Fewer than 3 distinct points -> they are returned as-is.
    """
    pts = np.asarray(points_xy, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points_xy must be (N, 2), got shape {pts.shape}")
    if pts.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64)
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    p = pts[order]
    keep = np.ones(p.shape[0], dtype=bool)
    if p.shape[0] > 1:
        keep[1:] = np.any(p[1:] != p[:-1], axis=1)
    p = p[keep]
    if p.shape[0] < 3:
        return p.copy()

    def cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    lower: list[np.ndarray] = []
    for pt in p:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], pt) <= 0.0:
            lower.pop()
        lower.append(pt)
    upper: list[np.ndarray] = []
    for pt in p[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], pt) <= 0.0:
            upper.pop()
        upper.append(pt)
    hull = lower[:-1] + upper[:-1]
    return np.asarray(hull, dtype=np.float64)


def polygon_mask(vertices_uv, shape) -> np.ndarray:
    """Rasterise a convex polygon into a bool mask (pixel frame).

    Input: ``vertices_uv`` ``(N, 2)`` pixels in convex position (as produced by
    :func:`convex_hull_2d`); ``shape`` is ``(H, W)``. Output: ``(H, W)`` bool, ``True`` for
    pixel coordinates inside the polygon (edges inclusive).

    Scan-line fill: for every image row inside the polygon's bounding box the horizontal
    span is ``[min, max]`` of the row crossings of the non-horizontal edges (a convex
    polygon crosses a row in exactly one segment). Rasterisation is restricted to that
    bounding box, so the prism search's many hypotheses stay cheap (spec §17).
    """
    height, width = int(shape[0]), int(shape[1])
    out = np.zeros((height, width), dtype=bool)
    verts = np.asarray(vertices_uv, dtype=np.float64)
    if verts.ndim != 2 or verts.shape[1] != 2:
        raise ValueError(f"vertices_uv must be (N, 2), got shape {verts.shape}")
    if verts.shape[0] < 3:
        return out
    if not np.all(np.isfinite(verts)):
        verts = verts[np.all(np.isfinite(verts), axis=1)]
        if verts.shape[0] < 3:
            return out
    u0 = max(0, int(np.floor(verts[:, 0].min())))
    u1 = min(width - 1, int(np.ceil(verts[:, 0].max())))
    v0 = max(0, int(np.floor(verts[:, 1].min())))
    v1 = min(height - 1, int(np.ceil(verts[:, 1].max())))
    if u1 < u0 or v1 < v0:
        return out
    cols = np.arange(u0, u1 + 1, dtype=np.float64)
    rows = np.arange(v0, v1 + 1, dtype=np.float64)
    x_min = np.full(rows.shape, np.inf, dtype=np.float64)
    x_max = np.full(rows.shape, -np.inf, dtype=np.float64)
    nxt = np.roll(verts, -1, axis=0)
    for (x1, y1), (x2, y2) in zip(verts, nxt, strict=True):
        if y1 == y2:
            continue  # horizontal edges do not define a row span
        low, high = (y1, y2) if y1 < y2 else (y2, y1)
        selected = (rows >= low) & (rows <= high)
        if not selected.any():
            continue
        crossings = x1 + (rows[selected] - y1) * (x2 - x1) / (y2 - y1)
        x_min[selected] = np.minimum(x_min[selected], crossings)
        x_max[selected] = np.maximum(x_max[selected], crossings)
    valid = np.isfinite(x_min) & np.isfinite(x_max) & (x_min <= x_max)
    inside = (
        valid[:, None]
        & (cols[None, :] >= x_min[:, None] - _EPS)
        & (cols[None, :] <= x_max[:, None] + _EPS)
    )
    out[v0 : v1 + 1, u0 : u1 + 1] = inside
    return out


def polygon_area(vertices_xy) -> float:
    """Signed area of a polygon (``> 0`` for counter-clockwise vertices in its frame).

    Input: ``(N, 2)`` planar coordinates (any frame — pixel ``uv`` or world ``xy``).
    Formula: ``A = 0.5 * sum_i (x_i * y_{i+1} - x_{i+1} * y_i)``.
    """
    verts = np.asarray(vertices_xy, dtype=np.float64)
    if verts.ndim != 2 or verts.shape[1] != 2:
        raise ValueError(f"vertices_xy must be (N, 2), got shape {verts.shape}")
    if verts.shape[0] < 3:
        return 0.0
    x, y = verts[:, 0], verts[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def polygon_centroid(vertices_xy) -> np.ndarray:
    """Area centroid of a simple polygon in ``(N, 2)`` coordinates.

    Input/output frame: whatever planar frame the vertices are in (pixel ``uv`` or world
    ``xy``), metres for world. Formula: ``C = (1/(6A)) * sum_i (v_i + v_{i+1}) * cross_i``
    with ``cross_i = x_i * y_{i+1} - x_{i+1} * y_i``. A degenerate (zero-area) polygon
    falls back to the mean of its vertices.
    """
    verts = np.asarray(vertices_xy, dtype=np.float64)
    if verts.ndim != 2 or verts.shape[1] != 2:
        raise ValueError(f"vertices_xy must be (N, 2), got shape {verts.shape}")
    if verts.shape[0] == 0:
        return np.full(2, np.nan)
    if verts.shape[0] < 3:
        return verts.mean(axis=0)
    nxt = np.roll(verts, -1, axis=0)
    cross = verts[:, 0] * nxt[:, 1] - nxt[:, 0] * verts[:, 1]
    area = 0.5 * float(np.sum(cross))
    if abs(area) <= _EPS:
        return verts.mean(axis=0)
    return np.sum((verts + nxt) * cross[:, None], axis=0) / (6.0 * area)


def _edge_side(px: float, py: float, ax: float, ay: float, ex: float, ey: float) -> float:
    """Signed side of ``(px, py)`` w.r.t. the directed edge ``(ax, ay) + t*(ex, ey)``."""
    return ex * (py - ay) - ey * (px - ax)


def _edge_crossing(
    p: tuple[float, float],
    q: tuple[float, float],
    dp: float,
    dq: float,
) -> tuple[float, float]:
    """Point where segment ``p-q`` crosses an edge, given both endpoints' edge sides."""
    t = dp / (dp - dq) if dp != dq else 0.0
    return (p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1]))


def convex_polygon_intersection(a_xy, b_xy) -> np.ndarray:
    """Intersect two convex polygons (Sutherland-Hodgman), ``(M, 2)`` or empty ``(0, 2)``.

    Input/output frame: the vertices' own planar frame (pixel or world), counter-clockwise
    or clockwise (orientation is normalised). Used to build the visual-hull cross-section
    of several silhouettes at one height (spec §9.2).
    """
    subject = np.asarray(a_xy, dtype=np.float64)
    clipper = np.asarray(b_xy, dtype=np.float64)
    for arr in (subject, clipper):
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"polygon vertices must be (N, 2), got {arr.shape}")
    if subject.shape[0] < 3 or clipper.shape[0] < 3:
        return np.zeros((0, 2), dtype=np.float64)
    if polygon_area(subject) < 0.0:
        subject = subject[::-1]
    if polygon_area(clipper) < 0.0:
        clipper = clipper[::-1]

    output: list[tuple[float, float]] = [(float(x), float(y)) for x, y in subject]
    for index in range(clipper.shape[0]):
        if not output:
            break
        ax, ay = (float(v) for v in clipper[index])
        bx, by = (float(v) for v in clipper[(index + 1) % clipper.shape[0]])
        edge_x, edge_y = bx - ax, by - ay

        source, output = output, []
        previous = source[-1]
        previous_side = _edge_side(previous[0], previous[1], ax, ay, edge_x, edge_y)
        for current in source:
            current_side = _edge_side(current[0], current[1], ax, ay, edge_x, edge_y)
            if current_side >= 0.0:
                if previous_side < 0.0:
                    output.append(
                        _edge_crossing(previous, current, previous_side, current_side)
                    )
                output.append(current)
            elif previous_side >= 0.0:
                output.append(_edge_crossing(previous, current, previous_side, current_side))
            previous, previous_side = current, current_side
    if len(output) < 3:
        return np.zeros((0, 2), dtype=np.float64)
    return np.asarray(output, dtype=np.float64)


# ---------------------------------------------------------------------------
# Camera-frame conveniences used by the fusion module
# ---------------------------------------------------------------------------


def world_to_camera(points_w, T_world_cam) -> np.ndarray:
    """Transform world points into the camera frame.

    Input frame: **world** ``(N, 3)`` metres. Output frame: **camera** (OpenCV), ``(N, 3)``.
    Formula: ``p_cam = inv(T_world_cam) @ [x, y, z, 1]^T``.
    """
    pts = _points3(points_w, name="points_w")
    t = _transform(T_world_cam)
    inverse = np.linalg.inv(t)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    homogeneous = np.concatenate((pts, ones), axis=1)
    return (inverse @ homogeneous.T).T[:, :3]


def rays_from_pixels(uv, K, T_world_cam) -> tuple[np.ndarray, np.ndarray]:
    """World-space rays through pixels (from the camera's optical centre).

    Input frame: **pixel** ``uv`` ``(N, 2)``. Output frame: **world** —
    ``origin`` ``(3,)`` (camera centre) and unit ``dirs`` ``(N, 3)``.
    Formula: ``d_cam = normalize(K^-1 @ [u, v, 1]^T)``,
    ``d_world = R_world_cam @ d_cam``, ``origin = t_world_cam``.
    """
    pix = _pixels(uv)
    k_inv = np.linalg.inv(_intrinsics(K))
    t = _transform(T_world_cam)
    homogeneous = np.concatenate((pix, np.ones((pix.shape[0], 1))), axis=1)
    cam = homogeneous @ k_inv.T
    norms = np.linalg.norm(cam, axis=1, keepdims=True)
    cam = cam / np.where(norms > 0.0, norms, 1.0)
    dirs = cam @ t[:3, :3].T
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs = dirs / np.where(norms > 0.0, norms, 1.0)
    return t[:3, 3].copy(), dirs
