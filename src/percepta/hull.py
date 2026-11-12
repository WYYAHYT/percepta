"""Silhouette-restricted visual hull on a ground-plane grid (P0).

Why this module exists
----------------------
The prism path used to take its cross-section as the *cone slice* at mid-height: where each
camera's silhouette rays cross the plane ``z = mid_z``. That polygon is the object's **shadow on
that plane**, not its footprint, and a shadow is larger than the object whenever the camera looks
down at all. Intersecting several shadows cancels part of the exaggeration — they disagree in
different directions — but for a two-view rig the result was still 2.6–2.9× the true area.

The prism is defined differently, and exactly: a plane cell belongs to the object **iff its whole
vertical column** ``[base_z, base_z + h]`` projects inside every mask. Sampling a cell at several
heights and requiring all of them to be inside is the discrete form of that test, and for a box it
is exact — the sides of the column are the box's sides.

Why a grid rather than polygons: the true footprint of the visual hull is not a convex polygon in
general (three views give a curved, faceted outline), so any convex-polygon representation loses
exactly the information that matters. A grid also lets the same code answer "what does each view
allow" for heights, which is what the height search needs.
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass

import numpy as np

__all__ = [
    "PlaneGrid",
    "hull_footprint",
    "column_inside",
    "footprint_polygon",
    "DEFAULT_GRID_CELLS",
]

#: Upper bound on grid cells, so a large silhouette cannot blow up memory. A 10 cm object gets
#: ~2 mm cells at this budget; the accuracy limit is the mask edge, not the grid.
DEFAULT_GRID_CELLS = 60_000
#: Cells narrower than this are not worth the memory (the mask edge is coarser anyway).
MIN_CELL_M = 0.0005
#: A column is tested at these fractions of its height, **strictly inside** the ends.
#:
#: Testing the ends themselves is wrong and was the bug behind a total failure on tall objects:
#: a column whose base sits on the table and whose top is the object's top surface has both ends
#: exactly on the silhouette boundary, where a pixel test reports "outside" — the mask covers the
#: object's interior, and the base/top edges are where the interior stops. Every column then
#: failed at every height, and the answer degenerated to the smallest candidate.
#:
#: Sampling interior fractions avoids the boundary entirely. Three levels still catch a silhouette
#: that is inside at the bottom and top but outside in between (a waist).
LEVEL_FRACTIONS = (0.15, 0.5, 0.85)


@dataclass(frozen=True)
class PlaneGrid:
    """Cells of a horizontal plane, addressed by integer index."""

    lo: np.ndarray  # (2,) minimum x, y of cell centres
    cell: float
    nx: int
    ny: int

    @classmethod
    def around(cls, lo_xy, hi_xy, cell_m: float) -> PlaneGrid:
        lo = np.asarray(lo_xy, dtype=np.float64).reshape(2)
        hi = np.asarray(hi_xy, dtype=np.float64).reshape(2)
        span = np.maximum(hi - lo, 1e-6)
        cell = max(float(cell_m), MIN_CELL_M)
        nx = int(math.ceil(float(span[0]) / cell)) + 1
        ny = int(math.ceil(float(span[1]) / cell)) + 1
        return cls(lo=lo, cell=cell, nx=max(1, nx), ny=max(1, ny))

    @classmethod
    def fit(cls, lo_xy, hi_xy, max_cells: int = DEFAULT_GRID_CELLS) -> PlaneGrid:
        """Cover world-XY bounds with cell centres while respecting a cell-count budget.

        Args:
            lo_xy: World ``[x, y]`` metres of the first cell centre.
            hi_xy: Upper world ``[x, y]`` bounds, covered by the last cell centres.
            max_cells: Integer budget of at least four, allowing two centres per axis.

        Returns:
            A grid with ``nx * ny <= max_cells`` and ``cell >= MIN_CELL_M``. Cell
            centres follow ``xy = lo + [column, row] * cell`` in row-major order.

        Raises:
            ValueError: The budget is not a representable integer of at least four, or
                bounds are nonfinite, reversed or have an unrepresentable span.

        Example:
            >>> grid = PlaneGrid.fit([0, 0], [1, 1], max_cells=4)
            >>> (grid.nx, grid.ny, grid.cell)
            (2, 2, 1.0)
        """
        try:
            budget = operator.index(max_cells)
            float(budget)
        except (TypeError, OverflowError) as exc:
            raise ValueError("max_cells must be an integer of at least 4") from exc
        if isinstance(max_cells, (bool, np.bool_)) or budget < 4:
            raise ValueError("max_cells must be an integer of at least 4")
        lo = np.asarray(lo_xy, dtype=np.float64).reshape(2)
        hi = np.asarray(hi_xy, dtype=np.float64).reshape(2)
        with np.errstate(over="ignore", invalid="ignore"):
            span = hi - lo
        if not np.isfinite(lo).all() or not np.isfinite(hi).all() or not np.isfinite(span).all() or np.any(span < 0):
            raise ValueError("grid bounds must be finite, ordered and have a finite span")
        span = np.maximum(span, 1e-6)
        area = float(span[0]) * float(span[1])
        estimate = math.sqrt(area / budget) if math.isfinite(area) else math.sqrt(float(span[0]) / budget) * math.sqrt(float(span[1]))
        cell = max(MIN_CELL_M, estimate)
        scale = float(span.max())
        grid = cls.around(lo, hi, cell) if math.isfinite(scale / cell) else None
        if grid is None or grid.size > budget:
            if budget == 4:
                cell = max(cell, scale)
            else:
                x, y = span / scale
                # ceil(span/cell)+1 <= span/cell+2; solve the resulting cell-count bound.
                ratio = (x + y + math.sqrt((x + y) ** 2 + (budget - 4) * x * y)) / (budget - 4)
                cell = math.nextafter(max(cell, scale * ratio), math.inf)
            grid = cls.around(lo, hi, cell)
        if np.any(lo + np.array([grid.nx - 1, grid.ny - 1]) * grid.cell < hi):
            grid = cls(lo=lo, cell=math.nextafter(grid.cell, math.inf), nx=grid.nx, ny=grid.ny)
        return grid

    @property
    def size(self) -> int:
        return self.nx * self.ny

    def xy(self) -> np.ndarray:
        """``(N, 2)`` world xy of every cell centre, row-major."""
        xs = self.lo[0] + np.arange(self.nx) * self.cell
        ys = self.lo[1] + np.arange(self.ny) * self.cell
        gx, gy = np.meshgrid(xs, ys, indexing="xy")
        return np.stack([gx.ravel(), gy.ravel()], axis=1)

    def points(self, cells: np.ndarray, z: float) -> np.ndarray:
        """Return world points for selected row-major cell indices at a fixed height.

        Args:
            cells: One-dimensional integer indices or a boolean selection over the grid.
                Integers retain NumPy order, duplicates and negative-index semantics.
            z: World height in metres.

        Returns:
            ``(N, 3)`` world XYZ metres; ``xy = lo + [column, row] * cell``.

        Example:
            >>> grid = PlaneGrid(np.array([0., 0.]), 0.5, 2, 2)
            >>> grid.points(np.array([3, 0, 3]), 2.).tolist()
            [[0.5, 0.5, 2.0], [0.0, 0.0, 2.0], [0.5, 0.5, 2.0]]
        """
        xy = self.xy()[cells]
        return np.column_stack([xy, np.full(len(xy), float(z))])

    def at_height(self, z: float) -> np.ndarray:
        return np.column_stack([self.xy(), np.full(self.size, float(z))])

    def cell_area(self) -> float:
        return self.cell * self.cell


def _inside_mask(view, points_w: np.ndarray) -> np.ndarray:
    """Which world points project inside ``view``'s mask (and in front of it)."""
    from . import geometry as geo

    camera_frame = geo.world_to_camera(points_w, view.T)
    depth = camera_frame[:, 2]
    uv = geo.project_points(camera_frame, view.K)
    height, width = view.mask.shape
    ok = np.isfinite(uv).all(axis=1) & (depth > 1e-9)
    cols = np.where(ok, np.rint(uv[:, 0]), -1).astype(np.int64)
    rows = np.where(ok, np.rint(uv[:, 1]), -1).astype(np.int64)
    ok &= (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
    out = np.zeros(len(points_w), dtype=bool)
    if ok.any():
        out[ok] = view.mask[rows[ok], cols[ok]]
    return out


def column_inside(
    views: list, grid: PlaneGrid, base_z: float, top_z: float
) -> np.ndarray:
    """Cells whose **entire** vertical column is inside every view's mask.

    This is the definition of a prism silhouette: the object occupies the cell from the base to
    ``top_z``, so the column must not leave any mask anywhere along its height. Testing only the
    base (the old shadow) admits cells the object never occupies; testing only the top admits the
    shadow at that depth instead.
    """
    if top_z <= base_z:
        return np.zeros(grid.size, dtype=bool)
    alive = np.ones(grid.size, dtype=bool)
    for fraction in LEVEL_FRACTIONS:
        if not alive.any():
            break
        height = base_z + (top_z - base_z) * float(fraction)
        pts = grid.at_height(height)[alive]
        keep = _inside_mask_any(views, pts)
        index = np.flatnonzero(alive)
        alive[index[~keep]] = False
    return alive


def _inside_mask_any(views: list, points_w: np.ndarray) -> np.ndarray:
    """Points inside **every** view's mask."""
    inside = np.ones(len(points_w), dtype=bool)
    for view in views:
        if not inside.any():
            break
        inside &= _inside_mask(view, points_w)
    return inside


def hull_footprint(
    views: list,
    grid: PlaneGrid,
    base_z: float,
    top_z: float,
) -> np.ndarray:
    """``(N, 2)`` world xy of the cells forming the prism footprint."""
    cells = column_inside(views, grid, base_z, top_z)
    if not cells.any():
        return np.zeros((0, 2))
    return grid.xy()[cells]


def footprint_polygon(points_xy: np.ndarray) -> np.ndarray:
    """Convex hull of a footprint, for callers that need a polygon (bbox, yaw, IoU).

    NOTE: the hull is *not* the footprint — a three-view visual hull is generally non-convex.
    Use it only for quantities that are hull-invariant or where an upper bound is acceptable;
    ``footprint_metrics`` reports the true area separately.
    """
    from . import geometry as geo

    if len(points_xy) < 3:
        return np.zeros((0, 2))
    return geo.convex_hull_2d(np.asarray(points_xy, dtype=np.float64))


@dataclass(frozen=True)
class FootprintMetrics:
    """Quantities a caller needs from a footprint, kept honest about what is an upper bound."""

    area_m2: float  # sum of cell areas: the visual hull's true footprint area
    centroid_xy: np.ndarray  # area-weighted centre of the cells
    bbox_min: np.ndarray  # tight axis-aligned bounds of the cell set
    bbox_max: np.ndarray
    yaw_rad: float
    cells: np.ndarray  # bool mask over the grid
    cell_count: int


def footprint_metrics(grid: PlaneGrid, cells: np.ndarray) -> FootprintMetrics | None:
    """Area, centroid, tight bounds and principal axis of a cell set."""
    from . import geometry as geo

    count = int(np.count_nonzero(cells))
    if count == 0:
        return None
    xy = grid.xy()[cells]
    area = count * grid.cell_area()
    centroid = xy.mean(axis=0)
    lo = xy.min(axis=0) - 0.5 * grid.cell
    hi = xy.max(axis=0) + 0.5 * grid.cell
    return FootprintMetrics(
        area_m2=float(area),
        centroid_xy=centroid,
        bbox_min=lo,
        bbox_max=hi,
        yaw_rad=float(geo.yaw_from_points(xy)),
        cells=cells,
        cell_count=count,
    )
