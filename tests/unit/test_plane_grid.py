"""Grid indexing preserves cell IDs and fitting always respects its memory budget."""

from __future__ import annotations

import numpy as np
import pytest

from percepta.hull import MIN_CELL_M, PlaneGrid


@pytest.mark.parametrize("indices", [[0, 3], [3, 0, 3], [-1, 0], []])
def test_points_uses_integer_indices_in_order_including_repeats_and_negative_indices(indices):
    grid = PlaneGrid(lo=np.array([10.0, 20.0]), cell=0.5, nx=3, ny=2)
    cells = np.asarray(indices, dtype=np.int64)
    expected_xy = grid.xy()[cells]
    assert np.array_equal(grid.points(cells, 2.0), np.column_stack([expected_xy, np.full(len(cells), 2.0)]))


def test_points_boolean_selection_matches_equivalent_integer_indices():
    grid = PlaneGrid(lo=np.array([10.0, 20.0]), cell=0.5, nx=3, ny=2)
    cells = np.array([True, False, False, True, False, True])
    assert np.array_equal(grid.points(cells, 2.0), grid.points(np.flatnonzero(cells), 2.0))


@pytest.mark.parametrize(
    "span,budget",
    [([10.0, 10.0], 10_000), ([1e6, 0.001], 60_000), ([0.001, 1e6], 60_000),
     ([1.0, 1.0], 4), ([1e6, 0.001], 5), ([0.001, 1e6], 6), ([0.0001, 0.0001], 4),
     ([4378090744888843.5, 6.151405443033033e-05], 20446), ([1e200, 1e200], 60_000),
     ([1e200, 1e-6], 60_000), ([1e-6, 1e200], 60_000), ([1e308, 1e-6], 10**308)],
)
def test_fit_respects_cell_budget_and_covers_both_world_bounds(span, budget):
    lo = np.array([10.0, -20.0])
    hi = lo + span
    grid = PlaneGrid.fit(lo, hi, max_cells=budget)
    assert 4 <= grid.size <= budget
    assert grid.cell >= MIN_CELL_M
    assert np.array_equal(grid.lo, lo)
    last = grid.lo + np.array([grid.nx - 1, grid.ny - 1]) * grid.cell
    assert np.all(last >= hi), (last, hi)


@pytest.mark.parametrize(
    "lo,hi",
    [([0.0, 0.0], [float("inf"), 1.0]), ([float("nan"), 0.0], [1.0, 1.0]),
     ([1.0, 0.0], [0.0, 1.0]), ([-1e308, 0.0], [1e308, 1.0])],
)
def test_fit_rejects_nonfinite_reversed_or_overflowing_world_bounds(lo, hi):
    with pytest.raises(ValueError, match="bounds"):
        PlaneGrid.fit(lo, hi)


@pytest.mark.parametrize("budget", [0, -1, 1, 2, 3, 5.5, True, "60000", float("nan"), float("inf")])
def test_fit_rejects_budgets_that_cannot_cover_two_by_two_cell_centres(budget):
    with pytest.raises(ValueError, match="max_cells"):
        PlaneGrid.fit([0.0, 0.0], [1.0, 1.0], max_cells=budget)


def test_fit_preserves_existing_grid_when_the_minimum_cell_already_fits():
    grid = PlaneGrid.fit([0.0, 0.0], [0.01, 0.02], max_cells=np.int64(60_000))
    expected = PlaneGrid.around([0.0, 0.0], [0.01, 0.02], MIN_CELL_M)
    assert grid.cell == expected.cell
    assert (grid.nx, grid.ny) == (expected.nx, expected.ny)
