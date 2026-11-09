"""Baseline vs improved: footprint accuracy of the prism path (P0).

The defect this measures: the old cross-section was the *cone slice* at mid-height — the set of
points where each camera's silhouette rays meet the plane z=mid_z. For an oblique camera that
polygon is the object's **shadow**, not its footprint: it is larger than the object by
``h/2 * (tan(ray) - tan(base))``. Intersecting several shadows cancels part of the exaggeration
(they disagree in different directions) but never all of it, and for a two-view rig the
projection is severe.

The fix builds the footprint the way a prism actually is: a plane cell belongs to the object iff
its whole vertical column [table_z, table_z + h] projects inside every mask. That is the
silhouette-restricted visual hull, and for a box it is exact.

Run:  python scripts/footprint_accuracy.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from percepta.fusion import fuse  # noqa: E402
from tests.synth_geom import camera_rig, cube_masks, mask_percepts  # noqa: E402

RIGS = [(0, 120, 240), (0, 180), (0, 90, 180, 270), (0, 60, 120)]
HEIGHTS = [0.05, 0.10, 0.15]
HALF = 0.05


def run(azimuths, top_height, elevation=20.0):
    centre = np.array([0.0, 0.0, top_height / 2.0])
    cals = camera_rig(azimuths, elevation_deg=elevation, target=(0, 0, 0.05))
    masks = cube_masks(cals, centre, HALF)
    percepts = mask_percepts(cals, masks)
    fused = fuse(percepts, cals, {"table_z": 0.0})
    if fused is None or fused.geometry is None:
        return None
    geometry = fused.geometry
    centroid_error = float(np.linalg.norm(np.array(geometry.centroid) - centre))
    height_error = abs(float(geometry.top_z or 0.0) - top_height)
    # footprint area vs the true square
    if geometry.bbox3d_min and geometry.bbox3d_max:
        area = (geometry.bbox3d_max[0] - geometry.bbox3d_min[0]) * (
            geometry.bbox3d_max[1] - geometry.bbox3d_min[1]
        )
    else:
        area = float("nan")
    return {
        "centroid_cm": centroid_error * 100,
        "height_cm": height_error * 100,
        "footprint_cm2": area * 1e4,
        "views": geometry.views_used,
        "method": fused.extras.get("method"),
    }


def main() -> int:
    true_area = (2 * HALF) ** 2 * 1e4
    print(f"true footprint {true_area:.1f} cm^2, cube side {2 * HALF * 100:.0f} cm")
    print(f"{'rig':<22}{'top':>6}{'centroid':>10}{'height':>9}{'footprint':>11}{'ratio':>8}")
    for azimuths in RIGS:
        for top_height in HEIGHTS:
            result = run(azimuths, top_height)
            label = "/".join(str(a) for a in azimuths)
            if result is None:
                print(f"{label:<22}{top_height:>6.2f}{'  none':>10}")
                continue
            ratio = result["footprint_cm2"] / true_area if true_area else float("nan")
            print(
                f"{label:<22}{top_height:>6.2f}"
                f"{result['centroid_cm']:>9.2f}cm{result['height_cm']:>8.2f}cm"
                f"{result['footprint_cm2']:>10.1f}{ratio:>8.2f}x"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
