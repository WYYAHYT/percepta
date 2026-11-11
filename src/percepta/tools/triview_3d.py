"""``triview_3d``: three calibrated views, fused into one ``Geometry``.

Same contract as every other tool: the policy picks this one and passes three images plus their
calibrations. The tool consumes exactly what it is given.

    policy --picks triview_3d, 3 images + calibration--> triview_3d --> Geometry
           <------------------------------------------------------------

Two strategies, because they answer different questions
-------------------------------------------------------
``method="pairs"`` runs the stereo matcher on every pair the three cameras define and merges the
resulting points in the world frame. It is the route that needs no new model, and it is honest
about what it is: three two-view measurements combined, not one three-view inference.

``method="mvs"`` is the seam for a native multi-view network (MVSAnywhere). It reports
``backend_not_configured`` until such a backend is wired, rather than pretending the pairwise
route is the same thing.

Why merging must happen in the world frame
------------------------------------------
The three depth maps live in three different camera frames and observe different surface
patches. Averaging their pixel values would be meaningless; the points are transformed to the
world first and only then combined. Each contributing pair reports its own precision, so the
merge weights by ``1/sigma_z^2`` instead of treating a 0.5 mm point and a 6 mm point as equally
trustworthy.

Why this refuses rather than degrades
-------------------------------------
If no pair produces usable depth there is no geometry and the tool returns ``[]``. If only one
pair works it still answers -- with ``pairs_used=1`` in ``extras``, so a caller can see that the
"three-view" result was really one pair, which is exactly the kind of silent degradation the
plan (section 5) demands be visible.
"""

from __future__ import annotations

import time
from itertools import combinations
from typing import Any

import numpy as np

from .. import geometry as geo
from ..schema import (
    Calibration,
    Geometry,
    Percept,
    Prompt,
    ToolCapabilities,
    ToolContext,
)
from .base import BaseTool, make_percept
from .stereo_3d import _rectify, _sgbm_disparity

TOOL_NAME = "triview_3d"
TOOL_VERSION = "0.1.0"

ERR_NEEDS_THREE_VIEWS = "triview_needs_three_calibrated_views"
ERR_BACKEND_NOT_CONFIGURED = "backend_not_configured"
ERR_NO_USABLE_PAIR = "no_usable_pair"


def _pair_points(
    cal_a: Calibration,
    cal_b: Calibration,
    image_a: np.ndarray,
    image_b: np.ndarray,
    *,
    num_disparities: int,
    block_size: int,
    sigma_d: float,
    min_valid_fraction: float,
) -> dict[str, Any] | None:
    """World points and their per-point depth sigma for one pair, or ``None``.

    The returned ``sigma`` array is what makes the merge meaningful: it lets the caller weight a
    long-baseline pair more than a short one instead of averaging them blind.
    """
    if image_a.shape[:2] != image_b.shape[:2]:
        return None
    try:
        K_rect, _R_a, info = _rectify(cal_a, cal_b, image_a.shape[:2])
    except Exception:
        return None

    focal = info["focal_rect_px"]
    baseline = info["baseline_m"]
    if not np.isfinite(focal) or not np.isfinite(baseline) or focal <= 0 or baseline <= 0:
        return None

    # A search range cannot exceed the image width (OpenCV asserts at the boundary).
    width_limit = 16 * (int(image_a.shape[1]) // 16) - 16
    num_disparities = max(16, min(num_disparities, width_limit))

    disparity, valid = _sgbm_disparity(
        image_a, image_b, num_disparities=num_disparities, block_size=block_size
    )
    valid_fraction = float(valid.mean())
    if valid_fraction < min_valid_fraction:
        return None

    depth = np.full(disparity.shape, np.nan, dtype=np.float64)
    np.divide(focal * baseline, disparity, out=depth, where=valid)

    rows, cols = np.nonzero(valid)
    dist = depth[rows, cols]
    keep = np.isfinite(dist) & (dist > 0.0)
    rows, cols, dist = rows[keep], cols[keep], dist[keep]
    if rows.size == 0:
        return None

    K_rect_inv = np.linalg.inv(K_rect)
    rays = np.stack(
        [cols.astype(np.float64), rows.astype(np.float64), np.ones(rows.size)], axis=1
    )
    # ``dist`` is distance along the optical axis, and the camera convention (spec §4) is that
    # the optical axis is -z. Multiplying a +z ray by a positive depth would put every point
    # BEHIND the camera, which is the sign error this comment exists to prevent.
    rays[:, 2] = -1.0
    rect_pts = (K_rect_inv @ rays.T).T * dist[:, None]
    a_pts = (info["R_a_to_rect"].T @ rect_pts.T).T
    T_world_a = np.asarray(cal_a.T_world_cam, dtype=np.float64)
    world = (T_world_a[:3, :3] @ a_pts.T).T + T_world_a[:3, 3]

    sigma = dist * dist / (focal * baseline) * sigma_d
    return {
        "points": world,
        "sigma": sigma,
        "focal": focal,
        "baseline": baseline,
        "valid_fraction": valid_fraction,
    }


class Triview3DTool(BaseTool):
    """Three calibrated views -> fused world points -> ``Geometry``.

    Parameters the policy may pass through ``context``:

    * ``triview_method`` -- ``"pairs"`` (default) or ``"mvs"``.
    * ``stereo_num_disparities`` / ``stereo_block_size`` -- forwarded to the matcher.
    * ``stereo_sigma_d_px`` -- assumed disparity noise, default 1.0.
    * ``min_valid_fraction`` -- per-pair gate; a pair below it does not contribute.
    * ``triview_max_pairs`` -- use only the best N pairs (by valid fraction). Default 3, i.e. all.
    """

    name = TOOL_NAME
    version = TOOL_VERSION
    kind = "geometry"

    def capabilities(self) -> ToolCapabilities:
        return ToolCapabilities(
            kind="geometry",
            prompts=set(),
            masks=False,
            boxes=False,
            points=False,
            depth=False,
            multi_view=True,
            geometry=True,
        )

    def load(self, config: dict[str, Any] | None = None) -> None:
        import cv2  # noqa: F401  the pairwise route needs OpenCV

        super().load(config or {})

    def perceive(self, ctx: ToolContext, prompts: list[Prompt]) -> list[Percept]:
        _ = prompts
        started = time.perf_counter()
        context = dict(ctx.context or {})
        for key, value in (self.config or {}).items():
            context.setdefault(key, value)

        calibrated = {
            name: cal for name, cal in (ctx.calibration or {}).items() if name in ctx.images
        }
        if len(calibrated) != 3:
            self._log(
                ctx,
                f"{ERR_NEEDS_THREE_VIEWS}: triview_3d takes exactly 3 calibrated views, got "
                f"{len(calibrated)} (images={sorted(ctx.images)}, "
                f"calibrated={sorted(ctx.calibration or {})}); use stereo_3d for two",
                level="warning",
            )
            return []

        method = str(context.get("triview_method", "pairs")).lower()
        if method != "pairs":
            self._log(
                ctx,
                f"{ERR_BACKEND_NOT_CONFIGURED}: triview method {method!r} is not wired yet; "
                "'pairs' is available",
                level="warning",
            )
            return []

        num_disparities = int(context.get("stereo_num_disparities", 640))
        block_size = int(context.get("stereo_block_size", 5))
        if block_size % 2 == 0:
            block_size += 1
        sigma_d = float(context.get("stereo_sigma_d_px", 1.0))
        min_valid = float(context.get("min_valid_fraction", 0.01))
        max_pairs = int(context.get("triview_max_pairs", 3))

        names = sorted(calibrated)
        results: list[tuple[str, str, dict[str, Any]]] = []
        for a, b in combinations(names, 2):
            pair = _pair_points(
                calibrated[a],
                calibrated[b],
                np.asarray(ctx.images[a]),
                np.asarray(ctx.images[b]),
                num_disparities=num_disparities,
                block_size=block_size,
                sigma_d=sigma_d,
                min_valid_fraction=min_valid,
            )
            if pair is not None:
                results.append((a, b, pair))

        if not results:
            self._log(
                ctx,
                f"{ERR_NO_USABLE_PAIR}: none of the 3 pairs produced usable depth "
                f"(min_valid_fraction={min_valid})",
                level="warning",
            )
            return []

        # Keep the strongest pairs when a caller limits work; a short-baseline pair adds noise
        # more than information, so ordering by valid fraction is a reasonable proxy for value.
        results.sort(key=lambda item: -item[2]["valid_fraction"])
        results = results[: max(1, max_pairs)]

        points = np.concatenate([item[2]["points"] for item in results], axis=0)
        sigma = np.concatenate([item[2]["sigma"] for item in results], axis=0)
        if points.size == 0:
            return []

        # Inverse-variance weighting: a 0.5 mm point and a 6 mm point must not count equally.
        weights = 1.0 / np.maximum(sigma, 1e-9) ** 2
        weights /= weights.sum()
        weighted_centroid = (points * weights[:, None]).sum(axis=0)

        bbox_min, bbox_max = geo.percentile_bbox3d(points)
        top_z = float(geo.top_z_from_points(points))
        yaw = float(geo.yaw_from_points(points[:, :2]))
        median_sigma = float(np.median(sigma))

        # Report how much of the three-view claim survives: a result built on one pair is a
        # two-view answer wearing a three-view label, and the caller should be able to tell.
        pair_names = [f"{a}-{b}" for a, b, _ in results]
        degrade = len(results) < len(list(combinations(names, 2)))

        geometry = Geometry(
            frame="world",
            centroid=[float(v) for v in weighted_centroid],
            bbox3d_min=[float(v) for v in bbox_min],
            bbox3d_max=[float(v) for v in bbox_max],
            top_z=top_z,
            yaw_rad=yaw,
            uncertainty_m=median_sigma,
            views_used=len({n for a, b, _ in results for n in (a, b)}),
            lateral_uncertainty_m=median_sigma * 0.5,
            height_uncertainty_m=median_sigma,
            view_disagreement_m=None,
            method="triview_pairs",
        )

        extras: dict[str, Any] = {
            "method": "triview",
            "strategy": "pairs",
            "pairs_used": len(results),
            "pairs": pair_names,
            "degraded_to_two_view": degrade,
            "points": int(len(points)),
            "median_sigma_z_m": median_sigma,
            "sigma_d_px": sigma_d,
            "per_pair": [
                {
                    "pair": f"{a}-{b}",
                    "baseline_m": info["baseline"],
                    "focal_rect_px": info["focal"],
                    "valid_fraction": info["valid_fraction"],
                    "points": int(len(info["points"])),
                    "median_sigma_z_m": float(np.median(info["sigma"])),
                }
                for a, b, info in results
            ],
            "warnings": [],
        }
        if degrade:
            extras["warnings"].append("triview_degraded_to_subset_of_pairs")

        self._log(
            ctx,
            "triview_3d {pairs}: {n} points, median sigma {s:.2f} mm, {ms:.1f} ms"
            "{deg}".format(
                pairs=pair_names,
                n=len(points),
                s=median_sigma * 1000.0,
                ms=(time.perf_counter() - started) * 1000.0,
                deg=" [degraded]" if degrade else "",
            ),
            level="debug",
        )

        provenance = self._provenance(
            ctx,
            started,
            camera=names[0],
            images={n: np.asarray(ctx.images[n]) for n in names},
            calibration_id=getattr(calibrated[names[0]], "id", None),
        )
        return [
            make_percept(
                percept_id="triview-" + "-".join(names),
                status="ok",
                camera=names[0],
                provenance=provenance,
                geometry=geometry,
                extras=extras,
            )
        ]

    @staticmethod
    def _log(ctx: ToolContext, message: str, *, level: str = "debug") -> None:
        """Best-effort structured log; a missing or odd logger must never fail a request."""
        logger: Any = getattr(ctx, "logger", None)
        if logger is None:
            return
        try:
            getattr(logger, level, logger.debug)(message, extra={"tool": TOOL_NAME})
        except (AttributeError, TypeError, ValueError):
            pass


__all__ = ["Triview3DTool", "TOOL_NAME", "TOOL_VERSION"]
