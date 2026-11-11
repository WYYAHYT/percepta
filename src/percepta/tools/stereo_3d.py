"""``stereo_3d``: depth from two calibrated views, through the existing Tool contract.

Where this fits
---------------
percepta exposes perception *tools*. The policy looks at the images it has and picks one,
passing parameters:

    policy --picks tool, passes 2 images + calibration--> stereo_3d --> Geometry
           <-----------------------------------------------------------

Nothing here decides which cameras to use or when. ``ctx.images`` and ``ctx.calibration`` hold
exactly what the caller supplied, and this tool consumes them. Two views in, metric depth and a
``Geometry`` out.

It does not pick a pair behind the caller's back, does not fall back to another method, and does
not invent a result when the pair cannot work: with no usable input it returns ``[]`` -- the
convention a geometry-kind tool uses for "no geometry" -- after logging a specific failure code.

Two backends, one contract
--------------------------
``stereo_backend="sgbm"`` uses OpenCV's semi-global matcher: no weights, no GPU, runs anywhere,
and is the honest baseline for "can stereo help at all here". ``"ffs"`` is the seam for a neural
matcher (Fast-FoundationStereo) and reports ``backend_not_configured`` until weights are wired,
rather than silently doing something else.

What is reported rather than assumed
------------------------------------
Disparity in the rectified frame maps to depth as ``Z = f * B / d``. ``f`` is the *rectified*
focal, not either camera's own ``fx``, and ``B`` is the baseline norm in the first camera's
frame. Both are derived from the supplied calibrations and echoed in ``extras``, so the
stereo-ness of a result can be checked from the output alone.
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

TOOL_NAME = "stereo_3d"
TOOL_VERSION = "0.1.0"

#: Failure codes, named so a caller can branch on them instead of parsing message strings --
#: the plan (section 4.5) asks for exactly this.
ERR_NEEDS_TWO_VIEWS = "stereo_needs_two_calibrated_views"
ERR_NO_OVERLAP = "no_overlap"
ERR_LOW_VALID_FRACTION = "low_valid_fraction"
ERR_BACKEND_NOT_CONFIGURED = "backend_not_configured"
ERR_INVALID_CALIBRATION = "invalid_calibration"


def _sgbm_disparity(
    left: np.ndarray,
    right: np.ndarray,
    *,
    num_disparities: int,
    block_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Disparity of ``right`` relative to ``left`` plus a validity mask, in pixels.

    OpenCV reports invalid matches inconsistently across modes; both the negative and the zero
    conventions collapse into one rule here: depth is meaningless wherever ``valid`` is False.
    """
    import cv2

    grey_l = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY) if left.ndim == 3 else left
    grey_r = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY) if right.ndim == 3 else right
    matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=int(num_disparities),
        blockSize=int(block_size),
        P1=8 * block_size * block_size,
        P2=32 * block_size * block_size,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        disp12MaxDiff=1,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    raw = matcher.compute(grey_l, grey_r)
    disparity = raw.astype(np.float32) / 16.0  # the fixed-point path scales by 16
    return disparity, disparity > 0.0


def _rectify(
    cal_a: Calibration, cal_b: Calibration, image_shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Rectified intrinsics, the rotation for camera A, and the derived baseline.

    ``Calibration`` carries no distortion coefficients -- RoboDojo's cameras are simulated
    pinholes -- so this rectifies orientation only. The returned rotation takes points in A's
    camera frame into the rectified frame; its transpose recovers A's frame for unprojection.
    """
    import cv2

    Ta = np.asarray(cal_a.T_world_cam, dtype=np.float64)
    Tb = np.asarray(cal_b.T_world_cam, dtype=np.float64)
    Ka = np.asarray(cal_a.K, dtype=np.float64)
    Kb = np.asarray(cal_b.K, dtype=np.float64)

    R_ab = Tb[:3, :3].T @ Ta[:3, :3]
    t_ab = Ta[:3, :3].T @ (Tb[:3, 3] - Ta[:3, 3])
    height, width = image_shape

    R1, _R2, P1, _P2, _Q, _roi1, _roi2 = cv2.stereoRectify(
        Ka,
        np.zeros(5),
        Kb,
        np.zeros(5),
        (int(width), int(height)),
        R_ab,
        t_ab.reshape(3, 1),
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0,
    )
    K_rect = np.asarray(P1[:3, :3], dtype=np.float64)
    info = {
        "focal_rect_px": float(P1[0, 0]),
        "baseline_m": float(np.linalg.norm(t_ab)),
        "R_a_to_rect": R1,
        "original_fx": {"a": float(Ka[0, 0]), "b": float(Kb[0, 0])},
    }
    return K_rect, R1, info


class Stereo3DTool(BaseTool):
    """Two calibrated views -> metric depth -> ``Geometry``.

    Parameters the policy may pass through ``context``:

    * ``stereo_left`` / ``stereo_right`` -- camera names. Required when more than two calibrated
      views are supplied; with exactly two the tool uses the only pair that exists. With three
      it refuses to guess, because picking a pair would silently discard the caller's choice.
    * ``stereo_backend`` -- ``"sgbm"`` (default) or ``"ffs"``.
    * ``stereo_num_disparities`` -- search range, multiple of 16. It must cover ``f*B/Z`` for the
      nearest distance of interest: about 400 px at 0.6 m on RoboDojo's rig.
    * ``stereo_block_size`` -- matcher window; forced odd.
    * ``stereo_mask`` -- optional ``{camera: bool array}`` limiting which pixels count, so a
      caller that already segmented the target gets object-local depth.
    * ``stereo_sigma_d_px`` -- assumed disparity noise, default 1.0.
    * ``min_valid_fraction`` -- below this the tool reports ``low_valid_fraction`` rather than a
      confident-looking geometry built on almost no points.
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
        import cv2  # noqa: F401  the SGBM backend needs OpenCV present

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
        if len(calibrated) < 2:
            self._log(
                ctx,
                f"{ERR_NEEDS_TWO_VIEWS}: got {len(calibrated)} calibrated view(s); "
                f"images={sorted(ctx.images)}, calibrated={sorted(ctx.calibration or {})}",
                level="warning",
            )
            return []

        left_name, right_name = self._resolve_pair(ctx, calibrated, context)
        if left_name is None:
            self._log(
                ctx,
                f"{ERR_NEEDS_TWO_VIEWS}: no pair selected and no unambiguous default; "
                "pass stereo_left/stereo_right",
                level="warning",
            )
            return []

        backend = str(context.get("stereo_backend", "sgbm")).lower()
        if backend != "sgbm":
            self._log(
                ctx,
                f"{ERR_BACKEND_NOT_CONFIGURED}: backend {backend!r} is not wired yet; "
                "'sgbm' is available",
                level="warning",
            )
            return []

        cal_l, cal_r = calibrated[left_name], calibrated[right_name]
        image_l = np.asarray(ctx.images[left_name])
        image_r = np.asarray(ctx.images[right_name])
        if image_l.shape[:2] != image_r.shape[:2]:
            self._log(
                ctx,
                f"{ERR_NO_OVERLAP}: image sizes differ, {left_name} {image_l.shape[:2]} vs "
                f"{right_name} {image_r.shape[:2]}",
                level="warning",
            )
            return []

        num_disparities = int(context.get("stereo_num_disparities", 640))
        if num_disparities <= 0 or num_disparities % 16 != 0:
            num_disparities = max(16, (num_disparities // 16) * 16)
        # A search range wider than the image is meaningless and makes OpenCV allocate a volume
        # far larger than memory (it asks for width-derived buffers of that width). Clamp it and
        # say so: silently shrinking a caller's parameter would hide a real mismatch between the
        # request and the images.
        # OpenCV needs the window to end strictly inside the image: equal to the width asserts.
        width_limit = 16 * (int(image_l.shape[1]) // 16) - 16
        if num_disparities > width_limit:
            self._log(
                ctx,
                f"stereo_num_disparities {num_disparities} exceeds the image width "
                f"{image_l.shape[1]}; clamping to {width_limit}",
                level="warning",
            )
            num_disparities = max(16, width_limit)
        block_size = int(context.get("stereo_block_size", 5))
        if block_size % 2 == 0:
            block_size += 1

        try:
            K_rect, _R_a, info = _rectify(cal_l, cal_r, image_l.shape[:2])
        except Exception as exc:
            self._log(ctx, f"{ERR_INVALID_CALIBRATION}: {exc!r}", level="warning")
            return []

        focal = info["focal_rect_px"]
        baseline = info["baseline_m"]
        if not np.isfinite(focal) or not np.isfinite(baseline) or focal <= 0 or baseline <= 0:
            self._log(
                ctx,
                f"{ERR_INVALID_CALIBRATION}: degenerate pair focal={focal} baseline={baseline}",
                level="warning",
            )
            return []

        need_06 = focal * baseline / 0.6
        if num_disparities < need_06:
            self._log(
                ctx,
                f"stereo search range {num_disparities} px is below the ~{need_06:.0f} px this "
                "pair needs at 0.6 m; near objects will not match",
                level="warning",
            )

        disparity, valid = _sgbm_disparity(
            image_l, image_r, num_disparities=num_disparities, block_size=block_size
        )

        mask = (context.get("stereo_mask") or {}).get(left_name)
        if mask is not None:
            mask_arr = np.asarray(mask, dtype=bool)
            if mask_arr.shape != valid.shape:
                self._log(
                    ctx,
                    f"stereo_mask shape {mask_arr.shape} != image {valid.shape}; ignoring",
                    level="warning",
                )
            else:
                valid = valid & mask_arr

        valid_fraction = float(valid.mean())
        min_valid = float(context.get("min_valid_fraction", 0.01))
        if valid_fraction < min_valid:
            self._log(
                ctx,
                f"{ERR_LOW_VALID_FRACTION}: only {valid_fraction * 100:.2f}% of pixels matched "
                f"(min_valid_fraction={min_valid})",
                level="warning",
            )
            return []

        depth = np.full(disparity.shape, np.nan, dtype=np.float64)
        np.divide(focal * baseline, disparity, out=depth, where=valid)

        rows, cols = np.nonzero(valid)
        dist = depth[rows, cols]
        keep = np.isfinite(dist) & (dist > 0.0)
        rows, cols, dist = rows[keep], cols[keep], dist[keep]
        if rows.size == 0:
            self._log(ctx, f"{ERR_LOW_VALID_FRACTION}: no finite depth values", level="warning")
            return []

        # Unproject in the rectified frame, rotate back to camera A, lift to the world.
        K_rect_inv = np.linalg.inv(K_rect)
        rays = np.stack(
            [cols.astype(np.float64), rows.astype(np.float64), np.ones(rows.size)], axis=1
        )
        rect_pts = (K_rect_inv @ rays.T).T * dist[:, None]
        a_pts = (info["R_a_to_rect"].T @ rect_pts.T).T
        T_world_a = np.asarray(cal_l.T_world_cam, dtype=np.float64)
        world = (T_world_a[:3, :3] @ a_pts.T).T + T_world_a[:3, 3]

        bbox_min, bbox_max = geo.percentile_bbox3d(world)
        centroid = [
            0.5 * (bbox_min[0] + bbox_max[0]),
            0.5 * (bbox_min[1] + bbox_max[1]),
            0.5 * (bbox_min[2] + bbox_max[2]),
        ]
        top_z = float(geo.top_z_from_points(world))
        yaw = float(geo.yaw_from_points(world[:, :2]))

        # A matched pixel's depth error is sigma_d * Z^2 / (f B). sigma_d is an assumption for
        # SGBM without a calibration set; it is stated and used consistently rather than dressed
        # up as a measured quantity.
        sigma_d = float(context.get("stereo_sigma_d_px", 1.0))
        sigma_z = dist * dist / (focal * baseline) * sigma_d
        height_unc = float(np.median(sigma_z))

        geometry = Geometry(
            frame="world",
            centroid=[float(v) for v in centroid],
            bbox3d_min=[float(v) for v in bbox_min],
            bbox3d_max=[float(v) for v in bbox_max],
            top_z=top_z,
            yaw_rad=yaw,
            uncertainty_m=height_unc,
            views_used=2,
            lateral_uncertainty_m=float(np.median(sigma_z) * 0.5),
            height_uncertainty_m=height_unc,
            view_disagreement_m=None,
            method="stereo_sgbm",
        )

        extras: dict[str, Any] = {
            "method": "stereo",
            "backend": "sgbm",
            "left": left_name,
            "right": right_name,
            "focal_rect_px": float(focal),
            "baseline_m": float(baseline),
            "num_disparities": num_disparities,
            "block_size": block_size,
            "sigma_d_px": sigma_d,
            "valid_fraction": valid_fraction,
            "points": int(rows.size),
            "median_depth_m": float(np.median(dist)),
            "median_sigma_z_m": float(np.median(sigma_z)),
            "disparity_p02": float(np.percentile(disparity[valid], 2)),
            "disparity_p98": float(np.percentile(disparity[valid], 98)),
            "warnings": [],
        }

        self._log(
            ctx,
            f"stereo_3d {left_name}->{right_name}: {rows.size} pts, {valid_fraction * 100:.1f}% valid, depth {float(np.median(dist)):.3f} m, "
            f"sigma_z {float(np.median(sigma_z)) * 1000.0:.1f} mm, {(time.perf_counter() - started) * 1000.0:.1f} ms",
            level="debug",
        )

        provenance = self._provenance(
            ctx,
            started,
            camera=left_name,
            images={left_name: image_l, right_name: image_r},
            calibration_id=getattr(cal_l, "id", None),
        )
        return [
            make_percept(
                percept_id=f"stereo-{left_name}-{right_name}",
                status="ok",
                camera=left_name,
                provenance=provenance,
                geometry=geometry,
                extras=extras,
            )
        ]

    # -- helpers -------------------------------------------------------------------

    def _resolve_pair(
        self, ctx: ToolContext, calibrated: dict[str, Calibration], context: dict[str, Any]
    ) -> tuple[str | None, str | None]:
        """Honour an explicit choice; use the only pair when exactly one exists.

        With three calibrated views there is no defensible default. Which pair to use is the
        caller's decision, and picking one here would silently discard the other camera, so the
        tool names the available pairs in the log and returns nothing.
        """
        left = context.get("stereo_left")
        right = context.get("stereo_right")
        if left is not None and right is not None:
            if left not in calibrated or right not in calibrated:
                self._log(
                    ctx,
                    f"stereo_left/stereo_right {left!r}/{right!r} are not both calibrated "
                    "and imaged",
                    level="warning",
                )
                return None, None
            return str(left), str(right)

        if len(calibrated) == 2:
            names = sorted(calibrated)
            return names[0], names[1]

        self._log(
            ctx,
            f"{len(calibrated)} calibrated views but no stereo_left/stereo_right: refusing to "
            "guess which pair the caller meant (available: "
            f"{[f'{a}-{b}' for a, b in combinations(sorted(calibrated), 2)]})",
            level="warning",
        )
        return None, None

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


__all__ = ["Stereo3DTool", "TOOL_NAME", "TOOL_VERSION"]
