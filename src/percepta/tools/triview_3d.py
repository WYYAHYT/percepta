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

``method="mvs"`` uses an independently deployed MVSAnywhere or DA3-BASE model. These
networks jointly consume three calibrated views; metric depth is returned on the input grid.

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
from ..backends.base import BackendRequestError, BackendUnavailable
from ..backends.multiview_backend import MultiViewBackend
from ..schema import (
    Calibration,
    Geometry,
    Percept,
    Prompt,
    ToolCapabilities,
    ToolContext,
)
from ..weights import WeightError, resolve_weights
from .base import BaseTool, ToolUnavailable, make_percept
from .stereo_3d import (
    ERR_INVALID_CALIBRATION,
    _disparity_points,
    _error_percept,
    _rectify,
    _remap_pair,
    _sgbm_disparity,
    _StereoInputError,
    _validate_calibration,
)

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
    """World visible-surface points and assumed rectified-depth sigma, or ``None``.

    Uses Z=f*B/d and p_world = T_world_a @ [R1.T @ p_rect, 1]. Malformed or
    reversed/vertical pairs raise _StereoInputError; no pair is silently swapped.
    """
    if image_a.shape[:2] != image_b.shape[:2]:
        raise _StereoInputError(ERR_INVALID_CALIBRATION, "stereo image sizes must agree")
    K_rect, _R_a, info = _rectify(cal_a, cal_b, image_a.shape[:2])
    rect_a, rect_b, _ = _remap_pair(image_a, image_b, info)
    focal = info["focal_rect_px"]
    baseline = info["baseline_m"]

    # A search range cannot exceed the image width (OpenCV asserts at the boundary).
    width_limit = 16 * (int(image_a.shape[1]) // 16) - 16
    num_disparities = max(16, min((num_disparities // 16) * 16, width_limit))

    disparity, valid = _sgbm_disparity(
        rect_a, rect_b, num_disparities=num_disparities, block_size=block_size
    )
    world, dist, valid = _disparity_points(disparity, valid, cal_a, cal_b, K_rect, info)
    valid_fraction = float(valid.mean())
    if dist.size == 0 or valid_fraction < min_valid_fraction:
        return None

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

    def __init__(self) -> None:
        super().__init__()
        self._backend: MultiViewBackend | None = None
        self._model_sha256: str | None = None
        self._neural_backend: str | None = None

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
        super().load(config or {})
        if self.config.get("triview_method", "pairs") == "pairs":
            import cv2  # noqa: F401  the pairwise route needs OpenCV

            return
        model = self.config.get("triview_backend")
        if model not in {"mvsanywhere", "da3"}:
            raise ToolUnavailable("triview mvs requires a deployed mvsanywhere or da3 backend")
        try:
            weights = resolve_weights(self.config.get("checkpoint"), config=self.config,
                                      verify=self.config.get("verify_weights", "if-declared"), what=f"{model} checkpoint")
            if weights.path is None:
                raise WeightError(f"{model} requires a local checkpoint")
            arguments = {"source_dir": self.config.get("source_dir"), "checkpoint": str(weights.path)}
            if model == "da3":
                model_config = resolve_weights(
                    self.config.get("model_config", str(weights.path.with_name("config.json"))),
                    expected_sha256=self.config.get("model_config_sha256"),
                    verify=self.config.get("verify_weights", "if-declared"), what="DA3 model config",
                )
                if model_config.path is None:
                    raise WeightError("DA3 requires a local model config")
                arguments.update(model_config=str(model_config.path), process_res=self.config.get("process_res", 504))
                factory = "percepta.backends.da3_adapter:DA3Adapter"
            else:
                arguments.update(image_height=self.config.get("image_height", 480), image_width=self.config.get("image_width", 640),
                                 num_refinement_steps=self.config.get("num_refinement_steps", 1))
                factory = "percepta.backends.mvsanywhere_adapter:MVSAnywhereAdapter"
            self._backend = MultiViewBackend(
                name=model, backend_factory=factory, backend_args=arguments,
                python=self.config.get("python", f"uv_envs/{model}/.venv/bin/python"),
                cuda_visible_devices=self.config.get("cuda_visible_devices"),
                startup_timeout_s=self.config.get("startup_timeout_s", 180), timeout_s=self.config.get("timeout_s", 180),
            )
            self._backend.start()
            self._model_sha256, self._neural_backend = weights.observed_sha256, model
        except (WeightError, BackendUnavailable, BackendRequestError, OSError, ValueError) as exc:
            self.close()
            raise ToolUnavailable(str(exc)) from exc

    def close(self) -> None:
        """Release only this tool's resident model process.

        Example:
            >>> tool = Triview3DTool()
            >>> tool.close()
        """
        if self._backend is not None:
            self._backend.close()
            self._backend = None
        self._neural_backend = None

    def perceive(self, ctx: ToolContext, prompts: list[Prompt]) -> list[Percept]:
        _ = prompts
        started = time.perf_counter()
        context = dict(ctx.context or {})
        for key, value in (self.config or {}).items():
            context.setdefault(key, value)

        calibrated = ctx.calibration
        if len(ctx.images) != 3 or set(calibrated) != set(ctx.images):
            return _error_percept(
                self,
                ctx,
                started,
                ERR_NEEDS_THREE_VIEWS,
                "triview_3d requires exactly three images and the same three calibrations",
            )
        try:
            for name, cal in calibrated.items():
                image = ctx.images[name]
                if cal.camera != name:
                    raise _StereoInputError(
                        ERR_INVALID_CALIBRATION, "calibration camera/key mismatch"
                    )
                if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
                    raise _StereoInputError(
                        ERR_INVALID_CALIBRATION, f"{name}: expected uint8 RGB image"
                    )
                _validate_calibration(cal, image.shape[:2])
        except _StereoInputError as exc:
            return _error_percept(self, ctx, started, exc.code, str(exc))

        method = str(context.get("triview_method", "pairs")).lower()
        if method == "mvs" and self._backend is not None:
            if context.get("triview_backend", self._neural_backend) != self._neural_backend:
                return _error_percept(self, ctx, started, ERR_BACKEND_NOT_CONFIGURED,
                                      "requested multiview model is not loaded in this deployment", status="unavailable")
            return self._perceive_neural(ctx, started)
        if method != "pairs":
            self._log(
                ctx,
                f"{ERR_BACKEND_NOT_CONFIGURED}: triview method {method!r} is not wired yet; "
                "'pairs' is available",
                level="warning",
            )
            return _error_percept(self, ctx, started, ERR_BACKEND_NOT_CONFIGURED,
                                  f"triview method {method!r} is not deployed", status="unavailable")

        try:
            import cv2  # noqa: F401  a neural-only deployment may not install this optional extra
        except ImportError as exc:
            raise ToolUnavailable("triview pairs requires the stereo extra (OpenCV)") from exc

        num_disparities = int(context.get("stereo_num_disparities", 640))
        block_size = int(context.get("stereo_block_size", 5))
        if block_size % 2 == 0:
            block_size += 1
        sigma_d = float(context.get("stereo_sigma_d_px", 1.0))
        min_valid = float(context.get("min_valid_fraction", 0.01))
        max_pairs = int(context.get("triview_max_pairs", 3))

        names = sorted(calibrated)
        results: list[tuple[str, str, dict[str, Any]]] = []
        pair_failures = []
        for a, b in combinations(names, 2):
            try:
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
            except _StereoInputError as exc:
                pair_failures.append({"pair": f"{a}-{b}", "code": exc.code, "message": str(exc)})
                self._log(ctx, f"{a}-{b}: {exc.code}: {exc}", level="warning")
                continue
            if pair is not None:
                results.append((a, b, pair))

        if not results and pair_failures:
            result = _error_percept(
                self,
                ctx,
                started,
                pair_failures[0]["code"],
                "no supported ordered stereo pair",
            )
            result[0].extras["pair_failures"] = pair_failures
            return result
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
            lateral_uncertainty_m=None,
            height_uncertainty_m=None,
            view_disagreement_m=None,
            method="triview_pairs",
        )

        extras: dict[str, Any] = {
            "method": "triview",
            "strategy": "pairs",
            "geometry_scope": "matched_visible_surface",
            "geometry_summary": "weighted sample centroid; bounds/top/yaw are not object shape",
            "uncertainty_model": "median Z^2*sigma_d/(f*B), assumed rectified-depth noise, not calibrated world-axis error",
            "uncertainty_excludes": [
                "calibration_error",
                "occlusion",
                "mismatches",
                "pair_correlation",
                "unseen_surfaces",
            ],
            "rectified": True,
            "pair_failures": pair_failures,
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
            "triview_3d {pairs}: {n} points, median sigma {s:.2f} mm, {ms:.1f} ms{deg}".format(
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

    def _perceive_neural(self, ctx: ToolContext, started: float) -> list[Percept]:
        """Convert metric camera +Z depth into world points with p_world=T_world_cam@p_cam_h."""
        names = sorted(ctx.images)
        images = [np.asarray(ctx.images[name]) for name in names]
        if any(image.shape != images[0].shape for image in images):
            return _error_percept(self, ctx, started, ERR_INVALID_CALIBRATION, "neural triview images must have equal shapes")
        K = np.asarray([ctx.calibration[name].K for name in names], dtype=np.float64)
        transforms = np.asarray([ctx.calibration[name].T_world_cam for name in names], dtype=np.float64)
        centres = transforms[:, :3, 3]
        if (any(np.linalg.norm(centres[a] - centres[b]) <= 1e-6 for a, b in ((0, 1), (0, 2), (1, 2)))
                or (self._neural_backend == "da3" and np.linalg.matrix_rank(centres - centres.mean(axis=0), tol=1e-6) < 2)):
            return _error_percept(self, ctx, started, ERR_INVALID_CALIBRATION,
                                  "metric multiview inference requires distinct centres; DA3 also requires non-collinear centres")
        try:
            result = self._backend.infer(np.stack(images), K, transforms)
        except BackendRequestError as exc:
            return _error_percept(self, ctx, started, ERR_INVALID_CALIBRATION, str(exc))
        except BackendUnavailable as exc:
            raise ToolUnavailable(str(exc)) from exc
        clouds, contributing = [], []
        for index, name in enumerate(names):
            depth = result.depth[index]
            valid = depth > 0
            if valid.any():
                clouds.append(geo.mask_to_world_points(valid, depth, K[index], transforms[index]))
                contributing.append(name)
        if not clouds:
            return []
        points = np.concatenate(clouds)
        lower, upper = geo.percentile_bbox3d(points)
        geometry = Geometry(
            frame="world", centroid=np.median(points, axis=0).tolist(), bbox3d_min=lower, bbox3d_max=upper,
            top_z=float(geo.top_z_from_points(points)), yaw_rad=float(geo.yaw_from_points(points[:, :2])),
            uncertainty_m=0.03, views_used=len(contributing), method=f"triview_{self._neural_backend}",
        )
        provenance = self._provenance(ctx, started, camera=names[0], images=dict(zip(names, images, strict=True)),
                                      model={"da3": "DA3-BASE", "mvsanywhere": "MVSAnywhere-HERO"}[self._neural_backend],
                                      model_sha256=self._model_sha256, calibration_id=ctx.calibration[names[0]].id)
        return [make_percept(
            percept_id="triview-" + "-".join(names), status="ok", camera=names[0], provenance=provenance, geometry=geometry,
            extras={"method": "triview", "strategy": "mvs", "backend": self._neural_backend,
                    "geometry_scope": "whole_scene_visible_surface", "points": len(points), "views": contributing,
                    "depth_unit": "m", "valid_fraction": float((result.depth > 0).mean()), "runtime": self._backend.describe(),
                    "confidence_available": result.confidence is not None,
                    "uncertainty_model": "uncalibrated 0.03 m assumption; model confidence is not a metric error bound",
                    "warnings": ["neural_depth_uncertainty_uncalibrated", "geometry_is_scene_visible_surface"]},
        )]

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
