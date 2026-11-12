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
not invent a result when the pair cannot work: no usable depth returns ``[]`` (no geometry).
Malformed input returns an ``invalid_request`` percept with a specific failure code.

Disparity backends
-----------------
``stereo_backend="sgbm"`` uses OpenCV without weights or a GPU. ``"foundation_stereo"`` uses
an isolated, preconfigured model worker. ``"ffs"`` selects the separately deployed
Fast-FoundationStereo model; it is not an alias and never silently falls back.

What is reported rather than assumed
------------------------------------
Disparity in the rectified frame maps to depth as ``Z = f * B / d``. ``f`` is the *rectified*
focal, not either camera's own ``fx``, and ``B`` is the positive, signed horizontal baseline
from the rectified projection matrices. Reversed and vertical pairs are rejected, never swapped.
The geometry summarizes matched visible surfaces, not a reconstructed object's shape.
"""

from __future__ import annotations

import time
from numbers import Real
from typing import Any

import numpy as np

from .. import geometry as geo
from ..backends.base import BackendRequestError, BackendUnavailable
from ..backends.foundation_stereo_backend import FoundationStereoBackend
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

TOOL_NAME = "stereo_3d"
TOOL_VERSION = "0.1.0"

#: Failure codes, named so a caller can branch on them instead of parsing message strings --
#: the plan (section 4.5) asks for exactly this.
ERR_NEEDS_TWO_VIEWS = "stereo_needs_two_calibrated_views"
ERR_NO_OVERLAP = "no_overlap"
ERR_LOW_VALID_FRACTION = "low_valid_fraction"
ERR_BACKEND_NOT_CONFIGURED = "backend_not_configured"
ERR_INVALID_CALIBRATION = "invalid_calibration"
ERR_INVALID_ORDER = "invalid_stereo_order"
ERR_REVERSED_ORDER = "reversed_stereo_order"
ERR_VERTICAL_RECTIFICATION = "vertical_rectification_unsupported"
ERR_INVALID_MASK = "invalid_stereo_mask"
ERR_INVALID_OPTIONS = "invalid_geometry_options"
ERR_SGBM_IMAGE_SIZE = "sgbm_image_too_small"


class _StereoInputError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _numeric_option(context, key, default, *, integer=False, minimum=None, maximum=None, positive=False):
    """Read a finite request number, rejecting booleans, strings and invalid ranges."""
    value = context.get(key, default)
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise _StereoInputError(ERR_INVALID_OPTIONS, f"{key} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise _StereoInputError(ERR_INVALID_OPTIONS, f"{key} must be a finite number") from exc
    if (not np.isfinite(number) or (integer and not number.is_integer())
            or (minimum is not None and number < minimum) or (maximum is not None and number > maximum)
            or (positive and number <= 0)):
        raise _StereoInputError(ERR_INVALID_OPTIONS, f"{key} has an unsupported value: {value}")
    return int(number) if integer else number


def _sgbm_options(context: dict[str, Any]) -> tuple[int, int]:
    """Validate the request's integer search range and normalize an even window to odd."""
    num_disparities = _numeric_option(context, "stereo_num_disparities", 640, integer=True, minimum=16)
    if num_disparities % 16:
        raise _StereoInputError(ERR_INVALID_OPTIONS, "stereo_num_disparities must be a positive multiple of 16")
    block_size = _numeric_option(context, "stereo_block_size", 5, integer=True, minimum=1)
    return num_disparities, block_size + (block_size % 2 == 0)


def _sgbm_search_range(num_disparities: int, block_size: int, image_shape: tuple[int, int]) -> int:
    """Clamp disparity to a legal 16-pixel range before OpenCV can allocate matcher buffers."""
    height, width = image_shape
    width_limit = 16 * ((width - 1) // 16)
    if width_limit < 16:
        raise _StereoInputError(ERR_SGBM_IMAGE_SIZE, "SGBM requires an image width greater than 16 pixels")
    if block_size > min(height, width):
        raise _StereoInputError(ERR_INVALID_OPTIONS, "stereo_block_size must fit within both image dimensions")
    return min(num_disparities, width_limit)


def _depth_sigma(depth: np.ndarray, focal: float, baseline: float, sigma_d: float) -> np.ndarray:
    """Return finite assumed rectified-Z noise in metres, sigma_Z = Z² sigma_d / (f B)."""
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        sigma = depth * (depth / (focal * baseline)) * sigma_d
    if not np.isfinite(sigma).all():
        raise _StereoInputError(ERR_INVALID_OPTIONS, "stereo_sigma_d_px produces non-finite depth uncertainty")
    return sigma


def _median_sigma(sigma: np.ndarray) -> float:
    """Median of nonempty finite nonnegative metre errors, without a two-value sum overflow."""
    lower, upper = (sigma.size - 1) // 2, sigma.size // 2
    low, high = np.partition(sigma, (lower, upper))[[lower, upper]]
    return float(low) + (float(high) - float(low)) * 0.5


def _error_percept(
    tool: BaseTool,
    ctx: ToolContext,
    started: float,
    code: str,
    message: str,
    *,
    status: str = "invalid_request",
) -> list[Percept]:
    """Keep input/backend errors distinct from an empty, usable-depth result."""
    return [
        make_percept(
            percept_id=f"{tool.name}-error",
            status=status,
            message=message,
            provenance=tool._provenance(ctx, started),
            extras={"code": code},
        )
    ]


def _validate_calibration(cal: Calibration, image_shape: tuple[int, int]) -> None:
    """Validate pinhole K and rigid camera->world T (p_world = T @ p_cam_h)."""
    K = np.asarray(cal.K, dtype=np.float64)
    T = np.asarray(cal.T_world_cam, dtype=np.float64)
    if tuple(image_shape) != (cal.height, cal.width):
        raise _StereoInputError(
            ERR_INVALID_CALIBRATION, f"{cal.camera}: image/calibration size mismatch"
        )
    if (
        K.shape != (3, 3)
        or not np.isfinite(K).all()
        or K[0, 0] <= 0
        or K[1, 1] <= 0
        or not np.allclose(K[2], [0, 0, 1], atol=1e-8, rtol=0)
        or not np.allclose([K[0, 1], K[1, 0]], 0, atol=1e-8, rtol=0)
    ):
        raise _StereoInputError(
            ERR_INVALID_CALIBRATION, f"{cal.camera}: invalid pinhole intrinsics"
        )
    if (
        T.shape != (4, 4)
        or not np.isfinite(T).all()
        or not np.allclose(T[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
        or not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-6, rtol=0)
        or not np.isclose(np.linalg.det(T[:3, :3]), 1.0, atol=1e-6, rtol=0)
    ):
        raise _StereoInputError(ERR_INVALID_CALIBRATION, f"{cal.camera}: T_world_cam must be rigid")


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
    try:
        raw = matcher.compute(grey_l, grey_r)
    except cv2.error as exc:
        raise _StereoInputError(ERR_INVALID_OPTIONS, f"unsupported SGBM image or matcher parameters: {exc}") from exc
    disparity = raw.astype(np.float32) / 16.0  # the fixed-point path scales by 16
    return disparity, disparity > 0.0


def _rectify(
    cal_a: Calibration, cal_b: Calibration, image_shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return K_rect, R1 and both views' rectification matrices, maps and support.

    OpenCV camera A -> B: p_b = R_b.T @ R_a @ p_a + R_b.T @ (C_a - C_b).
    R1/R2 rotate original +z camera points into rectified +z frames; P1/P2/Q
    describe those frames. Maps take rectified output pixels to raw input pixels.
    Calibration has no distortion (spec §4). Only positive horizontal disparity is supported.
    """
    import cv2

    _validate_calibration(cal_a, image_shape)
    _validate_calibration(cal_b, image_shape)
    Ta = np.asarray(cal_a.T_world_cam, dtype=np.float64)
    Tb = np.asarray(cal_b.T_world_cam, dtype=np.float64)
    Ka = np.asarray(cal_a.K, dtype=np.float64)
    Kb = np.asarray(cal_b.K, dtype=np.float64)

    R_ab = Tb[:3, :3].T @ Ta[:3, :3]
    t_ab = Tb[:3, :3].T @ (Ta[:3, 3] - Tb[:3, 3])
    if np.linalg.norm(t_ab) <= 1e-9:
        raise _StereoInputError(ERR_INVALID_CALIBRATION, "coincident camera centres")
    height, width = image_shape
    size = (int(width), int(height))
    try:
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            Ka,
            np.zeros(5),
            Kb,
            np.zeros(5),
            size,
            R_ab,
            t_ab.reshape(3, 1),
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0,
        )
    except cv2.error as exc:
        raise _StereoInputError(ERR_INVALID_CALIBRATION, f"rectification failed: {exc}") from exc
    if not all(np.isfinite(m).all() for m in (R1, R2, P1, P2, Q)):
        raise _StereoInputError(ERR_INVALID_CALIBRATION, "non-finite rectification matrices")
    if min(P1[0, 0], P1[1, 1], P2[0, 0], P2[1, 1]) <= 0:
        raise _StereoInputError(ERR_INVALID_CALIBRATION, "rectified focal must be positive")
    if not np.allclose(P1[:, :3], P2[:, :3], atol=1e-7, rtol=1e-7):
        raise _StereoInputError(ERR_INVALID_CALIBRATION, "rectified intrinsics must agree")
    if abs(P2[1, 3] / P2[1, 1] - P1[1, 3] / P1[1, 1]) > 1e-9:
        raise _StereoInputError(
            ERR_VERTICAL_RECTIFICATION, "horizontal stereo cannot match a vertical pair"
        )
    baseline = float(P1[0, 3] / P1[0, 0] - P2[0, 3] / P2[0, 0])
    if not np.isfinite(baseline) or not np.isfinite(float(P1[0, 0]) * baseline):
        raise _StereoInputError(
            ERR_INVALID_CALIBRATION, "non-finite rectified baseline/focal product"
        )
    if baseline < -1e-9:
        raise _StereoInputError(
            ERR_REVERSED_ORDER, "right camera is left of left camera; supply the correct order"
        )
    if baseline <= 1e-9:
        raise _StereoInputError(ERR_INVALID_CALIBRATION, "degenerate rectified baseline")

    map_a = cv2.initUndistortRectifyMap(Ka, None, R1, P1, size, cv2.CV_32FC1)
    map_b = cv2.initUndistortRectifyMap(Kb, None, R2, P2, size, cv2.CV_32FC1)
    # A pixel is supported only if the whole bilinear sampling footprint is in the source.
    source_support = np.ones(image_shape, np.float32)
    support = [
        cv2.remap(
            source_support, *maps, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        == 1.0
        for maps in (map_a, map_b)
    ]
    K_rect = np.asarray(P1[:3, :3], dtype=np.float64)
    info = {
        "focal_rect_px": float(P1[0, 0]),
        "baseline_m": baseline,
        "R_a_to_rect": R1,
        "R_b_to_rect": R2,
        "P1": P1,
        "P2": P2,
        "Q": Q,
        "roi_a": roi1,
        "roi_b": roi2,
        "map_a": map_a,
        "map_b": map_b,
        "support_a": support[0],
        "support_b": support[1],
        "original_fx": {"a": float(Ka[0, 0]), "b": float(Kb[0, 0])},
    }
    return K_rect, R1, info


def _remap_pair(
    image_a: np.ndarray,
    image_b: np.ndarray,
    info: dict[str, Any],
    left_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Raw RGB/mask -> rectified pixels; RGB is bilinear, the left mask nearest-neighbour."""
    import cv2

    if left_mask is not None:
        try:
            left_mask = np.asarray(left_mask)
            if left_mask.shape != image_a.shape[:2]:
                raise ValueError("left mask must have the original left image's (H, W)")
            if left_mask.dtype.kind not in "buif" or not np.isfinite(left_mask).all():
                raise ValueError("left mask must contain finite boolean/0/1 values")
            if not np.isin(left_mask, [0, 1]).all():
                raise ValueError("left mask must contain boolean/0/1 values")
        except (TypeError, ValueError) as exc:
            raise _StereoInputError(ERR_INVALID_MASK, str(exc)) from exc
        left_mask = cv2.remap(
            left_mask.astype(np.uint8),
            *info["map_a"],
            cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(bool)
    rect_a = cv2.remap(
        image_a, *info["map_a"], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    rect_b = cv2.remap(
        image_b, *info["map_b"], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    return rect_a, rect_b, left_mask


def _rectified_to_world(uv, depth, K_rect, R_a_to_rect, T_world_a) -> np.ndarray:
    """Rectified pixels + positive Z -> world metres (OpenCV +z).

    p_rect = unproject(uv, Z, K_rect); p_world = R_world_a @ R1.T @ p_rect + C_a.
    """
    rect_pts = geo.unproject_pixels(uv, depth, K_rect)
    T = np.asarray(T_world_a, dtype=np.float64)
    a_pts = rect_pts @ np.asarray(R_a_to_rect)
    return a_pts @ T[:3, :3].T + T[:3, 3]


def _disparity_points(
    disparity: np.ndarray,
    valid: np.ndarray,
    cal_a: Calibration,
    cal_b: Calibration,
    K_rect: np.ndarray,
    info: dict[str, Any],
    left_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rectified disparity -> world points, rectified Z and final (H,W) validity.

    Z = f*B/d, u_right = u_left-d. Both remap footprints and the right subpixel
    correspondence must be supported. Reject points behind either original +z camera.
    Points/depth follow row-major order of the returned validity mask.
    """
    valid = np.asarray(valid, dtype=bool) & info["support_a"]
    valid &= np.isfinite(disparity) & (disparity > 0)
    if left_mask is not None:
        valid &= left_mask
    rows, cols = np.nonzero(valid)
    right_u = cols - disparity[rows, cols].astype(np.float64)
    inside = (right_u >= 0) & (right_u <= disparity.shape[1] - 1)
    rows, cols, right_u = rows[inside], cols[inside], right_u[inside]
    supported = (
        info["support_b"][rows, np.floor(right_u).astype(int)]
        & info["support_b"][rows, np.ceil(right_u).astype(int)]
    )
    rows, cols = rows[supported], cols[supported]
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        dist = info["focal_rect_px"] * info["baseline_m"] / disparity[rows, cols].astype(np.float64)
        world = _rectified_to_world(
            np.column_stack((cols, rows)),
            dist,
            K_rect,
            info["R_a_to_rect"],
            cal_a.T_world_cam,
        )
    keep = np.isfinite(world).all(axis=1) & np.isfinite(dist) & (dist > 0)
    # Only transform finite points, avoiding invalid arithmetic for infinite depth.
    for cal in (cal_a, cal_b):
        front = np.zeros(keep.shape, dtype=bool)
        front[keep] = geo.world_to_camera(world[keep], cal.T_world_cam)[:, 2] > 0
        keep &= front
    valid = np.zeros(disparity.shape, dtype=bool)
    valid[rows[keep], cols[keep]] = True
    return world[keep], dist[keep], valid


class Stereo3DTool(BaseTool):
    """Two calibrated views -> metric depth -> ``Geometry``.

    Parameters the policy may pass through ``context``:

    * ``stereo_left`` / ``stereo_right`` -- optional ordered camera names. Supply both, naming
      exactly the two supplied images/calibrations. Absent, use sorted camera names; the signed
      baseline must still be positive. Extra views, reversed or vertical pairs are rejected.
    * ``stereo_backend`` -- ``"sgbm"`` (default), deployed ``"foundation_stereo"``, or ``"ffs"``.
    * ``stereo_num_disparities`` -- search range, multiple of 16. It must cover ``f*B/Z`` for the
      nearest distance of interest: about 400 px at 0.6 m on RoboDojo's rig.
    * ``stereo_block_size`` -- positive integer matcher window; forced odd and must fit the images.
    * ``stereo_mask`` -- optional ``{camera: bool array}``; only the selected left camera's mask
      is consumed. It must match the raw image size and is rectified with nearest-neighbour
      sampling before limiting the visible-surface points. Right masks are not used.
    * ``stereo_sigma_d_px`` -- finite, strictly positive assumed disparity noise, default 1.0.
    * ``min_valid_fraction`` -- finite number in [0,1]; below it the tool reports ``low_valid_fraction`` rather than a
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

    def __init__(self) -> None:
        super().__init__()
        self._backend: FoundationStereoBackend | None = None
        self._model_sha256: str | None = None
        self._config_sha256: str | None = None
        self._deployed_backend: str | None = None

    def load(self, config: dict[str, Any] | None = None) -> None:
        try:
            import cv2  # noqa: F401
        except ImportError as exc:
            raise ToolUnavailable("stereo_3d requires the stereo extra (OpenCV)") from exc

        super().load(config or {})
        backend = str(self.config.get("stereo_backend", "sgbm")).lower()
        if backend not in {"foundation_stereo", "ffs"}:
            return
        model_name = "FoundationStereo" if backend == "foundation_stereo" else "Fast-FoundationStereo"
        try:
            weights = resolve_weights(
                self.config.get("checkpoint"), config=self.config,
                verify=self.config.get("verify_weights", "if-declared"),
                what=f"{model_name} checkpoint",
            )
            if weights.path is None:
                raise WeightError(f"{model_name} requires a local checkpoint")
            model_config = resolve_weights(
                self.config.get("model_config", str(weights.path.with_name("cfg.yaml"))),
                expected_sha256=self.config.get("model_config_sha256"),
                verify=self.config.get("verify_weights", "if-declared"),
                what=f"{model_name} model config",
            )
            if model_config.path is None:
                raise WeightError(f"{model_name} requires a local model config")
            self._model_sha256 = weights.observed_sha256
            self._config_sha256 = model_config.observed_sha256
            factory = {}
            if backend == "ffs":
                factory = {
                    "backend_factory": "percepta.backends.fast_foundation_stereo_worker:FastFoundationStereoAdapter",
                    "backend_args": {
                        "source_dir": self.config.get("source_dir"), "checkpoint": str(weights.path),
                        "model_config": str(model_config.path), "valid_iters": self.config.get("valid_iters", 8),
                        "max_disp": self.config.get("max_disp", 416),
                    },
                }
            runtime = "foundation-stereo" if backend == "foundation_stereo" else "fast-foundation-stereo"
            self._backend = FoundationStereoBackend(
                python=self.config.get("python", f"uv_envs/{runtime}/.venv/bin/python"),
                source_dir=self.config.get("source_dir"),
                checkpoint=str(weights.path), model_config=str(model_config.path),
                hf_cache=self.config.get("hf_cache"),
                valid_iters=self.config.get("valid_iters", 32),
                cuda_visible_devices=self.config.get("cuda_visible_devices"),
                startup_timeout_s=self.config.get("startup_timeout_s", 180),
                timeout_s=self.config.get("timeout_s", 120),
                **factory,
            )
            self._backend.name = backend
            self._backend.start()
            self._deployed_backend = backend
        except (WeightError, BackendUnavailable, BackendRequestError, OSError, ValueError) as exc:
            self.close()
            raise ToolUnavailable(str(exc)) from exc

    def close(self) -> None:
        if self._backend is not None:
            self._backend.close()
            self._backend = None
        self._deployed_backend = None

    def perceive(self, ctx: ToolContext, prompts: list[Prompt]) -> list[Percept]:
        _ = prompts
        started = time.perf_counter()
        context = dict(ctx.context or {})
        for key, value in (self.config or {}).items():
            context.setdefault(key, value)

        calibrated = ctx.calibration
        if len(ctx.images) != 2 or set(calibrated) != set(ctx.images):
            return _error_percept(
                self,
                ctx,
                started,
                ERR_NEEDS_TWO_VIEWS,
                "stereo_3d requires exactly two images and the same two calibrations; no pair selection",
            )
        left_name, right_name = sorted(calibrated)
        if "stereo_left" in context or "stereo_right" in context:
            left_name, right_name = context.get("stereo_left"), context.get("stereo_right")
            if (
                not isinstance(left_name, str)
                or not isinstance(right_name, str)
                or left_name == right_name
                or {left_name, right_name} != set(ctx.images)
            ):
                return _error_percept(
                    self,
                    ctx,
                    started,
                    ERR_INVALID_ORDER,
                    "stereo_left and stereo_right must name exactly the supplied pair, once each",
                )

        backend = str(context.get("stereo_backend", "sgbm")).lower()
        if backend == "foundation_stereo" and self._backend is None:
            raise ToolUnavailable("FoundationStereo is not loaded in this deployment")
        if backend not in {"sgbm", "foundation_stereo", "ffs"} or (
            backend in {"foundation_stereo", "ffs"} and backend != self._deployed_backend
        ):
            self._log(
                ctx,
                f"{ERR_BACKEND_NOT_CONFIGURED}: backend {backend!r} is not wired yet; "
                "'sgbm' is available",
                level="warning",
            )
            return _error_percept(
                self,
                ctx,
                started,
                ERR_BACKEND_NOT_CONFIGURED,
                f"backend {backend!r} is not configured",
                status="unavailable",
            )

        cal_l, cal_r = calibrated[left_name], calibrated[right_name]
        image_l = np.asarray(ctx.images[left_name])
        image_r = np.asarray(ctx.images[right_name])
        for name, cal, image in ((left_name, cal_l, image_l), (right_name, cal_r, image_r)):
            if cal.camera != name:
                return _error_percept(
                    self,
                    ctx,
                    started,
                    ERR_INVALID_CALIBRATION,
                    f"calibration key {name!r} differs from camera {cal.camera!r}",
                )
            if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
                return _error_percept(
                    self,
                    ctx,
                    started,
                    ERR_INVALID_CALIBRATION,
                    f"{name}: expected uint8 RGB image",
                )
        if image_l.shape[:2] != image_r.shape[:2]:
            return _error_percept(
                self,
                ctx,
                started,
                ERR_INVALID_CALIBRATION,
                "stereo image sizes must agree",
            )
        try:
            sigma_d = _numeric_option(context, "stereo_sigma_d_px", 1.0, positive=True)
            min_valid = _numeric_option(context, "min_valid_fraction", 0.01, minimum=0, maximum=1)
            if backend == "sgbm":
                num_disparities, block_size = _sgbm_options(context)
                num_disparities = _sgbm_search_range(num_disparities, block_size, image_l.shape[:2])
            K_rect, _R_a, info = _rectify(cal_l, cal_r, image_l.shape[:2])
            masks = context.get("stereo_mask")
            if masks is not None and not isinstance(masks, dict):
                raise _StereoInputError(
                    ERR_INVALID_MASK, "stereo_mask must map camera names to masks"
                )
            if masks is not None and set(masks) - set(ctx.images):
                raise _StereoInputError(ERR_INVALID_MASK, "stereo_mask contains an unknown camera")
            mask = masks.get(left_name) if masks is not None else None
            rect_l, rect_r, mask = _remap_pair(image_l, image_r, info, mask)
        except _StereoInputError as exc:
            return _error_percept(self, ctx, started, exc.code, str(exc))

        focal = info["focal_rect_px"]
        baseline = info["baseline_m"]
        if backend in {"foundation_stereo", "ffs"}:
            try:
                disparity = self._backend.infer_pair(rect_l, rect_r)
            except (BackendUnavailable, BackendRequestError) as exc:
                raise ToolUnavailable(str(exc)) from exc
            valid = np.isfinite(disparity) & (disparity > 0)
            backend_details = {
                "runtime": self._backend.describe(),
                "model_config_sha256": self._config_sha256,
            }
        else:
            if context.get("stereo_num_disparities", 640) > num_disparities:
                self._log(
                    ctx,
                    f"stereo_num_disparities exceeds the image width {image_l.shape[1]}; clamping to {num_disparities}",
                    level="warning",
                )
            need_06 = focal * baseline / 0.6
            if num_disparities < need_06:
                self._log(
                    ctx,
                    f"stereo search range {num_disparities} px is below the ~{need_06:.0f} px this "
                    "pair needs at 0.6 m; near objects will not match",
                    level="warning",
                )
            try:
                disparity, valid = _sgbm_disparity(
                    rect_l, rect_r, num_disparities=num_disparities, block_size=block_size
                )
            except _StereoInputError as exc:
                return _error_percept(self, ctx, started, exc.code, str(exc))
            backend_details = {"num_disparities": num_disparities, "block_size": block_size}
        world, dist, valid = _disparity_points(disparity, valid, cal_l, cal_r, K_rect, info, mask)
        valid_fraction = float(valid.mean())
        if dist.size == 0 or valid_fraction < min_valid:
            self._log(
                ctx,
                f"{ERR_LOW_VALID_FRACTION}: only {valid_fraction * 100:.2f}% of pixels matched "
                f"(min_valid_fraction={min_valid})",
                level="warning",
            )
            return []

        bbox_min, bbox_max = geo.percentile_bbox3d(world)
        centroid = [
            0.5 * (bbox_min[0] + bbox_max[0]),
            0.5 * (bbox_min[1] + bbox_max[1]),
            0.5 * (bbox_min[2] + bbox_max[2]),
        ]
        top_z = float(geo.top_z_from_points(world))
        yaw = float(geo.yaw_from_points(world[:, :2]))

        # sigma_d is assumed disparity noise, not measured model accuracy.
        try:
            sigma_z = _depth_sigma(dist, focal, baseline, sigma_d)
        except _StereoInputError as exc:
            return _error_percept(self, ctx, started, exc.code, str(exc))
        height_unc = _median_sigma(sigma_z)

        geometry = Geometry(
            frame="world",
            centroid=[float(v) for v in centroid],
            bbox3d_min=[float(v) for v in bbox_min],
            bbox3d_max=[float(v) for v in bbox_max],
            top_z=top_z,
            yaw_rad=yaw,
            uncertainty_m=height_unc,
            views_used=2,
            lateral_uncertainty_m=None,
            height_uncertainty_m=None,
            view_disagreement_m=None,
            method=f"stereo_{backend}",
        )

        extras: dict[str, Any] = {
            "method": "stereo",
            "backend": backend,
            "masked": mask is not None,
            "geometry_scope": "matched_visible_surface",
            "geometry_summary": "percentile_bbox_midpoint; bounds/top/yaw describe samples, not object shape",
            "uncertainty_model": "median Z^2*sigma_d/(f*B), assumed rectified-depth noise, not calibrated world-axis error",
            "uncertainty_excludes": [
                "calibration_error",
                "occlusion",
                "mismatches",
                "unseen_surfaces",
            ],
            "rectified": True,
            "left_support_fraction": float(info["support_a"].mean()),
            "right_support_fraction": float(info["support_b"].mean()),
            "left": left_name,
            "right": right_name,
            "focal_rect_px": float(focal),
            "baseline_m": float(baseline),
            **backend_details,
            "sigma_d_px": sigma_d,
            "valid_fraction": valid_fraction,
            "points": int(dist.size),
            "median_depth_m": float(np.median(dist)),
            "median_sigma_z_m": height_unc,
            "disparity_p02": float(np.percentile(disparity[valid], 2)),
            "disparity_p98": float(np.percentile(disparity[valid], 98)),
            "warnings": [],
        }

        self._log(
            ctx,
            f"stereo_3d {left_name}->{right_name}: {dist.size} pts, {valid_fraction * 100:.1f}% valid, depth {float(np.median(dist)):.3f} m, "
            f"sigma_z {height_unc * 1000.0:.1f} mm, {(time.perf_counter() - started) * 1000.0:.1f} ms",
            level="debug",
        )

        provenance = self._provenance(
            ctx,
            started,
            camera=left_name,
            images={left_name: image_l, right_name: image_r},
            calibration_id=getattr(cal_l, "id", None),
            model={"foundation_stereo": "FoundationStereo", "ffs": "Fast-FoundationStereo"}.get(backend),
            model_sha256=self._model_sha256 if backend in {"foundation_stereo", "ffs"} else None,
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
