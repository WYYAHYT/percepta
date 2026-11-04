"""Action verification / evidence comparison (spec §10, acceptance S6, milestone M4).

``compare(before, after, action, options)`` answers one question: *did the observation
after an action provide the evidence that the action actually happened?* It never
raises on missing masks or geometry — a check that cannot be evaluated is ``False`` with
a numeric marker in ``details`` (the frozen ``VerificationResult`` has no ``message``
field, so a human-readable reason cannot be attached without changing the schema).

Checks (spec §10)
-----------------
``followed``
    Pixel-space motion. With ``shift = centroid(after) - centroid(before)`` in pixels
    (``u`` = column, ``v`` = row, origin top-left) and ``delta = action["ee_delta_px"]``
    (the end-effector's pixel displacement), the direction cosine is

        cos = (shift · delta) / (|shift| · |delta|)

    and the magnitude error is ``|shift - delta|`` (pixels). The check passes when
    ``cos >= followed_min_cos`` (default 0.5) and ``error <= followed_tol_px`` (default
    12 px). A zero shift with a zero delta (nothing expected, nothing observed) passes;
    exactly one of them being zero fails (direction is undefined).

``lifted``
    ``after.geometry.centroid.z - before.geometry.centroid.z >= required_lift_m``
    (default 0.10 m). Missing geometry or mismatched geometry frames -> ``False`` with
    ``details["geometry_available"] == 0.0`` / ``details["frame_mismatch"] == 1.0``.

``stable``
    Mask IoU between ``after`` and ``options["stable_with"]`` (a :class:`Percept`, an
    :class:`Observation`, a :class:`Mask`, or a bool ndarray) >= ``stable_iou`` (default
    0.6). Only computed when ``stable_with`` is given; per decision D11 two empty masks
    score 0.0 and therefore never count as stable.

``visible``
    ``after.status == "ok"`` and the first observation's area >= ``min_area_px``
    (default 1 px).

``options["required"]`` narrows the set of checks that must pass; by default every check
that applies (``followed``, ``lifted``, ``stable`` when requested, ``visible``) is
required. A failed ``not_found`` percept is **never** a pass (spec §10, last line), even
if ``required`` was narrowed.

Status of the result
--------------------
``status`` mirrors the governing percept status: ``ok`` on a pass; ``not_found`` when
either side is ``not_found``; ``invalid_request`` for malformed ``action``/``options``;
otherwise ``after.status`` (``occluded`` / ``unavailable`` / ``timeout`` /
``invalid_request`` propagate). When ``after.status == "ok"`` and only a substantive
check failed (e.g. the object did not lift far enough), ``status`` stays ``"ok"`` and the
verdict lives in ``ok``/``checks``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from .masks import mask_centroid, mask_iou, to_array
from .schema import Geometry, Mask, Observation, Percept, Status, VerificationResult

__all__ = ["compare"]

_CHECKS = ("followed", "lifted", "stable", "visible")
_EPS = 1e-9

_DEFAULTS: dict[str, Any] = {
    "followed_min_cos": 0.5,
    "followed_tol_px": 12.0,
    "required_lift_m": 0.10,
    "stable_iou": 0.6,
    "min_area_px": 1.0,
}


class _BadInput(ValueError):
    """Malformed ``action``/``options``; surfaced as ``status="invalid_request"``."""


# ---------------------------------------------------------------------------
# option / input parsing
# ---------------------------------------------------------------------------


def _number(options: Mapping[str, Any], name: str, *, low: float | None = None,
            high: float | None = None) -> float:
    value = options.get(name, _DEFAULTS[name])
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise _BadInput(f"options[{name!r}] must be a number") from exc
    if not math.isfinite(number):
        raise _BadInput(f"options[{name!r}] must be finite")
    if low is not None and number < low:
        raise _BadInput(f"options[{name!r}] must be >= {low}")
    if high is not None and number > high:
        raise _BadInput(f"options[{name!r}] must be <= {high}")
    return number


def _parse_delta(value: Any) -> np.ndarray | None:
    """``action["ee_delta_px"]`` as a finite 2-vector ``[du, dv]`` in pixels, else None."""
    if value is None or isinstance(value, (str, bytes, Mapping)):
        return None
    try:
        arr = np.asarray(list(value), dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if arr.shape != (2,) or not bool(np.all(np.isfinite(arr))):
        return None
    return arr


# ---------------------------------------------------------------------------
# evidence extraction (never raises on malformed percept payloads)
# ---------------------------------------------------------------------------


def _first_observation(percept: Percept) -> Observation | None:
    return percept.observations[0] if percept.observations else None


def _decode_mask(mask: Mask | None) -> np.ndarray | None:
    """Decode a schema mask to bool pixels; ``None`` when absent or undecodable."""
    if mask is None or mask.encoding == "none":
        return None
    try:
        return to_array(mask)
    except Exception:  # noqa: BLE001 - corrupt payloads degrade to a failed check
        return None


def _percept_mask(percept: Percept) -> np.ndarray | None:
    observation = _first_observation(percept)
    return None if observation is None else _decode_mask(observation.mask)


def _coerce_mask(value: Any) -> np.ndarray | None:
    """Accept the ``stable_with`` shapes documented above; ``None`` stays unmatchable."""
    if value is None:
        return None
    if isinstance(value, Percept):
        return _percept_mask(value)
    if isinstance(value, Observation):
        return _decode_mask(value.mask)
    if isinstance(value, Mask):
        return _decode_mask(value)
    if isinstance(value, np.ndarray):
        return np.asarray(value).astype(bool)
    return None


def _centroid_uv(percept: Percept) -> tuple[float, float] | None:
    """First observation's pixel centroid, falling back from ``centroid_uv`` to the mask."""
    observation = _first_observation(percept)
    if observation is None:
        return None
    if observation.centroid_uv is not None and len(observation.centroid_uv) >= 2:
        try:
            u = float(observation.centroid_uv[0])
            v = float(observation.centroid_uv[1])
        except (TypeError, ValueError):
            u = v = math.nan
        if math.isfinite(u) and math.isfinite(v):
            return (u, v)
    mask = _decode_mask(observation.mask)
    return None if mask is None else mask_centroid(mask)


def _area_px(percept: Percept) -> float | None:
    observation = _first_observation(percept)
    if observation is None:
        return None
    if observation.area_px is not None:
        try:
            return float(observation.area_px)
        except (TypeError, ValueError):
            return None
    mask = _decode_mask(observation.mask)
    return None if mask is None else float(np.count_nonzero(mask))


def _centroid_z(percept: Percept) -> tuple[float, str] | None:
    """``(z, frame)`` from ``geometry.centroid``; ``None`` when geometry is missing."""
    geometry: Geometry | None = percept.geometry
    if geometry is None or geometry.centroid is None or len(geometry.centroid) < 3:
        return None
    try:
        z = float(geometry.centroid[2])
    except (TypeError, ValueError):
        return None
    return (z, geometry.frame) if math.isfinite(z) else None


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------


def compare(
    before: Percept, after: Percept, action: dict, options: dict | None = None
) -> VerificationResult:
    """Compare two percepts around an action and return a :class:`VerificationResult`.

    ``before``/``after`` are the observations of the same object (usually the same
    camera and prompt) before and after the action; ``action`` must carry
    ``ee_delta_px`` = ``[du, dv]`` pixel displacement. ``options`` accepts
    ``followed_min_cos``, ``followed_tol_px``, ``required_lift_m``, ``stable_iou``,
    ``min_area_px``, ``stable_with`` and ``required`` (see the module docstring).
    """
    checks: dict[str, bool] = {}
    details: dict[str, float] = {}

    if not isinstance(before, Percept) or not isinstance(after, Percept):
        return VerificationResult(ok=False, status="invalid_request")
    if options is not None and not isinstance(options, Mapping):
        return VerificationResult(ok=False, status="invalid_request")

    opts: Mapping[str, Any] = options or {}
    try:
        min_cos = _number(opts, "followed_min_cos", low=-1.0, high=1.0)
        tol_px = _number(opts, "followed_tol_px", low=0.0)
        required_lift_m = _number(opts, "required_lift_m")
        stable_iou = _number(opts, "stable_iou", low=0.0, high=1.0)
        min_area_px = _number(opts, "min_area_px", low=0.0)
    except _BadInput:
        return VerificationResult(ok=False, status="invalid_request")

    # -- followed: pixel-space agreement with the commanded end-effector motion -----
    action_map: Mapping[str, Any] = action if isinstance(action, Mapping) else {}
    delta = _parse_delta(action_map.get("ee_delta_px"))
    ee_available = delta is not None
    details["ee_delta_available"] = 1.0 if ee_available else 0.0

    before_uv, after_uv = _centroid_uv(before), _centroid_uv(after)
    details["masks_available"] = 1.0 if before_uv is not None and after_uv is not None else 0.0
    checks["followed"] = False
    if ee_available and before_uv is not None and after_uv is not None:
        shift = np.array(
            [after_uv[0] - before_uv[0], after_uv[1] - before_uv[1]], dtype=np.float64
        )
        shift_norm = float(np.linalg.norm(shift))
        delta_norm = float(np.linalg.norm(delta))
        details["centroid_shift_px"] = shift_norm
        details["ee_delta_norm_px"] = delta_norm
        details["followed_error_px"] = float(np.linalg.norm(shift - delta))
        if shift_norm <= _EPS and delta_norm <= _EPS:
            # Nothing was expected and nothing happened: agreement by default.
            cos, magnitude_ok = 1.0, True
        elif shift_norm <= _EPS or delta_norm <= _EPS:
            cos, magnitude_ok = 0.0, False
        else:
            cos = float(np.dot(shift, delta) / (shift_norm * delta_norm))
            cos = max(-1.0, min(1.0, cos))
            magnitude_ok = details["followed_error_px"] <= tol_px
        details["direction_cos"] = cos
        checks["followed"] = bool(magnitude_ok and cos >= min_cos)

    # -- lifted: world-z displacement >= required_lift_m ----------------------------
    before_z, after_z = _centroid_z(before), _centroid_z(after)
    checks["lifted"] = False
    if before_z is None or after_z is None:
        details["geometry_available"] = 0.0
    else:
        details["geometry_available"] = 1.0
        if before_z[1] != after_z[1]:
            # z in different frames is not comparable; refuse rather than guess.
            details["frame_mismatch"] = 1.0
        else:
            lift_m = after_z[0] - before_z[0]
            details["lift_m"] = float(lift_m)
            checks["lifted"] = bool(lift_m >= required_lift_m)

    # -- stable: mask IoU against an optional previous after-mask ------------------
    stable_with = opts.get("stable_with")
    if stable_with is not None:
        checks["stable"] = False
        after_mask, other_mask = _percept_mask(after), _coerce_mask(stable_with)
        available = after_mask is not None and other_mask is not None
        details["stable_masks_available"] = 1.0 if available else 0.0
        if after_mask is not None and other_mask is not None:
            if after_mask.shape != other_mask.shape:
                details["stable_shape_mismatch"] = 1.0
            else:
                iou = float(mask_iou(after_mask, other_mask))
                details["mask_iou"] = iou
                checks["stable"] = bool(iou >= stable_iou)

    # -- visible: after is ok and its first observation covers enough pixels -------
    area = _area_px(after)
    checks["visible"] = False
    if area is not None:
        details["area_px"] = float(area)
        checks["visible"] = bool(after.status == "ok" and area >= min_area_px)

    # -- which checks are required? -------------------------------------------------
    malformed = False
    required_opt = opts.get("required")
    if required_opt is None:
        required = [name for name in _CHECKS if name in checks]
    else:
        if isinstance(required_opt, (str, bytes)) or not isinstance(
            required_opt, (list, tuple, set, frozenset)
        ):
            malformed = True
            required = []
        else:
            required = [str(name) for name in required_opt]
            if not required:
                malformed = True
            if any(name not in _CHECKS for name in required):
                malformed = True
    if "stable" in required and "stable" not in checks:
        # ``required=["stable"]`` without stable_with cannot be evaluated: say so
        # explicitly instead of silently passing or raising.
        checks["stable"] = False
        malformed = True

    # A malformed action makes ``followed`` unverifiable; that is a bad request when the
    # caller asked us to verify it.
    if "followed" in required and not ee_available:
        malformed = True

    passed = bool(required) and not malformed
    passed = passed and all(checks.get(name, False) for name in required)
    # §10 last line: not_found is never evidence of success, even if the caller narrowed
    # ``required``; a non-ok percept cannot be a passing verification either.
    if after.status == "not_found" or before.status == "not_found" or after.status != "ok":
        passed = False

    status: Status = _governing_status(before, after, ok=passed, malformed=malformed)
    return VerificationResult(ok=passed, checks=checks, details=details, status=status)


def _governing_status(
    before: Percept, after: Percept, *, ok: bool, malformed: bool
) -> Status:
    """Status of the verification result (see the module docstring's precedence)."""
    if ok:
        return "ok"
    if after.status == "not_found" or before.status == "not_found":
        return "not_found"
    if malformed:
        return "invalid_request"
    return after.status
