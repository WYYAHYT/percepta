"""Frozen data model for percepta (spec §3, plus §5 tool context and §10 verification result).

Every model here is the *interface*: field names are JSON names (snake_case), all models
round-trip through ``model_dump_json()`` / ``model_validate_json()``, and no tool may add
top-level fields of its own. Additions are optional fields only, recorded in the spec's
change log (§18).

Two deliberate conventions:

* ``allow_inf_nan=False`` on every model: the service contract forbids NaN/Inf in JSON
  (§6). Missing geometry is expressed as ``null``, never as a non-finite float.
* Enums are ``Literal`` aliases rather than ``enum.Enum``: they serialize to plain JSON
  strings and validate without ``.value`` gymnastics.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---------------------------------------------------------------------------
# §3.1 Enumerations
# ---------------------------------------------------------------------------

Status = Literal["ok", "not_found", "occluded", "unavailable", "timeout", "invalid_request"]
PromptType = Literal["text", "box", "point"]
MaskEncoding = Literal["rle", "png", "none"]
ImageEncoding = Literal["png", "jpeg", "raw"]
GeometryFrame = Literal["pixel", "camera", "world"]
ToolKind = Literal[
    "segmenter", "detector", "depth", "vlm_grounding", "geometry", "verifier", "fake"
]

SCHEMA_VERSION = "percepta/v1"
TOOL_API_VERSION = "percepta/tool/v1"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Frozen(BaseModel):
    """Common base: strict JSON numbers (no NaN/Inf) and attribute assignment allowed."""

    model_config = ConfigDict(allow_inf_nan=False, validate_assignment=False)


class _Forbid(Frozen):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")


# ---------------------------------------------------------------------------
# §3.2 Prompt
# ---------------------------------------------------------------------------


class Prompt(Frozen):
    """A single prompt. ``text`` / ``box`` / ``point`` are mutually exclusive by ``type``.

    * ``text``: noun phrase, not a sentence (spec §8.4).
    * ``box``: normalised ``[cx, cy, w, h]`` in 0..1 of the image.
    * ``point``: pixel ``[u, v]`` = ``[col, row]`` (spec §4.1).
    """

    type: PromptType
    text: str | None = None
    box: list[float] | None = None
    point: list[int] | None = None
    label: bool = True
    min_score: float = 0.2
    top_k: int = 1

    @model_validator(mode="after")
    def _check_payload(self) -> Prompt:
        if self.type == "text":
            if not self.text or not self.text.strip():
                raise ValueError("prompt.type='text' requires a non-empty 'text'")
        elif self.type == "box":
            if self.box is None or len(self.box) != 4:
                raise ValueError("prompt.type='box' requires 'box' = [cx, cy, w, h]")
            if any(b < 0.0 or b > 1.0 for b in self.box):
                raise ValueError("prompt.box must be normalised to 0..1")
            if self.box[2] <= 0.0 or self.box[3] <= 0.0:
                raise ValueError("prompt.box width/height must be > 0")
        elif self.type == "point":
            if self.point is None or len(self.point) != 2:
                raise ValueError("prompt.type='point' requires 'point' = [u, v]")
        if self.top_k < 1:
            raise ValueError("prompt.top_k must be >= 1")
        if not (0.0 <= self.min_score <= 1.0):
            raise ValueError("prompt.min_score must be within 0..1")
        return self


# ---------------------------------------------------------------------------
# §3.3 ImageRef / DepthRef / Calibration
# ---------------------------------------------------------------------------


class ImageRef(_Forbid):
    encoding: ImageEncoding
    data: str  # base64; for raw, base64 of uint8 RGB bytes
    shape: tuple[int, int, int] | None = None  # (H, W, 3), required when raw
    camera: str | None = None

    @model_validator(mode="after")
    def _check_raw_shape(self) -> ImageRef:
        if self.encoding == "raw":
            if self.shape is None:
                raise ValueError("ImageRef(encoding='raw') requires shape=(H, W, 3)")
            if len(self.shape) != 3 or self.shape[2] != 3:
                raise ValueError("ImageRef.shape must be (H, W, 3)")
        return self


class DepthRef(_Forbid):
    """float32 metres, same resolution as the RGB image, base64-encoded."""

    encoding: Literal["raw_f32"] = "raw_f32"
    data: str
    shape: tuple[int, int] = (0, 0)  # (H, W)

    @model_validator(mode="after")
    def _check_shape(self) -> DepthRef:
        if len(self.shape) != 2:
            raise ValueError("DepthRef.shape must be (H, W)")
        return self


class Calibration(_Forbid):
    """Camera intrinsics plus ``T_world_cam`` (camera -> world, spec §4.3)."""

    camera: str
    width: int
    height: int
    K: list[list[float]]  # 3x3 intrinsics, pixels
    T_world_cam: list[list[float]]  # 4x4, p_world = T_world_cam @ p_cam_h
    depth: DepthRef | None = None
    id: str | None = None

    @model_validator(mode="after")
    def _check_matrix_shapes(self) -> Calibration:
        if len(self.K) != 3 or any(len(row) != 3 for row in self.K):
            raise ValueError("Calibration.K must be 3x3")
        if len(self.T_world_cam) != 4 or any(len(row) != 4 for row in self.T_world_cam):
            raise ValueError("Calibration.T_world_cam must be 4x4")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Calibration.width/height must be > 0")
        if self.depth is not None and tuple(self.depth.shape) != (self.height, self.width):
            raise ValueError(
                f"DepthRef.shape {tuple(self.depth.shape)} != Calibration (H, W) "
                f"{(self.height, self.width)}"
            )
        return self


# ---------------------------------------------------------------------------
# §3.4 Mask / Observation / Geometry / Provenance / Percept
# ---------------------------------------------------------------------------


class Mask(_Forbid):
    encoding: MaskEncoding
    size: tuple[int, int] | None = None  # (H, W)
    counts: list[int] | None = None  # encoding="rle"
    data: str | None = None  # encoding="png", base64

    @model_validator(mode="after")
    def _check_payload(self) -> Mask:
        if self.encoding == "rle":
            if self.counts is None:
                raise ValueError("Mask(encoding='rle') requires counts")
            if self.size is None:
                raise ValueError("Mask(encoding='rle') requires size=(H, W)")
            if any(c < 0 for c in self.counts):
                raise ValueError("Mask.counts must be non-negative")
            if sum(self.counts) != int(self.size[0]) * int(self.size[1]):
                raise ValueError(
                    f"Mask.counts sum {sum(self.counts)} != H*W "
                    f"{int(self.size[0]) * int(self.size[1])}"
                )
        elif self.encoding == "png":
            if not self.data:
                raise ValueError("Mask(encoding='png') requires data")
        return self


class Observation(Frozen):
    """One instance: mask, box, score, 2D statistics."""

    object_id: str  # sha1(percept_id + index)
    score: float
    box_xyxy: list[float] | None = None  # pixels, x1/y1 exclusive outer edge (§4.1)
    mask: Mask | None = None
    area_px: int | None = None
    centroid_uv: list[float] | None = None  # pixels [u, v]
    label: str | None = None
    attributes: dict[str, float | str | bool] = Field(default_factory=dict)


class Geometry(Frozen):
    frame: GeometryFrame = "world"
    centroid: list[float] | None = None  # [x, y, z] metres
    bbox3d_min: list[float] | None = None
    bbox3d_max: list[float] | None = None
    top_z: float | None = None
    yaw_rad: float | None = None
    uncertainty_m: float | None = None
    views_used: int | None = None


class Provenance(Frozen):
    tool: str
    tool_version: str
    kind: ToolKind
    model: str | None = None
    model_sha256: str | None = None
    prompt: Prompt | None = None
    camera: str | None = None
    image_sha256: dict[str, str] = Field(default_factory=dict)
    calibration_id: str | None = None
    latency_ms: float = 0.0
    created_at: datetime = Field(default_factory=_utcnow)
    cache_hit: bool = False


class Percept(Frozen):
    schema_version: Literal["percepta/v1"] = SCHEMA_VERSION
    percept_id: str
    status: Status
    message: str | None = None
    camera: str | None = None
    observations: list[Observation] = Field(default_factory=list)
    geometry: Geometry | None = None
    provenance: Provenance
    extras: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# §3.5 Request / response
# ---------------------------------------------------------------------------


class PerceiveOptions(Frozen):
    tools: list[str] | None = None  # None = default pipeline
    fuse: bool = True
    masks: MaskEncoding = "rle"
    min_score: float = 0.2
    top_k: int = 1
    timeout_ms: int = 120_000
    cache: bool = True


class PerceiveRequest(Frozen):
    request_id: str | None = None
    images: dict[str, ImageRef]
    calibration: dict[str, Calibration] = Field(default_factory=dict)
    prompts: list[Prompt]
    context: dict[str, Any] = Field(default_factory=dict)
    options: PerceiveOptions = Field(default_factory=PerceiveOptions)

    @model_validator(mode="after")
    def _check_request(self) -> PerceiveRequest:
        if not self.images:
            raise ValueError("PerceiveRequest.images must contain at least one camera")
        if not self.prompts:
            raise ValueError("PerceiveRequest.prompts must contain at least one prompt")
        for cam, calib in self.calibration.items():
            if cam != calib.camera:
                raise ValueError(
                    f"calibration key {cam!r} does not match Calibration.camera {calib.camera!r}"
                )
            ref = self.images.get(cam)
            if ref is not None and ref.encoding == "raw" and ref.shape is not None:
                # Encoded images are size-checked after decoding; raw ones we can check here.
                if (int(ref.shape[0]), int(ref.shape[1])) != (calib.height, calib.width):
                    raise ValueError(
                        f"image {cam!r} shape {tuple(ref.shape[:2])} != calibration "
                        f"(H, W) {(calib.height, calib.width)} (spec §4.6)"
                    )
        return self


class PerceiveResponse(Frozen):
    request_id: str
    percepts: list[Percept] = Field(default_factory=list)
    fused: Percept | None = None
    timings_ms: dict[str, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# §5.1 Tool capabilities / context (frozen tool protocol)
# ---------------------------------------------------------------------------


class ToolCapabilities(Frozen):
    kind: ToolKind
    prompts: set[PromptType] = Field(default_factory=set)
    masks: bool = False
    boxes: bool = False
    points: bool = False
    depth: bool = False
    multi_view: bool = False
    geometry: bool = False


class ToolContext(Frozen):
    """Everything a tool may read during ``perceive()``.

    ``extras`` carries pipeline state between tools — spec §5.3 passes upstream percepts as
    ``ctx.extras["upstream"]``. It is an *optional additive* field over the frozen §5.1
    shape (recorded in the spec change log).
    """

    model_config = ConfigDict(allow_inf_nan=False, arbitrary_types_allowed=True)

    request_id: str
    images: dict[str, np.ndarray]  # decoded RGB uint8 (H, W, 3)
    calibration: dict[str, Calibration] = Field(default_factory=dict)
    context: dict[str, Any] = Field(default_factory=dict)
    options: PerceiveOptions = Field(default_factory=PerceiveOptions)
    logger: Any = None
    cache: Any = None
    extras: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# §10 Verification result
# ---------------------------------------------------------------------------


class VerificationResult(Frozen):
    ok: bool
    checks: dict[str, bool] = Field(default_factory=dict)  # followed/lifted/stable/visible
    details: dict[str, float] = Field(default_factory=dict)
    status: Status = "ok"
