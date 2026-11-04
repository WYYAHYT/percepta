"""Pipeline orchestration — the single path used by the HTTP service and ``perceive_local``.

Decision D12: the orchestration semantics live here, not twice. Callers are
:mod:`percepta.service.routes` (``POST /v1/perceive``) and
:meth:`percepta.client.PerceptaClient.perceive_local`.

Semantics implemented (spec §4.6, §5.3, §11.4, §16.4, decisions D1/D2/D6/D15):

* every :class:`~percepta.schema.ImageRef` is decoded to RGB ``uint8 (H, W, 3)`` and its
  ``(H, W)`` is compared with the matching ``Calibration``; a mismatch is a request-level
  ``invalid_request`` (HTTP 400), raised as :class:`PipelineRequestError`;
* each configured tool is called **once** per request with every camera in
  ``ToolContext.images`` (D6), while holding ``registry.lock_for(name)``;
* prompts a tool did not declare in ``capabilities().prompts`` never reach the tool: they
  become a ``status="invalid_request"`` Percept with ``message="unsupported_prompt"`` and
  ``extras["code"]="unsupported_prompt"`` (D1) instead of failing the request. A tool that
  declares **no** prompt types at all is prompt-agnostic (a geometry stage ignores prompts),
  so it is not rejected;
* earlier percepts reach later tools as ``ctx.extras["upstream"]`` (D2, spec §5.3);
* a tool whose ``capabilities().kind`` is ``"geometry"`` produces the response's ``fused``:
  its output is not added to ``percepts`` and :func:`percepta.fusion.fuse` is not called a
  second time (decision D15). Fusion only runs as a fallback when a fuse was requested and
  no geometry stage ran;
* ``ToolUnavailable`` / ``ToolLoadError`` become ``status="unavailable"`` Percepts — never
  ``not_found`` (spec §16.4). Only a tool that *looked* may report ``not_found``;
* ``options.fuse`` calls ``percepta.fusion.fuse`` if that module exists (owned by another
  agent); when it does not, ``fused`` stays ``None`` and ``geometry_unavailable`` is warned;
* ``timings_ms`` is filled per tool, ``warnings`` aggregates tool warnings,
  ``ambiguous_instances`` (via :func:`percepta.prompts.disambiguate`) and
  ``geometry_unavailable``;
* an optional duck-typed cache (``get``/``put``/``make_key``, see ``AGENTS.md`` §4) shortens
  repeated requests; a hit sets ``provenance.cache_hit=True`` and leaves the result content
  untouched (spec §11.4).

Coordinate note: this module never converts coordinates. It hands tools RGB ``(H, W, 3)``
arrays and the frozen :class:`~percepta.schema.Calibration` (``T_world_cam`` is camera→world,
spec §4.3); every pixel/metre conversion stays inside the tools and the fusion layer.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from typing import Any

import numpy as np

from .config import AppConfig
from .masks import array_sha256, decode_image
from .prompts import disambiguate
from .registry import ToolLoadError, ToolRegistry, UnknownTool
from .schema import (
    PerceiveOptions,
    PerceiveRequest,
    PerceiveResponse,
    Percept,
    Prompt,
    ToolCapabilities,
    ToolContext,
)
from .tools.base import (
    ToolError,
    ToolUnavailable,
    UnsupportedPrompt,
    build_provenance,
    make_percept,
)

__all__ = ["PipelineRunner", "PipelineRequestError", "prepare_tools"]

log = logging.getLogger("percepta.pipeline")


class PipelineRequestError(ValueError):
    """A request-level failure that maps to an HTTP 4xx (default 400 ``invalid_request``).

    Raised for problems that are not attributable to one tool: an undecodable image, an
    image whose ``(H, W)`` disagrees with its calibration (spec §4.6), a pipeline whose
    requested tools are all disabled, or an unknown tool name (spec §5.2). Both callers see
    the same exception; the HTTP route maps it to ``status_code`` with
    ``{"error": code, "message": message}``.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str = "invalid_request",
        status_code: int = 400,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = int(status_code)

    def as_dict(self) -> dict[str, Any]:
        return {"error": self.code, "message": self.message}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _add_warning(warnings: list[str], warning: str) -> None:
    """Append ``warning`` once, preserving the order warnings were discovered in."""
    if warning not in warnings:
        warnings.append(warning)


def _elapsed_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000.0, 3)


def _coerce_percepts(value: Any) -> list[Percept]:
    """Normalise whatever a tool returned into ``list[Percept]`` (tolerant, never silent)."""
    if value is None:
        return []
    if isinstance(value, Percept):
        return [value]
    return [item if isinstance(item, Percept) else Percept.model_validate(item) for item in value]


def _set_cache_hit(percept: Percept) -> Percept:
    """Copy a cached Percept with ``provenance.cache_hit=True`` (content otherwise intact)."""
    provenance = percept.provenance.model_copy(update={"cache_hit": True})
    return percept.model_copy(update={"provenance": provenance})


def _cache_key(
    cache: Any,
    image_hashes: dict[str, str],
    prompt: Prompt,
    tool: str,
    options: PerceiveOptions,
) -> str:
    """Cache key from ``sha256(image) + prompt + tool + effective options`` (spec §11.4).

    The cache module owns the key material; ``options`` is folded in because
    ``options.masks`` (and ``min_score``/``top_k``) change the cached payload. The 3-argument
    form is still accepted so an older cache implementation keeps working, and a local
    equivalent hash is the last resort. Get and put must use the same call path, otherwise
    every lookup misses silently.
    """
    make_key = getattr(cache, "make_key", None)
    if callable(make_key):
        for args in (
            (image_hashes, prompt, tool, options),
            (image_hashes, prompt, tool),
        ):
            try:
                return str(make_key(*args))
            except TypeError:  # signature drift between cache revisions
                continue
    payload = json.dumps(
        {
            "tool": tool,
            "prompt": prompt.model_dump(mode="json"),
            "options": options.model_dump(mode="json"),
            "images": dict(sorted(image_hashes.items())),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _build_cache_if_available(config: AppConfig) -> Any:
    """Build the process cache when ``percepta.cache`` exists (owned by the ``state`` agent)."""
    if not config.cache.enabled:
        return None
    try:
        from .cache import build_cache  # noqa: PLC0415 - optional, another agent owns it
    except ImportError:
        log.debug("percepta.cache is not available yet; caching disabled")
        return None
    try:
        return build_cache(config.cache)
    except Exception as exc:  # noqa: BLE001 - a broken cache must not break perception
        log.warning("failed to build cache: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------


def prepare_tools(registry: ToolRegistry, config: AppConfig) -> list[str]:
    """Load the tools serving will need; return the names that loaded.

    Shared startup step for :func:`percepta.service.app.create_app` and ``percepta probe``:
    first everything enabled under ``config.tools`` (:meth:`ToolRegistry.configure`), then the
    tools of the configured pipeline are warmed so a broken/missing model surfaces as
    ``error`` on ``/healthz`` *before* the first request. That is decision D9: serving the
    code-default pipeline without weights reports ``sam3`` as ``error`` and must not crash.

    Loading happens once per process (spec §5.4). Failures are logged and left on the
    registry entry's ``error`` field; they never raise.
    """
    loaded = list(registry.configure(config.tools))
    for name in config.pipeline.default:
        if name not in registry or not config.is_enabled(name):
            continue
        if registry.entry(name).instance is not None:
            if name not in loaded:
                loaded.append(name)
            continue
        try:
            registry.instance(name, config.tool_config(name))
            loaded.append(name)
        except ToolLoadError as exc:
            log.warning("pipeline tool %s is unavailable: %s", name, exc)
    return loaded


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class PipelineRunner:
    """Runs ``config.resolve_pipeline(options.tools)`` serially for one request (spec §5.3)."""

    def __init__(self, registry: ToolRegistry, config: AppConfig, cache: Any = None) -> None:
        self.registry = registry
        self.config = config
        self._cache = cache
        self._cache_resolved = cache is not None
        self._cache_warned = False
        self._closed = False

    # -- life cycle ------------------------------------------------------

    @property
    def cache(self) -> Any:
        """The configured cache, built lazily from ``config.cache`` when available."""
        if not self._cache_resolved:
            self._cache_resolved = True
            self._cache = _build_cache_if_available(self.config)
        return self._cache

    def close(self) -> None:
        """Release tool instances and drop the cache. Safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        try:
            self.registry.close_all()
        finally:
            cache, self._cache = self._cache, None
            close = getattr(cache, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:  # noqa: BLE001 - shutdown must not raise
                    log.warning("error closing cache: %s", exc)

    def __enter__(self) -> PipelineRunner:
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    # -- public API ------------------------------------------------------

    def perceive(self, request: PerceiveRequest) -> PerceiveResponse:
        """Decode, validate, run the pipeline, fuse, and assemble the response."""
        request_id = request.request_id or str(uuid.uuid4())
        images = self._decode_images(request)
        tools = self._resolve_pipeline(request)
        ctx = ToolContext(
            request_id=request_id,
            images=images,
            calibration=dict(request.calibration),
            context=dict(request.context),
            options=request.options,
            logger=log,
            cache=self.cache,
            extras={},
        )
        image_hashes = {camera: array_sha256(array) for camera, array in images.items()}

        percepts: list[Percept] = []
        warnings: list[str] = []
        timings: dict[str, float] = {}
        fused: Percept | None = None
        geometry_stage_ran = False

        for name in tools:
            upstream = list(percepts)
            ctx.extras["upstream"] = upstream  # D2: earlier percepts feed later tools
            produced, elapsed_ms = self._run_tool(
                name, ctx, request, image_hashes, upstream, warnings
            )
            timings[name] = round(timings.get(name, 0.0) + elapsed_ms, 3)
            if self._tool_kind(name) == "geometry":
                # D15: a geometry-kind stage *is* the response's `fused`; it is never kept in
                # `percepts` (§3.5 wants one percept per camera x prompt x tool) and fusion
                # must not run a second time.
                geometry_stage_ran = True
                fused, produced = self._take_geometry(fused, produced)
            percepts.extend(produced)

        percepts = self._postprocess(percepts, request, warnings)
        if request.options.fuse:
            if geometry_stage_ran:
                if fused is None:
                    _add_warning(warnings, "geometry_unavailable")
            else:
                fused = self._fuse(percepts, request, warnings, timings)
        self._collect_tool_warnings(percepts, fused, warnings)

        log.debug(
            "pipeline request done",
            extra={
                "request_id": request_id,
                "status": "ok",
                "latency_ms": sum(timings.values()),
            },
        )
        return PerceiveResponse(
            request_id=request_id,
            percepts=percepts,
            fused=fused,
            timings_ms=timings,
            warnings=warnings,
        )

    @staticmethod
    def _take_geometry(
        fused: Percept | None, produced: list[Percept]
    ) -> tuple[Percept | None, list[Percept]]:
        """Move a geometry stage's successful output into ``fused`` (decision D15).

        Errors from the geometry stage (``unavailable``/``invalid_request``) stay in
        ``percepts`` — a failed fusion must remain visible, never disappear into ``fused``.
        """
        candidates = [
            percept
            for percept in produced
            if percept.provenance.kind == "geometry" and percept.status == "ok"
        ]
        if not candidates:
            return fused, produced
        # A later geometry stage supersedes an earlier one.
        chosen = candidates[-1]
        return chosen, [percept for percept in produced if percept is not chosen]

    # -- request validation ----------------------------------------------

    def _decode_images(self, request: PerceiveRequest) -> dict[str, np.ndarray]:
        images: dict[str, np.ndarray] = {}
        for camera, ref in request.images.items():
            try:
                array = decode_image(ref)
            except Exception as exc:  # noqa: BLE001 - any decode failure is a bad request
                raise PipelineRequestError(
                    f"cannot decode image for camera {camera!r}: {exc}"
                ) from exc
            array = np.ascontiguousarray(array)
            if array.ndim != 3 or array.shape[2] != 3:
                raise PipelineRequestError(
                    f"image {camera!r} is not RGB (H, W, 3): shape {array.shape}"
                )
            calibration = request.calibration.get(camera)
            if calibration is not None:
                expected = (int(calibration.height), int(calibration.width))
                actual = (int(array.shape[0]), int(array.shape[1]))
                if actual != expected:
                    # spec §4.6: image (H, W) must match Calibration.height/width.
                    raise PipelineRequestError(
                        f"image {camera!r} shape (H, W) {actual} != calibration "
                        f"(H, W) {expected} (spec §4.6)"
                    )
            images[camera] = array
        return images

    def _resolve_pipeline(self, request: PerceiveRequest) -> list[str]:
        try:
            names = self.config.resolve_pipeline(request.options.tools)
        except ValueError as exc:
            raise PipelineRequestError(str(exc)) from exc
        if not names:
            raise PipelineRequestError("no tools left to run after applying config")
        for name in names:
            # spec §5.2: an unknown tool name is a caller error, not a tool failure. A name
            # that is configured but failed to register is handled as `unavailable`.
            if name not in self.registry and name not in self.config.tools:
                raise PipelineRequestError(f"unknown tool {name!r} (spec §5.2)")
        return names

    # -- one tool --------------------------------------------------------

    def _tool_kind(self, name: str) -> str | None:
        """The registry's recorded ``ToolKind`` for a loaded tool (``None`` if it never loaded)."""
        try:
            return self.registry.entry(name).kind
        except UnknownTool:
            return None

    def _run_tool(
        self,
        name: str,
        ctx: ToolContext,
        request: PerceiveRequest,
        image_hashes: dict[str, str],
        upstream: list[Percept],
        warnings: list[str],
    ) -> tuple[list[Percept], float]:
        started = time.perf_counter()
        try:
            tool = self.registry.instance(name, self.config.tool_config(name))
        except (UnknownTool, ToolLoadError) as exc:
            _add_warning(warnings, "tool_unavailable")
            log.warning("tool %s is unavailable: %s", name, exc)
            return (
                self._fail_percepts(
                    name,
                    ctx,
                    list(request.prompts),
                    version="0.0.0",
                    kind=self._kind_for(name),
                    message=f"{type(exc).__name__}: {exc}",
                    status="unavailable",
                    code="tool_unavailable",
                    started_at=started,
                ),
                _elapsed_ms(started),
            )

        version = str(getattr(tool, "version", "0.0.0"))
        try:
            caps = tool.capabilities()
        except Exception as exc:  # noqa: BLE001 - a broken declaration is unavailability
            _add_warning(warnings, "tool_unavailable")
            return (
                self._fail_percepts(
                    name,
                    ctx,
                    list(request.prompts),
                    version=version,
                    kind=self._kind_for(name),
                    message=f"capabilities() failed: {type(exc).__name__}: {exc}",
                    status="unavailable",
                    code="tool_unavailable",
                    started_at=started,
                ),
                _elapsed_ms(started),
            )

        if caps.prompts:
            supported = [prompt for prompt in request.prompts if prompt.type in caps.prompts]
            rejected = [prompt for prompt in request.prompts if prompt.type not in caps.prompts]
        else:
            # A tool that declares no prompt type is *prompt-agnostic* (geometry tools consume
            # upstream percepts and ignore `prompts` entirely, spec §5.3). Such a tool must
            # not turn every request into an `unsupported_prompt` Percept; only a tool that
            # declares a non-empty prompt set rejects the types outside it.
            supported = list(request.prompts)
            rejected = []

        out: list[Percept] = []
        for camera in sorted(ctx.images):
            for prompt in rejected:
                out.append(
                    self._rejected_prompt(
                        name, version, caps, camera, prompt, ctx, started_at=started
                    )
                )

        if supported:
            out.extend(
                self._call_tool(
                    name, version, caps, tool, ctx, supported, image_hashes, upstream, warnings
                )
            )
        return out, _elapsed_ms(started)

    def _call_tool(
        self,
        name: str,
        version: str,
        caps: ToolCapabilities,
        tool: Any,
        ctx: ToolContext,
        supported: list[Prompt],
        image_hashes: dict[str, str],
        upstream: list[Percept],
        warnings: list[str],
    ) -> list[Percept]:
        # The cache only applies to the first (image-conditioned) stage: a later stage also
        # depends on upstream percepts, which the spec §11.4 key does not cover.
        cache = self.cache if (ctx.options.cache and not upstream) else None

        cached: dict[int, list[Percept]] = {}
        misses: list[tuple[int, Prompt]] = []
        for index, prompt in enumerate(supported):
            hit = self._cache_get(cache, image_hashes, prompt, name, ctx.options) if cache else None
            if hit:
                cached[index] = hit
            else:
                misses.append((index, prompt))

        produced: list[Percept] = []
        if misses:
            try:
                with self.registry.lock_for(name):  # §11.2: per-tool inference lock
                    produced = _coerce_percepts(
                        tool.perceive(ctx, [prompt for _, prompt in misses])
                    )
            except UnsupportedPrompt as exc:
                _add_warning(warnings, "unsupported_prompt")
                return self._fail_percepts(
                    name,
                    ctx,
                    [prompt for _, prompt in misses],
                    version=version,
                    kind=caps.kind,
                    message="unsupported_prompt",
                    status="invalid_request",
                    code="unsupported_prompt",
                    extra={"detail": str(exc)},
                )
            except ToolUnavailable as exc:
                _add_warning(warnings, "tool_unavailable")
                return self._fail_percepts(
                    name,
                    ctx,
                    [prompt for _, prompt in misses],
                    version=version,
                    kind=caps.kind,
                    message=f"{type(exc).__name__}: {exc}",
                    status="unavailable",
                    code="tool_unavailable",
                )
            except ToolError as exc:
                # Any other declared tool failure is a tool problem, never `not_found`.
                _add_warning(warnings, "tool_unavailable")
                return self._fail_percepts(
                    name,
                    ctx,
                    [prompt for _, prompt in misses],
                    version=version,
                    kind=caps.kind,
                    message=f"{type(exc).__name__}: {exc}",
                    status="unavailable",
                    code="tool_unavailable",
                )

            if cache is not None:
                for _, prompt in misses:
                    group = [p for p in produced if p.provenance.prompt == prompt]
                    if group:
                        self._cache_put(cache, image_hashes, prompt, name, ctx.options, group)

        if not misses:
            return [p for index in range(len(supported)) for p in cached.get(index, [])]

        # Deterministic prompt-major order; anything the tool returned without a prompt in
        # provenance is appended in tool order rather than dropped.
        matched: set[int] = set()
        ordered: list[Percept] = []
        for index, prompt in enumerate(supported):
            if index in cached:
                ordered.extend(cached[index])
                continue
            group = [p for p in produced if p.provenance.prompt == prompt]
            for percept in group:
                matched.add(id(percept))
            ordered.extend(group)
        for percept in produced:
            if id(percept) not in matched:
                ordered.append(percept)
        return ordered

    # -- cache -----------------------------------------------------------

    def _cache_get(
        self,
        cache: Any,
        image_hashes: dict[str, str],
        prompt: Prompt,
        tool: str,
        options: PerceiveOptions,
    ) -> list[Percept] | None:
        try:
            key = _cache_key(cache, image_hashes, prompt, tool, options)
            value = cache.get(key)
        except Exception as exc:  # noqa: BLE001 - a broken cache must not fail a request
            self._warn_cache(exc)
            return None
        if not value:
            return None
        try:
            return [_set_cache_hit(percept) for percept in _coerce_percepts(value)]
        except Exception as exc:  # noqa: BLE001 - untrusted cache payload
            self._warn_cache(exc)
            return None

    def _cache_put(
        self,
        cache: Any,
        image_hashes: dict[str, str],
        prompt: Prompt,
        tool: str,
        options: PerceiveOptions,
        group: list[Percept],
    ) -> None:
        try:
            key = _cache_key(cache, image_hashes, prompt, tool, options)
            cache.put(key, list(group))
        except Exception as exc:  # noqa: BLE001
            self._warn_cache(exc)

    def _warn_cache(self, exc: Exception) -> None:
        if not self._cache_warned:
            self._cache_warned = True
            log.warning("cache disabled after error: %s", exc)
        self._cache = None
        self._cache_resolved = True

    # -- percept builders ------------------------------------------------

    def _kind_for(self, name: str) -> str:
        try:
            kind = self.registry.entry(name).kind
        except UnknownTool:
            kind = None
        if kind:
            return str(kind)
        # The tool never loaded, so its declared kind is unknown; keep the enum valid.
        return "fake" if name == "fake" else "segmenter"

    def _provenance(
        self,
        name: str,
        version: str,
        kind: str,
        started_at: float,
        camera: str | None,
        prompt: Prompt | None,
        ctx: ToolContext,
    ):
        images = {camera: ctx.images[camera]} if camera in ctx.images else {}
        calibration = ctx.calibration.get(camera) if camera is not None else None
        return build_provenance(
            tool=name,
            tool_version=version,
            kind=kind,
            started_at=started_at,
            camera=camera,
            prompt=prompt,
            images=images,
            calibration_id=(calibration.id if calibration is not None else None),
        )

    def _fail_percepts(
        self,
        name: str,
        ctx: ToolContext,
        prompts: list[Prompt],
        *,
        version: str,
        kind: str,
        message: str,
        status: str,
        code: str,
        started_at: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> list[Percept]:
        """One failure Percept per ``camera × prompt`` for a tool that could not deliver."""
        started = time.perf_counter() if started_at is None else started_at
        out: list[Percept] = []
        for camera in sorted(ctx.images):
            for prompt in prompts or [None]:
                extras = {"code": code, "tool": name}
                extras.update(extra or {})
                out.append(
                    make_percept(
                        percept_id=str(uuid.uuid4()),
                        status=status,
                        provenance=self._provenance(
                            name, version, kind, started, camera, prompt, ctx
                        ),
                        camera=camera,
                        message=message,
                        extras=extras,
                    )
                )
        return out

    def _rejected_prompt(
        self,
        name: str,
        version: str,
        caps: ToolCapabilities,
        camera: str,
        prompt: Prompt,
        ctx: ToolContext,
        *,
        started_at: float,
    ) -> Percept:
        """D1: an unsupported prompt is a Percept status, not a request failure."""
        return make_percept(
            percept_id=str(uuid.uuid4()),
            status="invalid_request",
            provenance=self._provenance(name, version, caps.kind, started_at, camera, prompt, ctx),
            camera=camera,
            message="unsupported_prompt",
            extras={
                "code": "unsupported_prompt",
                "tool": name,
                "prompt_type": prompt.type,
                "supported_prompts": sorted(caps.prompts),
            },
        )

    # -- post-processing --------------------------------------------------

    def _postprocess(
        self, percepts: list[Percept], request: PerceiveRequest, warnings: list[str]
    ) -> list[Percept]:
        """Apply §8.3 multi-instance disambiguation and mark ``ambiguous_instances``."""
        table_z = request.context.get("table_z")
        if not isinstance(table_z, (int, float)) or isinstance(table_z, bool):
            table_z = None
        out: list[Percept] = []
        for percept in percepts:
            if percept.status == "ok" and len(percept.observations) > 1:
                kept, ambiguous = disambiguate(
                    percept, context=request.context, table_z=table_z
                )
                if ambiguous:
                    _add_warning(warnings, "ambiguous_instances")
                if len(kept) != len(percept.observations):
                    percept = percept.model_copy(update={"observations": kept})
            out.append(percept)
        return out

    def _collect_tool_warnings(
        self, percepts: list[Percept], fused: Percept | None, warnings: list[str]
    ) -> None:
        candidates = list(percepts)
        if fused is not None:
            candidates.append(fused)  # the fused Percept carries its own extras["warnings"]
        for percept in candidates:
            if percept.status == "unavailable":
                _add_warning(warnings, "tool_unavailable")
            for warning in percept.extras.get("warnings") or []:
                if isinstance(warning, str):
                    _add_warning(warnings, warning)

    def _fuse(
        self,
        percepts: list[Percept],
        request: PerceiveRequest,
        warnings: list[str],
        timings: dict[str, float],
    ) -> Percept | None:
        """Run ``percepta.fusion.fuse`` when configured and importable (spec §5.3, §9)."""
        if not request.options.fuse or not percepts:
            return None
        try:
            from .fusion import fuse  # noqa: PLC0415 - optional module owned by `geometry`
        except ImportError:
            log.debug("percepta.fusion is not available yet")
            _add_warning(warnings, "geometry_unavailable")
            return None
        started = time.perf_counter()
        fused = fuse(
            percepts,
            request.calibration,
            request.context,
            request.options,
        )
        timings["fuse"] = _elapsed_ms(started)
        if fused is None:
            _add_warning(warnings, "geometry_unavailable")
            return None
        return fused if isinstance(fused, Percept) else Percept.model_validate(fused)
