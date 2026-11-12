"""``PipelineRunner`` unit tests (spec §5.3, §4.6, §11.4, §16.4; decisions D1/D2/D6/D12).

Covered here:

* D1 — an unsupported prompt is a Percept status, never a request failure;
* D6 — one tool call per request with every camera;
* D2 — earlier percepts reach later tools through ``ctx.extras["upstream"]``;
* §5.2 — an unknown tool name is a caller error; a *configured* tool that cannot load is
  ``unavailable`` (never ``not_found``);
* §4.6 — image/calibration size mismatch is a 400-class request error;
* §11.4 — cache miss/hit, the options-sensitivity of the key, and graceful degradation when
  the cache is missing or broken;
* §9 — fusion is optional and only wired when ``percepta.fusion`` imports.
"""

from __future__ import annotations

import sys
import time
import types
import uuid
from typing import Any

import numpy as np
import pytest

from percepta.config import AppConfig
from percepta.masks import encode_image
from percepta.pipeline import PipelineRequestError, PipelineRunner
from percepta.registry import ToolRegistry
from percepta.schema import (
    Geometry,
    PerceiveOptions,
    PerceiveRequest,
    Percept,
    Prompt,
    ToolCapabilities,
)
from percepta.tools.base import (
    BaseTool,
    ToolError,
    ToolUnavailable,
    UnsupportedPrompt,
    build_provenance,
    make_observation,
    make_percept,
)
from tests import synth

CAM = "cam_head"
TEXT = Prompt(type="text", text="red block")
BOX = Prompt(type="box", box=[0.5, 0.5, 0.25, 0.25])


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


class RecordingTool(BaseTool):
    """A minimal tool: one maskless observation per ``camera x prompt``, and a call log."""

    name = "recorder"
    version = "0.1.0"
    kind = "fake"

    def __init__(
        self,
        *,
        name: str = "recorder",
        prompts: tuple[str, ...] = ("text", "box", "point"),
        unavailable: bool = False,
        observations_per_prompt: int = 1,
    ) -> None:
        super().__init__()
        self.name = name
        self.supported_prompts = set(prompts)
        self.unavailable = unavailable
        self.observations_per_prompt = observations_per_prompt
        self.calls: list[list[Prompt]] = []
        self.upstreams: list[list[Percept]] = []

    def capabilities(self) -> ToolCapabilities:
        return ToolCapabilities(
            kind="fake",
            prompts=set(self.supported_prompts),
            masks=False,
            boxes=True,
            points=False,
        )

    def perceive(self, ctx, prompts):
        self.calls.append(list(prompts))
        self.upstreams.append(list(ctx.extras.get("upstream") or []))
        if self.unavailable:
            raise ToolUnavailable(f"no weights for {self.name}")
        started = time.perf_counter()
        out: list[Percept] = []
        for camera in sorted(ctx.images):
            for prompt in prompts:
                percept_id = str(uuid.uuid4())
                observations = [
                    make_observation(None, 0.5, version_percept_id=percept_id, index=index)
                    for index in range(self.observations_per_prompt)
                ]
                out.append(
                    make_percept(
                        percept_id=percept_id,
                        status="ok",
                        provenance=self._provenance(
                            ctx, started, camera=camera, prompt=prompt
                        ),
                        camera=camera,
                        observations=observations,
                    )
                )
        return out


class UnloadableTool(RecordingTool):
    """Importable, registered, but ``load()`` raises: it must surface as ``unavailable``."""

    name = "unloadable"

    def load(self, config: dict[str, Any]) -> None:
        raise RuntimeError("no weights on this host")


class GeometryStageTool(BaseTool):
    """A pipeline stage with ``kind="geometry"``, like the built-in ``multi_view_3d``."""

    name = "geom"
    version = "0.1.0"
    kind = "geometry"

    def __init__(self, *, returns_geometry: bool = True, unavailable: bool = False) -> None:
        super().__init__()
        self.returns_geometry = returns_geometry
        self.unavailable = unavailable
        self.calls: list[list[Prompt]] = []
        self.upstreams: list[list[Percept]] = []

    def capabilities(self) -> ToolCapabilities:
        return ToolCapabilities(
            kind="geometry",
            prompts=set(),
            masks=False,
            boxes=False,
            points=False,
            depth=True,
            multi_view=True,
            geometry=True,
        )

    def perceive(self, ctx, prompts):
        self.calls.append(list(prompts))
        self.upstreams.append(list(ctx.extras.get("upstream") or []))
        if self.unavailable:
            raise ToolUnavailable("geometry backend missing")
        if not self.returns_geometry:
            return []
        return [
            make_percept(
                percept_id=str(uuid.uuid4()),
                status="ok",
                provenance=build_provenance(
                    tool=self.name,
                    tool_version=self.version,
                    kind="geometry",
                    started_at=time.perf_counter(),
                ),
                geometry=Geometry(
                    frame="world",
                    centroid=[0.0, 0.0, 0.1],
                    uncertainty_m=0.02,
                    views_used=2,
                ),
            )
        ]


class DictCache:
    """Duck-typed cache with the ``AGENTS.md`` §4 shape (``get``/``put``/``make_key``)."""

    def __init__(self) -> None:
        self.store: dict[str, Any] = {}
        self.keys: list[str] = []

    def get(self, key: str) -> Any:
        return self.store.get(key)

    def put(self, key: str, value: Any) -> None:
        self.keys.append(key)
        self.store[key] = value

    @staticmethod
    def make_key(image_sha256, prompt, tool, options=None) -> str:
        images = sorted((image_sha256 or {}).items())
        masks = getattr(options, "masks", None)
        return f"{tool}|{prompt.model_dump_json()}|{images}|{masks}"


def config_for(
    default: list[str] | None = None,
    *,
    tools: dict[str, dict[str, Any]] | None = None,
    cache_enabled: bool = False,
) -> AppConfig:
    return AppConfig.model_validate(
        {
            "pipeline": {"default": default or ["fake"]},
            "tools": tools or {},
            "cache": {"enabled": cache_enabled},
            "metrics": {"enabled": False},
        }
    )


def make_runner(config: AppConfig, *tools: BaseTool, cache: Any = None) -> PipelineRunner:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool.name, tool)
    return PipelineRunner(registry, config, cache)


def request_for(
    prompts: list[Prompt],
    *,
    images: dict[str, np.ndarray] | None = None,
    calibration: dict[str, Any] | None = None,
    tools: list[str] | None = None,
    fuse: bool = False,
    masks: str = "rle",
    request_id: str | None = "req-pipeline",
    context: dict[str, Any] | None = None,
) -> PerceiveRequest:
    pictures = images if images is not None else {CAM: synth.red_square_scene()}
    calib = (
        calibration
        if calibration is not None
        else {camera: synth.make_calibration(camera) for camera in pictures}
    )
    return PerceiveRequest(
        request_id=request_id,
        images={camera: encode_image(image) for camera, image in pictures.items()},
        calibration=calib,
        prompts=prompts,
        context=dict(context or {}),
        options=PerceiveOptions(tools=tools, fuse=fuse, masks=masks),
    )


# ---------------------------------------------------------------------------
# D1 — capability gating never fails the request
# ---------------------------------------------------------------------------


def test_unsupported_prompt_becomes_a_percept_and_skips_the_tool() -> None:
    box_only = RecordingTool(name="box_only", prompts=("box",))
    runner = make_runner(config_for(["box_only"]), box_only)
    try:
        response = runner.perceive(request_for([TEXT], tools=["box_only"]))
    finally:
        runner.close()

    assert len(response.percepts) == 1
    percept = response.percepts[0]
    assert percept.status == "invalid_request"
    assert percept.message == "unsupported_prompt"
    assert percept.extras["code"] == "unsupported_prompt"
    assert percept.camera == CAM
    assert percept.provenance.prompt == TEXT
    assert box_only.calls == [], "the tool must not be called for a prompt it cannot handle"
    assert response.warnings == [], "an unsupported prompt must not fail the request"


def test_one_bad_prompt_does_not_spoil_a_good_one() -> None:
    box_only = RecordingTool(name="box_only", prompts=("box",))
    runner = make_runner(config_for(["box_only"]), box_only)
    try:
        response = runner.perceive(request_for([TEXT, BOX], tools=["box_only"]))
    finally:
        runner.close()

    statuses = [percept.status for percept in response.percepts]
    assert statuses.count("invalid_request") == 1
    assert statuses.count("ok") == 1
    assert len(box_only.calls) == 1
    assert [prompt.type for prompt in box_only.calls[0]] == ["box"]


def test_rejected_prompts_are_reported_per_camera() -> None:
    box_only = RecordingTool(name="box_only", prompts=("box",))
    runner = make_runner(config_for(["box_only"]), box_only)
    images = {"cam_head": synth.red_square_scene(), "cam_left": synth.red_square_scene()}
    try:
        response = runner.perceive(request_for([TEXT], images=images, tools=["box_only"]))
    finally:
        runner.close()
    assert {percept.camera for percept in response.percepts} == {"cam_head", "cam_left"}
    assert {percept.status for percept in response.percepts} == {"invalid_request"}
    _ = np  # keep numpy imported for the image fixtures above


def test_prompt_agnostic_tool_is_not_rejected() -> None:
    """A tool that declares no prompt types (e.g. a geometry tool) simply ignores prompts."""
    agnostic = RecordingTool(name="agnostic", prompts=())
    runner = make_runner(config_for(["agnostic"]), agnostic)
    try:
        response = runner.perceive(request_for([TEXT], tools=["agnostic"]))
    finally:
        runner.close()
    assert [percept.status for percept in response.percepts] == ["ok"]
    assert agnostic.calls and len(agnostic.calls[0]) == 1


# ---------------------------------------------------------------------------
# D6 — one call, every camera
# ---------------------------------------------------------------------------


def test_multi_camera_fan_out_uses_one_tool_call() -> None:
    runner = make_runner(config_for(["fake"]))
    images = {"cam_head": synth.red_square_scene(), "cam_left": synth.red_square_scene()}
    try:
        response = runner.perceive(request_for([TEXT], images=images, tools=["fake"]))
    finally:
        runner.close()

    assert len(response.percepts) == 2
    assert {percept.camera for percept in response.percepts} == {"cam_head", "cam_left"}
    for percept in response.percepts:
        assert percept.status == "ok"
        assert len(percept.observations) == 1
        observation = percept.observations[0]
        assert observation.area_px == 80 * 80
        assert observation.box_xyxy == [280.0, 200.0, 360.0, 280.0]
    assert set(response.timings_ms) == {"fake"}
    assert response.timings_ms["fake"] >= 0.0


# ---------------------------------------------------------------------------
# D2 — upstream percepts
# ---------------------------------------------------------------------------


def test_upstream_percepts_reach_later_tools() -> None:
    first = RecordingTool(name="first", prompts=("text",))
    second = RecordingTool(name="second", prompts=("text",))
    runner = make_runner(config_for(["first", "second"]), first, second)
    try:
        response = runner.perceive(request_for([TEXT], tools=["first", "second"]))
    finally:
        runner.close()

    assert first.upstreams == [[]], "the first tool has no upstream"
    assert len(second.upstreams) == 1
    upstream = second.upstreams[0]
    first_tool_percepts = [p for p in response.percepts if p.provenance.tool == "first"]
    assert [percept.percept_id for percept in upstream] == [
        percept.percept_id for percept in first_tool_percepts
    ]
    assert upstream, "the second tool must receive the first tool's percepts"
    assert all(percept.status == "ok" for percept in upstream)
    assert set(response.timings_ms) == {"first", "second"}


# ---------------------------------------------------------------------------
# Pipeline selection (§5.2, §5.3)
# ---------------------------------------------------------------------------


def test_options_tools_override_the_configured_pipeline() -> None:
    recorder = RecordingTool(name="recorder", prompts=("text",))
    runner = make_runner(config_for(["fake"]), recorder)
    try:
        response = runner.perceive(request_for([TEXT], tools=["recorder"]))
    finally:
        runner.close()
    assert recorder.calls, "the requested tool must run"
    assert set(response.timings_ms) == {"recorder"}


def test_unknown_tool_name_is_a_request_error() -> None:
    runner = make_runner(config_for(["fake"]))
    try:
        with pytest.raises(PipelineRequestError) as excinfo:
            runner.perceive(request_for([TEXT], tools=["does_not_exist"]))
    finally:
        runner.close()
    assert excinfo.value.status_code == 400
    assert excinfo.value.code == "invalid_request"
    assert "does_not_exist" in excinfo.value.message


def test_configured_but_unregistered_tool_is_unavailable() -> None:
    config = config_for(tools={"ghost": {"enabled": True}})
    runner = make_runner(config)
    try:
        response = runner.perceive(request_for([TEXT], tools=["ghost"]))
    finally:
        runner.close()
    assert response.percepts[0].status == "unavailable"
    assert "tool_unavailable" in response.warnings


def test_tool_that_fails_to_load_is_unavailable() -> None:
    config = config_for(tools={"unloadable": {"enabled": True}})
    registry = ToolRegistry()
    registry.register("unloadable", UnloadableTool)
    runner = PipelineRunner(registry, config)
    try:
        response = runner.perceive(request_for([TEXT], tools=["unloadable"]))
    finally:
        runner.close()
    percept = response.percepts[0]
    assert percept.status == "unavailable"
    assert percept.status != "not_found"
    assert "no weights" in (percept.message or "")
    assert percept.extras["code"] == "tool_unavailable"
    assert "tool_unavailable" in response.warnings


def test_tool_unavailable_at_inference_is_unavailable() -> None:
    tool = RecordingTool(name="broken_runtime", unavailable=True)
    runner = make_runner(config_for(["broken_runtime"]), tool)
    try:
        response = runner.perceive(request_for([TEXT], tools=["broken_runtime"]))
    finally:
        runner.close()
    assert response.percepts[0].status == "unavailable"
    assert "no weights" in (response.percepts[0].message or "")
    assert tool.calls, "the tool was called and failed"


def test_every_requested_tool_disabled_is_a_request_error() -> None:
    config = config_for(tools={"fake": {"enabled": False}})
    runner = make_runner(config)
    try:
        with pytest.raises(PipelineRequestError):
            runner.perceive(request_for([TEXT], tools=["fake"]))
    finally:
        runner.close()


# ---------------------------------------------------------------------------
# spec §4.6 — image/calibration size check
# ---------------------------------------------------------------------------


def test_image_calibration_size_mismatch_is_invalid_request() -> None:
    tool = RecordingTool(name="recorder", prompts=("text",))
    runner = make_runner(config_for(["recorder"]), tool)
    calibration = {CAM: synth.make_calibration(CAM, width=320, height=240)}
    try:
        with pytest.raises(PipelineRequestError) as excinfo:
            runner.perceive(request_for([TEXT], calibration=calibration, tools=["recorder"]))
    finally:
        runner.close()
    assert excinfo.value.status_code == 400
    assert "calibration" in excinfo.value.message
    assert tool.calls == [], "size validation happens before any tool call"


def test_undecodable_image_is_invalid_request() -> None:
    runner = make_runner(config_for(["fake"]))
    request = request_for([TEXT], tools=["fake"])
    request.images[CAM] = request.images[CAM].model_copy(update={"data": "AAAA"})
    try:
        with pytest.raises(PipelineRequestError) as excinfo:
            runner.perceive(request)
    finally:
        runner.close()
    assert excinfo.value.status_code == 400
    assert "decode" in excinfo.value.message


# ---------------------------------------------------------------------------
# Warnings, timings, request ids
# ---------------------------------------------------------------------------


def test_ambiguous_instances_are_warned_about() -> None:
    image = synth.blank_image()
    image[100:180, 200:280] = synth.RED
    image[300:380, 400:480] = synth.RED
    runner = make_runner(config_for(["fake"]))
    try:
        response = runner.perceive(
            request_for(
                [Prompt(type="text", text="red block", top_k=3)],
                images={CAM: image},
                calibration={},
                tools=["fake"],
            )
        )
    finally:
        runner.close()
    assert len(response.percepts) == 1
    assert len(response.percepts[0].observations) == 2
    assert "ambiguous_instances" in response.warnings


def test_request_id_is_echoed_or_generated() -> None:
    runner = make_runner(config_for(["fake"]))
    try:
        echoed = runner.perceive(request_for([TEXT], tools=["fake"], request_id="req-echo"))
        generated = runner.perceive(request_for([TEXT], tools=["fake"], request_id=None))
    finally:
        runner.close()
    assert echoed.request_id == "req-echo"
    assert generated.request_id and generated.request_id != "req-echo"


# ---------------------------------------------------------------------------
# §9 / fusion — optional module
# ---------------------------------------------------------------------------


def test_missing_fusion_module_adds_geometry_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "percepta.fusion", None)
    runner = make_runner(config_for(["fake"]))
    try:
        response = runner.perceive(request_for([TEXT], tools=["fake"], fuse=True))
    finally:
        runner.close()
    assert response.fused is None
    assert "geometry_unavailable" in response.warnings


def test_fusion_is_wired_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = make_percept(
        percept_id="fused-percept",
        status="ok",
        provenance=build_provenance(
            tool="stub", tool_version="0", kind="geometry", started_at=time.perf_counter()
        ),
    )
    stub = types.ModuleType("percepta.fusion")
    stub.fuse = lambda percepts, calibration, context, options=None: expected
    monkeypatch.setitem(sys.modules, "percepta.fusion", stub)

    runner = make_runner(config_for(["fake"]))
    try:
        response = runner.perceive(request_for([TEXT], tools=["fake"], fuse=True))
    finally:
        runner.close()
    assert response.fused is expected
    assert "fuse" in response.timings_ms


def test_fuse_false_skips_fusion(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = types.ModuleType("percepta.fusion")
    stub.fuse = lambda *args, **kwargs: pytest.fail("fuse() must not be called")
    monkeypatch.setitem(sys.modules, "percepta.fusion", stub)
    runner = make_runner(config_for(["fake"]))
    try:
        response = runner.perceive(request_for([TEXT], tools=["fake"], fuse=False))
    finally:
        runner.close()
    assert response.fused is None
    assert "fuse" not in response.timings_ms


# ---------------------------------------------------------------------------
# D15 — a geometry-kind stage is the response's `fused`, never a Percept
# ---------------------------------------------------------------------------


def _fail_fusion(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = types.ModuleType("percepta.fusion")
    stub.fuse = lambda *args, **kwargs: pytest.fail(
        "fusion.fuse must not run when a geometry-kind tool already ran"
    )
    monkeypatch.setitem(sys.modules, "percepta.fusion", stub)


def test_geometry_stage_output_becomes_fused_and_is_not_duplicated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fail_fusion(monkeypatch)
    masker = RecordingTool(name="masker", prompts=("text",))
    geometry = GeometryStageTool()
    runner = make_runner(config_for(["masker", "geom"]), masker, geometry)
    try:
        response = runner.perceive(request_for([TEXT], tools=["masker", "geom"], fuse=True))
    finally:
        runner.close()

    assert response.fused is not None
    assert response.fused.provenance.tool == "geom"
    assert response.fused.provenance.kind == "geometry"
    assert response.fused.geometry is not None
    assert response.fused.geometry.views_used == 2
    # No camera-less, observation-less geometry Percept leaks into `percepts` (§3.5).
    assert [percept.provenance.kind for percept in response.percepts] == ["fake"]
    assert [percept.camera for percept in response.percepts] == [CAM]
    assert "fuse" not in response.timings_ms
    assert set(response.timings_ms) == {"masker", "geom"}
    assert geometry.upstreams and geometry.upstreams[0], "the geometry stage saw upstream"


def test_geometry_stage_without_result_warns_once(monkeypatch: pytest.MonkeyPatch) -> None:
    _fail_fusion(monkeypatch)
    masker = RecordingTool(name="masker", prompts=("text",))
    geometry = GeometryStageTool(returns_geometry=False)
    runner = make_runner(config_for(["masker", "geom"]), masker, geometry)
    try:
        response = runner.perceive(request_for([TEXT], tools=["masker", "geom"], fuse=True))
    finally:
        runner.close()
    assert response.fused is None
    assert "geometry_unavailable" in response.warnings
    assert len(response.percepts) == 1


def test_geometry_stage_failure_stays_in_percepts(monkeypatch: pytest.MonkeyPatch) -> None:
    _fail_fusion(monkeypatch)
    masker = RecordingTool(name="masker", prompts=("text",))
    geometry = GeometryStageTool(unavailable=True)
    runner = make_runner(config_for(["masker", "geom"]), masker, geometry)
    try:
        response = runner.perceive(request_for([TEXT], tools=["masker", "geom"], fuse=True))
    finally:
        runner.close()
    assert response.fused is None
    geometry_percepts = [p for p in response.percepts if p.provenance.tool == "geom"]
    assert len(geometry_percepts) == 1
    assert geometry_percepts[0].status == "unavailable"
    assert geometry_percepts[0].camera == CAM
    assert "tool_unavailable" in response.warnings


def test_geometry_load_failure_keeps_declared_kind_and_never_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    from percepta.tools.fake import FakeTool

    class BrokenGeometry(GeometryStageTool):
        def load(self, config):
            raise ToolUnavailable("geometry checkpoint missing")

    _fail_fusion(monkeypatch)
    registry = ToolRegistry()
    registry.register("fake", FakeTool)
    registry.register("geom", BrokenGeometry)
    runner = PipelineRunner(registry, config_for(["fake", "geom"]))
    try:
        response = runner.perceive(request_for([TEXT], tools=["fake", "geom"], fuse=True))
    finally:
        runner.close()
    assert response.fused is None
    assert response.percepts[0].observations[0].mask is not None
    failed = [percept for percept in response.percepts if percept.provenance.tool == "geom"]
    assert len(failed) == 1
    assert failed[0].status == "unavailable"
    assert failed[0].provenance.kind == "geometry"
    assert "tool_unavailable" in response.warnings
    assert "geometry_unavailable" in response.warnings
    assert "fuse" not in response.timings_ms


def test_two_cameras_with_a_geometry_stage_keep_one_percept_per_camera(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fail_fusion(monkeypatch)
    masker = RecordingTool(name="masker", prompts=("text",))
    geometry = GeometryStageTool()
    runner = make_runner(config_for(["masker", "geom"]), masker, geometry)
    images = {"cam_head": synth.red_square_scene(), "cam_left": synth.red_square_scene()}
    try:
        response = runner.perceive(request_for([TEXT], images=images, tools=["masker", "geom"]))
    finally:
        runner.close()
    assert len(response.percepts) == 2, "2 cameras x 1 prompt, no geometry entry"
    assert response.fused is not None
    assert all(percept.provenance.kind != "geometry" for percept in response.percepts)


# ---------------------------------------------------------------------------
# §11.4 — cache
# ---------------------------------------------------------------------------


def test_second_identical_request_is_a_cache_hit() -> None:
    tool = RecordingTool(name="recorder", prompts=("text",))
    cache = DictCache()
    runner = make_runner(config_for(["recorder"]), tool, cache=cache)
    request = request_for([TEXT], tools=["recorder"])
    try:
        first = runner.perceive(request)
        second = runner.perceive(request)
    finally:
        runner.close()

    assert len(tool.calls) == 1, "the cached tool call must not run twice"
    assert [percept.provenance.cache_hit for percept in first.percepts] == [False]
    assert [percept.provenance.cache_hit for percept in second.percepts] == [True]
    # A hit does not change the payload.
    assert second.percepts[0].percept_id == first.percepts[0].percept_id
    assert second.percepts[0].status == first.percepts[0].status
    assert second.percepts[0].observations[0].score == first.percepts[0].observations[0].score
    assert len(cache.keys) == 1


def test_cache_key_distinguishes_mask_encoding() -> None:
    tool = RecordingTool(name="recorder", prompts=("text",))
    cache = DictCache()
    runner = make_runner(config_for(["recorder"]), tool, cache=cache)
    try:
        runner.perceive(request_for([TEXT], tools=["recorder"], masks="rle"))
        runner.perceive(request_for([TEXT], tools=["recorder"], masks="png"))
    finally:
        runner.close()
    assert len(tool.calls) == 2, "options.masks changes the payload, so the key must change"
    assert len(cache.keys) == 2


@pytest.mark.parametrize("mutate_hit", [False, True])
def test_cached_results_are_isolated_from_returned_nested_mutation(mutate_hit: bool) -> None:
    runner = make_runner(config_for(["fake"]), cache=DictCache())
    request = request_for([TEXT], tools=["fake"])
    try:
        response = runner.perceive(request)
        if mutate_hit:
            response = runner.perceive(request)
        percept = response.percepts[0]
        expected = percept.model_dump(exclude={"provenance": {"cache_hit"}})
        percept.observations[0].area_px = 9999
        percept.observations[0].mask.counts[0] = 9999
        percept.observations[0].attributes["changed"] = True
        percept.extras["changed"] = {"nested": [1]}
        fresh = runner.perceive(request).percepts[0]
    finally:
        runner.close()
    assert fresh.model_dump(exclude={"provenance": {"cache_hit"}}) == expected
    assert fresh.provenance.cache_hit is True


@pytest.mark.parametrize("cache_state", ["disabled", "cold", "hot", "mixed"])
def test_repeated_prompts_preserve_occurrences_without_multiplying_results(cache_state: str) -> None:
    tool = RecordingTool()
    cache = None if cache_state == "disabled" else DictCache()
    runner = make_runner(config_for([tool.name]), tool, cache=cache)
    other = Prompt(type="text", text="blue block")
    images = {CAM: synth.red_square_scene(), "cam_left": synth.red_square_scene()}
    prompts = [TEXT, other, TEXT]
    try:
        if cache_state in {"hot", "mixed"}:
            warm = prompts if cache_state == "hot" else [TEXT]
            runner.perceive(request_for(warm, images=images, tools=[tool.name]))
        response = runner.perceive(request_for(prompts, images=images, tools=[tool.name]))
    finally:
        runner.close()
    assert [(p.provenance.prompt.text, p.camera) for p in response.percepts] == [
        (prompt.text, camera) for prompt in prompts for camera in sorted(images)
    ]
    assert len({p.model_dump_json() for p in response.percepts}) >= 4
    assert all(len(call) == len({prompt.model_dump_json() for prompt in call}) for call in tool.calls)
    response.percepts[0].observations[0].area_px = 123
    assert response.percepts[4].observations[0].area_px != 123


@pytest.mark.parametrize("error", [UnsupportedPrompt, ToolUnavailable, ToolError])
@pytest.mark.parametrize("warm_cache", [False, True])
def test_failed_cache_misses_preserve_cached_and_repeated_prompt_results(
    monkeypatch: pytest.MonkeyPatch, error: type[ToolError], warm_cache: bool,
) -> None:
    tool, cache = RecordingTool(), DictCache()
    runner = make_runner(config_for([tool.name]), tool, cache=cache)
    other = Prompt(type="text", text="blue block")
    images = {CAM: synth.red_square_scene(), "cam_left": synth.red_square_scene()}
    prompts = [TEXT, other, other, TEXT]
    request = request_for(prompts, images=images, tools=[tool.name])
    original_perceive, failed_calls = tool.perceive, []

    def fail(_ctx, received):
        failed_calls.append(received)
        raise error("backend rejected this request")

    try:
        if warm_cache:
            runner.perceive(request_for([TEXT], images=images, tools=[tool.name]))
        monkeypatch.setattr(tool, "perceive", fail)
        response = runner.perceive(request)
        assert [(p.provenance.prompt.text, p.camera) for p in response.percepts] == [
            (prompt.text, camera) for prompt in prompts for camera in sorted(images)
        ]
        failed_status = "invalid_request" if error is UnsupportedPrompt else "unavailable"
        assert [p.status for p in response.percepts] == [
            "ok" if warm_cache and prompt == TEXT else failed_status for prompt in prompts for _ in images
        ]
        assert [p.provenance.cache_hit for p in response.percepts] == [
            warm_cache and prompt == TEXT for prompt in prompts for _ in images
        ]
        assert len(failed_calls) == 1
        assert len(failed_calls[0]) == (1 if warm_cache else 2)
        assert len(cache.keys) == int(warm_cache), "failed results must never enter the cache"
        response.percepts[2].extras["changed"] = True
        assert "changed" not in response.percepts[4].extras
        expected_warning = "unsupported_prompt" if error is UnsupportedPrompt else "tool_unavailable"
        assert expected_warning in response.warnings

        monkeypatch.setattr(tool, "perceive", original_perceive)
        recovered = runner.perceive(request)
        assert all(p.status == "ok" for p in recovered.percepts)
        assert [p.provenance.cache_hit for p in recovered.percepts] == [
            warm_cache and prompt == TEXT for prompt in prompts for _ in images
        ]
    finally:
        runner.close()


def test_broken_cache_degrades_to_no_cache() -> None:
    class ExplodingCache(DictCache):
        def get(self, key: str) -> Any:
            raise RuntimeError("cache backend down")

    tool = RecordingTool(name="recorder", prompts=("text",))
    runner = make_runner(config_for(["recorder"]), tool, cache=ExplodingCache())
    try:
        response = runner.perceive(request_for([TEXT], tools=["recorder"]))
        assert runner.cache is None, "the pipeline disables a cache that raises"
        second = runner.perceive(request_for([TEXT], tools=["recorder"]))
    finally:
        runner.close()
    assert response.percepts[0].status == "ok"
    assert second.percepts[0].status == "ok"
    assert len(tool.calls) == 2, "a broken cache must not lose results"


def test_missing_cache_module_still_perceives(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "percepta.cache", None)
    runner = make_runner(config_for(["fake"], cache_enabled=True))
    try:
        assert runner.cache is None
        response = runner.perceive(request_for([TEXT], tools=["fake"]))
    finally:
        runner.close()
    assert response.percepts[0].status == "ok"


def test_disabled_cache_still_perceives() -> None:
    runner = make_runner(config_for(["fake"], cache_enabled=False))
    try:
        assert runner.cache is None
        response = runner.perceive(request_for([TEXT], tools=["fake"]))
    finally:
        runner.close()
    assert response.percepts[0].status == "ok"
    assert response.percepts[0].provenance.cache_hit is False


def test_enabled_cache_is_built_from_config() -> None:
    runner = make_runner(config_for(["fake"], cache_enabled=True))
    try:
        cache = runner.cache
        assert cache is not None
        first = runner.perceive(request_for([TEXT], tools=["fake"]))
        second = runner.perceive(request_for([TEXT], tools=["fake"]))
    finally:
        runner.close()
    assert first.percepts[0].provenance.cache_hit is False
    assert second.percepts[0].provenance.cache_hit is True


# ---------------------------------------------------------------------------
# Life cycle
# ---------------------------------------------------------------------------


def test_close_is_idempotent_and_releases_tools() -> None:
    registry = ToolRegistry()
    runner = PipelineRunner(registry, config_for(["fake"]))
    runner.perceive(request_for([TEXT], tools=["fake"]))
    assert registry.entry("fake").instance is not None
    runner.close()
    runner.close()
    assert registry.entry("fake").instance is None
    assert registry.entry("fake").status == "lazy"
