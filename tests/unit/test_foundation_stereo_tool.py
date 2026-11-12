"""FoundationStereo contracts with real geometry, asset hashing, CPU workers and HTTP."""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from unittest.mock import Mock, create_autospec

import numpy as np
import pytest
from fastapi.testclient import TestClient

from percepta.backends.base import BackendRequestError, BackendUnavailable
from percepta.backends.foundation_stereo_backend import FoundationStereoBackend
from percepta.config import AppConfig
from percepta.masks import encode_image
from percepta.registry import ToolRegistry
from percepta.schema import (
    Calibration,
    PerceiveOptions,
    PerceiveRequest,
    PerceiveResponse,
    Prompt,
    ToolContext,
)
from percepta.service.app import create_app
from percepta.tools import stereo_3d as stereo
from percepta.tools.base import ToolUnavailable

cv2 = pytest.importorskip("cv2", reason="FoundationStereo geometry requires optional OpenCV")

ROOT = Path(__file__).resolve().parents[2]
FACTORY = "tests.unit.foundation_stereo_support:FakeStereoAdapter"
WIDTH, HEIGHT = 96, 64
FOCAL, BASELINE, DISPARITY = 80.0, 0.2, 8.0
CHECKPOINT_BYTES = b"test-only FoundationStereo checkpoint\x00\x01\xff"
CONFIG_BYTES = b"valid_iters: 7\nmax_disp: 192\n"


@pytest.fixture
def deployment(tmp_path):
    checkpoint = tmp_path / "model.pth"
    model_config = tmp_path / "custom.yaml"
    checkpoint.write_bytes(CHECKPOINT_BYTES)
    model_config.write_bytes(CONFIG_BYTES)
    return {
        "stereo_backend": "foundation_stereo",
        "python": sys.executable,
        "source_dir": str(tmp_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(CHECKPOINT_BYTES).hexdigest(),
        "model_config": str(model_config),
        "model_config_sha256": hashlib.sha256(CONFIG_BYTES).hexdigest(),
        "verify_weights": "always",
        "hf_cache": str(tmp_path / "hf-cache"),
        "valid_iters": 7,
        "cuda_visible_devices": "0",
        "startup_timeout_s": 10,
        "timeout_s": 5,
    }


@pytest.fixture
def pair():
    """OpenCV +Z cameras: Z=f*B/d=2 m; p_world=T_world_cam @ p_cam_h."""
    rng = np.random.default_rng(37)
    images = {
        name: rng.integers(0, 256, (HEIGHT, WIDTH, 3), dtype=np.uint8)
        for name in ("left", "right")
    }
    # FakeStereoAdapter returns left.R - right.R + offset, not learned disparity.
    images["left"][..., 0] = 28
    images["right"][..., 0] = 20
    calibration = {}
    for name, x in (("left", 0), ("right", BASELINE)):
        transform = np.eye(4)
        transform[0, 3] = x
        calibration[name] = Calibration(
            camera=name,
            width=WIDTH,
            height=HEIGHT,
            K=[[FOCAL, 0, WIDTH / 2], [0, FOCAL, HEIGHT / 2], [0, 0, 1]],
            T_world_cam=transform.tolist(),
            id=f"test-{name}",
        )
    return images, calibration


def _context(pair, **context):
    images, calibration = pair
    return ToolContext(
        request_id="foundation-tool-test",
        images=images,
        calibration=calibration,
        context=context,
        options=PerceiveOptions(cache=False),
    )


def _scene_prompts():
    return [Prompt(type="text", text="scene")]


@pytest.fixture
def tool():
    instance = stereo.Stereo3DTool()
    yield instance
    instance.close()


@pytest.fixture(autouse=True)
def no_sgbm(monkeypatch):
    matcher = Mock(side_effect=AssertionError("FoundationStereo must never fall back to SGBM"))
    monkeypatch.setattr(stereo, "_sgbm_disparity", matcher)
    return matcher


@pytest.fixture
def stub_backend(monkeypatch):
    backend = create_autospec(FoundationStereoBackend, instance=True)
    backend.infer_pair.side_effect = lambda left, right: np.full(
        left.shape[:2], DISPARITY, dtype=np.float32
    )
    backend.describe.return_value = {
        "backend": "foundation_stereo", "device": "cpu", "running": True, "valid_iters": 7,
    }
    constructor = Mock(return_value=backend)
    monkeypatch.setattr(stereo, "FoundationStereoBackend", constructor)
    return constructor, backend


@pytest.fixture
def cpu_backend_factory(monkeypatch):
    """Wrap only construction; the real resident process and wire protocol are unchanged."""
    existing = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, (str(ROOT), existing))))
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    backends = []

    def install(mode="normal"):
        def construct(**kwargs):
            backend = FoundationStereoBackend(
                **kwargs, backend_factory=FACTORY, backend_args={"mode": mode, "offset": 0.0}
            )
            backends.append(backend)
            return backend

        constructor = Mock(side_effect=construct)
        monkeypatch.setattr(stereo, "FoundationStereoBackend", constructor)
        return constructor, backends

    yield install
    for backend in backends:
        backend.close()
        assert backend._process is None
        if backend._stderr_thread is not None:
            assert not backend._stderr_thread.is_alive()


def _assert_metric_percept(percept):
    assert percept.status == "ok", percept.message
    assert percept.observations == []  # scene geometry, not prompt segmentation
    assert percept.geometry.frame == "world"
    assert percept.geometry.method == "stereo_foundation_stereo"
    assert percept.geometry.views_used == 2
    # Sign is essential: the fixture's original optical axes point along world +Z.
    assert percept.geometry.centroid[2] == pytest.approx(2.0)
    assert percept.geometry.bbox3d_min[2] == pytest.approx(2.0)
    assert percept.geometry.bbox3d_max[2] == pytest.approx(2.0)
    assert percept.geometry.top_z == pytest.approx(2.0)
    assert percept.geometry.uncertainty_m == pytest.approx(2.0**2 / (FOCAL * BASELINE))
    assert percept.extras["backend"] == "foundation_stereo"
    assert percept.extras["rectified"] is True
    assert percept.extras["focal_rect_px"] == pytest.approx(FOCAL)
    assert percept.extras["baseline_m"] == pytest.approx(BASELINE)
    assert percept.extras["median_depth_m"] == pytest.approx(2.0)
    assert percept.extras["disparity_p02"] == pytest.approx(DISPARITY)
    assert percept.extras["disparity_p98"] == pytest.approx(DISPARITY)
    assert percept.extras["points"] == HEIGHT * (WIDTH - int(DISPARITY))
    assert percept.extras["valid_fraction"] == pytest.approx((WIDTH - DISPARITY) / WIDTH)
    assert percept.extras["geometry_scope"] == "matched_visible_surface"
    assert "num_disparities" not in percept.extras
    assert percept.provenance.tool == "stereo_3d"
    assert percept.provenance.kind == "geometry"
    assert percept.provenance.model == "FoundationStereo"
    assert percept.provenance.model_sha256 == hashlib.sha256(CHECKPOINT_BYTES).hexdigest()
    assert percept.extras["model_config_sha256"] == hashlib.sha256(CONFIG_BYTES).hexdigest()
    assert percept.provenance.camera == "left"
    assert percept.provenance.calibration_id == "test-left"
    assert set(percept.provenance.image_sha256) == {"left", "right"}
    assert percept.provenance.prompt is None
    percept.model_dump_json()


@pytest.mark.parametrize("explicit_config", [True, False], ids=["explicit-cfg", "sibling-cfg"])
@pytest.mark.parametrize("declare_hashes", [True, False], ids=["verified", "observed"])
def test_metric_geometry_and_exact_asset_hashes(
    tool, deployment, pair, stub_backend, explicit_config, declare_hashes
):
    constructor, backend = stub_backend
    expected_config = Path(deployment["model_config"])
    if not explicit_config:
        expected_config = Path(deployment["checkpoint"]).with_name("cfg.yaml")
        Path(deployment.pop("model_config")).rename(expected_config)
    if not declare_hashes:
        deployment.pop("checkpoint_sha256")
        deployment.pop("model_config_sha256")
        deployment["verify_weights"] = "if-declared"
    tool.load(deployment)
    constructor.assert_called_once()
    assert constructor.call_args.kwargs["checkpoint"] == deployment["checkpoint"]
    assert constructor.call_args.kwargs["model_config"] == str(expected_config)
    backend.start.assert_called_once_with()
    backend.infer_pair.assert_not_called()  # load eagerly starts, but does not infer

    prompts = _scene_prompts() + [Prompt(type="text", text="another scene description")]
    percepts = tool.perceive(_context(pair), prompts)
    assert len(percepts) == 1  # not one result per prompt or camera
    _assert_metric_percept(percepts[0])
    assert percepts[0].extras["runtime"] == backend.describe.return_value
    assert tool.capabilities().prompts == set()
    backend.infer_pair.assert_called_once()


def _untrusted_context(deployment):
    parent = Path(deployment["checkpoint"]).parent / "request-controlled"
    return {
        "stereo_backend": "foundation_stereo",
        "python": str(parent / "python"),
        "checkpoint": str(parent / "checkpoint.pth"),
        "checkpoint_sha256": "0" * 64,
        "source_dir": str(parent / "source"),
        "model_config": str(parent / "cfg.yaml"),
        "model_config_sha256": "0" * 64,
        "hf_cache": str(parent / "cache"),
        "cuda_visible_devices": "99",
        "valid_iters": -1,
        "startup_timeout_s": 0,
        "timeout_s": 0,
        "verify_weights": "never",
        "backend_factory": "request_controlled:must_not_import",
        "backend_args": {"mode": "crash"},
    }


def test_context_cannot_reconfigure_or_reload_a_resident_tool(
    tool, deployment, pair, stub_backend
):
    constructor, backend = stub_backend
    tool.load(deployment)
    for context in ({}, _untrusted_context(deployment)):
        ctx = _context(pair, **context)
        _assert_metric_percept(tool.perceive(ctx, _scene_prompts())[0])
        assert ctx.context == context
    constructor.assert_called_once_with(
        **{key: deployment[key] for key in (
            "python", "source_dir", "checkpoint", "model_config", "hf_cache", "valid_iters",
            "cuda_visible_devices", "startup_timeout_s", "timeout_s",
        )}
    )
    backend.start.assert_called_once_with()
    assert tool.config == deployment
    assert backend.infer_pair.call_count == 2
    for call in backend.infer_pair.call_args_list:
        assert len(call.args) == 2 and call.kwargs == {}
        np.testing.assert_array_equal(call.args[0], pair[0]["left"])
        np.testing.assert_array_equal(call.args[1], pair[0]["right"])
    tool.close()
    tool.close()
    backend.close.assert_called_once_with()
    assert tool._backend is None
    with pytest.raises(ToolUnavailable, match="not loaded"):
        tool.perceive(_context(pair), _scene_prompts())
    assert constructor.call_count == 1


def test_undeployed_foundation_cannot_be_loaded_from_request_context(
    tool, deployment, pair, stub_backend
):
    constructor, backend = stub_backend
    tool.load({"stereo_backend": "sgbm"})
    # Even entirely valid executable/asset paths in a request are not deployment authority.
    with pytest.raises(ToolUnavailable, match="not loaded"):
        tool.perceive(_context(pair, **deployment), _scene_prompts())
    constructor.assert_not_called()
    backend.start.assert_not_called()
    backend.infer_pair.assert_not_called()


def test_foundation_receives_rectified_rgb_and_uses_rectified_mask_support(
    tool, deployment, pair, stub_backend
):
    """Independent OpenCV oracle: p_world=R_world_left @ R1.T @ p_rect + C_left."""
    _, backend = stub_backend
    images, calibration = pair
    rotation = cv2.Rodrigues(np.array([0.07, -0.12, 0.05]))[0]
    centre = np.array([0.2, -0.1, 0.3])
    left_transform, right_transform = np.eye(4), np.eye(4)
    left_transform[:3, :3], left_transform[:3, 3] = rotation, centre
    right_transform[:3, :3] = rotation @ cv2.Rodrigues(np.array([0.025, -0.035, 0.02]))[0]
    right_transform[:3, 3] = centre + rotation @ np.array([0.18, 0.015, 0.01])
    calibration["left"].T_world_cam = left_transform.tolist()
    calibration["right"].T_world_cam = right_transform.tolist()
    calibration["right"].K = [[75, 0, 44], [0, 78, 30], [0, 0, 1]]
    Ka, Kb = (
        np.array(calibration[name].K, dtype=np.float64) for name in ("left", "right")
    )
    R_ab = right_transform[:3, :3].T @ rotation
    t_ab = right_transform[:3, :3].T @ (centre - right_transform[:3, 3])
    R1, R2, P1, P2, *_ = cv2.stereoRectify(
        Ka, np.zeros(5), Kb, np.zeros(5), (WIDTH, HEIGHT), R_ab, t_ab.reshape(3, 1),
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
    )
    maps = [
        cv2.initUndistortRectifyMap(K, None, R, P, (WIDTH, HEIGHT), cv2.CV_32FC1)
        for K, R, P in ((Ka, R1, P1), (Kb, R2, P2))
    ]
    expected_rgb = [
        cv2.remap(images[name], *mapping, cv2.INTER_LINEAR)
        for name, mapping in zip(("left", "right"), maps, strict=True)
    ]
    raw_mask = np.zeros((HEIGHT, WIDTH), np.uint8)
    raw_mask[3:53, :78] = 1
    rect_mask = cv2.remap(raw_mask, *maps[0], cv2.INTER_NEAREST).astype(bool)
    assert not np.array_equal(raw_mask, rect_mask)
    for name, rectified in zip(("left", "right"), expected_rgb, strict=True):
        assert not np.array_equal(images[name], rectified)

    disparity = np.full((HEIGHT, WIDTH), 8.5, np.float32)
    disparity[20:25, 35:50] = 0
    disparity[30:35, 35:50] = WIDTH + 1  # positive, but right correspondence is out of frame

    def infer(left, right):
        np.testing.assert_array_equal(left, expected_rgb[0])
        np.testing.assert_array_equal(right, expected_rgb[1])
        return disparity

    backend.infer_pair.side_effect = infer
    tool.load(deployment)
    percept = tool.perceive(
        _context(pair, stereo_mask={"left": raw_mask.tolist()}, min_valid_fraction=0),
        _scene_prompts(),
    )[0]
    assert percept.status == "ok", percept.message
    backend.infer_pair.assert_called_once()
    support_a, support_b = [
        cv2.remap(np.ones((HEIGHT, WIDTH), np.float32), *mapping, cv2.INTER_LINEAR) == 1
        for mapping in maps
    ]
    rows, cols = np.indices(disparity.shape)
    right_u = cols - disparity
    valid = rect_mask & support_a & (disparity > 0) & (right_u >= 0) & (right_u <= WIDTH - 1)
    lower = np.clip(np.floor(right_u).astype(int), 0, WIDTH - 1)
    upper = np.clip(np.ceil(right_u).astype(int), 0, WIDTH - 1)
    valid &= support_b[rows, lower] & support_b[rows, upper]
    assert 0 < valid.sum() < rect_mask.sum()  # the matcher output is not accepted wholesale
    v, u = np.nonzero(valid)
    baseline = P1[0, 3] / P1[0, 0] - P2[0, 3] / P2[0, 0]
    depth = P1[0, 0] * baseline / disparity[valid].astype(np.float64)
    rays = np.column_stack((u, v, np.ones(u.size))) @ np.linalg.inv(P1[:3, :3]).T
    world = (rays * depth[:, None]) @ R1 @ rotation.T + centre
    for transform in (left_transform, right_transform):
        assert np.all(((world - transform[:3, 3]) @ transform[:3, :3])[:, 2] > 0)
    bounds = np.percentile(world, [1, 99], axis=0)
    np.testing.assert_allclose(percept.geometry.bbox3d_min, bounds[0])
    np.testing.assert_allclose(percept.geometry.bbox3d_max, bounds[1])
    np.testing.assert_allclose(percept.geometry.centroid, bounds.mean(axis=0))
    assert percept.extras["masked"] is True
    assert percept.extras["points"] == int(valid.sum())
    assert percept.extras["valid_fraction"] == pytest.approx(valid.mean())
    assert percept.extras["median_depth_m"] == pytest.approx(np.median(depth))


@pytest.mark.parametrize("phase", ["construct", "start", "infer"])
@pytest.mark.parametrize("error_type", [BackendUnavailable, BackendRequestError])
def test_backend_errors_are_tool_unavailable_without_fallback(
    tool, deployment, pair, stub_backend, phase, error_type
):
    constructor, backend = stub_backend
    error = error_type("deliberate backend failure")
    if phase == "construct":
        constructor.side_effect = error
    elif phase == "start":
        backend.start.side_effect = error
    else:
        backend.infer_pair.side_effect = error

    with pytest.raises(ToolUnavailable, match="deliberate backend failure") as raised:
        tool.load(deployment)
        tool.perceive(_context(pair), _scene_prompts())
    assert raised.value.__cause__ is error
    constructor.assert_called_once()
    if phase != "infer":
        backend.infer_pair.assert_not_called()
        assert tool._backend is None
    if phase == "start":
        backend.close.assert_called_once_with()


@pytest.mark.parametrize("asset", ["checkpoint", "model_config"])
@pytest.mark.parametrize("fault", ["missing", "mismatched", "unset", "remote"])
def test_missing_or_unusable_assets_fail_before_worker_construction(
    tool, deployment, stub_backend, asset, fault
):
    constructor, backend = stub_backend
    if fault == "missing":
        deployment[asset] += ".missing"
    elif fault == "mismatched":
        deployment[f"{asset}_sha256"] = "0" * 64
    elif fault == "unset":
        deployment[asset] = None
    else:
        deployment[asset] = "hf://test-only/not-a-local-asset"
    with pytest.raises(ToolUnavailable, match="FoundationStereo") as raised:
        tool.load(deployment)
    if fault == "mismatched":
        assert "checksum mismatch" in str(raised.value)
    constructor.assert_not_called()
    backend.start.assert_not_called()
    assert tool._backend is None


def test_absent_default_sibling_config_is_unavailable(tool, deployment, stub_backend):
    constructor, _ = stub_backend
    deployment.pop("model_config")  # custom.yaml exists, but the required sibling cfg.yaml does not
    with pytest.raises(ToolUnavailable, match="model config not found"):
        tool.load(deployment)
    constructor.assert_not_called()


@pytest.mark.parametrize("min_valid_fraction", [0, 0.01])
def test_all_zero_disparity_is_empty_geometry_not_a_segmentation_result(
    tool, deployment, pair, stub_backend, min_valid_fraction
):
    _, backend = stub_backend
    backend.infer_pair.side_effect = lambda left, right: np.zeros(left.shape[:2], np.float32)
    tool.load(deployment)
    assert tool.perceive(
        _context(pair, min_valid_fraction=min_valid_fraction), _scene_prompts()
    ) == []
    backend.infer_pair.assert_called_once()


@pytest.mark.parametrize("foundation_deployed", [False, True])
def test_ffs_is_not_an_alias_for_foundation_stereo(
    tool, deployment, pair, stub_backend, foundation_deployed
):
    constructor, backend = stub_backend
    tool.load(deployment if foundation_deployed else {})
    result = tool.perceive(_context(pair, stereo_backend="ffs"), _scene_prompts())
    assert len(result) == 1
    assert result[0].status == "unavailable"
    assert result[0].extras["code"] == stereo.ERR_BACKEND_NOT_CONFIGURED
    assert result[0].geometry is None
    backend.infer_pair.assert_not_called()
    if not foundation_deployed:
        constructor.assert_not_called()


def test_sgbm_remains_an_explicit_alternative(
    tool, deployment, pair, stub_backend, no_sgbm
):
    _, backend = stub_backend
    tool.load(deployment)
    no_sgbm.side_effect = None
    no_sgbm.return_value = (
        np.full((HEIGHT, WIDTH), DISPARITY, np.float32), np.ones((HEIGHT, WIDTH), bool)
    )
    result = tool.perceive(_context(pair, stereo_backend="sgbm"), _scene_prompts())[0]
    assert result.status == "ok"
    assert result.geometry.method == "stereo_sgbm"
    assert result.geometry.centroid[2] == pytest.approx(2.0)
    assert result.provenance.model is None and result.provenance.model_sha256 is None
    assert "runtime" not in result.extras and "model_config_sha256" not in result.extras
    no_sgbm.assert_called_once()
    backend.infer_pair.assert_not_called()


def _app(deployment):
    registry = ToolRegistry(discover=False)
    registry.register("stereo_3d", stereo.Stereo3DTool)
    config = AppConfig(
        tools={"stereo_3d": deployment},
        pipeline={"default": ["stereo_3d"]},
        cache={"enabled": False},
        warmup={"enabled": False},
        server={"token_env": ""},
    )
    return create_app(config, registry=registry)


def _request(pair, **context):
    images, calibration = pair
    return PerceiveRequest(
        images={name: encode_image(image, camera=name) for name, image in images.items()},
        calibration=calibration,
        prompts=_scene_prompts(),  # schema requires a prompt; stereo deliberately ignores it
        context=context,
        options=PerceiveOptions(tools=["stereo_3d"], fuse=True, cache=False),
    )


def test_real_cpu_worker_http_fused_geometry_reuse_and_shutdown(
    deployment, pair, cpu_backend_factory
):
    constructor, backends = cpu_backend_factory()
    app = _app(deployment)
    assert len(backends) == 1
    backend = backends[0]
    process = backend._process
    assert process is not None and process.poll() is None  # load starts eagerly
    assert backend.describe()["calls"] == 0
    assert process.pid != os.getpid()
    with TestClient(app) as client:
        for count, context in enumerate(({}, _untrusted_context(deployment)), start=1):
            request = _request(pair, **context)
            response = client.post("/v1/perceive", json=request.model_dump(mode="json"))
            assert response.status_code == 200, response.text
            body = PerceiveResponse.model_validate(response.json())
            assert body.percepts == []
            assert body.fused is not None
            _assert_metric_percept(body.fused)
            assert set(body.timings_ms) == {"stereo_3d"}  # no second fusion stage
            assert "tool_unavailable" not in body.warnings
            runtime = body.fused.extras["runtime"]
            assert runtime["running"] is True
            assert runtime["device"] == "cpu" and runtime["torch_imported"] is False
            assert runtime["pid"] == runtime["adapter_pid"] == process.pid
            assert runtime["calls"] == count
            assert runtime["python"] == deployment["python"]
    constructor.assert_called_once()
    assert backend.worker_config["valid_iters"] == deployment["valid_iters"]
    for key in ("source_dir", "checkpoint", "model_config"):
        assert backend.worker_config[key] == deployment[key]
    assert backend.describe()["state"] == "closed"
    assert backend._process is None and process.poll() is not None
    assert all(stream.closed for stream in (process.stdin, process.stdout, process.stderr))
    assert app.state.registry.entry("stereo_3d").instance is None


@pytest.mark.parametrize("failure", [
    "error", "crash", "startup_error", "missing_checkpoint", "missing_config",
])
def test_http_worker_failure_or_missing_assets_is_unavailable_inside_200(
    deployment, pair, cpu_backend_factory, failure
):
    missing_asset = failure.startswith("missing_")
    constructor, backends = cpu_backend_factory("normal" if missing_asset else failure)
    if missing_asset:
        key = "checkpoint" if failure == "missing_checkpoint" else "model_config"
        deployment[key] += ".missing"
    with TestClient(_app(deployment)) as client:
        response = client.post("/v1/perceive", json=_request(pair).model_dump(mode="json"))
        assert response.status_code == 200, response.text
        body = PerceiveResponse.model_validate(response.json())
        assert body.fused is None
        assert body.percepts
        assert "tool_unavailable" in body.warnings
        for percept in body.percepts:
            assert percept.status == "unavailable"
            assert percept.geometry is None and percept.observations == []
            assert percept.provenance.tool == "stereo_3d"
            assert percept.extras["code"] == "tool_unavailable"
            assert percept.message
            if failure == "error":
                assert "fake inference failure" in percept.message
        if not missing_asset and failure != "startup_error":
            # A failed resident must not be silently reconstructed on the next HTTP request.
            again = client.post("/v1/perceive", json=_request(pair).model_dump(mode="json"))
            assert again.status_code == 200
            repeated = PerceiveResponse.model_validate(again.json())
            assert repeated.fused is None and repeated.percepts
            assert all(p.status == "unavailable" for p in repeated.percepts)
            constructor.assert_called_once()
    if missing_asset:
        constructor.assert_not_called()
        assert backends == []
    else:
        assert backends
        assert all(backend._process is None for backend in backends)
