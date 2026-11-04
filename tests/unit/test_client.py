"""``PerceptaClient`` tests (spec §7, decision D8).

The HTTP half runs against a **real uvicorn server** on an ephemeral loopback port, because
the client is deliberately stdlib-only (``urllib.request``) and an ASGI in-process transport
would not exercise the actual socket path. The server serves ``configs/ci.yaml`` (``fake``
tool, decision D9): no GPU, no weights, no external network.

The local half exercises ``perceive_local()``, which must return the same numbers as the HTTP
path (decision D12).
"""

from __future__ import annotations

import json
import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from percepta.client import (
    PerceptaClient,
    PerceptaError,
    build_perceive_request,
)
from percepta.config import load_config
from percepta.masks import to_array
from percepta.registry import ToolRegistry
from percepta.schema import ImageRef, PerceiveOptions, PerceiveResponse, Prompt
from percepta.service.app import create_app
from tests import synth
from tests.contract.helpers import CI_CONFIG

CAM = "cam_head"
TEXT = Prompt(type="text", text="red block")
EXPECTED_AREA = 80 * 80
EXPECTED_BOX = [280.0, 200.0, 360.0, 280.0]
EXPECTED_CENTROID = [319.5, 239.5]


# ---------------------------------------------------------------------------
# Live server
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def server_url() -> Iterator[str]:
    """Serve the CI config on an ephemeral loopback port for the whole module."""
    import uvicorn

    app = create_app(load_config(CI_CONFIG))
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = int(probe.getsockname()[1])
    probe.close()

    server = uvicorn.Server(
        uvicorn.Config(
            app, host="127.0.0.1", port=port, log_level="warning", lifespan="off"
        )
    )
    thread = threading.Thread(target=server.run, name="percepta-uvicorn", daemon=True)
    thread.start()
    deadline = time.monotonic() + 20.0
    while not server.started:
        if time.monotonic() > deadline:  # pragma: no cover - environment failure
            raise RuntimeError("uvicorn did not start in time")
        time.sleep(0.02)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=15.0)


@pytest.fixture()
def client(server_url: str) -> Iterator[PerceptaClient]:
    http_client = PerceptaClient(server_url)
    try:
        yield http_client
    finally:
        http_client.close()


def _scene() -> np.ndarray:
    return synth.red_square_scene()


def _calibration() -> dict[str, Any]:
    return {CAM: synth.make_calibration(CAM)}


def _options() -> PerceiveOptions:
    return PerceiveOptions(tools=["fake"], fuse=False)


# ---------------------------------------------------------------------------
# Read-only endpoints
# ---------------------------------------------------------------------------


def test_health_info_and_tools(client: PerceptaClient) -> None:
    health = client.health()
    assert health["ok"] is True
    assert health["uptime_s"] >= 0.0
    assert any(tool["name"] == "fake" for tool in health["tools"])

    info = client.info()
    assert info["schema_version"] == "percepta/v1"
    assert info["version"]
    assert info["pipeline"]["default"] == ["fake", "multi_view_3d"]

    tools = client.tools()
    assert isinstance(tools, list) and tools
    fake = next(tool for tool in tools if tool["name"] == "fake")
    assert fake["capabilities"]["kind"] == "fake"
    assert "text" in fake["capabilities"]["prompts"]

    assert client.wait_until_ready(timeout_s=5.0) is True


# ---------------------------------------------------------------------------
# perceive() over real HTTP
# ---------------------------------------------------------------------------


def test_perceive_round_trip(client: PerceptaClient) -> None:
    response = client.perceive(
        images={CAM: _scene()},
        prompts=[TEXT],
        calibration=_calibration(),
        options=_options(),
    )
    assert isinstance(response, PerceiveResponse)
    assert response.request_id
    assert len(response.percepts) == 1
    percept = response.percepts[0]
    assert percept.status == "ok"
    assert percept.camera == CAM
    assert percept.provenance.tool == "fake"
    assert len(percept.observations) == 1
    observation = percept.observations[0]
    assert observation.area_px == EXPECTED_AREA
    assert observation.box_xyxy == EXPECTED_BOX
    assert observation.centroid_uv == EXPECTED_CENTROID
    assert observation.mask is not None
    assert int(to_array(observation.mask).sum()) == EXPECTED_AREA
    assert response.warnings == []


def test_perceive_accepts_a_file_path(client: PerceptaClient, tmp_path: Path) -> None:
    path = tmp_path / "scene.png"
    Image.fromarray(_scene()).save(path)

    from_path = client.perceive(
        images={CAM: str(path)},
        prompts=[TEXT],
        calibration=_calibration(),
        options=_options(),
    )
    from_array = client.perceive(
        images={CAM: _scene()},
        prompts=[TEXT],
        calibration=_calibration(),
        options=_options(),
    )
    assert from_path.percepts[0].observations[0].area_px == EXPECTED_AREA
    assert (
        from_path.percepts[0].observations[0].box_xyxy
        == from_array.percepts[0].observations[0].box_xyxy
    )
    assert (
        from_path.percepts[0].observations[0].centroid_uv
        == from_array.percepts[0].observations[0].centroid_uv
    )


def test_not_found_is_a_response_not_an_error(client: PerceptaClient) -> None:
    response = client.perceive(
        images={CAM: _scene()},
        prompts=["purple block"],
        calibration=_calibration(),
        options=_options(),
    )
    assert response.percepts[0].status == "not_found"
    assert response.percepts[0].observations == []


def test_bad_request_raises_a_typed_error(client: PerceptaClient) -> None:
    """A 400 is raised, carrying the parsed JSON body (spec §7, appendix A)."""
    wrong_calibration = {CAM: synth.make_calibration(CAM, width=320, height=240)}
    with pytest.raises(PerceptaError) as excinfo:
        client.perceive(
            images={CAM: _scene()},
            prompts=[TEXT],
            calibration=wrong_calibration,
            options=_options(),
        )
    error = excinfo.value
    assert error.status_code == 400
    assert error.code == "invalid_request"
    assert isinstance(error.body, dict)
    assert "calibration" in json.dumps(error.body)


# ---------------------------------------------------------------------------
# perceive_local() — same semantics, no HTTP (decision D12)
# ---------------------------------------------------------------------------


def test_perceive_local_matches_the_http_path(client: PerceptaClient) -> None:
    http_response = client.perceive(
        images={CAM: _scene()},
        prompts=[TEXT],
        calibration=_calibration(),
        options=_options(),
    )
    with PerceptaClient() as local_client:
        local_response = local_client.perceive_local(
            images={CAM: _scene()},
            prompts=[TEXT],
            calibration=_calibration(),
            options=_options(),
            config=load_config(CI_CONFIG),
        )

    http_observation = http_response.percepts[0].observations[0]
    local_observation = local_response.percepts[0].observations[0]
    assert local_response.percepts[0].status == http_response.percepts[0].status
    assert local_observation.area_px == http_observation.area_px
    assert local_observation.box_xyxy == http_observation.box_xyxy
    assert local_observation.centroid_uv == http_observation.centroid_uv
    assert local_observation.score == http_observation.score
    assert local_response.warnings == http_response.warnings


def test_perceive_local_reuses_one_runner() -> None:
    with PerceptaClient() as local_client:
        first = local_client.perceive_local(
            images={CAM: _scene()}, prompts=[TEXT], calibration=_calibration(),
            options=_options(),
        )
        runner = local_client._local_runner
        second = local_client.perceive_local(
            images={CAM: _scene()}, prompts=[TEXT], calibration=_calibration(),
            options=_options(),
        )
        assert local_client._local_runner is runner
        assert first.percepts[0].status == second.percepts[0].status == "ok"


def test_perceive_local_works_with_default_config_and_explicit_tools() -> None:
    """No config file needed: the code default pipeline is replaced by ``options.tools``."""
    with PerceptaClient() as local_client:
        response = local_client.perceive_local(
            images={CAM: _scene()},
            prompts=[TEXT],
            calibration={},
            options=PerceiveOptions(tools=["fake"], fuse=False),
        )
    assert response.percepts[0].status == "ok"
    assert response.percepts[0].observations[0].area_px == EXPECTED_AREA


# ---------------------------------------------------------------------------
# Retry policy (spec §7: bounded retries on 503/5xx only)
# ---------------------------------------------------------------------------


def test_retries_transient_5xx_then_succeeds() -> None:
    attempted: list[str] = []

    def flaky_raw_request(method: str, path: str, body: bytes | None):
        attempted.append(path)
        if len(attempted) <= 2:
            raise PerceptaError("busy", status_code=503, code="busy", body={"error": "busy"})
        payload = {"ok": True, "uptime_s": 0.0, "tools": []}
        return 200, json.dumps(payload).encode()

    retrying = PerceptaClient("http://127.0.0.1:1", retries=2, backoff_s=0.0)
    retrying._raw_request = flaky_raw_request  # type: ignore[method-assign]
    assert retrying.health()["ok"] is True
    assert len(attempted) == 3


def test_gives_up_after_the_retry_budget() -> None:
    attempted: list[str] = []

    def always_busy(method: str, path: str, body: bytes | None):
        attempted.append(path)
        raise PerceptaError("busy", status_code=503, code="busy")

    retrying = PerceptaClient("http://127.0.0.1:1", retries=2, backoff_s=0.0)
    retrying._raw_request = always_busy  # type: ignore[method-assign]
    with pytest.raises(PerceptaError) as excinfo:
        retrying.health()
    assert excinfo.value.status_code == 503
    assert len(attempted) == 3  # initial attempt + 2 retries


def test_does_not_retry_client_errors() -> None:
    attempted: list[str] = []

    def not_found(method: str, path: str, body: bytes | None):
        attempted.append(path)
        raise PerceptaError("not found", status_code=404, code="not_found")

    retrying = PerceptaClient("http://127.0.0.1:1", retries=2, backoff_s=0.0)
    retrying._raw_request = not_found  # type: ignore[method-assign]
    with pytest.raises(PerceptaError):
        retrying.health()
    assert len(attempted) == 1


def test_connection_failure_is_reported_without_a_status() -> None:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = int(probe.getsockname()[1])
    probe.close()  # nothing listens there now
    unreachable = PerceptaClient(f"http://127.0.0.1:{port}", timeout=2.0, retries=0)
    with pytest.raises(PerceptaError) as excinfo:
        unreachable.health()
    assert excinfo.value.status_code is None
    assert excinfo.value.code == "connection_error"


# ---------------------------------------------------------------------------
# Request construction helper
# ---------------------------------------------------------------------------


def test_build_perceive_request_accepts_many_image_forms(tmp_path: Path) -> None:
    path = tmp_path / "scene.png"
    Image.fromarray(_scene()).save(path)
    reference = ImageRef.model_validate(
        build_perceive_request({CAM: _scene()}, [TEXT]).images[CAM].model_dump(mode="json")
    )
    request = build_perceive_request(
        {CAM: _scene()},
        ["red block", {"type": "box", "box": [0.5, 0.5, 0.2, 0.2]}],
        calibration=_calibration(),
        context={"instruction": "pick up the block"},
        options={"tools": ["fake"], "fuse": False},
    )
    assert request.images[CAM].encoding == "png"
    assert [prompt.type for prompt in request.prompts] == ["text", "box"]
    assert request.prompts[0].text == "red block"
    assert request.calibration[CAM].camera == CAM
    assert request.options.tools == ["fake"]
    assert request.options.fuse is False
    assert request.context["instruction"] == "pick up the block"

    from_path = build_perceive_request({CAM: str(path)}, [TEXT])
    assert from_path.images[CAM].encoding == "png"
    assert reference.encoding == "png"


def test_build_perceive_request_rejects_unknown_image_types() -> None:
    with pytest.raises(TypeError):
        build_perceive_request({CAM: 42}, [TEXT])
    with pytest.raises(ValueError):
        build_perceive_request({}, [TEXT])


def test_local_runner_accepts_an_explicit_registry() -> None:
    """``perceive_local`` can be pointed at a purpose-built registry (used by tests/embedders)."""
    registry = ToolRegistry()
    registry.register("fake", _TrackingFactory())
    with PerceptaClient() as local_client:
        local_response = local_client.perceive_local(
            images={CAM: _scene()},
            prompts=[TEXT],
            options=PerceiveOptions(tools=["fake"], fuse=False),
            registry=registry,
            config=load_config(CI_CONFIG),
        )
    assert local_response.percepts[0].status == "ok"


class _TrackingFactory:
    """Factory shim: returns a fresh ``FakeTool`` so the registry treats it like a plugin."""

    def __call__(self):
        from percepta.tools.fake import FakeTool

        return FakeTool()
