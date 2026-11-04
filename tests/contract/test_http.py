"""S4 — HTTP contract: 200/400/401/413/503/504 and ``not_found`` staying inside a 200.

Everything runs against ``configs/ci.yaml`` (``fake`` tool, decision D9): no GPU, no network,
no weights. The error bodies follow appendix A: ``{"error": <code>, "message": ...}``.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from percepta.config import AppConfig, load_config
from percepta.registry import ToolRegistry
from percepta.schema import ToolCapabilities
from percepta.service.app import create_app
from percepta.tools.base import BaseTool
from tests import synth
from tests.contract import helpers
from tests.contract.helpers import CI_CONFIG


def _payload(**overrides) -> dict:
    return helpers.perceive_payload(
        {"cam_head": synth.red_square_scene()},
        [{"type": "text", "text": "red block"}],
        {"cam_head": synth.make_calibration("cam_head")},
        **overrides,
    )


def _assert_error(response, status: int, code: str) -> None:
    assert response.status_code == status, response.text
    body = response.json()
    assert body["error"] == code
    assert isinstance(body["message"], str) and body["message"]


# ---------------------------------------------------------------------------
# Happy path and introspection
# ---------------------------------------------------------------------------


def test_perceive_happy_path_returns_200(client: TestClient) -> None:
    payload = _payload()
    response = client.post("/v1/perceive", json=payload)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["request_id"] == payload["request_id"]
    assert body["percepts"], body
    assert body["percepts"][0]["status"] == "ok"
    assert body["percepts"][0]["camera"] == "cam_head"
    assert "fake" in body["timings_ms"]
    assert body["timings_ms"]["fake"] >= 0.0
    # The frozen schema fields must survive the JSON round trip.
    assert body["percepts"][0]["schema_version"] == "percepta/v1"
    assert body["percepts"][0]["provenance"]["tool"] == "fake"


def test_healthz_reflects_tool_status(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert isinstance(body["uptime_s"], float) and body["uptime_s"] >= 0.0
    tools = {tool["name"]: tool for tool in body["tools"]}
    assert "fake" in tools
    # ci.yaml enables `fake`, so the app must have loaded it at startup.
    assert tools["fake"]["status"] == "ok"
    assert set(tools["fake"]) == {"name", "version", "status"}
    # spec §11.3: the only allowed values are ok | loading | error.
    assert all(tool["status"] in {"ok", "loading", "error"} for tool in body["tools"])


def test_healthz_reports_error_for_unloadable_tool() -> None:
    class Broken(BaseTool):
        name = "broken"
        version = "0.0.1"
        kind = "fake"

        def capabilities(self) -> ToolCapabilities:  # pragma: no cover - never reached
            return ToolCapabilities(kind="fake")

        def load(self, config) -> None:
            raise RuntimeError("no weights")

        def perceive(self, ctx, prompts):  # pragma: no cover - never reached
            return []

    registry = ToolRegistry()
    registry.register("broken", Broken)
    config = load_config(CI_CONFIG)
    config.tools = {"broken": {"enabled": True}}
    with TestClient(create_app(config, registry=registry)) as test_client:
        body = test_client.get("/healthz").json()
    tools = {tool["name"]: tool for tool in body["tools"]}
    assert tools["broken"]["status"] == "error"


def test_disabled_tools_are_absent_but_failing_enabled_tools_are_visible() -> None:
    """§11.3: a tool explicitly disabled is not part of this deployment, so it is not listed.

    Regression for the bug where ``enabled: false`` tools stayed ``lazy`` and were mapped to
    ``ok`` — an operator would believe SAM3 was available when it had never been attempted.
    The other half of the pair: a tool that *is* part of the deployment (unlisted in
    ``tools:``) but cannot load must still appear, with ``status="error"``.
    """
    # ci.yaml sets `tools.sam3.enabled: false` -> absent from every tool listing.
    with TestClient(create_app(load_config(CI_CONFIG))) as test_client:
        health = {tool["name"]: tool for tool in test_client.get("/healthz").json()["tools"]}
        assert set(health) == {"fake", "multi_view_3d"}
        assert "sam3" not in health
        listed = {tool["name"] for tool in test_client.get("/v1/tools").json()["tools"]}
        assert listed == {"fake", "multi_view_3d"}
        info = {tool["name"] for tool in test_client.get("/v1/info").json()["tools"]}
        assert info == {"fake", "multi_view_3d"}

    # Default config: `sam3` is unlisted -> enabled -> present and honestly `error` (D9).
    with TestClient(create_app(AppConfig())) as test_client:
        rows = {tool["name"]: tool["status"] for tool in test_client.get("/healthz").json()["tools"]}
        assert rows["sam3"] == "error"
        assert rows["multi_view_3d"] == "ok"


def test_info_and_tools_endpoints(client: TestClient) -> None:
    info = client.get("/v1/info")
    assert info.status_code == 200
    info_body = info.json()
    assert info_body["schema_version"] == "percepta/v1"
    assert info_body["tool_api_version"] == "percepta/tool/v1"
    assert info_body["version"]
    assert info_body["pipeline"]["default"] == ["fake", "multi_view_3d"]
    assert "device" in info_body
    fake = {tool["name"]: tool for tool in info_body["tools"]}["fake"]
    assert fake["capabilities"]["kind"] == "fake"
    assert "text" in fake["capabilities"]["prompts"]

    tools = client.get("/v1/tools")
    assert tools.status_code == 200
    rows = {row["name"]: row for row in tools.json()["tools"]}
    assert rows["fake"]["status"] == "ok"
    assert rows["fake"]["loaded"] is True


# ---------------------------------------------------------------------------
# 400 — malformed request payloads
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        pytest.param({}, id="empty-object"),
        pytest.param({"images": {}, "prompts": []}, id="no-images-no-prompts"),
        pytest.param(
            {
                "images": {"cam": {"encoding": "png", "data": "not-base64"}},
                "prompts": [{"type": "text", "text": "cup"}],
                "calibration": {"cam": {"camera": "cam", "width": 0, "height": 1}},
            },
            id="bad-calibration",
        ),
        pytest.param(
            {
                "images": {"cam": {"encoding": "png", "data": "AAAA"}},
                "prompts": [{"type": "bogus", "text": "cup"}],
            },
            id="bad-prompt-type",
        ),
        pytest.param(
            {
                "images": {"cam": {"encoding": "png", "data": "AAAA"}},
                "prompts": [{"type": "box", "box": [0.1, 0.2]}],
            },
            id="bad-box",
        ),
    ],
)
def test_malformed_body_returns_400(client: TestClient, body: dict) -> None:
    response = client.post("/v1/perceive", json=body)
    _assert_error(response, 400, "invalid_request")


def test_size_mismatch_between_image_and_calibration_returns_400(client: TestClient) -> None:
    """spec §4.6: only known after decoding, so the pipeline owns this check."""
    payload = _payload()
    payload["calibration"]["cam_head"]["width"] = 320
    payload["calibration"]["cam_head"]["height"] = 240
    response = client.post("/v1/perceive", json=payload)
    _assert_error(response, 400, "invalid_request")
    assert "calibration" in response.json()["message"]


# ---------------------------------------------------------------------------
# 401 — optional bearer token
# ---------------------------------------------------------------------------


def test_missing_or_wrong_token_returns_401(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PERCEPTA_TOKEN", "s3cret")
    config = load_config(CI_CONFIG)
    with TestClient(create_app(config)) as test_client:
        payload = _payload()
        _assert_error(test_client.post("/v1/perceive", json=payload), 401, "unauthorized")
        _assert_error(
            test_client.post(
                "/v1/perceive", json=payload, headers={"Authorization": "Bearer wrong"}
            ),
            401,
            "unauthorized",
        )
        # /healthz stays open for probes; a valid token opens the data endpoints.
        assert test_client.get("/healthz").status_code == 200
        ok = test_client.post(
            "/v1/perceive", json=payload, headers={"Authorization": "Bearer s3cret"}
        )
        assert ok.status_code == 200, ok.text
        assert test_client.get("/v1/info", headers={"Authorization": "Bearer s3cret"}).status_code == 200


def test_no_token_configured_means_no_auth(client: TestClient) -> None:
    assert client.post("/v1/perceive", json=_payload()).status_code == 200


# ---------------------------------------------------------------------------
# 413 — body size limit
# ---------------------------------------------------------------------------


def test_payload_too_large_returns_413() -> None:
    config = load_config(CI_CONFIG)
    config.server.max_body_bytes = 1024
    with TestClient(create_app(config)) as test_client:
        response = test_client.post(
            "/v1/perceive",
            content=b"{" + b"x" * 4096 + b"}",
            headers={"Content-Type": "application/json"},
        )
        _assert_error(response, 413, "payload_too_large")
        # The limit is about bodies, not the service: /healthz still answers.
        assert test_client.get("/healthz").status_code == 200


def test_payload_too_large_without_content_length_returns_413() -> None:
    """A chunked body must not slip past the limit by omitting Content-Length."""
    config = load_config(CI_CONFIG)
    config.server.max_body_bytes = 1024
    with TestClient(create_app(config)) as test_client:
        response = test_client.post(
            "/v1/perceive",
            content=iter([b"y" * 2048]),
            headers={"Content-Type": "application/json"},
        )
        _assert_error(response, 413, "payload_too_large")


# ---------------------------------------------------------------------------
# 503 — admission control
# ---------------------------------------------------------------------------


def test_busy_when_max_queue_is_zero() -> None:
    config = load_config(CI_CONFIG)
    config.server.max_queue = 0
    with TestClient(create_app(config)) as test_client:
        _assert_error(test_client.post("/v1/perceive", json=_payload()), 503, "busy")


def test_busy_when_every_admission_slot_is_taken() -> None:
    config = load_config(CI_CONFIG)
    config.server.max_queue = 1
    app = create_app(config)
    with TestClient(app) as test_client:
        semaphore = app.state.service.semaphore
        assert semaphore is not None
        assert semaphore.acquire(blocking=False)  # simulate one in-flight request
        try:
            _assert_error(test_client.post("/v1/perceive", json=_payload()), 503, "busy")
        finally:
            semaphore.release()
        assert test_client.post("/v1/perceive", json=_payload()).status_code == 200


# ---------------------------------------------------------------------------
# 504 — request timeout
# ---------------------------------------------------------------------------


class _SlowTool(BaseTool):
    name = "slow"
    version = "0.0.1"
    kind = "fake"

    def capabilities(self) -> ToolCapabilities:
        return ToolCapabilities(kind="fake", prompts={"text"})

    def perceive(self, ctx, prompts):
        time.sleep(0.5)
        return []


def test_request_timeout_returns_504() -> None:
    registry = ToolRegistry()
    registry.register("slow", _SlowTool)
    config = AppConfig.model_validate(
        {
            "server": {"request_timeout_ms": 50},
            "pipeline": {"default": ["slow"]},
            "tools": {"slow": {"enabled": True}},
            "metrics": {"enabled": False},
        }
    )
    with TestClient(create_app(config, registry=registry)) as test_client:
        response = test_client.post("/v1/perceive", json=_payload())
        _assert_error(response, 504, "timeout")


# ---------------------------------------------------------------------------
# not_found is a Percept status, not an HTTP error (spec §6)
# ---------------------------------------------------------------------------


def test_not_found_stays_inside_http_200(client: TestClient) -> None:
    payload = _payload()
    payload["prompts"] = [{"type": "text", "text": "purple block"}]
    response = client.post("/v1/perceive", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["percepts"][0]["status"] == "not_found"
    assert body["percepts"][0]["observations"] == []


def test_unknown_tool_name_is_a_400(client: TestClient) -> None:
    """spec §5.2: an unknown tool name is a caller error, not a tool failure."""
    response = client.post("/v1/perceive", json=_payload(tools=["does_not_exist"]))
    _assert_error(response, 400, "invalid_request")


# ---------------------------------------------------------------------------
# /metrics (spec §6, §11.3)
# ---------------------------------------------------------------------------


def test_metrics_enabled_renders_prometheus_text(client: TestClient) -> None:
    assert client.post("/v1/perceive", json=_payload()).status_code == 200
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    text = response.text
    assert "# TYPE percepta_requests_total counter" in text or "percepta_requests_total" in text
    assert 'tool="fake"' in text
    assert 'status="ok"' in text
    assert "percepta_latency_ms_count" in text


def test_metrics_disabled_returns_404() -> None:
    config = load_config(CI_CONFIG)
    config.metrics.enabled = False
    with TestClient(create_app(config)) as test_client:
        _assert_error(test_client.get("/metrics"), 404, "not_found")
