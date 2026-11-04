"""S1 over HTTP — the synthetic red square through ``POST /v1/perceive``.

The acceptance criterion (spec §14 S1) is: a centred red square yields exactly one
observation whose ``area_px``, ``box_xyxy`` and ``centroid_uv`` are right to within 2 px, and
the HTTP path must return *the same numbers* as the in-process
:class:`~percepta.pipeline.PipelineRunner` path (decision D12).

Geometry is switched off (``options.fuse=false``, ``options.tools=["fake"]``) so the numbers
depend only on the fake tool and never on whichever optional geometry modules are present.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from percepta.config import AppConfig
from percepta.masks import to_array
from percepta.pipeline import PipelineRunner
from percepta.registry import ToolRegistry
from percepta.schema import Mask, PerceiveRequest, PerceiveResponse
from tests import synth, synth_geom
from tests.contract import helpers

#: The S1 fixture: an 80x80 red square centred on (320, 240) in a 640x480 black frame.
EXPECTED_AREA = 80 * 80
EXPECTED_BOX = [280.0, 200.0, 360.0, 280.0]
EXPECTED_CENTROID = [319.5, 239.5]
MAX_ERROR_PX = 2.0


def _s1_payload(cameras: tuple[str, ...] = ("cam_head",)) -> dict:
    scene = synth.red_square_scene()
    return helpers.perceive_payload(
        {camera: scene for camera in cameras},
        [{"type": "text", "text": "red block"}],
        {camera: synth.make_calibration(camera) for camera in cameras},
        tools=["fake"],
        fuse=False,
    )


def _assert_s1_observation(observation: dict) -> None:
    assert observation["area_px"] == EXPECTED_AREA
    box = np.asarray(observation["box_xyxy"], dtype=float)
    centroid = np.asarray(observation["centroid_uv"], dtype=float)
    assert box.shape == (4,)
    assert np.all(np.abs(box - np.asarray(EXPECTED_BOX)) <= MAX_ERROR_PX), box
    assert np.all(np.abs(centroid - np.asarray(EXPECTED_CENTROID)) <= MAX_ERROR_PX), centroid
    assert observation["score"] == pytest.approx(1.0, abs=1e-6)
    assert observation["label"] == "red block"


def test_red_square_over_http(client: TestClient, ci_config: AppConfig) -> None:
    payload = _s1_payload()
    response = client.post("/v1/perceive", json=payload)
    assert response.status_code == 200, response.text
    body = response.json()

    assert len(body["percepts"]) == 1, "one camera x one prompt x one tool -> one Percept"
    percept = body["percepts"][0]
    assert percept["status"] == "ok"
    assert percept["camera"] == "cam_head"
    assert len(percept["observations"]) == 1
    _assert_s1_observation(percept["observations"][0])

    # The mask travels as RLE (spec §3.2 default) and decodes back to the exact square.
    mask = Mask.model_validate(percept["observations"][0]["mask"])
    assert mask.encoding == "rle"
    assert tuple(mask.size) == (480, 640)
    array = to_array(mask)
    assert array.shape == (480, 640)
    assert int(array.sum()) == EXPECTED_AREA
    assert bool(array[240, 320])
    assert not bool(array[0, 0])

    # §3: every model must round-trip through JSON.
    revalidated = PerceiveResponse.model_validate_json(response.text)
    assert revalidated.request_id == payload["request_id"]
    assert revalidated.percepts[0].observations[0].area_px == EXPECTED_AREA
    assert body["warnings"] == []


def test_http_numbers_equal_the_in_process_path(
    client: TestClient, ci_config: AppConfig
) -> None:
    payload = _s1_payload()
    http_body = client.post("/v1/perceive", json=payload).json()
    http_observation = http_body["percepts"][0]["observations"][0]

    request = PerceiveRequest.model_validate(payload)
    runner = PipelineRunner(ToolRegistry(), ci_config)
    try:
        local = runner.perceive(request)
    finally:
        runner.close()

    assert len(local.percepts) == len(http_body["percepts"])
    local_percept = local.percepts[0]
    assert local_percept.status == http_body["percepts"][0]["status"]
    local_observation = local_percept.observations[0]
    assert local_observation.area_px == http_observation["area_px"]
    assert local_observation.box_xyxy == http_observation["box_xyxy"]
    assert local_observation.centroid_uv == http_observation["centroid_uv"]
    assert local_observation.score == http_observation["score"]
    assert local_percept.provenance.tool == http_body["percepts"][0]["provenance"]["tool"]
    assert local_percept.provenance.image_sha256 == http_body["percepts"][0]["provenance"][
        "image_sha256"
    ]
    assert local.warnings == http_body["warnings"]


def test_two_cameras_fan_out_over_http(client: TestClient) -> None:
    """Decision D6: one tool call covers every camera; the response has one Percept per view."""
    payload = _s1_payload(cameras=("cam_head", "cam_left"))
    body = client.post("/v1/perceive", json=payload).json()
    assert {percept["camera"] for percept in body["percepts"]} == {"cam_head", "cam_left"}
    assert len(body["percepts"]) == 2
    for percept in body["percepts"]:
        assert percept["status"] == "ok"
        _assert_s1_observation(percept["observations"][0])
    # The same tool instance handled both views in one call: one timing entry.
    assert list(body["timings_ms"]) == ["fake"]


def test_geometry_stage_supplies_fused_without_a_duplicate_entry(client: TestClient) -> None:
    """Decision D15 — ``multi_view_3d`` in the CI pipeline is the response's ``fused``.

    A calibrated two-camera request must return exactly ``cameras x prompts`` Percepts (the
    geometry stage contributes no camera-less entry), ``fused`` must carry a world-frame
    geometry from at least two views, and ``fusion.fuse`` must not run a second time after the
    geometry tool (no extra ``fuse`` timing entry).
    """
    calibrations, images, _center = synth_geom.two_view_roomcube()
    payload = helpers.perceive_payload(
        images,
        [{"type": "text", "text": "red block"}],
        calibrations,
        fuse=True,  # tools=None -> the ci.yaml default pipeline ["fake", "multi_view_3d"]
    )
    response = client.post("/v1/perceive", json=payload)
    assert response.status_code == 200, response.text
    body = response.json()

    assert len(body["percepts"]) == len(images) * len(payload["prompts"])
    assert {percept["camera"] for percept in body["percepts"]} == set(calibrations)
    assert all(percept["provenance"]["kind"] != "geometry" for percept in body["percepts"])

    assert body["fused"] is not None
    assert body["fused"]["provenance"]["tool"] == "multi_view_3d"
    assert body["fused"]["provenance"]["kind"] == "geometry"
    geometry = body["fused"]["geometry"]
    assert geometry is not None
    assert geometry["frame"] == "world"
    assert geometry["views_used"] >= 2
    assert geometry["uncertainty_m"] is not None

    assert "fuse" not in body["timings_ms"], "fusion must not run twice"
    assert set(body["timings_ms"]) == {"fake", "multi_view_3d"}
