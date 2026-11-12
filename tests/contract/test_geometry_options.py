"""Geometry request options and unsupported SGBM image sizes fail without server errors."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cv2", reason="geometry options require the optional OpenCV runtime")

from tests.contract.helpers import perceive_payload
from tests.unit.test_stereo_3d import _pair
from tests.unit.test_triview_3d import _three_view_scene


@pytest.mark.parametrize("tool", ["stereo_3d", "triview_3d"])
@pytest.mark.parametrize(
    "bad_options", [{"stereo_sigma_d_px": -1}, {"stereo_sigma_d_px": "nan"}, {"stereo_num_disparities": "abc"}]
)
def test_bad_geometry_numbers_are_invalid_request_percepts(client, tool, bad_options):
    if tool == "stereo_3d":
        left, right, calibration = _pair()
        images = {"left": left, "right": right}
    else:
        images, calibration = _three_view_scene()
    payload = perceive_payload(images, [{"type": "text", "text": "scene"}], calibration, tools=[tool],
                               fuse=True, context={"stereo_num_disparities": 256, **bad_options})
    response = client.post("/v1/perceive", json=payload)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["fused"] is None
    assert result["percepts"][0]["status"] == "invalid_request"
    assert result["percepts"][0]["extras"]["code"] == "invalid_geometry_options"


@pytest.mark.parametrize("tool", ["stereo_3d", "triview_3d"])
def test_small_sgbm_images_are_invalid_request_percepts(client, tool):
    if tool == "stereo_3d":
        _, _, calibration = _pair()
    else:
        _, calibration = _three_view_scene()
    images = {name: np.zeros((24, 8, 3), np.uint8) for name in calibration}
    calibration = {
        name: cal.model_copy(update={"width": 8, "height": 24, "K": [[50, 0, 4], [0, 50, 12], [0, 0, 1]]})
        for name, cal in calibration.items()
    }
    payload = perceive_payload(images, [{"type": "text", "text": "scene"}], calibration, tools=[tool], fuse=True)
    response = client.post("/v1/perceive", json=payload)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["fused"] is None
    assert result["percepts"][0]["status"] == "invalid_request"
    assert result["percepts"][0]["extras"]["code"] == "sgbm_image_too_small"
