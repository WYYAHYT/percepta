"""Reject nonphysical camera calibration at the HTTP request boundary."""

from __future__ import annotations

import numpy as np
import pytest

from tests import synth
from tests.contract.helpers import perceive_payload


@pytest.mark.parametrize(
    "field,value",
    [
        ("T_world_cam", np.zeros((4, 4)).tolist()),
        ("T_world_cam", np.diag([-1.0, 1.0, 1.0, 1.0]).tolist()),
        ("K", [[0, 0, 0], [0, 0, 0], [0, 0, 1]]),
        ("K", [[288, 1, 320], [0, 288, 240], [0, 0, 1]]),
    ],
)
def test_nonphysical_calibration_is_http400_before_tool_inference(client, field, value):
    calibration = synth.make_calibration("cam")
    payload = perceive_payload(
        {"cam": synth.red_square_scene()}, [{"type": "text", "text": "red"}], {"cam": calibration}
    )
    payload["calibration"]["cam"][field] = value
    response = client.post("/v1/perceive", json=payload)
    assert response.status_code == 400, response.text
    assert response.json()["error"] == "invalid_request"


def test_calibration_rounding_noise_remains_accepted_over_http(client):
    calibration = synth.make_calibration("cam")
    payload = perceive_payload(
        {"cam": synth.red_square_scene()}, [{"type": "text", "text": "red"}], {"cam": calibration}
    )
    payload["calibration"]["cam"]["T_world_cam"][0][0] += 2e-8
    payload["calibration"]["cam"]["K"][0][1] = 1e-10
    response = client.post("/v1/perceive", json=payload)
    assert response.status_code == 200, response.text
