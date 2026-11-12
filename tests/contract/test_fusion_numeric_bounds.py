"""Overflowing JSON numbers retain the existing fusion fallback and warning semantics."""

from __future__ import annotations

import pytest

from tests import synth_geom
from tests.contract.helpers import perceive_payload


@pytest.mark.parametrize("tools", [["fake"], ["fake", "multi_view_3d"]])
@pytest.mark.parametrize(
    "context,warning",
    [
        ({"table_z": 10**400}, "invalid_table_z"),
        ({"table_plane": [0, 0, 1, 10**400]}, "invalid_table_plane"),
        ({"prism_step_m": 10**400}, None),
    ],
)
def test_overflowing_fusion_numbers_use_existing_fallbacks_over_http(client, tools, context, warning):
    calibration, images, _ = synth_geom.two_view_roomcube(width=160, height=120)
    payload = perceive_payload(images, [{"type": "text", "text": "red"}], calibration, tools=tools, fuse=True)
    baseline = client.post("/v1/perceive", json=payload)
    assert baseline.status_code == 200, baseline.text
    baseline_fused = baseline.json()["fused"]
    assert baseline_fused is not None

    payload["context"] = context
    response = client.post("/v1/perceive", json=payload)
    assert response.status_code == 200, response.text
    result = response.json()
    fused = result["fused"]
    assert fused is not None and fused["status"] == "ok"
    assert fused["geometry"] == baseline_fused["geometry"]
    if warning is not None:
        assert warning in fused["extras"]["warnings"]
        assert warning in result["warnings"]
    assert "table_z_assumed" in result["warnings"]
