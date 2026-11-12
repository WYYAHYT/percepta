"""Schema contract tests: frozen shapes, validation, JSON round-trip, no NaN/Inf (§3)."""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest
from pydantic import ValidationError

from percepta.schema import (
    SCHEMA_VERSION,
    Calibration,
    DepthRef,
    Geometry,
    ImageRef,
    Mask,
    Observation,
    PerceiveOptions,
    PerceiveRequest,
    PerceiveResponse,
    Percept,
    Prompt,
    Provenance,
)


def _provenance() -> Provenance:
    return Provenance(tool="fake", tool_version="0.1.0", kind="fake", latency_ms=1.5)


def _observation(index: int = 0) -> Observation:
    return Observation(
        object_id=f"id-{index}",
        score=0.9,
        box_xyxy=[1.0, 2.0, 3.0, 4.0],
        area_px=4,
        centroid_uv=[2.0, 3.0],
        label="block",
    )


def _calibration(**overrides) -> Calibration:
    data = {
        "camera": "cam_head",
        "width": 640,
        "height": 480,
        "K": [[288.13, 0, 320], [0, 288.13, 240], [0, 0, 1]],
        "T_world_cam": [[1, 0, 0, 0], [0, 1, 0, -0.41], [0, 0, 1, 1.308], [0, 0, 0, 1]],
        "id": "station-arx-x5",
    }
    data.update(overrides)
    return Calibration(**data)


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def test_percept_json_round_trip():
    percept = Percept(
        percept_id="p-1",
        status="ok",
        camera="cam_head",
        observations=[_observation()],
        geometry=Geometry(centroid=[0.1, 0.2, 0.3], uncertainty_m=0.02, views_used=2),
        provenance=_provenance(),
    )
    assert percept.schema_version == SCHEMA_VERSION
    restored = Percept.model_validate_json(percept.model_dump_json())
    assert restored == percept


def test_response_json_round_trip_and_is_plain_json():
    response = PerceiveResponse(
        request_id="req-1",
        percepts=[
            Percept(percept_id="p-1", status="ok", provenance=_provenance(), observations=[_observation()])
        ],
        fused=None,
        timings_ms={"fake": 1.2},
        warnings=["ambiguous_instances"],
    )
    payload = response.model_dump_json()
    json.loads(payload)  # must be valid JSON
    assert PerceiveResponse.model_validate_json(payload) == response


def test_request_round_trip_matches_spec_example():
    request = PerceiveRequest(
        request_id="req-1",
        images={"cam_head": ImageRef(encoding="png", data="AA==")},
        calibration={"cam_head": _calibration()},
        prompts=[Prompt(type="text", text="bottle", min_score=0.2, top_k=1)],
        options=PerceiveOptions(masks="rle", fuse=False),
    )
    assert PerceiveRequest.model_validate_json(request.model_dump_json()) == request


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_prompt_requires_payload_for_its_type():
    with pytest.raises(ValidationError, match="requires a non-empty 'text'"):
        Prompt(type="text")
    with pytest.raises(ValidationError, match="requires 'box'"):
        Prompt(type="box")
    with pytest.raises(ValidationError, match="requires 'point'"):
        Prompt(type="point")
    with pytest.raises(ValidationError, match="normalised"):
        Prompt(type="box", box=[0.5, 0.5, 0.2, 2.0])
    with pytest.raises(ValidationError, match="top_k"):
        Prompt(type="text", text="cup", top_k=0)


def test_calibration_matrix_shapes():
    with pytest.raises(ValidationError, match="K must be 3x3"):
        _calibration(K=[[1, 0], [0, 1]])
    with pytest.raises(ValidationError, match="T_world_cam must be 4x4"):
        _calibration(T_world_cam=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])


@pytest.mark.parametrize(
    "intrinsics",
    [
        [[0, 0, 320], [0, 0, 240], [0, 0, 1]],
        [[0, 0, 320], [0, 288, 240], [0, 0, 1]],
        [[288, 0, 320], [0, -288, 240], [0, 0, 1]],
        [[288, 1, 320], [0, 288, 240], [0, 0, 1]],
        [[288, 0, 320], [1, 288, 240], [0, 0, 1]],
        [[288, 0, 320], [0, 288, 240], [0.1, 0, 1]],
    ],
)
def test_calibration_rejects_nonphysical_pinhole_intrinsics(intrinsics):
    with pytest.raises(ValidationError, match="pinhole"):
        _calibration(K=intrinsics)


@pytest.mark.parametrize(
    "transform",
    [
        np.zeros((4, 4)).tolist(),
        np.diag([2.0, 1.0, 1.0, 1.0]).tolist(),
        np.diag([-1.0, 1.0, 1.0, 1.0]).tolist(),
        [[1, 0.1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0.1, 0, 0, 1]],
    ],
)
def test_calibration_rejects_nonrigid_camera_to_world(transform):
    with pytest.raises(ValidationError, match="rigid"):
        _calibration(T_world_cam=transform)


def test_calibration_accepts_float32_rounding_noise_without_changing_values():
    original = _calibration()
    intrinsics = np.asarray(original.K)
    intrinsics[0, 1] = 1e-10
    intrinsics[2, 0] = -1e-10
    transform = np.asarray(original.T_world_cam)
    transform[0, 0] += 2e-8
    transform[3, 0] = 1e-10
    calibration = _calibration(K=intrinsics.tolist(), T_world_cam=transform.tolist())
    assert calibration.K == intrinsics.tolist()
    assert calibration.T_world_cam == transform.tolist()


def test_calibration_validation_preserves_slim_worker_imports():
    code = '''
import importlib.abc
import sys

class RejectHeavy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        blocked = {"PIL", "torch", "cv2", "yaml", "fastapi", "percepta.geometry", "percepta.masks"}
        if any(fullname == name or fullname.startswith(name + ".") for name in blocked):
            raise AssertionError("unexpected dependency: " + fullname)

sys.meta_path.insert(0, RejectHeavy())
from percepta.schema import Calibration
import percepta.backends.protocol
import percepta.backends.multiview_protocol

Calibration(camera="cam", width=8, height=8, K=[[100,0,4],[0,100,4],[0,0,1]],
            T_world_cam=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
'''
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


def test_calibration_depth_must_match_resolution():
    depth = DepthRef(data="AA==", shape=(10, 10))
    with pytest.raises(ValidationError, match="DepthRef.shape"):
        _calibration(depth=depth)


def test_raw_image_requires_shape():
    with pytest.raises(ValidationError, match="requires shape"):
        ImageRef(encoding="raw", data="AA==")
    with pytest.raises(ValidationError):  # 2-tuple shape is rejected by the type itself
        ImageRef(encoding="raw", data="AA==", shape=(4, 4))


def test_request_requires_images_and_prompts():
    with pytest.raises(ValidationError, match="at least one camera"):
        PerceiveRequest(images={}, prompts=[Prompt(type="text", text="cup")])
    with pytest.raises(ValidationError, match="at least one prompt"):
        PerceiveRequest(images={"cam": ImageRef(encoding="png", data="AA==")}, prompts=[])


def test_request_rejects_raw_resolution_mismatch():
    with pytest.raises(ValidationError, match="spec §4.6"):
        PerceiveRequest(
            images={"cam_head": ImageRef(encoding="raw", data="AA==", shape=(240, 320, 3))},
            calibration={"cam_head": _calibration()},
            prompts=[Prompt(type="text", text="cup")],
        )


def test_calibration_key_must_match_camera():
    with pytest.raises(ValidationError, match="does not match"):
        PerceiveRequest(
            images={"other": ImageRef(encoding="png", data="AA==")},
            calibration={"other": _calibration()},
            prompts=[Prompt(type="text", text="cup")],
        )


def test_no_nan_or_inf_in_json():
    with pytest.raises(ValidationError):
        Geometry(centroid=[float("nan"), 0.0, 0.0])
    with pytest.raises(ValidationError):
        Geometry(top_z=float("inf"))
    with pytest.raises(ValidationError):
        Observation(object_id="x", score=float("nan"))


def test_mask_counts_must_sum_to_area():
    with pytest.raises(ValidationError, match="counts sum"):
        Mask(encoding="rle", size=(2, 2), counts=[1, 1])


def test_status_and_kind_literals_are_enforced():
    with pytest.raises(ValidationError):
        Percept(percept_id="p", status="weird", provenance=_provenance())
    with pytest.raises(ValidationError):
        Provenance(tool="t", tool_version="1", kind="not_a_kind")
