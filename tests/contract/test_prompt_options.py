"""Global filtering defaults and explicit per-prompt overrides agree across transports."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from percepta.client import PerceptaClient, build_perceive_request
from percepta.config import AppConfig
from percepta.schema import PerceiveOptions, PerceiveRequest, Prompt


@pytest.mark.parametrize(
    ("overrides", "expected_min", "expected_top", "expected_count"),
    [({}, 0.97, 3, 2), ({"min_score": 0.2}, 0.2, 3, 3), ({"top_k": 1}, 0.97, 1, 1),
     ({"min_score": 0.2, "top_k": 1}, 0.2, 1, 1), ({"min_score": 0.98, "top_k": 2}, 0.98, 2, 1)],
)
def test_filter_defaults_and_explicit_prompt_values_agree_locally_and_over_http(
    client: TestClient, ci_config: AppConfig, overrides: dict, expected_min: float,
    expected_top: int, expected_count: int,
) -> None:
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    for index, red in enumerate((255, 235, 210)):
        image[4:10, 4 + 10 * index:10 + 10 * index] = (red, 0, 0)
    prompt = Prompt(type="text", text="red block", **overrides)
    original = prompt.model_dump()
    original_fields = prompt.model_fields_set.copy()
    options = {"tools": ["fake"], "fuse": False, "min_score": 0.97, "top_k": 3}
    request = build_perceive_request({"cam": image}, [prompt], options=options)
    sdk_payload = request.model_dump(mode="json")
    raw_payload = {**sdk_payload, "prompts": [{"type": "text", "text": "red block", **overrides}]}
    with PerceptaClient() as local_client:
        local = local_client.perceive_local({"cam": image}, [prompt], options=options, config=ci_config)
    responses = [local.model_dump(mode="json")]
    for payload in (sdk_payload, raw_payload):
        response = client.post("/v1/perceive", json=payload)
        assert response.status_code == 200, response.text
        responses.append(response.json())
    for response in responses:
        percept = response["percepts"][0]
        assert len(percept["observations"]) == expected_count
        assert percept["provenance"]["prompt"]["min_score"] == expected_min
        assert percept["provenance"]["prompt"]["top_k"] == expected_top
    assert request.prompts[0].min_score == expected_min
    assert request.prompts[0].top_k == expected_top
    restored = PerceiveRequest.model_validate_json(request.model_dump_json())
    assert restored.prompts == request.prompts
    assert prompt.model_dump() == original
    assert prompt.model_fields_set == original_fields


@pytest.mark.parametrize("bad", [{"min_score": -0.1}, {"min_score": 1.1}, {"top_k": 0}, {"top_k": -1}])
def test_invalid_global_filters_fail_validation_locally_and_over_http(client: TestClient, bad: dict) -> None:
    with pytest.raises(ValidationError):
        PerceiveOptions(**bad)
    request = build_perceive_request({"cam": np.zeros((4, 4, 3), dtype=np.uint8)}, ["red block"])
    payload = request.model_dump(mode="json")
    payload["options"].update(bad)
    response = client.post("/v1/perceive", json=payload)
    assert response.status_code == 400, response.text
    assert response.json()["error"] == "invalid_request"
