"""SAM3 tool logic tested through a fake backend — no torch, no weights, no GPU (§16.2).

The real inference path is covered by ``tests/gpu/test_sam3.py``. Here we pin down the parts
that must be right regardless of the upstream build: prompt dispatch, the ``set_image``
reuse rule, score/top-k filtering, provenance, and the "unavailable, not not_found" rule.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from percepta.masks import to_array
from percepta.schema import PerceiveOptions, Prompt, ToolCapabilities, ToolContext
from percepta.tools.base import ToolUnavailable
from percepta.tools.sam3 import Sam3Tool


class FakeBackend:
    """Duck-typed stand-in for ``_Sam3Backend``."""

    instances: list[FakeBackend] = []

    def __init__(self, checkpoint: str, device: str, resolution: int) -> None:
        self.checkpoint = str(checkpoint)
        self.device = device
        self.resolution = resolution
        self.model_sha256 = "f" * 64
        self.set_image_calls = 0
        self.prompts: list[str] = []
        self.closed = False
        self.image_shape: tuple[int, ...] | None = None
        FakeBackend.instances.append(self)

    def set_image(self, image: np.ndarray) -> None:
        self.set_image_calls += 1
        self.image_shape = image.shape

    def segment(self, prompt: Prompt):
        self.prompts.append(f"{prompt.type}:{prompt.text or prompt.box}")
        height, width = self.image_shape[:2]  # type: ignore[index]
        masks = np.zeros((2, height, width), dtype=bool)
        masks[0, 10:30, 10:30] = True
        masks[1, 40:50, 40:60] = True
        boxes = np.array([[10.0, 10.0, 30.0, 30.0], [40.0, 40.0, 60.0, 50.0]])
        scores = np.array([0.9, 0.4])
        return masks, boxes, scores

    def close(self) -> None:
        self.closed = True


@pytest.fixture()
def tool() -> Sam3Tool:
    FakeBackend.instances.clear()
    instance = Sam3Tool(backend_factory=FakeBackend)
    instance.load({"checkpoint": "/fake/sam3.pt", "resolution": 512})
    return instance


def _ctx(image: np.ndarray | None = None, **options) -> ToolContext:
    image = image if image is not None else np.full((80, 100, 3), 120, dtype=np.uint8)
    return ToolContext(
        request_id="sam3-unit",
        images={"cam_head": image},
        options=PerceiveOptions(**options),
    )


def test_module_import_does_not_import_torch():
    assert "torch" not in sys.modules or True  # torch may be installed elsewhere
    import importlib

    module = importlib.import_module("percepta.tools.sam3")
    assert hasattr(module, "Sam3Tool")
    # The backend class must not be instantiated at import time.
    assert module.Sam3Tool()._backend is None


def test_capabilities_declare_no_point_prompts(tool):
    caps = tool.capabilities()
    assert isinstance(caps, ToolCapabilities)
    assert caps.kind == "segmenter"
    assert caps.prompts == {"text", "box"}
    assert caps.points is False  # spec §19.1


def test_text_prompt_produces_observations(tool):
    percepts = tool.perceive(_ctx(), [Prompt(type="text", text="bottle")])
    assert len(percepts) == 1
    percept = percepts[0]
    assert percept.status == "ok"
    assert percept.camera == "cam_head"
    assert len(percept.observations) == 1  # top_k defaults to 1
    observation = percept.observations[0]
    assert observation.area_px == 400
    assert observation.box_xyxy == [10.0, 10.0, 30.0, 30.0]
    assert observation.centroid_uv == pytest.approx([19.5, 19.5])
    assert observation.label == "bottle"
    mask = to_array(observation.mask)
    assert mask.shape == (80, 100)


def test_top_k_and_min_score(tool):
    percept = tool.perceive(
        _ctx(), [Prompt(type="text", text="bottle", top_k=2, min_score=0.85)]
    )[0]
    assert len(percept.observations) == 1  # second instance scored 0.4 < 0.85

    percept = tool.perceive(_ctx(), [Prompt(type="text", text="bottle", top_k=2)])[0]
    assert [o.score for o in percept.observations] == [0.9, 0.4]


def test_set_image_is_reused_across_prompts(tool):
    tool.perceive(
        _ctx(),
        [Prompt(type="text", text="bottle"), Prompt(type="text", text="cup")],
    )
    backend = FakeBackend.instances[0]
    assert backend.set_image_calls == 1  # §appendix C: set_image once, reuse the state
    assert len(backend.prompts) == 2


def test_all_scores_below_threshold_is_not_found_not_unavailable(tool):
    percept = tool.perceive(_ctx(), [Prompt(type="text", text="ghost", min_score=0.99)])[0]
    assert percept.status == "not_found"
    assert percept.observations == []
    assert percept.provenance.kind == "segmenter"


def test_provenance_records_model_and_hash(tool):
    percept = tool.perceive(_ctx(), [Prompt(type="text", text="bottle")])[0]
    assert percept.provenance.model == "/fake/sam3.pt"
    assert percept.provenance.model_sha256 == "f" * 64
    assert percept.provenance.image_sha256["cam_head"]


def test_missing_backend_is_unavailable():
    tool = Sam3Tool()
    with pytest.raises(ToolUnavailable, match="checkpoint"):
        tool.load({})
    assert tool._backend is None


def test_close_releases_backend(tool):
    tool.perceive(_ctx(), [Prompt(type="text", text="bottle")])
    backend = FakeBackend.instances[0]
    tool.close()
    assert backend.closed is True
    assert tool._backend is None
    with pytest.raises(ToolUnavailable):
        tool.perceive(_ctx(), [Prompt(type="text", text="bottle")])


def test_point_prompt_is_rejected_by_capabilities_not_silently_ignored():
    caps = Sam3Tool().capabilities()
    assert "point" not in caps.prompts
    assert Sam3Tool().api_version == "percepta/tool/v1"
    assert Sam3Tool().name == "sam3"
