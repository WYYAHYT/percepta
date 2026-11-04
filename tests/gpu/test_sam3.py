"""SAM3 smoke test (spec §14 S9) — needs the real package and gated weights.

Run explicitly with ``pytest -m gpu``. Never part of the default CI run.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from percepta.registry import ToolRegistry

CHECKPOINT = os.environ.get("PERCEPTA_SAM3_CHECKPOINT", "/models/sam3.pt")

pytestmark = pytest.mark.gpu


def _product_image() -> np.ndarray:
    """A crude stand-in for a real product photo: a bottle-ish blob on a plain background."""
    image = np.full((480, 640, 3), 235, dtype=np.uint8)
    image[120:400, 280:360] = (40, 90, 160)
    image[110:135, 295:345] = (230, 230, 230)  # cap
    return image


@pytest.mark.skipif(
    not os.path.exists(CHECKPOINT), reason=f"SAM3 checkpoint missing at {CHECKPOINT}"
)
def test_sam3_text_prompt_smoke():
    registry = ToolRegistry()
    tool = registry.instance(
        "sam3", {"checkpoint": CHECKPOINT, "resolution": 1008, "min_score": 0.2}
    )
    try:
        from percepta.schema import Prompt, ToolContext

        ctx = ToolContext(request_id="gpu-smoke", images={"cam_head": _product_image()})
        percepts = tool.perceive(ctx, [Prompt(type="text", text="bottle")])
        assert len(percepts) == 1
        percept = percepts[0]
        assert percept.status in {"ok", "not_found"}
        for observation in percept.observations:
            assert observation.mask is not None
            assert tuple(observation.area_px or 0 for _ in [0])
            assert observation.box_xyxy is not None
        if percept.status == "ok":
            height, width = _product_image().shape[:2]
            from percepta.masks import to_array

            mask = to_array(percept.observations[0].mask)
            assert mask.shape == (height, width)
    finally:
        tool.close()


def test_sam3_without_checkpoint_is_unavailable_not_not_found():
    from percepta.registry import ToolLoadError

    registry = ToolRegistry()
    with pytest.raises(ToolLoadError):
        registry.instance("sam3", {})
