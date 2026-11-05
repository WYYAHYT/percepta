"""Real SAM3 over the subprocess backend (spec §14 S9).

Skipped unless a checkpoint and the isolated interpreter are both present, so the default
`pytest -m "not gpu"` run stays GPU-free.

    PERCEPTA_SAM3_CHECKPOINT=checkpoints/sam3.pt \
    PERCEPTA_SAM3_PYTHON=.venvs/sam3/bin/python \
    PERCEPTA_SAM3_BPE=checkpoints/bpe_simple_vocab_16e6.txt.gz \
    pytest -m gpu tests/gpu -q

Design notes worth keeping:

* A **real photograph** is required. SAM3 is trained on photos and legitimately finds nothing
  in a flat vector-drawing scene, so a synthetic fixture would "pass" while proving nothing.
* The image goes in as HWC numpy and is converted to PIL inside the adapter. Handing HWC
  numpy straight to ``Sam3Processor.set_image`` silently corrupts the shape (it becomes
  C=H, H=W, W=3) and every prompt returns zero instances.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from percepta.schema import Prompt, ToolContext

pytestmark = pytest.mark.gpu

CHECKPOINT = os.environ.get("PERCEPTA_SAM3_CHECKPOINT", "checkpoints/sam3.pt")
WORKER_PYTHON = os.environ.get("PERCEPTA_SAM3_PYTHON", ".venvs/sam3/bin/python")
BPE_PATH = os.environ.get("PERCEPTA_SAM3_BPE", "checkpoints/bpe_simple_vocab_16e6.txt.gz")
PHOTO = os.environ.get("PERCEPTA_SAM3_PHOTO", "")

_missing = [
    f"{name}={value}"
    for name, value in (
        ("checkpoint", CHECKPOINT),
        ("worker python", WORKER_PYTHON),
        ("bpe path", BPE_PATH),
        ("photo", PHOTO),
    )
    if not value or not Path(value).exists()
]
if _missing:
    pytest.skip("missing real-SAM3 assets: " + ", ".join(_missing), allow_module_level=True)


def _load_photo() -> np.ndarray:
    from PIL import Image

    with Image.open(PHOTO) as handle:
        return np.array(handle.convert("RGB"))


@pytest.fixture(scope="module")
def tool():
    from percepta.tools.sam3 import Sam3Tool

    instance = Sam3Tool()
    instance.load(
        {
            "backend": "subprocess",
            "python": WORKER_PYTHON,
            "checkpoint": CHECKPOINT,
            "bpe_path": BPE_PATH,
            "device": "cuda",
            "resolution": 1008,
            "confidence_threshold": 0.3,
            "cuda_visible_devices": os.environ.get("PERCEPTA_SAM3_GPU", "2"),
            "timeout_s": 900,
        }
    )
    try:
        yield instance
    finally:
        instance.close()


def _ctx(image: np.ndarray) -> ToolContext:
    return ToolContext(request_id="gpu-sam3", images={"cam_head": image})


def test_sam3_finds_objects_in_a_real_photo(tool):
    image = _load_photo()
    percept = tool.perceive(_ctx(image), [Prompt(type="text", text="flower", top_k=3)])[0]

    assert percept.status == "ok", percept.message
    assert percept.observations, "real SAM3 found nothing in a real photo"
    height, width = image.shape[:2]
    for observation in percept.observations:
        assert observation.mask is not None
        assert tuple(observation.mask.size or ()) == (height, width)
        assert observation.area_px and observation.area_px > 0
        assert observation.box_xyxy is not None
        # boxes must stay inside the frame
        x0, y0, x1, y1 = observation.box_xyxy
        assert 0 <= x0 < x1 <= width + 1
        assert 0 <= y0 < y1 <= height + 1


def test_sam3_respects_top_k_and_ordering(tool):
    image = _load_photo()
    percept = tool.perceive(_ctx(image), [Prompt(type="text", text="flower", top_k=2)])[0]
    assert len(percept.observations) <= 2
    scores = [observation.score for observation in percept.observations]
    assert scores == sorted(scores, reverse=True), "observations must be score-ordered"


def test_sam3_records_the_weight_hash_in_provenance(tool):
    """A subprocess worker keeps the model in another process; the parent must still record
    the exact bytes, otherwise a result cannot be traced to a checkpoint."""
    percept = tool.perceive(_ctx(_load_photo()), [Prompt(type="text", text="flower")])[0]
    assert percept.provenance.model_sha256
    assert len(percept.provenance.model_sha256) == 64


def test_sam3_unrecognised_prompt_is_not_found_not_an_error(tool):
    percept = tool.perceive(
        _ctx(_load_photo()), [Prompt(type="text", text="quantum flux capacitor", min_score=0.99)]
    )[0]
    assert percept.status in {"ok", "not_found"}
    assert percept.status != "unavailable"
