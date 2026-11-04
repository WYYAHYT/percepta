"""S1 (in-process) — the ``fake`` tool end-to-end, plus prompt modes and determinism."""

from __future__ import annotations

import numpy as np
import pytest

from percepta.schema import Prompt, ToolContext
from percepta.tools.fake import FakeTool, label_components
from tests import synth


def _tool(**config) -> FakeTool:
    tool = FakeTool()
    payload = {"tolerance": 40.0, "min_area_px": 16}
    payload.update(config)
    tool.load(payload)
    return tool


def _perceive(tool: FakeTool, images: dict[str, np.ndarray], prompts, **options):
    from percepta.schema import PerceiveOptions

    ctx = ToolContext(
        request_id="req-test",
        images=images,
        options=PerceiveOptions(**options),
    )
    return tool.perceive(ctx, prompts)


# ---------------------------------------------------------------------------
# S1: red square -> exactly one observation with exact geometry
# ---------------------------------------------------------------------------


def test_s1_red_square_exact_geometry():
    tool = _tool()
    image = synth.red_square_scene()  # 80x80 red square centred at (320, 240)
    percept = _perceive(tool, {"cam_head": image}, [Prompt(type="text", text="red block")])[0]

    assert percept.status == "ok"
    assert len(percept.observations) == 1
    obs = percept.observations[0]
    assert obs.area_px == 80 * 80
    assert obs.box_xyxy == pytest.approx([280.0, 200.0, 360.0, 280.0], abs=1e-6)
    assert obs.centroid_uv == pytest.approx([319.5, 239.5], abs=1e-6)
    assert obs.label == "red block"
    assert obs.score > 0.9
    assert obs.mask is not None and obs.mask.encoding == "rle"


def test_s1_only_one_observation_not_duplicated():
    tool = _tool()
    percepts = _perceive(
        tool, {"cam_head": synth.red_square_scene()}, [Prompt(type="text", text="red block")]
    )
    assert len(percepts) == 1  # one percept per (camera, prompt)
    assert len(percepts[0].observations) == 1  # one observation for one blob


# ---------------------------------------------------------------------------
# Provenance (§3.4)
# ---------------------------------------------------------------------------


def test_provenance_is_populated():
    tool = _tool()
    image = synth.red_square_scene()
    percept = _perceive(tool, {"cam_head": image}, [Prompt(type="text", text="red block")])[0]
    prov = percept.provenance
    assert prov.tool == "fake"
    assert prov.tool_version == "0.1.0"
    assert prov.kind == "fake"
    assert prov.camera == "cam_head"
    assert prov.latency_ms >= 0.0
    assert prov.image_sha256["cam_head"]
    assert prov.created_at.tzinfo is not None
    assert prov.cache_hit is False


# ---------------------------------------------------------------------------
# Prompt modes
# ---------------------------------------------------------------------------


def test_text_prompt_without_colour_uses_salient_mode():
    tool = _tool()
    percept = _perceive(tool, {"cam_head": synth.red_square_scene()}, [Prompt(type="text", text="bottle")])[0]
    assert percept.status == "ok"
    assert percept.extras["mode"] == "salient"
    assert len(percept.observations) == 1


def test_text_prompt_missing_colour_is_not_found():
    tool = _tool()
    percept = _perceive(tool, {"cam_head": synth.red_square_scene()}, [Prompt(type="text", text="blue block")])[0]
    assert percept.status == "not_found"
    assert percept.observations == []
    assert percept.message


def test_point_prompt_selects_containing_component():
    tool = _tool()
    image = synth.blank_image()
    image[100:140, 100:140] = synth.RED
    image[300:340, 400:440] = synth.GREEN
    percept = _perceive(tool, {"cam_head": image}, [Prompt(type="point", point=[410, 310])])[0]
    assert percept.status == "ok"
    assert len(percept.observations) == 1
    obs = percept.observations[0]
    assert obs.box_xyxy == pytest.approx([400.0, 300.0, 440.0, 340.0])


def test_box_prompt_restricts_search():
    tool = _tool()
    image = synth.blank_image()
    image[100:140, 100:140] = synth.RED
    image[300:340, 400:440] = synth.RED
    # Box covering only the top-left blob: normalised [cx, cy, w, h].
    box = [120 / 640, 120 / 480, 80 / 640, 80 / 480]
    percept = _perceive(tool, {"cam_head": image}, [Prompt(type="box", box=box)])[0]
    assert percept.status == "ok"
    assert len(percept.observations) == 1
    assert percept.observations[0].box_xyxy == pytest.approx([100.0, 100.0, 140.0, 140.0])


def test_top_k_and_min_score_are_applied():
    tool = _tool()
    image = synth.blank_image()
    image[50:90, 50:90] = synth.RED
    image[200:300, 200:300] = synth.RED  # bigger blob
    two = _perceive(
        tool, {"cam_head": image}, [Prompt(type="text", text="red block", top_k=2)]
    )[0]
    assert len(two.observations) == 2
    assert two.observations[0].area_px > two.observations[1].area_px  # sorted by score

    one = _perceive(
        tool, {"cam_head": image}, [Prompt(type="text", text="red block", top_k=1)]
    )[0]
    assert len(one.observations) == 1


# ---------------------------------------------------------------------------
# Options / cameras / determinism
# ---------------------------------------------------------------------------


def test_multiple_cameras_yield_one_percept_each():
    tool = _tool()
    images = {
        "cam_left": synth.red_square_scene(center=(100, 100)),
        "cam_right": synth.red_square_scene(center=(400, 300)),
    }
    percepts = _perceive(tool, images, [Prompt(type="text", text="red block")])
    assert [p.camera for p in percepts] == ["cam_left", "cam_right"]
    assert all(p.status == "ok" for p in percepts)


def test_masks_option_none_omits_mask_but_keeps_stats():
    tool = _tool()
    percept = _perceive(
        tool, {"cam_head": synth.red_square_scene()}, [Prompt(type="text", text="red block")], masks="none"
    )[0]
    obs = percept.observations[0]
    assert obs.mask is None
    assert obs.area_px == 6400
    assert obs.centroid_uv is not None


def test_object_ids_are_unique_and_stable_within_percept():
    tool = _tool()
    image = synth.blank_image()
    image[50:90, 50:90] = synth.RED
    image[200:300, 200:300] = synth.RED
    percept = _perceive(tool, {"cam_head": image}, [Prompt(type="text", text="red block", top_k=2)])[0]
    ids = [o.object_id for o in percept.observations]
    assert len(ids) == len(set(ids))
    replayed = _perceive(tool, {"cam_head": image}, [Prompt(type="text", text="red block", top_k=2)])[0]
    assert [o.box_xyxy for o in replayed.observations] == [o.box_xyxy for o in percept.observations]


# ---------------------------------------------------------------------------
# Connected components
# ---------------------------------------------------------------------------


def test_label_components_basic():
    mask = np.zeros((6, 6), dtype=bool)
    mask[0:2, 0:2] = True
    mask[4:6, 4:6] = True
    labels, count = label_components(mask)
    assert count == 2
    assert labels[0, 0] == labels[1, 1]
    assert labels[0, 0] != labels[4, 4]
    assert np.count_nonzero(labels) == 8


def test_label_components_connectivity_differs_on_diagonal():
    mask = np.zeros((4, 4), dtype=bool)
    mask[1, 1] = True
    mask[2, 2] = True
    assert label_components(mask, connectivity=4)[1] == 2
    assert label_components(mask, connectivity=8)[1] == 1


def test_label_components_empty():
    labels, count = label_components(np.zeros((5, 5), dtype=bool))
    assert count == 0
    assert not labels.any()
