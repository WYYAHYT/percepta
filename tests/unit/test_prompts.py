"""Prompt extraction and disambiguation (spec §8)."""

from __future__ import annotations

import pytest

from percepta.prompts import (
    disambiguate,
    extract_noun_phrase,
    from_instruction,
    is_noun_phrase,
)
from percepta.schema import Geometry, Observation, Percept, Provenance


@pytest.mark.parametrize(
    "instruction,expected",
    [
        ("pick up the red bottle", "red bottle"),
        ("grasp the cube", "cube"),
        ("move the blue block to the left", "blue block"),
        ("take the yellow sponge and put it in the drawer", "yellow sponge"),
        ("Please pick up the mug.", "mug"),
        ("bottle", "bottle"),
    ],
)
def test_extract_noun_phrase(instruction, expected):
    assert extract_noun_phrase(instruction) == expected


@pytest.mark.parametrize("instruction", ["", "   ", None])
def test_extract_noun_phrase_empty(instruction):
    assert extract_noun_phrase(instruction) is None


def test_is_noun_phrase_heuristics():
    assert is_noun_phrase("red bottle")
    assert is_noun_phrase("camera")
    assert not is_noun_phrase("pick up the bottle")  # starts with a verb
    assert not is_noun_phrase("where is the bottle?")  # a question
    assert not is_noun_phrase("")
    assert not is_noun_phrase("a very long descriptive phrase with too many words here")


def test_from_instruction_prefers_explicit_target():
    prompt = from_instruction("pick up the red bottle", target="bottle")
    assert prompt is not None
    assert prompt.type == "text"
    assert prompt.text == "bottle"


def test_from_instruction_derives_noun_phrase():
    prompt = from_instruction("pick up the red bottle", min_score=0.4, top_k=2)
    assert prompt is not None
    assert (prompt.text, prompt.min_score, prompt.top_k) == ("red bottle", 0.4, 2)


def test_from_instruction_returns_none_when_nothing_extractable():
    assert from_instruction("") is None


def test_from_instruction_geometry_prompts():
    box_prompt = from_instruction("", box=[0.5, 0.5, 0.2, 0.2])
    assert box_prompt is not None
    assert box_prompt.type == "box"
    assert box_prompt.box == [0.5, 0.5, 0.2, 0.2]

    point_prompt = from_instruction("", point=[12, 34])
    assert point_prompt is not None
    assert point_prompt.type == "point"
    assert point_prompt.point == [12, 34]


def test_from_instruction_llm_hook_is_reserved():
    with pytest.raises(NotImplementedError):
        from_instruction("pick up the cup", prompt_llm=object())


# ---------------------------------------------------------------------------
# §8.3 disambiguation
# ---------------------------------------------------------------------------


def _percept(observations, centroid=None) -> Percept:
    return Percept(
        percept_id="p",
        status="ok",
        observations=observations,
        geometry=None if centroid is None else Geometry(centroid=list(centroid)),
        provenance=Provenance(tool="fake", tool_version="0.1.0", kind="fake"),
    )


def _obs(score, object_id="o", **attributes) -> Observation:
    return Observation(
        object_id=object_id, score=score, attributes={k: v for k, v in attributes.items()}
    )


def test_disambiguate_single_observation_is_never_ambiguous():
    kept, ambiguous = disambiguate(_percept([_obs(0.9)]))
    assert len(kept) == 1 and ambiguous is False


def test_disambiguate_table_band_filters_by_height():
    percept = _percept(
        [
            _obs(0.9, "a", centroid_z=0.05),
            _obs(0.8, "b", centroid_z=1.20),  # on a shelf, not the table
        ]
    )
    kept, ambiguous = disambiguate(percept, table_z=0.0, table_band_m=0.15)
    assert [o.object_id for o in kept] == ["a"]
    assert ambiguous is False


def test_disambiguate_prefers_nearest_to_end_effector():
    percept = _percept(
        [
            _obs(0.9, "far", centroid_x=1.0, centroid_y=1.0),
            _obs(0.5, "near", centroid_x=0.1, centroid_y=0.1),
        ]
    )
    kept, ambiguous = disambiguate(percept, context={"ee_position_world": [0.0, 0.0, 0.2]})
    assert [o.object_id for o in kept] == ["near"]
    assert ambiguous is False


def test_disambiguate_reports_ambiguity_when_tied():
    percept = _percept([_obs(0.9, "a"), _obs(0.9, "b")])
    kept, ambiguous = disambiguate(percept)
    assert len(kept) == 2
    assert ambiguous is True
