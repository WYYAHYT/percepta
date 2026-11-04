"""Prompt construction and disambiguation (spec §8).

The only job here is turning an instruction like ``"pick up the red bottle"`` into a
:class:`~percepta.schema.Prompt` with a **noun phrase** (never a sentence), plus the
default multi-instance disambiguation policy of §8.3.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Any

from .schema import Observation, Percept, Prompt

__all__ = [
    "from_instruction",
    "NOUN_PATTERN_HINTS",
    "extract_noun_phrase",
    "is_noun_phrase",
    "disambiguate",
]

#: Verb frames the spec calls out; we grab whatever follows the determiner.
_NOUN_FRAME = re.compile(
    r"\b(?:pick\s+up|grasp|grab|move|take|lift|put|place|push|open|close|"
    r"point\s+at|look\s+at|segment|find)\b"
    r"(?:\s+(?:the|a|an|this|that|my))?\s+(?P<noun>[a-z0-9][a-z0-9 _\-]*)",
    re.IGNORECASE,
)

#: Trailing words that are relations/qualifiers, not part of the noun phrase.
_TRAILING = re.compile(r"\s+(?:to|into|onto|on|in|from|at|with|and|then|now|please)\b.*$", re.I)

#: A prompt that starts with a verb is a sentence, not a noun phrase (§8.4).
_VERB_START = re.compile(
    r"^(?:pick|grasp|grab|move|take|lift|put|place|push|open|close|find|segment|look|go|"
    r"bring|hand|give|is|are|was|were|do|does|can|could|should|would|will)\b",
    re.IGNORECASE,
)

#: Words kept for the "noun-ish" heuristic; purely a warning heuristic, never a hard gate.
NOUN_PATTERN_HINTS = frozenset(
    {
        "bottle",
        "cup",
        "bowl",
        "box",
        "can",
        "block",
        "cube",
        "banana",
        "apple",
        "orange",
        "sponge",
        "cloth",
        "towel",
        "drawer",
        "handle",
        "button",
        "switch",
        "door",
        "ball",
        "pen",
        "marker",
        "spoon",
        "fork",
        "knife",
        "plate",
        "mug",
        "pot",
        "pan",
        "bag",
        "camera",
        "robot",
        "table",
        "chair",
        "shelf",
    }
)


def extract_noun_phrase(instruction: str) -> str | None:
    """Best-effort noun phrase from a manipulation instruction.

    ``"pick up the red bottle and put it in the bin"`` -> ``"red bottle"``.
    Returns ``None`` when no frame matches (the caller then reports ``not_found``).
    """
    if not instruction or not instruction.strip():
        return None
    cleaned = instruction.strip().strip(".!?").strip()
    match = _NOUN_FRAME.search(cleaned)
    if match:
        noun = _TRAILING.sub("", match.group("noun")).strip()
        noun = noun.strip(".,!?;:")
        return noun or None
    # Fall back to "the last noun-looking token run" for frameless instructions.
    tokens = [t for t in re.split(r"[\s,]+", cleaned) if t and t.lower() not in _STOPWORDS]
    if tokens:
        return tokens[-1].strip(".,!?;:").lower() or None
    return None


_STOPWORDS = frozenset(
    {"the", "a", "an", "this", "that", "it", "them", "please", "now", "then", "and", "to"}
)


def is_noun_phrase(text: str) -> bool:
    """Lightweight validator for §8.4: reject sentences, accept short noun phrases."""
    if not text or not text.strip():
        return False
    stripped = text.strip()
    if len(stripped.split()) > 6:
        return False
    if stripped.endswith(("?", "!")):
        return False
    return not bool(_VERB_START.match(stripped))


def from_instruction(
    instruction: str,
    target: str | None = None,
    *,
    prompt_llm: Any = None,  # reserved hook, v0.1 does not implement it (§8.1)
    min_score: float = 0.2,
    top_k: int = 1,
    box: Sequence[float] | None = None,
    point: Sequence[int] | None = None,
) -> Prompt | None:
    """Build a :class:`Prompt` from a free-form instruction.

    Returns ``None`` (never raises) when nothing usable can be extracted; callers are
    expected to surface that as a ``not_found`` warning, per §8.1.
    """
    if prompt_llm is not None:  # pragma: no cover - documented no-op in v0.1
        raise NotImplementedError("prompt_llm hook is reserved for a later version")
    if box is not None:
        return Prompt(
            type="box", box=[float(v) for v in box], min_score=min_score, top_k=top_k
        )
    if point is not None:
        return Prompt(
            type="point",
            point=[int(point[0]), int(point[1])],
            min_score=min_score,
            top_k=top_k,
        )
    text = (target or extract_noun_phrase(instruction) or "").strip()
    if not text:
        return None
    return Prompt(type="text", text=text, min_score=min_score, top_k=top_k)


# ---------------------------------------------------------------------------
# §8.3 multi-instance disambiguation
# ---------------------------------------------------------------------------


def _instance_world_z(percept: Percept) -> float | None:
    if percept.geometry is None or percept.geometry.centroid is None:
        return None
    if len(percept.geometry.centroid) < 3:
        return None
    return float(percept.geometry.centroid[2])


def disambiguate(
    percept: Percept,
    *,
    context: dict[str, Any] | None = None,
    table_z: float | None = None,
    table_band_m: float = 0.15,
    ee_position_world: Sequence[float] | None = None,
) -> tuple[list[Observation], bool]:
    """Apply the default §8.3 policy; returns ``(kept_observations, ambiguous)``.

    Order of preference:

    1. sort by ``score`` (the caller usually already did this);
    2. keep instances whose world ``z`` is within ``table_band_m`` of ``table_z`` when a
       calibration is available;
    3. prefer the instance nearest ``context["ee_position_world"]``;
    4. if several remain, return all of them and let the caller mark ``ambiguous_instances``.
    """
    context = context or {}
    observations = sorted(percept.observations, key=lambda o: o.score, reverse=True)
    if len(observations) <= 1:
        return observations, False

    if ee_position_world is None:
        ee_position_world = context.get("ee_position_world")

    kept = observations
    band = float(context.get("table_band_m", table_band_m))
    if table_z is not None:
        filtered = []
        for obs in kept:
            z = None
            if obs.attributes.get("centroid_z") is not None:
                try:
                    z = float(obs.attributes["centroid_z"])
                except (TypeError, ValueError):
                    z = None
            if z is None:
                filtered.append(obs)  # cannot judge: keep rather than drop
            elif abs(z - float(table_z)) <= band:
                filtered.append(obs)
        if filtered:
            kept = filtered

    if ee_position_world is not None and len(kept) > 1:
        import math

        def distance(obs: Observation) -> float:
            cx = obs.attributes.get("centroid_x")
            cy = obs.attributes.get("centroid_y")
            if cx is None or cy is None:
                return float("inf")
            dx = float(cx) - float(ee_position_world[0])
            dy = float(cy) - float(ee_position_world[1])
            return math.hypot(dx, dy)

        best = min(kept, key=distance)
        if distance(best) != float("inf"):
            if distance(best) < float("inf"):
                # Keep the nearest plus any exact ties, so the caller can still report
                # ambiguity instead of silently committing to one instance.
                nearest = [o for o in kept if distance(o) == distance(best)]
                kept = nearest

    return kept, len(kept) > 1


def ambiguous_warning(observations: Iterable[Observation]) -> list[str]:
    return ["ambiguous_instances"] if len(list(observations)) > 1 else []
