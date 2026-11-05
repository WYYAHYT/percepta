"""percepta — a pluggable observation layer for robot agents.

The public surface is intentionally small: the frozen data model (§3), the mask codecs,
the tool registry, config, and the SDK client/CLI. Anything model-specific lives behind a
tool plugin so that CI runs with ``fake`` only (spec §16.2).

Typical use (spec appendix D)::

    from percepta import Prompt, PerceptaClient

    client = PerceptaClient("http://127.0.0.1:8790")
    resp = client.perceive(images={"cam_head": "head.png"}, prompts=[Prompt(type="text", text="bottle")])
"""

from __future__ import annotations

from typing import Any

from .schema import (
    SCHEMA_VERSION,
    TOOL_API_VERSION,
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
    ToolCapabilities,
    ToolContext,
    VerificationResult,
)

__version__ = "0.1.0"

#: Names re-exported lazily from submodules that pull optional or heavy dependencies.
#:
#: Why lazy: a model *worker* runs in an isolated virtualenv that may contain nothing but
#: numpy, and it needs ``percepta.backends.protocol`` + ``percepta.schema`` only. Importing
#: this package must therefore stay cheap — an eager ``from .config import ...`` drags in
#: pyyaml, and ``.masks`` drags in pillow, which breaks every slim worker
#: (``docs/model-isolation.md`` §6). Only ``schema`` (pydantic + stdlib) is eager.
_LAZY = {
    # config / registry
    "AppConfig": ".config",
    "load_config": ".config",
    "ToolRegistry": ".registry",
    "default_registry": ".registry",
    # masks (appendix B)
    "b64_bytes": ".masks",
    "bytes_b64": ".masks",
    "decode_image": ".masks",
    "decode_rle": ".masks",
    "encode_image": ".masks",
    "encode_rle": ".masks",
    "mask_area": ".masks",
    "mask_box_xyxy": ".masks",
    "mask_centroid": ".masks",
    "mask_iou": ".masks",
    "to_array": ".masks",
    "to_mask": ".masks",
    # heavier submodules
    "PerceptaClient": ".client",
    "compare": ".verification",
    "Geo": ".geometry",
}

__all__ = [
    "__version__",
    "SCHEMA_VERSION",
    "TOOL_API_VERSION",
    # schema §3
    "Calibration",
    "DepthRef",
    "Geometry",
    "ImageRef",
    "Mask",
    "Observation",
    "Percept",
    "PerceiveOptions",
    "PerceiveRequest",
    "PerceiveResponse",
    "Provenance",
    "Prompt",
    "ToolCapabilities",
    "ToolContext",
    "VerificationResult",
    # masks appendix B
    "b64_bytes",
    "bytes_b64",
    "decode_image",
    "decode_rle",
    "encode_image",
    "encode_rle",
    "mask_area",
    "mask_box_xyxy",
    "mask_centroid",
    "mask_iou",
    "to_array",
    "to_mask",
    # config / registry
    "AppConfig",
    "ToolRegistry",
    "default_registry",
    "load_config",
    # lazily imported
    "PerceptaClient",
    "compare",
]


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        import importlib

        module = importlib.import_module(_LAZY[name], __name__)
        try:
            value = getattr(module, name)
        except AttributeError:
            value = module
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))
