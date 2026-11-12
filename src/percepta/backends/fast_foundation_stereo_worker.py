"""Official Fast-FoundationStereo adapter for the existing isolated stereo worker."""

from __future__ import annotations

import hashlib
import json
import sys
from importlib.resources import files
from pathlib import Path

import numpy as np

from .foundation_stereo_worker import FoundationStereoAdapter, _offline

MANIFEST = files("percepta.backends").joinpath("fast_foundation_stereo_assets.json")
if not MANIFEST.is_file():  # Editable checkouts keep the single source manifest beside their uv project.
    MANIFEST = Path(__file__).resolve().parents[3] / "uv_envs/fast-foundation-stereo/assets.json"


def require_official_asset(path: str | Path, filename: str) -> Path:
    """Verify a local asset against the checked-in official research manifest.

    Args:
        path: Existing local model or configuration file.
        filename: Exact asset name in the deployment manifest.

    Returns:
        The resolved path whose complete SHA256 matches the pinned asset.

    Raises:
        ValueError: The asset name or its bytes differ from the pinned release.
        FileNotFoundError: The supplied asset is missing.

    Example:
        >>> require_official_asset("missing.pth", "model_best_bp2_serialize.pth")
        Traceback (most recent call last):
        ...
        FileNotFoundError: ...
    """
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Fast-FoundationStereo asset not found: {path}")
    assets = json.loads(MANIFEST.read_text())["checkpoint_assets"]
    expected = next((item["sha256"] for item in assets if item["filename"] == filename), None)
    if expected is None:
        raise ValueError(f"unrecognized official Fast-FoundationStereo asset: {filename}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError(f"official Fast-FoundationStereo asset hash mismatch: {path}")
    return path


class FastFoundationStereoAdapter(FoundationStereoAdapter):
    """Load the pinned research model and return original-resolution pixel disparity.

    Inputs are rectified uint8 RGB images. Output is d(v,u)=u_left-u_right;
    metric conversion remains in Stereo3DTool (Z=f*B/d, camera +z forward).

    Args:
        source_dir: Checkout of the manifest-pinned upstream source.
        checkpoint: Official hash-pinned serialized, structurally pruned model.
        model_config: Official cfg.yaml accompanying that checkpoint.
        valid_iters: Positive number of refinement iterations, matching the demo.
        max_disp: Positive multiple of 32 up to the checkpoint's fixed 416-pixel positional encoding.

    Example:
        >>> adapter = FastFoundationStereoAdapter(source_dir="missing", checkpoint="missing.pth")
        Traceback (most recent call last):
        ...
        FileNotFoundError: ...
    """

    def __init__(self, *, source_dir: str | None, checkpoint: str | None, model_config: str | None = None,
                 valid_iters: int = 8, max_disp: int = 416) -> None:
        _offline()
        if not source_dir or not Path(source_dir).is_dir():
            raise FileNotFoundError(f"Fast-FoundationStereo source_dir not found: {source_dir}")
        if type(valid_iters) is not int or valid_iters <= 0:
            raise ValueError("valid_iters must be a positive integer")
        if type(max_disp) is not int or max_disp <= 0 or max_disp % 32:
            raise ValueError("max_disp must be a positive multiple of 32")
        if max_disp > 416:
            raise ValueError("the official 23-36-37 checkpoint supports max_disp <= 416; its positional encoding is fixed")
        if not checkpoint:
            raise FileNotFoundError("Fast-FoundationStereo requires the official local checkpoint")
        checkpoint_path = require_official_asset(checkpoint, "model_best_bp2_serialize.pth")
        config_path = require_official_asset(model_config or checkpoint_path.with_name("cfg.yaml"), "cfg.yaml")
        sys.path.insert(0, str(Path(source_dir).resolve()))
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("Fast-FoundationStereo requires CUDA; no CPU/fallback model is supported")
        # Import registers upstream's legacy module aliases used by its serialized pruned model.
        import core.foundation_stereo  # noqa: F401
        from core.utils.utils import InputPadder
        from Utils import set_seed

        set_seed(0)
        # Upstream distributes a full pruned model, not a reconstructible state_dict. Only
        # the manifest-pinned official bytes reach this isolated, deployment-only pickle load.
        model = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        model.args.valid_iters, model.args.max_disp = valid_iters, max_disp
        self.model = model.cuda().eval()
        self.torch, self.padder_type = torch, InputPadder
        self.valid_iters = valid_iters
        self._info = {
            "model": "Fast-FoundationStereo",
            "torch_version": torch.__version__,
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
            "source_dir": str(Path(source_dir).resolve()),
            "checkpoint": str(checkpoint_path),
            "model_config": str(config_path),
            "max_disp": max_disp,
            "valid_iters": valid_iters,
            "device": "cuda:0",
            "device_name": torch.cuda.get_device_name(0),
            "model_latency_ms": None,
            "peak_cuda_memory_bytes": None,
        }
