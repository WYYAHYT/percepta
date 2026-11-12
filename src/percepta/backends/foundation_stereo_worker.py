"""Isolated FoundationStereo worker; stdout is reserved for the stereo wire protocol."""

from __future__ import annotations

import importlib
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np

from .stereo_protocol import (
    MAX_REQUEST_BYTES,
    ProtocolError,
    StereoResponse,
    decode_request,
    encode_response,
    read_frame,
    write_frame,
)


def _offline() -> None:
    # Upstream's edgenext_small requests pretrained timm weights even with a full
    # checkpoint. Deployment must populate its HF cache; never download or monkeypatch
    # timm to silently substitute an uninitialized model.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"


class FoundationStereoAdapter:
    """Rectified RGB pairs -> pixel disparity d(v,u) = u_left - u_right, not metric depth."""

    def __init__(
        self, *, source_dir: str | None, checkpoint: str | None,
        model_config: str | None = None, valid_iters: int = 32,
    ) -> None:
        _offline()
        if not source_dir or not Path(source_dir).is_dir():
            raise FileNotFoundError(f"FoundationStereo source_dir not found: {source_dir}")
        if not checkpoint or not Path(checkpoint).is_file():
            raise FileNotFoundError(f"FoundationStereo checkpoint not found: {checkpoint}")
        model_config = model_config or str(Path(checkpoint).with_name("cfg.yaml"))
        if not Path(model_config).is_file():
            raise FileNotFoundError(f"FoundationStereo model_config not found: {model_config}")
        if type(valid_iters) is not int or valid_iters <= 0:
            raise ValueError("valid_iters must be a positive integer")
        sys.path.insert(0, str(Path(source_dir).resolve()))
        import codecs
        from collections import defaultdict
        from typing import Any

        import torch
        from numpy.core.multiarray import scalar
        from omegaconf import DictConfig, ListConfig, OmegaConf
        from omegaconf.base import ContainerMetadata, Metadata
        from omegaconf.nodes import AnyNode

        if not torch.cuda.is_available():
            raise RuntimeError("FoundationStereo requires CUDA; no CPU/fallback model is supported")
        from core.foundation_stereo import FoundationStereo
        from core.utils.utils import InputPadder

        cfg = OmegaConf.load(model_config)
        if "vit_size" not in cfg:
            cfg.vit_size = "vitl"
        model = FoundationStereo(cfg)
        # The official training checkpoint includes NumPy metrics and OmegaConf metadata.
        torch.serialization.add_safe_globals([
            codecs.encode, scalar, np.dtype, np.dtypes.Float64DType,
            DictConfig, ListConfig, ContainerMetadata, Metadata, AnyNode,
            Any, dict, list, int, defaultdict,
        ])
        try:
            ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
        finally:
            torch.serialization.clear_safe_globals()
        model.load_state_dict(ckpt["model"], strict=True)
        self.model = model.cuda().eval()
        self.torch, self.padder_type = torch, InputPadder
        self.valid_iters = valid_iters
        self._info = {
            "model": "FoundationStereo",
            "torch_version": torch.__version__,
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
            "source_dir": str(Path(source_dir).resolve()),
            "checkpoint": str(Path(checkpoint).resolve()),
            "model_config": str(Path(model_config).resolve()),
            "vit_size": str(cfg.vit_size),
            "max_disp": int(cfg.max_disp),
            "valid_iters": valid_iters,
            "device": "cuda:0",
            "device_name": torch.cuda.get_device_name(0),
            "model_latency_ms": None,
            "peak_cuda_memory_bytes": None,
        }

    def infer_pair(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        torch = self.torch
        with torch.inference_mode():
            # RGB values remain 0..255; no resize, channel swap, normalization or clamp.
            image1 = torch.from_numpy(left).permute(2, 0, 1)[None].float().cuda()
            image2 = torch.from_numpy(right).permute(2, 0, 1)[None].float().cuda()
            padder = self.padder_type(image1.shape, divis_by=32, force_square=False)
            image1, image2 = padder.pad(image1, image2)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                disparity = self.model.forward(
                    image1, image2, iters=self.valid_iters, test_mode=True,
                )
            end.record()
            end.synchronize()
            # Events time only forward(), excluding input transfer/padding/CPU transfer/IPC.
            self._info["model_latency_ms"] = float(start.elapsed_time(end))
            self._info["peak_cuda_memory_bytes"] = int(torch.cuda.max_memory_allocated())
            result = padder.unpad(disparity.float()).cpu().numpy()
        if result.shape != (1, 1, *left.shape[:2]):
            raise RuntimeError(f"unexpected FoundationStereo output shape {result.shape}")
        return np.ascontiguousarray(result[0, 0], dtype=np.float32)

    def describe(self) -> dict:
        return dict(self._info)

    def close(self) -> None:
        self.model = None
        self.torch.cuda.empty_cache()


def build_backend(config: dict):
    """Only trusted deployment argv may select a module:factory and keyword arguments."""
    _offline()
    spec = config.get("backend_factory")
    if spec is None:
        return FoundationStereoAdapter(**{
            key: config.get(key) for key in ("source_dir", "checkpoint", "model_config")
        }, valid_iters=config.get("valid_iters", 32))
    if not isinstance(spec, str) or not all(spec.partition(":")[::2]) or ":" not in spec:
        raise ValueError("backend_factory must be module:attribute")
    module, _, attribute = spec.partition(":")
    return getattr(importlib.import_module(module), attribute)(**config.get("backend_args", {}))


class _RequestReader:
    """Enforce the stereo outer length before read_frame allocates/reads a payload."""

    def __init__(self, stream):
        self.stream = stream

    def read(self, size: int) -> bytes:
        if size > MAX_REQUEST_BYTES:
            raise ProtocolError("stereo request exceeds length limit")
        return self.stream.read(size)


def serve(config: dict, stdin, stdout) -> int:
    backend = None
    try:
        backend = build_backend(config)
        write_frame(stdout, encode_response(StereoResponse("ready", diagnostics=backend.describe())))
        reader = _RequestReader(stdin)
        while True:
            payload = read_frame(reader)
            if payload is None:
                return 0
            request = decode_request(payload)
            if request.op == "shutdown":
                return 0
            disparity = backend.infer_pair(request.left, request.right)
            if not isinstance(disparity, np.ndarray) or disparity.shape != request.left.shape[:2]:
                raise ProtocolError("adapter disparity does not match input shape")
            write_frame(stdout, encode_response(StereoResponse(
                "ok", disparity=disparity, diagnostics=backend.describe(),
            )))
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        try:
            write_frame(stdout, encode_response(StereoResponse(
                "error", error=f"{type(exc).__name__}: {exc}"[:8192],
            )))
        except (OSError, ProtocolError):
            pass
        return 1  # inference/protocol errors are terminal, never a fallback or not_found
    finally:
        if backend is not None:
            try:
                backend.close()
            except Exception:
                traceback.print_exc(file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    _offline()
    if len(argv) != 1:
        print("expected one deployment config JSON argument", file=sys.stderr)
        return 2
    try:
        config = json.loads(argv[0])
        if not isinstance(config, dict):
            raise ValueError("deployment config must be an object")
    except ValueError as exc:
        print(f"invalid deployment config: {exc}", file=sys.stderr)
        return 2
    # Keep a private protocol fd, then redirect BOTH Python prints and native writes to
    # fd 1 (including subprocesses started by upstream code) to the drained stderr pipe.
    with os.fdopen(os.dup(sys.stdout.fileno()), "wb") as wire:
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        sys.stdout = sys.stderr
        return serve(config, sys.stdin.buffer, wire)


if __name__ == "__main__":  # pragma: no cover - exercised in real subprocess tests
    raise SystemExit(main())
