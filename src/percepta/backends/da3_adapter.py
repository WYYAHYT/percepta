"""DA3-BASE worker adapter; metric scale comes from calibrated camera translations."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


def validate_image_shape(image_shape: tuple[int, int], process_res: int = 504) -> None:
    """Reject images whose DA3 resize would leave fewer than one 14-pixel patch.

    Args:
        image_shape: Positive input image height and width in pixels.
        process_res: Longest image edge used by the deployed DA3 processor.

    Raises:
        ValueError: The processed short edge would round down to zero patches.

    Example:
        >>> validate_image_shape((480, 640))
        >>> validate_image_shape((1, 1000))
        Traceback (most recent call last):
        ...
        ValueError: DA3 resize must retain at least one 14-pixel patch in each dimension
    """
    scale = process_res / max(image_shape)
    resized = [max(1, int(round(dimension * scale))) for dimension in image_shape]
    if any((dimension + 7) // 14 == 0 for dimension in resized):
        raise ValueError("DA3 resize must retain at least one 14-pixel patch in each dimension")


class DA3Adapter:
    """Run local DA3-BASE on three calibrated RGB views without downloading assets.

    Args:
        source_dir: Checkout containing ``src/depth_anything_3``.
        checkpoint: Local official DA3-BASE safetensors file.
        model_config: Local JSON model identity, defaulting to the checkpoint's ``config.json``.
        process_res: Longest processed image edge; DA3 rounds both dimensions to multiples of 14.

    Example:
        >>> adapter = DA3Adapter(source_dir="models/Depth-Anything-3", checkpoint="checkpoints/da3/model.safetensors")
        >>> adapter.describe()["model"]
        'DA3-BASE'
    """

    def __init__(self, *, source_dir: str, checkpoint: str, model_config: str | None = None, process_res: int = 504) -> None:
        source, weights = Path(source_dir).resolve(), Path(checkpoint).resolve()
        config = Path(model_config).resolve() if model_config else weights.with_name("config.json")
        if not (source / "src/depth_anything_3/api.py").is_file():
            raise FileNotFoundError(f"DA3 source_dir not found: {source}")
        if not weights.is_file() or not config.is_file():
            raise FileNotFoundError(f"DA3 requires local checkpoint and config: {weights}, {config}")
        if type(process_res) is not int or process_res < 28:
            raise ValueError("DA3 process_res must be an integer >= 28")
        identity = json.loads(config.read_text())
        if identity.get("model_name") != "da3-base":
            raise ValueError("only the official DA3-BASE model_config is supported")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        sys.path.insert(0, str(source / "src"))
        import torch
        from depth_anything_3.api import DepthAnything3
        from safetensors.torch import load_model

        if not torch.cuda.is_available():
            raise RuntimeError("DA3 requires CUDA; CPU inference is not a deployment fallback")
        model = DepthAnything3(model_name="da3-base")
        # Official safetensors deduplicates the shared auxiliary LayerNorm tensors.
        load_model(model, str(weights), strict=True, device="cpu")
        self.model, self.torch = model.to("cuda").eval(), torch
        self.process_res = process_res
        self._info = {
            "model": "DA3-BASE", "python_version": sys.version.split()[0], "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda, "device": "cuda:0", "device_name": torch.cuda.get_device_name(0),
            "source_dir": str(source), "checkpoint": str(weights), "process_res": process_res,
            "depth_unit": "m", "metric_scale_source": "input_camera_translations", "calls": 0,
        }

    def infer(self, images: np.ndarray, intrinsics: np.ndarray, T_world_cam: np.ndarray) -> dict[str, Any]:
        """Return depth on the original image grid, scaled to the supplied camera baseline.

        Args:
            images: Three equal-sized RGB uint8 images, shape ``(3,H,W,3)``.
            intrinsics: OpenCV pixel intrinsics, shape ``(3,3,3)``.
            T_world_cam: OpenCV camera-to-world rigid transforms, shape ``(3,4,4)``, translations in metres.
                Poses are rebased in float64: ``T_local_cam = inv(T_world_cam[0]) @ T_world_cam``.
                Upstream receives ``inv(T_local_cam)`` and divides depth by its fitted Sim(3) scale.
                Non-collinear camera centres establish scale; world Geometry still uses the original poses.

        Returns:
            Dict with finite nonnegative float32 ``depth`` and optional ``confidence`` arrays of shape
            ``(3,H,W)``, plus ``depth_unit="m"``. Zero depth marks invalid samples. Resampling uses the
            upstream convention ``u_processed = u_original * W_processed/W_original`` (likewise v).

        Raises:
            ValueError: Images/calibration are malformed or camera centres cannot determine scale.
            RuntimeError: Upstream returns incompatible output or loses the supplied calibration.

        Example:
            >>> result = adapter.infer(images, intrinsics, T_world_cam)
            >>> (result["depth"].shape == images.shape[:3], result["depth_unit"])
            (True, 'm')
        """
        if (not isinstance(images, np.ndarray) or images.dtype != np.uint8 or images.ndim != 4
                or images.shape[0] != 3 or images.shape[-1] != 3 or min(images.shape[1:3]) <= 0):
            raise ValueError("DA3 images must be uint8 (3,H,W,3)")
        validate_image_shape(images.shape[1:3], self.process_res)
        intrinsics, transforms = np.asarray(intrinsics), np.asarray(T_world_cam, dtype=np.float64)
        if intrinsics.shape != (3, 3, 3) or transforms.shape != (3, 4, 4):
            raise ValueError("DA3 requires three intrinsic matrices and camera-to-world transforms")
        if not np.isfinite(intrinsics).all() or not np.isfinite(transforms).all():
            raise ValueError("DA3 calibration must be finite")
        if (np.any(intrinsics[:, (0, 1), (0, 1)] <= 0)
                or not np.allclose(intrinsics[:, 2], [0, 0, 1], atol=1e-6)):
            raise ValueError("DA3 intrinsics require positive focal lengths and homogeneous last rows")
        rotation = transforms[:, :3, :3]
        if (not np.allclose(transforms[:, 3], [0, 0, 0, 1], atol=1e-6)
                or not np.allclose(rotation.transpose(0, 2, 1) @ rotation, np.eye(3), atol=1e-5)
                or not np.allclose(np.linalg.det(rotation), 1, atol=1e-5)):
            raise ValueError("DA3 T_world_cam must be rigid camera-to-world transforms")
        centres = transforms[:, :3, 3]
        if np.linalg.matrix_rank(centres - centres.mean(axis=0), tol=1e-6) < 2:
            raise ValueError("DA3 metric alignment requires three non-collinear camera centres")
        # Remove the arbitrary world origin before upstream converts poses to float32.
        local_poses = np.linalg.inv(transforms[0]) @ transforms
        local_poses[0] = np.eye(4)
        extrinsics = np.linalg.inv(local_poses)
        self.torch.cuda.synchronize()
        self.torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        prediction = self.model.inference(
            list(images), extrinsics=extrinsics, intrinsics=intrinsics.copy(), align_to_input_ext_scale=True,
            process_res=self.process_res, process_res_method="upper_bound_resize", ref_view_strategy="first",
            export_dir=None,
        )
        self.torch.cuda.synchronize()
        elapsed = (time.perf_counter() - started) * 1000
        depth = np.asarray(prediction.depth)
        if depth.ndim != 3 or depth.shape[0] != 3 or min(depth.shape[1:]) <= 0:
            raise RuntimeError(f"unexpected DA3 depth shape: {depth.shape}")
        if (prediction.extrinsics is None
                or not np.allclose(prediction.extrinsics, extrinsics[:, :3], atol=1e-4, rtol=1e-4)):
            raise RuntimeError("DA3 did not retain input camera poses after metric scale alignment")
        processed_k = intrinsics.copy().astype(np.float64)
        processed_k[:, 0] *= depth.shape[2] / images.shape[2]
        processed_k[:, 1] *= depth.shape[1] / images.shape[1]
        if prediction.intrinsics is None or not np.allclose(prediction.intrinsics, processed_k, atol=1e-3, rtol=1e-4):
            raise RuntimeError("DA3 processed intrinsics do not match its resize-only image grid")
        rows = np.minimum(np.arange(images.shape[1]) * depth.shape[1] // images.shape[1], depth.shape[1] - 1)
        cols = np.minimum(np.arange(images.shape[2]) * depth.shape[2] // images.shape[2], depth.shape[2] - 1)
        output = depth[:, rows[:, None], cols].astype(np.float32)
        output[~np.isfinite(output) | (output <= 0)] = 0
        confidence = None
        if prediction.conf is not None:
            conf = np.asarray(prediction.conf)
            if conf.shape != depth.shape:
                raise RuntimeError("DA3 confidence must match its depth map")
            confidence = conf[:, rows[:, None], cols].astype(np.float32)
            confidence[~np.isfinite(confidence) | (confidence < 0) | (output == 0)] = 0
        self._info.update({
            "calls": self._info["calls"] + 1, "inference_ms": elapsed,
            "peak_cuda_memory_bytes": int(self.torch.cuda.max_memory_allocated()),
            "processed_shape": list(depth.shape[1:]), "output_shape": list(output.shape[1:]),
            "valid_fraction": float((output > 0).mean()),
        })
        return {"depth": np.ascontiguousarray(output), "confidence": confidence, "depth_unit": "m"}

    def describe(self) -> dict[str, Any]:
        """Return JSON diagnostics about model identity, metric scaling, and the last call.

        Returns:
            Copy of the adapter's diagnostics dictionary.

        Example:
            >>> adapter.describe()["metric_scale_source"]
            'input_camera_translations'
        """
        return dict(self._info)

    def close(self) -> None:
        """Release this adapter's GPU model; repeated calls are harmless.

        Example:
            >>> adapter.close()
            >>> adapter.model is None
            True
        """
        self.model = None
        self.torch.cuda.empty_cache()
