"""Official MVSAnywhere HERO inference inside its isolated, pinned CUDA runtime."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from importlib.resources import files
from pathlib import Path

import numpy as np

MANIFEST = files("percepta.backends").joinpath("mvsanywhere_assets.json")
if not MANIFEST.is_file():  # Editable checkouts use the uv project's original manifest.
    MANIFEST = Path(__file__).resolve().parents[3] / "uv_envs/mvsanywhere/assets.json"


def _require_hash(path: Path, expected: str) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError(f"MVSAnywhere asset hash mismatch: {path}")


def _validate_inputs(images, intrinsics, poses) -> None:
    if not isinstance(images, np.ndarray) or images.dtype != np.uint8 or images.ndim != 4:
        raise ValueError("MVSAnywhere images must be uint8 [3,H,W,3]")
    if images.shape[0] != 3 or images.shape[-1] != 3 or min(images.shape[1:3]) < 1:
        raise ValueError("MVSAnywhere requires exactly three nonempty RGB views")
    if not isinstance(intrinsics, np.ndarray) or intrinsics.shape != (3, 3, 3) or not np.isfinite(intrinsics).all():
        raise ValueError("MVSAnywhere intrinsics must be finite [3,3,3]")
    if not np.allclose(intrinsics[:, 2], [0, 0, 1]) or np.any(intrinsics[:, [0, 1], [0, 1]] <= 0):
        raise ValueError("MVSAnywhere requires valid positive-focal OpenCV intrinsics")
    if not isinstance(poses, np.ndarray) or poses.shape != (3, 4, 4) or not np.isfinite(poses).all():
        raise ValueError("MVSAnywhere T_world_cam must be finite [3,4,4]")
    rotations = poses[:, :3, :3]
    if (not np.allclose(poses[:, 3], [0, 0, 0, 1])
            or not np.allclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=1e-4)
            or not np.allclose(np.linalg.det(rotations), 1, atol=1e-4)):
        raise ValueError("MVSAnywhere T_world_cam must contain proper rigid camera-to-world transforms")
    centres = poses[:, :3, 3]
    if any(np.linalg.norm(centres[a] - centres[b]) <= 1e-6 for a, b in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("MVSAnywhere requires nonzero metric baselines, not coincident camera centres")


def _prepare_cameras(intrinsics, poses, original_size, model_size):
    # OpenCV +Z-forward already matches upstream. Only K changes under image resizing;
    # src_T_cur = inv(T_world_src) @ T_world_cur cancels any common world-frame basis.
    matching_K = np.broadcast_to(np.eye(4), (3, 4, 4)).copy()
    matching_K[:, :3, :3] = intrinsics
    matching_K[:, 0] *= model_size[1] / original_size[1] / 4
    matching_K[:, 1] *= model_size[0] / original_size[0] / 4
    return matching_K.astype(np.float32), poses.astype(np.float32), np.linalg.inv(poses).astype(np.float32)


def _canonical_poses(poses: np.ndarray) -> np.ndarray:
    # The model uses only relative transforms. Remove the arbitrary world origin/rotation
    # in float64 before float32 inversion can amplify global-coordinate cancellation.
    precise = np.asarray(poses, dtype=np.float64)
    local = np.linalg.inv(precise[0]) @ precise
    local[0] = np.eye(4)
    return local


class MVSAnywhereAdapter:
    """Predict metric camera-Z depth for three calibrated RGB views using official HERO weights.

    Each reference jointly consumes both other images. Input and output camera axes are
    OpenCV (+X right, +Y down, +Z forward); ``p_world = T_world_cam @ p_cam_h``.
    The upstream model uses ``src_T_cur = inv(T_world_src) @ T_world_cur``. Its ScanNet
    convention therefore needs no extra global-axis flip for these already-OpenCV inputs.
    The arbitrary world basis is removed in float64 using ``inv(T_world_cam[0]) @ T_world_cam[i]``
    before float32 model input; metric scale and the original camera axes are preserved.

    Args:
        source_dir: Pinned, bootstrapped official MVSAnywhere checkout, including local DINO.
        checkpoint: Hash-pinned official HERO checkpoint; never a request-supplied file.
        image_height: Inference height, a positive multiple of 32.
        image_width: Inference width, a positive multiple of 32.
        num_refinement_steps: Additional official depth-range refinement iterations (0 to 2).

    Raises:
        ValueError: Configuration or source/checkpoint integrity is invalid.
        RuntimeError: CUDA is unavailable, or trained weights cannot be loaded strictly.

    Example:
        >>> adapter = MVSAnywhereAdapter(source_dir="models/mvsanywhere", checkpoint="checkpoints/mvsanywhere/mvsanywhere_hero.ckpt")
        >>> result = adapter.infer(images, K, T_world_cam)
        >>> result["depth"].shape == images.shape[:3], result["depth_unit"]
        (True, 'm')
    """

    def __init__(self, *, source_dir: str, checkpoint: str, image_height: int = 480,
                 image_width: int = 640, num_refinement_steps: int = 1) -> None:
        for value in (image_height, image_width):
            if type(value) is not int or value < 32 or value % 32:
                raise ValueError("MVSAnywhere inference dimensions must be positive multiples of 32")
        if type(num_refinement_steps) is not int or not 0 <= num_refinement_steps <= 2:
            raise ValueError("MVSAnywhere num_refinement_steps must be an integer from 0 to 2")
        source = Path(source_dir).resolve()
        weights = Path(checkpoint).resolve()
        manifest = json.loads(MANIFEST.read_text())
        for path, record in ((source, manifest["source"]), (source / "dinov2", manifest["dinov2"])):
            revision = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
            if revision != record["revision"]:
                raise ValueError(f"MVSAnywhere source revision mismatch: {path}")
        for patch in manifest["source"]["offline_patches"]:
            _require_hash(source / patch["path"], patch["sha256"])
        _require_hash(weights, manifest["checkpoint"]["sha256"])
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["XFORMERS_DISABLED"] = "1"
        sys.path.insert(0, str(source / "src"))
        import torch
        from torchvision import transforms

        if not torch.cuda.is_available():
            raise RuntimeError("MVSAnywhere requires a CUDA GPU; CPU or substitute-model fallback is unsupported")
        from mvsanywhere.experiment_modules.sr_depth_model import DepthModel

        # The official Lightning checkpoint serializes its Options dataclass. torch 2.1
        # needs the original loader; the immutable official hash is checked before unpickling.
        saved = torch.load(str(weights), map_location="cpu", weights_only=False)
        opts = saved["hyper_parameters"]["opts"]
        opts.da_weights_path = None
        self.model = DepthModel(opts)
        self.model.load_state_dict(saved["state_dict"], strict=True)
        self.model.depth_decoder.set_prediction_scale(1.0)
        self.model = self.model.cuda().eval()
        self.torch = torch
        self.image_size = (image_height, image_width)
        self.refinement_steps = num_refinement_steps
        self._input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize(self.image_size, antialias=False),
        ])
        self._info = {
            "model": "MVSAnywhere HERO", "model_sha256": manifest["checkpoint"]["sha256"],
            "source_revision": manifest["source"]["revision"], "device": "cuda:0",
            "device_name": torch.cuda.get_device_name(0), "python_version": sys.version.split()[0],
            "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
            "image_height": image_height, "image_width": image_width,
            "num_refinement_steps": num_refinement_steps, "depth_unit": "m",
            "camera_frame": "OpenCV +Z forward", "depth_semantics": "camera_z",
            "pose_normalization": "float64 first-camera frame: inv(T_world_cam[0]) @ T_world_cam[i]",
            "references_inferred": 3, "source_views_per_reference": 2,
            "confidence_semantics": "unavailable; sky probability is not depth confidence",
            "model_latency_ms": None, "peak_cuda_memory_bytes": None,
        }

    def infer(self, images: np.ndarray, intrinsics: np.ndarray, T_world_cam: np.ndarray) -> dict:
        """Infer all three original-grid depth maps; zero marks sky or invalid depth.

        Args:
            images: uint8 RGB ``[3,H,W,3]``, synchronized calibrated views.
            intrinsics: Finite OpenCV camera matrices ``[3,3,3]`` at the original image resolution.
            T_world_cam: Rigid camera-to-world transforms ``[3,4,4]`` with translation in metres.

        Returns:
            Dict with contiguous float32 ``depth[3,H,W]`` in metres, ``depth_unit='m'``
            and ``confidence=None``. No relative-depth scale is fitted from ground truth.

        Raises:
            ValueError: Images/calibration are invalid or metric baselines vanish.
            RuntimeError: The adapter was closed or inference returns malformed depth.

        Example:
            >>> result = adapter.infer(images, K, T_world_cam)
            >>> result["depth"].dtype == np.float32, result["confidence"] is None
            (True, True)
        """
        _validate_inputs(images, intrinsics, T_world_cam)
        if self.model is None:
            raise RuntimeError("MVSAnywhere adapter is closed")
        torch = self.torch
        matching_K, poses, _ = _prepare_cameras(intrinsics, _canonical_poses(T_world_cam), images.shape[1:3], self.image_size)
        with torch.inference_mode():
            # Preserve upstream's CPU torchvision arithmetic and contiguous CHW layout.
            # GPU normalization/channel-last convolutions round differently in FP16.
            rgb = torch.stack([self._input_transform(image) for image in images]).cuda()
            Ks = torch.from_numpy(matching_K).cuda()
            worlds = torch.from_numpy(poses).cuda()
            result = []
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            timings = []
            for reference in range(3):
                sources = [index for index in range(3) if index != reference]
                current = {
                    "image_b3hw": rgb[reference:reference + 1], "world_T_cam_b44": worlds[reference:reference + 1],
                    "cam_T_world_b44": torch.linalg.inv(worlds[reference:reference + 1]),
                    "K_matching_b44": Ks[reference:reference + 1],
                    "invK_matching_b44": torch.linalg.inv(Ks[reference:reference + 1]),
                }
                other = {
                    "image_b3hw": rgb[sources][None], "world_T_cam_b44": worlds[sources][None],
                    "cam_T_world_b44": torch.linalg.inv(worlds[sources][None]), "K_matching_b44": Ks[sources][None],
                    "invK_matching_b44": torch.linalg.inv(Ks[sources][None]),
                }
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    output = self.model("test", current, other, return_mask=True,
                                        num_refinement_steps=self.refinement_steps, estimate_depth_range=True)
                end.record()
                timings.append((start, end))
                depth = output["depth_pred_s0_b1hw"].float()
                sky = output["sky_pred_s0_b1hw"].float()
                if depth.shape != (1, 1, *self.image_size) or sky.shape != depth.shape:
                    raise RuntimeError(f"unexpected MVSAnywhere depth/sky shape: {depth.shape}, {sky.shape}")
                depth = torch.nn.functional.interpolate(depth, size=images.shape[1:3], mode="bilinear", align_corners=False)
                sky = torch.nn.functional.interpolate(sky, size=images.shape[1:3], mode="nearest")
                valid = torch.isfinite(depth) & (depth > 0) & torch.isfinite(sky) & (sky < 0.5)
                result.append(torch.where(valid, depth, 0)[0, 0])
            torch.cuda.synchronize()
            self._info["model_latency_ms"] = sum(float(start.elapsed_time(end)) for start, end in timings)
            self._info["peak_cuda_memory_bytes"] = int(torch.cuda.max_memory_allocated())
            depth_maps = torch.stack(result).cpu().numpy()
        return {"depth": np.ascontiguousarray(depth_maps, dtype=np.float32), "confidence": None, "depth_unit": "m"}

    def describe(self) -> dict:
        """Return JSON-safe runtime information and most recent GPU inference measurements.

        Returns:
            A copy of model identity, coordinate semantics, runtime versions and timings.

        Example:
            >>> adapter.describe()["references_inferred"]
            3
        """
        return dict(self._info)

    def close(self) -> None:
        """Release this adapter's GPU model; repeated calls are harmless.

        Example:
            >>> adapter.close()
            >>> adapter.close()
            # Both calls return None.
        """
        self.model = None
        self.torch.cuda.empty_cache()
