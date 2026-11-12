"""Exercise real DA3-BASE inference with a known three-camera textured-plane scene."""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from percepta.backends.da3_adapter import DA3Adapter  # noqa: E402


def calibrated_scene(height: int = 252, width: int = 336) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render one textured plane from three non-collinear OpenCV camera centres.

    Args:
        height: Image rows.
        width: Image columns.

    Returns:
        RGB uint8 ``(3,H,W,3)``, pixel K ``(3,3,3)``, and camera-to-world ``(3,4,4)``.
        Rays obey ``p_world = T_world_cam @ [Z * K^-1 @ [u,v,1],1]``; the plane is world Z=1 m.

    Example:
        >>> images, intrinsics, poses = calibrated_scene()
        >>> images.shape, poses[:, :3, 3].tolist()
        ((3, 252, 336, 3), [[-0.1, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]])
    """
    from PIL import Image

    rng = np.random.default_rng(42)
    texture = rng.integers(20, 235, (256, 256, 3), dtype=np.uint8)
    texture = np.asarray(Image.fromarray(texture).resize((1024, 1024), Image.Resampling.BICUBIC))
    poses = np.repeat(np.eye(4)[None], 3, axis=0)
    poses[:, :3, 3] = [[-0.1, 0, 0], [0.1, 0, 0], [0, 0.1, 0]]
    focal = 0.85 * width
    intrinsic = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])
    intrinsics = np.repeat(intrinsic[None], 3, axis=0)
    rows, cols = np.indices((height, width))
    images = []
    for pose in poses:
        x = (cols - width / 2) / focal + pose[0, 3]
        y = (rows - height / 2) / focal + pose[1, 3]
        tx = np.clip(((x + 1) * 512).astype(int), 0, 1023)
        ty = np.clip(((y + 1) * 512).astype(int), 0, 1023)
        images.append(texture[ty, tx])
    return np.stack(images), intrinsics, poses


def run(*, source_dir: str = "models/Depth-Anything-3", checkpoint: str = "checkpoints/da3/model.safetensors",
        process_res: int = 336) -> dict:
    """Verify CUDA imports, repeatable inference, original-grid output and metric scale binding.

    Args:
        source_dir: Pinned DA3 checkout.
        checkpoint: Official local DA3-BASE safetensors.
        process_res: Longest processed image edge.

    Returns:
        JSON report including CUDA tensor evidence, repeated-call timings, memory and scale ratio.

    Example:
        >>> report = run()
        >>> report["repeat_agrees"], report["metric_scale_agrees"]
        (True, True)
    """
    import torch
    import torchaudio
    import torchvision

    assert torch.cuda.is_available()
    tensor = torch.ones(1, device="cuda")
    assert tensor.item() == 1
    images, intrinsics, poses = calibrated_scene()
    originals = images.copy(), intrinsics.copy(), poses.copy()
    started = time.perf_counter()
    adapter = DA3Adapter(source_dir=source_dir, checkpoint=checkpoint, process_res=process_res)
    report = {"load_ms": (time.perf_counter() - started) * 1000, "cuda_tensor": str(tensor),
              "torchvision": torchvision.__version__, "torchaudio": torchaudio.__version__, "calls": []}
    try:
        predictions = []
        for _ in range(2):
            result = adapter.infer(images, intrinsics, poses)
            depth = result["depth"]
            assert depth.shape == images.shape[:3] and depth.dtype == np.float32
            assert result["depth_unit"] == "m" and np.isfinite(depth).all()
            assert (depth > 0).mean() > 0.5
            for observed, original in zip((images, intrinsics, poses), originals, strict=True):
                np.testing.assert_array_equal(observed, original)
            predictions.append(depth)
            report["calls"].append({**adapter.describe(), "median_depth_m": float(np.median(depth[depth > 0]))})
        np.testing.assert_allclose(predictions[0], predictions[1], atol=1e-3, rtol=1e-3)
        scaled_poses = poses.copy()
        scaled_poses[:, :3, 3] *= 2
        doubled = adapter.infer(images, intrinsics, scaled_poses)["depth"]
        valid = (predictions[0] > 0) & (doubled > 0)
        ratio = float(np.median(doubled[valid] / predictions[0][valid]))
        assert abs(ratio - 2) < 0.03, f"DA3 did not follow input metre scale: {ratio}"
        origin_differences = []
        for offset in (1e6, 1e7):
            shifted_poses = poses.copy()
            shifted_poses[:, :3, 3] += offset
            shifted = adapter.infer(images, intrinsics, shifted_poses)["depth"]
            np.testing.assert_allclose(shifted, predictions[0], atol=1e-3, rtol=1e-3)
            origin_differences.append(float(np.max(np.abs(shifted - predictions[0]))))
        report.update({"repeat_agrees": True, "metric_scale_agrees": True, "doubled_baseline_depth_ratio": ratio,
                       "world_origin_agrees": True, "world_origin_max_depth_differences_m": origin_differences,
                       "ground_truth_plane_z_m": 1.0,
                       "accuracy_note": "Synthetic smoke verifies execution and baseline scaling, not localization accuracy."})
    finally:
        adapter.close()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--process-res", type=int, default=336)
    args = parser.parse_args()
    with redirect_stdout(sys.stderr):
        report = run(process_res=args.process_res)
    print(json.dumps(report, indent=2, allow_nan=False))
