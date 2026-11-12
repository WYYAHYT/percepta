"""Run real, offline three-view HERO inference in the isolated MVSAnywhere interpreter."""

from __future__ import annotations

import argparse
import importlib.util
import json
import socket
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def official_triplet(source: Path, image_size: tuple[int, int] = (480, 640)) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read three hash-pinned official RGB images and their supplied metric calibration.

    Args:
        source: Pinned MVSAnywhere source checkout.
        image_size: Returned image ``(height,width)``; focal lengths/principal points scale with it.

    Returns:
        RGB uint8 ``[3,H,W,3]``, float64 OpenCV K ``[3,3,3]`` and camera-to-world ``[3,4,4]``.

    Example:
        >>> images, K, poses = official_triplet(ROOT / "models/mvsanywhere")
        >>> images.shape, K.shape, poses.shape
        ((3, 480, 640, 3), (3, 3, 3), (3, 4, 4))
    """
    from percepta.backends.mvsanywhere_adapter import _require_hash

    manifest = json.loads((ROOT / "uv_envs/mvsanywhere/assets.json").read_text())
    for asset in manifest["demo_assets"]:
        _require_hash(source / asset["path"], asset["sha256"])
    # Upstream provides a pickled calibration dict. Verify the immutable official bytes first;
    # this helper is only for deployment smoke, never the public image/request path.
    cameras = np.load(source / "demo_assets/cameras.npy", allow_pickle=True).item()
    names = ["cur_image.png", "src_image_1.png", "src_image_2.png"]
    frames = [Image.open(source / "demo_assets" / name).convert("RGB") for name in names]
    K = np.stack([cameras["cur_intrinsics"][:3, :3], *cameras["src_intrinsics"][:2, :3, :3]]).astype(np.float64)
    for index, frame in enumerate(frames):
        width, height = frame.size
        K[index, 0] *= image_size[1] / width
        K[index, 1] *= image_size[0] / height
    images = np.stack([np.asarray(frame.resize(image_size[::-1], Image.Resampling.BILINEAR)) for frame in frames])
    poses = np.stack([cameras["cur_pose"], *cameras["src_pose"][:2]]).astype(np.float64)
    return images, K, poses


def upstream_reference(adapter, images: np.ndarray, K: np.ndarray, poses: np.ndarray) -> np.ndarray:
    """Compare our first reference against the actual upstream in-memory hub forward path.

    Args:
        adapter: Resident official HERO adapter; reuse its strictly loaded trained model.
        images: The three original-grid uint8 RGB views.
        K: The three original-grid 3×3 intrinsic matrices.
        poses: The three camera-to-world matrices in metres.

    Returns:
        Float32 first-reference depth map in metres, with upstream sky pixels zeroed.

    Example:
        >>> reference = upstream_reference(adapter, images, K, poses)
        >>> reference.shape == images.shape[1:3]
        True
    """
    import torch
    from torchvision import transforms

    from percepta.backends.mvsanywhere_adapter import _canonical_poses

    source = ROOT / "models/mvsanywhere"
    spec = importlib.util.spec_from_file_location("mvsanywhere_official_hub", source / "hubconf.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    wrapper = module.MVSAnywhereInference.__new__(module.MVSAnywhereInference)
    torch.nn.Module.__init__(wrapper)
    wrapper.model = adapter.model
    wrapper.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    wrapper.input_transform = transforms.Compose([transforms.ToTensor(), wrapper.normalize])
    K4 = np.broadcast_to(np.eye(4), (3, 4, 4)).copy()
    K4[:, :3, :3] = K
    # Compare identical, explicitly normalized metric poses through the official path.
    poses = _canonical_poses(poses)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        reference = wrapper.preprocess_and_run(
            images[0], [images[1], images[2]], poses[0], poses[1:], K4[0].copy(), K4[1:].copy(),
            num_refinement_steps=adapter.refinement_steps, target_image_size=adapter.image_size, device="cuda",
        )
    depth = reference["depth"].float()
    sky = reference["sky_mask"].float()
    depth = torch.nn.functional.interpolate(depth[None, None], size=images.shape[1:3], mode="bilinear", align_corners=False)[0, 0]
    sky = torch.nn.functional.interpolate(sky[None, None], size=images.shape[1:3], mode="nearest")[0, 0]
    valid = torch.isfinite(depth) & (depth > 0) & torch.isfinite(sky) & (sky < 0.5)
    return torch.where(valid, depth, 0).cpu().numpy()


def run(repeats: int = 3) -> dict:
    """Verify CUDA, offline strict loading, repeated three-view maps and upstream agreement.

    Args:
        repeats: Number of complete three-reference inference calls, at least two.

    Returns:
        JSON-safe environment, loading, latency, memory and per-view metric-depth diagnostics.

    Example:
        >>> report = run(repeats=2)
        >>> report["repeated_consistent"], report["upstream_reference_consistent"]
        (True, True)
    """
    if repeats < 2:
        raise ValueError("repeats must be at least two")
    import torch

    from percepta.backends.mvsanywhere_adapter import MVSAnywhereAdapter, _canonical_poses

    assert torch.cuda.is_available(), "MVSAnywhere CUDA is unavailable"
    tensor = torch.ones(1, device="cuda")
    assert tensor.item() == 1
    images, K, poses = official_triplet(ROOT / "models/mvsanywhere")
    original_connect, original_create = socket.socket.connect, socket.create_connection

    def blocked(*args, **kwargs):
        raise RuntimeError("network access forbidden during prepared MVSAnywhere loading/inference")

    socket.socket.connect = blocked
    socket.create_connection = blocked
    adapter = None
    try:
        started = time.perf_counter()
        adapter = MVSAnywhereAdapter(
            source_dir=str(ROOT / "models/mvsanywhere"),
            checkpoint=str(ROOT / "checkpoints/mvsanywhere/mvsanywhere_hero.ckpt"),
        )
        report = {"load_s": time.perf_counter() - started, "cuda_tensor": str(tensor), "calls": []}
        first = None
        for _ in range(repeats):
            started = time.perf_counter()
            output = adapter.infer(images, K, poses)
            elapsed = time.perf_counter() - started
            depth = output["depth"]
            assert depth.shape == images.shape[:3] and depth.dtype == np.float32
            assert np.isfinite(depth).all() and np.all(depth >= 0) and output["depth_unit"] == "m"
            views = []
            for view in depth:
                valid = view > 0
                assert valid.mean() > 0.05
                views.append({"valid_fraction": float(valid.mean()), "median_depth_m": float(np.median(view[valid]))})
            if first is None:
                first = depth.copy()
            else:
                np.testing.assert_allclose(first, depth, rtol=1e-3, atol=1e-3)
            report["calls"].append({"elapsed_s": elapsed, "views": views, "runtime": adapter.describe()})
        expected = upstream_reference(adapter, images, K, poses)
        np.testing.assert_allclose(first[0], expected, rtol=1e-3, atol=1e-3)
        basis = np.array([[0, -1, 0, 1], [1, 0, 0, 2], [0, 0, 1, -1], [0, 0, 0, 1]], dtype=float)
        shifted = adapter.infer(images, K, basis @ poses)["depth"]
        np.testing.assert_allclose(first, shifted, rtol=3e-3, atol=3e-3)
        report.update({"repeated_consistent": True, "upstream_reference_consistent": True,
                       "world_basis_consistent": True, "network_blocked": True,
                       "upstream_reference_pose_normalization": "float64 first-camera frame",
                       "world_basis_pose_max_abs_diff": float(np.max(np.abs(_canonical_poses(poses) - _canonical_poses(basis @ poses)))),
                       "world_basis_depth_max_abs_diff_m": float(np.max(np.abs(first - shifted)))})
        return report
    finally:
        if adapter is not None:
            adapter.close()
        socket.socket.connect, socket.create_connection = original_connect, original_create


def main() -> None:
    """Write a real GPU smoke report from the isolated interpreter.

    Example:
        >>> main()
        # Prints JSON; --output also stores the same report for review.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.dumps(run(args.repeats), indent=2)
    if args.output:
        args.output.write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
