"""DA3's calibrated metre contract is testable without torch or model assets."""

import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from percepta.backends.da3_adapter import DA3Adapter


@pytest.fixture
def inputs():
    images = np.zeros((3, 6, 8, 3), dtype=np.uint8)
    intrinsics = np.repeat(np.array([[[20.0, 0, 4], [0, 20, 3], [0, 0, 1]]]), 3, axis=0)
    transforms = np.repeat(np.eye(4)[None], 3, axis=0)
    transforms[1, 0, 3], transforms[2, 1, 3] = 0.1, 0.1
    return images, intrinsics, transforms


@pytest.fixture
def adapter():
    adapter = DA3Adapter.__new__(DA3Adapter)
    adapter.process_res = 504
    adapter._info = {"calls": 0}
    adapter.torch = SimpleNamespace(cuda=SimpleNamespace(
        synchronize=lambda: None, reset_peak_memory_stats=lambda: None,
        max_memory_allocated=lambda: 12345, empty_cache=lambda: None,
    ))
    return adapter


def test_metric_scale_inverse_pose_and_original_pixel_grid(adapter, inputs):
    images, intrinsics, transforms = inputs
    depth = np.arange(36, dtype=np.float32).reshape(3, 3, 4)
    depth[0, 0, 0], depth[0, 0, 1] = np.nan, -1
    captured = {}

    def inference(image_list, **kwargs):
        captured.update(kwargs)
        assert len(image_list) == 3 and image_list[0].dtype == np.uint8
        processed_k = kwargs["intrinsics"].copy()
        processed_k[:, :2] *= 0.5
        return SimpleNamespace(depth=depth, conf=np.full_like(depth, 3),
                               extrinsics=kwargs["extrinsics"][:, :3], intrinsics=processed_k)

    adapter.model = SimpleNamespace(inference=inference)
    result = adapter.infer(*inputs)
    assert result["depth_unit"] == "m"
    assert result["depth"].dtype == np.float32 and result["depth"].shape == images.shape[:3]
    expected = np.repeat(np.repeat(np.maximum(np.nan_to_num(depth), 0), 2, axis=1), 2, axis=2)
    np.testing.assert_array_equal(result["depth"], expected)
    np.testing.assert_array_equal(result["confidence"][result["depth"] == 0], 0)
    assert captured["align_to_input_ext_scale"] is True
    assert captured["process_res_method"] == "upper_bound_resize"
    assert captured["ref_view_strategy"] == "first"
    np.testing.assert_allclose(captured["extrinsics"], np.linalg.inv(transforms))
    np.testing.assert_allclose(intrinsics[:, 0, 0], 20)
    assert adapter.describe()["calls"] == 1
    assert adapter.describe()["peak_cuda_memory_bytes"] == 12345


@pytest.mark.parametrize("bad", ["image_dtype", "image_shape", "focal", "homogeneous", "rotation", "nan", "collinear"])
def test_invalid_calibration_never_reaches_model(adapter, inputs, bad):
    images, intrinsics, transforms = inputs
    if bad == "image_dtype":
        images = images.astype(np.float32)
    elif bad == "image_shape":
        images = images[:2]
    elif bad == "focal":
        intrinsics[0, 0, 0] = 0
    elif bad == "homogeneous":
        transforms[0, 3, 3] = 2
    elif bad == "rotation":
        transforms[0, 0, 0] = -1
    elif bad == "nan":
        transforms[0, 0, 3] = np.nan
    else:
        transforms[2, 1, 3] = 0
    with pytest.raises(ValueError):
        adapter.infer(images, intrinsics, transforms)


@pytest.mark.parametrize("bad", ["pose", "intrinsics", "depth_shape", "confidence_shape"])
def test_upstream_calibration_or_grid_mismatch_is_rejected(adapter, inputs, bad):
    def inference(image_list, **kwargs):
        result = SimpleNamespace(depth=np.ones((3, 6, 8)), conf=None,
                                 extrinsics=kwargs["extrinsics"][:, :3], intrinsics=kwargs["intrinsics"])
        if bad == "pose":
            result.extrinsics = np.zeros((3, 3, 4))
        elif bad == "intrinsics":
            result.intrinsics = np.ones((3, 3, 3))
        elif bad == "depth_shape":
            result.depth = np.ones((2, 6, 8))
        else:
            result.conf = np.ones((3, 2, 2))
        return result

    adapter.model = SimpleNamespace(inference=inference)
    with pytest.raises(RuntimeError):
        adapter.infer(*inputs)


def test_missing_or_wrong_assets_fail_before_heavy_imports(tmp_path):
    source = tmp_path / "source"
    with pytest.raises(FileNotFoundError, match="source_dir"):
        DA3Adapter(source_dir=str(source), checkpoint=str(tmp_path / "missing"))
    (source / "src/depth_anything_3").mkdir(parents=True)
    (source / "src/depth_anything_3/api.py").touch()
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.touch()
    (tmp_path / "config.json").write_text(json.dumps({"model_name": "another-model"}))
    with pytest.raises(ValueError, match="DA3-BASE"):
        DA3Adapter(source_dir=str(source), checkpoint=str(checkpoint))


def test_import_does_not_load_torch():
    result = subprocess.run(
        [sys.executable, "-c", "import sys; import percepta.backends.da3_adapter; assert 'torch' not in sys.modules"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("shape", [(1, 1000), (1000, 1)])
def test_thin_images_fail_before_upstream_inference(adapter, inputs, shape):
    _, intrinsics, transforms = inputs

    def forbidden(*args, **kwargs):
        pytest.fail("a dimension smaller than one processed patch must not reach the model")

    adapter.model = SimpleNamespace(inference=forbidden)
    with pytest.raises(ValueError, match="patch"):
        adapter.infer(np.zeros((3, *shape, 3), np.uint8), intrinsics, transforms)


def test_model_camera_conditions_ignore_large_world_origin(adapter, inputs):
    images, intrinsics, transforms = inputs
    captured = []

    def inference(image_list, **kwargs):
        captured.append(kwargs["extrinsics"].astype(np.float32))
        return SimpleNamespace(depth=np.ones(images.shape[:3]), conf=None,
                               extrinsics=kwargs["extrinsics"][:, :3], intrinsics=kwargs["intrinsics"])

    adapter.model = SimpleNamespace(inference=inference)
    basis = np.array([[0, -1, 0, 4], [1, 0, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]], dtype=np.float64)
    for offset in (0., 1e6, 1e7):
        world = basis.copy()
        world[:3, 3] += offset
        poses = world @ transforms
        original = poses.copy()
        adapter.infer(images, intrinsics, poses)
        np.testing.assert_array_equal(poses, original)
        np.testing.assert_allclose(np.linalg.inv(captured[-1])[0], np.eye(4), atol=1e-7)
        np.testing.assert_allclose(captured[-1], captured[0], atol=1e-7, rtol=1e-6)


def test_smoke_views_share_the_same_calibrated_textured_plane():
    from scripts.da3_smoke import calibrated_scene

    images, intrinsics, poses = calibrated_scene(height=140, width=200)
    # f=170 px, baseline=0.2 m and plane Z=1 m imply left-right disparity=34 px.
    np.testing.assert_array_equal(images[0, :, 34:], images[1, :, :-34])
    np.testing.assert_allclose(intrinsics[:, 0, 0], 170)
    assert np.linalg.matrix_rank(poses[:, :3, 3] - poses[:, :3, 3].mean(axis=0)) == 2
