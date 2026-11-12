"""CPU checks of MVS coordinate conversion and invalid deployment/request rejection."""

import subprocess
import sys

import numpy as np
import pytest

from percepta.backends.mvsanywhere_adapter import (
    MVSAnywhereAdapter,
    _canonical_poses,
    _prepare_cameras,
    _validate_inputs,
)


def _inputs():
    images = np.zeros((3, 40, 60, 3), dtype=np.uint8)
    K = np.broadcast_to([[100.0, 0, 30], [0, 120, 20], [0, 0, 1]], (3, 3, 3)).copy()
    poses = np.broadcast_to(np.eye(4), (3, 4, 4)).copy()
    poses[:, 0, 3] = [-0.1, 0, 0.1]
    return images, K, poses


def test_opencv_positive_z_and_rigid_world_basis_are_preserved():
    images, K, poses = _inputs()
    _validate_inputs(images, K, poses)
    Ks, worlds, cameras = _prepare_cameras(K, poses, (40, 60), (480, 640))
    point_world = np.array([0.0, 0.0, 2.0, 1.0])
    points_camera = cameras @ point_world
    assert np.all(points_camera[:, 2] > 0)
    pixels = Ks[:, :3, :3] @ points_camera[:, :3, None]
    pixels = pixels[:, :2, 0] / pixels[:, 2, 0, None]
    np.testing.assert_allclose(pixels[:, 0], [93.3333333, 80, 66.6666667], atol=1e-5)
    np.testing.assert_allclose(worlds @ points_camera[..., None], np.broadcast_to(point_world, (3, 4))[..., None], atol=1e-7)
    # The trained ray metadata uses current-camera coordinates. A common world
    # rotation/translation cancels and must not introduce an extra camera-Z flip.
    basis = np.array([[0, -1, 0, 4], [1, 0, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]], dtype=float)
    _, shifted_worlds, shifted_cameras = _prepare_cameras(K, basis @ poses, (40, 60), (480, 640))
    np.testing.assert_allclose(shifted_cameras[1] @ shifted_worlds[0], cameras[1] @ worlds[0], atol=1e-6)


def test_pose_normalization_removes_world_basis_without_changing_metric_geometry():
    _, _, poses = _inputs()
    poses[1, :3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    basis = np.array([[0, 0, 1, 123.456], [1, 0, 0, -76.54], [0, 1, 0, 89.12], [0, 0, 0, 1]])
    original = poses.copy()
    local = _canonical_poses(poses)
    shifted = _canonical_poses(basis @ poses)
    np.testing.assert_array_equal(local[0], np.eye(4))
    np.testing.assert_allclose(local, shifted, atol=1e-10, rtol=0)
    np.testing.assert_array_equal(poses, original)
    for a, b in ((0, 1), (0, 2), (1, 2)):
        np.testing.assert_allclose(np.linalg.inv(local[b]) @ local[a], np.linalg.inv(poses[b]) @ poses[a], atol=1e-10)
        assert np.linalg.norm(local[a, :3, 3] - local[b, :3, 3]) == pytest.approx(
            np.linalg.norm(poses[a, :3, 3] - poses[b, :3, 3]), abs=1e-10)


@pytest.mark.parametrize("fault", ["two_views", "float_rgb", "nan_K", "negative_focal", "inverted_pose", "reflection", "zero_baseline"])
def test_invalid_inputs_fail_before_model_inference(fault):
    images, K, poses = _inputs()
    if fault == "two_views":
        images = images[:2]
    elif fault == "float_rgb":
        images = images.astype(np.float32)
    elif fault == "nan_K":
        K[0, 0, 0] = np.nan
    elif fault == "negative_focal":
        K[1, 1, 1] = -1
    elif fault == "inverted_pose":
        poses[0, 3, 0] = 1
    elif fault == "reflection":
        poses[0, 0, 0] = -1
    else:
        poses[0] = poses[1]
    with pytest.raises(ValueError):
        _validate_inputs(images, K, poses)


@pytest.mark.parametrize("kwargs", [{"image_height": 479}, {"image_width": True}, {"num_refinement_steps": -1}])
def test_invalid_deployment_options_fail_before_asset_loading(kwargs):
    with pytest.raises(ValueError):
        MVSAnywhereAdapter(source_dir="missing", checkpoint="missing", **kwargs)


def test_import_does_not_load_torch():
    result = subprocess.run(
        [sys.executable, "-c", "import sys; import percepta.backends.mvsanywhere_adapter; assert 'torch' not in sys.modules"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
