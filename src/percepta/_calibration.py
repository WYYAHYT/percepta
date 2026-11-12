"""Pure-NumPy validation shared by request schemas and metric geometry helpers."""

from __future__ import annotations

import numpy as np


def pinhole_intrinsics(K) -> np.ndarray:
    """Validate pixel intrinsics for the supported zero-skew pinhole camera.

    Args:
        K: Finite ``(3, 3)`` matrix mapping camera points to pixels via ``uv_h = K @ p_cam``.
            Focal lengths must be positive; the last row is ``[0, 0, 1]``.

    Returns:
        Float64 intrinsics without changing the caller's values.

    Raises:
        ValueError: The matrix is nonnumeric, nonfinite or not a supported pinhole camera.

    Example:
        >>> pinhole_intrinsics([[100, 0, 4], [0, 100, 4], [0, 0, 1]]).shape
        (3, 3)
    """
    try:
        matrix = np.asarray(K, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("K must contain numeric pinhole intrinsics") from exc
    if matrix.shape != (3, 3):
        raise ValueError("K must be 3x3 pinhole intrinsics")
    if (not np.isfinite(matrix).all() or matrix[0, 0] <= 0 or matrix[1, 1] <= 0
            or not np.allclose(matrix[2], [0, 0, 1], atol=1e-8, rtol=0)
            or not np.allclose(matrix[[0, 1], [1, 0]], 0, atol=1e-8, rtol=0)):
        raise ValueError("K must be zero-skew pinhole intrinsics with positive focal lengths")
    return matrix


def rigid_transform(T_world_cam) -> np.ndarray:
    """Validate a proper OpenCV camera-to-world rigid transform in metres.

    Args:
        T_world_cam: Finite ``(4, 4)`` matrix with ``p_world = T_world_cam @ p_cam_h``.
            Rotation must be orthonormal with determinant +1; the last row is ``[0, 0, 0, 1]``.

    Returns:
        Float64 transform without changing the caller's values. Rotation checks tolerate
        absolute floating-point error up to ``1e-6``; homogeneous-row checks use ``1e-8``.

    Raises:
        ValueError: The matrix is nonnumeric, nonfinite, nonrigid or a reflection.

    Example:
        >>> rigid_transform(np.eye(4)).shape
        (4, 4)
    """
    try:
        matrix = np.asarray(T_world_cam, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("T_world_cam must contain a numeric rigid transform") from exc
    if matrix.shape != (4, 4):
        raise ValueError("T_world_cam must be 4x4 rigid transform")
    rotation = matrix[:3, :3]
    with np.errstate(over="ignore", invalid="ignore"):
        if (not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
                or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0)
                or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6, rtol=0)):
            raise ValueError("T_world_cam must be a proper rigid camera-to-world transform")
    return matrix
