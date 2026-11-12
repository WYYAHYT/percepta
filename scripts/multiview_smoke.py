"""Exercise deployed three-view models through resident workers, tool geometry and real HTTP."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from percepta.config import load_config  # noqa: E402
from percepta.schema import Calibration, ToolContext  # noqa: E402
from percepta.tools.triview_3d import Triview3DTool  # noqa: E402
from scripts.foundation_stereo_smoke import live_client  # noqa: E402


def scene(model: str) -> tuple[dict, dict]:
    """Load calibrated model smoke RGB without importing torch in the parent process.

    Args:
        model: mvsanywhere uses its official sample; da3 uses a calibrated textured plane.

    Returns:
        RGB and Calibration dictionaries, with p_world = T_world_cam @ p_cam_h.

    Example:
        >>> images, calibration = scene("da3")
        >>> len(images), len(calibration)
        (3, 3)
    """
    if model == "mvsanywhere":
        from scripts.mvsanywhere_smoke import official_triplet

        frames, K, transforms = official_triplet(ROOT / "models/mvsanywhere")
    elif model == "da3":
        from scripts.da3_smoke import calibrated_scene

        frames, K, transforms = calibrated_scene()
    else:
        raise ValueError("expected da3 or mvsanywhere")
    images = {f"view_{i}": image for i, image in enumerate(frames)}
    calibration = {
        name: Calibration(camera=name, width=image.shape[1], height=image.shape[0], K=K[i].tolist(),
                          T_world_cam=transforms[i].tolist(), id=f"{model}-smoke")
        for i, (name, image) in enumerate(images.items())
    }
    return images, calibration


def run(config_path: str, *, http: bool = False) -> dict:
    """Verify two resident calls, world geometry, and optional actual loopback HTTP behavior.

    Args:
        config_path: Trusted YAML with a prepared three-view model.
        http: Also test HTTP/direct agreement, repeatability and worker termination.

    Returns:
        Strict JSON evidence with measured timings and model provenance.

    Example:
        >>> run("configs/local-da3.yaml")["resident_process_reused"]  # doctest: +SKIP
        True
    """
    config = load_config(config_path)
    deployment = config.tool_config("triview_3d")
    model = deployment["triview_backend"]
    images, calibration = scene(model)
    tool = Triview3DTool()
    report = {"backend": model, "calls": []}
    started = time.perf_counter()
    try:
        tool.load(deployment)
        report["load_ms"] = (time.perf_counter() - started) * 1000
        pid = tool._backend.describe()["pid"]
        for index in range(2):
            context = ToolContext(request_id=f"{model}-tool-{index}", images=images, calibration=calibration)
            started = time.perf_counter()
            percepts = tool.perceive(context, [])
            assert len(percepts) == 1 and percepts[0].status == "ok", str(percepts)
            percept = percepts[0]
            assert percept.geometry and percept.geometry.views_used == 3 and percept.provenance.model_sha256
            assert np.isfinite(percept.geometry.centroid).all()
            assert tool._backend.describe()["pid"] == pid
            report["calls"].append({"elapsed_ms": (time.perf_counter() - started) * 1000,
                                    "percept": percept.model_dump(mode="json")})
        local_geometry = percept.geometry
        np.testing.assert_allclose(report["calls"][0]["percept"]["geometry"]["centroid"], local_geometry.centroid,
                                   rtol=1e-3, atol=1e-4)
        report["resident_process_reused"] = True
    finally:
        tool.close()
    if http:
        with live_client(config) as (client, app):
            first = client.perceive(images, ["scene"], calibration)
            instance = app.state.registry.instance("triview_3d", deployment)
            process = instance._backend._process
            if model == "da3":
                thin_images = {name: np.zeros((1, 1000, 3), np.uint8) for name in images}
                thin_calibration = {name: cal.model_copy(update={"width": 1000, "height": 1}) for name, cal in calibration.items()}
                rejected = client.perceive(thin_images, ["scene"], thin_calibration)
                assert rejected.fused is None and rejected.percepts
                assert all(p.status == "invalid_request" for p in rejected.percepts)
                assert instance._backend._process is process and process.poll() is None
            second = client.perceive(images, ["scene"], calibration)
            assert first.fused and first.fused.geometry and second.fused and second.fused.geometry
            np.testing.assert_allclose(first.fused.geometry.centroid, local_geometry.centroid, rtol=1e-3, atol=1e-4)
            np.testing.assert_allclose(first.fused.geometry.centroid, second.fused.geometry.centroid, rtol=1e-3, atol=1e-4)
            process.kill()
            process.wait(timeout=5)
            failed = client.perceive(images, ["scene"], calibration)
            assert failed.fused is None and any(p.status == "unavailable" for p in failed.percepts)
            assert "tool_unavailable" in failed.warnings
            health = next(row for row in client.health()["tools"] if row["name"] == "triview_3d")
            assert health["status"] == "error"
            report["http"] = {"direct_geometry_agrees": True, "repeat_agrees": True,
                              "killed_worker_status": "unavailable", "killed_worker_health": "error"}
            if model == "da3":
                report["http"]["invalid_shape_preserves_worker"] = True
    return report


def main() -> None:
    """Run the selected deployment and print JSON evidence.

    Example:
        >>> main()  # doctest: +SKIP
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--http", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.config, http=args.http), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
