"""Exercise the real isolated Fast-FoundationStereo worker, geometry and loopback HTTP."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from percepta.config import load_config  # noqa: E402
from percepta.schema import Calibration, ToolContext  # noqa: E402
from percepta.tools.stereo_3d import Stereo3DTool  # noqa: E402
from scripts.foundation_stereo_smoke import live_client  # noqa: E402


def official_pair(source: Path) -> tuple[dict, dict]:
    """Read the upstream demo's rectified RGB pair and camera-to-world calibration.

    Args:
        source: Local Fast-FoundationStereo source checkout.

    Returns:
        Image and calibration dictionaries; +z is forward and p_world=T_world_cam@p_cam_h.

    Example:
        >>> images, calibration = official_pair(ROOT / "models/Fast-FoundationStereo")  # doctest: +SKIP
    """
    images = {}
    for name in ("left", "right"):
        with Image.open(source / "demo_data" / f"{name}.png") as image:
            images[name] = np.array(image.convert("RGB"))
    lines = (source / "demo_data/K.txt").read_text().splitlines()
    intrinsic = np.fromstring(lines[0], sep=" ").reshape(3, 3)
    baseline = float(lines[1])
    calibration = {}
    for name, x in (("left", 0.0), ("right", baseline)):
        transform = np.eye(4)
        transform[0, 3] = x
        height, width = images[name].shape[:2]
        calibration[name] = Calibration(camera=name, width=width, height=height, K=intrinsic.tolist(),
                                        T_world_cam=transform.tolist(), id="fast-foundation-stereo-official-demo")
    return images, calibration


def compare_upstream(backend, config: dict, images: dict) -> dict:
    """Compare worker disparity with the unmodified official demo while blocking network IO.

    Args:
        backend: Loaded resident stereo backend.
        config: Trusted tool deployment configuration.
        images: Official rectified RGB images.

    Returns:
        Pixelwise agreement and maximum absolute disparity difference in pixels.

    Example:
        >>> compare_upstream(backend, config, images)["agrees"]  # doctest: +SKIP
        True
    """
    actual = backend.infer_pair(images["left"], images["right"])
    source = Path(config["source_dir"]).resolve()
    code = (
        "import runpy,sys,socket,numpy as np,cv2\n"
        "def blocked(*args,**kwargs): raise RuntimeError('network access forbidden in smoke')\n"
        "socket.socket.connect=blocked\nsocket.create_connection=blocked\n"
        "cv2.imshow=lambda *a,**k:None\ncv2.waitKey=lambda *a,**k:0\n"
        "demo,checkpoint,output,iters,maxdisp=sys.argv[1:]\n"
        "sys.argv=[demo,'--model_dir',checkpoint,'--out_dir',output,'--get_pc','0',"
        "'--remove_invisible','0','--valid_iters',iters,'--max_disp',maxdisp]\n"
        "state=runpy.run_path(demo,run_name='__main__')\n"
        "np.save(output+'/reference.npy',state['disp'])\n"
    )
    with tempfile.TemporaryDirectory(prefix="percepta-ffs-reference-") as temporary:
        output = str(Path(temporary) / "demo-output")
        result = subprocess.run([
            config["python"], "-c", code, str(source / "scripts/run_demo.py"),
            str(Path(config["checkpoint"]).resolve()), output, str(config.get("valid_iters", 8)),
            str(config.get("max_disp", 416)),
        ], env=backend._worker_env(), capture_output=True, text=True, timeout=600)
        assert result.returncode == 0, result.stderr
        reference = np.load(Path(output) / "reference.npy", allow_pickle=False)
    clipped = np.maximum(actual, 0)  # The official demo clips only after raw model inference.
    np.testing.assert_allclose(clipped, reference, rtol=1e-3, atol=1e-3)
    return {"agrees": True, "maximum_absolute_difference_px": float(np.max(np.abs(clipped - reference))),
            "network_blocked_in_reference": True}


def run(config_path: str, repeats: int = 3, upstream: bool = False) -> dict:
    """Run repeated real model calls and assert valid metric geometry and process reuse.

    Args:
        config_path: Trusted local YAML deployment file.
        repeats: Positive number of geometry requests.
        upstream: Also compare with the official, unmodified run_demo.py.

    Returns:
        JSON-serializable model timings, geometry, provenance and optional reference comparison.

    Example:
        >>> run("configs/local-fast-foundation-stereo.yaml", repeats=2)["resident_process_reused"]  # doctest: +SKIP
        True
    """
    if repeats < 1:
        raise ValueError("repeats must be positive")
    config = load_config(config_path).tool_config("stereo_3d")
    images, calibration = official_pair(Path(config["source_dir"]))
    tool = Stereo3DTool()
    report = {"image_shape": list(images["left"].shape), "calls": []}
    started = time.perf_counter()
    try:
        tool.load(config)
        report["load_ms"] = (time.perf_counter() - started) * 1000
        process = tool._backend._process
        for index in range(repeats):
            context = ToolContext(request_id=f"ffs-smoke-{index}", images=images, calibration=calibration,
                                  context={"stereo_left": "left", "stereo_right": "right"})
            started = time.perf_counter()
            percepts = tool.perceive(context, [])
            assert len(percepts) == 1 and percepts[0].status == "ok", str(percepts)
            percept = percepts[0]
            assert percept.geometry and percept.geometry.views_used == 2 and percept.geometry.centroid[2] > 0
            assert percept.provenance.model_sha256
            assert percept.extras["backend"] == "ffs"
            assert tool._backend._process is process and process.poll() is None
            report["calls"].append({"elapsed_ms": (time.perf_counter() - started) * 1000,
                                    "geometry": percept.geometry.model_dump(mode="json"),
                                    "provenance": percept.provenance.model_dump(mode="json"), "extras": percept.extras})
        report["resident_process_reused"] = True
        if upstream:
            report["upstream"] = compare_upstream(tool._backend, config, images)
    finally:
        tool.close()
    return report


def run_http(config_path: str) -> dict:
    """Check loopback HTTP geometry, repeatability, empty masks and worker failure semantics.

    Args:
        config_path: Trusted local YAML deployment file.

    Returns:
        HTTP checks and the first fused geometry response.

    Example:
        >>> run_http("configs/local-fast-foundation-stereo.yaml")["repeat_agrees"]  # doctest: +SKIP
        True
    """
    config = load_config(config_path)
    images, calibration = official_pair(Path(config.tool_config("stereo_3d")["source_dir"]))
    order = {"stereo_left": "left", "stereo_right": "right"}
    with live_client(config) as (client, app):
        first = client.perceive(images, ["scene"], calibration, order)
        second = client.perceive(images, ["scene"], calibration, order)
        assert first.fused and first.fused.geometry and second.fused and second.fused.geometry
        assert first.fused.provenance.model_sha256
        np.testing.assert_allclose(first.fused.geometry.centroid, second.fused.geometry.centroid, rtol=1e-3, atol=1e-4)
        empty = client.perceive(images, ["scene"], calibration,
                               {**order, "stereo_mask": {"left": np.zeros(images["left"].shape[:2], bool).tolist()}})
        assert empty.fused is None or empty.fused.geometry is None
        tool = app.state.registry.instance("stereo_3d", config.tool_config("stereo_3d"))
        tool._backend._process.kill()
        tool._backend._process.wait(timeout=5)
        failed = client.perceive(images, ["scene"], calibration, order)
        assert any(percept.status == "unavailable" for percept in failed.percepts)
        assert failed.fused is None or failed.fused.geometry is None
    return {"fused": first.fused.model_dump(mode="json"), "repeat_agrees": True,
            "empty_mask_has_no_geometry": True, "killed_worker_status": "unavailable"}


def main() -> None:
    """Run the requested smoke checks and print a strict JSON evidence report.

    Example:
        >>> main()  # doctest: +SKIP
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/local-fast-foundation-stereo.yaml")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--http", action="store_true")
    parser.add_argument("--upstream", action="store_true")
    args = parser.parse_args()
    report = run(args.config, args.repeats, args.upstream)
    if args.http:
        report["http"] = run_http(args.config)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
