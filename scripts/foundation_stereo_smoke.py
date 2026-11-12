"""Run the deployed stereo tool on the official calibrated RGB pair."""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from percepta.config import load_config  # noqa: E402
from percepta.schema import Calibration, ToolContext  # noqa: E402
from percepta.tools.stereo_3d import Stereo3DTool  # noqa: E402


def official_pair(source: Path) -> tuple[dict, dict]:
    images = {}
    for name in ("left", "right"):
        with Image.open(source / "assets" / f"{name}.png") as image:
            images[name] = np.array(image.convert("RGB"))
    lines = (source / "assets/K.txt").read_text().splitlines()
    intrinsic = np.fromstring(lines[0], sep=" ").reshape(3, 3)
    baseline = float(lines[1])
    calibration = {}
    for name, x in (("left", 0.0), ("right", baseline)):
        transform = np.eye(4)
        transform[0, 3] = x
        height, width = images[name].shape[:2]
        calibration[name] = Calibration(
            camera=name,
            width=width,
            height=height,
            K=intrinsic.tolist(),
            T_world_cam=transform.tolist(),
            id="foundation-stereo-official-demo",
        )
    return images, calibration


def run(config_path: str, repeats: int = 3) -> dict:
    config = load_config(config_path)
    tool_config = config.tool_config("stereo_3d")
    images, calibration = official_pair(Path(tool_config["source_dir"]))
    tool = Stereo3DTool()
    report: dict = {"image_shape": list(images["left"].shape), "calls": []}
    started = time.perf_counter()
    try:
        tool.load(tool_config)
        report["load_ms"] = (time.perf_counter() - started) * 1000
        for index in range(repeats):
            context = ToolContext(
                request_id=f"foundation-stereo-smoke-{index}",
                images=images,
                calibration=calibration,
                context={"stereo_left": "left", "stereo_right": "right"},
            )
            started = time.perf_counter()
            percepts = tool.perceive(context, [])
            elapsed = (time.perf_counter() - started) * 1000
            assert len(percepts) == 1, "no geometry returned for the official image pair"
            percept = percepts[0]
            assert percept.status == "ok", percept.message
            assert percept.geometry is not None
            assert percept.geometry.views_used == 2
            assert percept.geometry.centroid[2] > 0, "world points must be in front of the cameras"
            assert percept.provenance.model_sha256, "checkpoint identity is missing"
            assert percept.extras["backend"] == "foundation_stereo"
            report["calls"].append(
                {
                    "elapsed_ms": elapsed,
                    "geometry": percept.geometry.model_dump(mode="json"),
                    "provenance": percept.provenance.model_dump(mode="json"),
                    "extras": percept.extras,
                }
            )
    finally:
        tool.close()
    return report


@contextmanager
def live_client(config):
    import uvicorn

    from percepta.client import PerceptaClient
    from percepta.service.app import create_app

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    app = create_app(config)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 30
        while not server.started:
            if not thread.is_alive() or time.monotonic() >= deadline:
                raise RuntimeError("loopback server did not start")
            thread.join(timeout=0.05)
        with PerceptaClient(
            f"http://127.0.0.1:{port}",
            token=os.environ.get(config.server.token_env),
            timeout=360,
            retries=0,
        ) as client:
            yield client, app
    finally:
        server.should_exit = True
        thread.join(timeout=15)
        assert not thread.is_alive(), "loopback server did not stop"


def run_http(config_path: str) -> dict:
    config = load_config(config_path)
    images, calibration = official_pair(Path(config.tool_config("stereo_3d")["source_dir"]))
    order = {"stereo_left": "left", "stereo_right": "right"}
    with live_client(config) as (client, app):
        first = client.perceive(images, ["scene"], calibration, order)
        second = client.perceive(images, ["scene"], calibration, order)
        assert first.fused and first.fused.geometry
        assert first.fused.provenance.model_sha256
        assert second.fused and second.fused.geometry
        np.testing.assert_allclose(
            first.fused.geometry.centroid, second.fused.geometry.centroid, rtol=1e-3, atol=1e-4
        )
        empty = client.perceive(
            images, ["scene"], calibration,
            {**order, "stereo_mask": {"left": np.zeros(images["left"].shape[:2], dtype=bool).tolist()}},
        )
        assert empty.fused is None or empty.fused.geometry is None
        tool = app.state.registry.instance("stereo_3d", config.tool_config("stereo_3d"))
        process = tool._backend._process
        process.kill()
        process.wait(timeout=5)
        failed = client.perceive(images, ["scene"], calibration, order)
        assert failed.fused is None or failed.fused.geometry is None
        assert any(p.status == "unavailable" for p in failed.percepts)
        assert tool._backend.describe()["state"] == "failed"
    broken = config.model_copy(deep=True)
    broken.tools["stereo_3d"]["checkpoint"] = str(ROOT / "checkpoints/foundation-stereo/missing.pth")
    with live_client(broken) as (client, _app):
        unavailable = client.perceive(images, ["scene"], calibration, order)
        assert unavailable.fused is None or unavailable.fused.geometry is None
        assert any(p.status == "unavailable" for p in unavailable.percepts)
        assert unavailable.warnings
    return {
        "fused": first.fused.model_dump(mode="json"),
        "repeat_agrees": True,
        "empty_mask_has_no_geometry": True,
        "missing_checkpoint_status": "unavailable",
        "killed_worker_status": "unavailable",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/local-foundation-stereo.yaml")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--http", action="store_true", help="also exercise real loopback HTTP")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    report = run(args.config, args.repeats)
    if args.http:
        report["http"] = run_http(args.config)
        np.testing.assert_allclose(
            report["calls"][0]["geometry"]["centroid"],
            report["http"]["fused"]["geometry"]["centroid"], rtol=1e-3, atol=1e-4,
        )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
