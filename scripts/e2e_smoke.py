"""Independent end-to-end smoke test used by the Lead during integration.

Not part of the pytest suite: it starts a real uvicorn server on an ephemeral port, drives
it through the SDK client, and checks the two paths (in-process and HTTP) agree.

    python3 scripts/e2e_smoke.py
"""

from __future__ import annotations

import json
import socket
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from percepta import Prompt  # noqa: E402
from percepta.client import PerceptaClient  # noqa: E402
from percepta.config import load_config  # noqa: E402
from percepta.pipeline import PipelineRunner  # noqa: E402
from percepta.registry import ToolRegistry  # noqa: E402
from percepta.schema import PerceiveRequest  # noqa: E402
from tests import synth  # noqa: E402


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def scene() -> np.ndarray:
    return synth.red_square_scene()


def wait_healthy(endpoint: str, timeout: float = 30.0) -> dict:
    deadline = time.time() + timeout
    last: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{endpoint}/healthz", timeout=1.0) as response:
                return json.loads(response.read())
        except Exception as exc:  # noqa: BLE001 - server still starting
            last = exc
            time.sleep(0.3)
    raise SystemExit(f"server never became healthy: {last}")


def main() -> int:
    config = load_config(ROOT / "configs" / "ci.yaml")

    # --- in-process path -------------------------------------------------------
    runner = PipelineRunner(ToolRegistry(), config)
    image = scene()
    request = PerceiveRequest(
        images={"cam_head": synth.image_ref(image)},
        prompts=[Prompt(type="text", text="red block")],
    )
    local = runner.perceive(request)
    runner.close()
    observation = local.percepts[0].observations[0]
    print(
        "local :",
        local.percepts[0].status,
        observation.area_px,
        observation.box_xyxy,
        observation.centroid_uv,
        local.percepts[0].provenance.tool,
    )

    # --- HTTP path -------------------------------------------------------------
    # Start uvicorn in a daemon thread around the real ASGI app, so the smoke test needs no
    # module-level app and no environment plumbing.
    import threading

    import uvicorn

    from percepta.service.app import create_app

    port = free_port()
    endpoint = f"http://127.0.0.1:{port}"
    app = create_app(config)
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        health = wait_healthy(endpoint)
        print("health:", health["ok"], [t["name"] + ":" + t["status"] for t in health["tools"]])
        client = PerceptaClient(endpoint)
        remote = client.perceive(
            images={"cam_head": image},
            prompts=[Prompt(type="text", text="red block")],
        )
        remote_obs = remote.percepts[0].observations[0]
        print(
            "http  :",
            remote.percepts[0].status,
            remote_obs.area_px,
            remote_obs.box_xyxy,
            remote_obs.centroid_uv,
            remote.percepts[0].provenance.tool,
        )
        assert remote_obs.area_px == observation.area_px
        assert remote_obs.box_xyxy == observation.box_xyxy
        assert remote_obs.centroid_uv == observation.centroid_uv
        assert client.tools()
        assert client.info()

        # not_found must stay inside HTTP 200 (spec §6)
        missing = client.perceive(
            images={"cam_head": image},
            prompts=[Prompt(type="text", text="blue block")],
        )
        assert missing.percepts[0].status == "not_found"
        print("nf    : HTTP 200 with status =", missing.percepts[0].status)
    finally:
        server.should_exit = True
        thread.join(timeout=10)
    print("E2E OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
