"""Opt-in DA3-BASE GPU inference with explicit metre-scale evidence."""

import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / "uv_envs/da3/.venv/bin/python"
pytestmark = pytest.mark.gpu


@pytest.mark.skipif(not PYTHON.exists() or not (ROOT / "checkpoints/da3/model.safetensors").exists(),
                    reason="DA3 runtime and official local weights are not deployed")
def test_da3_real_inference_and_metric_scale():
    """Execute the isolated model twice, then double calibration translations."""
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}
    code = (
        "import json,socket; "
        "socket.socket.connect=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('offline smoke')); "
        "from scripts.da3_smoke import run; print(json.dumps(run()))"
    )
    result = subprocess.run([str(PYTHON), "-c", code], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=360)
    assert result.returncode == 0, result.stderr + result.stdout
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["repeat_agrees"] and report["metric_scale_agrees"] and report["world_origin_agrees"]
    assert report["calls"][0]["model"] == "DA3-BASE"
    assert report["calls"][1]["calls"] == 2
