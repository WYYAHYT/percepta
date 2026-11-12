"""Real resident model, geometry and loopback HTTP for the two neural three-view backends."""

from pathlib import Path

import pytest

from scripts.multiview_smoke import run

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("model", ["da3", "mvsanywhere"])
def test_neural_triview_worker_and_http(model):
    if not (ROOT / f"uv_envs/{model}/.venv/bin/python").exists():
        pytest.skip("isolated model environment is not deployed")
    report = run(str(ROOT / f"configs/local-{model}.yaml"), http=True)
    assert report["resident_process_reused"] and report["http"]["direct_geometry_agrees"]
    assert report["http"]["killed_worker_status"] == "unavailable"
    assert report["http"]["killed_worker_health"] == "error"
    if model == "da3":
        assert report["http"]["invalid_shape_preserves_worker"]
