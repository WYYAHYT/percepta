"""Official, prepared HERO three-view inference; explicitly select with -m gpu."""

import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.gpu


@pytest.mark.skipif(
    not (ROOT / "uv_envs/mvsanywhere/.venv/bin/python").exists()
    or not (ROOT / "checkpoints/mvsanywhere/mvsanywhere_hero.ckpt").exists(),
    reason="Official MVSAnywhere runtime/checkpoint are not prepared",
)
def test_real_offline_three_view_inference_repeats_and_matches_upstream(tmp_path):
    report = tmp_path / "report.json"
    env = dict(os.environ, OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1")
    completed = subprocess.run(
        [str(ROOT / "uv_envs/mvsanywhere/.venv/bin/python"), str(ROOT / "scripts/mvsanywhere_smoke.py"),
         "--repeats", "2", "--output", str(report)],
        env=env, capture_output=True, text=True, timeout=600,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(report.read_text())
    assert result["repeated_consistent"] and result["upstream_reference_consistent"] and result["network_blocked"]
    assert result["world_basis_consistent"]
    assert len(result["calls"]) == 2
    for call in result["calls"]:
        assert len(call["views"]) == 3
        assert all(view["valid_fraction"] > 0.05 and view["median_depth_m"] > 0 for view in call["views"])
