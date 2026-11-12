"""Real Fast-FoundationStereo regression checks; request explicitly with pytest -m gpu."""

from pathlib import Path

import numpy as np
import pytest

from scripts.fast_foundation_stereo_smoke import run, run_http

pytestmark = pytest.mark.gpu
ROOT = Path(__file__).resolve().parents[2]
ASSETS = (
    "uv_envs/fast-foundation-stereo/.venv/bin/python",
    "checkpoints/fast-foundation-stereo/model_best_bp2_serialize.pth",
    "models/Fast-FoundationStereo/demo_data/left.png",
    "configs/local-fast-foundation-stereo.yaml",
)


@pytest.mark.skipif(any(not (ROOT / path).exists() for path in ASSETS), reason="Fast-FoundationStereo is not deployed")
def test_official_pair_is_repeatable_and_matches_upstream_without_network():
    report = run(str(ROOT / "configs/local-fast-foundation-stereo.yaml"), repeats=2, upstream=True)
    assert report["resident_process_reused"] and report["upstream"]["agrees"]
    first, second = report["calls"]
    for call in report["calls"]:
        assert call["extras"]["points"] > 100 and call["extras"]["median_depth_m"] > 0
        assert call["extras"]["valid_fraction"] > 0.01
        assert call["geometry"]["frame"] == "world"
        assert call["provenance"]["model_sha256"]
    np.testing.assert_allclose(first["geometry"]["centroid"], second["geometry"]["centroid"], atol=1e-4, rtol=1e-3)


@pytest.mark.skipif(any(not (ROOT / path).exists() for path in ASSETS), reason="Fast-FoundationStereo is not deployed")
def test_real_http_has_geometry_and_preserves_model_failure_semantics():
    report = run_http(str(ROOT / "configs/local-fast-foundation-stereo.yaml"))
    assert report["repeat_agrees"] and report["empty_mask_has_no_geometry"]
    assert report["killed_worker_status"] == "unavailable"
