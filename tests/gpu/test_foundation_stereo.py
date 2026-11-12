"""Real FoundationStereo inference; run explicitly with pytest -m gpu."""

import subprocess
from pathlib import Path

import numpy as np
import pytest

from scripts.foundation_stereo_smoke import run

pytestmark = pytest.mark.gpu

ROOT = Path(__file__).resolve().parents[2]
ASSETS = (
    "uv_envs/foundation-stereo/.venv/bin/python",
    "checkpoints/foundation-stereo/model_best_bp2.pth",
    "models/FoundationStereo/assets/left.png",
    "models/FoundationStereo/assets/right.png",
    "configs/local-foundation-stereo.yaml",
)


@pytest.mark.skipif(
    any(not (ROOT / path).exists() for path in ASSETS),
    reason="FoundationStereo runtime and official model assets are not deployed",
)
def test_official_pair_produces_repeatable_metric_geometry():
    report = run(str(ROOT / "configs/local-foundation-stereo.yaml"), repeats=2)
    first, second = report["calls"]
    for call in report["calls"]:
        assert call["extras"]["points"] > 100
        assert call["extras"]["median_depth_m"] > 0
        assert call["extras"]["valid_fraction"] > 0.01
        assert call["geometry"]["frame"] == "world"
        assert call["provenance"]["model_sha256"]
    np.testing.assert_allclose(
        first["geometry"]["centroid"], second["geometry"]["centroid"], atol=1e-4, rtol=1e-3
    )


@pytest.mark.skipif(
    any(not (ROOT / path).exists() for path in ASSETS),
    reason="FoundationStereo runtime and official model assets are not deployed",
)
def test_worker_matches_upstream_demo_without_network(tmp_path):
    from percepta.backends.foundation_stereo_backend import FoundationStereoBackend
    from percepta.config import load_config
    from scripts.foundation_stereo_smoke import official_pair

    config = load_config(ROOT / "configs/local-foundation-stereo.yaml").tool_config("stereo_3d")
    source = Path(config["source_dir"]).resolve()
    images, _ = official_pair(source)
    backend = FoundationStereoBackend(**{
        key: config[key] for key in (
            "python", "source_dir", "checkpoint", "model_config", "hf_cache",
            "valid_iters", "cuda_visible_devices", "startup_timeout_s", "timeout_s",
        )
    })
    try:
        actual = backend.infer_pair(images["left"], images["right"])
        first_pid = backend.describe()["pid"]
        repeated = backend.infer_pair(images["left"], images["right"])
        assert backend.describe()["pid"] == first_pid
        assert backend.describe()["max_disp"] == 416
        np.testing.assert_allclose(actual, repeated, rtol=1e-3, atol=1e-3)
        env = backend._worker_env()
    finally:
        backend.close()
    code = (
        "import runpy,sys,socket,numpy as np\n"
        "def blocked(*args,**kwargs): raise RuntimeError('network access forbidden in smoke')\n"
        "socket.socket.connect=blocked\n"
        "socket.create_connection=blocked\n"
        "demo,checkpoint,output=sys.argv[1:]\n"
        "sys.argv=[demo,'--ckpt_dir',checkpoint,'--out_dir',output,'--get_pc','0',"
        "'--remove_invisible','0','--valid_iters','32']\n"
        "state=runpy.run_path(demo,run_name='__main__')\n"
        "np.save(output+'/reference.npy',state['disp'])\n"
    )
    result = subprocess.run(
        [config["python"], "-c", code, str(source / "scripts/run_demo.py"),
         str(Path(config["checkpoint"]).resolve()), str(tmp_path)],
        env=env, capture_output=True, text=True, timeout=360,
    )
    assert result.returncode == 0, result.stderr
    reference = np.load(tmp_path / "reference.npy", allow_pickle=False)
    np.testing.assert_allclose(actual, reference, rtol=1e-3, atol=1e-3)
