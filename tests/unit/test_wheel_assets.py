"""Exercise the built wheel away from the source checkout, without GPU/model weights."""

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(shutil.which("uv") is None, reason="building the wheel requires the project's uv tool")
def test_installed_wheel_resolves_pinned_model_manifests(tmp_path):
    built = subprocess.run(["uv", "build", "--wheel", "--offline", "--out-dir", str(tmp_path / "dist")],
                           cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert built.returncode == 0, built.stderr
    wheel = next((tmp_path / "dist").glob("*.whl"))
    unpacked = tmp_path / "installed"
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(unpacked)
    code = """
import json
from pathlib import Path
from percepta.backends import fast_foundation_stereo_worker as ffs, mvsanywhere_adapter as mvs
assert Path(ffs.__file__).is_relative_to(Path.cwd() / 'installed')
assert Path(mvs.__file__).is_relative_to(Path.cwd() / 'installed')
assert json.loads(ffs.MANIFEST.read_text())['model'] == 'Fast-FoundationStereo 23-36-37'
assert json.loads(mvs.MANIFEST.read_text())['model'] == 'MVSAnywhere HERO'
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path,
                            env={**os.environ, "PYTHONPATH": str(unpacked)}, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
