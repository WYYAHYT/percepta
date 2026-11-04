"""Shared pytest configuration.

Tests run from a plain checkout (``PYTHONPATH`` injection) *and* from an installed package,
so CI does not depend on ``pip install -e .`` having happened first.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "gpu: requires a GPU and/or model weights; skipped by default"
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip ``-m gpu`` tests unless GPU runs were requested explicitly."""
    if config.getoption("-m") and "gpu" in str(config.getoption("-m")):
        return
    skip = pytest.mark.skip(reason="GPU/weights test; run with -m gpu")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)
