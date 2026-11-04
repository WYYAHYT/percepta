"""Fixtures for the contract suite: a FastAPI app built from ``configs/ci.yaml``.

No GPU, no network, no weights: ``configs/ci.yaml`` pins the ``fake`` segmenter (decision D9)
and the app is exercised through ``TestClient`` (an in-process ASGI transport).
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from percepta.config import AppConfig, load_config
from percepta.service.app import create_app
from tests.contract.helpers import CI_CONFIG


@pytest.fixture()
def ci_config() -> AppConfig:
    """The CI config: ``fake`` enabled, ``sam3`` disabled (spec §16.2, decision D9)."""
    return load_config(CI_CONFIG)


@pytest.fixture()
def client(ci_config: AppConfig) -> Iterator[TestClient]:
    """A ``TestClient`` over an app built from ``configs/ci.yaml``."""
    with TestClient(create_app(ci_config)) as test_client:
        yield test_client
