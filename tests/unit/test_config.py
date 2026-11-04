"""Configuration model and YAML loading (spec §11.1)."""

from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from percepta.config import AppConfig, example_config_dict, load_config


def test_defaults_are_loopback_only():
    config = AppConfig()
    assert config.server.host == "127.0.0.1"
    assert config.server.port == 8790
    assert config.server.max_queue == 8
    assert config.pipeline.default == ["sam3", "multi_view_3d"]
    assert config.cache.enabled is True
    assert config.metrics.enabled is False


def test_spec_example_config_validates():
    config = AppConfig.model_validate(example_config_dict())
    assert config.tools["sam3"]["checkpoint"] == "/models/sam3.pt"
    assert config.is_enabled("sam3") is True
    assert config.is_enabled("fake") is False
    assert config.logging.json_output is True  # YAML key is `json`


def test_yaml_round_trip(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(example_config_dict()), encoding="utf-8")
    config = load_config(path)
    assert config.server.port == 8790
    assert config.tools["fake"]["enabled"] is False


def test_load_config_without_path_gives_defaults():
    assert load_config(None) == AppConfig()


def test_unknown_key_is_rejected():
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"server": {"nonsense": 1}})


def test_resolve_pipeline_honours_enabled_flags():
    config = AppConfig.model_validate(
        {
            "pipeline": {"default": ["fake", "multi_view_3d"]},
            "tools": {"fake": {"enabled": True}},
        }
    )
    assert config.resolve_pipeline() == ["fake", "multi_view_3d"]


def test_resolve_pipeline_keeps_geometry_last():
    config = AppConfig()
    assert config.resolve_pipeline(["multi_view_3d", "fake"]) == ["fake", "multi_view_3d"]


def test_resolve_pipeline_errors_when_everything_requested_is_disabled():
    config = AppConfig.model_validate({"tools": {"fake": {"enabled": False}}})
    with pytest.raises(ValueError, match="none of the requested tools"):
        config.resolve_pipeline(["fake"])


def test_token_env_resolution(monkeypatch):
    config = AppConfig()
    assert config.server.resolve_token() is None
    monkeypatch.setenv("PERCEPTA_TOKEN", "secret")
    assert config.server.resolve_token() == "secret"


def test_device_config_exports_cuda_visible_devices(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    AppConfig().device.apply()
    import os

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"
