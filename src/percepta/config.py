"""Configuration model and YAML loading (spec §11.1).

Deliberately built on plain ``pydantic`` + ``pyyaml`` rather than ``pydantic-settings``:
that package is not in the spec's dependency list (§13) and adding it is not worth the
supply-chain surface for reading one YAML file. Environment overrides are limited to the
documented ``token_env`` indirection.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "ServerConfig",
    "DeviceConfig",
    "PipelineConfig",
    "CacheConfig",
    "LoggingConfig",
    "MetricsConfig",
    "AppConfig",
    "load_config",
    "example_config_dict",
]

DEFAULT_PIPELINE = ["sam3", "multi_view_3d"]
"""Spec §5.3: the default pipeline is ``[<default segmenter>, multi_view_3d]``."""


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ServerConfig(_Section):
    host: str = "127.0.0.1"  # loopback only; there is deliberately no 0.0.0.0 default
    port: int = 8790
    max_body_bytes: int = 33_554_432
    max_queue: int = 8
    request_timeout_ms: int = 120_000
    token_env: str = "PERCEPTA_TOKEN"

    def resolve_token(self) -> str | None:
        """Bearer token from the environment; ``None`` disables authentication."""
        value = os.environ.get(self.token_env) if self.token_env else None
        return value or None


class DeviceConfig(_Section):
    cuda_visible_devices: str = "0"

    def apply(self) -> None:
        """Export ``CUDA_VISIBLE_DEVICES`` before any model import happens."""
        if self.cuda_visible_devices:
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", self.cuda_visible_devices)


class PipelineConfig(_Section):
    default: list[str] = Field(default_factory=lambda: list(DEFAULT_PIPELINE))

    @model_validator(mode="after")
    def _check_names(self) -> PipelineConfig:
        if not self.default:
            raise ValueError("pipeline.default must list at least one tool")
        return self


class CacheConfig(_Section):
    enabled: bool = True
    max_entries: int = 256
    ttl_s: float = 300.0


class LoggingConfig(_Section):
    """YAML ``logging: {level, json}``.

    The YAML key is ``json`` (spec §11.1), but a field named ``json`` would shadow
    ``BaseModel.json``, so the attribute is ``json_output`` and ``json`` is its alias.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, populate_by_name=True)

    level: str = "info"
    json_output: bool = Field(default=True, alias="json")


class MetricsConfig(_Section):
    enabled: bool = False


class AppConfig(_Section):
    """The whole ``config.yaml`` for ``percepta serve`` / ``percepta probe``."""

    server: ServerConfig = Field(default_factory=ServerConfig)
    device: DeviceConfig = Field(default_factory=DeviceConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    tools: dict[str, dict[str, Any]] = Field(default_factory=dict)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)

    def tool_config(self, name: str) -> dict[str, Any]:
        return dict(self.tools.get(name) or {})

    def is_enabled(self, name: str) -> bool:
        cfg = self.tools.get(name)
        if cfg is None:
            return True  # unlisted tools keep their own defaults
        return bool(cfg.get("enabled", True))

    def resolve_pipeline(self, requested: list[str] | None = None) -> list[str]:
        """Pipeline for a request, dropping tools disabled in config.

        An explicitly requested pipeline that becomes empty is a caller error; falling back
        silently would hide a typo in ``options.tools``.
        """
        names = list(requested) if requested else list(self.pipeline.default)
        kept = [n for n in names if n != "multi_view_3d" and self.is_enabled(n)]
        if "multi_view_3d" in names and self.is_enabled("multi_view_3d"):
            kept.append("multi_view_3d")
        if requested and not kept:
            raise ValueError(f"none of the requested tools {requested} are enabled")
        return kept


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load ``config.yaml``; with no path, return defaults (no file required)."""
    if path is None:
        return AppConfig()
    text = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: config root must be a mapping")
    return AppConfig.model_validate(data)


def example_config_dict() -> dict[str, Any]:
    """The example from spec §11.1, for docs and round-trip tests."""
    return {
        "server": {
            "host": "127.0.0.1",
            "port": 8790,
            "max_body_bytes": 33_554_432,
            "max_queue": 8,
            "request_timeout_ms": 120_000,
            "token_env": "PERCEPTA_TOKEN",
        },
        "device": {"cuda_visible_devices": "0"},
        "pipeline": {"default": ["sam3", "multi_view_3d"]},
        "tools": {
            "sam3": {
                "enabled": True,
                "checkpoint": "/models/sam3.pt",
                "resolution": 1008,
                "min_score": 0.2,
            },
            "fake": {"enabled": False},
        },
        "cache": {"enabled": True, "max_entries": 256, "ttl_s": 300},
        "logging": {"level": "info", "json": True},
        "metrics": {"enabled": False},
    }
