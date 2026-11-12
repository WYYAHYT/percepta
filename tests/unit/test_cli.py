"""Perception CLI exits distinguish tool failures from normal empty detections."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from percepta.cli import main
from percepta.schema import PerceiveResponse, Percept, Provenance


@pytest.mark.parametrize("status", ["unavailable", "invalid_request", "timeout", "not_found", "ok"])
@pytest.mark.parametrize("fused", [False, True])
def test_cli_result_status_controls_exit_code(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], status: str, fused: bool,
) -> None:
    percept = Percept(percept_id="cli-test", status=status, provenance=Provenance(tool="fake", tool_version="1", kind="fake"))
    response = PerceiveResponse(request_id="cli-test", percepts=[] if fused else [percept], fused=percept if fused else None)
    monkeypatch.setattr("percepta.cli.PerceptaClient.perceive", lambda *_args, **_kwargs: response)
    code = main(["perceive", "--image", "cam=unused.png", "--prompt", "text:object", "--json"])
    assert code == (1 if status in {"unavailable", "invalid_request", "timeout"} else 0)
    assert json.loads(capsys.readouterr().out)["request_id"] == "cli-test"


def test_cli_real_missing_weights_fails_but_empty_fake_detection_succeeds(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    image = tmp_path / "scene.png"
    Image.new("RGB", (8, 8), (255, 0, 0)).save(image)
    config = tmp_path / "unavailable.yaml"
    config.write_text(
        f"pipeline:\n  default: [sam3]\ntools:\n  sam3:\n    checkpoint: {tmp_path / 'missing.pt'}\n"
        "cache:\n  enabled: false\n", encoding="utf-8",
    )
    common = ["perceive", "--local", "--image", f"cam={image}", "--prompt", "text:blue block", "--json", "--no-fuse"]
    assert main([*common, "--config", str(config)]) == 1
    assert json.loads(capsys.readouterr().out)["percepts"][0]["status"] == "unavailable"
    assert main([*common, "--tools", "fake"]) == 0
    assert json.loads(capsys.readouterr().out)["percepts"][0]["status"] == "not_found"
