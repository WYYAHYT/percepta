"""``percepta`` command line interface (spec §7).

Frozen CLI surface::

    percepta serve --config config.yaml
    percepta perceive --image cam_head=img.png --prompt text:bottle [--calib calib.json] [--json]
    percepta tools list
    percepta probe [--config config.yaml]

Built on :mod:`argparse` only (no ``click``). Exit codes: ``0`` success, ``1`` failure
(config error, tool error, HTTP error, failed probe), ``2`` bad usage (argparse),
``130`` interrupted.

Safety rule (spec §12): ``serve`` binds ``127.0.0.1`` by default and refuses any
non-loopback address unless ``--allow-remote`` is given explicitly.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import __version__
from .client import DEFAULT_ENDPOINT, PerceptaClient, PerceptaError
from .config import load_config
from .pipeline import prepare_tools
from .registry import ToolEntry, ToolRegistry
from .schema import Calibration, PerceiveOptions, PerceiveResponse, Prompt
from .weights import WeightError, resolve_weights

__all__ = ["main", "parse_prompt"]

_LOOPBACK_NAMES = {"localhost", "localhost.localdomain", "ip6-localhost"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """Parse ``argv`` and dispatch to a subcommand; returns the process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 0
    try:
        return int(handler(args) or 0)
    except KeyboardInterrupt:  # pragma: no cover - interactive
        print("percepta: interrupted", file=sys.stderr)
        return 130
    except PerceptaError as exc:
        print(f"percepta: {exc}", file=sys.stderr)
        body = getattr(exc, "body", None)
        if isinstance(body, dict) and body.get("message"):
            print(f"percepta: server said: {body['message']}", file=sys.stderr)
        return 1
    except (OSError, ValueError) as exc:
        print(f"percepta: error: {exc}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="percepta",
        description="Pluggable robot-perception layer: images + prompts -> serialisable percepts.",
    )
    parser.add_argument("--version", action="version", version=f"percepta {__version__}")
    subparsers = parser.add_subparsers(dest="command", metavar="{serve,perceive,tools,probe}")

    # -- serve -----------------------------------------------------------
    serve = subparsers.add_parser("serve", help="run the loopback HTTP service (spec §6)")
    serve.add_argument("--config", type=Path, default=None, help="YAML config file")
    serve.add_argument("--host", default=None, help="bind address (default: config server.host)")
    serve.add_argument("--port", type=int, default=None, help="bind port (default: config)")
    serve.add_argument(
        "--allow-remote",
        action="store_true",
        help="allow binding a non-loopback address (spec §12 forbids it by default)",
    )
    serve.add_argument("--log-level", default=None, help="override logging.level")
    serve.set_defaults(handler=_cmd_serve)

    # -- perceive --------------------------------------------------------
    perceive = subparsers.add_parser("perceive", help="run one perception request (spec §7)")
    perceive.add_argument(
        "--image",
        action="append",
        required=True,
        metavar="CAM=PATH",
        help="camera name and image file; repeat for multiple cameras",
    )
    perceive.add_argument(
        "--prompt",
        action="append",
        required=True,
        metavar="TYPE:VALUE",
        help="text:bottle | box:cx,cy,w,h (normalised) | point:u,v",
    )
    perceive.add_argument("--calib", type=Path, default=None, help="calibration JSON file")
    perceive.add_argument("--endpoint", default=None, help=f"server URL (default {DEFAULT_ENDPOINT})")
    perceive.add_argument("--token", default=None, help="bearer token, if the server needs one")
    perceive.add_argument("--config", type=Path, default=None, help="config file for --local")
    perceive.add_argument(
        "--local", action="store_true", help="run in-process instead of over HTTP"
    )
    perceive.add_argument("--tools", default=None, metavar="a,b", help="override the pipeline")
    perceive.add_argument("--no-fuse", action="store_true", help="disable multi-view fusion")
    perceive.add_argument("--json", dest="json_output", action="store_true", help="print JSON")
    perceive.set_defaults(handler=_cmd_perceive)

    # -- tools -----------------------------------------------------------
    tools = subparsers.add_parser("tools", help="inspect registered tools")
    tools_sub = tools.add_subparsers(dest="tools_command", metavar="{list}")
    tools_list = tools_sub.add_parser("list", help="list tools, versions and capabilities")
    tools_list.add_argument("--config", type=Path, default=None)
    tools_list.add_argument("--endpoint", default=None, help="query a running server instead")
    tools_list.add_argument("--token", default=None)
    tools_list.add_argument("--json", dest="json_output", action="store_true")
    tools_list.set_defaults(handler=_cmd_tools_list)

    # -- probe -----------------------------------------------------------
    probe = subparsers.add_parser("probe", help="check the config and tool availability")
    probe.add_argument("--config", type=Path, default=None)
    probe.add_argument("--endpoint", default=None, help="probe a running server instead")
    probe.add_argument("--token", default=None)
    probe.add_argument("--json", dest="json_output", action="store_true")
    probe.set_defaults(handler=_cmd_probe)

    # -- models ----------------------------------------------------------
    models = subparsers.add_parser("models", help="inspect model weight assets")
    models_sub = models.add_subparsers(dest="models_command", metavar="{check}")
    models_check = models_sub.add_parser(
        "check", help="verify configured checkpoints exist and match their sha256"
    )
    models_check.add_argument("--config", type=Path, default=None)
    models_check.add_argument(
        "--no-hash",
        action="store_true",
        help="only check existence, skip hashing (fast; misses corrupt files)",
    )
    models_check.add_argument("--json", dest="json_output", action="store_true")
    models_check.set_defaults(handler=_cmd_models_check)

    return parser


# ---------------------------------------------------------------------------
# Argument parsing helpers
# ---------------------------------------------------------------------------


def is_loopback(host: str) -> bool:
    """True for ``localhost`` and any literal address in a loopback range."""
    candidate = (host or "").strip().strip("[]")
    if not candidate:
        return False
    lowered = candidate.lower()
    if lowered in _LOOPBACK_NAMES or lowered.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(lowered).is_loopback
    except ValueError:
        return False


def parse_prompt(value: str) -> Prompt:
    """Parse ``text:bottle`` / ``box:cx,cy,w,h`` / ``point:u,v``.

    A value without a known ``type:`` prefix is treated as a text prompt, so
    ``--prompt bottle`` works too.
    """
    kind, sep, rest = value.partition(":")
    kind = kind.strip().lower()
    if not sep or kind not in {"text", "box", "point"}:
        if not value.strip():
            raise ValueError("prompt must not be empty")
        return Prompt(type="text", text=value.strip())
    rest = rest.strip()
    if kind == "text":
        if not rest:
            raise ValueError("text prompt must not be empty")
        return Prompt(type="text", text=rest)
    parts = [part for part in rest.replace(";", ",").split(",") if part.strip()]
    if kind == "box":
        if len(parts) != 4:
            raise ValueError("box prompt needs 4 comma-separated numbers (cx,cy,w,h)")
        return Prompt(type="box", box=[float(part) for part in parts])
    if len(parts) != 2:
        raise ValueError("point prompt needs 2 comma-separated integers (u,v)")
    return Prompt(type="point", point=[int(float(part)) for part in parts])


def _parse_images(values: Sequence[str]) -> dict[str, Path]:
    images: dict[str, Path] = {}
    for value in values:
        camera, sep, path = value.partition("=")
        if not sep or not camera.strip() or not path.strip():
            raise ValueError(f"--image expects CAM=PATH, got {value!r}")
        images[camera.strip()] = Path(path.strip())
    return images


def _load_calibration(path: Path | None) -> dict[str, Calibration]:
    if path is None:
        return {}
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(document, dict) and isinstance(document.get("cameras"), dict):
        document = document["cameras"]
    if isinstance(document, dict) and "K" in document and "camera" in document:
        single = Calibration.model_validate(document)
        return {single.camera: single}
    if not isinstance(document, dict):
        raise ValueError(f"{path}: calibration must be a JSON object keyed by camera")
    return {camera: Calibration.model_validate(value) for camera, value in document.items()}


def _options_from_args(args: argparse.Namespace) -> PerceiveOptions:
    tools = None
    if getattr(args, "tools", None):
        tools = [name.strip() for name in str(args.tools).split(",") if name.strip()]
    return PerceiveOptions(tools=tools, fuse=not getattr(args, "no_fuse", False))


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def _cmd_serve(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    host = args.host or config.server.host
    port = int(args.port or config.server.port)
    if not is_loopback(host) and not args.allow_remote:
        raise ValueError(
            f"refusing to bind non-loopback host {host!r} (spec §12); "
            "pass --allow-remote if you really mean it"
        )
    config.device.apply()  # must happen before any model import

    from .service.app import create_app  # fastapi/uvicorn are the optional `service` extra

    logging_config = config.logging
    app = create_app(config)
    import uvicorn

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=str(args.log_level or logging_config.level).lower(),
    )
    return 0


def _cmd_perceive(args: argparse.Namespace) -> int:
    images = _parse_images(args.image)
    prompts = [parse_prompt(value) for value in args.prompt]
    calibration = _load_calibration(args.calib)
    options = _options_from_args(args)

    if args.local:
        client = PerceptaClient()
        try:
            response = client.perceive_local(
                images=images,
                prompts=prompts,
                calibration=calibration,
                options=options,
                config=load_config(args.config),
            )
        finally:
            client.close()
    else:
        endpoint = args.endpoint or DEFAULT_ENDPOINT
        client = PerceptaClient(endpoint, token=args.token)
        response = client.perceive(
            images=images, prompts=prompts, calibration=calibration, options=options
        )

    if args.json_output:
        print(response.model_dump_json(indent=2))
    else:
        print(_format_response(response))
    return 0


def _cmd_tools_list(args: argparse.Namespace) -> int:
    if args.endpoint:
        rows = PerceptaClient(args.endpoint, token=args.token).tools()
    else:
        config = load_config(args.config)
        registry = ToolRegistry()
        registry.configure(config.tools)
        rows = [_local_tool_row(registry.entry(name)) for name in registry.names()]

    if args.json_output:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        return 0
    for row in rows:
        capabilities = row.get("capabilities") or {}
        prompts = ",".join(capabilities.get("prompts") or []) or "-"
        loaded = "yes" if row.get("loaded") else "no"
        print(
            f"{row['name']:<16} {row['version']:<8} {row['status']:<8} "
            f"{(row.get('kind') or '-'):<14} loaded={loaded:<4} prompts={prompts}"
        )
        if row.get("error"):
            print(f"{'':<16} error: {row['error']}")
    return 0


def _cmd_probe(args: argparse.Namespace) -> int:
    if args.endpoint:
        health = PerceptaClient(args.endpoint, token=args.token).health()
        bad = [tool["name"] for tool in health.get("tools", []) if tool.get("status") == "error"]
        report: dict[str, Any] = {
            "ok": bool(health.get("ok")) and not bad,
            "endpoint": args.endpoint,
            "uptime_s": health.get("uptime_s"),
            "tools": health.get("tools", []),
            "unavailable": bad,
        }
    else:
        config = load_config(args.config)
        registry = ToolRegistry()
        # Same startup step as `serve`: config tools + the pipeline warm-up, so the probe
        # reports what the service would really report on /healthz (decision D9).
        prepare_tools(registry, config)
        pipeline = config.resolve_pipeline()
        rows = [_local_tool_row(registry.entry(name)) for name in registry.names()]
        by_name = {row["name"]: row for row in rows}
        missing = [
            name
            for name in pipeline
            if (by_name.get(name) or {}).get("status") == "error" or name not in by_name
        ]
        report = {
            "ok": not missing,
            "config": str(args.config) if args.config else None,
            "pipeline": pipeline,
            "tools": rows,
            "unavailable": missing,
        }

    if args.json_output:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        status = "ok" if report["ok"] else "FAILED"
        print(f"percepta probe: {status}")
        print(f"  pipeline: {', '.join(report.get('pipeline') or []) or '-'}")
        for row in report["tools"]:
            print(f"  - {row['name']:<16} {row['status']:<8} {row.get('error') or ''}".rstrip())
        if report["unavailable"]:
            print(
                "  unavailable: " + ", ".join(report["unavailable"]),
                file=sys.stderr,
            )
            print(
                "  hint: run with --config configs/ci.yaml to use the weight-free fake tool",
                file=sys.stderr,
            )
    return 0 if report["ok"] else 1


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def _cmd_models_check(args: argparse.Namespace) -> int:
    """Verify every configured checkpoint before the service ever starts.

    This exists because a wrong or truncated checkpoint is the failure mode that hurts most:
    it can load successfully and then produce quietly wrong segments. Checking here turns
    that into an actionable message instead of a mysterious quality regression.
    """
    config = load_config(args.config)
    rows: list[dict[str, Any]] = []

    for name, raw in (config.tools or {}).items():
        tool_config = dict(raw or {})
        if not tool_config.get("enabled", True):
            continue
        location = tool_config.get("checkpoint")
        if not location:
            continue  # tools without weights (geometry) have nothing to check
        row: dict[str, Any] = {
            "tool": name,
            "backend": str(tool_config.get("backend", "inprocess")),
            "location": str(location),
        }
        try:
            spec = resolve_weights(
                location,
                expected_sha256=tool_config.get("checkpoint_sha256"),
                verify="never" if args.no_hash else str(tool_config.get("verify_weights", "if-declared")),
                what=f"{name} checkpoint",
            )
            row.update(spec.describe())
            row["ok"] = True
        except WeightError as exc:
            row.update({"ok": False, "error": str(exc)})
        rows.append(row)

    ok = all(row["ok"] for row in rows)
    report = {
        "ok": ok,
        "config": str(args.config) if args.config else None,
        "models": rows,
    }
    if args.json_output:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(f"percepta models check: {'ok' if ok else 'FAILED'}")
        if not rows:
            print("  (no tools with a `checkpoint` configured)")
        for row in rows:
            mark = "ok" if row["ok"] else "FAIL"
            sha = row.get("observed_sha256") or "-"
            verified = "verified" if row.get("verified") else "recorded-only"
            print(f"  [{mark}] {row['tool']} ({row['backend']}): {row['location']}")
            if row["ok"]:
                print(f"        sha256 {sha} ({verified})")
            else:
                print(f"        {row['error'].splitlines()[0]}")
    return 0 if ok else 1


def _local_tool_row(entry: ToolEntry) -> dict[str, Any]:
    """Same shape as ``GET /v1/tools`` without importing the optional service extra."""
    instance = entry.instance
    capabilities = None
    if instance is not None:
        try:
            caps = instance.capabilities()
        except Exception:  # noqa: BLE001 - introspection never fails the CLI
            caps = None
        if caps is not None:
            capabilities = {
                "kind": caps.kind,
                "prompts": sorted(caps.prompts),
                "masks": caps.masks,
                "boxes": caps.boxes,
                "points": caps.points,
                "depth": caps.depth,
                "multi_view": caps.multi_view,
                "geometry": caps.geometry,
            }
    return {
        "name": entry.name,
        "version": str(getattr(instance, "version", "0.0.0")) if instance else "0.0.0",
        "status": "ok" if entry.status in {"ok", "lazy"} else entry.status,
        "kind": entry.kind,
        "source": entry.source,
        "capabilities": capabilities,
        "loaded": instance is not None,
        "error": entry.error,
    }


def _format_response(response: PerceiveResponse) -> str:
    lines = [
        f"request_id: {response.request_id}",
        f"percepts: {len(response.percepts)}",
    ]
    for percept in response.percepts:
        provenance = percept.provenance
        camera = percept.camera or "-"
        prompt = provenance.prompt
        prompt_text = ""
        if prompt is not None:
            prompt_text = f" {prompt.type}={prompt.text or prompt.box or prompt.point}"
        lines.append(
            f"[{camera}] {provenance.tool} v{provenance.tool_version}{prompt_text}: "
            f"{percept.status} ({provenance.latency_ms:.1f} ms, "
            f"{len(percept.observations)} observation(s))"
            + (f" — {percept.message}" if percept.message else "")
        )
        for observation in percept.observations:
            box = (
                "[" + ", ".join(f"{value:.1f}" for value in observation.box_xyxy) + "]"
                if observation.box_xyxy
                else "-"
            )
            centroid = (
                "[" + ", ".join(f"{value:.1f}" for value in observation.centroid_uv) + "]"
                if observation.centroid_uv
                else "-"
            )
            lines.append(
                f"    score={observation.score:.3f} area={observation.area_px} "
                f"box={box} centroid={centroid}"
            )
    if response.fused is not None:
        geometry = response.fused.geometry
        lines.append(
            "fused: "
            + (
                f"centroid={geometry.centroid} uncertainty_m={geometry.uncertainty_m}"
                if geometry is not None
                else response.fused.status
            )
        )
    else:
        lines.append("fused: none")
    if response.warnings:
        lines.append("warnings: " + ", ".join(response.warnings))
    if response.timings_ms:
        timings = ", ".join(f"{name}={value:.1f}ms" for name, value in response.timings_ms.items())
        lines.append(f"timings_ms: {timings}")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
