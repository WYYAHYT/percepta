"""Python SDK: HTTP client + in-process shortcut (spec §7, decision D8).

``PerceptaClient`` speaks the frozen §6 HTTP contract using the standard library's
``urllib.request`` only — ``httpx`` is a test-only dependency (decision D8), so the runtime
package needs nothing beyond the stdlib for transport.

Two entry points do the same job:

* :meth:`PerceptaClient.perceive` — POST ``/v1/perceive`` to a (by default loopback) server;
* :meth:`PerceptaClient.perceive_local` — call :class:`percepta.pipeline.PipelineRunner`
  in-process, for single-machine low latency. Both return the same
  :class:`~percepta.schema.PerceiveResponse`, so downstream code cannot tell them apart.

``images`` values may be numpy RGB ``(H, W, 3)`` arrays, local file paths (or ``Path``
objects), raw ``bytes``, or an already-built :class:`~percepta.schema.ImageRef`; arrays and
files are PNG-encoded with :func:`percepta.masks.encode_image`.

Retries are bounded (2 by default, exponential backoff) and apply to transient 5xx only.
A ``not_found`` Percept is a successful 200 response and is never retried; 4xx responses
raise :class:`PerceptaError` immediately with the parsed error body attached.
"""

from __future__ import annotations

import io
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

from .config import AppConfig
from .masks import encode_image
from .schema import (
    Calibration,
    ImageRef,
    PerceiveOptions,
    PerceiveRequest,
    PerceiveResponse,
    Prompt,
)

__all__ = [
    "DEFAULT_ENDPOINT",
    "PerceptaClient",
    "PerceptaError",
    "build_perceive_request",
]

DEFAULT_ENDPOINT = "http://127.0.0.1:8790"
"""Loopback only by default (spec §6/§12); cross-machine access goes through an SSH tunnel."""

#: Statuses worth retrying: transient server-side failures. Never 4xx, never a 200.
RETRY_STATUSES = frozenset({500, 502, 503, 504})


class PerceptaError(RuntimeError):
    """An HTTP call failed (or the server could not be reached).

    ``status_code`` is the HTTP status when there was a response (``None`` for connection
    failures), ``code`` is the appendix-A error code when the server sent one, and ``body``
    is the parsed JSON body (or the raw text) for debugging.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        code: str | None = None,
        body: Any = None,
        url: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code
        self.body = body
        self.url = url

    def __str__(self) -> str:
        prefix = f"HTTP {self.status_code}: " if self.status_code is not None else ""
        code = f"[{self.code}] " if self.code else ""
        return f"{prefix}{code}{self.message}"


# ---------------------------------------------------------------------------
# Request construction (shared by HTTP and local paths)
# ---------------------------------------------------------------------------


def _as_prompt(item: Prompt | str | dict[str, Any]) -> Prompt:
    if isinstance(item, Prompt):
        return item
    if isinstance(item, str):
        return Prompt(type="text", text=item)
    return Prompt.model_validate(item)


def _as_calibration(value: Any) -> Calibration:
    return value if isinstance(value, Calibration) else Calibration.model_validate(value)


def _as_options(value: PerceiveOptions | dict[str, Any] | None) -> PerceiveOptions:
    if value is None:
        return PerceiveOptions()
    if isinstance(value, PerceiveOptions):
        return value
    return PerceiveOptions.model_validate(value)


def _read_path(path: str | Path) -> np.ndarray:
    """Load an image file as RGB uint8 ``(H, W, 3)``."""
    from PIL import Image  # pillow is a default dependency (spec §13)

    with Image.open(Path(path)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _as_image_ref(value: Any, camera: str) -> ImageRef:
    if isinstance(value, ImageRef):
        return value
    if isinstance(value, np.ndarray):
        return encode_image(value, encoding="png", camera=camera)
    if isinstance(value, (str, Path)):
        return encode_image(_read_path(value), encoding="png", camera=camera)
    if isinstance(value, (bytes, bytearray)):
        from PIL import Image

        with Image.open(io.BytesIO(bytes(value))) as image:
            return encode_image(
                np.asarray(image.convert("RGB"), dtype=np.uint8), encoding="png", camera=camera
            )
    raise TypeError(
        f"image {camera!r} must be a numpy array, a file path, bytes or an ImageRef; "
        f"got {type(value).__name__}"
    )


def build_perceive_request(
    images: dict[str, Any],
    prompts: list[Prompt | str | dict[str, Any]],
    calibration: dict[str, Calibration | dict[str, Any]] | None = None,
    context: dict[str, Any] | None = None,
    options: PerceiveOptions | dict[str, Any] | None = None,
    request_id: str | None = None,
) -> PerceiveRequest:
    """Assemble a validated :class:`PerceiveRequest` (encoding images as PNG)."""
    if not images:
        raise ValueError("images must contain at least one camera")
    encoded = {camera: _as_image_ref(value, camera) for camera, value in images.items()}
    calib = {name: _as_calibration(value) for name, value in (calibration or {}).items()}
    return PerceiveRequest(
        request_id=request_id,
        images=encoded,
        calibration=calib,
        prompts=[_as_prompt(item) for item in prompts],
        context=dict(context or {}),
        options=_as_options(options),
    )


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class PerceptaClient:
    """Synchronous client for a percepta server (spec §7)."""

    def __init__(
        self,
        endpoint: str = DEFAULT_ENDPOINT,
        token: str | None = None,
        timeout: float = 180.0,
        *,
        retries: int = 2,
        backoff_s: float = 0.25,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.token = token
        self.timeout = float(timeout)
        self.retries = max(0, int(retries))
        self.backoff_s = max(0.0, float(backoff_s))
        self._local_runner: Any = None
        self._local_key: tuple[int | None, int | None, int | None] | None = None

    # -- transport -------------------------------------------------------

    def _url(self, path: str) -> str:
        return f"{self.endpoint}{path if path.startswith('/') else '/' + path}"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _raw_request(self, method: str, path: str, body: bytes | None) -> tuple[int, bytes]:
        """One HTTP attempt. Raises :class:`PerceptaError` for any non-2xx/connection error."""
        url = self._url(path)
        request = urllib.request.Request(
            url, data=body, method=method, headers=self._headers()
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return int(response.status), response.read()
        except urllib.error.HTTPError as exc:
            payload = exc.read()
            parsed, code = _parse_error(payload)
            raise PerceptaError(
                parsed or f"request failed with HTTP {exc.code}",
                status_code=int(exc.code),
                code=code,
                body=payload and _maybe_json(payload),
                url=url,
            ) from None
        except urllib.error.URLError as exc:
            raise PerceptaError(
                f"cannot reach {url}: {exc.reason}", url=url, code="connection_error"
            ) from None

    def _request(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> tuple[int, bytes]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        last: PerceptaError | None = None
        for attempt in range(self.retries + 1):
            try:
                return self._raw_request(method, path, body)
            except PerceptaError as exc:
                # Bounded retry on transient 5xx only (spec §7); 4xx and not_found are final.
                if exc.status_code in RETRY_STATUSES and attempt < self.retries:
                    last = exc
                    time.sleep(self.backoff_s * (2**attempt))
                    continue
                raise
        raise last  # pragma: no cover - loop always returns or raises

    # -- read-only endpoints ---------------------------------------------

    def health(self) -> dict[str, Any]:
        """``GET /healthz`` — ``{"ok", "uptime_s", "tools"}``."""
        _status, data = self._request("GET", "/healthz")
        return json.loads(data)

    def info(self) -> dict[str, Any]:
        """``GET /v1/info`` — versions, tool capabilities, device."""
        _status, data = self._request("GET", "/v1/info")
        return json.loads(data)

    def tools(self) -> list[dict[str, Any]]:
        """``GET /v1/tools`` — the tool list (unwrapped from ``{"tools": [...]}``)."""
        _status, data = self._request("GET", "/v1/tools")
        document = json.loads(data)
        if isinstance(document, dict):
            return list(document.get("tools") or [])
        return list(document)

    def wait_until_ready(self, timeout_s: float = 10.0, interval_s: float = 0.1) -> bool:
        """Poll ``/healthz`` until it answers; returns ``False`` on timeout instead of raising."""
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        while True:
            try:
                health = self.health()
                if health.get("ok"):
                    return True
            except PerceptaError:
                pass
            if time.monotonic() >= deadline:
                return False
            time.sleep(max(0.01, float(interval_s)))

    # -- perception ------------------------------------------------------

    def perceive(
        self,
        images: dict[str, Any],
        prompts: list[Prompt | str | dict[str, Any]],
        calibration: dict[str, Calibration | dict[str, Any]] | None = None,
        context: dict[str, Any] | None = None,
        options: PerceiveOptions | dict[str, Any] | None = None,
    ) -> PerceiveResponse:
        """POST ``/v1/perceive`` and return the parsed response (spec §7)."""
        request = build_perceive_request(images, prompts, calibration, context, options)
        _status, data = self._request("POST", "/v1/perceive", request.model_dump(mode="json"))
        return PerceiveResponse.model_validate_json(data)

    def perceive_local(
        self,
        images: dict[str, Any],
        prompts: list[Prompt | str | dict[str, Any]],
        calibration: dict[str, Calibration | dict[str, Any]] | None = None,
        context: dict[str, Any] | None = None,
        options: PerceiveOptions | dict[str, Any] | None = None,
        *,
        config: AppConfig | None = None,
        registry: Any = None,
        cache: Any = None,
    ) -> PerceiveResponse:
        """Same request/response as :meth:`perceive`, but in-process (decision D12).

        The runner is created on first use and reused; pass ``config`` (and optionally
        ``registry``/``cache``) to override it. Without a config the code defaults apply
        (spec §5.3: ``["sam3", "multi_view_3d"]``), so pass ``options={"tools": ["fake"]}``
        for a weight-free pipeline.
        """
        runner = self._get_local_runner(config, registry, cache)
        request = build_perceive_request(images, prompts, calibration, context, options)
        return runner.perceive(request)

    def _get_local_runner(self, config: AppConfig | None, registry: Any, cache: Any) -> Any:
        """Reuse the in-process runner while the collaborator objects stay identical."""
        from .pipeline import PipelineRunner
        from .registry import ToolRegistry

        key = (
            id(config) if config is not None else None,
            id(registry) if registry is not None else None,
            id(cache) if cache is not None else None,
        )
        if self._local_runner is not None and key == self._local_key:
            return self._local_runner
        runner = PipelineRunner(
            registry if registry is not None else ToolRegistry(),
            config if config is not None else AppConfig(),
            cache,
        )
        self._local_runner = runner
        self._local_key = key
        return runner

    # -- life cycle ------------------------------------------------------

    def close(self) -> None:
        """Close the in-process runner (tools) if one was created. Idempotent."""
        runner, self._local_runner = self._local_runner, None
        self._local_key = None
        if runner is not None:
            runner.close()

    def __enter__(self) -> PerceptaClient:
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Error-body helpers
# ---------------------------------------------------------------------------


def _maybe_json(payload: bytes) -> Any:
    try:
        return json.loads(payload)
    except (ValueError, UnicodeDecodeError):
        return payload.decode("utf-8", "replace")


def _parse_error(payload: bytes) -> tuple[str | None, str | None]:
    """Return ``(message, code)`` from an appendix-A error body, tolerating anything else."""
    document = _maybe_json(payload)
    if isinstance(document, dict):
        message = document.get("message") or document.get("detail") or document.get("error")
        code = document.get("error") or document.get("code")
        return (str(message) if message is not None else None,
                str(code) if code is not None else None)
    if isinstance(document, str) and document:
        return document, None
    return None, None
