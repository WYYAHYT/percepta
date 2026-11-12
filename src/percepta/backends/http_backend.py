"""HTTP backend: the model runs as a separate service, possibly on another machine.

Same isolation as the subprocess backend, but the boundary is a socket instead of a pipe, so
the worker can live on a different host, be restarted independently, and be observed by
normal service tooling (``docs/model-isolation.md`` §4.1).

The wire format is the *same* binary protocol as the subprocess backend, carried in the body
of a POST. Reusing it means one encoder/decoder pair, and the worker code is identical — only
the transport differs.
"""

from __future__ import annotations

import logging
import urllib.error
import urllib.request
from http.client import HTTPException
from typing import Any

import numpy as np

from ..schema import Prompt
from .base import BackendRequestError, BackendUnavailable
from .protocol import (
    ProtocolError,
    RequestFrame,
    decode_response,
    encode_request,
)

__all__ = ["HttpBackend"]

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 120.0


class HttpBackend:
    """A :class:`~percepta.backends.base.ModelBackend` reached over HTTP."""

    name = "http"

    def __init__(
        self,
        endpoint: str = "http://127.0.0.1:8801",
        *,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        token: str | None = None,
        path: str = "/segment",
        prepare_path: str = "/set_image",
    ) -> None:
        self.endpoint = str(endpoint).rstrip("/")
        self.timeout_s = float(timeout_s)
        self.token = token
        self.path = path if path.startswith("/") else f"/{path}"
        self.prepare_path = prepare_path if prepare_path.startswith("/") else f"/{prepare_path}"
        self._failure: str | None = None

    # -- transport -------------------------------------------------------

    def _post(self, path: str, payload: bytes) -> bytes:
        if self._failure:
            raise BackendUnavailable(self._failure)
        url = f"{self.endpoint}{path}"
        request = urllib.request.Request(
            url,
            data=payload,
            method="POST",
            headers={
                "Content-Type": "application/octet-stream",
                "Accept": "application/octet-stream",
            },
        )
        if self.token:
            request.add_header("Authorization", f"Bearer {self.token}")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            body = exc.read()[:400].decode("utf-8", "replace")
            if 400 <= exc.code < 500:
                raise BackendRequestError(f"{url} rejected the request ({exc.code}): {body}") from exc
            raise BackendUnavailable(f"{url} failed ({exc.code}): {body}") from exc
        except urllib.error.URLError as exc:
            raise BackendUnavailable(
                f"model service unreachable at {self.endpoint}: {exc.reason}"
            ) from exc
        except TimeoutError as exc:
            raise BackendUnavailable(f"model service timed out after {self.timeout_s}s") from exc
        except (HTTPException, OSError) as exc:
            raise self._fail(f"model service connection failed: {exc}") from exc

    # -- ModelBackend ----------------------------------------------------

    def set_image(self, image_rgb: np.ndarray, depth: np.ndarray | None = None) -> None:
        payload = encode_request(RequestFrame(prompt_json="", image=image_rgb, depth=depth))
        body = self._post(self.prepare_path, payload)
        response = self._decode(body)
        if response.status == "error":
            raise self._map_error(response)

    def segment(self, prompt: Prompt) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
        payload = encode_request(RequestFrame(prompt_json=prompt.model_dump_json()))
        response = self._decode(self._post(self.path, payload))
        if response.status == "error":
            raise self._map_error(response)
        if response.masks is None:
            return np.zeros((0, 1, 1), dtype=bool), None, np.zeros(0, dtype=float)
        return response.masks, response.boxes, response.scores

    def _decode(self, body: bytes):
        try:
            return decode_response(body)
        except ProtocolError as exc:
            raise self._fail(f"model service returned a malformed frame: {exc}") from exc

    def _fail(self, message: str) -> BackendUnavailable:
        self._failure = message
        return BackendUnavailable(message)

    @property
    def health_error(self) -> str | None:
        """Report a terminal malformed-response or connection failure.

        Returns:
            Failure detail, or None before any terminal transport/protocol failure.

        Example:
            >>> HttpBackend().health_error is None
            True
        """
        return self._failure

    @staticmethod
    def _map_error(response) -> Exception:
        message = response.error or "model service reported an error"
        if response.error_kind == "invalid_request":
            return BackendRequestError(message)
        return BackendUnavailable(message)

    def close(self) -> None:
        """No connection is held open, so there is nothing to release."""

    def describe(self) -> dict[str, Any]:
        return {"backend": self.name, "endpoint": self.endpoint, "path": self.path}
