"""Structured logging for the service (spec §11.3).

Every request line carries the fields the spec asks for: ``request_id, tool, camera,
status, latency_ms, cache_hit``. Two formats are supported:

* JSON (``logging.json: true`` in the config) — one object per line, ready for a log
  shipper;
* plain text — human-readable when developing.

The handler is installed on the ``percepta`` logger only (never the root logger), so an
embedding application keeps control of its own logging, and re-configuring is idempotent.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

__all__ = ["JsonFormatter", "TextFormatter", "configure_logging", "log_event"]

#: Extra record attributes promoted into the JSON object, in a stable order.
EXTRA_FIELDS = ("request_id", "tool", "camera", "status", "latency_ms", "cache_hit", "detail")

_HANDLER_MARK = "_percepta_handler"
_LEVELS = {
    "critical": logging.CRITICAL,
    "error": logging.ERROR,
    "warning": logging.WARNING,
    "warn": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
    "trace": logging.DEBUG,
}


class JsonFormatter(logging.Formatter):
    """One JSON object per log record; unknown ``extra`` keys are ignored on purpose."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
        }
        for field in EXTRA_FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False, sort_keys=False)


class TextFormatter(logging.Formatter):
    """``LEVEL logger: message [field=value ...]`` — readable during development."""

    def __init__(self) -> None:
        super().__init__(fmt="%(levelname)s %(name)s: %(message)s")

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        extras = []
        for field in EXTRA_FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                extras.append(f"{field}={value}")
        if extras:
            text = f"{text} [{', '.join(extras)}]"
        if record.exc_info:
            text = f"{text}\n{self.formatException(record.exc_info)}"
        return text


def configure_logging(level: str = "info", json_output: bool = True) -> logging.Logger:
    """Install a single handler on the ``percepta`` logger and return that logger.

    Idempotent: calling it again replaces the handler this function installed earlier and
    leaves any other handler (including the root logger's) untouched.
    """
    logger = logging.getLogger("percepta")
    logger.setLevel(_LEVELS.get(str(level).lower(), logging.INFO))
    for handler in list(logger.handlers):
        if getattr(handler, _HANDLER_MARK, False):
            logger.removeHandler(handler)
            handler.close()
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(JsonFormatter() if json_output else TextFormatter())
    setattr(handler, _HANDLER_MARK, True)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def log_event(logger: logging.Logger, message: str, **fields: Any) -> None:
    """Log ``message`` with the spec's structured fields attached as ``extra``."""
    extra = {key: value for key, value in fields.items() if value is not None}
    logger.info(message, extra=extra)
