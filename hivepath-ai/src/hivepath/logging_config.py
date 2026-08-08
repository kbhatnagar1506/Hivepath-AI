"""Logging setup.

The previous codebase reported failures with bare ``print()`` calls and, in the
request path, with ``except Exception: pass``. Both are replaced by the standard
library logger configured here, so failures are attributable and can be shipped
to a log aggregator.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

from hivepath.config import Settings, get_settings

_CONFIGURED = False

_RESERVED = frozenset(
    logging.LogRecord("", 0, "", 0, "", None, None).__dict__
) | {"asctime", "message", "taskName"}


class JsonFormatter(logging.Formatter):
    """Single-line JSON output, suitable for log aggregation in deployment."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Surface any structured extras attached via logger.info(..., extra={...})
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(settings: Settings | None = None, *, force: bool = False) -> None:
    """Configure root logging once per process.

    Human-readable in development, JSON in staging and production.
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    settings = settings or get_settings()

    handler = logging.StreamHandler(sys.stdout)
    if settings.is_production or settings.environment.value == "staging":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-8s %(name)-38s %(message)s",
                datefmt="%H:%M:%S",
            )
        )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(settings.log_level)

    # These are chatty at DEBUG and rarely what we are debugging.
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module logger. Prefer ``get_logger(__name__)``."""
    return logging.getLogger(name)
